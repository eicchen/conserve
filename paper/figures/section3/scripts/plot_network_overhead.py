"""
Section 3 figure: PD-disagg network overhead vs input token size.

Network overhead = decoder_forward_start_time − prefiller_finish_time (per request),
i.e., the gap between the prefiller emitting the request's KV and the decoder
beginning compute on it. Captures KV-cache transfer + RPC/scheduling overhead.

Inputs:
  profiling/gpu_monitoring/Qwen3-0.6B/pd_disagg/<L>/
    prefiller_vllm_core_log.jsonl   - prefiller step_starts/ends with request ids
    decoder_forward_start_time.csv  - one unix timestamp per request

Two-regime model:
  low  L  : net ≈ constant overhead floor  (RPC + scheduling)
  high L  : net ≈ a · L + b                (linear KV transfer)
Cut at user-chosen knee (default L=512).

Outputs:
  network_overhead.{pdf,png}
  network_overhead_fit.txt
  network_overhead_table.csv
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "config"))
from config import GPU_MON_ROOT, MODEL_SHORT, MODEL_DATA_DIR

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR = (GPU_MON_ROOT / MODEL_SHORT / "pd_disagg_300W")
OUT = (MODEL_DATA_DIR / "paper" / "section3" / "fig3")
WARMUP_DROP = 16          # drop first N requests per L (server warmup)
KNEE_L = 1024             # knee: constant regime L<=KNEE_L, linear L>=KNEE_L


def parse_prefiller_finish(L: int) -> pd.DataFrame:
    """Return per-request prefiller step_end timestamps."""
    df = pd.read_json(DATA_DIR / str(L) / "prefiller_vllm_core_log.jsonl", lines=True)
    se = df[df["event"] == "step_end"].copy().reset_index(drop=True)
    se = se[se["executed_request_ids"].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
    # Each step finishes one (or more) requests. Take the LAST step_end per request:
    # that's where prefill completes (for chunked prefill, the final chunk).
    se["request_id"] = se["executed_request_ids"].apply(lambda lst: lst[0])
    se["ts"] = pd.to_datetime(se["timestamp"]).dt.tz_localize("America/Chicago")
    # When a request appears in finished_request_ids it has completed prefill.
    fin = se[se["finished_request_ids"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    fin["request_id"] = fin["finished_request_ids"].apply(lambda lst: lst[0])
    return fin[["request_id", "ts"]].drop_duplicates("request_id").rename(columns={"ts": "ts_finish"})


def parse_decoder_starts(L: int) -> pd.DataFrame:
    p = DATA_DIR / str(L) / "decoder_forward_start_time.csv"
    df = pd.read_csv(p, header=None, names=["unix_ts"])
    df["request_idx"] = np.arange(len(df))
    df["ts_decoder_start"] = pd.to_datetime(df["unix_ts"], unit="s", utc=True).dt.tz_convert("America/Chicago")
    return df[["request_idx", "ts_decoder_start"]]


def parse_prefiller_start(L: int) -> pd.DataFrame:
    """Return per-request prefiller request_start timestamps (engine log)."""
    df = pd.read_json(DATA_DIR / str(L) / "prefiller_vllm_engine_log.jsonl", lines=True)
    rs = df[df["event"] == "request_start"].copy()
    rs["ts_start"] = pd.to_datetime(rs["timestamp"]).dt.tz_localize("America/Chicago")
    return rs[["request_id", "ts_start"]].drop_duplicates("request_id")


def cell_latencies(L: int) -> pd.DataFrame:
    """Per-request DataFrame for one L cell, post-warmup, with columns:
      network_ms : decoder_forward_start − prefiller_finish  (KV transfer + RPC)
      ttft_ms    : decoder_forward_start − prefiller_request_start
                   (= prefill compute + network; the prefill-phase TTFT)

    The decoder_forward_start_time.csv is written in append mode by the proxy
    server, so a previously-failed run can leave stale lines at the top. We
    skip CSV rows that timestamp BEFORE the prefiller's first finish, then
    align row-by-row.
    """
    fin = parse_prefiller_finish(L).reset_index(drop=True)
    dec = parse_decoder_starts(L)
    if len(fin) == 0:
        return pd.DataFrame(columns=["network_ms", "ttft_ms"])
    fin = fin.merge(parse_prefiller_start(L), on="request_id", how="left")

    first_prefill_finish = fin["ts_finish"].iloc[0]
    stale_mask = dec["ts_decoder_start"] < first_prefill_finish
    if stale_mask.any():
        dec = dec[~stale_mask].reset_index(drop=True)

    n = min(len(fin), len(dec))
    fin = fin.iloc[:n].reset_index(drop=True)
    dec = dec.iloc[:n].reset_index(drop=True)
    network = (dec["ts_decoder_start"] - fin["ts_finish"]).dt.total_seconds() * 1000.0
    ttft = (dec["ts_decoder_start"] - fin["ts_start"]).dt.total_seconds() * 1000.0
    out = pd.DataFrame({"network_ms": network, "ttft_ms": ttft}).iloc[WARMUP_DROP:]
    out = out[np.isfinite(out["network_ms"]) & np.isfinite(out["ttft_ms"])]
    out = out[out["network_ms"] > 0]
    return out.reset_index(drop=True)


def linfit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def main():
    L_values = sorted(int(p.name) for p in DATA_DIR.iterdir() if p.is_dir())
    per_L = {}        # L -> network latency array (ms)
    per_L_pct = {}    # L -> network-as-%-of-TTFT array
    for L in L_values:
        try:
            df = cell_latencies(L)
            if len(df) > 0:
                v = df["network_ms"].to_numpy()
                per_L[L] = v
                per_L_pct[L] = (df["network_ms"] / df["ttft_ms"] * 100.0).to_numpy()
                print(f"L={L:>6}: n={len(v):>5}  p50={np.percentile(v,50):6.2f}  "
                      f"p99={np.percentile(v,99):6.2f}  "
                      f"net%={np.median(per_L_pct[L]):5.1f}")
        except Exception as e:
            print(f"L={L}: skipped ({e})")

    rows = []
    for L, v in per_L.items():
        rows.append({
            "L": L,
            "n": len(v),
            "mean_ms": float(np.mean(v)),
            "median_ms": float(np.median(v)),
            "p25_ms": float(np.percentile(v, 25)),
            "p75_ms": float(np.percentile(v, 75)),
            "p99_ms": float(np.percentile(v, 99)),
            "std_ms": float(np.std(v)),
            "net_pct_median": float(np.median(per_L_pct[L])),
        })
    table = pd.DataFrame(rows).sort_values("L").reset_index(drop=True)
    table.to_csv(OUT / "network_overhead_table.csv", index=False)

    # Knee at KNEE_L belongs to both regimes: constant fit over L<=KNEE_L,
    # linear fit over L>=KNEE_L.
    low_Ls = [L for L in per_L if L <= KNEE_L]
    high_Ls = [L for L in per_L if L >= KNEE_L]

    def stack(Ls):
        xs = np.concatenate([np.full(len(per_L[L]), L) for L in Ls]).astype(float)
        ys = np.concatenate([per_L[L] for L in Ls])
        return xs, ys

    x_low, y_low = stack(low_Ls)
    x_high, y_high = stack(high_Ls)
    low_mean = float(np.mean(y_low))
    low_std = float(np.std(y_low))
    slope_hi, b_hi, r2_hi = linfit(x_high, y_high)

    with open(OUT / "network_overhead_fit.txt", "w") as f:
        f.write("PD-disagg network overhead: two-regime model\n")
        f.write(f"Source: gpu_monitoring/{MODEL_SHORT}/pd_disagg_300W/<L>/\n")
        f.write(f"Warmup drop: first {WARMUP_DROP} requests per L\n")
        f.write(f"Knee cutoff L = {KNEE_L}\n")
        f.write(f"Low-regime  L: {low_Ls}\n")
        f.write(f"High-regime L: {high_Ls}\n\n")
        f.write(f"Low regime (constant): mean = {low_mean:.3f} ms, std = {low_std:.3f} ms\n")
        f.write(f"  n samples = {len(y_low)}\n\n")
        f.write(f"High regime (linear): net = a*L + b\n")
        f.write(f"  a (slope)  = {slope_hi*1000:.4f} us/token\n")
        f.write(f"  b (offset) = {b_hi:.4f} ms\n")
        f.write(f"  R^2        = {r2_hi:.5f}\n")
        f.write(f"  n samples  = {len(y_high)}\n")
        # KV bytes per token: 112 KiB (from HANDOFF). Effective bandwidth:
        kv_per_tok_bytes = 112 * 1024
        if slope_hi > 0:
            bw_GBps = kv_per_tok_bytes / (slope_hi * 1e-3) / (1024**3)
            f.write(f"\n  Implied effective transfer bandwidth at 112 KiB/token:\n")
            f.write(f"  {bw_GBps:.2f} GB/s\n")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(5, 3))
    for L in sorted(per_L):
        v = per_L[L]
        # subsample for plot if huge
        if len(v) > 500:
            v_plot = np.random.RandomState(0).choice(v, 500, replace=False)
        else:
            v_plot = v
        ax.scatter(np.full(len(v_plot), L), v_plot, s=4, alpha=0.15,
                   color="#4477AA", rasterized=True)

    # Constant line for low regime
    x_lo_line = np.array([min(L_values) * 0.7, KNEE_L])
    ax.plot(x_lo_line, np.full_like(x_lo_line, low_mean, dtype=float), color="#EE7733",
            lw=1.8, label=f"KV-transfer latency (≤ {KNEE_L} tokens): const $\\approx$ {low_mean:.1f} ms")
    # Linear fit for high regime. Sample densely (log-spaced) because the
    # x-axis is log: a linear function appears curved on log-x, so a 2-point
    # plot would render an incorrect chord.
    x_hi_line = np.logspace(np.log10(KNEE_L), np.log10(max(L_values) * 1.1), 200)
    ax.plot(x_hi_line, slope_hi * x_hi_line + b_hi, color="#CC3311", lw=1.8,
            label=f"KV-transfer latency (≥ {KNEE_L} tokens): \nlinear $\\approx$ {slope_hi*1000:.2f} µs/tok + {b_hi:.1f} ms  ($R^2$={r2_hi:.3f})")
    ax.axvline(KNEE_L, color="gray", ls="--", lw=0.7, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("KV-transfer latency (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Right axis: network overhead as a share of the prefill-phase TTFT
    # (TTFT = prefill compute + network).
    ax2 = ax.twinx()
    L_pct = sorted(per_L_pct)
    pct_med = [np.median(per_L_pct[L]) for L in L_pct]
    ax2.plot(L_pct, pct_med, color="#117733", lw=1.8, marker="o", ms=4,
             label="KV-transfer latency / TTFT")
    ax2.set_ylabel("KV-transfer share of TTFT (%)", color="#117733")
    ax2.tick_params(axis="y", labelcolor="#117733")
    ax2.set_ylim(0, 35)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "network_overhead.pdf", dpi=200)
    fig.savefig(OUT / "network_overhead.png", dpi=200)

    print("\n" + (OUT / "network_overhead_fit.txt").read_text())
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
