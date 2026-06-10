"""
Section 3 figure: decode latency is dominated by kernel-overhead baseline for
Qwen3-0.6B; both batch size and active KV cache exert only modest pressure
across the operating range we measured.

Inputs:
  profiling/gpu_monitoring/Qwen3-0.6B/decode/<submitted_b>/vllm_core_log.jsonl
  for submitted_b in {1, 2, 4, 8, 16, 32, 64}.

Outputs (in this directory):
  decode_flat.pdf, decode_flat.png      - latency vs active KV, lines per batch bucket
  decode_flat_table.csv                 - per-(B-bucket, KV-bin) median latency
  decode_flat_fit.txt                   - linear-model fit summary
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
GPU_MON_ROOT = Path("/data/projects/AgentScaling/gpu_monitoring")  # external data dir; outside the conserve repo


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = (GPU_MON_ROOT / "Qwen3-0.6B/decode")
OUT = Path(__file__).parent.parent / "output" / "300W"
SUBMITTED = [1, 2, 4, 8, 16, 32, 64]
INPUT_LEN = 8
WARMUP_DROP = 32  # decode steps to discard at the start of each run

# Bucket the *actual* per-step batch size into a manageable legend.
BATCH_BUCKETS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
BUCKET_COLORS = ["#332288", "#117733", "#88CCEE", "#DDCC77", "#CC6677", "#882255"]


def parse_decode(submitted_b):
    df = pd.read_json(BASE / str(submitted_b) / "vllm_core_log.jsonl", lines=True)
    df = df[df["event"] == "step_end"].reset_index(drop=True)
    df["latency_ms"] = df["timestamp"].diff().dt.total_seconds() * 1000.0

    pos = {}
    kv_total = np.zeros(len(df), dtype=np.int64)
    batch_active = np.zeros(len(df), dtype=np.int32)
    pure_decode = np.zeros(len(df), dtype=bool)
    for i, rids in enumerate(df["executed_request_ids"].tolist()):
        if not isinstance(rids, list) or not rids:
            continue
        s = 0
        has_prefill = False
        for r in rids:
            k = pos.get(r, 0)
            if k == 0:
                has_prefill = True
            s += INPUT_LEN + max(k - 1, 0)
            pos[r] = k + 1
        batch_active[i] = len(rids)
        kv_total[i] = s
        pure_decode[i] = not has_prefill
    df["batch_active"] = batch_active
    df["active_kv_total"] = kv_total
    df["pure_decode"] = pure_decode
    df = df.iloc[1:]
    df = df[df["pure_decode"]].iloc[WARMUP_DROP:]
    return df[["batch_active", "active_kv_total", "latency_ms"]].dropna()


def bucket_label(lo, hi):
    return f"B = {lo}" if lo == hi else f"B ∈ [{lo}, {hi}]"


def main():
    parts = [parse_decode(b) for b in SUBMITTED]
    data = pd.concat(parts, ignore_index=True)
    data = data[np.isfinite(data["latency_ms"])]

    X = np.column_stack([np.ones(len(data)), data["batch_active"], data["active_kv_total"]])
    y = data["latency_ms"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma = coef
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(((y - yhat) ** 2).mean()))

    rows = []
    for lo, hi in BATCH_BUCKETS:
        sl = data[(data["batch_active"] >= lo) & (data["batch_active"] <= hi)]
        if not len(sl):
            continue
        kv = sl["active_kv_total"].to_numpy()
        lat = sl["latency_ms"].to_numpy()
        edges = np.unique(np.quantile(kv, np.linspace(0, 1, 12)))
        for i in range(len(edges) - 1):
            mask = (kv >= edges[i]) & (kv <= edges[i + 1])
            if mask.sum() < 30:
                continue
            rows.append({
                "B_lo": lo, "B_hi": hi,
                "kv_lo": int(edges[i]), "kv_hi": int(edges[i + 1]),
                "kv_mid": float((edges[i] + edges[i + 1]) / 2),
                "n": int(mask.sum()),
                "lat_p25": float(np.percentile(lat[mask], 25)),
                "lat_p50": float(np.percentile(lat[mask], 50)),
                "lat_p75": float(np.percentile(lat[mask], 75)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "decode_flat_table.csv", index=False)

    with open(OUT / "decode_flat_fit.txt", "w") as f:
        f.write("Qwen3-0.6B decode latency: planar fit\n")
        f.write(f"Total decode steps: {len(data)}\n")
        f.write(f"batch_active range: {int(data['batch_active'].min())} to {int(data['batch_active'].max())}\n")
        f.write(f"active_kv_total range: {int(data['active_kv_total'].min())} to {int(data['active_kv_total'].max())}\n")
        f.write(f"latency_ms p50/p99: {float(np.median(y)):.3f} / {float(np.percentile(y, 99)):.3f}\n\n")
        f.write("Model:  lat (ms) = alpha + beta * B + gamma * sum_KV\n")
        f.write(f"  alpha = {alpha:.4f} ms             (kernel-launch / dispatch floor)\n")
        f.write(f"  beta  = {beta*1000:.3f} us per active request   (FFN / per-token compute)\n")
        f.write(f"  gamma = {gamma*1000:.4f} us per active KV token (attention memory bandwidth)\n")
        f.write(f"  R^2   = {r2:.4f}    RMSE = {rmse:.3f} ms\n\n")
        f.write("Per-batch-bucket headline:\n")
        for lo, hi in BATCH_BUCKETS:
            sl = data[(data["batch_active"] >= lo) & (data["batch_active"] <= hi)]
            if not len(sl):
                continue
            f.write(f"  {bucket_label(lo, hi):<14} n={len(sl):>8}  "
                    f"KV med={int(sl['active_kv_total'].median()):>6}  "
                    f"lat p50={sl['latency_ms'].median():.2f} ms  "
                    f"p99={sl['latency_ms'].quantile(0.99):.2f} ms\n")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for (lo, hi), color in zip(BATCH_BUCKETS, BUCKET_COLORS):
        sub = table[(table["B_lo"] == lo) & (table["B_hi"] == hi)]
        if not len(sub):
            continue
        ax.fill_between(sub["kv_mid"], sub["lat_p25"], sub["lat_p75"],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(sub["kv_mid"], sub["lat_p50"], color=color, lw=1.6, marker="o",
                ms=4, label=bucket_label(lo, hi))

    ax.axhline(alpha, color="black", ls="--", lw=0.9, alpha=0.7,
               label=f"baseline $\\alpha \\approx$ {alpha:.1f} ms")

    ax.set_xscale("log")
    ax.set_xlabel("Total active KV cache (tokens)")
    ax.set_ylabel("Decode step latency (ms)")
    ax.set_ylim(0, max(25, table["lat_p75"].max() * 1.05))
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    fit_text = (f"$\\mathrm{{lat}} = {alpha:.2f} + {beta*1000:.1f}\\,\\mu s/\\mathrm{{req}}\\cdot B"
                f" + {gamma*1000:.3f}\\,\\mu s/\\mathrm{{tok}}\\cdot\\mathrm{{KV}}$"
                f"   ($R^2$={r2:.2f})")
    ax.text(0.98, 0.04, fit_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))

    fig.tight_layout()
    fig.savefig(OUT / "decode_flat.pdf", dpi=200)
    fig.savefig(OUT / "decode_flat.png", dpi=200)

    print(open(OUT / "decode_flat_fit.txt").read())


if __name__ == "__main__":
    main()
