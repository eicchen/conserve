"""Section 3 figure: per-step decode latency drift as KV cache grows.

Reproduces the analysis from gpu_monitoring/plot.ipynb, cell 12. Runs the
Qwen3-0.6B offline decode trace at B=64; for each decode step_id we compute
the geometric-mean latency across the B requests in that step, then fit a
linear regression on (step_id, gmean_latency) over step_id >= 750. The
positive slope captures the linear cost of attending over an ever-growing
KV cache during sustained decode.

Inputs:
  profiling/gpu_monitoring/Qwen3-0.6B/decode/64/vllm_core_log.jsonl

Outputs:
  decode_step_drift.{pdf,png}
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
GPU_MON_ROOT = Path("/data/projects/AgentScaling/gpu_monitoring")  # external data dir; outside the conserve repo


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from sklearn.linear_model import LinearRegression

DATA = (GPU_MON_ROOT / "Qwen3-0.6B/decode/64/vllm_core_log.jsonl")
OUT = (REPO_ROOT / "paper/figures/section3/output/300W")
WARMUP_STEPS = 750   # drop early steps where the engine is still ramping
BATCH_SIZE = 64


def read_df(path):
    """Pair step_start / step_end events, explode executed and finished
    request ids, and compute per-request per-step latency (ms). Matches the
    `read_df(..., offline=True)` helper in gpu_monitoring/plot.ipynb cell 2."""
    df = pd.read_json(path, lines=True)
    starts = df[df.event == "step_start"]["timestamp"].reset_index(drop=True)
    ends   = df[df.event == "step_end"].copy().reset_index(drop=True)
    ends["start_ts"] = starts

    events = pd.concat([
        ends[ends.executed_request_ids.str.len() > 0][["start_ts", "timestamp", "executed_request_ids"]]
            .explode("executed_request_ids")
            .rename(columns={"executed_request_ids": "request_id"})
            .assign(type="exec"),
        ends[ends.finished_request_ids.str.len() > 0][["start_ts", "timestamp", "finished_request_ids"]]
            .explode("finished_request_ids")
            .rename(columns={"finished_request_ids": "request_id"})
            .assign(type="finish"),
    ], ignore_index=True)

    events["type_order"] = events["type"].map({"exec": 0, "finish": 1})
    events = events.sort_values(["timestamp", "type_order"]).drop(columns="type_order")
    events["step_id"] = events["type"].eq("finish").groupby(events["request_id"]).cumsum()

    exec_first = events[events.type == "exec"].groupby(["request_id", "step_id"], as_index=False).first()
    finish_rows = events[events.type == "finish"].copy()
    finish_rows["step_id"] -= 1

    latency = exec_first.merge(
        finish_rows[["request_id", "step_id", "timestamp"]],
        on=["request_id", "step_id"], how="inner",
        suffixes=("_start", "_finish"),
    )
    latency["latency"] = latency["timestamp_finish"] - latency["start_ts"]
    latency["request_id"] = latency["request_id"].astype("int32")
    latency = (latency[["request_id", "step_id", "start_ts", "timestamp_finish", "latency"]]
               .sort_values(["request_id", "start_ts"])
               .reset_index(drop=True))
    latency["latency_ms"] = latency["latency"].dt.total_seconds() * 1000
    return latency


def main():
    df = read_df(DATA)
    print(f"Loaded {len(df):,} (request, step) latency rows from {DATA}")

    df = df[df["step_id"] >= WARMUP_STEPS]
    g = df.groupby("step_id")["latency_ms"]
    summary = pd.DataFrame({
        "mean":   g.mean(),
        "median": g.median(),
        "gmean":  g.apply(gmean),
        "p5":     g.quantile(0.05),
        "p95":    g.quantile(0.95),
        "p99":    g.quantile(0.99),
    }).sort_index()

    # Variance summary (averaged across step_ids).
    avg_p5_p95 = float((summary["p95"] - summary["p5"]).mean())
    avg_p99    = float(summary["p99"].mean())
    avg_med    = float(summary["median"].mean())
    print(f"Spread summary across step_ids:")
    print(f"  mean p5-p95 spread    = {avg_p5_p95:.3f} ms")
    print(f"  mean p99              = {avg_p99:.3f} ms")
    print(f"  spread / median       = {avg_p5_p95/avg_med*100:.1f}%")
    print(f"Central tendency (mean across step_ids):")
    print(f"  per-step mean   = {summary['mean'].mean():.3f} ms")
    print(f"  per-step median = {summary['median'].mean():.3f} ms")
    print(f"  per-step gmean  = {summary['gmean'].mean():.3f} ms")

    # ── Headline plot: mean curve + p5-p95 band + p99 line ─────────────────
    fig, ax = plt.subplots(figsize=(5, 3))

    # Thicker gray horizontal line at y=30 to mark where the scale flips
    # from linear to log.
    ax.axhline(30.0, color="#888888", lw=1.4, alpha=0.7, zorder=2)

    # p5-p95 spread band (light).
    ax.fill_between(summary.index, summary["p5"], summary["p95"],
                    color="#4477AA", alpha=0.20, linewidth=0,
                    label="P5–P95 TBT Spread")
    # p99 tail line.
    ax.plot(summary.index, summary["p99"],
            color="#BB5566", lw=0.7, ls="--", alpha=0.7, label="P99 TBT")
    # Per-step mean.
    ax.plot(summary.index, summary["mean"],
            color="#AA4499", lw=1.3, label="Mean TBT")

    # Symlog: linear 10-30, log above 30.
    max_p99 = float(summary["p99"].max())
    ax.set_yscale("symlog", linthresh=30.0, linscale=1.0)
    ax.set_yticks([10, 15, 20, 25, 30, 50, 100, 150, 200, 250, 300])
    ax.set_yticklabels(["10", "15", "20", "25", "30",
                         "50", "100", "150", "200", "250", "300"],
                        fontsize=9)
    ax.set_ylim(10, max_p99 * 1.05)

    

    ax.set_xlabel("Decode Iteration Counter")
    ax.set_ylabel("TBT (ms)")
    ax.grid(True, alpha=0.25, linestyle=":", which="both")
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=3,
              bbox_to_anchor=(0.5, 1.12), handlelength=1.5,
              columnspacing=0.9, handletextpad=0.4)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "decode_step_drift.pdf", dpi=200)
    fig.savefig(OUT / "decode_step_drift.png", dpi=200)
    print(f"\nSaved {OUT}/decode_step_drift.pdf / .png")

    # ── raw-point scatter (all (request, step) latencies kept after warmup) ─
    Xr = df["step_id"].to_numpy().reshape(-1, 1)
    yr = df["latency_ms"].to_numpy()
    model_raw = LinearRegression().fit(Xr, yr)
    yhat_r = model_raw.predict(Xr)
    ss_res_r = float(((yr - yhat_r) ** 2).sum())
    ss_tot_r = float(((yr - yr.mean()) ** 2).sum())
    r2_r = 1.0 - ss_res_r / ss_tot_r if ss_tot_r > 0 else float("nan")
    print(f"Linear fit on raw rows  (n = {len(yr):,}):")
    print(f"  intercept = {model_raw.intercept_:.4f} ms")
    print(f"  slope     = {model_raw.coef_[0]*1000:.4f} us/step")
    print(f"  R^2       = {r2_r:.4f}")

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.scatter(df["step_id"], df["latency_ms"], s=1, alpha=0.015,
               color="#4477AA", rasterized=True,
               label=f"per-(request, step) latency  (n = {len(df):,})")
    # Draw the fit at the endpoints only (sorted), so the line renders cleanly.
    xs = np.array([df["step_id"].min(), df["step_id"].max()]).reshape(-1, 1)
    ax.plot(xs.ravel(), model_raw.predict(xs), color="#CC3311", lw=1.8,
            label=(f"linear fit:  {model_raw.intercept_:.1f} + "
                   f"{model_raw.coef_[0]*1000:.2f}$\\,\\mu$s$\\cdot$step  "
                   f"($R^2$={r2_r:.3f})"))
    ax.set_xlabel("Decode step ID")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3, linestyle=":")
    leg = ax.legend(loc="upper center", frameon=False, fontsize=9)
    # Override the scatter handle's alpha in the legend so it's actually visible.
    for h in leg.legend_handles:
        try: h.set_alpha(1.0)
        except Exception: pass
    fig.tight_layout()
    fig.savefig(OUT / "decode_step_drift_raw.pdf", dpi=200)
    fig.savefig(OUT / "decode_step_drift_raw.png", dpi=200)
    print(f"Saved {OUT}/decode_step_drift_raw.pdf / .png")


if __name__ == "__main__":
    main()
