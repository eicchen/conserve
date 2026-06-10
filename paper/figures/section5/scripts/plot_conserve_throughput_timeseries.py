"""ConServe output-token throughput vs time at rps=1.634.

For each cap (300/300 and 300/200), bin every decoded token's emission
timestamp (from the decoder core logs) into 1-second buckets and plot
tokens/sec over time. A rolling-mean overlay smooths the per-second curve.

Dotted vertical = last iter-0 arrival time (drain phase begins).
Dashed vertical = last iter-4 completion (workload done).
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
RPS = "1.634"
POL = "adaptive_3eng"
CFGS = [
    ("p300_d300", "300/300", "#117733"),
    ("p300_d200", "300/200", "#88CCEE"),
]
BUCKET = 1.0      # seconds per bin
ROLL_S = 10       # rolling-mean window in seconds


def collect(cfg):
    rd = s5.RPS_SWEEP / cfg / POL / f"rps_{RPS}"
    ps = pd.read_csv(rd / "per_step_latency.csv")
    t0 = float(ps["start_time"].min())
    iter0 = ps[ps.step_id == 0]
    iter4 = ps[ps.step_id == 4]
    last_arr  = float(iter0["start_time"].max()) - t0
    last_done = float(iter4["end_time"].max())   - t0

    # All decode-token emission times across the 3 decoders, relative to t0.
    token_ts = []
    for lf in sorted(glob.glob(os.path.join(str(rd), "decoder*_vllm_core_log.jsonl"))):
        for st, en, ex, fin in s5.parse_core_log(lf):
            for _rid in fin:
                token_ts.append(en - t0)
    token_ts = np.asarray(token_ts)
    if len(token_ts) == 0:
        return None
    edges = np.arange(0, max(last_done, token_ts.max()) + BUCKET, BUCKET)
    counts, _ = np.histogram(token_ts, bins=edges)
    tps = counts / BUCKET
    centers = (edges[:-1] + edges[1:]) / 2.0
    # rolling mean
    win = int(round(ROLL_S / BUCKET))
    rolled = pd.Series(tps).rolling(win, min_periods=max(1, win // 2)).mean().to_numpy()
    return dict(
        centers=centers, tps=tps, rolled=rolled,
        last_arr=last_arr, last_done=last_done,
        total_tokens=int(token_ts.size),
        span=last_done,
    )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))

    for cfg, cap_lab, color in CFGS:
        d = collect(cfg)
        if d is None:
            continue
        ax.plot(d["centers"], d["tps"], color=color, lw=0.7, alpha=0.35)
        ax.plot(d["centers"], d["rolled"], color=color, lw=1.8,
                label=f"ConServe @ {cap_lab}  ({d['total_tokens']:,} tokens, "
                      f"span {d['span']:.0f}s, avg {d['total_tokens']/d['span']:.0f} tok/s)")
        ax.axvline(d["last_arr"],  color=color, ls=":",  lw=0.9, alpha=0.7)
        ax.axvline(d["last_done"], color=color, ls="--", lw=0.9, alpha=0.7)

    ax.set_xlabel("Time since first iter-0 (s)", fontsize=10)
    ax.set_ylabel("Output tokens / sec  (across 3 decoders)", fontsize=10)
    ax.set_title(f"ConServe output throughput @ rps={RPS}", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3, ls=":")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.text(0.5, 0.005,
             "thin = per-second bins; thick = 10s rolling mean.  "
             "dotted = last iter-0 arrival; dashed = workload done.",
             ha="center", fontsize=8, style="italic", color="#444")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(OUT / "conserve_throughput_timeseries.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "conserve_throughput_timeseries.png", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    print("Saved conserve_throughput_timeseries.pdf / .png\n")

    print(f"{'cap':>8}  {'span(s)':>8}  {'tokens':>9}  {'avg tok/s':>9}  "
          f"{'peak roll':>9}  {'p50 roll':>9}")
    for cfg, cap_lab, _ in CFGS:
        d = collect(cfg)
        if d is None: continue
        valid = d["rolled"][~np.isnan(d["rolled"])]
        print(f"{cap_lab:>8}  {d['span']:>8.1f}  {d['total_tokens']:>9,}  "
              f"{d['total_tokens']/d['span']:>9.0f}  "
              f"{valid.max():>9.0f}  {np.percentile(valid,50):>9.0f}")


if __name__ == "__main__":
    main()
