"""
Plot the input/output-token profile of the mini_swe_agent trace, per turn.

Single panel, x-axis = turn index; at each turn two paired violins:
  Appended input tokens (turn-1: prompt; turn-2+: new chunk)
  Output tokens

Outputs (written to BENCHMARK_TRACE_DIR, alongside mini_swe_agent_trace.json —
same per-benchmark model_outputs dir the trace was read from, so plotting
benchmarks one after another doesn't overwrite each other's output):
  trace_profile_<BENCHMARK_SHORT>.png
  trace_profile_summary_<BENCHMARK_SHORT>.csv
"""

import json
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "config"))
from config import BENCHMARK_SHORT, BENCHMARK_TRACE_DIR

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TRACE = BENCHMARK_TRACE_DIR / "mini_swe_agent_trace.json"
OUT = BENCHMARK_TRACE_DIR
# Suffixed with BENCHMARK_SHORT for consistency with the old shared-output-dir
# naming, even though OUT is now already per-benchmark.
PNG_PATH = OUT / f"trace_profile_{BENCHMARK_SHORT}.png"
CSV_PATH = OUT / f"trace_profile_summary_{BENCHMARK_SHORT}.csv"


def load_trace():
    raw = json.loads(TRACE.read_text())
    by_conv = defaultdict(list)
    for r in raw:
        by_conv[r["conv_id"]].append(r)
    rows = []
    for cid, turns in by_conv.items():
        turns.sort(key=lambda x: x["iter_id"])
        cumulative_in = 0
        cumulative_out = 0
        for t in turns:
            appended = t["in_token_size"]
            out = t["out_token_size"]
            # total_input at this turn = all previous in chunks + all previous outputs + this new chunk
            total_input = cumulative_in + cumulative_out + appended
            rows.append({
                "conv_id": cid,
                "iter_id": t["iter_id"],
                "appended_input": appended,
                "total_input": total_input,
                "out_token_size": out,
                "carried_over": total_input - appended,  # = cumulative_in + cumulative_out
            })
            cumulative_in += appended
            cumulative_out += out
    return pd.DataFrame(rows)


def main():
    df = load_trace()
    df.to_csv(CSV_PATH, index=False)
    print(f"loaded {len(df)} turns across {df['conv_id'].nunique()} convs")
    summary = (df.groupby("iter_id")
                 .agg(n=("conv_id", "size"),
                      total_input_p50=("total_input", "median"),
                      total_input_mean=("total_input", "mean"),
                      total_input_min=("total_input", "min"),
                      total_input_max=("total_input", "max"),
                      out_p50=("out_token_size", "median"),
                      out_p99=("out_token_size", lambda x: float(np.percentile(x, 99))),
                      appended_p50=("appended_input", "median"),
                      carried_p50=("carried_over", "median"))
                 .reset_index())
    print(summary.to_string(index=False))

    iters = sorted(df["iter_id"].unique())
    positions = [it + 1 for it in iters]
    appended_by_turn = [df.loc[df["iter_id"] == it, "appended_input"].to_numpy()
                        for it in iters]
    output_by_turn = [df.loc[df["iter_id"] == it, "out_token_size"].to_numpy()
                      for it in iters]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def style_violin(vp, color):
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        for k in ("cbars", "cmins", "cmaxes", "cmedians"):
            if k in vp:
                vp[k].set_color(color)
                vp[k].set_linewidth(1.2)

    offset = 0.22
    width = 0.40
    YMIN = 10
    YMAX = 100_000

    vp_in = ax.violinplot(
        [np.log10(np.clip(v, YMIN, None)) for v in appended_by_turn],
        positions=[p - offset for p in positions], widths=width,
        showmedians=True, showextrema=True)
    style_violin(vp_in, "#4477AA")

    vp_out = ax.violinplot(
        [np.log10(np.clip(v, YMIN, None)) for v in output_by_turn],
        positions=[p + offset for p in positions], widths=width,
        showmedians=True, showextrema=True)
    style_violin(vp_out, "#117733")

    yt = [10, 100, 1_000, 10_000, 100_000]
    ax.set_yticks(np.log10(yt))
    ax.set_yticklabels([f"{v:,}" for v in yt], fontsize=9)
    ax.set_ylim(np.log10(YMIN), np.log10(YMAX))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], fontsize=9)
    ax.set_xlabel("Turn index", fontsize=11)
    ax.set_ylabel("Tokens (log scale)", fontsize=11)
    ax.grid(True, axis="y", which="both", alpha=0.3, linestyle=":")

    legend_handles = [
        Patch(facecolor="#4477AA", edgecolor="#4477AA", alpha=0.55,
              label="Appended input"),
        Patch(facecolor="#117733", edgecolor="#117733", alpha=0.55,
              label="Output"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11, frameon=False)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=200)
    print(f"\nSaved {PNG_PATH.name} and {CSV_PATH.name}")


if __name__ == "__main__":
    main()
