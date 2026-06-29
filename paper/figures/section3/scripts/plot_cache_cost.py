"""
Plot Exp 1: prefix-cache miss vs hit prefill latency.

Each cell of the data has 2 prefill events (miss then hit) — easy to extract
from the per-cell engine log via the first step_end after each iteration_start.
With max_tokens=2 there are exactly 2 step_ends per iteration (prefill +
1 decode). We take the first per iteration as the prefill latency.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_SHORT, MODEL_DATA_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", type=str,
                 default=str(MODEL_DATA_DIR / "paper" / "section3" / "profiling"))
_ap.add_argument("--out", type=str,
                 default=str(MODEL_DATA_DIR / "paper" / "section3" / "fig2"))
_args = _ap.parse_args()
DATA = Path(_args.dir) / "cache_cost_data"
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)


def parse_cell(cell):
    eng = pd.read_json(DATA / cell["engine_log"], lines=True)
    # Walk events; first step_end after each iter_start is prefill.
    iter_idx = -1
    sip = -1
    rows = []
    for _, row in eng.iterrows():
        if row["event"] == "iteration_start":
            iter_idx += 1
            sip = -1
        elif row["event"] == "step_end":
            sip += 1
            rows.append({"iter": iter_idx, "sip": sip,
                         "latency_ms": row["step_duration_ms"]})
    df = pd.DataFrame(rows)
    # iter 0 = miss, iter 1 = hit
    miss = df.loc[(df["iter"] == 0) & (df["sip"] == 0), "latency_ms"].iloc[0]
    hit = df.loc[(df["iter"] == 1) & (df["sip"] == 0), "latency_ms"].iloc[0]
    return miss, hit


def main():
    plan = json.loads((DATA / "plan.json").read_text())
    rows = []
    for cell in plan["cells"]:
        miss, hit = parse_cell(cell)
        rows.append({"L": cell["L"], "rep": cell["rep"],
                     "miss_ms": miss, "hit_ms": hit})
    df = pd.DataFrame(rows)
    summary = (df.groupby("L")
                 .agg(miss_p50=("miss_ms", "median"), miss_mean=("miss_ms", "mean"),
                      miss_std=("miss_ms", "std"),
                      hit_p50=("hit_ms", "median"), hit_mean=("hit_ms", "mean"),
                      hit_std=("hit_ms", "std"),
                      n=("miss_ms", "size"))
                 .reset_index())
    summary.to_csv(OUT / "cache_cost_table.csv", index=False)
    print(summary.to_string(index=False))

    # Hit is ~constant across all L (block-table assembly only).
    # Miss is super-linear in L (attention compute is O(L^2)) so no clean
    # linear fit; we annotate per-L speedup ratios instead.
    hit_const = float(summary["hit_p50"].mean())

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    # Per-rep scatter
    ax.scatter(df["L"], df["miss_ms"], s=18, alpha=0.55, color="#CC3311",
               label="cache miss (raw)")
    ax.scatter(df["L"], df["hit_ms"], s=18, alpha=0.55, color="#117733",
               label="cache hit (raw)")
    # Per-L medians
    ax.plot(summary["L"], summary["miss_p50"], color="#CC3311", lw=1.8,
            marker="o", ms=5)
    ax.plot(summary["L"], summary["hit_p50"], color="#117733", lw=1.8,
            marker="s", ms=5)
    ax.axhline(hit_const, color="#117733", ls="--", lw=1.0, alpha=0.7,
               label=f"hit const $\\approx$ {hit_const:.1f} ms")

    # Configure axes BEFORE measuring transform for the bracket annotation.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prompt length L (tokens)")
    ax.set_ylabel("Prefill latency (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.canvas.draw()  # finalize transforms (log scale + tight_layout applied)

    # Hand-drawn bracket: total control over extent. Spans miss↔hit at the
    # rightmost L; wings stick out a fixed multiplicative amount on the log x.
    last = summary.iloc[-1]
    ratio = last["miss_p50"] / last["hit_p50"]
    bracket_x = last["L"]
    wing_x = last["L"] * 0.93   # how far the wings extend leftward (log-x)
    ax.plot([bracket_x, bracket_x], [last["miss_p50"], last["hit_p50"]],
            color="#222", lw=1.0, solid_capstyle="round")
    ax.plot([wing_x, bracket_x], [last["miss_p50"], last["miss_p50"]],
            color="#222", lw=1.0, solid_capstyle="round")
    ax.plot([wing_x, bracket_x], [last["hit_p50"], last["hit_p50"]],
            color="#222", lw=1.0, solid_capstyle="round")

    # Text label, freely positioned. Adjust TEXT_X / TEXT_Y to taste.
    TEXT_X = last["L"] * 0.5 + 10000
    TEXT_Y = 150
    ax.text(TEXT_X, TEXT_Y, f"{ratio:.0f}×\nspeedup",
            ha="right", va="center", fontsize=9, color="#222")
    fig.savefig(OUT / "cache_cost.pdf", dpi=200)
    fig.savefig(OUT / "cache_cost.png", dpi=200)

    # Ratio
    summary["miss_over_hit"] = summary["miss_p50"] / summary["hit_p50"]
    print()
    print("miss/hit ratio:")
    print(summary[["L", "miss_p50", "hit_p50", "miss_over_hit"]].to_string(index=False))


if __name__ == "__main__":
    main()
