"""
Combined 2x2 interference figure:
  rows = sweep axis    (top: B_decode with L_decoder=78; bottom: L_decoder with B=8)
  cols = cache state   (left: miss, right: hit)

Reuses the per-cell summaries that plot_interference.py and
plot_interference_kv.py write to interference_summary.csv and
interference_kv_summary.csv.
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "profiling"))
from config import MODEL_SHORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = (REPO_ROOT / "paper/figures/section3/output" / MODEL_SHORT / "300W")
STEP_COL = "prefill_step_mean_ms"


def main():
    df_B = pd.read_csv(OUT / "interference_summary.csv")
    df_L = pd.read_csv(OUT / "interference_kv_summary.csv")
    df_B = df_B[df_B["L_prefill"] != 1024]
    df_L = df_L[df_L["L_prefill"] != 1024]

    L_pref_vals = sorted(set(df_B["L_prefill"].unique()) | set(df_L["L_prefill"].unique()))
    cmap = plt.get_cmap("viridis")
    L_colors = {L: cmap(i / max(1, len(L_pref_vals) - 1)) for i, L in enumerate(L_pref_vals)}

    # Shared y-range across both panels
    y_vals = pd.concat([df_B[STEP_COL], df_L[STEP_COL]]).dropna()
    y_lo, y_hi = float(y_vals.min()) * 0.7, float(y_vals.max()) * 1.4

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2), sharey=True)

    # Top: B sweep — mean line (miss solid / hit dashed) + raw per-rep dots.
    # B=0 (prefill alone, no decoders) is dropped: it can't sit on a log x-axis.
    ax = axes[0]
    for L in sorted(df_B["L_prefill"].unique()):
        for cache_hit, ls in [(False, "-"), (True, "--")]:
            s = df_B[(df_B["L_prefill"] == L) & (df_B["cache_hit"] == cache_hit)
                     & (df_B["B_decode"] >= 1)]
            ax.scatter(s["B_decode"], s[STEP_COL], s=5, alpha=0.2,
                       color=L_colors[L], rasterized=True)
            g = s.groupby("B_decode")[STEP_COL].mean().sort_index()
            ax.plot(g.index, g.values, lw=1.8, color=L_colors[L], linestyle=ls)
    ax.set_title("a. Varying decoder batch size", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.set_ylabel("Iteration latency (ms)", fontsize=12)
    ax.set_xlabel("Batch size", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    # Bottom: L_decoder sweep at B=8 — mean line + raw per-rep dots.
    ax = axes[1]
    for L in sorted(df_L["L_prefill"].unique()):
        for cache_hit, ls in [(False, "-"), (True, "--")]:
            s = df_L[(df_L["L_prefill"] == L) & (df_L["cache_hit"] == cache_hit)]
            ax.scatter(s["L_decoder"], s[STEP_COL], s=5, alpha=0.2,
                       color=L_colors[L], rasterized=True)
            g = s.groupby("L_decoder")[STEP_COL].mean().sort_index()
            ax.plot(g.index, g.values, lw=1.8, color=L_colors[L], linestyle=ls)
    ax.set_title("b. Varying decoder context length", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.set_xlabel("Individual context length (tokens)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    # Legend: row 1 = "Prefill tokens" inline title + the prefill-length
    # colors; row 2 = the prefix-cache miss/hit linestyle key.
    L_title_handle = Line2D([], [], linestyle="none", marker="none",
                          label="Prefill tokens")
    L_handles = [Line2D([0], [0], color=L_colors[L], lw=1.8, label=str(L))
                 for L in L_pref_vals]
    style_title_handles = Line2D([], [], linestyle="none", marker="none",
                          label="Prefix-cache")
    style_handles = [Line2D([0], [0], color="gray", lw=1.8, linestyle="-",
                            label="miss"),
                     Line2D([0], [0], color="gray", lw=1.8, linestyle="--",
                            label="hit")]
    # Block 1: "Prefill tokens" title + the prefill-length colors, one row.
    fig.legend(handles=[L_title_handle] + L_handles,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=1 + len(L_handles), frameon=False, fontsize=12,
               handletextpad=0.4, columnspacing=1.2)
    # Block 2: the miss/hit linestyle key, its own row just below.
    fig.legend(handles=[style_title_handles] + style_handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=1 + len(style_handles), frameon=False, fontsize=12,
               handletextpad=0.4, columnspacing=1.2)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "interference_combined.pdf", dpi=200)
    fig.savefig(OUT / "interference_combined.png", dpi=200)
    print("Saved interference_combined.pdf / .png")


if __name__ == "__main__":
    main()
