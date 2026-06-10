"""
Heatmap showing the per-cell relative difference (200W - 300W) / 300W in mean
decode-step latency, as a percentage. Positive (warm) = 300W is faster;
negative (cool) = 300W is slower (noise-level; decode is memory-bandwidth
bound, so the power cap barely matters for this workload).

Mirrors plot_decode_grid.py's heatmap layout: B on y, L on x.
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SEC3 = (REPO_ROOT / "paper/figures/section3")
OUT = SEC3 / "output"                       # write the comparison plot at the top level
DATA_200W = SEC3 / "output" / "200W" / "decode_grid_cell_summary.csv"
DATA_300W = SEC3 / "output" / "300W" / "decode_grid_cell_summary.csv"


def main():
    df200 = pd.read_csv(DATA_200W)
    df300 = pd.read_csv(DATA_300W)
    m = df200.merge(df300, on=["cell_idx", "B", "L"], suffixes=("_200", "_300"))
    # Relative difference: (200W - 300W) / 300W, as a percentage.
    m["diff_pct"] = (m["lat_mean_200"] - m["lat_mean_300"]) / m["lat_mean_300"] * 100.0

    pivot = m.pivot(index="B", columns="L", values="diff_pct")
    pivot = pivot.sort_index(ascending=False)
    # Keep only power-of-2 batch sizes on the y-axis (mirror plot_decode_grid).
    pivot = pivot.loc[[b for b in pivot.index if b > 0 and (b & (b - 1)) == 0]]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    abs_max = float(np.nanmax(np.abs(pivot.values)))
    # Symmetric divergent colormap centered at 0.
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.set_xlabel("Per-request context length  L  (tokens)", fontsize=12)
    ax.set_ylabel("Batch size  B", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("∆ Mean TBT (%)",
                   fontsize=9)

    # Per-cell annotation.
    n_cells = pivot.notna().sum().sum()
    # annot_fs = 8 if n_cells <= 40 else (7 if n_cells <= 80 else 6)
    annot_fs = 10
    for i, B in enumerate(pivot.index):
        for j, L in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                # Choose text color based on luminance of background color.
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=annot_fs, color=txt_color, fontweight="bold")

    # Iso-active-KV diagonal at B·L = 64k (mirror plot_decode_grid).
    B_grid = np.array(pivot.index, dtype=float)
    L_idx = {L: i for i, L in enumerate(pivot.columns)}
    for kv_iso in [65536]:
        pts = []
        for L in pivot.columns:
            B_target = kv_iso / L
            if B_target < min(B_grid) or B_target > max(B_grid):
                continue
            log_Bs = np.log2(B_grid)
            row = float(np.interp(np.log2(B_target), log_Bs[::-1],
                                  np.arange(len(B_grid))[::-1]))
            pts.append((L_idx[L], row))
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="black", lw=1.8, ls="--", alpha=0.75)
            kx, ky = xs[0], ys[0]
            label = f"B·L = {kv_iso//1024}k"
            ax.text(kx - 0.15, ky - 0.35, label, color="white", fontsize=10,
                    ha="left", va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="0.15",
                              ec="white", lw=0.4))

    fig.tight_layout()
    fig.savefig(OUT / "decode_grid_heatmap_diff.pdf", dpi=200)
    fig.savefig(OUT / "decode_grid_heatmap_diff.png", dpi=200)

    # Summary stats for the caption / paper text.
    diffs = m["diff_pct"].to_numpy()
    print(f"n cells: {len(diffs)}")
    print(f"relative diff (200W − 300W) / 300W  [%]:")
    print(f"  min/max     = {diffs.min():.2f} / {diffs.max():.2f}")
    print(f"  mean / std  = {diffs.mean():.3f} / {diffs.std():.3f}")
    print(f"  median      = {np.median(diffs):.3f}")
    print(f"  mean ratio (200W/300W) = {m.eval('lat_mean_200/lat_mean_300').mean():.4f}")


if __name__ == "__main__":
    main()
