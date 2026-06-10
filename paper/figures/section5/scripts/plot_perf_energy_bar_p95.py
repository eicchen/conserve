"""Section-5 perf-vs-energy summary at rps=1.634 — bar chart, p95 variant.

Same 2x2 layout and styling as plot_perf_energy_bar.py, but the three latency
panels show p95 (instead of gmean) of TTFET, last-turn TBT, and E2E. TPJ uses
the idle-corrected accounting (workload_energy_full_corrected).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_perf_energy_bar import (
    POLICIES, POWER_CFGS, RPS, draw_panel,
    workload_energy_full_corrected,
)

OUT = Path(__file__).parent.parent / "output"


def collect():
    base = s5.load_baseline()
    data = {}
    for cfg, _, _ in POWER_CFGS:
        for pol, lab, _ in POLICIES:
            df = s5.load_run(cfg, pol, RPS).set_index("conv_id").join(base)
            ttfet_n = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
            e2e_n   = (df["e2e"]   / df["base_e2e"]).dropna().to_numpy()
            tbt_n   = s5.lastturn_tbt_tokens(cfg, pol, RPS)
            tokens = s5.workload_tokens_full_context(cfg, pol, RPS)
            energy = workload_energy_full_corrected(cfg, pol, RPS)
            data[(cfg, lab)] = dict(
                ttfet_p95=float(np.percentile(ttfet_n, 95)),
                tbt_p95=float(np.percentile(tbt_n, 95)),
                e2e_p95=float(np.percentile(e2e_n, 95)),
                tpj=tokens / energy,
            )
    return data


def main():
    data = collect()

    fig, axes = plt.subplots(2, 2, figsize=(5, 5),
                              sharex="col",
                              gridspec_kw=dict(wspace=0.18, hspace=0.18))
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 0].sharey(axes[0, 0])

    # 1-7 linear (SLO-relevant), 7-150 log so Full Disagg p95 ~104 still fits.
    LAT_YLIM = (1, 150)
    draw_panel(axes[0, 0], data, "ttfet_p95", "TTFET p95",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=True, show_xtick_labels=False)
    draw_panel(axes[0, 1], data, "tbt_p95", "Last-turn TBT p95",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=False, show_xtick_labels=False)
    draw_panel(axes[1, 0], data, "e2e_p95", "E2E p95",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=True, show_xtick_labels=True)
    draw_panel(axes[1, 1], data, "tpj", "Tokens per Joule",
               ylim=(20, 85),
               show_ylabel_ticks=True, show_xtick_labels=True)

    cap_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="300W / 300W"),
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="300W / 200W"),
    ]
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=0.9,
                        label="SLO (5$\\times$)")
    fig.legend(handles=cap_handles + [slo_handle],
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=9, handlelength=1.3,
               columnspacing=1.0, handletextpad=0.3)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.13)
    fig.savefig(OUT / "perf_energy_bar_p95.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "perf_energy_bar_p95.png", dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    print("Saved perf_energy_bar_p95.pdf / .png\n")

    print(f"{'policy':>14}  {'cap':>8}  "
          f"{'TTFET p95':>10}  {'TBT p95':>9}  {'E2E p95':>9}  {'TPJ':>7}")
    for pol, lab, _ in POLICIES:
        for cfg, cap_lab, _ in POWER_CFGS:
            d = data[(cfg, lab)]
            print(f"{lab:>14}  {cap_lab:>8}  "
                  f"{d['ttfet_p95']:>10.2f}  {d['tbt_p95']:>9.2f}  "
                  f"{d['e2e_p95']:>9.2f}  {d['tpj']:>7.2f}")


if __name__ == "__main__":
    main()
