"""Section-5 perf-vs-energy summary at rps=1.634 — bar chart.

2x2 panels for single-column paper layout:
  TTFET gmean | Last-turn TBT gmean
  E2E gmean   | Tokens per Joule

Each panel shows 4 policies x 2 power configs (8 bars), solid fill for
300W/300W and hatched (//) for 300W/200W. Color encodes the policy.

Energy uses workload_energy_full_corrected (AMPD virtual-prefill aware AND
prefiller-idle aware — bills GPU 0 idle baseline during the part of the live
span beyond the prefiller's busy window for ConServe and AMPD).
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_powercap import workload_energy_corrected


def _gpu0_idle_power(rd_str, t0, t1):
    """Mean GPU 0 power (W) within [t0, t1] from the live run's dcgmi trace.
    For ConServe/AMPD, GPU 0 is unused in the live run so this samples the
    idle baseline (~77 W on H100)."""
    vals = []
    with open(f"{rd_str}/dcgmi_trace.tsv") as f:
        for line in f:
            p = line.split()
            if len(p) < 5 or p[1] != "GPU" or p[2] != "0":
                continue
            try:
                ts = datetime.fromisoformat(p[0]).timestamp()
                pw = float(p[3])
            except (ValueError, IndexError):
                continue
            if t0 <= ts <= t1:
                vals.append(pw)
    return float(np.mean(vals)) if vals else 77.0


def workload_energy_full_corrected(cfg, policy, rps):
    """workload_energy_corrected + GPU 0 idle baseline billed for the part of
    the live span beyond the prefiller's serving window. Applies only to
    ConServe / AMPD (Collocated / Full Disagg already have GPU 0 in the live
    run, no idle window to bill)."""
    base = workload_energy_corrected(cfg, policy, rps)
    needs_idle = (policy == "adaptive_3eng"
                  or policy.startswith("per_turn_adaptive_disagg_decoders"))
    if not needs_idle:
        return base
    rd = s5.RPS_SWEEP / cfg / policy / f"rps_{rps}"
    ps = pd.read_csv(rd / "per_step_latency.csv")
    t0, t1 = float(ps.start_time.min()), float(ps.end_time.max())
    live_span = t1 - t0
    pref_dir = s5.prefiller_dir_for(cfg, rps)
    ps_p = pd.read_csv(pref_dir / "per_step_latency.csv")
    pref_busy_span = float(ps_p.end_time.max() - ps_p.start_time.min())
    extra_idle = max(0.0, live_span - pref_busy_span)
    if extra_idle <= 0:
        return base
    p_idle = _gpu0_idle_power(str(rd), t0, t1)
    return base + p_idle * extra_idle

OUT = Path(__file__).parent.parent / "output"
RPS = "1.634"

POLICIES = [
    ("no_disagg",                              "Collocated",   "#4477AA"),
    ("all_disagg",                             "Full Disagg",  "#CC3311"),
    ("per_turn_adaptive_disagg_decoders_p10",  "AMPD",         "#AA4499"),
    ("adaptive_3eng",                          "ConServe",     "#117733"),
]
POWER_CFGS = [
    ("p300_d300", "300/300", ""),
    ("p300_d200", "300/200", "//"),
]


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
                ttfet_g=float(gmean(ttfet_n)),
                tbt_g=float(gmean(tbt_n)),
                e2e_g=float(gmean(e2e_n)),
                ttfet_p95=float(np.percentile(ttfet_n, 95)),
                tbt_p95=float(np.percentile(tbt_n, 95)),
                e2e_p95=float(np.percentile(e2e_n, 95)),
                tpj=tokens / energy,
            )
    return data


def draw_panel(ax, data, metric, title, slo_line=None, ylog=False,
               ylim=None, show_ylabel_ticks=True, show_xtick_labels=True):
    n_policies = len(POLICIES)
    n_caps = len(POWER_CFGS)
    bar_w = 0.36
    x_base = np.arange(n_policies)

    for ci, (cfg, cfg_lab, hatch) in enumerate(POWER_CFGS):
        offset = (ci - (n_caps - 1) / 2.0) * bar_w
        for pi, (pol, plabel, color) in enumerate(POLICIES):
            v = data[(cfg, plabel)][metric]
            ax.bar(x_base[pi] + offset, v, bar_w,
                   color=color if hatch == "" else "white",
                   edgecolor=color, hatch=hatch, linewidth=1.0)

    if slo_line is not None:
        ax.axhline(slo_line, color="#BB5566", ls="--", lw=0.9, alpha=0.8)
    if ylog:
        # Symlog: linear 1-7 (the SLO-relevant range), log above 7 so the
        # Full Disagg / AMPD over-SLO bars still fit on the same panel.
        ax.set_yscale("symlog", linthresh=7.0, linscale=1.0)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 10, 25, 50, 100])
        ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7", "10", "25", "50", "100"],
                            fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8)

    ax.set_xticks(x_base)
    if show_xtick_labels:
        ax.set_xticklabels([lab for _, lab, _ in POLICIES], fontsize=8,
                           rotation=20, ha="right")
    else:
        ax.set_xticklabels([])
    if not show_ylabel_ticks:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(title, fontsize=10, pad=2)
    ax.grid(True, axis="y", which="both", alpha=0.25, ls=":")


def main():
    data = collect()

    fig, axes = plt.subplots(2, 2, figsize=(5, 5),
                              sharex="col",
                              gridspec_kw=dict(wspace=0.18, hspace=0.18))
    # Share y across the three latency panels (TTFET, TBT, E2E gmean).
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 0].sharey(axes[0, 0])

    LAT_YLIM = (1, 25)   # 1-7 linear, 7-25 log; Full Disagg gm caps at ~19
    draw_panel(axes[0, 0], data, "ttfet_g", "TTFET gmean",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=True, show_xtick_labels=False)
    draw_panel(axes[0, 1], data, "tbt_g", "Last-turn TBT gmean",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=False, show_xtick_labels=False)
    draw_panel(axes[1, 0], data, "e2e_g", "E2E gmean",
               slo_line=5.0, ylog=True, ylim=LAT_YLIM,
               show_ylabel_ticks=True, show_xtick_labels=True)
    draw_panel(axes[1, 1], data, "tpj", "Tokens per Joule",
               ylim=(20, 85),
               show_ylabel_ticks=True, show_xtick_labels=True)

    # Legend: power-cap fills (2) + SLO line. Policy colors are read off the
    # shared x-tick labels in the bottom row, so the legend stays minimal.
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

    # Manual margins (avoid tight_layout's warning when sharing y across non-
    # adjacent axes) + bbox_inches='tight' on save trims the outer white area.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.13)
    fig.savefig(OUT / "perf_energy_bar.pdf", dpi=200, bbox_inches="tight",
                pad_inches=0.03)
    fig.savefig(OUT / "perf_energy_bar.png", dpi=200, bbox_inches="tight",
                pad_inches=0.03)
    print("Saved perf_energy_bar.pdf / .png\n")

    print(f"{'policy':>14}  {'cap':>8}  "
          f"{'TTFET gm':>9}  {'TBT gm':>8}  {'E2E gm':>8}  "
          f"{'TTFET p95':>10}  {'TBT p95':>9}  {'E2E p95':>9}  "
          f"{'TPJ':>7}")
    for pol, lab, _ in POLICIES:
        for cfg, cap_lab, _ in POWER_CFGS:
            d = data[(cfg, lab)]
            print(f"{lab:>14}  {cap_lab:>8}  "
                  f"{d['ttfet_g']:>9.2f}  {d['tbt_g']:>8.2f}  {d['e2e_g']:>8.2f}  "
                  f"{d['ttfet_p95']:>10.2f}  {d['tbt_p95']:>9.2f}  {d['e2e_p95']:>9.2f}  "
                  f"{d['tpj']:>7.2f}")


if __name__ == "__main__":
    main()
