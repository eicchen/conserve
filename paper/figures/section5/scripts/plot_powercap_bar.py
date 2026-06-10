"""Bar-chart version of the power-cap comparison.

4 panels in a single row (TTFET p95, Last-turn TBT p95, E2E p95, Tokens / J).
x-axis: rps. Within each rps tick, 8 grouped bars: 4 policies x 2 power
configs. Color = policy; solid fill = 300W/300W, hatched (//) = 300W/200W.

Reuses plot_powercap.py's data collection (including the AMPD prefiller
energy correction).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_powercap import workload_energy_corrected   # AMPD-aware

OUT = Path(__file__).parent.parent / "output"
RPS = [0.5, 0.75, 1, 1.25, 1.5, 1.634]

POLICIES = [
    ("no_disagg",                              "Collocated",  "#4477AA"),
    ("all_disagg",                             "Full Disagg", "#CC3311"),
    ("per_turn_adaptive_disagg_decoders_p10",  "AMPD",        "#AA4499"),
    ("adaptive_3eng",                          "ConServe",   "#117733"),
]
POWER_CFGS = [
    ("p300_d300", "300/300", ""),
    ("p300_d200", "300/200", "//"),
]
P95 = lambda x: np.percentile(x, 95)


def collect_data():
    base = s5.load_baseline()
    base4 = s5.baseline_lastturn_tbt()

    # data[(cfg, policy_label)][metric][rps]
    data = {}
    for cfg, _, _ in POWER_CFGS:
        for _, lab, _ in POLICIES:
            data[(cfg, lab)] = {"TTFET_p95": {}, "TBT_p95": {}, "E2E_p95": {},
                                 "TTFET_slo": {}, "TBT_slo": {}, "E2E_slo": {},
                                 "tpj": {}}

    for cfg, _, _ in POWER_CFGS:
        for pol, lab, _ in POLICIES:
            for rps in RPS:
                run_dir = s5.RPS_SWEEP / cfg / pol / f"rps_{rps}"
                if not (run_dir / "per_step_latency.csv").exists():
                    continue
                df = s5.load_run(cfg, pol, rps).set_index("conv_id").join(base)
                ttfet = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
                e2e   = (df["e2e"]   / df["base_e2e"]).dropna().to_numpy()
                tbt   = s5.lastturn_tbt_tokens(cfg, pol, rps)
                data[(cfg, lab)]["TTFET_p95"][rps] = P95(ttfet) if len(ttfet) else np.nan
                data[(cfg, lab)]["TBT_p95"][rps]   = P95(tbt)   if len(tbt)   else np.nan
                data[(cfg, lab)]["E2E_p95"][rps]   = P95(e2e)   if len(e2e)   else np.nan

                vt = df[["ttfet", "slo_ttfet"]].dropna()
                data[(cfg, lab)]["TTFET_slo"][rps] = float((vt["ttfet"] > vt["slo_ttfet"]).mean() * 100)
                ve = df[["e2e", "slo_e2e"]].dropna()
                data[(cfg, lab)]["E2E_slo"][rps]   = float((ve["e2e"] > ve["slo_e2e"]).mean() * 100)
                gaps = s5._iter4_ordered_gaps(str(run_dir), pol)
                viol = tot = 0
                for conv, g in gaps.items():
                    if conv in base4 and g:
                        tot += 1
                        viol += np.mean(g) > 5.0 * base4[conv]
                data[(cfg, lab)]["TBT_slo"][rps] = 100.0 * viol / tot if tot else float("nan")

                tokens = s5.workload_tokens_full_context(cfg, pol, rps)
                energy = workload_energy_corrected(cfg, pol, rps)
                data[(cfg, lab)]["tpj"][rps] = tokens / energy
    return data


def draw_groups(ax, data, metric, ylabel, ylog=False, slo_line=None, ylim=None):
    n_bars_per_group = len(POLICIES) * len(POWER_CFGS)
    bar_w = 0.10
    x_base = np.arange(len(RPS))
    for pi, (pol, plabel, color) in enumerate(POLICIES):
        for ci, (cfg, cfg_lab, hatch) in enumerate(POWER_CFGS):
            ys = [data[(cfg, plabel)][metric].get(r, np.nan) for r in RPS]
            # Bar position: each (policy, cfg) gets a slot in the group of 8.
            slot = pi * len(POWER_CFGS) + ci
            offset = (slot - (n_bars_per_group - 1) / 2.0) * bar_w
            ax.bar(x_base + offset, ys, bar_w,
                   color=color if hatch == "" else "white",
                   edgecolor=color, hatch=hatch, linewidth=1.0,
                   label=f"{plabel} {cfg_lab}")

    if slo_line is not None:
        ax.axhline(slo_line, color="#BB5566", ls="--", lw=1.0, alpha=0.8,
                   label="SLO (5x baseline)")
    if metric.endswith("_p95"):
        ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)

    if ylog:
        ax.set_yscale("symlog", linthresh=5.0, linscale=2.0)
        ax.set_yticks([1, 2, 3, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["1", "2", "3", "5", "10", "20", "50", "100"], fontsize=8)
        ax.set_ylim(bottom=0.5, top=200)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(r) for r in RPS], fontsize=8)
    ax.set_xlabel("Request rate (conv / s)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.25, ls=":")
    ax.tick_params(axis="y", labelsize=8)


def main():
    data = collect_data()

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 7.6),
                              gridspec_kw=dict(wspace=0.27, hspace=0.32))

    # Row 0: p95 normalized latencies + TPJ
    draw_groups(axes[0, 0], data, "TTFET_p95", "TTFET p95 (normalized)",
                ylog=True, slo_line=5.0)
    axes[0, 0].set_title("TTFET p95", fontsize=10)

    draw_groups(axes[0, 1], data, "TBT_p95", "Last-turn TBT p95 (normalized)",
                ylog=True, slo_line=5.0)
    axes[0, 1].set_title("Last-turn TBT p95", fontsize=10)

    draw_groups(axes[0, 2], data, "E2E_p95", "E2E p95 (normalized)",
                ylog=True, slo_line=5.0)
    axes[0, 2].set_title("E2E p95", fontsize=10)

    draw_groups(axes[0, 3], data, "tpj", "tokens / J", ylim=(0, 90))
    axes[0, 3].set_title("Tokens per Joule", fontsize=10)

    # Row 1: SLO violation %
    draw_groups(axes[1, 0], data, "TTFET_slo", "TTFET SLO viol. (%)", ylim=(0, 105))
    axes[1, 0].set_title("TTFET SLO violation %", fontsize=10)

    draw_groups(axes[1, 1], data, "TBT_slo", "Last-turn TBT SLO viol. (%)", ylim=(0, 105))
    axes[1, 1].set_title("Last-turn TBT SLO violation %", fontsize=10)

    draw_groups(axes[1, 2], data, "E2E_slo", "E2E SLO viol. (%)", ylim=(0, 105))
    axes[1, 2].set_title("E2E SLO violation %", fontsize=10)

    # Use the 8th panel for a per-rps TPJ improvement vs Collocated@300/300 (just
    # to fill the slot informatively — easier than leaving it blank).
    ax_imp = axes[1, 3]
    ax_imp.set_title("TPJ gain vs Collocated@300/300 (%)", fontsize=10)
    n_bars_per_group = len(POLICIES) * len(POWER_CFGS)
    bar_w = 0.10
    x_base = np.arange(len(RPS))
    baseline = data[("p300_d300", "Collocated")]["tpj"]
    for pi, (pol, plabel, color) in enumerate(POLICIES):
        for ci, (cfg, cfg_lab, hatch) in enumerate(POWER_CFGS):
            ys = []
            for r in RPS:
                v = data[(cfg, plabel)]["tpj"].get(r)
                b = baseline.get(r)
                ys.append((v / b - 1) * 100 if (v is not None and b) else np.nan)
            slot = pi * len(POWER_CFGS) + ci
            offset = (slot - (n_bars_per_group - 1) / 2.0) * bar_w
            ax_imp.bar(x_base + offset, ys, bar_w,
                       color=color if hatch == "" else "white",
                       edgecolor=color, hatch=hatch, linewidth=1.0)
    ax_imp.axhline(0.0, color="black", lw=0.8)
    ax_imp.set_xticks(x_base)
    ax_imp.set_xticklabels([str(r) for r in RPS], fontsize=8)
    ax_imp.set_xlabel("Request rate (conv / s)", fontsize=9)
    ax_imp.set_ylabel("Δ TPJ (%)", fontsize=9)
    ax_imp.grid(True, axis="y", alpha=0.25, ls=":")
    ax_imp.tick_params(axis="y", labelsize=8)

    # Legend at top: 4 policy colors + 2 cap fills.
    policy_handles = [Patch(facecolor=c, edgecolor=c, label=lab)
                      for _, lab, c in POLICIES]
    cap_handles = [
        Patch(facecolor="lightgray", edgecolor="black", label="300W / 300W (uncapped)"),
        Patch(facecolor="white",     edgecolor="black", hatch="//",
              label="300W / 200W (decoder cap)"),
    ]
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=1.0,
                        label="SLO (5x baseline)")
    fig.legend(handles=policy_handles + cap_handles + [slo_handle],
               labels=[h.get_label() for h in policy_handles + cap_handles + [slo_handle]],
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=7, frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "powercap_bar.pdf", dpi=200)
    fig.savefig(OUT / "powercap_bar.png", dpi=200)
    print("Saved powercap_bar.pdf / .png")


if __name__ == "__main__":
    main()
