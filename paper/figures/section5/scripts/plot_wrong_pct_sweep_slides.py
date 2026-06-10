"""Section-5 wrong-pct sensitivity — SLIDE variant (2x2 layout, larger fonts).

Same data and curves as plot_wrong_pct_sweep.py, just rearranged into a 2x2
grid suitable for presentation slides:

  a. gmean latencies         |  b. p95 latencies
  c. SLO violation %         |  d. Energy efficiency (tokens / J)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_powercap import workload_energy_corrected

OUT = Path(__file__).parent.parent / "output"
CFG = "p300_d300"
RPS = "1.634"
PCTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Big font knobs (one place to tweak for slides).
TITLE_FS  = 14
LABEL_FS  = 13
TICK_FS   = 11
LEGEND_FS = 12
ANNOT_FS  = 11


def policy_for(pct):
    if pct == 0.0:
        return "adaptive_3eng"
    return f"per_turn_adaptive_disagg_decoders_p{int(pct*100):02d}"


def main():
    base = s5.load_baseline()
    base4 = s5.baseline_lastturn_tbt()

    norm = {m: {} for m in ("TTFET", "TBT", "E2E")}
    slo  = {m: {} for m in ("TTFET", "TBT", "E2E")}
    tpj  = {}

    for pct in PCTS:
        pol = policy_for(pct)
        run_dir = s5.RPS_SWEEP / CFG / pol / f"rps_{RPS}"
        df = s5.load_run(CFG, pol, RPS).set_index("conv_id").join(base)
        norm["TTFET"][pct] = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
        norm["E2E"][pct]   = (df["e2e"]   / df["base_e2e"]).dropna().to_numpy()
        norm["TBT"][pct]   = s5.lastturn_tbt_tokens(CFG, pol, RPS)

        vt = df[["ttfet", "slo_ttfet"]].dropna()
        slo["TTFET"][pct] = float((vt["ttfet"] > vt["slo_ttfet"]).mean() * 100)
        ve = df[["e2e", "slo_e2e"]].dropna()
        slo["E2E"][pct]   = float((ve["e2e"] > ve["slo_e2e"]).mean() * 100)
        gaps = s5._iter4_ordered_gaps(str(run_dir), pol)
        viol = tot = 0
        for conv, g in gaps.items():
            if conv in base4 and g:
                tot += 1
                viol += np.mean(g) > 5.0 * base4[conv]
        slo["TBT"][pct] = 100.0 * viol / tot if tot else float("nan")

        tokens = s5.workload_tokens_full_context(CFG, pol, RPS)
        energy = workload_energy_corrected(CFG, pol, RPS)
        tpj[pct] = tokens / energy

    P95 = lambda x: np.percentile(x, 95)

    cols = [("TTFET", "TTFET",         "#117733", "o"),
            ("TBT",   "Last-turn TBT", "#4477AA", "s"),
            ("E2E",   "E2E",           "#CC3311", "^")]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True,
                              gridspec_kw=dict(wspace=0.22, hspace=0.32))

    # (0,0) gmean
    ax = axes[0, 0]
    for m, label, color, mk in cols:
        ys = [gmean(norm[m][p]) for p in PCTS]
        ax.plot(PCTS, ys, color=color, marker=mk, ms=7, lw=2.2, label=label)
    ax.axhline(5.0, color="#BB5566", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(1.0, color="gray",    ls=":",  lw=1.0, alpha=0.7)
    ax.set_title("a. Normalized gmean latencies", fontsize=TITLE_FS)
    ax.set_ylim(2, 6)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(True, alpha=0.22, ls=":")

    # (0,1) p95
    ax = axes[0, 1]
    for m, label, color, mk in cols:
        ys = [P95(norm[m][p]) for p in PCTS]
        ax.plot(PCTS, ys, color=color, marker=mk, ms=7, lw=2.2, label=label)
    ax.axhline(5.0, color="#BB5566", ls="--", lw=1.2, alpha=0.8)
    ax.set_title("b. Normalized p95 latencies", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(True, alpha=0.22, ls=":")

    # (1,0) SLO violation %
    ax = axes[1, 0]
    for m, label, color, mk in cols:
        ys = [slo[m][p] for p in PCTS]
        ax.plot(PCTS, ys, color=color, marker=mk, ms=7, lw=2.2, label=label)
    ax.set_title("c. SLO violation", fontsize=TITLE_FS)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(True, alpha=0.22, ls=":")

    # (1,1) Tokens per Joule
    ax = axes[1, 1]
    ys = [tpj[p] for p in PCTS]
    ax.plot(PCTS, ys, color="#AA4499", marker="D", ms=7, lw=2.2,
            label="AMPD tokens / J")
    conserve_tpj = tpj[0.0]
    ax.axhline(conserve_tpj, color="#117733", ls="--", lw=2.2, alpha=0.85)
    ax.text(PCTS[-1] + 0.02, conserve_tpj - 0.4, "ConServe",
            color="#117733", fontsize=ANNOT_FS, fontweight="bold",
            ha="right", va="center")
    ax.set_title("d. Energy efficiency", fontsize=TITLE_FS)
    ax.set_ylabel("tokens / J", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(True, alpha=0.22, ls=":")

    # Shared axis treatment: ConServe band on QoS panels, x-tick labels on
    # the bottom row only.
    bottom_row = {(1, 0), (1, 1)}
    qos_panels = {(0, 0), (0, 1), (1, 0)}
    XTICKS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    XLABELS = [f"{int(p*100)}%" for p in XTICKS]
    for (i, j), ax in np.ndenumerate(axes):
        ax.set_xticks(XTICKS)
        if (i, j) in bottom_row:
            ax.set_xticklabels(XLABELS, fontsize=TICK_FS)
            ax.set_xlabel("Wrong-prediction rate", fontsize=LABEL_FS)
        else:
            ax.set_xticklabels([])
        if (i, j) in qos_panels:
            ax.axvspan(-0.012, 0.012, color="#117733", alpha=0.10, zorder=0)
            ymax = ax.get_ylim()[1]
            ax.text(0, ymax * 0.97, "ConServe", ha="center", va="top",
                    fontsize=ANNOT_FS, color="#117733", fontweight="bold",
                    rotation=90)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=1.2,
                        label="SLO (5$\\times$ baseline)")
    handles.append(slo_handle)
    labels.append(slo_handle.get_label())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=LEGEND_FS,
               handlelength=2.0, columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "wrong_pct_sweep_slides.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "wrong_pct_sweep_slides.png", dpi=200,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Saved wrong_pct_sweep_slides.pdf / .png")


if __name__ == "__main__":
    main()
