"""Section-5: sensitivity of the AMPD-style per_turn baseline to its
wrong-prediction rate, at the saturation operating point (p300_d300, rps_1.634).

For each `wrong_pred_pct ∈ {0, 0.05, ..., 0.50}` we show:
  - gmean / p95 normalized TTFET, last-turn TBT, E2E (3 curves)
  - SLO violation % for each of TTFET / TBT / E2E
  - tokens-per-joule (energy efficiency)

wrong_pct=0 reuses the adaptive_3eng results (which equals "no wrong predict").
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # noqa: F401  (used for the SLO legend entry)
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_powercap import workload_energy_corrected   # AMPD virtual-prefill aware

OUT = Path(__file__).parent.parent / "output"
CFG = "p300_d300"
RPS = "1.634"
PCTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PCT_LABELS = [f"{int(p*100)}" for p in PCTS]


def policy_for(pct):
    if pct == 0.0:
        return "adaptive_3eng"
    return f"per_turn_adaptive_disagg_decoders_p{int(pct*100):02d}"


def main():
    base = s5.load_baseline()
    base4 = s5.baseline_lastturn_tbt()

    # data[metric][pct] = array of per-conv normalized values
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
        # Use the AMPD-aware corrected energy so the virtual-prefill busy time
        # is billed at higher wrong_pct (otherwise TPJ looks artificially flat).
        energy = workload_energy_corrected(CFG, pol, RPS)
        tpj[pct] = tokens / energy

    P95 = lambda x: np.percentile(x, 95)

    cols = [("TTFET", "TTFET", "#117733", "o"),
            ("TBT",   "Last-turn TBT", "#4477AA", "s"),
            ("E2E",   "E2E",   "#CC3311", "^")]

    def _draw(axes, variant):
        # gmean normalized
        ax = axes[0]
        for m, label, color, mk in cols:
            ys = [gmean(norm[m][p]) for p in PCTS]
            ax.plot(PCTS, ys, color=color, marker=mk, ms=5, lw=1.8, label=label)
        ax.axhline(5.0, color="#BB5566", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.set_title("a. Normalized gmean latencies", fontsize=10)
        # Extend top so the rotated "ConServe" banner has room above the dots
        # (gmean data tops out ~5.5; without this, the banner runs into them).
        ax.set_ylim(2, 6)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.22, ls=":")

        # p95 normalized
        ax = axes[1]
        for m, label, color, mk in cols:
            ys = [P95(norm[m][p]) for p in PCTS]
            ax.plot(PCTS, ys, color=color, marker=mk, ms=5, lw=1.8, label=label)
        ax.axhline(5.0, color="#BB5566", ls="--", lw=1.0, alpha=0.8)
        ax.set_title("b. Normalized p95 latencies", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.22, ls=":")

        # SLO violation %
        ax = axes[2]
        for m, label, color, mk in cols:
            ys = [slo[m][p] for p in PCTS]
            ax.plot(PCTS, ys, color=color, marker=mk, ms=5, lw=1.8, label=label)
        ax.set_title("c. SLO violation", fontsize=10)
        ax.set_ylim(0, 60)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.22, ls=":")

        # Tokens per Joule
        ax = axes[3]
        ys = [tpj[p] for p in PCTS]
        # AMPD curve across the full sweep — at 0% AMPD reduces to ConServe so
        # the curve naturally starts on the ConServe baseline.
        ax.plot(PCTS, ys, color="#AA4499", marker="D", ms=5, lw=1.8,
                label="AMPD tokens / J")
        # Horizontal reference line: ConServe tokens / J (= pct=0 cell).
        conserve_tpj = tpj[0.0]
        ax.axhline(conserve_tpj, color="#117733", ls="--", lw=2, alpha=0.85)
        # Right-edge annotation for the ConServe baseline.
        ax.text(PCTS[-1]+0.02, conserve_tpj-0.5, "ConServe", color="#117733",
                fontsize=8, fontweight="bold", ha="right", va="center")
        ax.set_title("d. Energy efficiency", fontsize=10)
        ax.set_ylabel("tokens / J", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.22, ls=":")

        # Common x-axis treatment + variant-specific 0%-annotation.
        for k, ax in enumerate(axes):
            ax.set_xticks(PCTS)
            if variant == "opt1":
                # Tick-label augment: replace the "0%" tick text.
                labels_x = ["0%\n(= ConServe)"] + [f"{int(p*100)}%" for p in PCTS[1:]]
            else:
                labels_x = [f"{int(p*100)}%" for p in PCTS]
            ax.set_xticklabels(labels_x, fontsize=8, rotation=30)
            ax.set_xlabel("Wrong-prediction rate", fontsize=10)

            # Banner only on the QoS panels (a, b, c) — skip the tokens/J panel.
            if variant == "opt3" and k < len(axes) - 1:
                # Thin shaded band at 0% with a small text label at the top.
                ax.axvspan(-0.012, 0.012, color="#117733", alpha=0.10, zorder=0)
                ymax = ax.get_ylim()[1]
                ax.text(0, ymax * 0.97, "ConServe", ha="center", va="top",
                        fontsize=8, color="#117733", fontweight="bold",
                        rotation=90)

    # MLSys single-column: ~3.3" wide. Two layouts to compare.
    XTICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    XLABELS = [f"{int(p*100)}%" for p in XTICKS]

    def _common_axis_treatment(ax, is_bottom):
        ax.set_xticks(XTICKS)
        if is_bottom:
            ax.set_xticklabels(XLABELS, fontsize=8)
            ax.set_xlabel("Wrong-prediction rate", fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=7)

    # Vertical 4x1 layout (single MLSys column).
    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    _draw(axes, "opt3")
    for i, ax in enumerate(axes):
        _common_axis_treatment(ax, is_bottom=(i == len(axes) - 1))
    handles, labels = axes[0].get_legend_handles_labels()
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=1.0,
                        label="SLO (5$\\times$ baseline)")
    handles.append(slo_handle)
    labels.append(slo_handle.get_label())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "wrong_pct_sweep.pdf", dpi=200)
    fig.savefig(OUT / "wrong_pct_sweep.png", dpi=200)
    plt.close(fig)
    print("Saved wrong_pct_sweep.pdf / .png\n")

    # Tabular dump
    print(f"{'pct':>5}  {'gm.TTFET':>9}  {'gm.TBT':>8}  {'gm.E2E':>8}  "
          f"{'p95.TTFET':>10}  {'p95.TBT':>9}  {'p95.E2E':>9}  "
          f"{'SLO.TTFET':>10}  {'SLO.TBT':>9}  {'SLO.E2E':>9}  {'tok/J':>7}")
    for p in PCTS:
        print(f"{int(p*100):>4}%  "
              f"{gmean(norm['TTFET'][p]):>9.2f}  "
              f"{gmean(norm['TBT'][p]):>8.2f}  "
              f"{gmean(norm['E2E'][p]):>8.2f}  "
              f"{P95(norm['TTFET'][p]):>10.2f}  "
              f"{P95(norm['TBT'][p]):>9.2f}  "
              f"{P95(norm['E2E'][p]):>9.2f}  "
              f"{slo['TTFET'][p]:>9.1f}%  "
              f"{slo['TBT'][p]:>8.1f}%  "
              f"{slo['E2E'][p]:>8.1f}%  "
              f"{tpj[p]:>7.2f}")


if __name__ == "__main__":
    main()
