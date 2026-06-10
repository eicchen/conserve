"""Section-5 perf-vs-energy Pareto summary at rps=1.634.

Single panel:
  x = E2E p95 (normalized to per-conv baseline; lower is better)
  y = tokens / J (corrected; higher is better)

One marker per (policy, cap). The four policies have distinct colors; 300/300
uses a filled marker and 300/200 uses a hollow marker. A thin arrow connects
each policy's 300/300 -> 300/200 pair so the cap effect is visually obvious.
Vertical red dashed line marks the 5x SLO. Corner annotations show the "good"
direction.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5
from plot_powercap import workload_energy_corrected   # AMPD-aware accounting

OUT = Path(__file__).parent.parent / "output"
RPS = "1.634"

# (cfg, policy, label, color, marker)
POLICIES = [
    ("no_disagg",                              "Collocated",  "#4477AA", "s"),
    ("all_disagg",                             "Full Disagg", "#CC3311", "^"),
    ("per_turn_adaptive_disagg_decoders_p10",  "AMPD",        "#AA4499", "v"),
    ("adaptive_3eng",                          "ConServe",    "#117733", "o"),
]
POWER_CFGS = [
    ("p300_d300", "300/300", True),    # filled
    ("p300_d200", "300/200", False),   # hollow
]


def collect():
    base = s5.load_baseline()
    data = {}
    for cfg, _, _ in POWER_CFGS:
        for pol, lab, *_ in POLICIES:
            rd = s5.RPS_SWEEP / cfg / pol / f"rps_{RPS}"
            df = s5.load_run(cfg, pol, RPS).set_index("conv_id").join(base)
            e2e_n = (df["e2e"] / df["base_e2e"]).dropna().to_numpy()
            tokens = s5.workload_tokens_full_context(cfg, pol, RPS)
            energy = workload_energy_corrected(cfg, pol, RPS)
            data[(cfg, lab)] = dict(
                e2e_p95=float(np.percentile(e2e_n, 95)),
                tpj=tokens / energy,
            )
    return data


def main():
    data = collect()

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.4))

    for pol, lab, color, mk in POLICIES:
        pts = [data[(cfg, lab)] for cfg, *_ in POWER_CFGS]
        x = [p["e2e_p95"] for p in pts]
        y = [p["tpj"] for p in pts]
        # arrow from 300/300 -> 300/200
        ax.annotate("",
                    xy=(x[1], y[1]), xytext=(x[0], y[0]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                     lw=1.0, alpha=0.55,
                                     shrinkA=8, shrinkB=8))
        # 300/300 (filled)
        ax.plot(x[0], y[0], color=color, marker=mk, ms=10,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.4, lw=0)
        # 300/200 (hollow)
        ax.plot(x[1], y[1], color=color, marker=mk, ms=10,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.4, lw=0)
        # short label by the 300/300 point
        ax.annotate(lab, xy=(x[0], y[0]),
                    xytext=(8, -2), textcoords="offset points",
                    fontsize=9, color=color, ha="left", va="center")

    ax.axvline(5.0, color="#BB5566", ls="--", lw=1.0, alpha=0.8,
               label="SLO (5$\\times$ baseline)")

    ax.set_xlabel("E2E p95 (normalized) — lower is better", fontsize=10)
    ax.set_ylabel("Tokens / Joule — higher is better", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", alpha=0.25, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "50", "100"])

    # "Better" corner indicator
    ax.annotate("", xy=(0.06, 0.94), xytext=(0.18, 0.82),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(0.06, 0.94, "  better", transform=ax.transAxes,
            fontsize=8, ha="left", va="bottom", color="black")

    # Legend: filled = 300/300, hollow = 300/200
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", lw=0, ms=8,
               markerfacecolor="gray", markeredgecolor="gray",
               label="300W / 300W"),
        Line2D([0], [0], marker="o", color="gray", lw=0, ms=8,
               markerfacecolor="white", markeredgecolor="gray",
               markeredgewidth=1.4,
               label="300W / 200W"),
        Line2D([0], [0], color="#BB5566", ls="--", lw=1.0,
               label="SLO (5$\\times$)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              frameon=True, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "perf_energy_pareto.pdf", dpi=200)
    fig.savefig(OUT / "perf_energy_pareto.png", dpi=200)
    print("Saved perf_energy_pareto.pdf / .png\n")

    print(f"{'policy':>14}  {'cap':>8}  {'E2E_p95':>9}  {'TPJ':>7}")
    for pol, lab, *_ in POLICIES:
        for cfg, cap_lab, *_ in POWER_CFGS:
            d = data[(cfg, lab)]
            print(f"{lab:>14}  {cap_lab:>8}  {d['e2e_p95']:>9.2f}  {d['tpj']:>7.2f}")


if __name__ == "__main__":
    main()
