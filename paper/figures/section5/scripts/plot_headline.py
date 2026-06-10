"""Section-5 headline: 3-row x 3-col QoS grid, statistic vs request rate.

  rows : gmean | p95 | SLO violation %
  cols : TTFET | Last-turn TBT | E2E

Compares Collocated / Full Disagg / AMPD / ConServe at uncapped 300W/300W.
Normalized = observed / per-conv standalone baseline. Red dashed = 5x SLO.

Y axes are shared within each row, and the gmean/p95 rows share the same
log-scale axis so the two statistics are directly comparable.

Data collection is cached to headline_data_cache.pkl so re-plot iterations
are instant. Delete the cache or set HEADLINE_REBUILD=1 in the env to force
re-collection (e.g. after re-running an underlying experiment cell).
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
CACHE = Path(__file__).parent.parent / "cache" / "headline_data_cache.pkl"
RPS = [0.5, 0.75, 1, 1.25, 1.5, 1.634]
# (cfg, policy, label, color, marker)
COMBOS = [
    ("p300_d300", "no_disagg",                                 "Collocated",   "#4477AA", "s"),
    ("p300_d300", "all_disagg",                                "Full Disagg",  "#CC3311", "^"),
    ("p300_d300", "per_turn_adaptive_disagg_decoders_p10",     "AMPD",         "#AA4499", "v"),
    ("p300_d300", "adaptive_3eng",                             "ConServe",     "#117733", "o"),
]
COLS = [("TTFET", "TTFET"), ("TBT", "Last-turn TBT"), ("E2E", "E2E")]
P95 = lambda x: np.percentile(x, 95)


def collect_data():
    """Read every (cfg, policy, rps) cell and return the (norm, slo) dicts.
    Slow — touches all dcgmi/core/log files."""
    base = s5.load_baseline()
    base4 = s5.baseline_lastturn_tbt()
    norm = {c: {lab: {} for _, _, lab, *_ in COMBOS} for c, _ in COLS}
    slo = {c: {lab: {} for _, _, lab, *_ in COMBOS} for c, _ in COLS}
    for cfg, pol, label, *_ in COMBOS:
        for rps in RPS:
            run_dir = s5.RPS_SWEEP / cfg / pol / f"rps_{rps}"
            if not (run_dir / "per_step_latency.csv").exists():
                continue
            df = s5.load_run(cfg, pol, rps).set_index("conv_id").join(base)
            norm["TTFET"][label][rps] = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
            norm["E2E"][label][rps] = (df["e2e"] / df["base_e2e"]).dropna().to_numpy()
            norm["TBT"][label][rps] = s5.lastturn_tbt_tokens(cfg, pol, rps)

            vt = df[["ttfet", "slo_ttfet"]].dropna()
            slo["TTFET"][label][rps] = float((vt["ttfet"] > vt["slo_ttfet"]).mean() * 100)
            ve = df[["e2e", "slo_e2e"]].dropna()
            slo["E2E"][label][rps] = float((ve["e2e"] > ve["slo_e2e"]).mean() * 100)
            gaps = s5._iter4_ordered_gaps(str(run_dir), pol)
            viol = tot = 0
            for conv, g in gaps.items():
                if conv in base4 and g:
                    tot += 1
                    viol += np.mean(g) > 5.0 * base4[conv]
            slo["TBT"][label][rps] = 100.0 * viol / tot if tot else float("nan")
    return norm, slo


def load_or_collect():
    if CACHE.exists() and not os.environ.get("HEADLINE_REBUILD"):
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("combos_key") == _combos_key() and cached.get("rps") == RPS:
            print(f"Loaded cached data from {CACHE.name} "
                  f"(delete or set HEADLINE_REBUILD=1 to refresh)")
            return cached["norm"], cached["slo"]
        print("Cache key mismatch (COMBOS or RPS changed); re-collecting.")
    norm, slo = collect_data()
    with open(CACHE, "wb") as f:
        pickle.dump({"combos_key": _combos_key(), "rps": RPS,
                     "norm": norm, "slo": slo}, f)
    print(f"Saved cache to {CACHE.name}")
    return norm, slo


def _combos_key():
    """Stable tuple of (cfg, policy, label) — invalidates cache if any combo changes."""
    return tuple((c, p, l) for c, p, l, *_ in COMBOS)


def main():
    norm, slo = load_or_collect()

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(3, 3, wspace=0.08, hspace=0.28,
                          left=0.12, right=0.99, top=0.9, bottom=0.1)

    # Shared y across all 3 cols of a row; gmean (row 0) and p95 (row 1) share
    # the same log-scale axis since both are normalized latency.
    qos_axes = [[None] * 3 for _ in range(3)]
    qos_axes[0][0] = fig.add_subplot(gs[0, 0])
    qos_axes[1][0] = fig.add_subplot(gs[1, 0], sharex=qos_axes[0][0],
                                       sharey=qos_axes[0][0])
    qos_axes[2][0] = fig.add_subplot(gs[2, 0], sharex=qos_axes[0][0])
    for j in range(1, 3):
        qos_axes[0][j] = fig.add_subplot(gs[0, j], sharex=qos_axes[0][0],
                                          sharey=qos_axes[0][0])
        qos_axes[1][j] = fig.add_subplot(gs[1, j], sharex=qos_axes[0][0],
                                          sharey=qos_axes[0][0])
        qos_axes[2][j] = fig.add_subplot(gs[2, j], sharex=qos_axes[0][0],
                                          sharey=qos_axes[2][0])

    rowdefs = [("gmean", gmean), ("p95", P95), ("SLO violation (%)", None)]
    for i, (rlabel, stat) in enumerate(rowdefs):
        for j, (col, ctitle) in enumerate(COLS):
            ax = qos_axes[i][j]
            for cfg, pol, label, color, mk in COMBOS:
                if stat is not None:
                    xs = [r for r in RPS
                          if norm[col][label].get(r) is not None and len(norm[col][label][r])]
                    ys = [stat(norm[col][label][r]) for r in xs]
                else:
                    xs = [r for r in RPS if r in slo[col][label]]
                    ys = [slo[col][label][r] for r in xs]
                ax.plot(xs, ys, color=color, marker=mk, ms=4, lw=1.5,
                        alpha=0.75, label=label)
            if stat is not None:
                ax.axhline(5.0, color="#BB5566", ls="--", lw=0.8, alpha=0.8)
                ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
                ax.set_yscale("symlog", linthresh=5.0, linscale=2.0)
                ax.set_yticks([1, 5, 10, 100])
                ax.set_yticklabels(["1", "5", "10", "100"], fontsize=9)
                ax.set_ylim(bottom=0.8)
            else:
                ax.set_yscale("symlog", linthresh=25.0, linscale=2.0)
                ax.set_yticks([0, 25, 100])
                ax.set_yticklabels(["0", "25", "100"], fontsize=9)
                ax.set_ylim(0, 110)
            ax.grid(True, which="both", alpha=0.22, ls=":")
            ax.tick_params(axis="both", labelsize=8)
            if i == 0:
                ax.set_title(ctitle, fontsize=10, pad=2)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=10)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i == len(rowdefs) - 1:
                ax.set_xticks([0.5, 1, 1.5])
                ax.set_xticklabels(["0.5", "1", "1.5"], fontsize=9)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    handles, labels = qos_axes[0][0].get_legend_handles_labels()
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=0.8,
                        label="SLO (5$\\times$)")
    handles.append(slo_handle)
    labels.append(slo_handle.get_label())
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, frameon=False, fontsize=9, handlelength=1.5,
               columnspacing=0.9, handletextpad=0.3)
    fig.supxlabel("Request rate (conv / s)", fontsize=10, y=0.02)
    fig.savefig(OUT / "headline.pdf", dpi=200)
    fig.savefig(OUT / "headline.png", dpi=200)
    print("Saved headline.pdf / .png\n")

    short = [lab for _, _, lab, *_ in COMBOS]
    for col, ctitle in COLS:
        print(f"=== {ctitle} ===")
        for rlabel, stat in rowdefs:
            hdr = " ".join(f"{s:>13}" for s in short)
            print(f"  {rlabel}:  " + hdr)
            for rps in RPS:
                vals = []
                for cfg, pol, label, *_ in COMBOS:
                    if stat is not None:
                        v = norm[col][label].get(rps)
                        vals.append(f"{stat(v):.2f}" if v is not None and len(v) else "-")
                    else:
                        s = slo[col][label].get(rps)
                        vals.append(f"{s:.0f}%" if s is not None else "-")
                row = " ".join(f"{v:>13}" for v in vals)
                print(f"    rps {rps:>5}: {row}")
        print()


if __name__ == "__main__":
    main()
