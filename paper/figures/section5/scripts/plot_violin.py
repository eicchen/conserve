"""Diagnostic violins (uncapped, p300_d300): the full per-conv / per-token
distributions behind the section-5 headline statistics.

Row 1 — normalized TTFET, one point per conversation.
Row 2 — normalized last-turn TBT, one point per *decode token* in iter-4
        (raw, NOT averaged per conv).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
CFG = "p300_d300"
RPS = [0.5, 0.75, 1, 1.25, 1.5, 1.634]
POLICIES = [
    ("adaptive_3eng", "adaptive (ours)", "#117733"),
    ("no_disagg",     "no-disagg",       "#4477AA"),
    ("all_disagg",    "all-disagg",      "#CC3311"),
]
GROUP_W = 4.0


def main():
    base = s5.load_baseline()
    ttfet, tbt = {}, {}      # [policy][rps] -> array
    for pol, _, _ in POLICIES:
        ttfet[pol], tbt[pol] = {}, {}
        for rps in RPS:
            run_dir = s5.RPS_SWEEP / CFG / pol / f"rps_{rps}"
            if not (run_dir / "per_step_latency.csv").exists():
                continue
            df = s5.load_run(CFG, pol, rps).set_index("conv_id").join(base)
            ttfet[pol][rps] = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
            tbt[pol][rps] = s5.lastturn_tbt_tokens(CFG, pol, rps)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    for ax, (data, ylabel) in zip(
            axes, [(ttfet, "Normalized TTFET\n(per conversation)"),
                   (tbt, "Normalized last-turn TBT\n(per decode token)")]):
        for pi, (pol, _, color) in enumerate(POLICIES):
            arrs, poss = [], []
            for ri, rps in enumerate(RPS):
                v = data[pol].get(rps)
                if v is None or len(v) == 0:
                    continue
                arrs.append(np.clip(v, 1e-3, None))
                poss.append(ri * GROUP_W + pi)
            if not arrs:
                continue
            vp = ax.violinplot(arrs, positions=poss, widths=0.85,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set(facecolor=color, alpha=0.6, edgecolor=color)
            vp["cmedians"].set(color="black", lw=1.2)
        ax.axhline(5.0, color="#BB5566", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.25, ls=":")

    axes[1].set_xticks([ri * GROUP_W + 1 for ri in range(len(RPS))])
    axes[1].set_xticklabels([str(r) for r in RPS])
    axes[1].set_xlabel("Request rate (conversations / s)")
    axes[0].legend(handles=[Patch(facecolor=c, alpha=0.6, label=l)
                            for _, l, c in POLICIES],
                   loc="upper left", frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "violin_ttfet_tbt.pdf", dpi=200)
    fig.savefig(OUT / "violin_ttfet_tbt.png", dpi=200)
    print("Saved violin_ttfet_tbt.pdf / .png")
    for pol, _, _ in POLICIES:
        npts = sum(len(tbt[pol].get(r, [])) for r in RPS)
        print(f"  {pol:>14}: {npts} last-turn token samples pooled")


if __name__ == "__main__":
    main()
