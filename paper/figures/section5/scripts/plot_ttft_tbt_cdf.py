"""Section-5: TTFT and TBT distributions at saturation (RPS 1.634), side by
side. Four lines: Collocated, Full Disagg, AMPD, ConServe — all at 300W/300W.

Left  — CDF of per-request TTFT (all iters pooled). Prefix caching makes
        Collocated / ConServe mostly fast; Full Disagg pays the PD handoff on
        every iter, so its whole distribution is shifted high.
Right — CDF of per-token TBT over all iterations. Collocated has the fat tail
        (chunked-prefill bursts colliding with decode).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
RPS = "1.634"
# (cfg, policy, label, color)
COMBOS = [
    ("p300_d300", "no_disagg",                                 "Collocated",   "#4477AA"),
    ("p300_d300", "all_disagg",                                "Full Disagg",  "#CC3311"),
    ("p300_d300", "per_turn_adaptive_disagg_decoders_p10",     "AMPD",         "#AA4499"),
    ("p300_d300", "adaptive_3eng",                             "ConServe",     "#117733"),
]


def cdf(ax, v, color, label):
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=color, lw=1.9, label=label)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

    for cfg, pol, label, color in COMBOS:
        run_dir = str(s5.RPS_SWEEP / cfg / pol / f"rps_{RPS}")
        # adaptive: iter-0 TTFT comes from the matching prefiller_sweep trace
        # (where the prefill+KV-transfer is actually paid); iters 1-4 from the
        # adaptive decoder logs. Same convention as the headline TTFET.
        if pol == "adaptive_3eng":
            ttft = s5.adaptive_all_ttft(cfg, RPS) * 1000.0
        elif pol.startswith("per_turn_adaptive_disagg_decoders"):
            # iter-0 from the recorded prefiller trace + queueing from
            # synthetic blocks; iter 1-4 from the per_turn decoder logs with the
            # wrong-predict pauses added back (engine request_start is post-pause).
            ttft = s5.per_turn_all_ttft(cfg, pol, RPS) * 1000.0
        else:
            ttft = s5.all_request_ttft(run_dir, pol) * 1000.0
        tbt = s5.all_tbt_raw(run_dir, pol) * 1000.0         # ms, all iters
        cdf(axes[0], ttft, color, label)
        cdf(axes[1], tbt, color, label)
        print(f"{label:>34}: TTFT n={len(ttft):>5} p50={np.percentile(ttft,50):7.1f}ms "
              f"p99={np.percentile(ttft,99):8.1f}ms  |  "
              f"TBT n={len(tbt):>6} p50={np.percentile(tbt,50):6.1f}ms "
              f"p99={np.percentile(tbt,99):7.1f}ms")

    for ax, xlabel in [(axes[0], "TTFT (ms)"), (axes[1], "TBT (ms)")]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.25, ls=":")
    axes[0].set_ylabel("CDF")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "ttft_tbt_cdf.pdf", dpi=200)
    fig.savefig(OUT / "ttft_tbt_cdf.png", dpi=200)
    print("\nSaved ttft_tbt_cdf.pdf / .png")


if __name__ == "__main__":
    main()
