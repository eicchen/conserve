"""Order-sweep version of the AMPD vs ConServe tail diagnosis.

Same idea as plot_ampd_vs_adapt_tail.py but for the 10-seed order sweep at
p300_d300 (one rps point per seed). Overlays median ± [p25, p75] band across
seeds for each policy.

Panels (shared x-axis = seconds since each seed's first iter-0):
  (1) In-flight conversation count
  (2) Cumulative iter-4 completions
  (3) Per-seed total span (bar comparison)
"""

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

ROOT = (REPO_ROOT / "conserve/output/order_sweep/p300_d300")
OUT = Path(__file__).parent.parent / "output"
SEEDS = list(range(10))
POLICIES = [
    ("per_turn_adaptive_disagg_decoders_p10", "AMPD",      "#AA4499"),
    ("adaptive_3eng",                          "ConServe", "#117733"),
]


def per_seed_timeseries(policy, seed):
    rd = ROOT / policy / f"seed_{seed}"
    ps = pd.read_csv(rd / "per_step_latency.csv")
    t0 = float(ps["start_time"].min())
    iter0 = ps[ps.step_id == 0]
    iter4 = ps[ps.step_id == 4]
    arrivals = (iter0["start_time"].to_numpy() - t0)
    completions = (iter4["end_time"].to_numpy() - t0)
    return dict(
        arrivals=np.sort(arrivals),
        completions=np.sort(completions),
        last_arrival=float(arrivals.max()),
        last_completion=float(completions.max()),
    )


def step_to_grid(times_in, values_in, grid):
    """Step-function values_in sampled on grid (right-continuous, last-value-held)."""
    out = np.zeros_like(grid, dtype=float)
    j = 0
    cur = 0
    for i, t in enumerate(grid):
        while j < len(times_in) and times_in[j] <= t:
            cur = values_in[j]; j += 1
        out[i] = cur
    return out


def aggregate(policy, grid):
    """Median, p25, p75 of in-flight count over seeds on a common time grid.
    Also returns cumulative completions on the same grid."""
    inflight_stack, cum_stack = [], []
    spans, last_arrs = [], []
    for s in SEEDS:
        d = per_seed_timeseries(policy, s)
        # in-flight events
        events = [(t, +1) for t in d["arrivals"]] + [(t, -1) for t in d["completions"]]
        events.sort(key=lambda e: (e[0], -e[1]))
        ts, cur, inf = [], 0, []
        for t, delta in events:
            cur += delta
            ts.append(t); inf.append(cur)
        inflight_stack.append(step_to_grid(np.array(ts), np.array(inf), grid))
        cum_stack.append(step_to_grid(d["completions"], np.arange(1, len(d["completions"]) + 1), grid))
        spans.append(d["last_completion"])
        last_arrs.append(d["last_arrival"])
    inflight = np.stack(inflight_stack)  # [seeds, grid]
    cum = np.stack(cum_stack)
    return dict(
        inflight_med=np.median(inflight, axis=0),
        inflight_p25=np.percentile(inflight, 25, axis=0),
        inflight_p75=np.percentile(inflight, 75, axis=0),
        cum_med=np.median(cum, axis=0),
        cum_p25=np.percentile(cum, 25, axis=0),
        cum_p75=np.percentile(cum, 75, axis=0),
        spans=np.array(spans),
        last_arrs=np.array(last_arrs),
    )


def main():
    grid = np.arange(0, 420 + 1, 1.0)
    agg = {lab: aggregate(pol, grid) for pol, lab, _ in POLICIES}

    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.65], hspace=0.32)
    ax_inf = fig.add_subplot(gs[0])
    ax_cum = fig.add_subplot(gs[1], sharex=ax_inf)
    ax_bar = fig.add_subplot(gs[2])

    # In-flight panel
    for pol, lab, color in POLICIES:
        a = agg[lab]
        ax_inf.fill_between(grid, a["inflight_p25"], a["inflight_p75"],
                            color=color, alpha=0.18)
        ax_inf.plot(grid, a["inflight_med"], color=color, lw=1.6, label=lab)
        ax_inf.axvline(np.median(a["last_arrs"]), color=color, ls=":", lw=1.0, alpha=0.8)
        ax_inf.axvline(np.median(a["spans"]),     color=color, ls="--", lw=1.0, alpha=0.8)
    ax_inf.set_ylabel("In-flight\nconversations", fontsize=11)
    ax_inf.set_title("In-flight conversations vs time — order sweep p300_d300 (10 seeds, "
                     "median + IQR band)", fontsize=11)
    ax_inf.grid(True, alpha=0.3, ls=":")
    ax_inf.legend(loc="upper right", fontsize=10, frameon=False)

    # Cumulative completions panel
    for pol, lab, color in POLICIES:
        a = agg[lab]
        ax_cum.fill_between(grid, a["cum_p25"], a["cum_p75"],
                            color=color, alpha=0.18)
        ax_cum.plot(grid, a["cum_med"], color=color, lw=1.6, label=lab)
        ax_cum.axvline(np.median(a["last_arrs"]), color=color, ls=":", lw=1.0, alpha=0.8)
        ax_cum.axvline(np.median(a["spans"]),     color=color, ls="--", lw=1.0, alpha=0.8)
    ax_cum.set_xlabel("Time since first iter-0 (s)", fontsize=11)
    ax_cum.set_ylabel("Cumulative iter-4\ncompletions", fontsize=11)
    ax_cum.grid(True, alpha=0.3, ls=":")

    # Per-seed span bar comparison
    spans_a = agg["AMPD"]["spans"]
    spans_b = agg["ConServe"]["spans"]
    x = np.arange(len(SEEDS))
    w = 0.4
    ax_bar.bar(x - w/2, spans_a, w, color="#AA4499", label="AMPD")
    ax_bar.bar(x + w/2, spans_b, w, color="#117733", label="ConServe")
    ax_bar.axhline(np.median(spans_a), color="#AA4499", ls="--", lw=1.0, alpha=0.7,
                   label=f"median AMPD = {np.median(spans_a):.1f} s")
    ax_bar.axhline(np.median(spans_b), color="#117733", ls="--", lw=1.0, alpha=0.7,
                   label=f"median ConServe = {np.median(spans_b):.1f} s")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=9, rotation=20)
    ax_bar.set_ylabel("Workload span\n(s)", fontsize=11)
    ax_bar.set_title("Per-seed total span", fontsize=11)
    ax_bar.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
    ax_bar.grid(True, axis="y", alpha=0.3, ls=":")

    fig.text(0.5, 0.005,
             "vertical lines: dotted = median last-iter-0 arrival;  "
             "dashed = median last iter-4 completion",
             ha="center", fontsize=9, style="italic", color="#444")

    fig.tight_layout(rect=(0, 0.018, 1, 1))
    fig.savefig(OUT / "ampd_vs_adapt_tail_order.pdf", dpi=200)
    fig.savefig(OUT / "ampd_vs_adapt_tail_order.png", dpi=200)
    print("Saved ampd_vs_adapt_tail_order.pdf / .png\n")

    print("=== per-seed span (s) ===")
    print(f"{'seed':>6} {'AMPD':>8} {'ConServe':>10} {'Δ (Adapt-AMPD)':>16}")
    for i, s in enumerate(SEEDS):
        print(f"{s:>6} {spans_a[i]:>8.1f} {spans_b[i]:>10.1f} {spans_b[i]-spans_a[i]:>+16.1f}")
    print(f"\nmedian:  AMPD={np.median(spans_a):.1f}  ConServe={np.median(spans_b):.1f}  "
          f"Δ={np.median(spans_b) - np.median(spans_a):+.1f}")
    print(f"mean:    AMPD={spans_a.mean():.1f}  ConServe={spans_b.mean():.1f}  "
          f"Δ={spans_b.mean() - spans_a.mean():+.1f}")


if __name__ == "__main__":
    main()
