"""3-way in-flight comparison at rps=1.634 / p300_d300:

  ConServe (oracle peak-KV reservation, original)
  ConServe_per_turn_kv (AMPD-style KV but no wrong-predict pauses)
  AMPD (per-turn KV + wrong-predict pauses)

Each policy has 3 trials in output/var_check/p300_d300/rps_1.634/. We overlay
the median (across trials) of in-flight count and cumulative iter-4
completions, plus a per-trial span bar. This isolates the effect of the KV
reservation strategy from the wrong-predict effect.
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

OUT = Path(__file__).parent.parent / "output"
ROOT = (REPO_ROOT / "conserve/output/var_check/p300_d300/rps_1.634")
TRIALS = [0, 1, 2]

# (sub-dir, label, color)
POLICIES = [
    ("adaptive_3eng",                                  "ConServe (oracle KV)",   "#117733"),
    ("adaptive_3eng_per_turn_kv",                       "ConServe (per-turn KV)", "#88CCEE"),
    ("per_turn_adaptive_disagg_decoders_p10",          "AMPD",                    "#AA4499"),
]


def per_trial_timeseries(policy_dir, trial):
    rd = ROOT / policy_dir / f"trial_{trial}"
    ps = pd.read_csv(rd / "per_step_latency.csv")
    t0 = float(ps["start_time"].min())
    iter0 = ps[ps.step_id == 0]
    iter4 = ps[ps.step_id == 4]
    arrivals = np.sort(iter0["start_time"].to_numpy() - t0)
    completions = np.sort(iter4["end_time"].to_numpy() - t0)
    return dict(
        arrivals=arrivals,
        completions=completions,
        last_arrival=float(arrivals.max()),
        last_completion=float(completions.max()),
    )


def step_to_grid(times_in, values_in, grid):
    out = np.zeros_like(grid, dtype=float)
    j, cur = 0, 0
    for i, t in enumerate(grid):
        while j < len(times_in) and times_in[j] <= t:
            cur = values_in[j]; j += 1
        out[i] = cur
    return out


def aggregate(policy_dir, grid):
    inflight_stack, cum_stack = [], []
    spans, last_arrs = [], []
    for tr in TRIALS:
        d = per_trial_timeseries(policy_dir, tr)
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
    return dict(
        inflight=np.stack(inflight_stack),
        cum=np.stack(cum_stack),
        spans=np.array(spans),
        last_arrs=np.array(last_arrs),
    )


def main():
    grid = np.arange(0, 420 + 1, 1.0)
    agg = {label: aggregate(pol_dir, grid) for pol_dir, label, _ in POLICIES}

    fig = plt.figure(figsize=(9.5, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.55], hspace=0.32)
    ax_inf = fig.add_subplot(gs[0])
    ax_cum = fig.add_subplot(gs[1], sharex=ax_inf)
    ax_bar = fig.add_subplot(gs[2])

    for pol_dir, label, color in POLICIES:
        a = agg[label]
        med = np.median(a["inflight"], axis=0)
        p25 = np.percentile(a["inflight"], 25, axis=0)
        p75 = np.percentile(a["inflight"], 75, axis=0)
        ax_inf.fill_between(grid, p25, p75, color=color, alpha=0.18)
        ax_inf.plot(grid, med, color=color, lw=1.6, label=label)
        ax_inf.axvline(np.median(a["last_arrs"]), color=color, ls=":", lw=0.9, alpha=0.7)
        ax_inf.axvline(np.median(a["spans"]),     color=color, ls="--", lw=0.9, alpha=0.7)
    ax_inf.set_ylabel("In-flight\nconversations", fontsize=11)
    ax_inf.set_title("3-way comparison at rps=1.634, p300_d300 — 3 trials each (median + IQR)",
                     fontsize=11)
    ax_inf.grid(True, alpha=0.3, ls=":")
    ax_inf.legend(loc="upper right", fontsize=10, frameon=False)

    for pol_dir, label, color in POLICIES:
        a = agg[label]
        med = np.median(a["cum"], axis=0)
        p25 = np.percentile(a["cum"], 25, axis=0)
        p75 = np.percentile(a["cum"], 75, axis=0)
        ax_cum.fill_between(grid, p25, p75, color=color, alpha=0.18)
        ax_cum.plot(grid, med, color=color, lw=1.6, label=label)
        ax_cum.axvline(np.median(a["last_arrs"]), color=color, ls=":", lw=0.9, alpha=0.7)
        ax_cum.axvline(np.median(a["spans"]),     color=color, ls="--", lw=0.9, alpha=0.7)
    ax_cum.set_ylabel("Cumulative iter-4\ncompletions", fontsize=11)
    ax_cum.set_xlabel("Time since first iter-0 (s)", fontsize=11)
    ax_cum.grid(True, alpha=0.3, ls=":")

    # span bar — 3 trials × 3 policies
    width = 0.25
    x = np.arange(len(TRIALS))
    for i, (pol_dir, label, color) in enumerate(POLICIES):
        offset = (i - 1) * width
        ax_bar.bar(x + offset, agg[label]["spans"], width, color=color, label=label)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"trial {t}" for t in TRIALS], fontsize=10)
    ax_bar.set_ylabel("Span (s)", fontsize=11)
    ax_bar.set_title("Per-trial workload span", fontsize=11)
    ax_bar.legend(loc="lower right", fontsize=8, ncol=3, frameon=False)
    ax_bar.grid(True, axis="y", alpha=0.3, ls=":")

    fig.text(0.5, 0.005,
             "vertical lines: dotted = median last-iter-0 arrival;  dashed = median last iter-4 completion",
             ha="center", fontsize=9, style="italic", color="#444")

    fig.tight_layout(rect=(0, 0.018, 1, 1))
    fig.savefig(OUT / "kv_strategy_compare.pdf", dpi=200)
    fig.savefig(OUT / "kv_strategy_compare.png", dpi=200)
    print("Saved kv_strategy_compare.pdf / .png\n")

    print("=== per-trial spans (s) ===")
    print(f"{'trial':>6} " + " ".join(f"{lab[:24]:>26}" for _, lab, _ in POLICIES))
    for i, tr in enumerate(TRIALS):
        row = f"{tr:>6} "
        for pol_dir, label, _ in POLICIES:
            row += f"{agg[label]['spans'][i]:>26.1f}"
        print(row)
    print(f"\n{'median':>6} " + " ".join(f"{np.median(agg[lab]['spans']):>26.1f}"
                                          for _, lab, _ in POLICIES))
    print(f"{'mean':>6} "   + " ".join(f"{agg[lab]['spans'].mean():>26.1f}"
                                        for _, lab, _ in POLICIES))


if __name__ == "__main__":
    main()
