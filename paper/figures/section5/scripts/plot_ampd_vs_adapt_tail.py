"""Diagnose why AMPD's workload-level wall span is shorter than ConServe's at
rps=1.634, 300/300, even though AMPD's per-conv E2E is longer.

Hypothesis: ConServe lets iter-0 arrivals in faster than AMPD (no virtual-
prefill back-pressure), so it carries more concurrent in-flight convs around
saturation. vLLM batches grow with concurrency → step duration grows → tail
wallclock stretches even though per-step work is fairly stable.

Plot, both runs overlaid against wall time t=0 at first iter-0 start:
  (1) In-flight conversation count over time (arrivals − completions).
  (2) vLLM step duration on decoders (rolling mean, ms).
  (3) Mean per-step batch size on decoders (rolling mean).
  (4) Cumulative iter-0 arrivals (post-KV-gate). AMPD lags here whenever the
      decoder-side KV pressure delays iter-0 admission.
  (5) Cumulative iter-4 completions over time.

Markers: x=last iter-0 arrival per run (drain phase begins),
         x=last iter-4 completion per run (workload done).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
RPS = 1.634
CFG = "p300_d300"
RUNS = [
    ("per_turn_adaptive_disagg_decoders_p10", "AMPD",      "#AA4499"),
    ("adaptive_3eng",                          "ConServe", "#117733"),
]
ROLL = 200   # rolling window in number of steps


def collect(cfg, policy, rps):
    rd = s5.RPS_SWEEP / cfg / policy / f"rps_{rps}"
    ps = pd.read_csv(rd / "per_step_latency.csv")
    t0 = float(ps["start_time"].min())

    iter0 = ps[ps.step_id == 0].sort_values("start_time").copy()
    iter4 = ps[ps.step_id == 4].sort_values("end_time").copy()
    arrivals = (iter0["start_time"].to_numpy() - t0)
    completions = (iter4["end_time"].to_numpy() - t0)

    # In-flight per-conv count over time: arrivals = step_0 start, departures =
    # step_4 end (when this conv's last token leaves).
    events = []
    for t in arrivals:
        events.append((t, +1))
    for t in completions:
        events.append((t, -1))
    events.sort(key=lambda e: (e[0], -e[1]))
    times, inflight = [], []
    cur = 0
    for t, d in events:
        cur += d
        times.append(t)
        inflight.append(cur)
    times = np.asarray(times)
    inflight = np.asarray(inflight)

    # vLLM step time-series across the decoders.
    steps = []
    import glob, os
    for lf in sorted(glob.glob(str(rd / "decoder*_vllm_core_log.jsonl"))):
        for st, en, ex, fin in s5.parse_core_log(lf):
            steps.append((st - t0, en - st, len(ex)))
    sdf = pd.DataFrame(steps, columns=["t", "dur", "batch"]).sort_values("t").reset_index(drop=True)
    sdf["dur_roll_ms"] = sdf["dur"].rolling(ROLL, min_periods=20).mean() * 1000.0
    sdf["batch_roll"]   = sdf["batch"].rolling(ROLL, min_periods=20).mean()

    return dict(
        t_arrivals=arrivals,
        t_completions=np.sort(completions),
        t_inflight=times, n_inflight=inflight,
        step_df=sdf,
        last_arrival=float(arrivals.max()),
        last_completion=float(completions.max()),
    )


def main():
    data = {lab: collect(CFG, pol, RPS) for pol, lab, _ in RUNS}

    fig, axes = plt.subplots(5, 1, figsize=(9, 11.5), sharex=True,
                              gridspec_kw=dict(hspace=0.18))

    # (1) In-flight count
    ax = axes[0]
    for pol, lab, color in RUNS:
        d = data[lab]
        ax.step(d["t_inflight"], d["n_inflight"], where="post",
                color=color, lw=1.3, label=lab)
        ax.axvline(d["last_arrival"], color=color, ls=":", lw=0.9, alpha=0.8)
        ax.axvline(d["last_completion"], color=color, ls="--", lw=0.9, alpha=0.8)
    ax.set_ylabel("In-flight\nconversations", fontsize=10)
    ax.set_title(f"AMPD vs ConServe at rps={RPS}, {CFG.replace('p300_d', '300/')}W",
                 fontsize=11)
    ax.grid(True, alpha=0.3, ls=":")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # (2) Rolling vLLM step duration
    ax = axes[1]
    for pol, lab, color in RUNS:
        sdf = data[lab]["step_df"]
        ax.plot(sdf["t"], sdf["dur_roll_ms"], color=color, lw=1.2, label=lab)
    ax.set_ylabel(f"vLLM step duration\n(rolling {ROLL}, ms)", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")

    # (3) Rolling batch size
    ax = axes[2]
    for pol, lab, color in RUNS:
        sdf = data[lab]["step_df"]
        ax.plot(sdf["t"], sdf["batch_roll"], color=color, lw=1.2, label=lab)
    ax.set_ylabel(f"vLLM batch size\n(rolling {ROLL}, # reqs)", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")

    # (4) Cumulative iter-0 arrivals (post-KV-gate)
    ax = axes[3]
    for pol, lab, color in RUNS:
        d = data[lab]
        ts = np.sort(d["t_arrivals"])
        ys = np.arange(1, len(ts) + 1)
        ax.step(ts, ys, where="post", color=color, lw=1.3, label=lab)
        ax.axvline(d["last_arrival"], color=color, ls=":", lw=0.9, alpha=0.8)
    # Reference: ideal Poisson at the nominal rps from the trace.
    if len(RUNS) and "n_inflight" in data[RUNS[0][1]]:
        n_total = len(data[RUNS[0][1]]["t_arrivals"])
        ax.plot([0, n_total / RPS], [0, n_total], color="gray", ls=":", lw=0.8,
                alpha=0.7, label=f"ideal {RPS} conv/s")
    ax.set_ylabel("Cumulative iter-0\narrivals", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # (5) Cumulative iter-4 completions
    ax = axes[4]
    for pol, lab, color in RUNS:
        d = data[lab]
        ts = d["t_completions"]
        ys = np.arange(1, len(ts) + 1)
        ax.step(ts, ys, where="post", color=color, lw=1.3, label=lab)
        ax.axvline(d["last_arrival"], color=color, ls=":", lw=0.9, alpha=0.8)
        ax.axvline(d["last_completion"], color=color, ls="--", lw=0.9, alpha=0.8)
    ax.set_ylabel("Cumulative iter-4\ncompletions", fontsize=10)
    ax.set_xlabel("Time since first iter-0 (s)", fontsize=10)
    ax.grid(True, alpha=0.3, ls=":")

    # explanation of vertical lines
    fig.text(0.5, 0.01,
             "vertical lines: dotted = last iter-0 arrival (drain phase begins);  "
             "dashed = last iter-4 completion (workload done)",
             ha="center", fontsize=9, style="italic", color="#444")

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(OUT / "ampd_vs_adapt_tail.pdf", dpi=200)
    fig.savefig(OUT / "ampd_vs_adapt_tail.png", dpi=200)
    print("Saved ampd_vs_adapt_tail.pdf / .png")

    # short text summary
    for pol, lab, _ in RUNS:
        d = data[lab]
        sdf = d["step_df"]
        # "tail" = window between last_arrival and last_completion
        tail_mask = (sdf["t"] >= d["last_arrival"]) & (sdf["t"] <= d["last_completion"])
        avg_tail_dur = sdf.loc[tail_mask, "dur"].mean() * 1000
        avg_tail_batch = sdf.loc[tail_mask, "batch"].mean()
        # peak in-flight
        peak_inflight = int(d["n_inflight"].max())
        print(f"{lab:>11}  last_arr={d['last_arrival']:.1f}s  "
              f"last_done={d['last_completion']:.1f}s  "
              f"drain_phase={d['last_completion']-d['last_arrival']:.1f}s  "
              f"peak_inflight={peak_inflight}  "
              f"tail_step_dur={avg_tail_dur:.2f} ms  "
              f"tail_batch={avg_tail_batch:.1f}")


if __name__ == "__main__":
    main()
