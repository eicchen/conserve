"""Diagnostic: per-turn TTFT box plot, Collocated vs ConServe at saturation
(p300_d300, rps 1.634). Five turn groups (iter 0..4), two boxes per group.

Collocated: TTFT pulled from the run's engine/core logs (every engine does
prefill + decode end-to-end).
ConServe:  iter-0 from the matching prefiller_sweep trace (decoder first
token − prefiller first_exec; first_exec excludes queueing from rps_2
over-saturation, since 1.634 is the saturation point with no real queue);
iter 1-4 from the adaptive decoder logs (decoder first token − decoder
request_start).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
CFG = "p300_d300"
RPS = "1.634"


def collocated_ttft_per_iter():
    run_dir = str(s5.RPS_SWEEP / CFG / "no_disagg" / f"rps_{RPS}")
    _, tt = s5.core_events(run_dir, "no_disagg")
    rs = s5.request_starts(run_dir, '*_vllm_engine_log.jsonl')
    per = {i: [] for i in range(5)}
    for rid, times in tt.items():
        if not s5._is_conv_request(rid) or not times or rid not in rs:
            continue
        _, it = s5.parse_request_id(rid)
        per[it].append(min(times) - rs[rid])
    return {i: np.array(v) for i, v in per.items()}


def adaptive_ttft_per_iter():
    # iter 0 from prefiller_sweep
    pref_dir = str(s5.prefiller_dir_for(CFG, RPS))
    _, tt_p = s5.core_events(pref_dir, "all_disagg")  # decoder-only
    pref_fe = {}
    for st, en, ex, fin in s5.parse_core_log(
            f"{pref_dir}/prefiller_vllm_core_log.jsonl"):
        for rid in ex:
            pref_fe.setdefault(rid, st)
    # iter 1-4 from adaptive run
    run_dir = str(s5.RPS_SWEEP / CFG / "adaptive_3eng" / f"rps_{RPS}")
    _, tt_a = s5.core_events(run_dir, "adaptive_3eng")
    rs_a = s5.request_starts(run_dir, '*_vllm_engine_log.jsonl')

    per = {i: [] for i in range(5)}
    for rid, times in tt_p.items():
        if not s5._is_conv_request(rid) or not times or rid not in pref_fe:
            continue
        if s5.parse_request_id(rid)[1] == 0:
            per[0].append(min(times) - pref_fe[rid])
    for rid, times in tt_a.items():
        if not s5._is_conv_request(rid) or not times or rid not in rs_a:
            continue
        it = s5.parse_request_id(rid)[1]
        if it != 0:
            per[it].append(min(times) - rs_a[rid])
    return {i: np.array(v) for i, v in per.items()}


def main():
    co = collocated_ttft_per_iter()
    ad = adaptive_ttft_per_iter()

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    positions_co = [i - 0.18 for i in range(5)]
    positions_ad = [i + 0.18 for i in range(5)]
    bp_co = ax.boxplot([co[i] * 1000 for i in range(5)], positions=positions_co,
                       widths=0.30, patch_artist=True, showfliers=True,
                       flierprops=dict(marker='.', ms=2, alpha=0.4))
    bp_ad = ax.boxplot([ad[i] * 1000 for i in range(5)], positions=positions_ad,
                       widths=0.30, patch_artist=True, showfliers=True,
                       flierprops=dict(marker='.', ms=2, alpha=0.4))
    for b in bp_co['boxes']:
        b.set_facecolor("#4477AA"); b.set_edgecolor("#1F3A5F"); b.set_alpha(0.7)
    for b in bp_ad['boxes']:
        b.set_facecolor("#117733"); b.set_edgecolor("#0A4D24"); b.set_alpha(0.7)
    for bp in (bp_co, bp_ad):
        for m in bp['medians']:
            m.set_color("black"); m.set_linewidth(1.4)

    ax.set_yscale("log")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"iter {i}" for i in range(5)])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"Per-turn TTFT, p300_d300 @ rps {RPS}", fontsize=10)
    ax.grid(True, axis="y", which="both", alpha=0.22, ls=":")
    ax.legend([bp_co['boxes'][0], bp_ad['boxes'][0]],
              ["Collocated", "ConServe"], loc="upper right", fontsize=9,
              frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_per_turn.pdf", dpi=200)
    fig.savefig(OUT / "ttft_per_turn.png", dpi=200)
    print("Saved ttft_per_turn.pdf / .png\n")

    print(f"{'iter':>5} {'n':>5}  "
          f"{'Collocated p50/p95/p99 (ms)':>34}    {'ConServe p50/p95/p99 (ms)':>34}")
    for i in range(5):
        c = co[i] * 1000; a = ad[i] * 1000
        cs = f"{np.percentile(c,50):8.1f} / {np.percentile(c,95):8.1f} / {np.percentile(c,99):8.1f}"
        as_ = f"{np.percentile(a,50):8.1f} / {np.percentile(a,95):8.1f} / {np.percentile(a,99):8.1f}"
        print(f"{i:>5} {len(c):>5}  {cs:>34}    {as_:>34}")


if __name__ == "__main__":
    main()
