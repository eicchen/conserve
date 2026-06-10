"""Diagnostic: iter-0 TTFT box plot, Collocated vs ConServe, with the network /
queue hops removed (compute-only).

Collocated: per request, find the engine that served it; TTFT = first_finish −
            first_exec on that engine (excludes the engine's request_start →
            first_exec queueing).
ConServe:  iter-0 from the matching prefiller_sweep. TTFT = (prefiller's
            last-step end − prefiller's first_exec)  +  (decoder's first_finish
            − decoder's first_exec). I.e., prefill compute on the prefiller +
            first-decode-step time on the decoder, with the KV-transfer /
            decoder-queue gap stripped out.

Saturation point, p300_d300, rps 1.634.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
CFG = "p300_d300"
RPS = "1.634"


def collocated_iter0_compute():
    """For each iter-0 rid in Collocated, find the colocated engine that served
    it and return (first_finish - first_exec) in seconds."""
    run_dir = s5.RPS_SWEEP / CFG / "no_disagg" / f"rps_{RPS}"
    out = []
    # build per-engine {rid: (first_exec, first_finish)}
    for lf in sorted(glob.glob(str(run_dir / "*_vllm_core_log.jsonl"))):
        first_exec, first_fin = {}, {}
        for st, en, ex, fin in s5.parse_core_log(lf):
            for rid in ex:
                first_exec.setdefault(rid, st)
            for rid in fin:
                first_fin.setdefault(rid, en)
        for rid, st in first_exec.items():
            if not s5._is_conv_request(rid):
                continue
            if s5.parse_request_id(rid)[1] != 0:
                continue
            if rid in first_fin:
                out.append(first_fin[rid] - st)
    return np.array(out)


def adaptive_iter0_compute():
    """Iter-0 in ConServe = prefill compute on prefiller + first decode step
    on decoder. Pulled from the matching prefiller_sweep trace."""
    pref_dir = s5.prefiller_dir_for(CFG, RPS)
    # prefiller: first_exec + last step_end where the rid appears in
    # executed or finished (chunked prefill spans several steps)
    pref_first_exec, pref_last_end = {}, {}
    for st, en, ex, fin in s5.parse_core_log(
            os.path.join(str(pref_dir), "prefiller_vllm_core_log.jsonl")):
        for rid in ex:
            pref_first_exec.setdefault(rid, st)
            pref_last_end[rid] = max(pref_last_end.get(rid, 0.0), en)
        for rid in fin:
            pref_last_end[rid] = max(pref_last_end.get(rid, 0.0), en)
    # decoder: first_exec + first_finish on whichever decoder takes over
    dec_first_exec, dec_first_fin = {}, {}
    for lf in sorted(glob.glob(str(pref_dir / "decoder*_vllm_core_log.jsonl"))):
        for st, en, ex, fin in s5.parse_core_log(lf):
            for rid in ex:
                dec_first_exec.setdefault(rid, st)
            for rid in fin:
                dec_first_fin.setdefault(rid, en)
    out = []
    for rid, pfe in pref_first_exec.items():
        if not s5._is_conv_request(rid):
            continue
        if s5.parse_request_id(rid)[1] != 0:
            continue
        if rid in pref_last_end and rid in dec_first_exec and rid in dec_first_fin:
            prefill_compute = pref_last_end[rid] - pfe
            first_decode    = dec_first_fin[rid] - dec_first_exec[rid]
            out.append(prefill_compute + first_decode)
    return np.array(out)


def main():
    co = collocated_iter0_compute() * 1000
    ad = adaptive_iter0_compute() * 1000

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bp = ax.boxplot([co, ad], positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='.', ms=3, alpha=0.5))
    bp['boxes'][0].set_facecolor("#4477AA"); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor("#117733"); bp['boxes'][1].set_alpha(0.7)
    for m in bp['medians']:
        m.set_color("black"); m.set_linewidth(1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Collocated", "ConServe"])
    ax.set_ylabel("Iter-0 TTFT, compute only (ms)")
    ax.set_title(f"Iter-0 TTFT excl. network/queue\n(p300_d300 @ rps {RPS})", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "ttft_iter0_no_network.pdf", dpi=200)
    fig.savefig(OUT / "ttft_iter0_no_network.png", dpi=200)
    print("Saved ttft_iter0_no_network.pdf / .png\n")

    for name, v in [("Collocated", co), ("ConServe", ad)]:
        print(f"  {name:>11}  n={len(v):4d}  "
              f"p50={np.percentile(v,50):7.1f}ms  "
              f"p95={np.percentile(v,95):7.1f}ms  "
              f"p99={np.percentile(v,99):7.1f}ms  "
              f"mean={np.mean(v):7.1f}ms")


if __name__ == "__main__":
    main()
