"""Section-5 per-conversation metric extraction for the rps_sweep result set.

For one run dir + policy, `load_run` returns a per-conv DataFrame with columns:
    conv_id, e2e, ttfet, ttft0, tbt_mean        (seconds)

Definitions
-----------
E2E   = iter4 end  − iter0 start                 (per_step_latency.csv)
TTFET = iter4 first-token − iter0 start           (time to first effective token)
TTFT0 = iter0 first-token − iter0 request_start
TBT   = mean per-token inter-arrival gap over all iters

Adaptive correction
-------------------
adaptive_disagg's per_step start_time is post-prefiller (decoder receives the
request). For a fair E2E / TTFET vs no_disagg, we prepend a one-time per-conv
correction = prefill compute + KV transfer:
    correction = decoder_first_exec − prefiller_first_exec
sourced from the matching prefiller_sweep run. Measured from prefiller
first_exec (not request_start) so the prefiller's queueing delay is excluded —
at the saturation operating point the prefiller runs steady-state, and
no_disagg's replayed arrival is the prefiller first_exec too, so both policies
share that reference. no_disagg / all_disagg need no correction.
"""

import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())


import numpy as np
import pandas as pd

ROOT = (REPO_ROOT / "conserve/output")
RPS_SWEEP = ROOT / "rps_sweep"
BASELINE_JSON = ROOT / "baseline" / "baseline_summary.json"
TTFET_BASELINE_CACHE = Path(__file__).parent.parent / "cache" / "ttfet_baseline.json"
LASTTURN_TBT_BASELINE_CACHE = Path(__file__).parent.parent / "cache" / "lastturn_tbt_baseline.json"
LASTTURN_TBT_STEPS_CACHE = Path(__file__).parent.parent / "cache" / "lastturn_tbt_baseline_steps.json"


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def parse_request_id(rid):
    """'cmpl-{conv}-{iter}-0' -> (conv, iter)."""
    p = rid.split('-')
    return int(p[1]), int(p[2])


def parse_core_log(path):
    """-> list of (step_start, step_end, executed_ids, finished_ids)."""
    steps = []
    lines = [l.strip() for l in open(path) if l.strip()]
    i = 0
    while i < len(lines) - 1:
        try:
            s = json.loads(lines[i]); e = json.loads(lines[i + 1])
        except json.JSONDecodeError:
            i += 1; continue
        if s.get('event') == 'step_start' and e.get('event') == 'step_end':
            steps.append((_ts(s['timestamp']), _ts(e['timestamp']),
                          e.get('executed_request_ids') or [],
                          e.get('finished_request_ids') or []))
            i += 2
        else:
            i += 1
    return steps


def core_events(run_dir, policy=None):
    """Merge core logs in run_dir.

    -> first_exec{rid: first step_start}, token_times{rid: [step_end per token]}

    For all_disagg a request is split across the prefiller (prefill) and a
    decoder (decode); token_times is then taken from the DECODER core logs
    only, so the prefiller's prefill-completion isn't mistaken for the first
    decode token. first_exec is merged over all engines.
    """
    decoder_only = (policy == "all_disagg")
    first_exec, token_times = {}, {}
    for lf in sorted(glob.glob(os.path.join(run_dir, '*_vllm_core_log.jsonl'))):
        is_dec = os.path.basename(lf).startswith('decoder')
        for st, en, ex, fin in parse_core_log(lf):
            for rid in ex:
                first_exec.setdefault(rid, st)
            if decoder_only and not is_dec:
                continue
            for rid in fin:
                token_times.setdefault(rid, []).append(en)
    return first_exec, token_times


def request_starts(run_dir, pattern='*_vllm_engine_log.jsonl'):
    """{rid: request_start ts} merged over engine logs matching pattern."""
    rs = {}
    for lf in sorted(glob.glob(os.path.join(run_dir, pattern))):
        for line in open(lf):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('event') == 'request_start':
                rs.setdefault(d['request_id'], _ts(d['timestamp']))
    return rs


def per_conv_metrics(run_dir, policy=None):
    """Raw (uncorrected) per-conv metrics from one run dir.

    For all_disagg, decode tokens are read from the decoder logs and arrivals
    from the prefiller engine log (the request spans both)."""
    ps = pd.read_csv(os.path.join(run_dir, 'per_step_latency.csv'))
    first_exec, token_times = core_events(run_dir, policy)
    rs_pat = ('prefiller_vllm_engine_log.jsonl' if policy == "all_disagg"
              else '*_vllm_engine_log.jsonl')
    rstart = request_starts(run_dir, rs_pat)
    rows = []
    for conv, g in ps.groupby('conv_id'):
        steps = {int(r.step_id): r for r in g.itertuples()}
        if 0 not in steps or 4 not in steps:
            continue
        t0 = steps[0].start_time
        e2e = steps[4].end_time - t0

        rid0 = f'cmpl-{conv}-0-0'
        ttft0 = float('nan')
        if rid0 in token_times and rid0 in rstart:
            ttft0 = min(token_times[rid0]) - rstart[rid0]

        rid4 = f'cmpl-{conv}-4-0'
        ttfet = float('nan')
        if rid4 in token_times:
            ttfet = min(token_times[rid4]) - t0

        tbts = []
        for it in range(5):
            tt = sorted(token_times.get(f'cmpl-{conv}-{it}-0', []))
            tbts += [tt[k] - tt[k - 1] for k in range(1, len(tt))]
        tbt_mean = float(np.mean(tbts)) if tbts else float('nan')

        rows.append(dict(conv_id=int(conv), e2e=e2e, ttfet=ttfet,
                         ttft0=ttft0, tbt_mean=tbt_mean))
    return pd.DataFrame(rows)


def per_token_tbt(run_dir):
    """Long-form per-token TBT (seconds) for one run dir -> DataFrame[conv_id,iter_id,tbt]."""
    _, token_times = core_events(run_dir)
    rows = []
    for rid, times in token_times.items():
        try:
            conv, it = parse_request_id(rid)
        except (ValueError, IndexError):
            continue
        tt = sorted(times)
        for k in range(1, len(tt)):
            rows.append(dict(conv_id=conv, iter_id=it, tbt=tt[k] - tt[k - 1]))
    return pd.DataFrame(rows)


def adaptive_correction(prefiller_dir):
    """{conv: prefill compute + KV transfer} for iter-0
       = decoder first_exec − prefiller first_exec.

    Measured from prefiller *first_exec* (not request_start), so it excludes the
    prefiller's queueing delay: at the saturation operating point the prefiller
    runs steady-state with no growing queue, and no_disagg's replayed arrival is
    likewise the prefiller first_exec — so both policies share that reference.
    """
    pref_fe = {}
    for st, en, ex, fin in parse_core_log(
            os.path.join(prefiller_dir, 'prefiller_vllm_core_log.jsonl')):
        for rid in ex:
            pref_fe.setdefault(rid, st)
    dec_fe = {}
    for lf in sorted(glob.glob(os.path.join(prefiller_dir, 'decoder*_vllm_core_log.jsonl'))):
        for st, en, ex, fin in parse_core_log(lf):
            for rid in ex:
                dec_fe.setdefault(rid, st)
    corr = {}
    for rid, fe in pref_fe.items():
        conv, it = parse_request_id(rid)
        if it == 0 and rid in dec_fe:
            corr[conv] = dec_fe[rid] - fe
    return corr


def prefiller_dir_for(power_cfg, rps):
    """Matching prefiller_sweep dir for an adaptive run. RPS 1.634 replays rps_2."""
    sub = "prefiller_p200" if power_cfg == "p200_d200" else "prefiller_p300"
    trace_rps = "2" if str(rps) == "1.634" else str(rps)
    return RPS_SWEEP / sub / f"rps_{trace_rps}"


def load_run(power_cfg, policy, rps):
    """Per-conv metrics for one (power_cfg, policy, rps) cell, adaptive-corrected.

    For per_turn_adaptive_disagg_decoders_p<NN>, additionally adds the queueing
    delay that the virtual prefiller blocks imposed on each conv's iter-0
    (derived from synthetic_prefiller_per_step_latency.csv vs the recorded
    prefiller trace). This is the modeled side-effect of wrong-predict turn 2+
    requests occupying the prefiller and pushing back other convs' iter-0s."""
    run_dir = RPS_SWEEP / power_cfg / policy / f"rps_{rps}"
    df = per_conv_metrics(str(run_dir), policy)
    if policy == "adaptive_3eng" or policy.startswith("per_turn_adaptive_disagg_decoders"):
        corr = adaptive_correction(str(prefiller_dir_for(power_cfg, rps)))
        df["corr"] = df["conv_id"].map(corr)
        df["e2e"] = df["e2e"] + df["corr"].fillna(0.0)
        df["ttfet"] = df["ttfet"] + df["corr"].fillna(0.0)
    if policy.startswith("per_turn_adaptive_disagg_decoders"):
        queue = synthetic_iter0_queueing(power_cfg, policy, rps)
        df["queue"] = df["conv_id"].map(queue)
        df["e2e"] = df["e2e"] + df["queue"].fillna(0.0)
        df["ttfet"] = df["ttfet"] + df["queue"].fillna(0.0)
    df["power_cfg"], df["policy"], df["rps"] = power_cfg, policy, float(rps)
    return df


def synthetic_iter0_queueing(power_cfg, policy, rps):
    """{conv: extra queueing seconds the virtual prefiller blocks added to this
    conv's iter-0} for a per_turn run. Computed as the difference between the
    synthetic prefiller trace's iter-0 start (relative to its first event) and
    the recorded prefiller trace's iter-0 start (relative to its first event).
    Blocks only push events forward, so values are >= 0."""
    syn_path = RPS_SWEEP / power_cfg / policy / f"rps_{rps}" / "synthetic_prefiller_per_step_latency.csv"
    orig_path = prefiller_dir_for(power_cfg, rps) / "per_step_latency.csv"
    syn = pd.read_csv(syn_path)
    orig = pd.read_csv(orig_path)
    orig = orig[orig["step_id"] == 0]
    syn_rel = (syn["start_time"] - syn["start_time"].min()).values
    orig_rel = (orig["start_time"] - orig["start_time"].min()).values
    syn_conv = syn["conv_id"].astype(int).values
    orig_conv = orig["conv_id"].astype(int).values
    orig_map = dict(zip(orig_conv, orig_rel))
    return {int(c): float(s - orig_map[int(c)]) for c, s in zip(syn_conv, syn_rel)
            if int(c) in orig_map}


# --------------------------------------------------------------------------
# Baseline (per-conv standalone numbers + 5x SLO)
# --------------------------------------------------------------------------

def load_baseline():
    """Per-conv baseline means + 5x SLO. Returns DataFrame indexed by conv_id with:
       base_e2e, slo_e2e, base_ttft, slo_ttft, base_tbt, slo_tbt,
       base_ttfet, slo_ttfet."""
    j = json.loads(BASELINE_JSON.read_text())
    rows = []
    for cid, m in j["per_conv"].items():
        rows.append(dict(
            conv_id=int(cid),
            base_e2e=m["conv_e2e_s"]["mean"],   slo_e2e=m["conv_e2e_s"]["slo_5x_mean"],
            base_ttft=m["iter0_ttft_s"]["mean"], slo_ttft=m["iter0_ttft_s"]["slo_5x_mean"],
            base_tbt=m["tbt_mean_s"]["mean"],    slo_tbt=m["tbt_mean_s"]["slo_5x_mean"],
        ))
    df = pd.DataFrame(rows).set_index("conv_id")
    ttfet = baseline_ttfet()
    df["base_ttfet"] = pd.Series(ttfet)
    df["slo_ttfet"] = df["base_ttfet"] * 5.0
    return df


def baseline_ttfet():
    """Per-conv TTFET baseline = mean over the 20 baseline runs. Cached to json."""
    if TTFET_BASELINE_CACHE.exists():
        return {int(k): v for k, v in json.loads(TTFET_BASELINE_CACHE.read_text()).items()}
    base = ROOT / "baseline"
    run_dirs = sorted(glob.glob(str(base / "p300" / "order_seed*"))) + \
               sorted(glob.glob(str(base / "p300_run2" / "order_seed*")))
    acc = {}
    for rd in run_dirs:
        df = per_conv_metrics(rd)
        for r in df.itertuples():
            if not np.isnan(r.ttfet):
                acc.setdefault(r.conv_id, []).append(r.ttfet)
    ttfet = {c: float(np.mean(v)) for c, v in acc.items()}
    TTFET_BASELINE_CACHE.write_text(json.dumps(ttfet, indent=2))
    return ttfet


def baseline_lastturn_tbt():
    """Per-conv baseline last-turn (iter-4) mean TBT, mean over the 20 baseline
    runs. Cached. (Per-conv mean only used as the normalizer; observed data is
    kept per-token elsewhere.)"""
    if LASTTURN_TBT_BASELINE_CACHE.exists():
        return {int(k): v for k, v in json.loads(LASTTURN_TBT_BASELINE_CACHE.read_text()).items()}
    base = ROOT / "baseline"
    run_dirs = sorted(glob.glob(str(base / "p300" / "order_seed*"))) + \
               sorted(glob.glob(str(base / "p300_run2" / "order_seed*")))
    acc = {}
    for rd in run_dirs:
        t = per_token_tbt(rd)
        g = t[t.iter_id == 4].groupby("conv_id").tbt.mean()
        for c, v in g.items():
            acc.setdefault(int(c), []).append(float(v))
    res = {c: float(np.mean(v)) for c, v in acc.items()}
    LASTTURN_TBT_BASELINE_CACHE.write_text(json.dumps(res, indent=2))
    return res


def _iter4_ordered_gaps(run_dir, policy=None):
    """{conv: [ordered iter-4 inter-token gaps]} for one run dir."""
    _, token_times = core_events(run_dir, policy)
    out = {}
    for rid, times in token_times.items():
        try:
            conv, it = parse_request_id(rid)
        except (ValueError, IndexError):
            continue
        if it != 4:
            continue
        tt = sorted(times)
        out[conv] = [tt[k] - tt[k - 1] for k in range(1, len(tt))]
    return out


def _is_conv_request(rid):
    """True only for real cmpl-{conv}-{iter}-0 ids (skips cmpl-warmup-* etc.)."""
    try:
        parse_request_id(rid)
        return True
    except (ValueError, IndexError):
        return False


_ENERGY_CACHE_FILE = Path(__file__).parent.parent / "cache" / "energy_cache.json"
_energy_cache = (json.loads(_ENERGY_CACHE_FILE.read_text())
                 if _ENERGY_CACHE_FILE.exists() else {})


def run_energy_joules(run_dir, t0=None, t1=None, gpus=None):
    """Total GPU energy (J) in window [t0, t1] (epoch s) from dcgmi TOTEC
    cumulative counters, summed over the listed GPUs (or all if gpus is None).
    Whole trace used if t0/t1 are None. Result cached to energy_cache.json so
    repeat calls don't re-parse the (~600 MB) dcgmi traces."""
    key = json.dumps([str(run_dir), t0, t1, sorted(gpus) if gpus else None])
    if key in _energy_cache:
        return _energy_cache[key]
    per_gpu = {}
    with open(os.path.join(run_dir, "dcgmi_trace.tsv")) as f:
        for line in f:
            p = line.split()
            if len(p) < 5 or p[1] != "GPU":
                continue
            try:
                ts = _ts(p[0])
                totec = float(p[4])
            except ValueError:
                continue
            per_gpu.setdefault(p[2], []).append((ts, totec))
    total = 0.0
    for g, rows in per_gpu.items():
        if gpus is not None and g not in gpus:
            continue
        rows.sort()
        lo = rows[0][0] if t0 is None else t0
        hi = rows[-1][0] if t1 is None else t1
        e0 = next((v for ts, v in rows if ts >= lo), rows[0][1])
        e1 = next((v for ts, v in reversed(rows) if ts <= hi), rows[-1][1])
        total += e1 - e0
    joules = total / 1000.0   # mJ -> J
    _energy_cache[key] = joules
    _ENERGY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ENERGY_CACHE_FILE.write_text(json.dumps(_energy_cache, indent=2))
    return joules


def workload_energy(power_cfg, policy, rps):
    """Total system energy (J) for one workload pass.

    no_disagg / all_disagg : all 4 GPUs from the run, over its serving window.
    adaptive_3eng          : decoder GPUs 1-3 from the adaptive replay PLUS
                             the prefiller GPU 0 from the matching
                             prefiller_sweep run (where the prefiller actually
                             does the iter-0 prefills; the replay's own GPU 0
                             is an idle prefiller engine and would not exist
                             in a real adaptive deployment). The serving
                             window for each run is per_step.start_time.min ->
                             end_time.max, which already excludes the
                             prefix-cache warm-up (it finishes ~5s before the
                             first real conversation arrives).
    """
    run_dir = RPS_SWEEP / power_cfg / policy / f"rps_{rps}"
    ps = pd.read_csv(run_dir / "per_step_latency.csv")
    t0, t1 = float(ps["start_time"].min()), float(ps["end_time"].max())
    # adaptive_3eng and per_turn_adaptive_disagg_decoders_p<NN> share the same
    # decoder-only architecture: decoder GPUs from the live run + prefiller GPU 0
    # from the matching prefiller_sweep run.
    if policy == "adaptive_3eng" or policy.startswith("per_turn_adaptive_disagg_decoders"):
        E_dec = run_energy_joules(str(run_dir), t0, t1, gpus=["1", "2", "3"])
        pref_dir = prefiller_dir_for(power_cfg, rps)
        ps_p = pd.read_csv(pref_dir / "per_step_latency.csv")
        t0p, t1p = float(ps_p["start_time"].min()), float(ps_p["end_time"].max())
        E_pref = run_energy_joules(str(pref_dir), t0p, t1p, gpus=["0"])
        return E_dec + E_pref
    return run_energy_joules(str(run_dir), t0, t1)


def workload_tokens_full_context(power_cfg, policy, rps):
    """Total tokens processed (full-context input) + generated (output),
    summed over all (conv, iter). For iter k, the input is the cumulative
    history-so-far + this iter's prompt; the output is this iter's max_tokens.
    Same value across policies (same replayed trace)."""
    run_dir = RPS_SWEEP / power_cfg / policy / f"rps_{rps}"
    ps = pd.read_csv(run_dir / "per_step_latency.csv")
    tot = 0
    for _, g in ps.groupby("conv_id"):
        cum = 0
        for r in g.sort_values("step_id").itertuples():
            inp = cum + int(r.prompt_tokens)
            tot += inp + int(r.max_tokens)
            cum += int(r.prompt_tokens) + int(r.max_tokens)
    return tot


def all_request_ttft(run_dir, policy=None):
    """Per-request TTFT (s) for every conversation request (all iters pooled):
       first decode token − request arrival. For all_disagg, decode tokens are
       the decoder logs and arrival is the prefiller engine log. Prefix-cache
       warm-up requests (cmpl-warmup-*) are excluded."""
    _, token_times = core_events(run_dir, policy)
    rs_pat = ('prefiller_vllm_engine_log.jsonl' if policy == "all_disagg"
              else '*_vllm_engine_log.jsonl')
    rstart = request_starts(run_dir, rs_pat)
    out = []
    for rid, times in token_times.items():
        if _is_conv_request(rid) and rid in rstart and times:
            out.append(min(times) - rstart[rid])
    return np.array(out)


def adaptive_all_ttft(power_cfg, rps):
    """Per-request TTFT (s) for adaptive_3eng, all iters pooled, computed the
    same way as the headline TTFET correction:

      * iter-0  : from the matching prefiller_sweep trace (every iter-0 there
                  goes through the prefiller). TTFT = first decode token on
                  decoder − prefiller *first_exec*. Decoder-only token_times so
                  the prefiller's prefill-completion isn't mistaken for the
                  first decode token. first_exec (not request_start) excludes
                  prefiller queueing: 1.634 RPS is the prefiller saturation
                  point and has no growing queue, but the matched prefiller
                  trace happens to be rps_2 (closest above 1.634), which IS
                  over-saturated; using first_exec strips that artifact, same
                  as `adaptive_correction` does for TTFET.
      * iter 1-4: from the adaptive run's decoder logs (no prefiller path —
                  these iters hit the decoder's prefix cache directly).
                  TTFT = first decode token − decoder request_start."""
    out = []
    pref_dir = str(prefiller_dir_for(power_cfg, rps))
    _, tt_p = core_events(pref_dir, "all_disagg")  # decoder-only token_times
    # prefiller first_exec per rid
    pref_fe = {}
    for st, en, ex, fin in parse_core_log(
            os.path.join(pref_dir, 'prefiller_vllm_core_log.jsonl')):
        for rid in ex:
            pref_fe.setdefault(rid, st)
    for rid, times in tt_p.items():
        if not _is_conv_request(rid) or not times or rid not in pref_fe:
            continue
        if parse_request_id(rid)[1] == 0:
            out.append(min(times) - pref_fe[rid])

    run_dir = str(RPS_SWEEP / power_cfg / "adaptive_3eng" / f"rps_{rps}")
    _, tt_a = core_events(run_dir, "adaptive_3eng")
    rs_a = request_starts(run_dir, '*_vllm_engine_log.jsonl')
    for rid, times in tt_a.items():
        if not _is_conv_request(rid) or not times or rid not in rs_a:
            continue
        if parse_request_id(rid)[1] != 0:
            out.append(min(times) - rs_a[rid])
    return np.array(out)


_WRONG_PREDICT_RE = re.compile(
    r"\[conv (\d+) iter (\d+)\] WRONG PREDICT.*?-> pause ([\d.]+) ms")


def parse_wrong_predict_pauses(run_dir):
    """Parse run.log for {(conv_id, iter_id): pause_seconds} entries written by
    the per_turn runner. Empty dict if run.log is missing (e.g., for
    adaptive_3eng)."""
    out = {}
    log_path = Path(run_dir) / "run.log"
    if not log_path.exists():
        return out
    with open(log_path) as f:
        for line in f:
            m = _WRONG_PREDICT_RE.search(line)
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3)) / 1000.0
    return out


def per_turn_all_ttft(power_cfg, policy, rps):
    """Per-request TTFT (s) for a per_turn_adaptive_disagg_decoders_p<NN> run.
       iter-0  : from the matching prefiller_sweep trace (compute + KV transfer)
                 + the synthetic queueing caused by virtual prefill blocks for
                 that conv.
       iter 1-4: from the per_turn run's decoder logs, plus the wrong-predict
                 pause for that specific (conv, iter) if it was wrong-predicted
                 (the engine-side request_start is post-pause, so we add the
                 sleep duration back to recover the client-observed TTFT)."""
    out = []
    pref_dir = str(prefiller_dir_for(power_cfg, rps))
    _, tt_p = core_events(pref_dir, "all_disagg")
    pref_fe = {}
    for st, en, ex, fin in parse_core_log(
            os.path.join(pref_dir, 'prefiller_vllm_core_log.jsonl')):
        for rid in ex:
            pref_fe.setdefault(rid, st)
    queueing = synthetic_iter0_queueing(power_cfg, policy, rps)
    for rid, times in tt_p.items():
        if not _is_conv_request(rid) or not times or rid not in pref_fe:
            continue
        conv, it = parse_request_id(rid)
        if it == 0:
            out.append(min(times) - pref_fe[rid] + queueing.get(conv, 0.0))

    run_dir = str(RPS_SWEEP / power_cfg / policy / f"rps_{rps}")
    _, tt_a = core_events(run_dir, "adaptive_3eng")
    rs_a = request_starts(run_dir, '*_vllm_engine_log.jsonl')
    pauses = parse_wrong_predict_pauses(run_dir)
    for rid, times in tt_a.items():
        if not _is_conv_request(rid) or not times or rid not in rs_a:
            continue
        conv, it = parse_request_id(rid)
        if it != 0:
            out.append(min(times) - rs_a[rid] + pauses.get((conv, it), 0.0))
    return np.array(out)


def all_tbt_raw(run_dir, policy=None):
    """Raw per-token TBT (s) over ALL iterations, pooled over all convs.
       Warm-up requests excluded."""
    _, token_times = core_events(run_dir, policy)
    out = []
    for rid, times in token_times.items():
        if not _is_conv_request(rid):
            continue
        tt = sorted(times)
        out += [tt[k] - tt[k - 1] for k in range(1, len(tt))]
    return np.array(out)


def lastturn_tbt_raw(run_dir, policy=None):
    """Raw per-token last-turn (iter-4) TBT in seconds, pooled over all convs."""
    out = []
    for _, gaps in _iter4_ordered_gaps(run_dir, policy).items():
        out.extend(gaps)
    return np.array(out)


def baseline_lastturn_tbt_steps():
    """Per-conv, per-token-index baseline last-turn (iter-4) TBT:
       {conv: [mean gap at token index 0, 1, ...]}, averaged over the 20
       baseline runs. Used for strict per-step normalization. Cached."""
    if LASTTURN_TBT_STEPS_CACHE.exists():
        return {int(k): v for k, v in json.loads(LASTTURN_TBT_STEPS_CACHE.read_text()).items()}
    base = ROOT / "baseline"
    run_dirs = sorted(glob.glob(str(base / "p300" / "order_seed*"))) + \
               sorted(glob.glob(str(base / "p300_run2" / "order_seed*")))
    acc = {}   # conv -> list of per-run gap lists
    for rd in run_dirs:
        for conv, gaps in _iter4_ordered_gaps(rd).items():
            acc.setdefault(conv, []).append(gaps)
    res = {}
    for conv, runs in acc.items():
        maxlen = max((len(g) for g in runs), default=0)
        per_idx = []
        for i in range(maxlen):
            vals = [g[i] for g in runs if i < len(g)]
            per_idx.append(float(np.mean(vals)))
        res[conv] = per_idx
    LASTTURN_TBT_STEPS_CACHE.write_text(json.dumps(res, indent=2))
    return res


def lastturn_tbt_tokens(power_cfg, policy, rps):
    """Raw per-token last-turn (iter-4) TBT for one cell, normalized to each
    conv's baseline last-turn MEAN TBT. Returns a 1-D array pooled over all
    convs and token indices."""
    base4 = baseline_lastturn_tbt()
    run_dir = RPS_SWEEP / power_cfg / policy / f"rps_{rps}"
    vals = []
    for conv, gaps in _iter4_ordered_gaps(str(run_dir), policy).items():
        b = base4.get(conv)
        if b:
            for g in gaps:
                vals.append(g / b)
    return np.array(vals)


if __name__ == "__main__":
    # Smoke test: a baseline run's E2E/TTFT should match baseline_summary.json.
    import sys
    bl = json.loads(BASELINE_JSON.read_text())["per_conv"]
    rd = str(ROOT / "baseline" / "p300" / "order_seed0")
    df = per_conv_metrics(rd).set_index("conv_id")
    e_obs, e_base, t_obs, t_base = [], [], [], []
    for c in df.index:
        if str(c) in bl:
            e_obs.append(df.loc[c, "e2e"]);   e_base.append(bl[str(c)]["conv_e2e_s"]["mean"])
            t_obs.append(df.loc[c, "ttft0"]); t_base.append(bl[str(c)]["iter0_ttft_s"]["mean"])
    e_obs, e_base = np.array(e_obs), np.array(e_base)
    t_obs, t_base = np.array(t_obs), np.array(t_base)
    print(f"baseline/p300/order_seed0  vs  baseline_summary.json  ({len(e_obs)} convs)")
    print(f"  E2E  : obs/base ratio  median={np.median(e_obs/e_base):.4f}  "
          f"mean={np.mean(e_obs/e_base):.4f}")
    print(f"  TTFT0: obs/base ratio  median={np.nanmedian(t_obs/t_base):.4f}  "
          f"mean={np.nanmean(t_obs/t_base):.4f}")
    print(f"  TTFET baseline: computing over 20 runs...")
    tf = baseline_ttfet()
    print(f"  TTFET baseline: {len(tf)} convs, "
          f"median={np.median(list(tf.values())):.3f}s")
