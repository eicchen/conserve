"""
Parse TTFT and TBT from vllm_core_log.jsonl files for three scheduling protocols:
  - no_disagg
  - all_disagg
  - adaptive_disagg  (combined from adaptive_disagg_first + adaptive_disagg_rest)

Saves parsed CSVs and plots to parsed_metrics/.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), '1p3d_Qwen-0.6B_real-workload')
OUT_DIR  = os.path.join(BASE_DIR, 'parsed_metrics')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_core_log(filepath):
    """Read *_vllm_core_log.jsonl -> list of step dicts (start_time, end_time, ...)."""
    steps = []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines) - 1:
        try:
            s = json.loads(lines[i])
            e = json.loads(lines[i + 1])
        except json.JSONDecodeError:
            i += 1
            continue
        if s.get('event') == 'step_start' and e.get('event') == 'step_end':
            steps.append({
                'start_time':      datetime.fromisoformat(s['timestamp']).timestamp(),
                'end_time':        datetime.fromisoformat(e['timestamp']).timestamp(),
                'executed':        e.get('executed_request_ids', []),
                'finished':        e.get('finished_request_ids', []),
                'finish_reasons':  e.get('finish_reasons', []),
                'num_cached_tokens': e.get('num_cached_tokens', []),
            })
            i += 2
        else:
            i += 1
    return steps


def parse_request_id(req_id):
    """'cmpl-{conv_id}-{iter_id}-0' -> (conv_id, iter_id)"""
    parts = req_id.split('-')
    return int(parts[1]), int(parts[2])


def steps_to_events(steps, exclude_ids=None):
    """
    Extract per-request first-execution time, token times, and first cached-token count.

    Returns
    -------
    first_exec    : {req_id: step_start timestamp when request first seen}
    token_times   : {req_id: [step_end timestamp per generated token]}
    cached_first  : {req_id: num_cached_tokens at first token generation}
    """
    exclude_ids = exclude_ids or set()
    first_exec   = {}
    token_times  = {}
    cached_first = {}
    for step in steps:
        for rid in step['executed']:
            if rid not in exclude_ids:
                first_exec.setdefault(rid, step['start_time'])
        cached_list = step.get('num_cached_tokens', [])
        for i, rid in enumerate(step['finished']):
            if rid not in exclude_ids:
                token_times.setdefault(rid, []).append(step['end_time'])
                if rid not in cached_first and i < len(cached_list):
                    cached_first[rid] = cached_list[i]
    return first_exec, token_times, cached_first


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_records_nodisagg(first_exec, token_times, cached_first=None):
    """
    TTFT = first_token_time - first_exec_time
    TBT_k = token_k - token_(k-1)
    """
    cached_first = cached_first or {}
    records = []
    for rid, times in token_times.items():
        if rid not in first_exec:
            continue
        times = sorted(times)
        conv_id, iter_id = parse_request_id(rid)
        ttft = times[0] - first_exec[rid]
        tbts = [times[k] - times[k-1] for k in range(1, len(times))]
        records.append(dict(
            request_id=rid, conv_id=conv_id, iter_id=iter_id,
            ttft=ttft,
            tbt_mean=float(np.mean(tbts)) if tbts else float('nan'),
            tbt_list=tbts,
            num_tokens=len(times),
            cached_tokens=cached_first.get(rid, float('nan')),
        ))
    return pd.DataFrame(records)


def build_records_disagg(prefill_first_exec, prefill_end_time,
                         decode_token_times, cached_first=None, iter_id_filter=None):
    """
    TTFT  = first_decode_token_time - prefill_first_exec_time
    TBT_2 = 2nd_decode_token_time  - prefill_end_time   (captures KV-transfer gap)
    TBT_k = token_k - token_(k-1) on decoder  (k >= 3)
    """
    cached_first = cached_first or {}
    records = []
    for rid, times in decode_token_times.items():
        if rid not in prefill_first_exec:
            continue
        conv_id, iter_id = parse_request_id(rid)
        if iter_id_filter is not None and iter_id != iter_id_filter:
            continue
        times = sorted(times)
        ttft  = times[0] - prefill_first_exec[rid]
        tbts  = []
        if len(times) >= 2 and rid in prefill_end_time:
            tbts.append(times[1] - prefill_end_time[rid])
        for k in range(2, len(times)):
            tbts.append(times[k] - times[k-1])
        records.append(dict(
            request_id=rid, conv_id=conv_id, iter_id=iter_id,
            ttft=ttft,
            tbt_mean=float(np.mean(tbts)) if tbts else float('nan'),
            tbt_list=tbts,
            num_tokens=len(times),
            cached_tokens=cached_first.get(rid, float('nan')),
        ))
    return pd.DataFrame(records)


def to_flat_df(df, protocol):
    """Explode tbt_list into per-token rows alongside TTFT rows."""
    rows = []
    for _, row in df.iterrows():
        rows.append(dict(protocol=protocol, request_id=row['request_id'],
                         conv_id=row['conv_id'], iter_id=row['iter_id'],
                         metric='ttft', value=row['ttft'], token_index=1))
        for i, tbt in enumerate(row['tbt_list']):
            rows.append(dict(protocol=protocol, request_id=row['request_id'],
                             conv_id=row['conv_id'], iter_id=row['iter_id'],
                             metric='tbt', value=tbt, token_index=i + 2))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-protocol parsers
# ---------------------------------------------------------------------------

def parse_no_disagg():
    """All GPUs run prefill+decode for their assigned requests — merge all logs."""
    folder    = os.path.join(BASE_DIR, 'no_disagg')
    log_files = sorted(glob.glob(os.path.join(folder, '*_vllm_core_log.jsonl')))
    print(f'[no_disagg] {len(log_files)} log files')
    all_fe, all_tt, all_cf = {}, {}, {}
    for lf in log_files:
        fe, tt, cf = steps_to_events(parse_core_log(lf))
        all_fe.update(fe)
        all_tt.update(tt)
        all_cf.update(cf)
    df = build_records_nodisagg(all_fe, all_tt, all_cf)
    print(f'  -> {len(df)} requests parsed')
    return df


def parse_all_disagg():
    """Prefiller does prefill; decoders do decode. Cross-log timing."""
    folder = os.path.join(BASE_DIR, 'all_disagg')

    # prefiller
    pf_steps = parse_core_log(os.path.join(folder, 'prefiller_vllm_core_log.jsonl'))
    pf_first_exec, pf_end_time = {}, {}
    for step in pf_steps:
        for rid in step['executed']:
            pf_first_exec.setdefault(rid, step['start_time'])
        for rid in step['finished']:
            pf_end_time[rid] = step['end_time']

    # decoders
    dec_logs = sorted(glob.glob(os.path.join(folder, 'decoder*_vllm_core_log.jsonl')))
    print(f'[all_disagg] {len(dec_logs)} decoder logs')
    dec_tt, dec_cf = {}, {}
    for lf in dec_logs:
        _, tt, cf = steps_to_events(parse_core_log(lf))
        dec_tt.update(tt)
        dec_cf.update(cf)

    df = build_records_disagg(pf_first_exec, pf_end_time, dec_tt, cached_first=dec_cf)
    print(f'  -> {len(df)} requests parsed')
    return df


def parse_adaptive_disagg():
    """
    adaptive_disagg_first:
      - prefiller: prefill for all iter_id=0 requests
      - decoders:  exactly 1 token per iter_id=0 request  →  TTFT for first iterations

    adaptive_disagg_rest:
      - iter_id=0: dummy prefill steps (to reload KV cache) + full decode tokens (TBT for first iters)
      - iter_id>=1: full prefill + decode on decoder GPUs  →  TTFT and TBT for rest iterations

    NOTE: the two sub-experiments ran at different wall-clock times, so timestamps are NOT
    comparable across phases.  TBT for iter_id=0 is computed entirely within adaptive_disagg_rest.

    Returns a DataFrame with columns:
      request_id, conv_id, iter_id, ttft, tbt_mean, tbt_list, num_tokens
    and an extra boolean column 'is_first_iter' (iter_id == 0).
    """
    # ── adaptive_disagg_first: TTFT for iter_id=0 ────────────────────────────
    f1 = os.path.join(BASE_DIR, 'adaptive_disagg_first')

    pf_steps = parse_core_log(os.path.join(f1, 'prefiller_vllm_core_log.jsonl'))
    pf_fe = {}   # prefill first-exec time per request
    for step in pf_steps:
        for rid in step['executed']:
            pf_fe.setdefault(rid, step['start_time'])

    dec_logs_f1 = sorted(glob.glob(os.path.join(f1, 'decoder*_vllm_core_log.jsonl')))
    print(f'[adaptive_disagg_first] {len(dec_logs_f1)} decoder logs')
    first_token_time = {}   # req_id -> time of the single decode token
    first_cached_f1  = {}   # req_id -> cached_tokens at that token
    for lf in dec_logs_f1:
        _, tt, cf = steps_to_events(parse_core_log(lf))
        for rid, times in tt.items():
            first_token_time[rid] = sorted(times)[0]
        first_cached_f1.update(cf)

    # TTFT for iter_id=0
    records_first_iter = []
    for rid, t1 in first_token_time.items():
        conv_id, iter_id = parse_request_id(rid)
        if iter_id != 0 or rid not in pf_fe:
            continue
        records_first_iter.append(dict(
            request_id=rid, conv_id=conv_id, iter_id=iter_id,
            ttft=t1 - pf_fe[rid],
            # TBT and cached_tokens will be filled from adaptive_disagg_rest below
            tbt_mean=float('nan'), tbt_list=[], num_tokens=1,
            cached_tokens=first_cached_f1.get(rid, float('nan')),
        ))
    ttft_first = {r['request_id']: r['ttft'] for r in records_first_iter}
    print(f'  -> TTFT computed for {len(records_first_iter)} iter_id=0 requests')

    # ── adaptive_disagg_rest: TBT for iter_id=0, and full metrics for iter_id>=1 ─
    f2 = os.path.join(BASE_DIR, 'adaptive_disagg_rest')
    dec_logs_f2 = sorted(glob.glob(os.path.join(f2, 'decoder*_vllm_core_log.jsonl')))
    print(f'[adaptive_disagg_rest]  {len(dec_logs_f2)} decoder logs')

    # Collect events separately for iter_id=0 and iter_id>=1
    id0_token_times = {}   # rid -> [token times] from adaptive_disagg_rest
    rest_fe  = {}          # first_exec for iter_id>=1
    rest_tt  = {}          # token_times for iter_id>=1
    rest_cf  = {}          # cached_first for iter_id>=1

    for lf in dec_logs_f2:
        steps = parse_core_log(lf)
        for step in steps:
            executed    = step['executed']
            finished    = step['finished']
            reasons     = step['finish_reasons']
            cached_list = step['num_cached_tokens']

            for rid in executed:
                conv_id, iter_id = parse_request_id(rid)
                if iter_id >= 1:
                    rest_fe.setdefault(rid, step['start_time'])
                # iter_id=0 exec-only steps are dummy prefill — skip for first_exec

            for i, (rid, reason) in enumerate(zip(finished, reasons)):
                conv_id, iter_id = parse_request_id(rid)
                cached_val = cached_list[i] if i < len(cached_list) else float('nan')
                if iter_id == 0:
                    # finish_reason=1 means dummy prefill completion — skip
                    # finish_reason=None means actual decode token — keep
                    if reason is None:
                        id0_token_times.setdefault(rid, []).append(step['end_time'])
                else:
                    rest_tt.setdefault(rid, []).append(step['end_time'])
                    rest_cf.setdefault(rid, cached_val)

    # Fill TBT for iter_id=0 from adaptive_disagg_rest token times
    for r in records_first_iter:
        rid = r['request_id']
        times = sorted(id0_token_times.get(rid, []))
        tbts = [times[k] - times[k-1] for k in range(1, len(times))]
        r['tbt_mean']  = float(np.mean(tbts)) if tbts else float('nan')
        r['tbt_list']  = tbts
        r['num_tokens'] = 1 + len(times)   # 1 from adaptive_disagg_first + rest here

    # TTFT + TBT for iter_id>=1 (non-disaggregated, like no_disagg)
    df_rest_iters = build_records_nodisagg(rest_fe, rest_tt, rest_cf)
    print(f'  -> iter_id>=1 requests: {len(df_rest_iters)}')

    df_first_iters = pd.DataFrame(records_first_iter)
    df_first_iters['is_first_iter'] = True
    df_rest_iters['is_first_iter']  = False

    df = pd.concat([df_first_iters, df_rest_iters], ignore_index=True)
    print(f'  -> total adaptive_disagg requests: {len(df)}')
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PROTOCOLS = ['no_disagg', 'all_disagg', 'adaptive_disagg']
COLORS    = {'no_disagg': '#4C72B0', 'all_disagg': '#DD8452', 'adaptive_disagg': '#55A868'}
LABELS    = {'no_disagg': 'No Disagg', 'all_disagg': 'All Disagg',
             'adaptive_disagg': 'Adaptive Disagg'}


def cdf(values):
    s = np.sort(values)
    p = np.linspace(0, 1, len(s))
    return s, p


def sel(flat_all, proto, metric):
    return flat_all[(flat_all['protocol'] == proto) & (flat_all['metric'] == metric)]['value'].dropna().values


def plot_cdf(flat_all):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['ttft', 'tbt']):
        for proto in PROTOCOLS:
            vals = sel(flat_all, proto, metric)
            if len(vals) == 0:
                continue
            x, y = cdf(vals)
            ax.plot(x, y, label=LABELS[proto], color=COLORS[proto], linewidth=1.8)
        ax.set_xlabel(f'{metric.upper()} (s)')
        ax.set_ylabel('CDF')
        ax.set_title(f'{metric.upper()} CDF')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cdf.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'cdf.png'), bbox_inches='tight', dpi=150)
    print('Saved cdf.pdf / cdf.png')
    plt.show()


def plot_boxplot(flat_all):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['ttft', 'tbt']):
        data = [sel(flat_all, p, metric) for p in PROTOCOLS]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2),
                        showfliers=False)
        for patch, proto in zip(bp['boxes'], PROTOCOLS):
            patch.set_facecolor(COLORS[proto])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(PROTOCOLS) + 1))
        ax.set_xticklabels([LABELS[p] for p in PROTOCOLS], rotation=10, ha='right')
        ax.set_ylabel(f'{metric.upper()} (s)')
        ax.set_title(f'{metric.upper()} Distribution')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.suptitle('Latency Distribution by Protocol', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'boxplot.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'boxplot.png'), bbox_inches='tight', dpi=150)
    print('Saved boxplot.pdf / boxplot.png')
    plt.show()


def plot_percentile_bars(flat_all):
    pcts = [50, 90, 99]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['ttft', 'tbt']):
        x     = np.arange(len(pcts))
        width = 0.25
        for i, proto in enumerate(PROTOCOLS):
            vals = sel(flat_all, proto, metric)
            heights = [np.percentile(vals, p) for p in pcts]
            ax.bar(x + i * width, heights, width, label=LABELS[proto],
                   color=COLORS[proto], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'p{p}' for p in pcts])
        ax.set_ylabel('Latency (s)')
        ax.set_title(f'{metric.upper()} Percentiles')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.suptitle('Latency Percentiles by Protocol', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'percentiles.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'percentiles.png'), bbox_inches='tight', dpi=150)
    print('Saved percentiles.pdf / percentiles.png')
    plt.show()


def plot_adaptive_breakdown(df_adaptive_disagg):
    """CDF breakdown: adaptive_disagg first-iter vs rest-iter, vs no_disagg and all_disagg."""
    flat_ad = to_flat_df(df_adaptive_disagg, 'adaptive_disagg')
    is_first_map = df_adaptive_disagg.set_index('request_id')['is_first_iter'].to_dict()
    flat_ad['is_first_iter'] = flat_ad['request_id'].map(is_first_map)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = [
        ('Adaptive: first iter (disagg)',  flat_ad['is_first_iter'] == True,  '#E66100'),
        ('Adaptive: rest iters (no disagg)', flat_ad['is_first_iter'] == False, '#5D3A9B'),
    ]
    for ax, metric in zip(axes, ['ttft', 'tbt']):
        for label, mask, color in groups:
            vals = flat_ad[mask & (flat_ad['metric'] == metric)]['value'].dropna().values
            if len(vals) == 0:
                continue
            x, y = cdf(vals)
            ax.plot(x, y, label=label, color=color, linewidth=1.8)
        ax.set_xlabel(f'{metric.upper()} (s)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Adaptive Disagg — {metric.upper()} breakdown')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'adaptive_breakdown.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'adaptive_breakdown.png'), bbox_inches='tight', dpi=150)
    print('Saved adaptive_breakdown.pdf / adaptive_breakdown.png')
    plt.show()


def plot_ttft_first_vs_rest(flat_all, df_adaptive_disagg):
    """
    Key comparison plot: TTFT of first iterations vs rest iterations across all protocols.
    For no_disagg and all_disagg: split by iter_id==0 vs iter_id>0.
    For adaptive_disagg: use is_first_iter flag.
    """
    # Build flat with iter type label
    dfs = []

    for df_proto, proto in [(df_no_disagg_g, 'no_disagg'), (df_all_disagg_g, 'all_disagg')]:
        flat = to_flat_df(df_proto, proto)
        iter_map = df_proto.set_index('request_id')['iter_id'].to_dict()
        flat['is_first_iter'] = flat['request_id'].map(iter_map) == 0
        dfs.append(flat)

    flat_ad = to_flat_df(df_adaptive_disagg, 'adaptive_disagg')
    is_first_map = df_adaptive_disagg.set_index('request_id')['is_first_iter'].to_dict()
    flat_ad['is_first_iter'] = flat_ad['request_id'].map(is_first_map)
    dfs.append(flat_ad)

    combined = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    styles = {
        ('no_disagg',       True):  ('#4C72B0', '-',  'No Disagg — first iter'),
        ('no_disagg',       False): ('#4C72B0', '--', 'No Disagg — rest iters'),
        ('all_disagg',      True):  ('#DD8452', '-',  'All Disagg — first iter'),
        ('all_disagg',      False): ('#DD8452', '--', 'All Disagg — rest iters'),
        ('adaptive_disagg', True):  ('#55A868', '-',  'Adaptive — first iter (disagg)'),
        ('adaptive_disagg', False): ('#55A868', '--', 'Adaptive — rest iters (no disagg)'),
    }

    for ax, metric in zip(axes, ['ttft', 'tbt']):
        for (proto, is_first), (color, ls, label) in styles.items():
            mask = (combined['protocol'] == proto) & \
                   (combined['is_first_iter'] == is_first) & \
                   (combined['metric'] == metric)
            vals = combined[mask]['value'].dropna().values
            if len(vals) == 0:
                continue
            x, y = cdf(vals)
            ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.6)
        ax.set_xlabel(f'{metric.upper()} (s)')
        ax.set_ylabel('CDF')
        ax.set_title(f'{metric.upper()}: first iter vs rest iters')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ttft_first_vs_rest.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'ttft_first_vs_rest.png'), bbox_inches='tight', dpi=150)
    print('Saved ttft_first_vs_rest.pdf / ttft_first_vs_rest.png')
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse — also exposed as module-level for plot_ttft_first_vs_rest
    df_no_disagg_g       = parse_no_disagg()
    df_all_disagg_g      = parse_all_disagg()
    df_adaptive_disagg_g = parse_adaptive_disagg()

    # Save per-protocol CSVs (drop list column)
    for df, name in [(df_no_disagg_g, 'no_disagg'),
                     (df_all_disagg_g, 'all_disagg'),
                     (df_adaptive_disagg_g, 'adaptive_disagg')]:
        save_cols = [c for c in df.columns if c != 'tbt_list']
        df[save_cols].to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)

    # Combined flat table
    flat_all = pd.concat([
        to_flat_df(df_no_disagg_g,       'no_disagg'),
        to_flat_df(df_all_disagg_g,      'all_disagg'),
        to_flat_df(df_adaptive_disagg_g, 'adaptive_disagg'),
    ], ignore_index=True)
    flat_all.to_csv(os.path.join(OUT_DIR, 'all_protocols_flat.csv'), index=False)
    print(f'\nSaved all_protocols_flat.csv  ({len(flat_all)} rows)')

    # Summary table
    summary = flat_all.groupby(['protocol', 'metric'])['value'].agg(
        count='count', mean='mean', median='median',
        p90=lambda x: np.percentile(x, 90),
        p99=lambda x: np.percentile(x, 99),
    ).round(4)
    print('\n--- Summary ---')
    print(summary.to_string())

    # ── Summary split by first vs rest iter ────────────────────────────────
    def flat_with_iter_flag(df, proto):
        flat = to_flat_df(df, proto)
        if 'is_first_iter' in df.columns:
            m = df.set_index('request_id')['is_first_iter'].to_dict()
        else:
            m = {r: (df.set_index('request_id').loc[r, 'iter_id'] == 0)
                 for r in df['request_id']}
        flat['is_first_iter'] = flat['request_id'].map(m)
        return flat

    flat_split = pd.concat([
        flat_with_iter_flag(df_no_disagg_g,       'no_disagg'),
        flat_with_iter_flag(df_all_disagg_g,       'all_disagg'),
        flat_with_iter_flag(df_adaptive_disagg_g,  'adaptive_disagg'),
    ], ignore_index=True)
    flat_split.to_csv(os.path.join(OUT_DIR, 'all_protocols_split.csv'), index=False)

    summary2 = flat_split.groupby(['protocol', 'is_first_iter', 'metric'])['value'].agg(
        count='count', mean='mean', median='median',
        p90=lambda x: np.percentile(x, 90),
        p99=lambda x: np.percentile(x, 99),
    ).round(4)
    print('\n--- Summary (first iter vs rest) ---')
    print(summary2.to_string())

    # Plots
    plot_cdf(flat_all)
    plot_boxplot(flat_all)
    plot_percentile_bars(flat_all)
    plot_adaptive_breakdown(df_adaptive_disagg_g)
    plot_ttft_first_vs_rest(flat_all, df_adaptive_disagg_g)

    print('\nDone. Outputs in:', OUT_DIR)
