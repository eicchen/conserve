"""Given one prefiller_sweep run directory (one seed's logs), produce the two
per-conv arrival-trace JSONs that the replay policies consume:

  iter0_prefill_start_arrival_trace.json   — for no_disagg_oracle / all_disagg
  iter1_decoding_start_arrival_trace.json  — for adaptive_disagg_decoders /
                                              per_turn_adaptive_disagg_decoders

Each entry is `{"conv_id": int, "offset_sec": float}` sorted by offset, with
the earliest event at offset 0. The prefill-start file uses the prefiller's
first-execution time for iter-0; the decoding-start file uses the matched
decoder's first-execution time for iter-0 (so the replays' arrival pattern
mirrors the original disaggregated timing).

Usage:
    python extract_arrival_traces.py <seed_log_dir> [<seed_log_dir> ...]

Each seed_log_dir must contain prefiller_vllm_{engine,core}_log.jsonl plus
the decoderN_vllm_{engine,core}_log.jsonl files. Multiple dirs can be passed
to extract traces in a batch (e.g., for a 10-seed prefiller sweep)."""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def parse_rid(rid: str):
    parts = rid.split("-")
    if len(parts) != 4 or parts[0] != "cmpl":
        return None, None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None, None


def first_exec_for_iter0(eng_log: Path, core_log: Path) -> dict:
    """Return {conv_id: first_exec_ts} for iter-0 requests on one engine."""
    iter0_rids = set()
    with open(eng_log) as f:
        for line in f:
            d = json.loads(line)
            if d["event"] != "request_start":
                continue
            rid = d["request_id"]
            conv, it = parse_rid(rid)
            if conv is not None and it == 0:
                iter0_rids.add(rid)
    fe = {}
    with open(core_log) as f:
        for line in f:
            d = json.loads(line)
            if d["event"] != "step_end":
                continue
            ts = parse_ts(d["timestamp"])
            for rid in d.get("executed_request_ids", []):
                if rid in iter0_rids and rid not in fe:
                    fe[rid] = ts
    return {parse_rid(rid)[0]: ts for rid, ts in fe.items()}


def extract(seed_dir: Path) -> None:
    pref_start = first_exec_for_iter0(
        seed_dir / "prefiller_vllm_engine_log.jsonl",
        seed_dir / "prefiller_vllm_core_log.jsonl",
    )
    dec_start = {}
    for dn in ("decoder1", "decoder2", "decoder3"):
        e = seed_dir / f"{dn}_vllm_engine_log.jsonl"
        c = seed_dir / f"{dn}_vllm_core_log.jsonl"
        if e.exists():
            dec_start.update(first_exec_for_iter0(e, c))

    common = sorted(set(pref_start) & set(dec_start))
    if not common:
        print(f"  WARNING: no common conv_ids in {seed_dir}; skipping")
        return

    t0p = min(pref_start[c] for c in common)
    t0d = min(dec_start[c]  for c in common)
    pref_trace = sorted(
        [{"conv_id": c, "offset_sec": pref_start[c] - t0p} for c in common],
        key=lambda e: e["offset_sec"],
    )
    dec_trace = sorted(
        [{"conv_id": c, "offset_sec": dec_start[c] - t0d} for c in common],
        key=lambda e: e["offset_sec"],
    )

    (seed_dir / "iter0_prefill_start_arrival_trace.json").write_text(
        json.dumps(pref_trace, indent=2))
    (seed_dir / "iter1_decoding_start_arrival_trace.json").write_text(
        json.dumps(dec_trace, indent=2))

    gaps = [dec_start[c] - pref_start[c] for c in common]
    print(f"  {seed_dir.name}: prefill span={pref_trace[-1]['offset_sec']:6.2f}s  "
          f"decode span={dec_trace[-1]['offset_sec']:6.2f}s  "
          f"mean prefill→decode gap={statistics.mean(gaps)*1000:.0f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_arrival_traces.py "
              "<seed_log_dir> [<seed_log_dir> ...]", file=sys.stderr)
        sys.exit(2)
    for arg in sys.argv[1:]:
        extract(Path(arg))
