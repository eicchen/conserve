"""Virtual prefiller queue for per_turn_adaptive_disagg_decoders.

Live in-memory simulation of the prefiller during the decoder-side run.
Seeded from a recorded `adaptive_disagg_prefiller` per_step_latency.csv,
which gives the original (start_time, end_time) for every iter-0 prefill
the real prefiller serviced. The simulator then layers wrong-predict
blocks on top, where each block:

  - waits for all currently-in-flight (simulated) iter-0 prefills to
    finish (busy_until = max effective_end of in-flight events),
  - then exclusively occupies the prefiller for prefill_hit_ms,
  - delays any subsequent recorded iter-0 whose effective start falls
    inside [block_start, block_end].

Two outputs:
  1. `query_and_block(T_wp_wall, prefill_hit_ms)` returns the queueing
     delay seen by the wrong-predicted conv (so the decoder runner can
     sleep that long before doing the real prefill).
  2. `write_synthetic_trace(out_path)` at end of run emits a synthetic
     prefiller per_step_latency.csv with the shifted iter-0 timings; the
     analysis layer reads this in place of the original prefiller trace
     when scoring per_turn_adaptive_disagg_decoders.
"""

import asyncio
import csv
from pathlib import Path


def _effective_start(rel_arrival: float, blocks):
    """Push rel_arrival forward through any block whose [start, end)
    contains it. Blocks are sorted by start. After being pushed, the new
    position may fall inside another block, so we sweep until stable."""
    result = rel_arrival
    changed = True
    while changed:
        changed = False
        for bs, be in blocks:
            if bs <= result < be:
                result = be
                changed = True
                break
            if bs >= result:
                # Sorted blocks; no later one can contain `result` if its
                # start is already past `result`.
                break
    return result


class VirtualPrefiller:
    def __init__(self, recorded_iter0_events, prompt_max_tokens_by_conv=None):
        """recorded_iter0_events: list of (conv_id, rel_start_sec, duration_sec),
        where rel_start_sec is measured from the first event's start (so the
        first arrival is at rel_time 0). Sorted by rel_start_sec.
        prompt_max_tokens_by_conv: optional {conv_id: (prompt_tokens, max_tokens)}
        used only when writing the synthetic CSV."""
        self.recorded = sorted(recorded_iter0_events, key=lambda x: x[1])
        self.prompt_max_tokens = prompt_max_tokens_by_conv or {}
        self.t0_wall = None
        self.blocks = []        # sorted list of (block_rel_start, block_rel_end)
        self.lock = asyncio.Lock()

    def set_t0(self, t0_wall: float):
        """Anchor wall-clock time of rel_time=0. Call when the first live
        conversation is scheduled."""
        if self.t0_wall is None:
            self.t0_wall = t0_wall

    def _to_rel(self, wall_clock: float) -> float:
        assert self.t0_wall is not None, "set_t0 must be called before query"
        return wall_clock - self.t0_wall

    def _busy_until_rel(self, rel_T: float) -> float:
        """Latest effective_end among events that are in-flight at rel_T."""
        busy = rel_T
        for _, rec_start, rec_dur in self.recorded:
            eff_start = _effective_start(rec_start, self.blocks)
            if eff_start > rel_T:
                # sorted by rec_start; eff_start can only grow further. But
                # blocks may shift later events earlier-in-list to start
                # later than this one, so we can't early-break safely.
                continue
            eff_end = eff_start + rec_dur
            if eff_start <= rel_T < eff_end:
                if eff_end > busy:
                    busy = eff_end
        return busy

    async def query_and_block(self, T_wp_wall: float, prefill_hit_ms: float) -> float:
        """Inject a wrong-predict block at wall-clock T_wp_wall. Returns the
        queueing delay (seconds) the wrong-predicted conv must wait before
        the virtual prefill can start — i.e., time to drain in-flight
        prefills. The block itself adds another prefill_hit_ms of wait on
        top (the caller adds that, plus KV/bookkeeping, separately)."""
        async with self.lock:
            rel_T = self._to_rel(T_wp_wall)
            busy_rel = self._busy_until_rel(rel_T)
            block_start = busy_rel
            block_end = block_start + prefill_hit_ms / 1000.0
            # keep self.blocks sorted by start
            self.blocks.append((block_start, block_end))
            self.blocks.sort(key=lambda b: b[0])
            return busy_rel - rel_T   # queueing delay (>= 0)

    def write_synthetic_trace(self, out_path):
        """Emit a synthetic prefiller per_step_latency.csv with the shifted
        iter-0 timings. start_time / end_time are in the live run's
        wall-clock so the analysis can join with the decoder trace."""
        assert self.t0_wall is not None, "no live arrivals seen; nothing to write"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conv_id", "step_id", "prompt_tokens", "max_tokens",
                        "start_time", "end_time", "latency_sec"])
            for conv_id, rec_start, rec_dur in self.recorded:
                eff_start_rel = _effective_start(rec_start, self.blocks)
                eff_end_rel = eff_start_rel + rec_dur
                sim_start = self.t0_wall + eff_start_rel
                sim_end = self.t0_wall + eff_end_rel
                pt, mt = self.prompt_max_tokens.get(conv_id, ("", ""))
                w.writerow([conv_id, 0, pt, mt,
                            f"{sim_start:.6f}", f"{sim_end:.6f}",
                            f"{rec_dur:.4f}"])


def load_recorded_iter0(prefiller_trace_dir):
    """Read per_step_latency.csv from a recorded adaptive_disagg_prefiller
    run dir; return (events, prompt_max_tokens) where events is a list of
    (conv_id, rel_start_sec, duration_sec)."""
    path = Path(prefiller_trace_dir) / "per_step_latency.csv"
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["step_id"]) != 0:
                continue
            rows.append((
                int(row["conv_id"]),
                int(row["prompt_tokens"]) if row["prompt_tokens"] else 0,
                int(row["max_tokens"]) if row["max_tokens"] else 0,
                float(row["start_time"]),
                float(row["end_time"]),
            ))
    if not rows:
        raise ValueError(f"No iter-0 rows found in {path}")
    rows.sort(key=lambda x: x[3])
    t_ref = rows[0][3]
    events = [(c, st - t_ref, en - st) for c, _, _, st, en in rows]
    prompt_max = {c: (pt, mt) for c, pt, mt, _, _ in rows}
    return events, prompt_max
