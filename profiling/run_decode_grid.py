"""
Controlled (B, L) decode-grid profile for Qwen3-0.6B.

For each (per-step batch size B, per-request KV length L) cell:
  - Submit B identical prompts of length L tokens (built from a cell-unique
    salt + a long run of a fixed filler token).
  - vLLM with prefix caching: rep-1 prefills once; reps 2..K hit the cache.
  - Generate N_decode tokens. With max_tokens=N_decode the engine produces
    one prefill step plus N_decode-1 decode steps per rep, but with B
    identical prompts the steps after prefill are pure-decode at batch B.
  - Per-cell logs are written separately so analysis can identify each step's
    operating point (B fixed, per-request KV = L + step_idx).

Inputs:  none
Outputs in OUT_DIR:
  plan.json                       - grid spec and ordered cell list
  engine_cell_NN.jsonl            - per-cell vLLM engine step durations
  core_cell_NN.jsonl              - per-cell core log (step_end events)
  dcgmi_trace.tsv                 - GPU telemetry across whole sweep
  run_log.txt                     - human-readable timing log per cell
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch.distributed as dist

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
from paths import MODEL_DIR, PROFILING_DATA_DIR


from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUT_DIR = (REPO_ROOT / "paper/figures/section3/output/300W/decode_grid_data")

B_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
L_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
KV_BUDGET = 200_000

K_REPS = 8
N_DECODE = 64

ROPE_FACTOR = 2.0           # native 32768 * 2 = 65536 max model len
MAX_MODEL_LEN = 65536

FILLER_TOKEN = 100
SALT_BASE = 300


_PROMPTS_CACHE: dict = {}

def load_prompts(L: int):
    if L not in _PROMPTS_CACHE:
        p = Path(f"{PROFILING_DATA_DIR}/prompts_{L}x2048.json")
        with open(p) as f:
            _PROMPTS_CACHE[L] = json.load(f)
    return _PROMPTS_CACHE[L]


def build_prompt(cell_idx: int, L: int, tokenizer) -> str:
    """Pick a unique natural-language prompt of length ~L tokens.

    Earlier versions of this script built prompts by decoding `[salt] + [100]*(L-1)`.
    Qwen's tokenizer merges runs of the same id back into a single token on
    re-encode (about 4 consecutive `[100]`s -> 1), so the engine actually saw
    only ~L/4 tokens of KV per request. That biased the planar-fit gamma
    coefficient by ~4x. Using natural-language prompts from prompts_{L}x2048.json
    keeps the per-request KV faithful to L.
    """
    prompts = load_prompts(L)
    return prompts[cell_idx % len(prompts)]["prompt"]


def build_cells():
    """Return canonical ordered cell list (B, L). Order is by index, so cell_idx is stable."""
    out = []
    for B in B_VALUES:
        for L in L_VALUES:
            if B * L > KV_BUDGET:
                continue
            out.append((B, L))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="0-based shard index for parallel runs")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of shards (cells split round-robin by cell_idx mod num_shards)")
    ap.add_argument("--pilot", action="store_true",
                    help="Run a tiny 3-cell pilot subset and exit")
    ap.add_argument("--enable-dcgmi", action="store_true",
                    help="Capture dcgmi telemetry to dcgmi_trace.tsv (disabled by default for parallel runs)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells = build_cells()
    all_indexed = list(enumerate(all_cells))

    if args.pilot:
        # Three small cells, one per shard if num_shards >= 3. Shard 0 takes
        # the largest (B=1, L=32768) since it's the slowest prefill; the rest
        # are cheap.
        pilot_set = [(1, 32768), (16, 8192), (128, 512)]
        pilot_indexed = [(i, c) for (i, c) in all_indexed if c in pilot_set]
        # Round-robin pilot cells across shards
        sharded = [(i, c) for shard_pos, (i, c) in enumerate(pilot_indexed)
                   if shard_pos % args.num_shards == args.shard_idx]
    else:
        sharded = [(i, c) for (i, c) in all_indexed
                   if i % args.num_shards == args.shard_idx]

    # Write a global plan file from shard 0 only to avoid races.
    if args.shard_idx == 0:
        with open(out_dir / "plan.json", "w") as f:
            json.dump({
                "model": MODEL,
                "B_values": B_VALUES, "L_values": L_VALUES,
                "kv_budget": KV_BUDGET,
                "k_reps": K_REPS, "n_decode": N_DECODE,
                "rope_factor": ROPE_FACTOR, "max_model_len": MAX_MODEL_LEN,
                "filler_token": FILLER_TOKEN, "salt_base": SALT_BASE,
                "num_shards": args.num_shards,
                "cells": [
                    {"cell_idx": i, "B": B, "L": L, "kv_at_step0": B * L,
                     "shard": i % args.num_shards}
                    for i, (B, L) in all_indexed
                ],
            }, f, indent=2)

    print(f"=== Decode grid (shard {args.shard_idx} of {args.num_shards}) ===")
    print(f"  this shard runs {len(sharded)} of {len(all_cells)} cells  "
          f"(K={K_REPS}, N_decode={N_DECODE})")
    for cell_idx, (B, L) in sharded:
        print(f"    cell {cell_idx:>2}  B={B:>3}  L={L:>6}  active_KV_step0={B*L:>7}")
    print()

    MAX_TOK = max(L_VALUES) + 1024
    MAX_SEQS = max(B_VALUES) * K_REPS + 16

    # NOTE: requires VLLM_ENABLE_V1_MULTIPROCESSING=0 (set by launcher) so the
    # engine runs in-process (InprocClient). In that mode the core_log_file
    # kwarg passed to generate() reaches the engine via direct method call.
    llm = LLM(
        model=MODEL,
        dtype="auto",
        download_dir=MODEL_DIR,
        rope_scaling={"rope_type": "dynamic", "factor": ROPE_FACTOR},
        max_num_batched_tokens=MAX_TOK,
        max_num_seqs=MAX_SEQS,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=1.2,
        top_p=1.0,
        max_tokens=N_DECODE,
        logit_bias={
            2: -100, 13: -100,
            151643: -100, 151644: -100, 151645: -100,
        },
    )

    dcgmi_proc = None
    if args.enable_dcgmi:
        dcgmi_file = out_dir / f"dcgmi_trace_shard{args.shard_idx}.tsv"
        dcgmi_cmd = ["bash", "-c",
                     f"dcgmi dmon -e 155,156,157,1130,1131,1132,1133,150,140,151,152,153 -d 1 "
                     f"| ts '%Y-%m-%dT%H:%M:%.S' >> {dcgmi_file}"]
        dcgmi_proc = subprocess.Popen(dcgmi_cmd, preexec_fn=os.setsid)
        time.sleep(3)

    run_log = open(out_dir / f"run_log_shard{args.shard_idx}.txt", "w")
    try:
        t_start = time.time()
        for cell_idx, (B, L) in sharded:
            prompt = build_prompt(cell_idx, L, tokenizer)
            actual_len = len(tokenizer.encode(prompt))
            if actual_len != L:
                print(f"[cell {cell_idx:>2}] note: prompt re-tokenized to {actual_len} (asked {L})",
                      flush=True)
            fixed_batches = [
                [prompt for _ in range(B)]
                for _ in range(K_REPS)
            ]
            engine_log = out_dir / f"engine_cell_{cell_idx:03d}.jsonl"
            core_log = out_dir / f"core_cell_{cell_idx:03d}.jsonl"

            t0 = time.time()
            try:
                llm.generate(
                    fixed_batches=fixed_batches,
                    sampling_params=sampling,
                    engine_log_file=str(engine_log),
                    core_log_file=str(core_log),
                    use_tqdm=False,
                )
            except Exception as e:
                msg = f"[cell {cell_idx:>2}] B={B:>3} L={L:>6} FAILED: {e}"
                print(msg, flush=True)
                run_log.write(msg + "\n")
                continue
            dt = time.time() - t0

            msg = f"[cell {cell_idx:>2}] B={B:>3} L={L:>6} active_KV={B*L:>7} done in {dt:5.1f}s"
            print(msg, flush=True)
            run_log.write(msg + "\n")
            run_log.flush()
        total = time.time() - t_start
        print(f"\nShard {args.shard_idx} done in {total:.1f}s")
        run_log.write(f"\nTotal wall time: {total:.1f}s\n")
    finally:
        run_log.close()
        if dcgmi_proc is not None:
            try:
                os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGTERM)
                dcgmi_proc.wait(timeout=5)
            except Exception as e:
                print(f"dcgmi cleanup: {e}")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
