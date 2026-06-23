"""
Experiment 1: prefix-cache hit vs miss cost.

For each L in a sweep, submit the same prompt twice in succession to a vLLM
engine with prefix caching enabled. The first call pays full prefill (miss);
the second hits the cache (~ block-table assembly only). Report both
latencies, demonstrating that the "cached compound prefix" trick in
adaptive_disagg has near-zero iter-0 prefill cost.

Output:
  cache_cost_data/plan.json
  cache_cost_data/cell_<idx>_engine.jsonl
  cache_cost_data/cell_<idx>_core.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
from config import MODEL_DIR, PROFILING_DATA_DIR, MODEL


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # local override

# --out points at a cache_cost_data dir (or a per-shard subdir); --l-values
# restricts this process to a subset of L for GPU-parallel sharding. cell_idx
# is computed from the L's global index so shard cell files never collide.
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", type=str,
                 default=f"{REPO_ROOT}/paper/figures/section3/output/300W/cache_cost_data")
_ap.add_argument("--l-values", type=str, default="",
                 help="comma-separated L subset for this shard; empty = all")
_args = _ap.parse_args()
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

L_VALUES = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288,
            16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536]
N_REPLICATES = 5  # measure miss/hit pair N times per L

MY_L_VALUES = ([int(x) for x in _args.l_values.split(",")]
               if _args.l_values else list(L_VALUES))


def build_prompt(cell_idx: int, L: int, tokenizer) -> str:
    """Pick a prompt that tokenizes to ~L tokens.

    The prompts_{L}x2048.json files have 2048 distinct natural-language
    prompts each tokenizing to ~L tokens; picking by cell_idx guarantees each
    cell gets a unique full-content prompt, so cache misses are true full
    misses (not partial hits where only the salt block differs).

    L in {40960, 49152, 57344} have no dedicated prompt file; fall back to
    truncating a prompts_65536x2048.json prompt to the first L tokens.
    """
    p = Path(f"{PROFILING_DATA_DIR}/prompts_{L}x2048.json")
    if p.exists():
        with open(p) as f:
            prompts = json.load(f)
        return prompts[cell_idx % len(prompts)]["prompt"]
    src = Path(f"{PROFILING_DATA_DIR}/prompts_65536x2048.json")
    with open(src) as f:
        prompts = json.load(f)
    text = prompts[cell_idx % len(prompts)]["prompt"]
    ids = tokenizer.encode(text, add_special_tokens=False)[:L]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    llm = LLM(
        model=MODEL,
        dtype="auto",
        download_dir=MODEL_DIR,
        rope_scaling={"rope_type": "dynamic", "factor": 2.5},
        max_num_batched_tokens=67584,
        max_num_seqs=64,
        max_model_len=81920,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=1.2, top_p=1.0, max_tokens=2,
        logit_bias={151643: -100, 151644: -100, 151645: -100},
    )

    # Warmup: discarded generations so the first measured cell doesn't pay
    # cold-start (CUDA init, cuBLAS autotuning, allocator growth). Without
    # this the first few cells (smallest L) read artificially slow.
    print("warmup...", flush=True)
    for wl in [512, 8192, 32768]:
        wp = build_prompt(0, wl, tokenizer)
        llm.generate(fixed_batches=[[wp], [wp]], sampling_params=sp,
                     engine_log_file="/tmp/cache_cost_warmup_engine.jsonl",
                     core_log_file="/tmp/cache_cost_warmup_core.jsonl",
                     use_tqdm=False)
    print("warmup done", flush=True)

    # Per-L, per-replicate: a unique salt so each (L, rep) gets a fresh
    # cache-miss prompt; the second submission within the cell is the hit.
    # cell_idx = (global L index) * N_REPLICATES + rep -> stable & unique
    # whether this process runs all L or just a shard's subset.
    plan_cells = []
    for L in MY_L_VALUES:
        L_idx = L_VALUES.index(L)
        for rep in range(N_REPLICATES):
            cell_idx = L_idx * N_REPLICATES + rep
            prompt = build_prompt(cell_idx, L, tokenizer)
            engine_log = OUT / f"cell_{cell_idx:03d}_engine.jsonl"
            core_log = OUT / f"cell_{cell_idx:03d}_core.jsonl"
            # First call: cache miss. Second call (same prompt): hit.
            fixed_batches = [[prompt], [prompt]]
            t0 = time.time()
            llm.generate(
                fixed_batches=fixed_batches,
                sampling_params=sp,
                engine_log_file=str(engine_log),
                core_log_file=str(core_log),
                use_tqdm=False,
            )
            dt = time.time() - t0
            plan_cells.append({
                "cell_idx": cell_idx, "L": L, "rep": rep,
                "wall_s": round(dt, 2),
                "engine_log": engine_log.name, "core_log": core_log.name,
            })
            print(f"  cell {cell_idx:>3}  L={L:>5}  rep={rep}  {dt:.2f}s", flush=True)

    with open(OUT / "plan.json", "w") as f:
        json.dump({
            "model": MODEL, "L_values": L_VALUES, "n_replicates": N_REPLICATES,
            "cells": plan_cells,
        }, f, indent=2)
    print(f"\nWrote {len(plan_cells)} cells to {OUT}")


if __name__ == "__main__":
    main()
