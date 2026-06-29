"""
Exp 2 & Exp 3: prefill x decode interference, with and without prefix-cache hit.

For each (B_decode, L_prefill, cache_condition) cell:
  1. Submit B_decode background decoder requests (short prompt, long max_tokens).
  2. Wait 1 second so decoders enter steady-state decode.
  3. Submit one prefill request:
       - cache=miss: a unique prompt of length L_prefill (not previously seen)
       - cache=hit : a prompt that was pre-warmed before the cell started
  4. Wait for prefill to complete, then for all decoders.
  5. Cancel any still-running decoders (cap their max_tokens at completion).

Then we parse the engine core_log and recover, per cell:
  - prefill TTFT under load
  - decoder TBT during the prefill window

Outputs:
  interference_data/server_engine.jsonl
  interference_data/server_core.jsonl
  interference_data/cells.json  (cell metadata: request_ids, submission times)
  interference_data/server.log
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_DIR, MODEL_DATA_DIR, MODEL, MODEL_SHORT, TENSOR_PARALLEL_SIZE, PROFILE


import aiohttp

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # local override

# Shard-aware: each shard runs a strided subset of the replicates on its own
# GPU/port/output dir, so the 3 free GPUs can run in parallel. Merge afterward.
_ap = argparse.ArgumentParser()
_ap.add_argument("--n-shards", type=int, default=1)
_ap.add_argument("--shard-id", type=int, default=0)
_ap.add_argument("--port", type=int, default=7700)
_ap.add_argument("--out", type=str,
                 default=str(MODEL_DATA_DIR / "paper" / "section3" / "profiling" / "interference_data"))
_args = _ap.parse_args()
PORT = _args.port
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

B_DECODE_VALUES = [0, 1, 2, 4, 8, 16, 32]
L_PREFILL_VALUES = [1024, 4096, 16384, 32768]
DECODE_PROMPT_LEN = 8       # short, so decoders prefill quickly
DECODE_MAX_TOKENS = 300     # ~5s of decode at 18 ms/step
PREFILL_MAX_TOKENS = 2      # produce 1 decode token after prefill
DECODE_STEADY_DELAY = 1.0   # seconds to wait after submitting decoders
N_REPLICATES = 64           # repeat each (B, L_prefill, cache) cell this many times


def start_server():
    """Start vllm serve in background. Returns (process, env)."""
    _tmp = MODEL_DATA_DIR / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(_tmp)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "localhost",
        "--port", str(PORT),
        "--download-dir", str(MODEL_DIR),
        *PROFILE.vllm_serve_flags,
        "--max-num-batched-tokens", "33792",
        "--max-num-seqs", "128",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--engine-log-file", str(OUT / "server_engine.jsonl"),
        "--core-log-file", str(OUT / "server_core.jsonl"),
        "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
    ]
    log = open(OUT / "server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    return proc, log


async def wait_for_server(timeout_s=600):
    url = f"http://localhost:{PORT}/v1/models"
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise RuntimeError("Server did not come up in time")


def stop_server(proc, log_file):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    log_file.close()


_PROMPT_CACHE: dict = {}

def load_prompts(L: int):
    """Load the prompts_{L}x2048.json file (cached across calls)."""
    if L not in _PROMPT_CACHE:
        p = MODEL_DATA_DIR / "long_prompts" / f"prompts_{L}x2048.json"
        with open(p) as f:
            _PROMPT_CACHE[L] = json.load(f)
    return _PROMPT_CACHE[L]


def build_text(salt: int, L: int, tokenizer, *, unique: bool = True) -> str:
    """Get a unique-content prompt of length ~L tokens.

    unique=True: pick a distinct natural-language prompt from the L-sized
                 prompt file (2048 distinct prompts available). Ensures every
                 block is a true cache miss.
    unique=False: short filler form, used only for tiny decoder prompts where
                  content does not matter.
    """
    if unique:
        prompts = load_prompts(L)
        return prompts[salt % len(prompts)]["prompt"]
    return tokenizer.decode([salt] + [100] * (L - 1), skip_special_tokens=True)


async def submit(session, request_id, prompt, max_tokens):
    """Submit one completion request. Returns (request_id, client_submit_time, client_finish_time)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.2,
        "top_p": 1.0,
        "stream": False,
        "logit_bias": {str(tid): -100 for tid in PROFILE.eos_token_ids},
        "request_id": request_id,
    }
    url = f"http://localhost:{PORT}/v1/completions"
    t_submit = time.time()
    async with session.post(url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=600)) as r:
        await r.read()
    t_done = time.time()
    return request_id, t_submit, t_done


async def warm_cache(session, prompt):
    """Submit one trivial request to seed the prefix cache for `prompt`."""
    rid = f"warm_{uuid.uuid4().hex[:8]}"
    await submit(session, rid, prompt, PREFILL_MAX_TOKENS)


async def run_cell(session, cell_idx, B_decode, L_prefill, cache_hit, tokenizer):
    """Run one cell and return its metadata."""
    decoder_prompt = build_text(900 + cell_idx, DECODE_PROMPT_LEN, tokenizer, unique=False)
    # Use a per-cell salt offset for unique decoder rids
    decoder_rids = [f"c{cell_idx:03d}_dec{i}" for i in range(B_decode)]
    prefill_rid = f"c{cell_idx:03d}_pref"

    # Build prefill prompt with cell-unique random tokens so every block is
    # a fresh cache miss. If cache_hit, we pre-warm before submitting decoders.
    prefill_prompt = build_text(50_000 + cell_idx, L_prefill, tokenizer, unique=True)

    if cache_hit:
        await warm_cache(session, prefill_prompt)

    # Submit decoders
    decoder_tasks = [
        asyncio.create_task(submit(session, rid, decoder_prompt, DECODE_MAX_TOKENS))
        for rid in decoder_rids
    ]
    t_decoders_submitted = time.time()

    # Wait for decoders to reach steady state
    await asyncio.sleep(DECODE_STEADY_DELAY)

    # Submit prefill
    t_prefill_submit = time.time()
    prefill_task = asyncio.create_task(
        submit(session, prefill_rid, prefill_prompt, PREFILL_MAX_TOKENS)
    )

    # Wait for prefill to complete
    _, _, t_prefill_done = await prefill_task

    # Optionally let decoders finish too so the engine queue clears.
    await asyncio.gather(*decoder_tasks, return_exceptions=True)

    return {
        "cell_idx": cell_idx,
        "B_decode": B_decode,
        "L_prefill": L_prefill,
        "cache_hit": cache_hit,
        "decoder_rids": decoder_rids,
        "prefill_rid": prefill_rid,
        "t_decoders_submitted": t_decoders_submitted,
        "t_prefill_submit": t_prefill_submit,
        "t_prefill_done": t_prefill_done,
    }


async def main():
    # Need a tokenizer to build prompts of exact token length. Use HF directly
    # (the server hasn't started yet, and we want this offline).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, cache_dir=MODEL_DIR, trust_remote_code=True,
    )

    # Start server
    print("Starting server...", flush=True)
    proc, log_file = start_server()
    try:
        await wait_for_server(timeout_s=600)
        print("Server up. Running cells...", flush=True)

        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)) as session:
            # Build the cell list. For each (B_decode, L_prefill) we run BOTH
            # cache=miss and cache=hit, so the same plot pair is comparable.
            # cell_idx = rep*CELLS_PER_REP + local keeps it globally unique
            # across shards, so per-cell prompt salts never collide.
            CELLS_PER_REP = len(B_DECODE_VALUES) * len(L_PREFILL_VALUES) * 2
            my_reps = list(range(_args.shard_id, N_REPLICATES, _args.n_shards))
            print(f"shard {_args.shard_id}/{_args.n_shards}: reps {my_reps}", flush=True)
            results = []
            t0 = time.time()
            for rep in my_reps:
                local = 0
                for cache_hit in [False, True]:
                    for L_prefill in L_PREFILL_VALUES:
                        for B_decode in B_DECODE_VALUES:
                            cell_idx = rep * CELLS_PER_REP + local
                            t_cell_start = time.time()
                            meta = await run_cell(session, cell_idx, B_decode,
                                                  L_prefill, cache_hit, tokenizer)
                            meta["rep"] = rep
                            dt = time.time() - t_cell_start
                            cond = "hit" if cache_hit else "miss"
                            print(f"  rep {rep:>2} cell {cell_idx:>5} {cond:>4} "
                                  f"B_dec={B_decode:>2} L={L_prefill:>5}  {dt:.1f}s",
                                  flush=True)
                            results.append(meta)
                            local += 1
            total = time.time() - t0
            print(f"\nAll {len(results)} cells done in {total:.1f}s")

            with open(OUT / "cells.json", "w") as f:
                json.dump({
                    "model": MODEL,
                    "decode_prompt_len": DECODE_PROMPT_LEN,
                    "decode_max_tokens": DECODE_MAX_TOKENS,
                    "prefill_max_tokens": PREFILL_MAX_TOKENS,
                    "decode_steady_delay": DECODE_STEADY_DELAY,
                    "n_replicates": N_REPLICATES,
                    "B_decode_values": B_DECODE_VALUES,
                    "L_prefill_values": L_PREFILL_VALUES,
                    "cells": results,
                }, f, indent=2)
    finally:
        print("Stopping server...", flush=True)
        stop_server(proc, log_file)


if __name__ == "__main__":
    asyncio.run(main())
