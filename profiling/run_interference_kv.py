"""
Follow-up to run_interference.py: instead of varying B with tiny L_decoder=8,
fix B=8 and sweep L_decoder ∈ {1024, 4096, 16384, 32768} to characterize how
the active KV cache of concurrent decoders affects the prefill-step duration.

For each cell (L_decoder, L_prefill, cache_hit):
  1. Submit 8 decoders with a shared natural-language prompt of length
     L_decoder; prefix caching makes the first one prefill (slow) and the
     other 7 cache-hit (fast). They then decode in lockstep with each request
     carrying L_decoder + ~steady-state tokens of KV.
  2. Wait ~max(2s, prefill_time*1.5) so all decoders are in steady-state decode.
  3. Submit 1 prefill request of length L_prefill with cell-unique content
     (miss) or pre-warmed (hit).
  4. Measure the engine-step duration of the step that executes the prefill.

Outputs:
  interference_kv_data/server_engine.jsonl
  interference_kv_data/server_core.jsonl
  interference_kv_data/cells.json
  interference_kv_data/server.log
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import time
import sys
import uuid
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
from config import MODEL_DIR, PROFILING_DATA_DIR, MODEL


import aiohttp

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # local override

# Shard-aware: each shard runs a strided subset of the replicates on its own
# GPU/port/output dir, so the 3 free GPUs can run in parallel. Merge afterward.
_ap = argparse.ArgumentParser()
_ap.add_argument("--n-shards", type=int, default=1)
_ap.add_argument("--shard-id", type=int, default=0)
_ap.add_argument("--port", type=int, default=7700)
_ap.add_argument("--out", type=str,
                 default=f"{REPO_ROOT}/paper/figures/section3/output/300W/interference_kv_data")
_args = _ap.parse_args()
PORT = _args.port
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

B_DECODE = 8
L_DECODER_VALUES = [1024, 4096, 16384, 32768]
L_PREFILL_VALUES = [1024, 4096, 16384, 32768]
PREFILL_MAX_TOKENS = 2
DECODE_MAX_TOKENS = 200  # long enough that decoders are still alive when we submit prefill
N_REPLICATES = 64        # repeat each (L_decoder, L_prefill, cache) cell this many times


def setup_wait_seconds(L_decoder: int) -> float:
    """How long to wait after submitting decoders so they're in steady-state decode."""
    # First decoder must prefill L_decoder tokens. Rough cost from prefill_linearity:
    # ~25 µs/tok for L<=8k, super-linear above. Cap conservatively.
    table = {1024: 1.5, 4096: 1.8, 16384: 2.5, 32768: 5.0}
    return table.get(L_decoder, 2.0)


_PROMPTS_CACHE: dict = {}

def load_prompts(L: int):
    if L not in _PROMPTS_CACHE:
        p = Path(f"{PROFILING_DATA_DIR}/prompts_{L}x2048.json")
        with open(p) as f:
            _PROMPTS_CACHE[L] = json.load(f)
    return _PROMPTS_CACHE[L]


def get_prompt(salt: int, L: int) -> str:
    """Pick a unique natural-language prompt of length ~L tokens by salt."""
    prompts = load_prompts(L)
    return prompts[salt % len(prompts)]["prompt"]


def start_server():
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "localhost",
        "--port", str(PORT),
        "--download-dir", MODEL_DIR,
        "--rope-scaling", '{"rope_type":"dynamic","factor":2.0}',
        "--max-num-batched-tokens", "33792",
        "--max-num-seqs", "32",
        "--disable-log-requests",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--engine-log-file", str(OUT / "server_engine.jsonl"),
        "--core-log-file", str(OUT / "server_core.jsonl"),
        "--gpu-memory-utilization", "0.9",
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


async def submit(session, request_id, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.2,
        "top_p": 1.0,
        "stream": False,
        "logit_bias": {"151643": -100, "151644": -100, "151645": -100},
        "request_id": request_id,
    }
    url = f"http://localhost:{PORT}/v1/completions"
    t_submit = time.time()
    async with session.post(url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=600)) as r:
        await r.read()
    return request_id, t_submit, time.time()


async def warm_cache(session, prompt):
    await submit(session, f"warm_{uuid.uuid4().hex[:8]}", prompt, PREFILL_MAX_TOKENS)


async def run_cell(session, cell_idx, L_decoder, L_prefill, cache_hit):
    # Shared decoder prompt (so prefix caching makes 7/8 prefills near-free).
    decoder_prompt = get_prompt(60_000 + cell_idx, L_decoder)
    decoder_rids = [f"c{cell_idx:03d}_dec{i}" for i in range(B_DECODE)]
    prefill_rid = f"c{cell_idx:03d}_pref"
    prefill_prompt = get_prompt(70_000 + cell_idx, L_prefill)

    if cache_hit:
        await warm_cache(session, prefill_prompt)

    # Submit decoders
    decoder_tasks = [
        asyncio.create_task(submit(session, rid, decoder_prompt, DECODE_MAX_TOKENS))
        for rid in decoder_rids
    ]
    # Wait long enough for them to reach steady-state decode
    await asyncio.sleep(setup_wait_seconds(L_decoder))

    # Submit prefill
    t_prefill_submit = time.time()
    prefill_task = asyncio.create_task(
        submit(session, prefill_rid, prefill_prompt, PREFILL_MAX_TOKENS)
    )
    _, _, t_prefill_done = await prefill_task

    # Let decoders finish so queue clears
    await asyncio.gather(*decoder_tasks, return_exceptions=True)

    return {
        "cell_idx": cell_idx,
        "B_decode": B_DECODE,
        "L_decoder": L_decoder,
        "L_prefill": L_prefill,
        "cache_hit": cache_hit,
        "decoder_rids": decoder_rids,
        "prefill_rid": prefill_rid,
        "t_prefill_submit": t_prefill_submit,
        "t_prefill_done": t_prefill_done,
    }


async def main():
    print("Starting server...", flush=True)
    proc, log_file = start_server()
    try:
        await wait_for_server(timeout_s=600)
        print("Server up. Running cells...", flush=True)
        async with aiohttp.ClientSession() as session:
            # cell_idx = rep*CELLS_PER_REP + local keeps it globally unique
            # across shards, so per-cell prompt salts never collide.
            CELLS_PER_REP = len(L_DECODER_VALUES) * len(L_PREFILL_VALUES) * 2
            my_reps = list(range(_args.shard_id, N_REPLICATES, _args.n_shards))
            print(f"shard {_args.shard_id}/{_args.n_shards}: reps {my_reps}", flush=True)
            results = []
            t0 = time.time()
            for rep in my_reps:
                local = 0
                for cache_hit in [False, True]:
                    for L_prefill in L_PREFILL_VALUES:
                        for L_decoder in L_DECODER_VALUES:
                            cell_idx = rep * CELLS_PER_REP + local
                            t_cell = time.time()
                            meta = await run_cell(session, cell_idx, L_decoder,
                                                  L_prefill, cache_hit)
                            meta["rep"] = rep
                            cond = "hit" if cache_hit else "miss"
                            print(f"  rep {rep:>2} cell {cell_idx:>5} {cond:>4} "
                                  f"L_dec={L_decoder:>5} L_pref={L_prefill:>5}  "
                                  f"{time.time()-t_cell:.1f}s", flush=True)
                            results.append(meta)
                            local += 1
            print(f"\nAll {len(results)} cells in {time.time()-t0:.1f}s")
            with open(OUT / "cells.json", "w") as f:
                json.dump({
                    "model": MODEL,
                    "B_decode": B_DECODE,
                    "n_replicates": N_REPLICATES,
                    "L_decoder_values": L_DECODER_VALUES,
                    "L_prefill_values": L_PREFILL_VALUES,
                    "prefill_max_tokens": PREFILL_MAX_TOKENS,
                    "decode_max_tokens": DECODE_MAX_TOKENS,
                    "cells": results,
                }, f, indent=2)
    finally:
        print("Stopping server...", flush=True)
        stop_server(proc, log_file)


if __name__ == "__main__":
    asyncio.run(main())
