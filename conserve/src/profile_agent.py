import asyncio
import aiohttp
import json
import time
import os
import subprocess
import signal
import csv
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())


# =============================
# CONFIG
# =============================
# PROMPTS_FILE = f"{REPO_ROOT}/conserve/input/prompts_40x5.json"
# LENGTHS_FILE = f"{REPO_ROOT}/conserve/input/prompt_lengths_40x5.json"
PROMPTS_FILE = f"{REPO_ROOT}/conserve/input/mini_swe_agent_trace.json"
OUT_DCGMI_FILE = "dcgmi_trace.tsv"
LATENCY_LOG_FILE = "per_step_latency.csv"

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ENGINE = "http://127.0.0.1:9101"  # single engine

RPS = 1.0  # conversations per second
REQUEST_INTERVAL = 1.0 / RPS
TIMEOUT = aiohttp.ClientTimeout(total=3600)

# KV cache budget per engine in tokens.
# Default: 19278 blocks × 16 tokens/block (measured from decoder logs at 0.9 GPU util).
# Set via --kv-budget-tokens to match your actual hardware.
KV_BUDGET_TOKENS = 308_448


class KvTracker:
    """Tracks KV-cache token usage per engine and routes new conversations to the
    engine with the most headroom that can still fit the request."""

    def __init__(self, num_engines: int, budget_per_engine: int):
        self._budget = budget_per_engine
        self._used = [0] * num_engines
        self._lock = asyncio.Lock()

    async def best_engine_for(self, needed: int) -> int:
        """Block until some engine has >= needed tokens free, then atomically reserve them."""
        while True:
            async with self._lock:
                free = [self._budget - u for u in self._used]
                candidates = [i for i, f in enumerate(free) if f >= needed]
                if candidates:
                    chosen = max(candidates, key=lambda i: free[i])
                    self._used[chosen] += needed
                    return chosen
            await asyncio.sleep(0.05)

    async def wait_for_engine(self, engine_idx: int, needed: int):
        """Block until a specific engine has >= needed tokens free, then reserve."""
        while True:
            async with self._lock:
                if self._budget - self._used[engine_idx] >= needed:
                    self._used[engine_idx] += needed
                    return
            await asyncio.sleep(0.05)

    async def add(self, engine_idx: int, delta: int):
        async with self._lock:
            self._used[engine_idx] += delta

    async def release(self, engine_idx: int, tokens: int):
        async with self._lock:
            self._used[engine_idx] = max(0, self._used[engine_idx] - tokens)

DCGMI_CMD = [
    "bash", "-c",
    "dcgmi dmon -e 155,156,157,1130,1131,1132,1133,150,140,151,152,153,158,159,1110,1111,1112,858,100,101,102,110,111,1120,203,204,206,207,1100,1101,1102,1103,1104 -d 1 | ts '%Y-%m-%dT%H:%M:%.S' >> "
]
REQUEST_COUNT = 256

# =============================
# LOAD PROMPTS AND LENGTHS
# =============================
with open(PROMPTS_FILE, "r") as f:
    raw_data = json.load(f)
PROMPT_DATA = {
    (entry['conv_id'], entry['iter_id']): {
        'in_token_size': entry['in_token_size'],
        'out_token_size': entry['out_token_size'],
        'prompt': entry['prompt']
    } for entry in raw_data if entry['conv_id'] < REQUEST_COUNT
}
CONV_COUNT = max(i for i, _ in PROMPT_DATA.keys())
ITER_COUNT = [0] * (CONV_COUNT + 1)
for i, _ in PROMPT_DATA.keys():
    ITER_COUNT[i] += 1


def parse_args():
    import argparse
    def csv_ints(s):
        return [int(x) for x in s.split(",")]
    def csv_strs(s):
        return [x.strip() for x in s.split(",")]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[7100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[7200])
    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--proxy-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--proxy-port", type=csv_ints, default=[9101])
    parser.add_argument("--baseline", type=str, default="no_batching", choices=["no_batching", "no_disagg", "all_disagg","adaptive_disagg_first","adaptive_disagg_rest"])
    parser.add_argument("--disagg-first-log-file", type=str, default=None, help="File for disagg first log")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use")
    parser.add_argument("--kv-budget-tokens", type=int, default=None,
                        help="KV cache budget per engine in tokens (default: KV_BUDGET_TOKENS constant)")

    args = parser.parse_args()
    assert args.baseline in ["no_batching", "no_disagg", "all_disagg","adaptive_disagg_first","adaptive_disagg_rest"]
    assert args.num_decoders == len(args.decoder_host)
    assert args.num_decoders == len(args.decoder_port)
    global MODEL, KV_BUDGET_TOKENS
    MODEL = args.model
    if args.kv_budget_tokens is not None:
        KV_BUDGET_TOKENS = args.kv_budget_tokens
    return args

def prep_outputs(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Update file paths to use the output directory
    global OUT_DCGMI_FILE, LATENCY_LOG_FILE
    OUT_DCGMI_FILE = os.path.join(output_dir, OUT_DCGMI_FILE)
    LATENCY_LOG_FILE = os.path.join(output_dir, LATENCY_LOG_FILE)

    # Rewrite dcgmi log file
    if os.path.exists(OUT_DCGMI_FILE):
        os.remove(OUT_DCGMI_FILE)

    if os.path.exists(LATENCY_LOG_FILE):
        os.remove(LATENCY_LOG_FILE)

    # Initialize the CSV with header
    with open(LATENCY_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conv_id", "step_id", "prompt_tokens", "max_tokens", "start_time", "end_time", "latency_sec"])
    print(f"Writing conversation logs to: {LATENCY_LOG_FILE}")
    return output_dir

def start_dcgmi():
    print("Starting GPU monitoring...")
    dcgmi_cmd = DCGMI_CMD.copy()
    dcgmi_cmd[-1] += str(OUT_DCGMI_FILE)
    dcgmi_proc = subprocess.Popen(
        dcgmi_cmd,
        preexec_fn=os.setsid
    )
    time.sleep(5)
    print("GPU monitoring started")
    print(f"Writing GPU monitoring logs to: {OUT_DCGMI_FILE}")
    return dcgmi_proc

def stop_dcgmi(dcgmi_proc):
    print("Stopping GPU monitoring...")
    try:
        os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGTERM)
        dcgmi_proc.wait(timeout=5)
        print("GPU monitoring stopped")
    except Exception as e:
        print(f"Failed to terminate dcgmi process group: {e}")
        try:
            os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGKILL)
        except Exception as e2:
            print(f"Failed to kill dcgmi process group: {e2}")

def log_step_latency(conv_id, step_id, prompt_tokens, max_tokens, start_time, end_time, latency):
    with open(LATENCY_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([conv_id, step_id, prompt_tokens, max_tokens, start_time, end_time, f"{latency:.4f}"])
    print(f"[conv {conv_id} step {step_id}] latency logged: {latency:.2f}s")


# =============================
# SEND ONE REQUEST
# =============================
async def send_request(engine_host, engine_port, session, prompt, max_tokens, request_id: str =None):
    url = f"http://{engine_host}:{engine_port}/v1/completions"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        'logit_bias': {
            2: -100,
            13: -100,
            128001: -100,
            128009: -100
        },
    }
    if request_id is not None:
        payload["request_id"] = request_id

    # print(f"Posting {payload} to {url}", flush=True)

    async with session.post(url, json=payload) as resp:
        resp_text = await resp.text()
        resp_json = json.loads(resp_text)
        # print(resp_json, flush=True)
        output_text = resp_json["choices"][0]["text"]
        completion_tokens = resp_json.get("usage", {}).get("completion_tokens", max_tokens)

    return output_text, completion_tokens

# =============================
# RUN ONE CONVERSATION
# =============================
async def run_conversation_first(conv_id, host, port, max_tokens=2, logging=True):
    conversation_history = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        # Only first inference
        iter_id = 0
        data = PROMPT_DATA[(conv_id, iter_id)]
        prompt, prompt_tokens = (
            data['prompt'],
            data['in_token_size'],
        )
        conversation_history.append(prompt)

        print(f"[conv {conv_id} iter {iter_id}] -> {ENGINE} (max_tokens={max_tokens})")
        t0 = time.time()
        output_text, _ = await send_request(host, port, session, prompt, max_tokens, request_id=f"{conv_id}-{iter_id}")
        conversation_history.append(output_text)
        t1 = time.time()

        # Log latency
        if logging:
            log_step_latency(conv_id, iter_id, prompt_tokens, max_tokens, t0, t1, t1 - t0)

    return conversation_history

async def run_conversation_serial(conv_id, host, port, conversation_history=None, start_iter_id = 0):
    conversation_history = [] if conversation_history is None else conversation_history
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for iter_id in range(start_iter_id, ITER_COUNT[conv_id]):
            data = PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'],
                data['in_token_size'],
                data['out_token_size'],
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (input_tokens={prompt_tokens}, max_tokens={out_tokens}, history_coung={len(conversation_history)})")
            t0 = time.time()
            output_text, _ = await send_request(host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)


async def run_conversation_kv_aware(conv_id, tracker: KvTracker, engines):
    needed = PROMPT_DATA[(conv_id, 0)]['in_token_size']
    engine_idx = await tracker.best_engine_for(needed)
    host, port = engines[engine_idx]

    kv_used = needed
    conversation_history = []

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for iter_id in range(ITER_COUNT[conv_id]):
            data = PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = data['prompt'], data['in_token_size'], data['out_token_size']
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (input_tokens={prompt_tokens}, max_tokens={out_tokens}, engine={engine_idx})")
            t0 = time.time()
            output_text, completion_tokens = await send_request(
                host, port, session, full_prompt, out_tokens,
                request_id=f"{conv_id}-{iter_id}"
            )
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

            # KV grows by new prompt tokens + generated tokens each iter.
            # iter 0 already reserved prompt_tokens[0]; only completion_tokens is new.
            added = completion_tokens if iter_id == 0 else prompt_tokens + completion_tokens
            await tracker.add(engine_idx, added)
            kv_used += added

    await tracker.release(engine_idx, kv_used)


async def run_conversation_kv_aware_all_disagg(
    conv_id, prefill_tracker: KvTracker, decode_tracker: KvTracker, host, port
):
    """KV-aware runner for all_disagg mode.

    Two separate budgets:
    - prefill_tracker (1 engine): the prefiller holds the full prompt in KV during prefill,
      then transfers it to the decoder. We conservatively hold the reservation until the
      response arrives since we cannot detect when prefill ends in non-streaming mode.
    - decode_tracker (num_decoders engines): the decoder holds prompt + output tokens for
      the entire decode phase. We pick the decoder with the most headroom at each iteration.

    Since all_disagg has no prefix cache, the full accumulated prompt is re-prefilled at
    every iteration. accum_prompt tracks that growing size so we can reserve accurately.

    Reservation lifecycle per iteration:
      1. Release the previous iteration's reservations (both prefill and decode).
      2. Wait for a decoder with enough headroom (the bottleneck), then the prefiller.
      3. Send the request; after it returns, update accum_prompt with actual output tokens.
    """
    conversation_history = []
    accum_prompt = 0      # running total tokens in conversation context
    kv_prefill   = 0      # tokens currently reserved on prefill_tracker
    kv_decode    = 0      # tokens currently reserved on decode_tracker
    decode_eng   = None

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for iter_id in range(ITER_COUNT[conv_id]):
            data = PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            # No prefix cache: prefiller receives the full accumulated prompt this iter.
            accum_prompt   += prompt_tokens
            needed_prefill  = accum_prompt
            needed_decode   = accum_prompt + out_tokens

            # Release previous iter's reservations before re-acquiring for new sizes.
            if kv_prefill > 0:
                await prefill_tracker.release(0, kv_prefill)
            if kv_decode > 0 and decode_eng is not None:
                await decode_tracker.release(decode_eng, kv_decode)

            # Acquire decoder first (the real bottleneck), then prefiller.
            decode_eng = await decode_tracker.best_engine_for(needed_decode)
            await prefill_tracker.wait_for_engine(0, needed_prefill)
            kv_prefill = needed_prefill
            kv_decode  = needed_decode

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} "
                  f"(accum_prompt={accum_prompt}, out={out_tokens}, decode_eng={decode_eng})")
            t0 = time.time()
            output_text, completion_tokens = await send_request(
                host, port, session, full_prompt, out_tokens,
                request_id=f"{conv_id}-{iter_id}"
            )
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

            # Model output becomes part of the next iteration's prompt context.
            accum_prompt += completion_tokens

    # Release the final iteration's reservations.
    if kv_prefill > 0:
        await prefill_tracker.release(0, kv_prefill)
    if kv_decode > 0 and decode_eng is not None:
        await decode_tracker.release(decode_eng, kv_decode)


async def run_conversation_adaptive_disagg(
    conv_id, decode_tracker: KvTracker, decoders, compound_text
):
    """KV-aware runner for adaptive_disagg_rest mode.

    All decoders share a pre-warmed compound_text prefix in their KV cache.
    Each conversation starts from iter_id=1, using compound_text as the base
    history entry. compound_token_count is pre-consumed in decode_tracker so
    only the additional tokens per conversation are tracked here.

    The conversation stays pinned to one decoder across all iterations to keep
    the prefix cache valid.
    """
    if ITER_COUNT[conv_id] <= 1:
        return

    pt_1 = PROMPT_DATA[(conv_id, 1)]['in_token_size']
    decoder_idx = await decode_tracker.best_engine_for(pt_1)
    host, port = decoders[decoder_idx]
    kv_extra = pt_1

    conversation_history = [compound_text]

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        for iter_id in range(1, ITER_COUNT[conv_id]):
            data = PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (decoder={decoder_idx}, kv_extra={kv_extra})")
            t0 = time.time()
            output_text, completion_tokens = await send_request(
                host, port, session, full_prompt, out_tokens,
                request_id=f"{conv_id}-{iter_id}"
            )
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

            # iter_id == 1: pt_1 was reserved at start; only completion_tokens is new.
            # iter_id > 1: nothing was pre-reserved; add full prompt + completion.
            added = completion_tokens if iter_id == 1 else prompt_tokens + completion_tokens
            await decode_tracker.add(decoder_idx, added)
            kv_extra += added

    await decode_tracker.release(decoder_idx, kv_extra)


# =============================
# SCHEDULE CONVERSATIONS WITH DELAY
# =============================
SCHEDULE_DISPATCH = {}

def register(baseline_name):
    def decorator(fn):
        SCHEDULE_DISPATCH[baseline_name] = fn
        return fn  # IMPORTANT: return the function unchanged
    return decorator

@register("no_batching")
async def schedule_conversations_no_batching(args):
    engine_count = len(args.decoder_host) + 1
    for conv_id in range(0, CONV_COUNT, engine_count):
        tasks = [asyncio.create_task(run_conversation_serial(conv_id, args.prefiller_host[0], args.prefiller_port[0]))]
        for i in range(args.num_decoders):
            tasks.append(asyncio.create_task(run_conversation_serial(conv_id+1+i, args.decoder_host[i], args.decoder_port[i])))
        await asyncio.gather(*tasks)

@register("no_disagg")
async def schedule_conversations_no_disagg(args):
    engine_count = len(args.decoder_host) + 1
    engines = [(args.prefiller_host[0], args.prefiller_port[0])] + list(zip(args.decoder_host, args.decoder_port))
    tracker = KvTracker(engine_count, KV_BUDGET_TOKENS)
    tasks = [
        asyncio.create_task(run_conversation_kv_aware(conv_id, tracker, engines))
        for conv_id in range(CONV_COUNT)
    ]
    await asyncio.gather(*tasks)

@register("all_disagg")
async def schedule_conversations_all_disagg(args):
    host, port = args.proxy_host[0], args.proxy_port[0]
    prefill_tracker = KvTracker(1, KV_BUDGET_TOKENS)
    decode_tracker  = KvTracker(args.num_decoders, KV_BUDGET_TOKENS)
    tasks = [
        asyncio.create_task(
            run_conversation_kv_aware_all_disagg(conv_id, prefill_tracker, decode_tracker, host, port)
        )
        for conv_id in range(CONV_COUNT)
    ]
    await asyncio.gather(*tasks)
    
@register("adaptive_disagg_first")
async def schedule_conversations_adaptive_disagg_first(args):
    tasks = []
    host, port = args.proxy_host[0], args.proxy_port[0]
    for conv_id in range(CONV_COUNT):
        tasks.append(asyncio.create_task(run_conversation_first(conv_id, host, port)))
        await asyncio.sleep(REQUEST_INTERVAL)
    await asyncio.gather(*tasks)

@register("adaptive_disagg_rest")
async def schedule_conversations_adaptive_disagg_rest(args):
    """Warm up all decoders with a shared compound_text prefix, then run
    conversations from iter_id=1 with KV-aware scheduling.

    Warm-up:
      - prompt  = iter-0 prompt with the fewest tokens (min_size)
      - max_tokens = max_size - min_size so compound_text totals max_size tokens
      - temperature=0 guarantees identical output on all decoders → the prefix
        is permanently cached and never evicted by competing requests

    decode_tracker is pre-loaded with compound_token_count on each decoder so
    per-conversation reservations only account for additional tokens.
    """
    decoders = list(zip(args.decoder_host, args.decoder_port))

    # ── Compute iter-0 prompt size range ────────────────────────────────────
    iter0_entries = [
        (conv_id, PROMPT_DATA[(conv_id, 0)])
        for conv_id in range(CONV_COUNT)
        if (conv_id, 0) in PROMPT_DATA
    ]
    _, min_entry = min(iter0_entries, key=lambda x: x[1]['in_token_size'])
    min_size = min_entry['in_token_size']
    max_size = max(d['in_token_size'] for _, d in iter0_entries)
    warmup_max_tokens = max_size - min_size
    warmup_prompt = min_entry['prompt']

    print(f"Warm-up: min_size={min_size}, max_size={max_size}, warmup_max_tokens={warmup_max_tokens}")

    # ── Warm up all decoders simultaneously ─────────────────────────────────
    async def warmup_one(host, port):
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            return await send_request(
                host, port, session, warmup_prompt, warmup_max_tokens,
                request_id="warmup"
            )

    warmup_results = await asyncio.gather(*(warmup_one(h, p) for h, p in decoders))

    warmup_output, warmup_completion = warmup_results[0]
    compound_text = warmup_prompt + warmup_output   # no separator — tokens are contiguous
    compound_token_count = min_size + warmup_completion
    print(f"Warm-up done: compound_token_count={compound_token_count} (warmup_completion={warmup_completion})")

    # ── Pre-consume compound_token_count in the tracker for each decoder ────
    decode_tracker = KvTracker(args.num_decoders, KV_BUDGET_TOKENS)
    for i in range(args.num_decoders):
        await decode_tracker.add(i, compound_token_count)

    # ── Launch all conversations from iter_id=1 ─────────────────────────────
    tasks = [
        asyncio.create_task(
            run_conversation_adaptive_disagg(conv_id, decode_tracker, decoders, compound_text)
        )
        for conv_id in range(CONV_COUNT)
    ]
    await asyncio.gather(*tasks)


async def schedule_conversations(args):
    fn = SCHEDULE_DISPATCH.get(args.baseline)
    if fn is None:
        raise ValueError(f"Invalid baseline: {args.baseline}")
    await fn(args)

# =============================
# MAIN
# =============================
async def main():
    
    args = parse_args()
    prep_outputs(args)

    # Start dcgmi monitoring
    dcgmi_proc = start_dcgmi()
    try:
        await schedule_conversations(args)
    finally:
        stop_dcgmi(dcgmi_proc)

if __name__ == "__main__":
    asyncio.run(main())
