import asyncio
import json
import random
import time

import aiohttp

random.seed(42)

import input_loader as cfg
from conversation import (
    KvTracker,
    run_conversation_prefiller,
    run_conversation_serial,
    run_conversation_no_disagg_oracle,
    run_conversation_kv_aware_all_disagg,
    run_conversation_adaptive_disagg_oracle,
    run_conversation_adaptive_disagg_decoders,
    run_conversation_adaptive_disagg_decoders_per_turn_kv,
    run_conversation_per_turn_adaptive_disagg_decoders,
)
from request import send_request
from virtual_prefiller import VirtualPrefiller, load_recorded_iter0

N_COMPOUND = 4   # number of distinct compound contexts to pre-warm in adaptive_disagg_oracle

SCHEDULE_DISPATCH = {}


def register(baseline_name):
    def decorator(fn):
        SCHEDULE_DISPATCH[baseline_name] = fn
        return fn
    return decorator


@register("baseline")
async def schedule_conversations_baseline(args):
    """Standalone single-conversation latency: each conversation runs at batch
    size 1 on its assigned engine (no co-tenants, no Poisson burst).

    Convs are shuffled (seeded reproducibly via --order-seed, default 0) and
    dealt round-robin across the engine pool, so each GPU gets a random subset
    of conv_ids — avoids systematic bias where (e.g.) every 4th conv lands on
    the prefiller GPU. Engines are on separate GPUs, so each conv's latency
    reflects truly isolated single-conv processing.
    """
    engines = [(args.prefiller_host[0], args.prefiller_port[0])] + list(zip(args.decoder_host, args.decoder_port))
    engine_count = len(engines)

    seed = getattr(args, 'order_seed', None)
    if seed is None: seed = 0
    rng = random.Random(seed)
    conv_order = list(range(cfg.CONV_COUNT))
    rng.shuffle(conv_order)
    print(f"baseline: shuffled conv→GPU mapping with seed={seed}, "
          f"first 10 conv_ids = {conv_order[:10]}")

    # Group shuffled conv_ids into engine_count parallel slots; each slot is one
    # GPU's serial queue. All slots run in parallel, each at batch=1.
    slots = [[] for _ in range(engine_count)]
    for i, c in enumerate(conv_order):
        slots[i % engine_count].append(c)

    async def run_slot(slot, host, port):
        for c in slot:
            await run_conversation_serial(c, host, port)

    await asyncio.gather(*(run_slot(slots[i], engines[i][0], engines[i][1])
                            for i in range(engine_count)))


@register("no_disagg_oracle")
async def schedule_conversations_no_disagg_oracle(args):
    engine_count = len(args.decoder_host) + 1
    engines = [(args.prefiller_host[0], args.prefiller_port[0])] + list(zip(args.decoder_host, args.decoder_port))
    tracker = KvTracker(engine_count, cfg.KV_BUDGET_TOKENS)
    tasks = []
    if getattr(args, 'arrival_trace', None):
        with open(args.arrival_trace) as f:
            trace = json.load(f)
        trace = sorted(trace, key=lambda e: e['offset_sec'])
        print(f"Arrival-trace replay: {len(trace)} conversations from {args.arrival_trace}")
        t_start = time.time()
        for entry in trace:
            wait = entry['offset_sec'] - (time.time() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(asyncio.create_task(
                run_conversation_no_disagg_oracle(entry['conv_id'], tracker, engines)))
    else:
        for conv_id in range(cfg.CONV_COUNT):
            tasks.append(asyncio.create_task(run_conversation_no_disagg_oracle(conv_id, tracker, engines)))
            await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


@register("all_disagg")
async def schedule_conversations_all_disagg(args):
    host, port = args.proxy_host[0], args.proxy_port[0]
    prefill_tracker = KvTracker(1, cfg.KV_BUDGET_TOKENS)
    decode_tracker = KvTracker(args.num_decoders, cfg.KV_BUDGET_TOKENS)
    tasks = []
    if getattr(args, 'arrival_trace', None):
        with open(args.arrival_trace) as f:
            trace = json.load(f)
        trace = sorted(trace, key=lambda e: e['offset_sec'])
        print(f"Arrival-trace replay: {len(trace)} conversations from {args.arrival_trace}")
        t_start = time.time()
        for entry in trace:
            wait = entry['offset_sec'] - (time.time() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(asyncio.create_task(
                run_conversation_kv_aware_all_disagg(
                    entry['conv_id'], prefill_tracker, decode_tracker, host, port)
            ))
    else:
        for conv_id in range(cfg.CONV_COUNT):
            tasks.append(asyncio.create_task(
                run_conversation_kv_aware_all_disagg(conv_id, prefill_tracker, decode_tracker, host, port)
            ))
            await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


@register("adaptive_disagg_prefiller")
async def schedule_conversations_adaptive_disagg_prefiller(args):
    host, port = args.proxy_host[0], args.proxy_port[0]

    # Build conversation launch order. With --order-seed, shuffle reproducibly
    # using a private RNG so the global Poisson interarrival sequence stays the
    # same across seeds (only the conv_id ↔ arrival-slot assignment changes).
    conv_order = list(range(cfg.CONV_COUNT))
    if getattr(args, 'order_seed', None) is not None:
        rng = random.Random(args.order_seed)
        rng.shuffle(conv_order)
        print(f"Conversation order shuffled with seed={args.order_seed}: "
              f"first 10 = {conv_order[:10]}")

    tasks = []
    for conv_id in conv_order:
        tasks.append(asyncio.create_task(run_conversation_prefiller(conv_id, host, port)))
        await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


@register("adaptive_disagg_decoders")
async def schedule_conversations_adaptive_disagg_decoders(args):
    """Warm decoders with N_COMPOUND prebuilt compound prefixes (file-based, same
    mechanism as adaptive_disagg_oracle), then run full conversations whose iter-0
    is cached prefill + real decode (decoder-side workload of real PD-disagg).
    Each conv is round-robin assigned to one of the compounds and pinned to a
    single decoder for the whole conversation.

    Arrivals: replay --arrival-trace if provided (simulates real prefiller's
    iter-0 completion timings driving the decoder side); otherwise Poisson at RPS.
    """
    decoders = list(zip(args.decoder_host, args.decoder_port))

    # ── Load pre-built compound prompts from file ───────────────────────────
    with open(cfg.COMPOUND_PROMPTS_FILE) as f:
        raw_compounds = json.load(f)
    compound_texts = [raw_compounds[i]['prompt'] for i in range(min(N_COMPOUND, len(raw_compounds)))]
    n_compound = len(compound_texts)
    print(f"Loaded {n_compound} pre-built compound prompts "
          f"({n_compound * len(decoders)} prefill warm-ups across {len(decoders)} decoders)...")

    # Tokenize each compound once so warm-up and per-conv iter-0 send the same
    # token IDs (guaranteed exact prefix match for cache hits).
    if cfg.MOCK_INFERENCE:
        compound_tokens_list = [list(range(raw_compounds[i]['estimated_tokens']))
                                for i in range(n_compound)]
    else:
        compound_tokens_list = []
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            url = f"http://{decoders[0][0]}:{decoders[0][1]}/tokenize"
            for ct in compound_texts:
                async with session.post(url, json={"prompt": ct}) as resp:
                    compound_tokens_list.append((await resp.json())["tokens"])

    # ── Warm up: prefill each compound on every decoder ─────────────────────
    async def warmup_one(host, port, prompt, n_decode, tag):
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            return await send_request(host, port, session, prompt, n_decode,
                                      request_id=f"warmup-{tag}")

    warmup_tasks = []
    for i, ct_tokens in enumerate(compound_tokens_list):
        for h, p in decoders:
            warmup_tasks.append(warmup_one(h, p, ct_tokens, 1, f"{i}-{h}:{p}"))
    await asyncio.gather(*warmup_tasks)
    print(f"Warm-up done: {n_compound} compounds (~{len(compound_tokens_list[0]):,} tokens each) "
          "pinned in prefix cache on every decoder.")

    # Brief settle period — let engines finish post-prefill bookkeeping.
    await asyncio.sleep(5)

    decode_tracker = KvTracker(args.num_decoders, cfg.KV_BUDGET_TOKENS)

    # ── Launch conversations: replay or Poisson ─────────────────────────────
    tasks = []
    if getattr(args, 'arrival_trace', None):
        with open(args.arrival_trace) as f:
            trace = json.load(f)
        trace = sorted(trace, key=lambda e: e['offset_sec'])
        print(f"Arrival-trace replay: {len(trace)} conversations from {args.arrival_trace}")
        t_start = time.time()
        for entry in trace:
            wait = entry['offset_sec'] - (time.time() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            conv_id = entry['conv_id']
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_adaptive_disagg_decoders(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx])
            ))
    else:
        for conv_id in range(cfg.CONV_COUNT):
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_adaptive_disagg_decoders(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx])
            ))
            await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


@register("adaptive_disagg_decoders_per_turn_kv")
async def schedule_conversations_adaptive_disagg_decoders_per_turn_kv(args):
    """Same as adaptive_disagg_decoders but each conv reserves KV per-turn
    (AMPD-style) instead of oracle-peak upfront. Diagnostic baseline used to
    isolate the effect of the KV-reservation strategy on the in-flight
    ramp-up shape."""
    decoders = list(zip(args.decoder_host, args.decoder_port))

    with open(cfg.COMPOUND_PROMPTS_FILE) as f:
        raw_compounds = json.load(f)
    compound_texts = [raw_compounds[i]['prompt'] for i in range(min(N_COMPOUND, len(raw_compounds)))]
    n_compound = len(compound_texts)
    print(f"Loaded {n_compound} pre-built compound prompts "
          f"({n_compound * len(decoders)} prefill warm-ups across {len(decoders)} decoders)...")

    if cfg.MOCK_INFERENCE:
        compound_tokens_list = [list(range(raw_compounds[i]['estimated_tokens']))
                                for i in range(n_compound)]
    else:
        compound_tokens_list = []
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            url = f"http://{decoders[0][0]}:{decoders[0][1]}/tokenize"
            for ct in compound_texts:
                async with session.post(url, json={"prompt": ct}) as resp:
                    compound_tokens_list.append((await resp.json())["tokens"])

    async def warmup_one(host, port, prompt, n_decode, tag):
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            return await send_request(host, port, session, prompt, n_decode,
                                      request_id=f"warmup-{tag}")

    warmup_tasks = []
    for i, ct_tokens in enumerate(compound_tokens_list):
        for h, p in decoders:
            warmup_tasks.append(warmup_one(h, p, ct_tokens, 1, f"{i}-{h}:{p}"))
    await asyncio.gather(*warmup_tasks)
    print(f"Warm-up done: {n_compound} compounds (~{len(compound_tokens_list[0]):,} tokens each) "
          "pinned in prefix cache on every decoder.")

    await asyncio.sleep(5)

    decode_tracker = KvTracker(args.num_decoders, cfg.KV_BUDGET_TOKENS)

    tasks = []
    if getattr(args, 'arrival_trace', None):
        with open(args.arrival_trace) as f:
            trace = json.load(f)
        trace = sorted(trace, key=lambda e: e['offset_sec'])
        print(f"Arrival-trace replay: {len(trace)} conversations from {args.arrival_trace}")
        t_start = time.time()
        for entry in trace:
            wait = entry['offset_sec'] - (time.time() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            conv_id = entry['conv_id']
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_adaptive_disagg_decoders_per_turn_kv(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx])
            ))
    else:
        for conv_id in range(cfg.CONV_COUNT):
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_adaptive_disagg_decoders_per_turn_kv(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx])
            ))
            await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


@register("per_turn_adaptive_disagg_decoders")
async def schedule_conversations_per_turn_adaptive_disagg_decoders(args):
    """AMPD-style baseline. Same warm-up + scheduling as adaptive_disagg_decoders,
    but a fixed-seed fraction `--wrong-pred-pct` of turn-2+ (iter_id >= 1)
    requests are 'wrongly' routed to a virtual prefiller. Two effects:

      Decoder side: that conversation's task pauses for
          queueing_delay + wrong_predict_disagg_wait_ms
      where queueing_delay = time the virtual prefiller needs to drain its
      in-flight iter-0 prefills (computed live from the recorded
      adaptive_disagg_prefiller trace). After the pause the decoder runs
      its real cache-reuse prefill + decode normally.

      Prefiller side (simulated): the wrong-predict blocks the prefiller
      for prefill_hit_ms(lhist+lincr). Any recorded iter-0 whose original
      arrival falls inside that block window gets pushed back. At end of
      run we emit a synthetic prefiller per_step_latency.csv reflecting
      these shifts; the analysis layer reads this instead of the original
      prefiller trace.

    Wrong-predict iter set is picked once at startup with --wrong-pred-seed
    (default 42) so reruns at the same (cfg, rps, pct, seed) are reproducible.
    Candidates are all turn-2+ (iter_id >= 1) -- turn-1 (iter 0) always goes
    through the matching `adaptive_disagg_prefiller` run, captured in the
    recorded trace at --prefiller-trace-dir."""
    decoders = list(zip(args.decoder_host, args.decoder_port))

    # ── Load pre-built compound prompts (same path as adaptive_disagg_decoders) ─
    with open(cfg.COMPOUND_PROMPTS_FILE) as f:
        raw_compounds = json.load(f)
    compound_texts = [raw_compounds[i]['prompt'] for i in range(min(N_COMPOUND, len(raw_compounds)))]
    n_compound = len(compound_texts)
    print(f"Loaded {n_compound} pre-built compound prompts "
          f"({n_compound * len(decoders)} prefill warm-ups across {len(decoders)} decoders)...")

    if cfg.MOCK_INFERENCE:
        compound_tokens_list = [list(range(raw_compounds[i]['estimated_tokens']))
                                for i in range(n_compound)]
    else:
        compound_tokens_list = []
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            url = f"http://{decoders[0][0]}:{decoders[0][1]}/tokenize"
            for ct in compound_texts:
                async with session.post(url, json={"prompt": ct}) as resp:
                    compound_tokens_list.append((await resp.json())["tokens"])

    async def warmup_one(host, port, prompt, n_decode, tag):
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            return await send_request(host, port, session, prompt, n_decode,
                                      request_id=f"warmup-{tag}")

    warmup_tasks = []
    for i, ct_tokens in enumerate(compound_tokens_list):
        for h, p in decoders:
            warmup_tasks.append(warmup_one(h, p, ct_tokens, 1, f"{i}-{h}:{p}"))
    await asyncio.gather(*warmup_tasks)
    print(f"Warm-up done: {n_compound} compounds (~{len(compound_tokens_list[0]):,} tokens each) "
          "pinned in prefix cache on every decoder.")

    await asyncio.sleep(5)

    decode_tracker = KvTracker(args.num_decoders, cfg.KV_BUDGET_TOKENS)

    # ── Pick wrong-predict iters once, deterministically ────────────────────
    wrong_pred_pct = getattr(args, 'wrong_pred_pct', 0.0)
    wrong_pred_seed = getattr(args, 'wrong_pred_seed', 42)
    pred_rng = random.Random(wrong_pred_seed)
    wrong_iters = set()
    n_candidates = 0
    for conv_id in range(cfg.CONV_COUNT):
        for iter_id in range(1, cfg.ITER_COUNT[conv_id]):
            n_candidates += 1
            if pred_rng.random() < wrong_pred_pct:
                wrong_iters.add((conv_id, iter_id))
    print(f"per_turn_adaptive_disagg_decoders: wrong_pred_pct={wrong_pred_pct}, "
          f"seed={wrong_pred_seed}, n_wrong={len(wrong_iters)}/{n_candidates}")

    # ── Spin up the virtual prefiller from the matching recorded trace ──────
    virtual_prefiller = None
    prefiller_trace_dir = getattr(args, 'prefiller_trace_dir', None)
    if prefiller_trace_dir:
        events, prompt_max = load_recorded_iter0(prefiller_trace_dir)
        virtual_prefiller = VirtualPrefiller(events, prompt_max)
        print(f"VirtualPrefiller seeded from {prefiller_trace_dir} "
              f"({len(events)} iter-0 events).")
    elif wrong_pred_pct > 0.0:
        print("WARNING: wrong_pred_pct > 0 but --prefiller-trace-dir not set; "
              "queueing delays cannot be simulated and no synthetic prefiller "
              "trace will be written.")

    tasks = []
    if getattr(args, 'arrival_trace', None):
        with open(args.arrival_trace) as f:
            trace = json.load(f)
        trace = sorted(trace, key=lambda e: e['offset_sec'])
        print(f"Arrival-trace replay: {len(trace)} conversations from {args.arrival_trace}")
        t_start = time.time()
        for entry in trace:
            wait = entry['offset_sec'] - (time.time() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            conv_id = entry['conv_id']
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_per_turn_adaptive_disagg_decoders(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx],
                    wrong_iters, virtual_prefiller)
            ))
    else:
        for conv_id in range(cfg.CONV_COUNT):
            idx = conv_id % n_compound
            tasks.append(asyncio.create_task(
                run_conversation_per_turn_adaptive_disagg_decoders(
                    conv_id, decode_tracker, decoders, compound_tokens_list[idx],
                    wrong_iters, virtual_prefiller)
            ))
            await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)

    # ── Write synthetic prefiller trace for the analysis layer ──────────────
    if virtual_prefiller is not None and virtual_prefiller.t0_wall is not None:
        out_path = f"{args.output_dir}/synthetic_prefiller_per_step_latency.csv"
        virtual_prefiller.write_synthetic_trace(out_path)
        print(f"Wrote synthetic prefiller trace -> {out_path} "
              f"({len(virtual_prefiller.blocks)} wrong-predict blocks applied)")


@register("adaptive_disagg_oracle")
async def schedule_conversations_adaptive_disagg_oracle(args):
    """Pre-warm N_COMPOUND distinct compound contexts on every engine (prefiller +
    decoders), then run each conversation with oracle KV reservation. The prefiller
    is treated as a 4th decoder so we have the same engine pool as no_disagg_oracle
    — this is the fair comparison that isolates the contention-elimination benefit
    of cached iter-0 prefill.
    """
    # If --num-prefillers > 0, the prefiller is included in the engine pool
    # (repurposed as a decoder, warmed with the same compounds). Set --num-prefillers 0
    # to run with only the decoders for an apples-to-apples 3-engine comparison.
    if args.num_prefillers > 0:
        engines = [(args.prefiller_host[0], args.prefiller_port[0])] + list(zip(args.decoder_host, args.decoder_port))
    else:
        engines = list(zip(args.decoder_host, args.decoder_port))

    # ── Load pre-built compound prompts from file ───────────────────────────
    # Each compound is a single text >= 25k tokens, prepared by
    # prepare_compound_prompts.py. We skip the old decode-based warmup entirely
    # and just do one prefill per compound per engine to lock the blocks in the
    # prefix cache.
    with open(cfg.COMPOUND_PROMPTS_FILE) as f:
        raw_compounds = json.load(f)
    compound_texts = [raw_compounds[i]['prompt'] for i in range(min(N_COMPOUND, len(raw_compounds)))]
    n_compound = len(compound_texts)
    n_engines = len(engines)
    print(f"Loaded {n_compound} pre-built compound prompts ({n_compound * n_engines} prefill warm-ups across {n_engines} engines)...")

    # Tokenize each compound once so warm-up and per-conv iter-0 send the same
    # token IDs (guaranteed exact prefix match for cache hits).
    if cfg.MOCK_INFERENCE:
        compound_tokens_list = [list(range(raw_compounds[i]['estimated_tokens']))
                                for i in range(n_compound)]
    else:
        compound_tokens_list = []
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            url = f"http://{engines[0][0]}:{engines[0][1]}/tokenize"
            for ct in compound_texts:
                async with session.post(url, json={"prompt": ct}) as resp:
                    compound_tokens_list.append((await resp.json())["tokens"])

    # ── Single warm-up: prefill the whole compound on every engine ──────────
    # max_tokens=1 means vLLM does prefill across the full compound (indexing all
    # blocks in the prefix cache) and produces one throwaway token. No decode
    # phase, so warm-up is dominated by prefill compute only — much faster than
    # the previous decode-based approach.
    async def warmup_one(host, port, prompt, n_decode, tag):
        async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
            return await send_request(host, port, session, prompt, n_decode,
                                      request_id=f"warmup-{tag}")

    warmup_tasks = []
    for i, ct_tokens in enumerate(compound_tokens_list):
        for h, p in engines:
            warmup_tasks.append(warmup_one(h, p, ct_tokens, 1, f"{i}-{h}:{p}"))
    await asyncio.gather(*warmup_tasks)
    print(f"Warm-up done: {n_compound} compounds (each ~{len(compound_tokens_list[0]):,} tokens) "
          "pinned in prefix cache on every engine.")

    # Brief settle period — let the engines finish any post-prefill bookkeeping
    # (KV index updates, allocator state) before the experiment starts.
    await asyncio.sleep(5)

    # compound_texts physically occupy ~n_compound * max_size tokens of KV per
    # engine. The tracker intentionally does NOT account for them — admission
    # uses each conv's own peak_kv so gating matches no_disagg_oracle.
    decode_tracker = KvTracker(n_engines, cfg.KV_BUDGET_TOKENS)

    # Round-robin assign each conv to a compound context.
    tasks = []
    for conv_id in range(cfg.CONV_COUNT):
        idx = conv_id % n_compound
        tasks.append(asyncio.create_task(
            run_conversation_adaptive_disagg_oracle(
                conv_id, decode_tracker, engines, compound_tokens_list[idx]
            )
        ))
        await asyncio.sleep(random.expovariate(cfg.RPS))
    await asyncio.gather(*tasks)


async def schedule_conversations(args):
    fn = SCHEDULE_DISPATCH.get(args.baseline)
    if fn is None:
        raise ValueError(f"Unknown baseline: {args.baseline}")
    await fn(args)
