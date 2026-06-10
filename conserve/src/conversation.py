import asyncio
import time

import aiohttp

import input_loader as cfg
from per_turn_cost_model import wrong_predict_disagg_wait_ms
from request import send_request, log_step_latency


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


async def run_conversation_prefiller(conv_id, host, port, max_tokens=2, logging=True):
    conversation_history = []
    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        iter_id = 0
        data = cfg.PROMPT_DATA[(conv_id, iter_id)]
        prompt, prompt_tokens = data['prompt'], data['in_token_size']
        conversation_history.append(prompt)

        print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (max_tokens={max_tokens})")
        t0 = time.time()
        output_text, _ = await send_request(host, port, session, prompt, max_tokens,
                                            request_id=f"{conv_id}-{iter_id}")
        conversation_history.append(output_text)
        t1 = time.time()

        if logging:
            log_step_latency(conv_id, iter_id, prompt_tokens, max_tokens, t0, t1, t1 - t0)

    return conversation_history


async def run_conversation_serial(conv_id, host, port, conversation_history=None, start_iter_id=0):
    conversation_history = [] if conversation_history is None else conversation_history
    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        for iter_id in range(start_iter_id, cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} "
                  f"(input_tokens={prompt_tokens}, max_tokens={out_tokens}, "
                  f"history_count={len(conversation_history)})")
            t0 = time.time()
            output_text, _ = await send_request(host, port, session, full_prompt, out_tokens,
                                                request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)


async def run_conversation_no_disagg_oracle(conv_id, tracker: KvTracker, engines):
    # Oracle: reserve the full conversation lifecycle upfront.
    # Only admit when there is budget for every iteration's prompt + max output.
    # This is not realistic (future iterations are unknown at admission time) but
    # gives a clean upper-bound baseline free of mid-flight preemptions.
    # Each iter's in_token_size is the NEW user turn (incremental, not the
    # accumulated prompt), so total KV = sum(new in + out) over all iters.
    peak_kv = sum(
        cfg.PROMPT_DATA[(conv_id, i)]['in_token_size']
        + cfg.PROMPT_DATA[(conv_id, i)]['out_token_size']
        for i in range(cfg.ITER_COUNT[conv_id])
    )
    engine_idx = await tracker.best_engine_for(peak_kv)
    host, port = engines[engine_idx]
    conversation_history = []

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        for iter_id in range(cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = data['prompt'], data['in_token_size'], data['out_token_size']
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} "
                  f"(input_tokens={prompt_tokens}, max_tokens={out_tokens}, engine={engine_idx})")
            t0 = time.time()
            output_text, completion_tokens = await send_request(
                host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

    await tracker.release(engine_idx, peak_kv)


async def run_conversation_kv_aware_all_disagg(
    conv_id, prefill_tracker: KvTracker, decode_tracker: KvTracker, host, port
):
    """KV-aware runner for all_disagg mode.

    Every iteration is PD-disaggregated with no KV reuse across iterations.
    Reserve before inference, release immediately after — nothing carries over.
    accum_prompt grows each iteration because the full history is re-sent.
    """
    conversation_history = []
    accum_prompt = 0

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        for iter_id in range(cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            accum_prompt += prompt_tokens
            needed_prefill = accum_prompt
            needed_decode = accum_prompt + out_tokens

            decode_eng = await decode_tracker.best_engine_for(needed_decode)
            await prefill_tracker.wait_for_engine(0, needed_prefill)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} "
                  f"(accum_prompt={accum_prompt}, out={out_tokens}, decode_eng={decode_eng})")
            t0 = time.time()
            output_text, completion_tokens = await send_request(
                host, port, session, full_prompt, out_tokens,
                request_id=f"{conv_id}-{iter_id}", decoder_index=decode_eng)
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

            await decode_tracker.release(decode_eng, needed_decode)
            await prefill_tracker.release(0, needed_prefill)

            accum_prompt += completion_tokens


async def run_conversation_adaptive_disagg_oracle(
    conv_id, decode_tracker: KvTracker, decoders, compound_tokens
):
    """Oracle variant of adaptive_disagg.

    Reserves the full expected KV upfront, mirroring no_disagg_oracle's accounting:
      peak_kv = sum(in_token_size + out_token_size) over all iters
    Each iter's in_token_size is the NEW user turn fragment (incremental), so
    the total is the union of all fresh tokens added across the conversation.

    Iter-0 simulates PD-disagg's split. Each decoder is pre-warmed with a single
    compound_text spanning max_size tokens. For each conversation, iter-0 sends
    `compound_tokens[:in_token_size_0]` as the prompt (a true token-level prefix
    of the cached pool, so prefill is a guaranteed cache hit — free, like prefill
    happening on a dedicated prefiller) and decodes `out_token_size_0` tokens for
    real (mirroring real PD-disagg's decoder workload). With temperature=0 the
    output is deterministic and matches the corresponding slice of compound_text,
    so the cached prefix remains valid for iter 1+.
    """
    if cfg.ITER_COUNT[conv_id] < 1:
        return

    peak_kv = sum(
        cfg.PROMPT_DATA[(conv_id, i)]['in_token_size']
        + cfg.PROMPT_DATA[(conv_id, i)]['out_token_size']
        for i in range(cfg.ITER_COUNT[conv_id])
    )
    decoder_idx = await decode_tracker.best_engine_for(peak_kv)
    host, port = decoders[decoder_idx]

    conversation_history = []

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        # ── iter-0: cached-prefill + real-decode (PD-disagg simulation) ─────
        data0 = cfg.PROMPT_DATA[(conv_id, 0)]
        n_in  = data0['in_token_size']
        n_out = data0['out_token_size']
        iter0_prompt_tokens = compound_tokens[:n_in]   # token-ID slice of the pool

        print(f"[conv {conv_id} iter 0] -> {host}:{port} (decoder={decoder_idx}, "
              f"cached prefill of {n_in} tokens, decode {n_out})")
        t0 = time.time()
        # echo=True returns "input_text + output_text" so conversation_history reflects
        # exactly what this conv processed (n_in + n_out tokens), not the full compound.
        echo_text, _ = await send_request(
            host, port, session, iter0_prompt_tokens, n_out,
            request_id=f"{conv_id}-0", echo=True)
        t1 = time.time()
        log_step_latency(conv_id, 0, n_in, n_out, t0, t1, t1 - t0)

        # echo_text is the detokenized input + the output, matching the n_in + n_out
        # tokens this conv actually processed. iter 1+ prefills will hit the prefix
        # cache for the n_in tokens (compound prefix) and re-prefill the small output
        # portion at most.
        conversation_history.append(echo_text)

        # ── iter 1..N: normal flow on the same decoder ──────────────────────
        for iter_id in range(1, cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (decoder={decoder_idx})")
            t0 = time.time()
            output_text, _ = await send_request(
                host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

    await decode_tracker.release(decoder_idx, peak_kv)


async def run_conversation_adaptive_disagg_decoders(
    conv_id, decode_tracker: KvTracker, decoders, compound_tokens
):
    """KV-aware runner for adaptive_disagg_decoders mode.

    Simulates the decoder-side workload of real adaptive_disagg PD-disagg:
      - iter-0: prefill is "free" (compound prefix → cache hit, mimicking the
                prefiller transferring KV) + real decode of n_out tokens
      - iter-1+: real prefill of accumulated history + real decode

    All decoders share a pre-warmed compound prefix. The conversation stays
    pinned to one decoder so the prefix cache remains valid across iters.
    Reserves peak_kv upfront (oracle-style) for clean accounting.
    """
    if cfg.ITER_COUNT[conv_id] < 1:
        return

    peak_kv = sum(
        cfg.PROMPT_DATA[(conv_id, i)]['in_token_size']
        + cfg.PROMPT_DATA[(conv_id, i)]['out_token_size']
        for i in range(cfg.ITER_COUNT[conv_id])
    )
    decoder_idx = await decode_tracker.best_engine_for(peak_kv)
    host, port = decoders[decoder_idx]

    conversation_history = []

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        # ── iter-0: cached-prefill + real-decode (decoder-side workload) ────
        data0 = cfg.PROMPT_DATA[(conv_id, 0)]
        n_in  = data0['in_token_size']
        n_out = data0['out_token_size']
        iter0_prompt_tokens = compound_tokens[:n_in]

        print(f"[conv {conv_id} iter 0] -> {host}:{port} (decoder={decoder_idx}, "
              f"cached prefill of {n_in} tokens, decode {n_out})")
        t0 = time.time()
        echo_text, _ = await send_request(
            host, port, session, iter0_prompt_tokens, n_out,
            request_id=f"{conv_id}-0", echo=True)
        t1 = time.time()
        log_step_latency(conv_id, 0, n_in, n_out, t0, t1, t1 - t0)
        conversation_history.append(echo_text)

        # ── iter 1..N: normal prefill + decode on the same decoder ──────────
        for iter_id in range(1, cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (decoder={decoder_idx})")
            t0 = time.time()
            output_text, _ = await send_request(
                host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

    await decode_tracker.release(decoder_idx, peak_kv)


async def run_conversation_adaptive_disagg_decoders_per_turn_kv(
    conv_id, decode_tracker: KvTracker, decoders, compound_tokens
):
    """Same as run_conversation_adaptive_disagg_decoders except the KV
    reservation is per-turn (AMPD-style) rather than oracle-peak. The conv
    reserves only iter-0's footprint at start and then grows kv_held
    incrementally per iter — this lets new convs admit faster early in a run
    when long-tail convs haven't reached their expensive iters yet. No
    wrong-predict logic; otherwise identical to adaptive_disagg_decoders.
    """
    if cfg.ITER_COUNT[conv_id] < 1:
        return

    data0 = cfg.PROMPT_DATA[(conv_id, 0)]
    n_in  = data0['in_token_size']
    n_out = data0['out_token_size']
    decoder_idx = await decode_tracker.best_engine_for(n_in + n_out)
    host, port = decoders[decoder_idx]
    kv_held = n_in + n_out

    conversation_history = []

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        iter0_prompt_tokens = compound_tokens[:n_in]
        print(f"[conv {conv_id} iter 0] -> {host}:{port} (decoder={decoder_idx}, "
              f"cached prefill of {n_in} tokens, decode {n_out})")
        t0 = time.time()
        echo_text, _ = await send_request(
            host, port, session, iter0_prompt_tokens, n_out,
            request_id=f"{conv_id}-0", echo=True)
        t1 = time.time()
        log_step_latency(conv_id, 0, n_in, n_out, t0, t1, t1 - t0)
        conversation_history.append(echo_text)

        for iter_id in range(1, cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            await decode_tracker.add(decoder_idx, prompt_tokens + out_tokens)
            kv_held += prompt_tokens + out_tokens

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (decoder={decoder_idx})")
            t0 = time.time()
            output_text, _ = await send_request(
                host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)

    await decode_tracker.release(decoder_idx, kv_held)


async def run_conversation_per_turn_adaptive_disagg_decoders(
    conv_id, decode_tracker: KvTracker, decoders, compound_tokens, wrong_iters,
    virtual_prefiller=None,
):
    """Per-turn AMPD-style baseline. Same as adaptive_disagg_decoders except:

      1. KV-aware scheduling uses *current* usage rather than oracle-peak —
         reserve just iter-0 at conv start and grow as the conversation
         progresses, mirroring the per-turn KV-budget model the AMPD paper
         describes.
      2. For each (conv_id, iter_id) in `wrong_iters` (iter_id >= 1) we
         pause *this conversation's* task on the decoder for
            queueing_delay + wrong_predict_disagg_wait_ms
         where `queueing_delay` is queried from `virtual_prefiller` (= time
         to drain the simulated prefiller's in-flight iter-0s). The
         virtual prefiller also records the wrong-predict block so other
         convs' iter-0s in the synthetic prefiller trace get pushed back.
         Other concurrent conversations on the same decoder are not paused.
         After the pause, the decoder does its real cache-reuse prefill +
         decode for this turn.
    """
    if cfg.ITER_COUNT[conv_id] < 1:
        return

    # ── KV-aware scheduling: reserve just iter-0 worth, grow on the fly ──
    data0 = cfg.PROMPT_DATA[(conv_id, 0)]
    n_in  = data0['in_token_size']
    n_out = data0['out_token_size']
    decoder_idx = await decode_tracker.best_engine_for(n_in + n_out)
    host, port = decoders[decoder_idx]
    kv_held = n_in + n_out

    # Anchor virtual prefiller's wall-clock reference to the first conv
    # that actually reaches the decoder (i.e., post-Poisson-arrival).
    if virtual_prefiller is not None:
        virtual_prefiller.set_t0(time.time())

    conversation_history = []
    # lhist for the cost model: history KV already on the decoder for this
    # conv, starting from the warmed-up compound prefix.
    cumulative_tokens = len(compound_tokens)

    async with aiohttp.ClientSession(timeout=cfg.TIMEOUT) as session:
        # ── iter-0: cached prefill + real decode (decoder-side workload) ────
        iter0_prompt_tokens = compound_tokens[:n_in]
        print(f"[conv {conv_id} iter 0] -> {host}:{port} (decoder={decoder_idx}, "
              f"cached prefill of {n_in} tokens, decode {n_out})")
        t0 = time.time()
        echo_text, _ = await send_request(
            host, port, session, iter0_prompt_tokens, n_out,
            request_id=f"{conv_id}-0", echo=True)
        t1 = time.time()
        log_step_latency(conv_id, 0, n_in, n_out, t0, t1, t1 - t0)
        conversation_history.append(echo_text)
        cumulative_tokens += n_in + n_out

        # ── iter 1..N: real prefill + decode, with optional wrong-predict pause ─
        for iter_id in range(1, cfg.ITER_COUNT[conv_id]):
            data = cfg.PROMPT_DATA[(conv_id, iter_id)]
            prompt, prompt_tokens, out_tokens = (
                data['prompt'], data['in_token_size'], data['out_token_size']
            )
            conversation_history.append(prompt)
            full_prompt = "\n".join(conversation_history)

            # Reserve KV for what's about to land on the decoder.
            await decode_tracker.add(decoder_idx, prompt_tokens + out_tokens)
            kv_held += prompt_tokens + out_tokens

            if (conv_id, iter_id) in wrong_iters:
                base_ms = wrong_predict_disagg_wait_ms(cumulative_tokens, prompt_tokens)
                queueing_s = 0.0
                if virtual_prefiller is not None:
                    # Inject the block into the virtual prefiller and read back
                    # the queueing delay caused by in-flight recorded prefills.
                    from per_turn_cost_model import prefill_hit_ms
                    block_ms = prefill_hit_ms(cumulative_tokens + prompt_tokens)
                    queueing_s = await virtual_prefiller.query_and_block(time.time(), block_ms)
                total_s = queueing_s + base_ms / 1000.0
                print(f"[conv {conv_id} iter {iter_id}] WRONG PREDICT "
                      f"(lhist={cumulative_tokens}, lincr={prompt_tokens}) "
                      f"-> pause {total_s*1000:.1f} ms "
                      f"(queueing {queueing_s*1000:.1f} ms + base {base_ms:.1f} ms) "
                      f"before decoder")
                await asyncio.sleep(total_s)

            print(f"[conv {conv_id} iter {iter_id}] -> {host}:{port} (decoder={decoder_idx})")
            t0 = time.time()
            output_text, _ = await send_request(
                host, port, session, full_prompt, out_tokens, request_id=f"{conv_id}-{iter_id}")
            t1 = time.time()
            conversation_history.append(output_text)
            log_step_latency(conv_id, iter_id, prompt_tokens, out_tokens, t0, t1, t1 - t0)
            cumulative_tokens += prompt_tokens + out_tokens

    await decode_tracker.release(decoder_idx, kv_held)
