import asyncio
import csv
import json
import random
import string
import time

import aiohttp

import input_loader as cfg


async def send_request_mock(engine_host, engine_port, session, prompt, max_tokens,
                            request_id: str = None, decoder_index: int = None,
                            echo: bool = False):
    await asyncio.sleep(1.0)
    # ~4 chars per token as a rough approximation
    output_text = ''.join(random.choices(string.ascii_lowercase + '  ', k=max_tokens * 4))
    if echo:
        # Mock: fabricate an echoed prompt of approximately the right length.
        prompt_len = len(prompt) if isinstance(prompt, list) else max(1, len(prompt) // 4)
        echo_text = ''.join(random.choices(string.ascii_lowercase + '  ', k=prompt_len * 4))
        output_text = echo_text + output_text
    return output_text, max_tokens


async def send_request(engine_host, engine_port, session, prompt, max_tokens,
                       request_id: str = None, decoder_index: int = None,
                       echo: bool = False):
    if cfg.MOCK_INFERENCE:
        return await send_request_mock(engine_host, engine_port, session, prompt, max_tokens,
                                       request_id, decoder_index, echo)

    url = f"http://{engine_host}:{engine_port}/v1/completions"
    payload = {
        "model": cfg.MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "logit_bias": {
            2: -100,
            13: -100,
            128001: -100,
            128009: -100,
        },
    }
    if request_id is not None:
        payload["request_id"] = request_id
    if decoder_index is not None:
        payload["decoder_index"] = decoder_index
    if echo:
        payload["echo"] = True

    async with session.post(url, json=payload) as resp:
        resp_json = json.loads(await resp.text())
        # When echo=True, choices[0].text contains the detokenized prompt + completion.
        output_text = resp_json["choices"][0]["text"]
        completion_tokens = resp_json.get("usage", {}).get("completion_tokens", max_tokens)

    return output_text, completion_tokens


def log_step_latency(conv_id, step_id, prompt_tokens, max_tokens, start_time, end_time, latency):
    with open(cfg.LATENCY_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([conv_id, step_id, prompt_tokens, max_tokens, start_time, end_time, f"{latency:.4f}"])
    print(f"[conv {conv_id} step {step_id}] latency logged: {latency:.2f}s")
