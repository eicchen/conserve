import json
import aiohttp
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())


# ── File paths ────────────────────────────────────────────────────────────────
PROMPTS_FILE = f"{REPO_ROOT}/conserve/input/mini_swe_agent_trace.json"
COMPOUND_PROMPTS_FILE = f"{REPO_ROOT}/conserve/input/compound_prompts.json"
OUT_DCGMI_FILE = "dcgmi_trace.tsv"
LATENCY_LOG_FILE = "per_step_latency.csv"

MOCK_INFERENCE = False  # set via --mock; skips vLLM and returns random text after 1 s

# ── Serving config ────────────────────────────────────────────────────────────
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
RPS = 1.5
REQUEST_INTERVAL = 1.0 / RPS
TIMEOUT = aiohttp.ClientTimeout(total=3600)
KV_BUDGET_TOKENS = 308_448  # 19278 blocks × 16 tokens/block at 0.9 GPU util

# ── Hardware monitoring ───────────────────────────────────────────────────────
REQUEST_COUNT = 256
DCGMI_CMD = [
    "bash", "-c",
    "dcgmi dmon -e 155,156,157,1130,1131,1132,1133,150,140,151,152,153,158,159,1110,1111,1112,858,100,101,102,110,111,1120,203,204,206,207,1100,1101,1102,1103,1104 -d 1 | ts '%Y-%m-%dT%H:%M:%.S' >> "
]

# ── Load prompts ──────────────────────────────────────────────────────────────
with open(PROMPTS_FILE, "r") as f:
    _raw = json.load(f)

PROMPT_DATA = {
    (entry['conv_id'], entry['iter_id']): {
        'in_token_size':  entry['in_token_size'],
        'out_token_size': entry['out_token_size'],
        'prompt':         entry['prompt'],
    }
    for entry in _raw if entry['conv_id'] < REQUEST_COUNT
}

CONV_COUNT = max(i for i, _ in PROMPT_DATA.keys())
ITER_COUNT = [0] * (CONV_COUNT + 1)
for i, _ in PROMPT_DATA.keys():
    ITER_COUNT[i] += 1


def apply_iter_cap(max_iters):
    """Cap each conversation to the first `max_iters` iterations.
    Re-filters PROMPT_DATA and rebuilds ITER_COUNT in place."""
    global PROMPT_DATA, ITER_COUNT
    if max_iters is None:
        return
    PROMPT_DATA = {k: v for k, v in PROMPT_DATA.items() if k[1] < max_iters}
    ITER_COUNT = [0] * (CONV_COUNT + 1)
    for i, _ in PROMPT_DATA.keys():
        ITER_COUNT[i] += 1
