"""
Self-contained prefill linearity profile for Qwen3-0.6B.

For each L in L_VALUES, submit N_PROMPTS_PER_L fresh prompts of length L
(batch=1, max_tokens=2) and capture per-step engine timings. Writes one
per-L subdirectory of engine/core logs that plot_prefill_linearity.py
can parse.

Designed to be run twice — once per power-cap configuration — with the
output dir distinguishing the two. Path constants below point at the
section3/300W/ subdir to match the rest of the layout.
"""

import argparse
import os
import json
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
from paths import MODEL_DIR, PROFILING_DATA_DIR, MODEL


os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams

# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # local override
OUT = (REPO_ROOT / "paper/figures/section3/output/300W/prefill_profile_data")
OUT.mkdir(parents=True, exist_ok=True)

L_VALUES = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288,
            16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536]
N_PROMPTS_PER_L = 100
# rope_scaling factor 2.5 -> max_model_len = 32768 * 2.5 = 81920 (>= 65536 + 2 slack)
ROPE_FACTOR = 2.5
MAX_MODEL_LEN = 81920

# L values that have their own prompts_{L}x2048.json file.
HAVE_FILES = {128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288,
              16384, 20480, 24576, 28672, 32768, 65536}


def load_prompts(L: int, tokenizer):
    """Return list of prompt texts of length ~L.

    For L values without a dedicated file (40960, 49152, 57344), take the
    first L tokens of the 65536 prompts.
    """
    if L in HAVE_FILES:
        p = Path(f"{PROFILING_DATA_DIR}/prompts_{L}x2048.json")
        return [d["prompt"] for d in json.loads(p.read_text())[:N_PROMPTS_PER_L]]
    src = json.loads(Path(f"{PROFILING_DATA_DIR}/prompts_65536x2048.json").read_text())
    out = []
    for d in src[:N_PROMPTS_PER_L]:
        ids = tokenizer.encode(d["prompt"], add_special_tokens=False)[:L]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L-list", type=str, default=None,
                    help="Comma-separated subset of L values to run this shard. "
                         "Default: all of L_VALUES.")
    args = ap.parse_args()
    if args.L_list:
        my_L = [int(x) for x in args.L_list.split(",") if x.strip()]
    else:
        my_L = list(L_VALUES)

    llm = LLM(
        model=MODEL,
        dtype="auto",
        download_dir=MODEL_DIR,
        rope_scaling={"rope_type": "dynamic", "factor": ROPE_FACTOR},
        max_num_batched_tokens=max(L_VALUES) + 2048,
        max_num_seqs=4,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    sp = SamplingParams(
        temperature=1.2, top_p=1.0, max_tokens=2,
        logit_bias={151643: -100, 151644: -100, 151645: -100},
    )

    plan = {"model": MODEL, "L_values": list(my_L),
            "n_prompts_per_L": N_PROMPTS_PER_L,
            "rope_factor": ROPE_FACTOR, "max_model_len": MAX_MODEL_LEN,
            "cells": []}

    tokenizer = llm.get_tokenizer()
    for L in my_L:
        texts = load_prompts(L, tokenizer)
        # Each prompt as its own batch so they prefill sequentially (no
        # batched interference), matching how the legacy gpu_monitoring
        # prefill profile worked.
        fixed_batches = [[t] for t in texts]

        cell_dir = OUT / str(L)
        cell_dir.mkdir(parents=True, exist_ok=True)
        engine_log = cell_dir / "engine_log.jsonl"
        core_log = cell_dir / "core_log.jsonl"

        t0 = time.time()
        llm.generate(
            fixed_batches=fixed_batches,
            sampling_params=sp,
            engine_log_file=str(engine_log),
            core_log_file=str(core_log),
            use_tqdm=False,
        )
        dt = time.time() - t0
        msg = f"L={L:>6}: {N_PROMPTS_PER_L} prompts in {dt:.1f}s ({dt/N_PROMPTS_PER_L*1000:.0f} ms/prompt wall)"
        print(msg, flush=True)
        plan["cells"].append({"L": L, "n_prompts": N_PROMPTS_PER_L,
                              "wall_s": round(dt, 2),
                              "engine_log": str(engine_log.relative_to(OUT)),
                              "core_log": str(core_log.relative_to(OUT))})

    shard_id = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    with open(OUT / f"plan_shard_{shard_id}.json", "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
