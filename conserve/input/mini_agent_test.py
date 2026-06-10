"""Run mini-swe-agent over the SWE-bench bm25_13K dataset and collect the raw
per-problem JSON outputs that we later convert into the serving experiment
input trace.

Where this script fits in the pipeline:

    SWE-bench_bm25_13K               (HuggingFace dataset, 2,294 problems)
        |
        |  THIS SCRIPT  --  one DefaultAgent run per problem, 16 in parallel
        v
    output/SWE-bench_bm25_13K/       (one JSON per problem; each contains the
                                      full message list + per-assistant-turn
                                      token usage from the LLM endpoint)
        |
        |  output/plot_dataset.ipynb (cells 5-10: flatten assistant turns,
        |                             pad each prompt with Gutenberg filler to
        |                             defeat prefix caching, sort by
        |                             (conv_id, iter_id))
        v
    input/mini_swe_agent_trace.json  (22,805 records: in/out token sizes +
                                      padded prompts, consumed by every
                                      serving experiment in
                                      output/{rps_sweep, order_sweep}/)

Pre-requisites
--------------
*  The external `minisweagent` Python package must be installed in the
   active environment (see CONFIG_GUIDE in the upstream repo for which
   YAML config it loads via `get_config_from_spec("swebench")`).
*  The "swebench" config spec assumes an HTTP-reachable model endpoint
   (default: an OpenAI-compatible server on localhost). Stand that up first.
*  HuggingFace's `datasets` library will download SWE-bench bm25_13K on first
   run into HF_CACHE_DIR.

Usage
-----
    cd $REPO_ROOT/conserve
    python mini_agent_test.py            # full sweep over all 2,294 problems

Re-running will overwrite existing per-problem JSON files (one per
instance_id), so to resume an interrupted sweep you can run as-is.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())

DATASET_NAME  = "princeton-nlp/SWE-bench_bm25_13K"
DATASET_SPLIT = "test"
HF_CACHE_DIR  = REPO_ROOT / "conserve" / "datasets"
OUTPUT_DIR    = REPO_ROOT / "conserve" / "output" / "SWE-bench_bm25_13K"

# Parallelism for the agent runs. The bottleneck is usually the LLM endpoint,
# not local CPU; tune to whatever your serving stack handles.
MAX_WORKERS = 16


async def run_task(item, idx, agent_factory, sem, executor, pbar):
    """Run one SWE-bench problem through a fresh DefaultAgent, writing the
    per-problem trace JSON to OUTPUT_DIR. Bounded by `sem` for concurrency."""
    async with sem:
        out_path = OUTPUT_DIR / f"output_{item.get('instance_id', idx)}_{idx}.json"
        agent = agent_factory(out_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: agent.run(item["problem_statement"], context=item["text"]),
        )
        pbar.update(1)


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The model + env are stateful; share them across all agent instances and
    # only create a fresh DefaultAgent (which holds per-conversation state)
    # per task.
    config = get_config_from_spec("swebench")
    model  = get_model(config=config.get("model", {}))
    env    = LocalEnvironment(**config.get("environment", {}))

    def agent_factory(out_path: Path) -> DefaultAgent:
        return DefaultAgent(
            model, env,
            **config.get("agent", {}),
            output_path=str(out_path),
        )

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT,
                            cache_dir=str(HF_CACHE_DIR))

    sem      = asyncio.Semaphore(MAX_WORKERS)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pbar     = tqdm(total=len(dataset),
                    desc=f"{DATASET_NAME} -> {OUTPUT_DIR.name}")
    try:
        await asyncio.gather(*[
            run_task(item, idx, agent_factory, sem, executor, pbar)
            for idx, item in enumerate(dataset)
        ])
    finally:
        pbar.close()
        executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
