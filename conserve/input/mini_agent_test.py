"""Run mini-swe-agent over a SWE-bench dataset and collect the raw per-problem
JSON outputs that we later convert into the serving experiment input trace.

Where this script fits in the pipeline:

    <DATASET_HF_NAME>                 (HuggingFace dataset, e.g. princeton-nlp/SWE-bench_bm25_13K)
        |
        |  THIS SCRIPT  --  one DefaultAgent run per problem, up to 16 in parallel
        |  python mini_agent_test.py --dataset <DATASET_HF_NAME> --split <SPLIT>
        v
    model_outputs/<MODEL_SHORT>/benchmarks/<BENCHMARK>/swe_output/
                                      (one JSON per problem; each contains the
                                       full message list + per-assistant-turn
                                       token usage from the LLM endpoint)
        |
        |  paper/figures/section3/notebooks/fig1_trace_profile.ipynb
        |  Step 2: flatten assistant turns, sort by (conv_id, iter_id)
        v
    model_outputs/<MODEL_SHORT>/benchmarks/<BENCHMARK>/
        mini_swe_agent_trace.json  (records: in/out token sizes,
                                   consumed by every serving experiment
                                   in output/{rps_sweep, order_sweep}/)
    BENCHMARK and MODEL_SHORT come from config/config.env.
    DATASET_HF_NAME and DATASET_SPLIT are set in fig1_trace_profile.ipynb.

Pre-requisites
--------------
*  The external `minisweagent` Python package must be installed in the
   active environment (see CONFIG_GUIDE in the upstream repo for which
   YAML config it loads via `get_config_from_spec("swebench")`).
*  The "swebench" config spec assumes an HTTP-reachable model endpoint
   (default: an OpenAI-compatible server on localhost:8000). Stand that up first.
*  HuggingFace's `datasets` library will download the dataset on first run
   into HF_CACHE_DIR.

Timing
------
Expect ~4 min/problem wall-clock at scale; bash execution in LocalEnvironment
dominates LLM call time. At MAX_WORKERS=16, the full 2,294-problem SWE-bench
sweep takes roughly 10-15 hours. Problems whose accumulated context exceeds the
model's token limit are skipped with a logged warning rather than crashing.

Usage
-----
    cd $REPO_ROOT/conserve
    python conserve/input/mini_agent_test.py   # full sweep over all 2,294 problems
    python conserve/input/mini_agent_test.py --max-problems 10 --max-turns 5  # quick test

Re-running will overwrite existing per-problem JSON files, so re-launching
after a crash resumes the sweep (already-complete files are overwritten but
that is fast since the agent finishes in one pass).
"""

import argparse
import asyncio
import sys
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

sys.path.insert(0, str(REPO_ROOT / "config"))
from config import BENCHMARK, BENCHMARK_TRACE_DIR, MODEL  # noqa: E402

HF_CACHE_DIR = REPO_ROOT / "conserve" / "datasets"

# Parallelism for the agent runs. The bottleneck is usually the LLM endpoint,
# not local CPU; tune to whatever your serving stack handles.
MAX_WORKERS = 16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=BENCHMARK,
                   help=f"HuggingFace dataset path (default: {BENCHMARK})")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Stop after this many problems (default: all)")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Cap agent turns per problem (default: unlimited)")
    return p.parse_args()


async def run_task(item, idx, agent_factory, sem, executor, pbar, output_dir):
    """Run one SWE-bench problem through a fresh DefaultAgent, writing the
    per-problem trace JSON to output_dir. Bounded by `sem` for concurrency."""
    async with sem:
        out_path = output_dir / f"output_{item.get('instance_id', idx)}_{idx}.json"
        agent = agent_factory(out_path)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                executor,
                lambda: agent.run(item["problem_statement"], context=item["text"]),
            )
        except Exception as e:
            print(f"\nSkipping {item.get('instance_id', idx)}: {type(e).__name__}: {e}",
                  flush=True)
        pbar.update(1)


async def main():
    args = parse_args()
    output_dir = BENCHMARK_TRACE_DIR / "swe_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config_from_spec("swebench")

    # Override model to use local vLLM endpoint instead of the Anthropic API
    # that the upstream swebench spec targets.
    model_config = config.get("model", {})
    model_config["model_name"] = f"openai/{MODEL}"
    model_config.setdefault("model_kwargs", {}).update({
        "api_base": "http://localhost:8000/v1",
        "api_key": "dummy",
    })
    model = get_model(config=model_config)
    env   = LocalEnvironment(**config.get("environment", {}))

    agent_kwargs = config.get("agent", {})
    if args.max_turns is not None:
        agent_kwargs = {**agent_kwargs, "step_limit": args.max_turns}

    def agent_factory(out_path: Path) -> DefaultAgent:
        return DefaultAgent(
            model, env,
            **agent_kwargs,
            output_path=str(out_path),
        )

    dataset = load_dataset(args.dataset, split=args.split,
                           cache_dir=str(HF_CACHE_DIR))
    if args.max_problems is not None:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))

    sem      = asyncio.Semaphore(MAX_WORKERS)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pbar     = tqdm(total=len(dataset), desc=f"{args.dataset} -> {output_dir.name}")
    try:
        await asyncio.gather(*[
            run_task(item, idx, agent_factory, sem, executor, pbar, output_dir)
            for idx, item in enumerate(dataset)
        ])
    finally:
        pbar.close()
        executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
