"""Standalone copy of conserve/input/mini_agent_test.py for the fig1 sbatch
pipeline (see run_fig1_trace.sbatch in this directory).

Differs from the canonical conserve/input/mini_agent_test.py in two ways:
  1. Imports the local config.py copy in this directory (which honors
     CONSERVE_CONFIG_ENV) instead of config/config.py, so each sbatch job's
     isolated BENCHMARK override is picked up correctly.
  2. The vLLM endpoint port is read from VLLM_PORT (default 8000) instead of
     being hardcoded, so concurrently submitted jobs can each bind their own
     server without colliding on localhost:8000.

The canonical conserve/input/mini_agent_test.py is untouched; this copy is
intentionally kept in sync by hand for the sbatch pipeline only.

---

Run mini-swe-agent over a SWE-bench dataset and collect the raw per-problem
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
        |  build_agent_trace.py (this directory)
        |  Step 2: flatten assistant turns, sort by (conv_id, iter_id)
        v
    model_outputs/<MODEL_SHORT>/benchmarks/<BENCHMARK>/
        mini_swe_agent_trace.json  (records: in/out token sizes,
                                   consumed by every serving experiment
                                   in output/{rps_sweep, order_sweep}/)
    BENCHMARK and MODEL_SHORT come from config.env (or its CONSERVE_CONFIG_ENV
    override — see run_fig1_trace.sbatch).

Pre-requisites
--------------
*  The external `minisweagent` Python package must be installed in the
   active environment (see CONFIG_GUIDE in the upstream repo for which
   YAML config it loads via `get_config_from_spec("swebench")`).
*  The "swebench" config spec assumes an HTTP-reachable model endpoint
   (default: an OpenAI-compatible server on localhost:$VLLM_PORT). Stand
   that up first.
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
    python mini_agent_test.py   # full sweep over all 2,294 problems
    python mini_agent_test.py --max-problems 10 --max-turns 5  # quick test

Re-running will overwrite existing per-problem JSON files, so re-launching
after a crash resumes the sweep (already-complete files are overwritten but
that is fast since the agent finishes in one pass).
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in SCRIPT_DIR.parents
                 if (p / ".conserve_root").exists())

sys.path.insert(0, str(SCRIPT_DIR))
from config import BENCHMARK, MODEL, MODEL_DATA_DIR  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "conserve" / "input"))
from benchmark_adapters import checkout_task_repo, cleanup_task_repo, get_adapter  # noqa: E402

HF_CACHE_DIR = REPO_ROOT / "conserve" / "datasets"

# Parallelism for the agent runs. The bottleneck is usually the LLM endpoint,
# not local CPU; tune to whatever your serving stack handles.
MAX_WORKERS = 16


def _non_negative_int(value):
    """Argparse type validator that rejects negative integers."""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value!r}")
    return ivalue


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=BENCHMARK,
                   help=f"HuggingFace dataset path (default: {BENCHMARK})")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--max-problems", type=_non_negative_int, default=None,
                   help="Stop after this many problems (default: all)")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Cap agent turns per problem (default: unlimited)")
    p.add_argument("--resume", action="store_true",
                   help="Skip tasks whose output already reached a terminal "
                        "state (for resuming after a Slurm time-limit kill). "
                        "Only safe across runs with the same --dataset/--split "
                        "and an unchanged-or-larger --max-problems.")
    return p.parse_args()


def _task_output_is_done(out_path: Path) -> bool:
    """True if out_path holds a trajectory that reached a terminal state
    (last message role == "exit"). Missing, unreadable, or truncated files
    are treated as not-done so they get rerun. DefaultAgent.save() writes
    out_path after every turn, not just at the end, so existence alone
    doesn't mean the task finished."""
    if not out_path.exists():
        return False
    try:
        with open(out_path) as fh:
            data = json.load(fh)
        messages = data.get("messages", [])
        return bool(messages) and messages[-1].get("role") == "exit"
    except (json.JSONDecodeError, OSError):
        return False


@contextmanager
def task_workspace(base_cwd: str, task_id: str, task):
    """Create a per-task scratch directory under base_cwd, removing it on
    exit regardless of how the task ends. Pairs creation and cleanup in one
    place so a full sweep can't leave per-task directories behind. Yields
    None when the benchmark config has no cwd to scope (nothing to create
    or clean up).

    For SWE-bench-family tasks (task.repo/task.base_commit set), the
    directory is populated with an actual checkout of that repo at that
    commit via checkout_task_repo, instead of being left empty."""
    if not base_cwd:
        yield None
        return
    path = Path(base_cwd) / task_id
    if task.repo and task.base_commit:
        checkout_task_repo(task.repo, task.base_commit, path)
    else:
        path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if task.repo and task.base_commit:
            cleanup_task_repo(task.repo, path)
        else:
            shutil.rmtree(path, ignore_errors=True)


async def run_task(task, idx, agent_factory, sem, executor, pbar, output_dir, base_cwd):
    """Run one benchmark task through a fresh DefaultAgent, writing the
    per-problem trace JSON to output_dir. Bounded by `sem` for concurrency.

    The try/except spans workspace setup (task_workspace's checkout_task_repo
    can fail, e.g. an unreachable base_commit) through the agent run, so one
    bad task is skipped and logged instead of propagating out of gather()
    and killing every other concurrently running task."""
    async with sem:
        task_id = f"{task.instance_id}_{idx}"
        out_path = output_dir / f"output_{task_id}.json"
        try:
            with task_workspace(base_cwd, task_id, task) as task_cwd:
                agent = agent_factory(out_path, task_cwd)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor,
                    lambda: agent.run(task.prompt),
                )
        except Exception as e:
            print(f"\nSkipping {task.instance_id}: {type(e).__name__}: {e}",
                  flush=True)
        pbar.update(1)


async def main():
    args = parse_args()
    # Output directory name follows whatever --dataset was actually run, not
    # the static BENCHMARK in config/config.env, so ad-hoc runs against a
    # different benchmark don't collide with another benchmark's outputs.
    benchmark_short = args.dataset.split("/")[-1]
    output_dir = MODEL_DATA_DIR / "benchmarks" / benchmark_short / "swe_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, config_spec = get_adapter(args.dataset)
    config = get_config_from_spec(config_spec)

    # Override model to use local vLLM endpoint instead of the Anthropic API
    # that the upstream swebench spec targets. VLLM_PORT lets concurrent runs
    # (e.g. different benchmarks in parallel sbatch jobs) each point at their
    # own server instead of colliding on the default port.
    vllm_port = os.environ.get("VLLM_PORT", "8000")
    model_config = config.get("model", {})
    model_config["model_name"] = f"openai/{MODEL}"
    model_config.setdefault("model_kwargs", {}).update({
        "api_base": f"http://localhost:{vllm_port}/v1",
        "api_key": "dummy",
    })
    model = get_model(config=model_config)
    env_config = config.get("environment", {})
    # The configured cwd (e.g. /tmp/lcb_workspace, /testbed) isn't created by
    # anything in this pipeline, and for SWE-bench-family configs is a
    # top-level path (/testbed) a non-root user can't even mkdir. Rather
    # than trust the literal configured path, only use it as a signal that
    # this benchmark wants a scoped working directory at all, and redirect
    # to a writable scratch root; task_workspace() then scopes a fresh
    # subdirectory per task (MAX_WORKERS tasks run at once) so parallel
    # agents don't clobber each other's files, and removes it when the task
    # ends.
    base_cwd = (str(Path(tempfile.gettempdir()) / "mini_agent_workspace" / benchmark_short)
                if env_config.get("cwd") else "")

    agent_kwargs = config.get("agent", {})
    if args.max_turns is not None:
        agent_kwargs = {**agent_kwargs, "step_limit": args.max_turns}

    def agent_factory(out_path: Path, task_cwd: Path | None) -> DefaultAgent:
        task_env_config = dict(env_config)
        if task_cwd is not None:
            task_env_config["cwd"] = str(task_cwd)
        env = LocalEnvironment(**task_env_config)
        return DefaultAgent(
            model, env,
            **agent_kwargs,
            output_path=str(out_path),
        )

    tasks = loader(args.dataset, args.split, HF_CACHE_DIR, args.max_problems)

    indexed_tasks = list(enumerate(tasks))
    n_done = 0
    if args.resume:
        remaining = []
        for idx, task in indexed_tasks:
            out_path = output_dir / f"output_{task.instance_id}_{idx}.json"
            if _task_output_is_done(out_path):
                n_done += 1
            else:
                remaining.append((idx, task))
        indexed_tasks = remaining
        print(f"--resume: {n_done}/{len(tasks)} tasks already done, "
              f"{len(indexed_tasks)} remaining", flush=True)

    sem      = asyncio.Semaphore(MAX_WORKERS)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pbar     = tqdm(total=len(tasks), initial=n_done, desc=f"{args.dataset} -> {output_dir.name}")
    try:
        await asyncio.gather(*[
            run_task(task, idx, agent_factory, sem, executor, pbar, output_dir, base_cwd)
            for idx, task in indexed_tasks
        ])
    finally:
        pbar.close()
        executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
