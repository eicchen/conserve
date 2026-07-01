"""Per-benchmark task loading + minisweagent config spec selection.

Shared by conserve/input/mini_agent_test.py and its sbatch-local copy
(sbatch/fig1/mini_agent_test.py). This module has no
CONSERVE_CONFIG_ENV-specific behavior (see sbatch/fig1/config.py's
docstring for why some files are duplicated instead), so it lives in one
place and both copies import it directly.
"""

import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent

# Shared local bare-clone cache for SWE-bench-family repos. One clone per
# unique repo, reused across every task/instance that touches it via `git
# worktree add` — avoids re-cloning from GitHub per task and avoids fully
# duplicating repo history to disk per task.
#
# This cache is process-global (unlike config.env/VLLM_PORT, which
# run_fig1_trace.sbatch gives each job its own copy of): separate sbatch
# jobs for different benchmarks — or the same benchmark split across
# multiple jobs — all share it. A same-process threading.Lock only
# serializes tasks within one job, so first-time clones use
# clone-to-temp-dir-then-atomic-rename below to stay correct even when two
# separate job processes race to clone the same repo simultaneously.
REPO_CACHE_DIR = SCRIPT_DIR.parent / "datasets" / "repo_cache"
_mirror_lock = threading.Lock()


@dataclass
class Task:
    instance_id: str
    prompt: str
    # SWE-bench-family only: which repo/commit to check out as the task's
    # working directory. None for benchmarks (e.g. livecodebench) that don't
    # need a real repo checkout.
    repo: str | None = None
    base_commit: str | None = None


def _load_swebench_family(dataset: str, split: str, cache_dir: Path,
                           max_problems: int | None) -> list[Task]:
    """princeton-nlp/SWE-bench_* and ScaleAI/SWE-bench_Pro: both expose a
    `problem_statement` field with the raw issue text. (SWE-bench_bm25_13K
    additionally has a `text` field with a BM25-retrieval-augmented
    rendering, but nothing in the swebench.yaml instance_template renders
    it, so it's intentionally not read here.) `repo` + `base_commit` are the
    fields the upstream SWE-bench harness uses to build each instance's
    Docker image; here they're used instead to check out a plain git
    worktree (see checkout_task_repo)."""
    split_spec = f"{split}[:{max_problems}]" if max_problems is not None else split
    ds = load_dataset(dataset, split=split_spec, cache_dir=str(cache_dir))
    return [
        Task(instance_id=item.get("instance_id", str(idx)),
             prompt=item["problem_statement"],
             repo=item["repo"],
             base_commit=item["base_commit"])
        for idx, item in enumerate(ds)
    ]


def _mirror_path(repo: str) -> Path:
    return REPO_CACHE_DIR / f"{repo.replace('/', '__')}.git"


def _ensure_repo_mirror(repo: str) -> Path:
    """Return a local bare mirror clone of `repo` (e.g. 'astropy/astropy'),
    cloning it once if not already cached.

    Correct under two layers of concurrency: a threading.Lock skips
    redundant clones from other tasks in this process, and — since the
    cache is shared across separate job processes too (see REPO_CACHE_DIR
    docstring) — the actual clone lands in a unique temp dir first and is
    only made visible via an atomic rename. If another process's clone
    lands first, our rename fails (destination exists) and we discard the
    redundant clone instead of racing to write the same directory."""
    mirror_path = _mirror_path(repo)
    if mirror_path.exists():
        return mirror_path
    with _mirror_lock:
        if mirror_path.exists():
            return mirror_path
        REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = REPO_CACHE_DIR / f".tmp-{os.getpid()}-{uuid.uuid4().hex}-{mirror_path.name}"
        subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(tmp_path)],
            check=True, capture_output=True, text=True,
        )
        try:
            tmp_path.rename(mirror_path)
        except OSError:
            # Another process's clone won the race and is already at
            # mirror_path; ours is redundant.
            shutil.rmtree(tmp_path, ignore_errors=True)
    return mirror_path


def checkout_task_repo(repo: str, base_commit: str, dest: Path) -> None:
    """Materialize `repo` at `base_commit` into `dest` as a detached git
    worktree off the shared local mirror — fast and network-free after the
    mirror's first clone."""
    mirror_path = _ensure_repo_mirror(repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(dest), base_commit],
        cwd=mirror_path, check=True, capture_output=True, text=True,
    )


def cleanup_task_repo(repo: str, dest: Path) -> None:
    """Remove a worktree created by checkout_task_repo and prune the
    mirror's worktree metadata."""
    mirror_path = _mirror_path(repo)
    subprocess.run(["git", "worktree", "remove", "--force", str(dest)],
                   cwd=mirror_path, check=False, capture_output=True, text=True)
    subprocess.run(["git", "worktree", "prune"],
                   cwd=mirror_path, check=False, capture_output=True, text=True)


LIVECODEBENCH_CONFIG = str(SCRIPT_DIR / "configs" / "livecodebench.yaml")


def _load_livecodebench(dataset: str, split: str, cache_dir: Path,
                         max_problems: int | None) -> list[Task]:
    split_spec = f"{split}[:{max_problems}]" if max_problems is not None else split
    ds = load_dataset(dataset, split=split_spec, cache_dir=str(cache_dir))
    tasks = []
    for idx, item in enumerate(ds):
        prompt = (
            f"## {item['question_title']}\n\n"
            f"{item['question_content']}\n\n"
            f"### Starter code\n```python\n{item['starter_code']}\n```\n\n"
            f"### Public test cases (JSON)\n```json\n{item['public_test_cases']}\n```\n"
        )
        tasks.append(Task(instance_id=item.get("question_id", str(idx)), prompt=prompt))
    return tasks


ADAPTERS: dict[str, tuple[Callable, str]] = {
    "princeton-nlp/SWE-bench_bm25_13K": (_load_swebench_family, "swebench"),
    "ScaleAI/SWE-bench_Pro": (_load_swebench_family, "swebench"),
    "livecodebench/code_generation": (_load_livecodebench, LIVECODEBENCH_CONFIG),
}


def get_adapter(dataset: str) -> tuple[Callable, str]:
    if dataset in ADAPTERS:
        return ADAPTERS[dataset]
    if dataset.startswith("princeton-nlp/SWE-bench"):
        return _load_swebench_family, "swebench"
    raise KeyError(
        f"No benchmark adapter registered for {dataset!r}. "
        f"Add one to ADAPTERS in {__file__}."
    )
