#!/bin/bash
#SBATCH --job-name=trace-benchmark
#SBATCH --partition=seec-own
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --output=sbatch/fig1/logs/fig1_trace_%j.out
#SBATCH --error=sbatch/fig1/logs/fig1_trace_%j.err
#
# Submit from the repo root so the relative --output/--error paths above resolve:
#   sbatch sbatch/fig1/run_fig1_trace.sbatch [BENCHMARK] [SPLIT] [MAX_PROBLEMS] [MAX_TURNS]
#
# Stitches the 3-shell workflow described in
# paper/figures/section3/notebooks/fig1_trace_profile.ipynb into one batch
# job. This directory holds standalone copies of config.py / config.sh /
# mini_agent_test.py / build_agent_trace.py — the canonical versions under
# config/ and conserve/input/ are untouched. The copies add two small,
# sbatch-only hooks (kept out of the canonical files so the normal
# interactive 3-shell workflow stays exactly as documented in the notebook):
#   - CONSERVE_CONFIG_ENV: points config.py/config.sh at an isolated
#     config.env copy instead of the shared repo one.
#   - VLLM_PORT: lets mini_agent_test.py target a non-default server port.
#
# Steps:
#   1. serve the model with vLLM
#   2. run the mini-swe-agent sweep over a benchmark dataset
#   3. stitch the per-problem JSONs into mini_swe_agent_trace.json
#      (plotting is skipped here — see Concurrency note below; run
#      paper/figures/section3/scripts/plot_trace.py by hand afterward,
#      once per benchmark, with config/config.env's BENCHMARK set to match)
#
# Concurrency: safe to submit multiple jobs with different BENCHMARK values at
# once. Each job works off a private copy of config/config.env (via
# CONSERVE_CONFIG_ENV) and binds vLLM to a freshly allocated port (via
# VLLM_PORT), so jobs never race on the shared config file or collide on
# localhost:8000 even when scheduled on the same node. MODEL/MODEL_DIR/
# MODELS_ROOT are still shared across jobs — only point this at different
# MODELs if they don't share a GPU/port allocation.
# Step 3 passes --no-plot because paper/figures/section3/scripts/plot_trace.py
# (deliberately left canonical/untouched, unlike config.py/config.sh/
# mini_agent_test.py/build_agent_trace.py in this directory) always reads
# BENCHMARK from the shared config/config.env — it does not honor this job's
# CONSERVE_CONFIG_ENV override — and always writes to the same fixed
# paper/figures/section3/output/trace_profile.{png,pdf} path. Running it from
# two concurrent jobs would plot the wrong benchmark's data and/or clobber
# each other's output file.
#
# Arguments (all optional)
# -------------------------
#   BENCHMARK     HuggingFace dataset path, e.g. princeton-nlp/SWE-bench_bm25_13K.
#                 Defaults to whatever is already set in config/config.env.
#                 REMINDER: benchmarks planned for this pipeline are
#                 princeton-nlp/SWE-bench_bm25_13K (current default),
#                 livecodebench/code_generation, ScaleAI/SWE-bench_Pro, and
#                 harborframework/terminal-bench-2.0 — see the commented-out
#                 BENCHMARK= options in config/config.env.
#   SPLIT         Dataset split (default: test)
#   MAX_PROBLEMS  Cap problems for a quick smoke test (default: unset = full sweep)
#   MAX_TURNS     Cap agent turns/problem for a quick smoke test (default: unset)
#
# Examples
#   sbatch sbatch/fig1/run_fig1_trace.sbatch
#       # full sweep against the benchmark already configured in config.env
#   sbatch sbatch/fig1/run_fig1_trace.sbatch princeton-nlp/SWE-bench_bm25_13K test 10 5
#       # quick smoke test (10 problems, 5 turns) against a different benchmark
#   for b in princeton-nlp/SWE-bench_bm25_13K princeton-nlp/SWE-bench_bm25_13K; do
#       sbatch sbatch/fig1/run_fig1_trace.sbatch "$b"
#   done
#       # two benchmarks at once — each gets its own config copy + port

set -euo pipefail

# Non-interactive Slurm shells don't source ~/.bashrc, so conda's init hook
# never runs — activate explicitly or `vllm` won't be on PATH.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate conserve
set -u

# NOTE: don't derive paths from BASH_SOURCE/$0 — Slurm copies the submitted
# script into a per-job spool dir (e.g. /var/spool/slurmd/job<ID>/) and runs
# it from there, so BASH_SOURCE points at that spool copy, not this repo.
# SLURM_SUBMIT_DIR (the directory sbatch was invoked from) is what's reliable.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -e "$REPO_ROOT/.conserve_root" ]]; then
    echo "REPO_ROOT ($REPO_ROOT) has no .conserve_root — submit this job from the repo root" >&2
    exit 1
fi
SCRIPT_DIR="$REPO_ROOT/sbatch/fig1"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

BENCHMARK_OVERRIDE="${1:-}"
SPLIT="${2:-test}"
MAX_PROBLEMS="${3:-}"
MAX_TURNS="${4:-}"

# Job-private config.env copy — never edit the shared repo file, so
# concurrently submitted jobs (e.g. different BENCHMARK per job) don't race.
JOB_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/conserve-fig1-${SLURM_JOB_ID:-$$}-XXXXXX")"
JOB_CONFIG_ENV="$JOB_TMP_DIR/config.env"
cp "$REPO_ROOT/config/config.env" "$JOB_CONFIG_ENV"
if [[ -n "$BENCHMARK_OVERRIDE" ]]; then
    sed -i "s|^BENCHMARK=.*|BENCHMARK=${BENCHMARK_OVERRIDE}|" "$JOB_CONFIG_ENV"
fi
export CONSERVE_CONFIG_ENV="$JOB_CONFIG_ENV"

# Job-private port — avoids colliding with another job's vLLM server on the
# same node. Bind-and-release is the standard "find a free port" pattern;
# there's a brief window before vllm rebinds it, acceptable here.
export VLLM_PORT="${VLLM_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

source "$SCRIPT_DIR/config.sh"   # exports MODEL, MODEL_DIR, BENCHMARK, BENCHMARK_SHORT, VLLM_SERVE_FLAGS, ...
cd "$REPO_ROOT/conserve"

BENCH_LOG_DIR="$LOG_DIR/$BENCHMARK_SHORT"
mkdir -p "$BENCH_LOG_DIR"
for _ext in out err; do
    _f="$LOG_DIR/fig1_trace_${SLURM_JOB_ID:-local}.$_ext"
    [[ -f "$_f" ]] && mv "$_f" "$BENCH_LOG_DIR/"
done

echo "MODEL=$MODEL  BENCHMARK=$BENCHMARK  SPLIT=$SPLIT  VLLM_PORT=$VLLM_PORT"
echo "Job config: $JOB_CONFIG_ENV"
[[ -n "$MAX_PROBLEMS" ]] && echo "MAX_PROBLEMS=$MAX_PROBLEMS  MAX_TURNS=$MAX_TURNS  (smoke test)"

# ── Shell 1 (backgrounded) — serve the model ────────────────────────────────
VLLM_LOG="$BENCH_LOG_DIR/vllm_serve_${SLURM_JOB_ID:-local}.log"
vllm serve "$MODEL" --download-dir "$MODEL_DIR" --port "$VLLM_PORT" \
    "${VLLM_SERVE_FLAGS[@]}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 512 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -rf "$JOB_TMP_DIR"
}
trap cleanup EXIT

echo "Waiting for vLLM server to come up (log: $VLLM_LOG) ..."
for _ in $(seq 1 180); do   # 180 x 10s = 30 min startup budget
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server process died before becoming healthy — see $VLLM_LOG" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM server did not become healthy within the startup budget — see $VLLM_LOG" >&2
    exit 1
fi
echo "vLLM server healthy."

# ── Shell 2 — run the agent sweep ────────────────────────────────────────────
# mini_agent_test.py (local copy) reads VLLM_PORT (exported above) to target
# this job's own server instead of the default localhost:8000.
AGENT_ARGS=(--dataset "$BENCHMARK" --split "$SPLIT")
[[ -n "$MAX_PROBLEMS" ]] && AGENT_ARGS+=(--max-problems "$MAX_PROBLEMS")
[[ -n "$MAX_TURNS" ]] && AGENT_ARGS+=(--max-turns "$MAX_TURNS")

python "$SCRIPT_DIR/mini_agent_test.py" "${AGENT_ARGS[@]}"

# ── Shell 3 — stitch outputs (plotting deferred, see Concurrency note above) ─
python "$SCRIPT_DIR/build_agent_trace.py" --no-plot
