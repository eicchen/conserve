#!/bin/bash
# Loads config.env and resolves relative paths against the repo root.
# Safe to source from any subdirectory — REPO_ROOT is computed from this file's location.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/config/config.env" ]]; then
    while IFS='=' read -r key val; do
        # Skip blank lines and comment lines
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        key="${key// /}"  # strip spaces from key
        # config.env is primary — always overwrite any inherited env var
        printf -v "$key" '%s' "$val"
    done < "$REPO_ROOT/config/config.env"
fi

# Resolve relative paths against REPO_ROOT
[[ "$MODEL_DIR"   != /* ]] && MODEL_DIR="$REPO_ROOT/$MODEL_DIR"
[[ "$MODELS_ROOT" != /* ]] && MODELS_ROOT="$REPO_ROOT/$MODELS_ROOT"
[[ "$GPU_MON_ROOT" != /* ]] && GPU_MON_ROOT="$REPO_ROOT/$GPU_MON_ROOT"

# Derive compound paths not stored verbatim in config.env
MODEL_SHORT="${MODEL##*/}"
BENCHMARK="${BENCHMARK:-princeton-nlp/SWE-bench_bm25_13K}"
BENCHMARK_SHORT="${BENCHMARK##*/}"
BENCHMARK_TRACE_DIR="$MODELS_ROOT/$MODEL_SHORT/benchmarks/$BENCHMARK_SHORT"

export MODEL MODEL_SHORT MODEL_DIR MODELS_ROOT GPU_TYPE GPU_MON_ROOT TENSOR_PARALLEL_SIZE PREFILLER_GPU_MEM_UTIL DECODER_GPU_MEM_UTIL BENCHMARK BENCHMARK_SHORT BENCHMARK_TRACE_DIR

# Load model-specific vLLM serve flags from model_specific_configs.toml.
# config.sh requires Python 3 with tomli/tomllib available (satisfied by both
# conserve envs — the package is installed as a vLLM transitive dependency).
# `python config.py --sh-vars` prints a bash declaration:
#   VLLM_SERVE_FLAGS=(<flag> ...)  — static per-model/env flags for `vllm serve`
# VLLM_SERVE_FLAGS is a bash array and cannot be exported; every script that
# needs it must source config.sh directly (all launch scripts already do this).
eval "$(python "$REPO_ROOT/config/config.py" --sh-vars)"
