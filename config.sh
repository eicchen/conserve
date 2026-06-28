#!/bin/bash
# Loads config.env and resolves relative paths against the repo root.
# Safe to source from any subdirectory — REPO_ROOT is computed from this file's location.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$REPO_ROOT/config.env" ]]; then
    while IFS='=' read -r key val; do
        # Skip blank lines and comment lines
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        key="${key// /}"  # strip spaces from key
        # config.env is primary — always overwrite any inherited env var
        printf -v "$key" '%s' "$val"
    done < "$REPO_ROOT/config.env"
fi

# Resolve relative paths against REPO_ROOT
[[ "$MODEL_DIR"   != /* ]] && MODEL_DIR="$REPO_ROOT/$MODEL_DIR"
[[ "$MODELS_ROOT" != /* ]] && MODELS_ROOT="$REPO_ROOT/$MODELS_ROOT"
[[ "$GPU_MON_ROOT" != /* ]] && GPU_MON_ROOT="$REPO_ROOT/$GPU_MON_ROOT"

export MODEL MODEL_DIR MODELS_ROOT GPU_TYPE GPU_MON_ROOT TENSOR_PARALLEL_SIZE

gpu_range() {
    local group=$1 tp=${TENSOR_PARALLEL_SIZE:-1}
    local start=$((group * tp))
    seq -s, $start $((start + tp - 1))
}
