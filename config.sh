#!/bin/bash
# Loads config.env and resolves relative paths against the repo root.
# Safe to source from any subdirectory — REPO_ROOT is computed from this file's location.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$REPO_ROOT/config.env" ]]; then
    while IFS='=' read -r key val; do
        # Skip blank lines and comment lines
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        key="${key// /}"  # strip spaces from key
        # Only set if not already exported in the calling environment
        [[ -v "$key" ]] || printf -v "$key" '%s' "$val"
    done < "$REPO_ROOT/config.env"
fi

# Resolve relative paths against REPO_ROOT
[[ "$MODEL_DIR"          != /* ]] && MODEL_DIR="$REPO_ROOT/$MODEL_DIR"
[[ "$PROFILING_DATA_DIR" != /* ]] && PROFILING_DATA_DIR="$REPO_ROOT/$PROFILING_DATA_DIR"
[[ "$GPU_MON_ROOT"       != /* ]] && GPU_MON_ROOT="$REPO_ROOT/$GPU_MON_ROOT"

export MODEL MODEL_DIR PROFILING_DATA_DIR GPU_MON_ROOT
