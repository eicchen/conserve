#!/bin/bash
# Parallel interference sweep: N_REPLICATES sharded across GPUs 0,1,2.
# Phase 1 = B-sweep (run_interference.py); Phase 2 = L_decoder-sweep
# (run_interference_kv.py). Each phase runs 3 shards in parallel (one per GPU),
# then merges the per-shard outputs back into the canonical data dir.

# Walk up from script location to find the .conserve_root marker.
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
source "$REPO_ROOT/config/config.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$(which python3)}"
MODEL_SHORT="${MODEL##*/}"
SEC3=$REPO_ROOT/model_outputs/$MODEL_SHORT/paper/section3/profiling

run_phase() {
    local script=$1 base=$2
    echo "=== $(date -Iseconds) phase start: $script ==="
    mkdir -p "$base"
    local pids=() logs=()
    for s in 0 1 2; do
        local log="$base/launcher_gpu${s}.log"
        CUDA_VISIBLE_DEVICES=$s "$PY" "$SCRIPT_DIR/$script" \
            --n-shards 3 --shard-id "$s" --port $((7701 + s)) \
            --out "$base/shard$s" > "$log" 2>&1 &
        pids+=($!)
        logs+=("$log")
    done
    local fail=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "  shard $i (pid ${pids[$i]}) FAILED — see ${logs[$i]}"
            fail=1
        fi
    done
    pkill -9 -f "[v]llm serve" 2>/dev/null || true
    pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
    sleep 10
    if [ "$fail" -ne 0 ]; then
        echo "=== $(date -Iseconds) ERROR: a shard of $script failed ==="
        return 1
    fi
    "$PY" "$SCRIPT_DIR/merge_interference_shards.py" "$base" 3
    echo "=== $(date -Iseconds) phase done: $script ==="
}

PHASE="${1:-all}"  # all | 1 | 2

if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    run_phase run_interference.py    "$SEC3/interference_data"    || exit 1
fi
if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    run_phase run_interference_kv.py "$SEC3/interference_kv_data" || exit 1
fi
echo "=== $(date -Iseconds) ALL DONE ==="
