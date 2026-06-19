#!/bin/bash
# Parallel interference sweep: N_REPLICATES sharded across GPUs 0,1,2.
# Phase 1 = B-sweep (run_interference.py); Phase 2 = L_decoder-sweep
# (run_interference_kv.py). Each phase runs 3 shards in parallel (one per GPU),
# then merges the per-shard outputs back into the canonical data dir.

# Walk up from script location to find the .conserve_root marker.
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
source "$REPO_ROOT/config.sh"

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$(dirname "$(which python3)"):$PATH"
export TMPDIR=/tmp

SEC3=$REPO_ROOT/paper/figures/section3/output/300W

run_phase() {
    local script=$1 base=$2
    echo "=== $(date -Iseconds) phase start: $script ==="
    local pids=()
    for s in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$s python3 "$script" \
            --n-shards 3 --shard-id "$s" --port $((7701 + s)) \
            --out "$base/shard$s" &
        pids+=($!)
    done
    local fail=0
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    pkill -9 -f "[v]llm serve" 2>/dev/null
    pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null
    sleep 10
    if [ "$fail" -ne 0 ]; then
        echo "=== $(date -Iseconds) ERROR: a shard of $script failed ==="
        return 1
    fi
    python3 merge_interference_shards.py "$base" 3
    echo "=== $(date -Iseconds) phase done: $script ==="
}

run_phase run_interference.py    "$SEC3/interference_data"    || exit 1
run_phase run_interference_kv.py "$SEC3/interference_kv_data" || exit 1
echo "=== $(date -Iseconds) ALL DONE ==="
