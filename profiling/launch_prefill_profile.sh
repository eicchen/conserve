#!/bin/bash
# Launch the prefill linearity sweep in parallel across 3 GPUs (1, 2, 3).
# Cells are split with greedy LPT (longest-processing-time-first) to balance
# wall-clock load — biggest L on each GPU first, smaller L appended to
# whichever GPU has the lowest cumulative cost so far.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
PY="${PY:-$(which python3)}"
MODEL_SHORT="${MODEL##*/}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/model_outputs/$MODEL_SHORT/paper/section3/profiling/prefill_profile_data}"
mkdir -p "$OUT_DIR"

# Static LPT assignment (computed offline based on expected per-L runtimes at 300W).
# Each line: GPU_id   L1,L2,...
declare -A SHARDS=(
    [1]="65536,28672,16384,6144,512"
    [2]="57344,32768,24576,10240,2048,128"
    [3]="49152,40960,20480,12288,8192,4096,1024,256"
)

declare -a PIDS=()
declare -a LOGS=()
for GPU in 1 2 3; do
    L_LIST="${SHARDS[$GPU]}"
    LOG="$OUT_DIR/launcher_gpu${GPU}.log"
    echo "GPU $GPU  ←  L = ${L_LIST}"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT_DIR/run_prefill_profile.py" \
        --L-list "$L_LIST" > "$LOG" 2>&1 &
    PIDS+=($!)
    LOGS+=("$LOG")
done
echo
echo "Waiting for shards..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "  shard $i (pid ${PIDS[$i]}) OK"
    else
        echo "  shard $i (pid ${PIDS[$i]}) FAILED — see ${LOGS[$i]}"
        FAILED=1
    fi
done
[ $FAILED -eq 0 ] && echo "All shards done." || { echo "Some shards failed."; exit 1; }
