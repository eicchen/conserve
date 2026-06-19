#!/bin/bash
# Launch 4 parallel shards of the decode-grid sweep, one per GPU.
# Cells are split round-robin by cell_idx mod 4 so each GPU runs ~8-9 cells.
#
# Usage: ./launch_decode_grid.sh            # full sweep
#        ./launch_decode_grid.sh --pilot    # 3-cell smoke test (still split across 4 GPUs)
#
# Outputs interleave into the same OUT_DIR (each cell has a unique cell_idx,
# so per-cell log filenames don't collide). Per-shard logs land in
# run_log_shard{0,1,2,3}.txt.


# Walk up from script location to find the .conserve_root marker.
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
source "$REPO_ROOT/config.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$(which python3)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/paper/figures/section3/output/300W/decode_grid_data}"
# Override GPUS env to control which GPUs to use (space-separated list).
# Default uses all 4. Set GPUS="1 2 3" to skip GPU 0.
GPUS_STR="${GPUS:-0 1 2 3}"
read -r -a GPUS_ARR <<< "$GPUS_STR"
NUM_SHARDS=${#GPUS_ARR[@]}
EXTRA_ARGS="$@"

mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"
echo "Launching $NUM_SHARDS shards on GPUs: ${GPUS_ARR[*]}"
echo

declare -a PIDS=()
for SHARD in "${!GPUS_ARR[@]}"; do
    GPU=${GPUS_ARR[$SHARD]}
    LOG="$OUT_DIR/launcher_gpu${GPU}.log"
    # VLLM_ENABLE_V1_MULTIPROCESSING=0 forces InprocClient so the custom
    # core_log_file kwarg (consumed in-process by `llm_engine.step`) takes
    # effect. Under MPClient the engine subprocess has its own observability
    # config and the log path never reaches it.
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT_DIR/run_decode_grid.py" \
        --out-dir "$OUT_DIR" \
        --shard-idx $SHARD \
        --num-shards $NUM_SHARDS \
        $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  shard $SHARD (GPU $GPU) -> pid $PID, log $LOG"
done
echo

echo "Waiting for shards to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "  shard $i (pid $PID) OK"
    else
        echo "  shard $i (pid $PID) FAILED -- see $OUT_DIR/launcher_gpu${i}.log"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo
    echo "All shards completed."
    echo "Outputs in $OUT_DIR"
else
    echo
    echo "Some shards failed; check launcher logs."
    exit 1
fi
