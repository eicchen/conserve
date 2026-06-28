#!/bin/bash
# Launch PD-disagg profiling sweep for fig3 (network overhead vs input length).
# Runs a unit profile (proxy + prefiller + decoder + profiling script) for each L
# in L_VALUES, skipping L values where output already exists
# (dcgmi_trace.tsv + decoder_forward_start_time.csv).
# Power cap must be 300 W: sudo nvidia-smi -i 0,1 -pl 300

set -euo pipefail

# Walk up from script location to find the .conserve_root marker
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"

source "$REPO_ROOT/config.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_LOGS_DIR="$SCRIPT_DIR/logs"
MODEL_SHORT="${MODEL##*/}"

NORMAL_L_VALUES=(128 256 512 1024 2048 4096 6144 8192)
LONG_L_VALUES=(10240 12288 16384 20480 24576 28672 32768 40960 49152 57344 65536)

if [[ "${INCLUDE_LONG:-0}" == "1" ]]; then
    L_VALUES=("${NORMAL_L_VALUES[@]}" "${LONG_L_VALUES[@]}")
else
    L_VALUES=("${NORMAL_L_VALUES[@]}")
fi

DISAGG_OUT_BASE="${DISAGG_OUT_BASE:-$GPU_MON_ROOT/$MODEL_SHORT/pd_disagg_300W}"

echo "Model:       $MODEL"
echo "Output base: $DISAGG_OUT_BASE"
echo "Normal L:    ${NORMAL_L_VALUES[*]}"
echo "Long L:      ${LONG_L_VALUES[*]}"
echo "Running L:   ${L_VALUES[*]}  (INCLUDE_LONG=${INCLUDE_LONG:-0})"
echo

PIDS=()

kill_all() {
    echo "Stopping all vLLM/proxy processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    # Kill any vllm/proxy processes that escaped the tracked PIDs (e.g. EngineCore workers)
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "disagg_proxy_server" 2>/dev/null || true
    # Belt-and-suspenders: kill anything still holding the serving ports
    lsof -ti :7100 -ti :7200 -ti :7300 -ti :7400 -ti :7500 -ti :9101 2>/dev/null \
        | xargs -r kill -9 2>/dev/null || true
    PIDS=()
    echo "All processes stopped."
}

wait_for_port_free() {
    local port=$1
    local timeout=120
    local start
    start=$(date +%s)
    while lsof -ti tcp:"$port" >/dev/null 2>&1; do
        if (( $(date +%s) - start >= timeout )); then
            echo "WARNING: port $port still in use after ${timeout}s"
            return 1
        fi
        sleep 2
    done
}

wait_for_server() {
    local port=$1
    local timeout_seconds=1200
    local start_time
    start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi
        sleep 1
    done
}

run_unit() {
    local IN_TOKEN_SIZE=$1
    local LOG_DIR="$DISAGG_OUT_BASE/$IN_TOKEN_SIZE"
    echo "MODEL:          $MODEL"
    echo "IN_TOKEN_SIZE:  $IN_TOKEN_SIZE"
    echo "LOG_DIR:        $LOG_DIR"

    mkdir -p "$SCRIPT_LOGS_DIR"

    python3 "$REPO_ROOT/conserve/common/disagg_proxy_server.py" \
        --host localhost \
        --port 9101 \
        --prefiller-host localhost \
        --prefiller-port 7100 \
        --num-prefillers 1 \
        --decoder-host localhost \
        --decoder-port 7200 \
        --decoder-init-port 7300 \
        --decoder-alloc-port 7400 \
        --proxy-host localhost \
        --proxy-port 7500 \
        --num-decoders 1 \
        > >(tee "$SCRIPT_LOGS_DIR/proxy.log") 2>&1 &
    PIDS+=($!)

    LOG_DIR=$LOG_DIR bash "$REPO_ROOT/conserve/common/disagg_vllm_launcher.sh" decoder "$MODEL" \
        > >(tee "$SCRIPT_LOGS_DIR/decoder.log") 2>&1 &
    PIDS+=($!)

    LOG_DIR=$LOG_DIR bash "$REPO_ROOT/conserve/common/disagg_vllm_launcher.sh" prefiller "$MODEL" \
        > >(tee "$SCRIPT_LOGS_DIR/prefiller.log") 2>&1 &
    PIDS+=($!)

    wait_for_server 7200
    wait_for_server 7100
    wait_for_server 9101

    python3 "$REPO_ROOT/profiling/disagg_profile.py" --in-token-size "$IN_TOKEN_SIZE"

    DECODER_CSV="/data/projects/AgentScaling/gpu_monitoring/decoder_forward_start_time.csv"
    if [[ -f "$DECODER_CSV" ]]; then
        mv "$DECODER_CSV" "$LOG_DIR/"
    fi

    kill_all

    # Ensure ports are fully released before returning (so the next unit starts clean)
    for port in 7100 7200 9101; do
        wait_for_port_free "$port" || true
    done
}

trap 'kill_all; exit 1' INT TERM

FAILED=0
for L in "${L_VALUES[@]}"; do
    LOG_DIR_CHECK="$DISAGG_OUT_BASE/$L"
    if [[ -f "$LOG_DIR_CHECK/prefiller_vllm_core_log.jsonl" \
       && -f "$LOG_DIR_CHECK/prefiller_vllm_engine_log.jsonl" \
       && -f "$LOG_DIR_CHECK/decoder_forward_start_time.csv" ]]; then
        echo "L=$L: already complete, skipping"
        continue
    fi
    echo "--- L=$L: starting ---"
    if DISAGG_OUT_BASE="$DISAGG_OUT_BASE" run_unit "$L"; then
        echo "--- L=$L: done ---"
    else
        echo "--- L=$L: FAILED, cleaning up before next iteration ---"
        kill_all
        for port in 7100 7200 9101; do
            wait_for_port_free "$port" || true
        done
        FAILED=1
    fi
    sleep 5
done

[ $FAILED -eq 0 ] && echo "All L values complete." || { echo "Some L values failed."; exit 1; }
