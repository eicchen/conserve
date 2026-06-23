#!/bin/bash
PIDS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments:
#   $1: profiling mode. One of: "baseline", "no_disagg_oracle", "all_disagg",
#                                "adaptive_disagg_prefiller", "adaptive_disagg_decoders",
#                                "adaptive_disagg_oracle",
#                                "per_turn_adaptive_disagg_decoders"
# Env vars honoured by "per_turn_adaptive_disagg_decoders":
#   WRONG_PRED_PCT      fraction of turn-2+ wrongly routed   (default 0.0)
#   WRONG_PRED_SEED     RNG seed for the wrong-predict picks (default 42)
#   PREFILLER_TRACE_DIR matching adaptive_disagg_prefiller log dir; required
#                       when WRONG_PRED_PCT > 0 for queueing-delay simulation
#                       and to emit the synthetic prefiller trace.
#   $2: number of decoders
#   $3: path to log directory

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

# To switch models, edit config.env at the repo root (or set MODEL= before running).
source "$SCRIPT_DIR/../config.sh"
MAX_NUM_BATCHED_TOKENS=2944

ensure_arg_exists() {
    valid_args=("baseline" "no_disagg_oracle" "all_disagg" "adaptive_disagg_prefiller" "adaptive_disagg_decoders" "adaptive_disagg_decoders_per_turn_kv" "adaptive_disagg_oracle" "per_turn_adaptive_disagg_decoders")
    if [[ ! " ${valid_args[*]} " =~ " $1 " ]]; then
        echo "Error: invalid first argument: $1"
        echo "Valid modes: ${valid_args[*]}"
        exit 1
    fi

    if [[ $# -ne 3 ]]; then
        echo "Error: This mode requires exactly three arguments (got $# arguments)."
        echo "Usage: $0 <mode> <num_decoders> <log_dir>"
        echo "  mode: One of: ${valid_args[*]}"
        echo "  num_decoders: Number of decoder processes to launch"
        echo "  log_dir: Path to log directory"
        exit 1
    fi

    if [[ -z "${PREFILLER_DEVICE_ID:-}" ]]; then
        echo "Error: PREFILLER_DEVICE_ID environment variable is not set."
        exit 1
    fi

    if ! [[ "$PREFILLER_DEVICE_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: PREFILLER_DEVICE_ID must be an integer. Got: '$PREFILLER_DEVICE_ID'"
        exit 1
    fi

    if [[ -z "${DECODER_DEVICE_IDS:-}" ]]; then
        echo "Error: DECODER_DEVICE_IDS environment variable is not set."
        exit 1
    fi

    if ! [[ "$DECODER_DEVICE_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Error: DECODER_DEVICE_IDS must be a comma-separated list of integers. Got: '$DECODER_DEVICE_IDS'"
        exit 1
    fi
}

# Argument: number of decoders
parse_device_ids() {
    IFS=',' read -ra DECODER_DEVICES_ARR <<< "$DECODER_DEVICE_IDS"
    if [ ${#DECODER_DEVICES_ARR[@]} -ne "$1" ]; then
        echo "Error: Expected $1 decoders, but got ${#DECODER_DEVICES_ARR[@]}."
        exit 1
    fi
}

# Build comma-separated decoder host/port strings from DECODER_DEVICES_ARR.
# Sets globals DECODER_HOSTS and DECODER_PORTS.
build_decoder_args() {
    local hosts="" ports=""
    for device in "${DECODER_DEVICES_ARR[@]}"; do
        hosts="${hosts:+$hosts,}localhost"
        ports="${ports:+$ports,}$((7199 + device))"
    done
    DECODER_HOSTS="$hosts"
    DECODER_PORTS="$ports"
}

# Argument: number of gpus needed
check_num_gpus() {
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt $1 ]; then
        echo "You need at least $1 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    python -c "import $1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ "$1" == "nixl" ]; then
            echo "$1 is not installed. Please refer to https://github.com/ai-dynamo/nixl for installation."
        else
            echo "$1 is not installed. Please install it via pip install $1."
        fi
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1 EXIT   # prevent re-entrancy

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            pkill -TERM -P "$pid"
            kill "$pid" 2>/dev/null
        fi
    done

    sleep 2

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            pkill -P "$pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    kill -- -$$ 2>/dev/null

    echo "All processes stopped."
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=1200
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server"
      return 1
    fi

    sleep 1
  done
}

# Used by all_disagg and adaptive_disagg_prefiller (PD disaggregation via proxy).
launch_disagg_proxy_engines() {
    local log_dir=$1
    local num_decoders=$2

    python3 common/disagg_proxy_server.py \
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
        --num-decoders "$num_decoders" \
        > >(tee logs/proxy.log) 2>&1 &
    PIDS+=($!)

    for device in "${DECODER_DEVICES_ARR[@]}"; do
        echo "Decoder device: $device"
        LOG_DIR="$log_dir" DECODER_DEVICE_ID="$device" bash common/disagg_vllm_launcher.sh "decoder${device}" "$MODEL" \
            > >(tee "logs/decoder${device}.log") 2>&1 &
        PIDS+=($!)
    done

    LOG_DIR="$log_dir" bash common/disagg_vllm_launcher.sh prefiller "$MODEL" \
        > >(tee logs/prefiller.log) 2>&1 &
    PIDS+=($!)

    for device in "${DECODER_DEVICES_ARR[@]}"; do
        wait_for_server $((7199 + device))
    done
    wait_for_server 7100
    wait_for_server 9101

    echo "==================================================="
    echo "All servers are up. You can send requests now..."
    echo "==================================================="
}

# Used by no_disagg_oracle, adaptive_disagg_decoders, adaptive_disagg_oracle (direct
# vLLM engines with prefix caching).
launch_engines() {
    local log_dir=$1

    CUDA_VISIBLE_DEVICES=${PREFILLER_DEVICE_ID} \
        vllm serve "$MODEL" \
        --port 7100 \
        --dtype auto \
        --trust-remote-code \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-num-seqs 1024 \
        --enforce-eager \
        --engine-log-file "$log_dir/prefiller_vllm_engine_log.jsonl" \
        --core-log-file   "$log_dir/prefiller_vllm_core_log.jsonl" \
        > >(tee logs/prefiller.log) 2>&1 &
    PIDS+=($!)

    for device in "${DECODER_DEVICES_ARR[@]}"; do
        echo "Decoder device: $device"
        CUDA_VISIBLE_DEVICES="$device" \
            vllm serve "$MODEL" \
            --port $((7199 + device)) \
            --dtype auto \
            --trust-remote-code \
            --download-dir "$MODEL_DIR" \
            --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --max-num-seqs 1024 \
            --enforce-eager \
            --engine-log-file "$log_dir/decoder${device}_vllm_engine_log.jsonl" \
            --core-log-file   "$log_dir/decoder${device}_vllm_core_log.jsonl" \
            > >(tee "logs/decoder${device}.log") 2>&1 &
        PIDS+=($!)
    done

    for device in "${DECODER_DEVICES_ARR[@]}"; do
        wait_for_server $((7199 + device))
    done
    wait_for_server 7100

    echo "==================================================="
    echo "All servers are up. You can send requests now..."
    echo "==================================================="
}

main() {
    ensure_arg_exists "$@"
    parse_device_ids "$2"
    build_decoder_args
    check_num_gpus "$(( $2 + 1 ))"
    ensure_python_library_installed lmcache
    ensure_python_library_installed nixl
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM
    trap cleanup EXIT

    mkdir -p logs

    local mode=$1
    local num_decoders=$2
    local log_dir=$3

    if [[ "$mode" == "all_disagg" || "$mode" == "adaptive_disagg_prefiller" ]]; then
        launch_disagg_proxy_engines "$log_dir" "$num_decoders"
        python3 src/main.py \
            --baseline "$mode" \
            --num-decoders "$num_decoders" \
            --decoder-host "$DECODER_HOSTS" \
            --decoder-port "$DECODER_PORTS" \
            --proxy-host localhost \
            --proxy-port 9101 \
            ${MAX_ITERS:+--max-iters $MAX_ITERS} \
            ${RPS:+--rps $RPS} \
            ${ARRIVAL_TRACE:+--arrival-trace $ARRIVAL_TRACE} \
            ${ORDER_SEED:+--order-seed $ORDER_SEED} \
            --output-dir "$log_dir" \
            --model "$MODEL"
    else
        # no_disagg_oracle, adaptive_disagg_decoders, adaptive_disagg_oracle,
        # per_turn_adaptive_disagg_decoders, baseline
        launch_engines "$log_dir"
        python3 src/main.py \
            --baseline "$mode" \
            --num-decoders "$num_decoders" \
            --decoder-host "$DECODER_HOSTS" \
            --decoder-port "$DECODER_PORTS" \
            --prefiller-host localhost \
            --prefiller-port 7100 \
            --num-prefillers "${NUM_PREFILLERS:-1}" \
            ${MAX_ITERS:+--max-iters $MAX_ITERS} \
            ${RPS:+--rps $RPS} \
            ${ARRIVAL_TRACE:+--arrival-trace $ARRIVAL_TRACE} \
            ${WRONG_PRED_PCT:+--wrong-pred-pct $WRONG_PRED_PCT} \
            ${WRONG_PRED_SEED:+--wrong-pred-seed $WRONG_PRED_SEED} \
            ${PREFILLER_TRACE_DIR:+--prefiller-trace-dir $PREFILLER_TRACE_DIR} \
            --output-dir "$log_dir" \
            --model "$MODEL"
    fi
}

main "$@"
