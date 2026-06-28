#!/bin/bash
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
source "$REPO_ROOT/config.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# configs/ lives one level up from common/ (conserve/conserve/configs/)
CONFIGS_DIR="$SCRIPT_DIR/../configs"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
MAX_NUM_BATCHED_TOKENS=2944
# Prefiller may need a larger batch budget when profiling long single prompts;
# overridable via env, defaults preserve the original 8192 used by adaptive_agentic_serving.
PREFILLER_MAX_NUM_BATCHED_TOKENS="${PREFILLER_MAX_NUM_BATCHED_TOKENS:-8192}"
mkdir -p $LOG_DIR

# NOTE: For correct KV cache transfer, ensure all processes use the same PYTHONHASHSEED to keep the hash of the KV cache consistent across processes.
export PYTHONHASHSEED=0

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
else
    echo "Using model: $2"
    MODEL=$2
fi


if [[ $1 == "prefiller" ]]; then
    # Prefiller listens on port 7100
    prefill_config_file=$CONFIGS_DIR/lmcache-prefiller-config.yaml
    echo prefiller device: CUDA_VISIBLE_DEVICES=$(gpu_range ${PREFILLER_DEVICE_ID:-0})
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$prefill_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$(gpu_range ${PREFILLER_DEVICE_ID:-0}) \
        vllm serve $MODEL \
        --port 7100 \
        --trust-remote-code \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens $PREFILLER_MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs 1024 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' \
        --engine-log-file $LOG_DIR/prefiller_vllm_engine_log.jsonl \
        --core-log-file $LOG_DIR/prefiller_vllm_core_log.jsonl \
        --gpu-memory-utilization 0.6


elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 7200
    decode_config_file=$CONFIGS_DIR/lmcache-decoder-config.yaml

    echo decoder device: CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1})
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1}) \
        vllm serve $MODEL \
        --port 7200 \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs 1024 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        --engine-log-file $LOG_DIR/decoder_vllm_engine_log.jsonl \
        --core-log-file $LOG_DIR/decoder_vllm_core_log.jsonl \
        --gpu-memory-utilization 0.8


elif [[ $1 == "decoder1" ]]; then
    # Decoder listens on port 7200
    decode_config_file=$CONFIGS_DIR/lmcache-decoder-1-config.yaml

    echo decoder1 device: CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1})
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1}) \
        vllm serve $MODEL \
        --port 7200 \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs 1024 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        --engine-log-file $LOG_DIR/decoder1_vllm_engine_log.jsonl \
        --core-log-file $LOG_DIR/decoder1_vllm_core_log.jsonl \
        --gpu-memory-utilization 0.8


elif [[ $1 == "decoder2" ]]; then
    # Decoder listens on port 7200
    decode_config_file=$CONFIGS_DIR/lmcache-decoder-2-config.yaml

    echo decoder2 device: CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1})
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1}) \
        vllm serve $MODEL \
        --port 7201 \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs 1024 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        --engine-log-file $LOG_DIR/decoder2_vllm_engine_log.jsonl \
        --core-log-file $LOG_DIR/decoder2_vllm_core_log.jsonl \
        --gpu-memory-utilization 0.8
elif [[ $1 == "decoder3" ]]; then
    # Decoder listens on port 7200
    decode_config_file=$CONFIGS_DIR/lmcache-decoder-3-config.yaml

    echo decoder3 device: CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1})
    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$(gpu_range ${DECODER_DEVICE_ID:-1}) \
        vllm serve $MODEL \
        --port 7202 \
        --download-dir "$MODEL_DIR" \
        --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs 1024 \
        --disable-log-requests \
        --enforce-eager \
        --no-enable-prefix-caching \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-1} \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        --engine-log-file $LOG_DIR/decoder3_vllm_engine_log.jsonl \
        --core-log-file $LOG_DIR/decoder3_vllm_core_log.jsonl \
        --gpu-memory-utilization 0.8
else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
