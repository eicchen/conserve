# Standalone copy of config/config.sh for the fig1 sbatch pipeline.
#
# Differs from the canonical config/config.sh in three ways:
#   1. REPO_ROOT is found by walking up to the .conserve_root marker rather
#      than assuming this file lives directly under <repo>/config/.
#   2. CONSERVE_CONFIG_ENV can point this at an isolated config.env copy
#      instead of the shared repo one (see config.py in this directory) —
#      run_fig1_trace.sbatch uses this so concurrently submitted jobs
#      (different BENCHMARK each) never race on the same file.
#   3. The VLLM_SERVE_FLAGS eval calls the local config.py copy in this
#      directory instead of config/config.py, so the two stay consistent.
#
# The canonical config/config.sh is untouched; this copy is intentionally
# kept in sync by hand for the sbatch pipeline only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SCRIPT_DIR"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"

CONFIG_ENV_FILE="${CONSERVE_CONFIG_ENV:-$REPO_ROOT/config/config.env}"

if [[ -f "$CONFIG_ENV_FILE" ]]; then
    while IFS='=' read -r key val; do
        # Skip blank lines and comment lines
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        key="${key// /}"  # strip spaces from key
        # config.env is primary — always overwrite any inherited env var
        printf -v "$key" '%s' "$val"
    done < "$CONFIG_ENV_FILE"
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

# Load model-specific vLLM serve flags from model_specific_configs.toml via
# the local config.py copy (honors CONSERVE_CONFIG_ENV, unlike config/config.py).
eval "$(python "$SCRIPT_DIR/config.py" --sh-vars)"
