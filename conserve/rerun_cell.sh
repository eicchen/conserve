#!/usr/bin/env bash
# Re-run a single experiment cell (one policy × cap × rps) multiple times
# into trial_* subdirectories. Useful for:
#   * variance characterization at a specific operating point
#   * picking the "median trial" or shortest-span trial to swap back into the
#     canonical output tree (output/rps_sweep/<cap>/<policy>/rps_<rps>/)
#   * sweeping AMPD's WRONG_PRED_PCT at a fixed cell (set the env var on the
#     invocation; one rerun_cell.sh call per pct value)
#
# Usage:
#   ./rerun_cell.sh <policy> <cap> <rps> [trials]
#
# Arguments:
#   policy   profile_1pxd.sh policy name. One of (see run_sweep.sh header):
#              no_disagg_oracle | all_disagg
#              adaptive_disagg_decoders | adaptive_disagg_decoders_per_turn_kv
#              per_turn_adaptive_disagg_decoders
#
#   cap      p300_d300 | p300_d200 | p200_d200
#
#   rps      arrival rate (matches a folder name under
#            output/rps_sweep/prefiller_p300/rps_<rps>/).
#            Saturation operating point is 1.634, which replays rps_2.
#
#   trials   number of trial dirs to produce. Default 3. (Env var TRIALS
#            takes precedence; CLI arg overrides env if both are set.)
#
# Env vars (optional):
#   WRONG_PRED_PCT      AMPD wrong-predict rate (default 0.10 for AMPD)
#   WRONG_PRED_SEED     AMPD wrong-predict seed (default 42)
#   POLICY_TAG          override the policy-name segment in the output path.
#                       Useful when running two variants of the same policy
#                       (e.g., POLICY_TAG=adaptive_3eng_per_turn_kv to put
#                       per-turn-KV trials next to the oracle-KV trials).
#   NUM_DECODERS        default 3
#
# Output layout:
#   output/var_check/<cap>/rps_<rps>/<policy_tag>/trial_<N>/
#
# Resumable: any trial whose per_step_latency.csv already exists is skipped.
#
# Pre-req: GPU power caps already applied. Script verifies them on launch.
#
# Examples:
#   ./rerun_cell.sh adaptive_disagg_decoders p300_d300 1.634 5
#   TRIALS=5 ./rerun_cell.sh per_turn_adaptive_disagg_decoders p300_d300 1.634
#   POLICY_TAG=adaptive_3eng_per_turn_kv \
#       ./rerun_cell.sh adaptive_disagg_decoders_per_turn_kv p300_d200 1.634 3
#   for p in 0.05 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
#       WRONG_PRED_PCT=$p ./rerun_cell.sh per_turn_adaptive_disagg_decoders \
#                                          p300_d300 1.634 1
#   done

set -uo pipefail

POLICY=${1:?Usage: $0 <policy> <cap> <rps> [trials]}
CAP=${2:?Usage: $0 <policy> <cap> <rps> [trials]}
RPS=${3:?Usage: $0 <policy> <cap> <rps> [trials]}
TRIALS=${4:-${TRIALS:-3}}

# Walk up to repo root.
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
PROJECT="$REPO_ROOT/conserve"

export PATH=/data/projects/jerry/conda/envs/agent-scaling/bin:$PATH
export PREFILLER_DEVICE_ID=0
export DECODER_DEVICE_IDS=1,2,3
export MAX_ITERS=${MAX_ITERS:-5}

NUM_DECODERS=${NUM_DECODERS:-3}

case "$CAP" in
    p300_d300) CAP_P=300; CAP_D=300 ;;
    p300_d200) CAP_P=300; CAP_D=200 ;;
    p200_d200) CAP_P=200; CAP_D=200 ;;
    *) echo "Unknown cap '$CAP'. Use: p300_d300 | p300_d200 | p200_d200"; exit 1 ;;
esac

if [[ "$POLICY" == "per_turn_adaptive_disagg_decoders" ]]; then
    export WRONG_PRED_PCT=${WRONG_PRED_PCT:-0.10}
    export WRONG_PRED_SEED=${WRONG_PRED_SEED:-42}
    pct_int=$(awk -v p="$WRONG_PRED_PCT" 'BEGIN{printf "%02d", p*100}')
    DEFAULT_POLICY_TAG="${POLICY}_p${pct_int}"
else
    DEFAULT_POLICY_TAG="$POLICY"
fi
POLICY_TAG=${POLICY_TAG:-$DEFAULT_POLICY_TAG}

# Pick the matched prefiller arrival trace. rps=1.634 replays rps_2.
trace_rps=$RPS
[[ "$RPS" == "1.634" ]] && trace_rps=2
PREF_TRACES_BASE="$PROJECT/output/rps_sweep/prefiller_p300"
TRACE_DIR="$PREF_TRACES_BASE/rps_${trace_rps}"

case "$POLICY" in
    no_disagg_oracle|all_disagg)
        export ARRIVAL_TRACE="$TRACE_DIR/iter0_prefill_start_arrival_trace.json" ;;
    *)
        export ARRIVAL_TRACE="$TRACE_DIR/iter1_decoding_start_arrival_trace.json" ;;
esac
if [[ "$POLICY" == "per_turn_adaptive_disagg_decoders" ]]; then
    export PREFILLER_TRACE_DIR="$TRACE_DIR"
fi
[[ -f "$ARRIVAL_TRACE" ]] || { echo "FATAL: missing arrival trace $ARRIVAL_TRACE"; exit 1; }

OUT_BASE="$PROJECT/output/var_check/$CAP/rps_${RPS}/$POLICY_TAG"
cd "$PROJECT"
mkdir -p "$OUT_BASE"

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
hr() { echo; echo "=========== $(ts) — $* ==========="; }

verify_caps() {
    local ok=1
    while IFS=', ' read -r idx lim; do
        local want=$CAP_D
        [[ "$idx" == "0" ]] && want=$CAP_P
        if ! awk -v l="$lim" -v w="$want" 'BEGIN{exit !(l>=w-1 && l<=w+1)}'; then
            echo "  GPU$idx power limit = ${lim}W, expected ${want}W"
            ok=0
        fi
    done < <(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits)
    if [[ $ok -ne 1 ]]; then
        echo "FATAL: power caps don't match '$CAP' (need GPU0=${CAP_P}W, GPU1-3=${CAP_D}W)."
        exit 1
    fi
    echo "Power caps verified (GPU0=${CAP_P}W, GPU1-3=${CAP_D}W)."
}

echo "POLICY=$POLICY  POLICY_TAG=$POLICY_TAG"
echo "CAP=$CAP  RPS=$RPS  TRIALS=$TRIALS"
[[ -n "${WRONG_PRED_PCT:-}" ]] && echo "WRONG_PRED_PCT=$WRONG_PRED_PCT"
echo "ARRIVAL_TRACE=$ARRIVAL_TRACE"
echo "OUT_BASE=$OUT_BASE"
verify_caps

for t in $(seq 0 $((TRIALS-1))); do
    LOG_DIR="$OUT_BASE/trial_$t"
    if [[ -f "$LOG_DIR/per_step_latency.csv" ]]; then
        echo "  SKIP (exists): $LOG_DIR"
        continue
    fi
    mkdir -p "$LOG_DIR"
    hr "[trial $t] $POLICY @ $CAP rps=$RPS → $LOG_DIR"
    ./profile_1pxd.sh "$POLICY" "$NUM_DECODERS" "$LOG_DIR" > "$LOG_DIR/run.log" 2>&1
    sleep 10
done

unset ARRIVAL_TRACE PREFILLER_TRACE_DIR

hr "DONE — $TRIALS trial(s) at $OUT_BASE"
