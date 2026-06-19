#!/usr/bin/env bash
# Generic experiment sweep driver. Loops profile_1pxd.sh over a chosen
# (policy, power-cap, mode) configuration. Replaces the per-experiment
# overnight_*.sh / run_all_disagg_*.sh scripts.
#
# Usage:
#   ./run_sweep.sh <policy> <cap> <mode>
#
# Arguments:
#   policy   profile_1pxd.sh policy name. One of:
#              baseline                         - per-conv standalone runs
#              no_disagg_oracle                 - Collocated
#              all_disagg                       - Full Disagg
#              adaptive_disagg_prefiller        - prefiller-side trace gen
#              adaptive_disagg_decoders         - ConServe (oracle KV)
#              adaptive_disagg_decoders_per_turn_kv  - ConServe (per-turn KV)
#              per_turn_adaptive_disagg_decoders     - AMPD
#
#   cap      Power-cap configuration. One of:
#              p300_d300   GPU0 @ 300W, GPU1-3 @ 300W   (uncapped baseline)
#              p300_d200   GPU0 @ 300W, GPU1-3 @ 200W   (decoder cap)
#              p200_d200   all GPUs @ 200W              (full cap)
#
#   mode     What to sweep. One of:
#              rps        6-point RPS sweep (matches paper config)
#              order      10-seed order sweep at saturation RPS=2
#              both       rps then order
#
# Env vars (optional):
#   RPS_LIST            override default 6-point: "0.5 0.75 1 1.25 1.5 1.634"
#   SEED_LIST           override default 10-seed: "0 1 ... 9"
#   WRONG_PRED_PCT      AMPD wrong-predict rate (default 0.10 for AMPD)
#   WRONG_PRED_SEED     AMPD wrong-predict seed (default 42)
#   EXTRACT_TRACES      =1 → run extract_arrival_traces.py on every output
#                        seed dir (only meaningful with
#                        adaptive_disagg_prefiller / order mode)
#   NUM_DECODERS        decoder count for profile_1pxd.sh (default 3)
#
# Pre-req: GPU power caps already applied. The script verifies them and
# aborts on mismatch.
#
# Examples:
#   ./run_sweep.sh per_turn_adaptive_disagg_decoders p300_d300 both
#   ./run_sweep.sh baseline p200_d200 order
#   ./run_sweep.sh all_disagg p300_d200 rps
#   EXTRACT_TRACES=1 ./run_sweep.sh adaptive_disagg_prefiller p200_d200 order

set -uo pipefail

POLICY=${1:?Usage: $0 <policy> <cap> <mode>}
CAP=${2:?Usage: $0 <policy> <cap> <mode>}
MODE=${3:?Usage: $0 <policy> <cap> <mode>}

# Walk up to repo root.
_d="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
while [ "$_d" != "/" ] && [ ! -e "$_d/.conserve_root" ]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"
PROJECT="$REPO_ROOT/conserve"

export PATH="$(dirname "$(which python3)"):$PATH"
export PREFILLER_DEVICE_ID=0
export DECODER_DEVICE_IDS=1,2,3
export MAX_ITERS=${MAX_ITERS:-5}

NUM_DECODERS=${NUM_DECODERS:-3}
PYTHON="${PYTHON:-$(command -v python3)}"

# Resolve (CAP_P, CAP_D) from cap name.
case "$CAP" in
    p300_d300) CAP_P=300; CAP_D=300 ;;
    p300_d200) CAP_P=300; CAP_D=200 ;;
    p200_d200) CAP_P=200; CAP_D=200 ;;
    *) echo "Unknown cap '$CAP'. Use: p300_d300 | p300_d200 | p200_d200"; exit 1 ;;
esac

# Defaults for the sweep ranges.
RPS_LIST=${RPS_LIST:-"0.5 0.75 1 1.25 1.5 1.634"}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4 5 6 7 8 9"}

# AMPD defaults — only used when policy is per_turn_adaptive_disagg_decoders.
if [[ "$POLICY" == "per_turn_adaptive_disagg_decoders" ]]; then
    export WRONG_PRED_PCT=${WRONG_PRED_PCT:-0.10}
    export WRONG_PRED_SEED=${WRONG_PRED_SEED:-42}
    pct_int=$(awk -v p="$WRONG_PRED_PCT" 'BEGIN{printf "%02d", p*100}')
    POLICY_SUFFIX="_p${pct_int}"
else
    POLICY_SUFFIX=""
fi

# Where each kind of output lives.
PREF_TRACES_BASE="$PROJECT/output/rps_sweep/prefiller_p300"
SEED_TRACES_BASE="$PROJECT/output/order_sweep/perfiller_p300"   # note: legacy "perfiller" spelling

if [[ "$POLICY" == "baseline" ]]; then
    SHORTCAP="${CAP%%_*}"                                          # p300 / p200
    OUT_RPS=""                                                     # baseline has no rps sweep
    OUT_ORDER="$PROJECT/output/baseline/${SHORTCAP}"
elif [[ "$POLICY" == "adaptive_disagg_prefiller" ]]; then
    OUT_RPS="$PROJECT/output/rps_sweep/prefiller_${CAP%%_*}"        # rps_sweep/prefiller_p300
    OUT_ORDER="$PROJECT/output/order_sweep/perfiller_${CAP%%_*}"    # order_sweep/perfiller_p300
else
    OUT_RPS="$PROJECT/output/rps_sweep/${CAP}/${POLICY}${POLICY_SUFFIX}"
    OUT_ORDER="$PROJECT/output/order_sweep/${CAP}/${POLICY}${POLICY_SUFFIX}"
fi

cd "$PROJECT"
[[ -n "$OUT_RPS"   ]] && mkdir -p "$OUT_RPS"
[[ -n "$OUT_ORDER" ]] && mkdir -p "$OUT_ORDER"

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
        echo "Apply with:"
        echo "  sudo nvidia-smi -i 0 -pl ${CAP_P}"
        echo "  sudo nvidia-smi -i 1,2,3 -pl ${CAP_D}"
        exit 1
    fi
    echo "Power caps verified (GPU0=${CAP_P}W, GPU1-3=${CAP_D}W)."
}

# Run one (policy, log_dir) cell, skipping if already complete.
run_one() {
    local log_dir=$1
    if [[ -f "$log_dir/per_step_latency.csv" ]]; then
        echo "  SKIP (exists): $log_dir"
        return
    fi
    mkdir -p "$log_dir"
    hr "→ $log_dir"
    ./profile_1pxd.sh "$POLICY" "$NUM_DECODERS" "$log_dir" > "$log_dir/run.log" 2>&1
    sleep 10
}

# Configure ARRIVAL_TRACE / PREFILLER_TRACE_DIR before each call to run_one().
# This depends on the policy: prefiller-policy generates its own arrivals via
# RPS; replay policies read from prefiller traces.
setup_replay_env_rps() {
    local rps=$1
    local trace_rps=$rps
    [[ "$rps" == "1.634" ]] && trace_rps=2

    if [[ "$POLICY" == "adaptive_disagg_prefiller" ]]; then
        # Prefiller generates its own arrivals from a fixed RPS.
        unset ARRIVAL_TRACE PREFILLER_TRACE_DIR
        export RPS=$rps
        return
    fi

    # Replay policies pick which arrival trace to use based on whether they
    # are "colocated" (read iter0 arrivals — happens to be the prefill start)
    # or "decoder-side" (read iter1 arrivals — when the decoder gets work).
    local arr_file
    case "$POLICY" in
        no_disagg_oracle|all_disagg)
            arr_file="iter0_prefill_start_arrival_trace.json" ;;
        *)
            arr_file="iter1_decoding_start_arrival_trace.json" ;;
    esac
    export ARRIVAL_TRACE="$PREF_TRACES_BASE/rps_${trace_rps}/${arr_file}"
    unset RPS
    # AMPD also needs the matched prefiller trace dir for VirtualPrefiller.
    if [[ "$POLICY" == "per_turn_adaptive_disagg_decoders" ]]; then
        export PREFILLER_TRACE_DIR="$PREF_TRACES_BASE/rps_${trace_rps}"
    else
        unset PREFILLER_TRACE_DIR
    fi
}

setup_replay_env_order() {
    local seed=$1
    case "$POLICY" in
        baseline)
            # Baseline doesn't replay an arrival trace; it varies the
            # conversation ordering via ORDER_SEED instead.
            unset ARRIVAL_TRACE PREFILLER_TRACE_DIR RPS
            export ORDER_SEED=$seed ;;
        adaptive_disagg_prefiller)
            # Prefiller generates its own arrivals; ORDER_SEED varies the conv
            # ordering for the per-seed sweep.
            unset ARRIVAL_TRACE PREFILLER_TRACE_DIR
            export ORDER_SEED=$seed
            export RPS=${ORDER_RPS:-2} ;;
        no_disagg_oracle|all_disagg)
            unset ORDER_SEED RPS PREFILLER_TRACE_DIR
            export ARRIVAL_TRACE="$SEED_TRACES_BASE/seed_${seed}/iter0_prefill_start_arrival_trace.json" ;;
        per_turn_adaptive_disagg_decoders)
            unset ORDER_SEED RPS
            export ARRIVAL_TRACE="$SEED_TRACES_BASE/seed_${seed}/iter1_decoding_start_arrival_trace.json"
            export PREFILLER_TRACE_DIR="$SEED_TRACES_BASE/seed_${seed}" ;;
        *)
            unset ORDER_SEED RPS PREFILLER_TRACE_DIR
            export ARRIVAL_TRACE="$SEED_TRACES_BASE/seed_${seed}/iter1_decoding_start_arrival_trace.json" ;;
    esac
}

# ── Validate inputs ──────────────────────────────────────────────────────────
case "$MODE" in
    rps|order|both) ;;
    *) echo "Unknown mode '$MODE'. Use: rps | order | both"; exit 1 ;;
esac

echo "POLICY=$POLICY  CAP=$CAP  MODE=$MODE"
echo "RPS_LIST=\"$RPS_LIST\""
echo "SEED_LIST=\"$SEED_LIST\""
[[ -n "${WRONG_PRED_PCT:-}" ]] && echo "WRONG_PRED_PCT=$WRONG_PRED_PCT  WRONG_PRED_SEED=${WRONG_PRED_SEED:-(unset)}"
echo "OUT_RPS=$OUT_RPS"
echo "OUT_ORDER=$OUT_ORDER"
echo
echo "GPU power limits:"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
verify_caps

# ── Mode: rps ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "rps" || "$MODE" == "both" ]]; then
    if [[ -z "$OUT_RPS" ]]; then
        echo "WARNING: policy '$POLICY' has no rps sweep layout — skipping rps phase."
    else
        hr "RPS sweep: $POLICY × $CAP × { $RPS_LIST }"
        for rps in $RPS_LIST; do
            setup_replay_env_rps "$rps"
            run_one "$OUT_RPS/rps_${rps}"
        done
        unset RPS ARRIVAL_TRACE PREFILLER_TRACE_DIR
    fi
fi

# ── Mode: order ──────────────────────────────────────────────────────────────
if [[ "$MODE" == "order" || "$MODE" == "both" ]]; then
    hr "Order sweep: $POLICY × $CAP × seeds { $SEED_LIST }"
    for seed in $SEED_LIST; do
        setup_replay_env_order "$seed"
        if [[ "$POLICY" == "baseline" ]]; then
            log_dir="$OUT_ORDER/order_seed${seed}"
        else
            log_dir="$OUT_ORDER/seed_${seed}"
        fi
        run_one "$log_dir"

        # After a prefiller run, optionally extract the iter0/iter1 arrival
        # traces that downstream replay policies consume.
        if [[ "$POLICY" == "adaptive_disagg_prefiller" && "${EXTRACT_TRACES:-0}" == "1" ]]; then
            "$PYTHON" common/extract_arrival_traces.py "$log_dir"
        fi
    done
    unset ORDER_SEED RPS ARRIVAL_TRACE PREFILLER_TRACE_DIR
fi

hr "DONE — $POLICY @ $CAP, mode=$MODE"
[[ -n "$OUT_RPS"   ]] && echo "  RPS outputs:    $OUT_RPS/rps_*/"
[[ -n "$OUT_ORDER" ]] && echo "  Order outputs:  $OUT_ORDER/"
