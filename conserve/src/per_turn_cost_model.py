"""AMPD wrong-predict cost model for `per_turn_adaptive_disagg_rest`.

When a turn-2+ request is "wrongly" routed to the prefiller, the cost
modelled here is what the decoder would have to wait for the prefiller to
service it under an AMPD-style flow:

    T_total = T_kv_read(lhist)             # decoder -> prefiller, history KV
            + T_incr_pre                   # prefiller does incremental prefill
                                           # with history KV already in place
            + T_kv_write(lincr)            # prefiller -> decoder, new KV
            + T_kv_mgmt(lhist + lincr)     # block lookup / pack / append on
                                           # both sides; non-marginal in
                                           # practice and not provided by
                                           # Dynamo+NIXL out of the box.

Constants come from `paper/figures/section3`:
  - NETWORK_*       from network_overhead_fit.txt
  - PREFILL_CACHE_HIT_FLOOR_MS from cache_cost_table.csv (hit_p50, ~15 ms,
    flat across L; valid because after the KV read the prefiller has the
    full history KV cached, so incremental prefill on a small lincr is
    dominated by the per-step floor).
"""

# Two-regime KV transfer model (PD-disagg network overhead).
NETWORK_LOW_CONST_MS = 4.849                # mean for L <= 1024
NETWORK_KNEE_L = 1024
NETWORK_HIGH_SLOPE_US_PER_TOKEN = 0.985     # us per token
NETWORK_HIGH_OFFSET_MS = 5.27               # ms

# Prefill with prefix-cache hit (cache_cost_table.csv hit_p50 column,
# measured up to L=65536). Two regimes:
#   - L <~ 12k tokens: flat floor ~14.5 ms (cache verify + first-token sample)
#   - L >~ 12k tokens: grows linearly (still cheap per token but the cached
#     context-length scan dominates the floor). Linear fit on L>=16384 gives
#     slope ~0.95 us/token, intercept ~5.6 ms.
# We use this curve at (lhist + lincr) as a proxy for incremental prefill on
# top of a pre-loaded history KV: the new lincr tokens are usually small
# (100-500), so the dominant cost is scanning/attending the cached context,
# not prefilling the new tokens themselves.
PREFILL_HIT_LOW_CONST_MS = 14.5
PREFILL_HIT_KNEE_L = 12000
PREFILL_HIT_HIGH_SLOPE_US_PER_TOKEN = 0.95
PREFILL_HIT_HIGH_OFFSET_MS = 5.6


def prefill_hit_ms(L: int) -> float:
    """Predicted incremental-prefill latency (ms) for context length L,
    assuming history KV is already in place on the worker."""
    if L <= PREFILL_HIT_KNEE_L:
        return PREFILL_HIT_LOW_CONST_MS
    return PREFILL_HIT_HIGH_SLOPE_US_PER_TOKEN * L / 1000.0 + PREFILL_HIT_HIGH_OFFSET_MS

# Per-side KV bookkeeping. Models: identifying blocks to send + packing them
# on the source, then appending them into the destination's running cache.
# Scaled with cache footprint via per-block overhead.
KV_MGMT_BASE_MS = 2.0
KV_MGMT_PER_BLOCK_US = 5.0
KV_BLOCK_SIZE_TOKENS = 16


def network_ms(L: int) -> float:
    """Predicted KV transfer latency (ms) for moving L tokens of KV cache."""
    if L <= NETWORK_KNEE_L:
        return NETWORK_LOW_CONST_MS
    return NETWORK_HIGH_SLOPE_US_PER_TOKEN * L / 1000.0 + NETWORK_HIGH_OFFSET_MS


def kv_mgmt_ms(cumulative_tokens: int) -> float:
    """Bookkeeping cost summed across both sides (decoder packs + prefiller
    appends + later vice-versa). Scales with the number of blocks involved."""
    per_side = (KV_MGMT_BASE_MS
                + KV_MGMT_PER_BLOCK_US * cumulative_tokens
                  / KV_BLOCK_SIZE_TOKENS / 1000.0)
    return 2.0 * per_side


def wrong_predict_disagg_wait_ms(lhist: int, lincr: int) -> float:
    """Total added latency (ms) for one wrong-predict turn-2+ request."""
    return (network_ms(lhist)
            + prefill_hit_ms(lhist + lincr)
            + network_ms(lincr)
            + kv_mgmt_ms(lhist + lincr))
