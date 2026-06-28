# Multi-Model Support

## Design

Model-specific vLLM parameters (context extension mechanism, EOS token IDs) are declared
once in `config/config.py` as a `ModelProfile` registry keyed by model ID. Profiling
scripts call `PROFILE.context_kwargs(MAX_MODEL_LEN)` and reference `PROFILE.eos_token_ids`
instead of hardcoding values. `PROFILE` is resolved at import time from `MODEL` in
`config/config.env`.

## Adding a New Model

1. Confirm the model's `max_position_embeddings` (check its HuggingFace `config.json`).
2. In `config/config.py`, add an entry to `_PROFILES`:

```python
"vendor/ModelName": ModelProfile(
    native_ctx=<max_position_embeddings>,
    eos_token_ids=[<id1>, <id2>, ...],          # token IDs to suppress early stop
    supports_dynamic_rope=<True|False>,          # True if model supports dynamic RoPE extension
),
```

3. Set `MODEL=vendor/ModelName` in `config/config.env` (and adjust `TENSOR_PARALLEL_SIZE`).
4. Run profiling scripts normally — `PROFILE` resolves automatically.

## ModelProfile Fields

| Field | Type | Description |
|---|---|---|
| `native_ctx` | `int` | Model's native `max_position_embeddings` |
| `eos_token_ids` | `list[int]` | Token IDs to suppress in `logit_bias` (prevents early stop) |
| `supports_dynamic_rope` | `bool` | Whether to add `rope_scaling` when `max_model_len > native_ctx` |

## context_kwargs(max_model_len)

Returns a dict of vLLM `LLM()` kwargs for context length. Splat into `LLM(...)`:

```python
llm = LLM(
    model=MODEL,
    ...,
    **PROFILE.context_kwargs(MAX_MODEL_LEN),
)
```

If `supports_dynamic_rope=True` and `max_model_len > native_ctx`, returns:
```python
{
    "max_model_len": max_model_len,
    "rope_scaling": {"rope_type": "dynamic", "factor": max_model_len / native_ctx},
}
```
Otherwise returns `{"max_model_len": max_model_len}`.

## Registered Models

| Model | native_ctx | eos_token_ids | supports_dynamic_rope |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 32768 | 151643, 151644, 151645 | True |
| `Qwen/Qwen3.6-27B` | 262144 | 151643, 151644, 151645 | False |

## Notes

- `run_interference.py` / `run_interference_kv.py` call vLLM via the HTTP API (string-keyed
  logit_bias required for JSON): `{str(tid): -100 for tid in PROFILE.eos_token_ids}`.
- `vllm serve` in launch scripts (`launch_interference.sh`) does not currently pass
  `--rope-scaling`. For Qwen3-0.6B the native 32K context is sufficient for interference
  experiments. If a future model or experiment needs extended context for serve-based scripts,
  add a model-specific `--rope-scaling` arg in those launchers.
- `MAX_MODEL_LEN` is a per-script constant (workload-specific). The profile provides the
  mechanism; the script controls the target context length.
