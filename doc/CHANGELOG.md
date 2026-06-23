# Changelog

## Centralized `MODEL` in `profiling/paths.py`

`MODEL = "Qwen/Qwen3-0.6B"` was duplicated in all 7 profiling scripts. Moved to
`profiling/paths.py` as `MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")` (env-var
override supported).

- `run_cache_cost.py`, `run_prefill_profile.py`, `run_decode_grid.py`, `run_interference.py`,
  `run_interference_kv.py` — import `MODEL` from `paths`; local definition replaced with a
  commented-out override line
- `prefill_profile.py` — imports `MODEL`, assigns `MODEL_PATH = MODEL`
- `decode_profile.py` — imports `MODEL`, assigns `MODEL_PATH = MODEL`;
  `MODEL_SUFFIX` now derived as `MODEL_PATH.split("/")[-1]` (was hardcoded `"Qwen3-0.6B"`)

## `run_decode_grid.py`: suppress NCCL destroy_process_group warning

Added `torch.distributed.destroy_process_group()` call in the `finally` block of
`profiling/run_decode_grid.py`. Without it the NCCL C++ destructor emits a warning
to stderr after Python exits, which produces a non-zero exit code and causes
`launch_decode_grid.sh` to falsely report a shard as FAILED even when all cells
completed successfully.

## vLLM patch expanded: `fixed_batches` + per-cell JSONL logging

Ported all custom patches from the original reference environment
(`/data/projects/jerry/conda/envs/agent-scaling/`) into the `conserve` conda env's
vLLM 0.11.0 install.  Nine files were overwritten wholesale.  The critical gap was
`entrypoints/llm.py`, which caused every cell in `fig4_decode_grid.ipynb` to fail
with `LLM.generate() got an unexpected keyword argument 'fixed_batches'`.

### Files updated

| File | What changed |
|---|---|
| `vllm/entrypoints/llm.py` | **New** — `fixed_batches`, `engine_log_file`, `core_log_file` params on `generate()`; `_run_engine_with_fixed_batch()` method; `using_customized_engine` / `trace_df` / `start_timestamp` state vars; `datetime`, `json`, `os` imports added |
| `vllm/v1/engine/llm_engine.py` | `step()` changed to `step(*args, **kwargs)` so `core_log_file` threads down the call chain |
| `vllm/v1/engine/core_client.py` | `InprocClient.get_output()` and `SyncMPClient.get_output()` changed to accept `*args, **kwargs`; `engine_log_file` / `core_log_file` forwarded into `make_async_mp_client` |
| `vllm/v1/engine/core.py` | `step()` accepts optional `core_log_file` kwarg overriding `self.core_log_file` at call time; init-time setup simplified; removed debug prints |
| `vllm/v1/engine/async_llm.py` | `generate()` accepts `engine_log_file` / `core_log_file` and forwards them to `make_async_mp_client`; `request_start` logging uses `with open(...)` instead of keeping a file handle |
| `vllm/entrypoints/openai/api_server.py` | `engine_log_file` / `core_log_file` from `engine_args` forwarded to `make_async_mp_client` |
| `vllm/engine/arg_utils.py` | Default values for `--engine-log-file` / `--core-log-file` restored to hardcoded AgentScaling paths (reference convention; `None` defaults dropped) |
| `vllm/v1/core/sched/scheduler.py` | Debug `print` statements from reference re-introduced (`Scheduler config`, `New request tokens`) |
| `vllm/v1/worker/gpu_model_runner.py` | Decoder-only forward-start-time CSV logging added (writes to `/data/projects/AgentScaling/gpu_monitoring/decoder_forward_start_time.csv` when KV-consumer mode is active) |

### How `fixed_batches` works

`llm.generate(fixed_batches=[[prompt]*B]*K, engine_log_file=..., core_log_file=...)`
bypasses the normal batch-all-at-once path and instead runs `_run_engine_with_fixed_batch()`,
which feeds each of the K batches into the engine only after the previous batch has fully
finished.  This gives exact control over the active batch size during decode.
`core_log_file` is threaded down to `EngineCore.step()` on each scheduler tick so each
decode-grid cell writes its own JSONL file.

---

## Section 3 figure notebooks
- Added `paper/figures/section3/notebooks/fig{1..8}_*.ipynb` — one notebook per figure covering all 8 section-3 figures
- Each notebook: markdown call-order cell → optional prereq data-collection cell (separate, skippable) → `%run` plotting cell → inline display cell
- Figs 3 and 6 have no prereq cell (data lives in `GPU_MON_ROOT`)
- Figs 7 and 8 prereq cells document 200W data collection via `--out`/`--dir` args on `run_cache_cost.py` / `plot_cache_cost.py`

## Path fixes — removed hardcoded env paths
All scripts now resolve the Python interpreter from the active conda environment
instead of hardcoding a user-specific path. `conda activate conserve` is all
that's needed before running any script.
- `profiling/launch_decode_grid.sh` — `PY=$(which python3)`
- `profiling/launch_prefill_profile.sh` — `PY=$(which python3)`
- `profiling/launch_interference.sh` — `PATH=$(dirname $(which python3)):$PATH`
- `profiling/run_interference.py` — `env["PATH"] = os.path.dirname(sys.executable) + ...`
- `profiling/run_interference_kv.py` — same as above; added `import sys`
- `conserve/run_sweep.sh` — `PATH=$(dirname $(which python3)):$PATH`
- `conserve/rerun_cell.sh` — `PATH=$(dirname $(which python3)):$PATH`

### Centralized path config
All `AgentScaling/` paths moved to two global config files — override with env vars, no per-file edits needed for new users.
- **`profiling/paths.py`** — `MODEL_DIR`, `PROFILING_DATA_DIR`, `GPU_MON_ROOT`; all profiling Python scripts import from here
- **`config.sh`** (repo root) — same three vars as shell exports; all launcher scripts source it after setting `REPO_ROOT`
- `profiling/run_{cache_cost,decode_grid,interference,interference_kv,prefill_profile}.py` — replaced inline paths with imported constants
- `profiling/{decode_profile,prefill_profile}.py` — replaced inline paths with imported constants
- `paper/figures/section3/scripts/plot_{decode_step_drift,decode_flat,decode_knee,network_overhead}.py` — `GPU_MON_ROOT` now imported via `sys.path` insert to `profiling/`
- `conserve/profile_1pxd.sh` — sources `config.sh` (replaces hardcoded `MODEL_DIR`)
- `conserve/common/disagg_vllm_launcher.sh` — 5× `--download-dir` now uses `$MODEL_DIR`
- `profiling/launch_{decode_grid,interference}.sh` — source `config.sh` after `REPO_ROOT`
- `profiling/launch_prefill_profile.sh` — sources `config.sh` via `SCRIPT_DIR`

---

### Environment / Setup
- Installed `lmcache==0.4.7` in `conserve` conda env; `nixl==1.3.0` pulled in as dependency (no manual git-clone needed)
- Re-pinned `transformers==4.57.1` after lmcache upgraded it to 5.x (breaks vLLM 0.11.0)
- Confirmed vLLM patch applied to all 4 files in local `conserve` env: `arg_utils.py`, `async_llm.py`, `core.py`, `core_client.py`

### doc/README_updated.md
- Added ordering note: profiling before sweeps
- Moved Section-3 microbenchmarks subsection before sweep subsections to reflect correct order
- Expanded sweep subsection to show internal ordering: prefiller → extract traces → replay policies (baseline independent)
- Fixed wrong power cap in prefiller order-sweep example: `p200_d200` → `p300_d300`
- Setup section: removed manual `git clone nixl` step; added transformers re-pin command
- "One experiment cell" section: added `PREFILLER_DEVICE_ID` and `DECODER_DEVICE_IDS` to the example command and env-var list (both are required but were missing from the original docs)

### doc/WORKFLOW.md
- Part 4: added "Entry points" table explaining `profile_1pxd.sh` / `run_sweep.sh` / `rerun_cell.sh` and when to use each
- Part 4: added "Quick test (one cell)" subsection with a full `profile_1pxd.sh` example including required env vars (`PREFILLER_DEVICE_ID`, `DECODER_DEVICE_IDS`)
- "How data collection actually works" section: cross-referenced the new Entry points section for env var details
- Fixed "four phases" → "five parts" in intro (the doc has 5 numbered Parts)
- Fixed Step 2.1 `--download-dir` path: was `/data/projects/eicchen/hf_cache`, corrected to `/data/projects/AgentScaling/models` (where model weights live)
- Fixed Step 2.2/2.3 working directory: added `cd` comment clarifying inner `conserve/` vs repo root
- Fixed Part 3 merge commands: removed `merge_decode_grid_shards.py` (script does not exist; only `merge_cache_cost_shards.py` and `merge_interference_shards.py` are present)
- Step 4.2 output path comment: clarified that `perfiller_p300` (typo spelling) is correct — it matches `SEED_TRACES_BASE` hardcoded in `run_sweep.sh` line 92
- Added `run_sweep.sh` parameter reference tables (positional args and env var overrides) derived from the script source
- Fixed "three things in parallel" in mechanics section: replaced with accurate three sequential stages; clarified that dcgmi is started by `main.py` (not `profile_1pxd.sh`) after vLLM servers pass health checks

---

## Missing files / documentation gaps

Items required to run the project that are not yet documented or present in this fork:

- **`PROFILING_DATA_DIR/prompts_*x2048.json`** (~2.5 GB) — pre-tokenized prompt files for each prefill length (128–65536 tokens). Required by `profiling/run_prefill_profile.py`. Stored at `/data/projects/AgentScaling/data/profiling/` on bbq (shared with original repo, no copy needed). Regeneration script: `profiling/generate_long_prompts.py`. Added entry to `doc/README_updated.md` Input Data table.
- **`conserve/input/mini_swe_agent_trace.json`** (~162 MB) — not generated or present in this fork. Requires running `mini_agent_test.py` + notebook cells against a live vLLM instance. No automation script.
- **`conserve/input/compound_prompts.json`** (~1.5 MB) — not generated or present in this fork. Regenerate with `conserve/src/prepare_compound_prompts.py` after `mini_swe_agent_trace.json` exists.
- **vLLM patch** — applied manually to the `conserve` conda env; no patch file or apply script committed to the repo. A new environment would require re-applying by hand against `vllm==0.11.0`. Four files changed: `vllm/engine/arg_utils.py` (added `--engine-log-file` / `--core-log-file` CLI flags), `vllm/v1/engine/core_client.py` (propagates flags to observability config), `vllm/v1/engine/async_llm.py` (emits `request_start` JSONL on `generate()`), `vllm/v1/engine/core.py` (emits `step_start` / `step_end` JSONL on every scheduler step). All changes marked `# NOTE(Jerry)`. See `doc/README_updated.md § vLLM Modifications` for full detail.
- **`paper/figures/section3/output/`** — all profiling run outputs are gitignored; no instructions for re-running the full Section 3 pipeline end-to-end (prefill profile → decode grid → interference → plots) in one command.
