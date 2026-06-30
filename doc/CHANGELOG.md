# Changelog

## 2026-06-29 — Benchmark trace path restructure + fig1 self-contained notebook

Moved `mini_swe_agent_trace.json` from `conserve/input/` into the per-model data tree; added `BENCHMARK` config key; expanded `fig1_trace_profile.ipynb` to cover the full data-collection pipeline inline.

### Trace path restructure
- `mini_swe_agent_trace.json` moved to `model_outputs/<MODEL_SHORT>/benchmark_trace/<BENCHMARK>/trace/`
- `BENCHMARK=SWE-bench_bm25_13K` added to `config/config.env`; `BENCHMARK_TRACE_DIR = MODEL_DATA_DIR / "benchmark_trace" / BENCHMARK / "trace"` added to `config/config.py` (stamped into `os.environ`) and derived in `config/config.sh` (exported)
- All consumers updated to import `BENCHMARK_TRACE_DIR` from `config/config.py`: `conserve/src/input_loader.py`, `src/prepare_compound_prompts.py`, `src/profile_agent.py`, `paper/figures/section3/scripts/plot_trace.py`
- `CLAUDE.md` and `doc/PAPER_TO_CODE.md` updated to reflect new path

### `fig1_trace_profile.ipynb` — self-contained pipeline
- Notebook now covers all four steps inline: (1) download `princeton-nlp/SWE-bench_bm25_13K` via `load_dataset` with stale-cache detection; (2) flatten per-problem JSONs, compute `uncached_tokens`, Gutenberg-pad prompts, save to `BENCHMARK_TRACE_DIR`; (3) verify PUA-anchor prefix-caching defeat; (4) run `plot_trace.py`
- `conserve/output/plot_dataset.ipynb` is now legacy for trace generation
- `doc/PAPER_TO_CODE.md`: expanded Fig 1 data-collection description inline with step-by-step detail

## 2026-06-29 — Config/ folder + multi-model vLLM serve flags

Moved `config.env`, `config.sh`, `profiling/config.py` into `config/` and eliminated all per-model hardcoding from launchers and profiling scripts.

### config/ folder restructure
- `config.env`, `config.sh` moved from repo root; `profiling/config.py` moved to `config/config.py`; `model_profiles.toml` renamed to `model_specific_configs.toml`
- All shell scripts updated to `source "$REPO_ROOT/config/config.sh"`
- All Python profiling scripts: added `sys.path.insert(0, str(REPO_ROOT / "config"))` before importing; updated to use `PROFILE`
- All 13 `paper/figures/section3/scripts/` plot scripts: same `sys.path` fix
- All section3 notebooks (figs 2–8): same `sys.path` fix
- `profiling/long_prompts/generate_long_prompts.py`: same `sys.path` fix; added `SEED_PATH` constant

### Model-specific vLLM serve flags via TOML
- `config/model_specific_configs.toml`: added `vllm_serve_flags` list to each model section — Qwen3-0.6B: `["--rope-scaling", '{"rope_type":"dynamic","factor":2.0}', "--disable-log-requests"]`; Qwen3.6-27B: `["--no-enable-log-requests"]`
- `config/config.py`: `ModelProfile` gains `vllm_serve_flags` field; `--sh-vars` CLI mode emits `VLLM_SERVE_FLAGS=(...)` bash array
- `config/config.sh`: evals `config.py --sh-vars` at source time to set `VLLM_SERVE_FLAGS` array
- `conserve/common/disagg_vllm_launcher.sh`: all 5 role blocks replace hardcoded `--rope-scaling`/`--disable-log-requests` with `"${VLLM_SERVE_FLAGS[@]}"`; log args moved to per-role `_log_args` array; GPU mem util now configurable via `PREFILLER_GPU_MEM_UTIL`/`DECODER_GPU_MEM_UTIL`
- `conserve/profile_1pxd.sh launch_engines()`: same `VLLM_SERVE_FLAGS` replacement
- `profiling/run_interference.py`, `run_interference_kv.py`: `*PROFILE.vllm_serve_flags` in cmd list; `logit_bias` uses `PROFILE.eos_token_ids`

### All profiling Python scripts — hardcoded values removed
- `rope_scaling` now conditional: only added when `MAX_MODEL_LEN > PROFILE.native_ctx`
- `logit_bias` uses `{tid: -100 for tid in PROFILE.eos_token_ids}` (string keys in HTTP-API scripts)
- Affected: `decode_profile.py`, `prefill_profile.py`, `disagg_profile.py`, `run_prefill_profile.py`, `run_decode_grid.py`, `run_cache_cost.py`

### `doc/technical/` (new folder)
- `tensor_parallelism.md`, `vllm_changes.md`, `WORKFLOW_Qwen3-0.6B.md` moved here from `doc/`; `multi_model.md` added

## TP GPU device assignment — Part 2 (launcher gpu_range wiring + notebook validation cells)

> **Superseded by "Model-specific vLLM serve flags" entry above.** The `gpu_range()` approach was removed during the config/ refactor; a replacement mechanism for `CUDA_VISIBLE_DEVICES` expansion at TP>1 is TBD. The items that remain in effect from this entry are: root-walker + `source config/config.sh` added to `disagg_vllm_launcher.sh`; `check_num_gpus` scaled by `TENSOR_PARALLEL_SIZE` in `profile_1pxd.sh`; config/validation cells added to section 3 notebooks (Figs 4, 5, 6). See `doc/technical/tensor_parallelism.md` for the TBD section and the exact sites to update.

- **`config.sh`**: added `gpu_range()` function — **removed in subsequent refactor**.
- **`conserve/common/disagg_vllm_launcher.sh`**: added root-walker + `source config.sh` ✓ (still in effect, path updated to `config/config.sh`); replaced `CUDA_VISIBLE_DEVICES=` with `$(gpu_range ...)` — **removed in subsequent refactor**.
- **`conserve/profile_1pxd.sh`** (`launch_engines()`): replaced `CUDA_VISIBLE_DEVICES=` with `$(gpu_range ...)` — **removed in subsequent refactor**; `check_num_gpus` multiplied by `TENSOR_PARALLEL_SIZE` ✓ (still in effect).
- **`profiling/launch_decode_grid.sh`**, **`profiling/launch_prefill_profile.sh`**, **`profiling/launch_interference.sh`**: `$(gpu_range ...)` — **removed in subsequent refactor**.
- **Notebooks (Figs 4, 5, 6)**: config/validation cell added ✓ (still in effect; imports `TENSOR_PARALLEL_SIZE`, queries `nvidia-smi`, sets per-launcher env var). Fig 2 cell was not added.

## Directory renames: `models/` → `model_outputs/`, `models_download/` → `models/`

Renamed the two top-level data directories to eliminate the ambiguity introduced in "Per-model folder reorganization" (see below), where `models/` meant per-model runtime data and `models_download/` meant HF weights — the opposite of what the names suggest.

- **`model_outputs/`** (renamed from `models/`): per-model runtime data — `long_prompts/`, `paper/section3/profiling/`, `paper/section3/fig{N}/`. `MODELS_ROOT` in `config.env` updated from `models` to `model_outputs`.
- **`models/`** (renamed from `models_download/`): HF weight cache only (`MODEL_DIR`). `MODEL_DIR` in `config.env` updated from `models_download` to `models`.
- **`profiling/config.py`**: fallback defaults updated to match (`MODEL_DIR` → `"models"`, `MODELS_ROOT` → `"model_outputs"`).
- **Launch scripts** (`launch_prefill_profile.sh`, `launch_decode_grid.sh`, `launch_interference.sh`): fallback `OUT_DIR`/`SEC3` paths updated from `$REPO_ROOT/models/` to `$REPO_ROOT/model_outputs/`.
- **`CLAUDE.md`**, **`doc/PAPER_TO_CODE.md`**, **section3 notebook markdown cells**: all path references updated.

## Update section 3 notebook markdown headers and `doc/PAPER_TO_CODE.md`

- **`paper/figures/section3/notebooks/fig{1..8}_*.ipynb` markdown cells**: updated output paths, data-collection paths, and prereq notes to reflect the current `model_outputs/<MODEL_SHORT>/paper/section3/` layout. Added ⚠️ notes to fig7 and fig8 documenting that their collect cells and `plot_prefill_powercap_ratio.py` / `plot_decode_grid_diff.py` still use the pre-reorganization `paper/figures/section3/output/<MODEL_SHORT>/` paths. The final `Image()` display cells in fig4, fig5, fig6 also remain stale (reference old `300W/` paths); the actual plots are written to the right location by the scripts.
- **`doc/PAPER_TO_CODE.md`**: Section 3 table rewritten to use notebooks as the primary interface column; data/output paths updated to `model_outputs/`; added Section 3 Data Layout block; fixed `profiling/paths.py` → `profiling/config.py` in Key Supporting Files.

## Add `profiling/launch_disagg_profile.sh` (fig3 data collection launcher)

Ported the missing sweep launcher for PD-disagg network-overhead profiling (Section 3, Fig 3). The reference repo had only `disagg_profile.sh` (a bare loop) and `unit_disagg_profile.sh` (single-L runner); neither was reachable from the notebook without adaptation.

- **New file**: `profiling/launch_disagg_profile.sh` — sources `config.sh`, iterates over the 19 L values matching the reference A40 data (128–65536, same set as the prefill linearity sweep), skips L values where `dcgmi_trace.tsv` and `decoder_forward_start_time.csv` already exist, calls `unit_disagg_profile.sh` sequentially with a 5 s cooldown between runs, and exits non-zero if any L fails.
- Output lands in `$GPU_MON_ROOT/<MODEL_SHORT>/pd_disagg_300W/<L>/` (overridable via `DISAGG_OUT_BASE`), matching what `plot_network_overhead.py` reads.

## GPU-typed profiling data layout (`profiling/gpu_profiling/`)

Replaced the flat `gpu_monitoring/` root with `profiling/gpu_profiling/<GPU_TYPE>/` so A40 and H100 timing data coexist without collision.

- **New directories**: `profiling/gpu_profiling/A40/` and `profiling/gpu_profiling/H100/` (H100 empty; populated when H100 experiments run).
- **`config.env`**: added `GPU_TYPE=A40`; updated `GPU_MON_ROOT=profiling/gpu_profiling/A40`.
- **`config.sh`**: added `GPU_TYPE` to the `export` line.
- **`profiling/config.py`**: added `GPU_TYPE = _get("GPU_TYPE", "A40")`; updated `GPU_MON_ROOT` default to `profiling/gpu_profiling/<GPU_TYPE>`; exports `GPU_TYPE` to `os.environ`. Switch to H100 by setting `GPU_TYPE=H100` and `GPU_MON_ROOT=profiling/gpu_profiling/H100` in `config.env` — no script changes needed.

## Section 3 per-figure output restructure

Separated raw profiling data from figure outputs within each model's data tree. All paths are relative to `models/<MODEL_SHORT>/`.

- **New structure**: raw data in `paper/section3/profiling/<data_type>/` (e.g. `profiling/prefill_profile_data/`); per-figure outputs in `paper/section3/fig{N}/` (fig2 = prefill + cache cost, fig3 = network overhead, fig4 = decode grid, fig5 = interference, fig6 = step drift). Old `profiling/section3/300W/` tree removed.
- **Data migration**: moved existing raw data dirs and figure files from `profiling/section3/300W/` to their new locations for both `Qwen3-0.6B` and `Qwen3.6-27B`.
- **Collection scripts** (`run_prefill_profile.py`, `run_decode_grid.py`, `run_cache_cost.py`, `run_interference.py`, `run_interference_kv.py`): updated default output paths from `"profiling"/"section3"/"300W"/<data_type>` to `"paper"/"section3"/"profiling"/<data_type>`.
- **Launch scripts** (`launch_prefill_profile.sh`, `launch_decode_grid.sh`, `launch_interference.sh`): updated `OUT_DIR` / `SEC3` defaults to match new profiling path.
- **Plot scripts** (all scripts in `paper/figures/section3/scripts/`): `DATA` now reads from `paper/section3/profiling/<data_type>/`; `OUT` now writes to `paper/section3/fig{N}/`. `plot_decode_flat.py` and `plot_decode_knee.py` also fixed: replaced hardcoded `"Qwen3-0.6B"` in `BASE` path with `MODEL_SHORT`, added `MODEL_DATA_DIR` import. `plot_cache_cost.py` gains a `--out` argument (default `fig2/`) separate from `--dir` (profiling dir).
- Cross-model figures (fig7, fig8, fig1) left unchanged — they depend on 200W+300W data not yet restructured.

## Per-model folder reorganization: `models/` and `models_download/`

> **Superseded by "Directory renames" entry above.** The directory names introduced here were subsequently renamed: `models/` → `model_outputs/` (per-model data) and `models_download/` → `models/` (HF cache).

- **`models/` (new)**: centralized per-model data directory. Each model gets `models/<MODEL_SHORT>/long_prompts/` (prompt JSON files) and `models/<MODEL_SHORT>/profiling/section3/300W/` (raw profiling data + notebook-generated figures).
- **`models_download/` (renamed from `models/`)**: HF weight cache only, pointed to by `MODEL_DIR`. No functional change — just renamed to clarify its role.
- **`config.env`**: replaced `PROFILING_DATA_DIR=data/profiling` with `MODELS_ROOT=models`; renamed `MODEL_DIR` default from `models` to `models_download`.
- **`profiling/config.py`**: replaced `PROFILING_DATA_DIR` with `MODELS_ROOT` and derived `MODEL_DATA_DIR = MODELS_ROOT / MODEL_SHORT`. Both are exported to `os.environ`. Removed `PROFILING_DATA_DIR` export entirely.
- **`config.sh`**: exports `MODELS_ROOT` instead of `PROFILING_DATA_DIR`.
- **Profiling scripts** (`run_prefill_profile.py`, `run_decode_grid.py`, `run_cache_cost.py`, `run_interference.py`, `run_interference_kv.py`, `decode_profile.py`, `prefill_profile.py`, `generate_long_prompts.py`): replaced all `PROFILING_DATA_DIR` references with `MODEL_DATA_DIR / "long_prompts"`; replaced all `paper/figures/section3/output/<MODEL_SHORT>/300W/` output paths with `MODEL_DATA_DIR / "profiling" / "section3" / "300W"`.
- **Launch scripts** (`launch_prefill_profile.sh`, `launch_decode_grid.sh`, `launch_interference.sh`): updated fallback `OUT_DIR`/`SEC3` to `$REPO_ROOT/models/$MODEL_SHORT/profiling/section3/300W/…`.
- **Plot scripts** (`paper/figures/section3/scripts/`, 9 files): added `MODEL_DATA_DIR` to imports; replaced `paper/figures/section3/output/<MODEL_SHORT>/300W` paths with `MODEL_DATA_DIR / "profiling" / "section3" / "300W"`.
- **Data moved**: `paper/figures/section3/output/300W/` → `models/Qwen3-0.6B/profiling/section3/300W/` (most complete 0.6B dataset); `paper/figures/section3/output/Qwen3.6-27B/300W/` → `models/Qwen3.6-27B/profiling/section3/300W/`; `profiling/long_prompts/prompts_8192x2048.json` → `models/Qwen3-0.6B/long_prompts/prompts_8192x2048.json`. Redundant `paper/figures/section3/output/Qwen3-0.6B/` (strict subset) removed.
- **Note**: `paper/figures/section3/notebooks/*.ipynb` still reference old paths and will need manual updates before re-running.

## Portable prompt-file layout and GPU monitoring documentation

- **`profiling/long_prompts/generate_long_prompts.py`** — new script (moved from the reference repo). Generates `prompts_{L}x2048.json` for L > 8192 by cyclically concatenating the 8192-token seed file. All paths come from `config.py` (`MODEL`, `MODEL_DIR`, `PROFILING_DATA_DIR`). Reads seed from and writes output to `PROFILING_DATA_DIR/<MODEL_SHORT>/` so different models never overwrite each other. Accepts `--targets L1,L2,...` to override the default set.
- **`profiling/run_prefill_profile.py`** — `load_prompts()` now reads from `PROFILING_DATA_DIR/<MODEL_SHORT>/` instead of the flat `PROFILING_DATA_DIR` root, matching the layout written by `generate_long_prompts.py`.
- **`doc/WORKFLOW.md`** — added "GPU Monitoring Data (`GPU_MON_ROOT`)" subsection at the end of Part 3. Explains what the directory contains (`decode/<B>/` and `pd_disagg_300W/<L>/`), what each type of run captures, which figure scripts consume each subtree, the multi-model namespacing under `<MODEL_SHORT>`, and the commands to collect decode data via `decode_profile.py`. Replaces scattered one-line references that previously left the concept undefined.
- **`CLAUDE.md`** — corrected three stale facts: `generate_long_prompts.py` now exists at `profiling/long_prompts/`; prompt files live at `PROFILING_DATA_DIR/<MODEL_SHORT>/`; `decode_profile.py` does produce figs 4/6 gpu-monitoring data (only fig3 pd_disagg data is missing). Added portability rule: no hardcoded machine-specific paths in source or docs.

## Multi-model support for Section 3 data-collection and figure scripts

- **`profiling/config.py`**: Added `MODEL_SHORT = MODEL.split("/")[-1]` (e.g. `Qwen3.6-27B` from `Qwen/Qwen3.6-27B`).
- **`paper/figures/section3/scripts/` (all 10 plot scripts)**: Output directories changed from the hardcoded `output/300W/` (or `output/200W/`) to `output/<MODEL_SHORT>/300W/` (and `output/<MODEL_SHORT>/200W/`). Figures from different models now land in separate subdirectories instead of overwriting each other.
- **`plot_network_overhead.py`** and **`plot_decode_step_drift.py`**: Input data paths in `GPU_MON_ROOT` were hardcoded to `Qwen3-0.6B/…`; now use `MODEL_SHORT` so the scripts read from the correct model's monitoring directory.
- **`plot_prefill_linearity.py`** and **`plot_prefill_with_cache.py`**: These two scripts had no `REPO_ROOT` or config import; added both, replacing `Path(__file__).parent.parent` relative paths with explicit `REPO_ROOT`-anchored paths.
- **`plot_prefill_linearity.py`**: Hardcoded model name in `prefill_linearity_fit.txt` output replaced with `MODEL_SHORT`.
- No changes to `plot_trace.py` (workload-distribution plot; not model-specific).
- **`profiling/run_prefill_profile.py`**: Updated hardcoded `OUT` path to `output/<MODEL_SHORT>/300W/prefill_profile_data`.
- **`profiling/run_cache_cost.py`**, **`run_decode_grid.py`**, **`run_interference.py`**, **`run_interference_kv.py`**: Updated argparse `--out`/`--out-dir` defaults from `output/300W/…` to `output/<MODEL_SHORT>/300W/…`.
- **`profiling/launch_decode_grid.sh`**, **`profiling/launch_interference.sh`**: Added `MODEL_SHORT="${MODEL##*/}"` after sourcing `config.sh`; updated fallback `OUT_DIR`/`SEC3` to include the model subfolder.
- **`profiling/launch_prefill_profile.sh`**: Fixed `Permission denied` crash — logs were written to `/tmp/prefill_gpuN.log` which is not writable. Added `MODEL_SHORT`, computed `OUT_DIR` (mirrors the Python script's output path), runs `mkdir -p`, and writes logs to `OUT_DIR/launcher_gpuN.log` (consistent with `launch_decode_grid.sh`).
- **Notebooks (fig2–fig8 display cells; fig7/fig8 collect cells)**: Updated all hardcoded `output/300W/` and `output/200W/` paths to use `MODEL_SHORT` from config. Fig7 collect cell was functionally broken (200W data written to wrong directory); fig8 collect cell's skip-if-exists check always evaluated to False.

## Benchmark switching: dynamic OUTPUT_DIR and WORKFLOW.md section

- **`input/mini_agent_test.py`**: `OUTPUT_DIR` is now derived from `DATASET_NAME.split("/")[-1]` so outputs never collide when switching benchmarks. Added commented-out entries for three planned datasets (`ScaleAI/SWE-bench_Pro`, `harborframework/terminal-bench-2.0`, `livecodebench/code_generation`) with HuggingFace links.
- **`doc/WORKFLOW.md`**: Added "Switching benchmarks" section (before Part 2) documenting the one-line swap, the four planned datasets, and the column-name caveat for new benchmarks.

## 2026-06-23 — WORKFLOW.md: factual corrections from repository audit

Audited every Part 3 command and §1.2 prereqs against the actual scripts; fixed
seven factual errors:

- **Part 3 — missing `run_cache_cost.py` step**: this script generates
  `cache_cost_data/` (prefix-cache hit-vs-miss latency) which feeds into
  `per_turn_cost_model.py`. It was not mentioned at all. Added as an explicit
  step after the three launch scripts.
- **Removed stale `merge_cache_cost_shards.py` no-arg call**: requires
  `<base_dir> <n_shards>` arguments; calling it bare would fail. Documented the
  correct sharded invocation inline in a comment next to `run_cache_cost.py`.
- **Removed redundant `merge_interference_shards.py` call**: `launch_interference.sh`
  already invokes it internally at the end of each phase; calling it again
  separately is a no-op at best.
- **Wrong attribution for cost-model source files**: `network_overhead_fit.txt`
  and `cache_cost_table.csv` are written by the section 3 **plot scripts**
  (`plot_network_overhead.py` and `plot_cache_cost.py`), not by the merge step.
  Corrected source attribution in both the Part 3 intro bullet and the Step 2
  update instruction.
- **Wrong paths for cost-model source files**: both files live in
  `paper/figures/section3/output/300W/`, not `paper/figures/section3/output/`.
  Fixed both path references.
- **Incorrect profiling prompt-file prerequisite scope**: all three launch
  scripts (decode grid, prefill profile, interference) and `run_cache_cost.py`
  need `prompts_*x2048.json`; the previous note said only `launch_prefill_profile.sh`
  needed them.
- **`profiling/generate_long_prompts.py` does not exist** in this repo (it
  lives in AgentScaling's data dir). Replaced with a `scp` transfer command.
- **`adaptive_disagg_oracle` policy added to the `run_sweep.sh` parameter
  table**: it is registered in `scheduler.py`, listed in `profile_1pxd.sh`
  `valid_args`, and appears in `main.py` choices, but was absent from the table.
  Added with the correct arrival trace type (`iter1_decoding_start`).
- **`ts` (moreutils) added to §1.2 hardware prerequisites**: the dcgmi pipeline
  pipes through `ts` to timestamp each output line; without it dcgmi data
  collection will fail silently.

## 2026-06-23 — WORKFLOW.md: new-machine portability improvements

Revised `doc/WORKFLOW.md` to be runnable on a fresh cluster without manual
investigation:

- **New §1.0** — "Machine-specific configuration" explains `config.env` (the
  three variables to edit before anything else), shows the `source config.sh &&
  mkdir -p` bootstrap, and calls out the `gpu_model_runner.py` hardcoded path
  upfront.
- **§1.1** — CUDA version note rewritten to be machine-agnostic; explains how
  to verify `torch.cuda.is_available()` rather than assuming CUDA 12.5. HF_HOME
  example path replaced with a generic placeholder.
- **§1.2** — Added dcgmi installation requirement and verification command;
  clarified that `PREFILLER_DEVICE_ID` / `DECODER_DEVICE_IDS` can override
  default GPU index assignments; noted that power-capped clusters may need
  `run_sweep.sh` cap-verification adjusted.
- **§1.3** — Replaced "pre-downloaded to /data/projects/AgentScaling/models"
  with instructions to set `MODEL_DIR` in `config.env` and download weights
  via `huggingface-cli`.
- **§1.4** — Added new-machine vLLM patch workflow: tar the 9 patched files on
  the source machine, scp to new machine, extract into vllm site-packages.
  Added post-patch verification command. Corrected `gpu_model_runner.py`
  hardcoded path note (literal `/data/projects/AgentScaling/gpu_monitoring/`,
  not `GPU_MON_ROOT`).
- **§2.1** — Replaced hardcoded `--download-dir /data/projects/AgentScaling/models`
  with `source config.sh && vllm serve "$MODEL" --download-dir "$MODEL_DIR"`.
- **§3** — Added prerequisite note for `launch_prefill_profile.sh`: requires
  `PROFILING_DATA_DIR/prompts_*x2048.json` files (~2.5 GB); pointed to
  `profiling/generate_long_prompts.py` as the generation path.

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

## `launch_interference.sh` / `run_interference*.py` — robustness fixes

- `launch_interference.sh` restructured to match canonical pattern (`SCRIPT_DIR`, `PY` variable, `set -euo pipefail`, no `cd`, absolute script paths, `pkill || true`); removed `export TMPDIR=/tmp` and `export PATH` manipulation
- Added phase argument to `launch_interference.sh`: `bash launch_interference.sh 1` or `2` to run a single phase; default runs both
- `run_interference.py` / `run_interference_kv.py`: fixed `env["TMPDIR"]` from `"/tmp"` (unwritable on bbq) to `MODEL_DATA_DIR / "tmp"`
- `run_interference.py` / `run_interference_kv.py`: added `aiohttp.TCPConnector(force_close=True)` to prevent `ServerDisconnectedError` from stale keep-alive connections during long cells
- `fig5_interference.ipynb`: added `cell-collect-kv` cell to run phase 2 independently; updated header to document the phase argument

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

## `config.env` — single source of truth for all global config

Introduced `config.env` at the repo root as the single canonical config file read
by both the shell and Python sides. Renamed `profiling/paths.py` → `profiling/config.py`
to reflect its broader role.

### What changed

- **`config.env`** (new) — defines `MODEL`, `MODEL_DIR`, `PROFILING_DATA_DIR`,
  `GPU_MON_ROOT`. Relative path defaults resolve from the repo root, so the repo
  works out of the box on a new machine without editing any script. To switch models,
  edit the `MODEL=` line here (or set `MODEL=` before running any script).
- **`config.sh`** — rewritten to parse `config.env` instead of hardcoding defaults.
  Now self-computes `REPO_ROOT` from its own location, so callers no longer need to
  pre-set it. Resolves relative paths against `REPO_ROOT` before exporting.
  Env vars already set in the caller's environment take precedence over `config.env`.
- **`profiling/config.py`** (renamed from `paths.py`) — same precedence logic as
  `config.sh`: env var → `config.env` → built-in fallback. Relative paths resolved
  against the repo root via `Path(__file__).parent.parent`.
- **`profiling/paths.py`** — deleted; all imports updated to `from config import`.
- **`conserve/profile_1pxd.sh`** — removed inline `MODEL=` definition; model now
  comes from `config.env` via `config.sh`.

### Files with updated imports (11 total)
`profiling/decode_profile.py`, `profiling/prefill_profile.py`,
`profiling/run_cache_cost.py`, `profiling/run_decode_grid.py`,
`profiling/run_interference.py`, `profiling/run_interference_kv.py`,
`profiling/run_prefill_profile.py`,
`paper/figures/section3/scripts/plot_decode_flat.py`,
`paper/figures/section3/scripts/plot_decode_knee.py`,
`paper/figures/section3/scripts/plot_decode_step_drift.py`,
`paper/figures/section3/scripts/plot_network_overhead.py`

## Known issues / remaining hardcodes

Items that are still hardcoded or lack a single source of truth:

- **`conserve/src/profile_agent.py:25` — hardcoded `ENGINE` URL separate from argparse default.**
  `ENGINE = "http://127.0.0.1:9101"` is set at module level independently of the
  `--proxy-port` argparse default (also 9101) on line 115. The two are not linked;
  changing the port requires updating both.

- **Ports have no single source of truth.**
  Ports 7100 (prefiller), 7200–7202 (decoders), 9101 (disagg proxy) are repeated
  as defaults in `profile_1pxd.sh`, `disagg_vllm_launcher.sh`, `main.py`, and
  `profile_agent.py`. All four files must be updated in sync if ports change.

- **NCCL `destroy_process_group` warning on vLLM server shutdown.**
  After `profile_1pxd.sh` exits, one warning per vLLM server appears:
  ```
  [rank0]:[W ProcessGroupNCCL.cpp:1538] WARNING: destroy_process_group() was not
  called before program exit, which can leak resources.
  ```
  Root cause: `MultiprocExecutor.shutdown()` closes the death pipe (signalling
  workers to begin their `finally` cleanup) and then immediately calls
  `p.terminate()` (SIGTERM). If SIGTERM arrives while `destroy_model_parallel()`
  or `destroy_distributed_environment()` is executing, Python's signal handler
  raises `SystemExit`, escaping the `finally` block before
  `torch.distributed.destroy_process_group()` is called. Cosmetic only — does not
  affect experiment data or exit codes.

## Missing files / documentation gaps

Items required to run the project that are not yet documented or present in this fork:

- **`PROFILING_DATA_DIR/prompts_*x2048.json`** (~2.5 GB) — pre-tokenized prompt files for each prefill length (128–65536 tokens). Required by `profiling/run_prefill_profile.py`. Stored at `/data/projects/AgentScaling/data/profiling/` on bbq (shared with original repo, no copy needed). Regeneration script: `profiling/generate_long_prompts.py`. Added entry to `doc/README_updated.md` Input Data table.
- **`conserve/input/mini_swe_agent_trace.json`** (~162 MB) — not generated or present in this fork. Requires running `mini_agent_test.py` + notebook cells against a live vLLM instance. No automation script.
- **`conserve/input/compound_prompts.json`** (~1.5 MB) — not generated or present in this fork. Regenerate with `conserve/src/prepare_compound_prompts.py` after `mini_swe_agent_trace.json` exists.
- **vLLM patch** — applied manually to the `conserve` conda env; no patch file or apply script committed to the repo. A new environment would require re-applying by hand against `vllm==0.11.0`. Four files changed: `vllm/engine/arg_utils.py` (added `--engine-log-file` / `--core-log-file` CLI flags), `vllm/v1/engine/core_client.py` (propagates flags to observability config), `vllm/v1/engine/async_llm.py` (emits `request_start` JSONL on `generate()`), `vllm/v1/engine/core.py` (emits `step_start` / `step_end` JSONL on every scheduler step). All changes marked `# NOTE(Jerry)`. See `doc/README_updated.md § vLLM Modifications` for full detail.
- **`paper/figures/section3/output/`** — all profiling run outputs are gitignored; no instructions for re-running the full Section 3 pipeline end-to-end (prefill profile → decode grid → interference → plots) in one command.
