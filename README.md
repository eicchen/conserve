# ConServe

Research artifact for **ConServe**, an SLO-aware scheduler for multi-turn
agent serving on disaggregated LLM-inference clusters. This repository
contains everything needed to reproduce the experiments and figures in the
paper.

## What's in the repo

```
conserve/                            ← repo root (this dir contains .conserve_root marker)
├── conserve/                        ← the scheduler + experiment driver
│   ├── src/                         ← Python implementation
│   │   ├── scheduler.py             ← arrival driver + per-policy schedulers
│   │   ├── conversation.py          ← per-conversation runners (Collocated /
│   │   │                              Full Disagg / AMPD / ConServe / …)
│   │   ├── input_loader.py          ← loads mini_swe_agent_trace.json into
│   │   │                              the PROMPT_DATA / ITER_COUNT globals
│   │   ├── virtual_prefiller.py     ← AMPD's modelled prefiller queue
│   │   ├── per_turn_cost_model.py   ← analytical prefill / decode cost model
│   │   ├── main.py                  ← CLI entry point for one experiment
│   │   └── ...
│   ├── common/                      ← shared scripts (vLLM launcher,
│   │                                  disagg proxy, arrival-trace extractor)
│   ├── configs/                     ← LMCache YAML configs (1 prefiller + 3
│   │                                  decoders)
│   ├── input/                       ← mini_swe_agent input trace lives here
│   │                                  after download (not in git; see below)
│   ├── output/                      ← per-experiment output dirs (gitignored
│   │                                  except for the analysis notebooks)
│   ├── profile_1pxd.sh              ← the workhorse: brings up 1 prefiller +
│   │                                  N decoders, runs one experiment
│   ├── run_sweep.sh                 ← parameterized sweep driver (RPS sweep,
│   │                                  10-seed order sweep) — see header for
│   │                                  the arg matrix
│   └── rerun_cell.sh                ← parameterized N-trial replay of a
│                                      single (policy, cap, rps) cell
├── profiling/                       ← microbenchmarks behind section 3
│   ├── run_*.py / launch_*.sh       ← cache-cost / decode-grid /
│   │                                  interference sweeps
│   ├── merge_*_shards.py            ← stitch parallel shard outputs
│   ├── decode_profile.py            ← drift trace producer (KV-grows-with-
│   │                                  step latency timeseries)
│   └── prefill_profile.py(+.sh)     ← prefill-side counterpart
└── paper/figures/                   ← figure-generation pipelines
    ├── section3/{scripts,output,logs}
    └── section5/{scripts,output,cache}
```

The root `.conserve_root` marker file is what every script walks up to find
the repo root. The convention (see `.conserve_root` for details) is:

```python
REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
```

## External inputs (not in git)

Two input files are intentionally **not** committed to the repo because they
are large and easy to regenerate or host externally:

| file | size | role |
|---|---|---|
| `conserve/input/mini_swe_agent_trace.json` | ~162 MB | per-conversation prompt + token-size trace, derived from running mini-swe-agent over SWE-bench bm25_13K |
| `conserve/input/compound_prompts.json` | ~1.5 MB | Project-Gutenberg-padded compound prefix prompts (used by all scheduler runners to defeat prefix caching) |

**To obtain them**, ask the maintainers (or, if you want to regenerate from
scratch):

1. Run `conserve/input/mini_agent_test.py` end-to-end. This invokes
   `minisweagent` on the full SWE-bench bm25_13K test split (2,294 problems)
   and writes one JSON per problem to
   `conserve/output/SWE-bench_bm25_13K/`. Takes several hours and needs an
   HTTP-reachable LLM endpoint.
2. Run the `output/plot_dataset.ipynb` notebook. Cells 5–10 flatten the
   per-problem JSONs into the per-(conv, iter) records, pad each prompt with
   Project Gutenberg text to defeat prefix caching, and write the result as
   `prompt_records_sorted.json`. Rename / move that to
   `conserve/input/mini_swe_agent_trace.json`.
3. Run `conserve/src/prepare_compound_prompts.py` to generate the compound
   prompts file.

## Setup

```bash
# 1. Conda environment — Python 3.10, vLLM, transformers, etc.
conda create -n conserve python=3.10
conda activate conserve
pip install vllm tqdm pandas numpy scipy scikit-learn matplotlib \
            datasets transformers minisweagent

# 2. Place the input files (see "External inputs" above):
#    conserve/input/mini_swe_agent_trace.json
#    conserve/input/compound_prompts.json
```

The repo also requires:

- **4 GPUs** for the production sweeps (1 prefiller GPU + 3 decoder GPUs).
- **NVIDIA dcgmi** installed (for power/energy traces).
- **sudo nvidia-smi -pl** privileges (to apply power caps).

## Running experiments

### One experiment cell

```bash
cd conserve/
./profile_1pxd.sh <policy> <num_decoders> <output_dir>
```

`policy` is one of: `baseline`, `no_disagg_oracle`, `all_disagg`,
`adaptive_disagg_prefiller`, `adaptive_disagg_decoders`,
`adaptive_disagg_decoders_per_turn_kv`, `adaptive_disagg_oracle`,
`per_turn_adaptive_disagg_decoders`.

Environment variables tune the behaviour — see the script header for the
full list. Important ones:

- `ARRIVAL_TRACE` — path to the per-conv arrival trace JSON to replay
- `MAX_ITERS` — cap conversation length (default 5)
- `RPS` — arrival rate (only used when ARRIVAL_TRACE is not set)
- `WRONG_PRED_PCT` / `WRONG_PRED_SEED` — AMPD wrong-predict knobs
- `PREFILLER_TRACE_DIR` — matched prefiller log dir (AMPD VirtualPrefiller)
- `ORDER_SEED` — alternate ordering seed for the 10-seed order sweep

### Sweeps

`run_sweep.sh` orchestrates the production sweep matrix:

```bash
# Per-turn AMPD sweep at 300/300 (uncapped): both RPS and 10-seed order sweep
./run_sweep.sh per_turn_adaptive_disagg_decoders p300_d300 both

# ConServe at 300/200 (decoder cap), RPS sweep only
./run_sweep.sh adaptive_disagg_decoders_per_turn_kv p300_d200 rps

# Baseline (10 normalization runs) at 200/200
./run_sweep.sh baseline p200_d200 order

# Prefiller order sweep at 200/200 + extract per-seed arrival traces
EXTRACT_TRACES=1 ./run_sweep.sh adaptive_disagg_prefiller p200_d200 order
```

See the script header for the full arg matrix and env-var overrides
(`RPS_LIST`, `SEED_LIST`, `WRONG_PRED_PCT`, etc.). The script verifies the
GPU power caps match the requested `cap` argument and aborts on mismatch, so
you can't accidentally produce mislabelled output.

### Variance characterization (single-cell N-trial reruns)

```bash
./rerun_cell.sh <policy> <cap> <rps> [trials]

# Examples:
./rerun_cell.sh adaptive_disagg_decoders p300_d300 1.634 5
TRIALS=5 ./rerun_cell.sh per_turn_adaptive_disagg_decoders p300_d300 1.634

# Sweeping AMPD's wrong-predict rate at the saturation operating point:
for p in 0.05 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
    WRONG_PRED_PCT=$p ./rerun_cell.sh \
        per_turn_adaptive_disagg_decoders p300_d300 1.634 1
done
```

Outputs land in `output/var_check/<cap>/rps_<rps>/<policy_tag>/trial_<N>/`.
Existing trials are skipped on re-runs (so re-launching after a crash
resumes).

### Section-3 microbenchmarks (cache cost, decode grid, interference)

Under `profiling/`:

```bash
cd profiling/
./launch_decode_grid.sh         # decode-step latency × batch × KV grid
./launch_prefill_profile.sh     # prefill linearity
./launch_interference.sh        # collocated decode + prefill interference
```

Each launcher splits work across the available GPUs and writes raw vLLM
logs into `paper/figures/section3/output/<cap>/<experiment>_data/`. The
`merge_*_shards.py` scripts stitch the parallel-shard outputs into a single
summary CSV per cell.

## Regenerating figures

The figure pipelines read from the per-cell output dirs (or in section 5's
case, from the per-cell `per_step_latency.csv` summaries — which are tracked
in git, see `.gitignore`).

```bash
# Section 5 figures (headline, power-cap bar, wrong-pct sweep, etc.)
cd paper/figures/section5/scripts
python plot_headline.py
python plot_powercap.py
python plot_powercap_bar.py
python plot_wrong_pct_sweep.py
python plot_perf_energy_bar.py
python plot_perf_energy_bar_p95.py
# ... see scripts/ for the full list

# Section 3 figures
cd paper/figures/section3/scripts
python plot_trace.py                   # input-trace violin profile
python plot_decode_step_drift.py       # decode-step latency drift (section 3)
python plot_cache_cost.py
python plot_decode_grid.py
python plot_prefill_powercap_ratio.py  # 200W vs 300W prefill comparison
# ... see scripts/ for the full list
```

Most plotting scripts cache their input data collection in
`paper/figures/section5/cache/` to make iteration on figure aesthetics
fast (~1 s instead of ~55 s). Delete the cache or set
`HEADLINE_REBUILD=1 python plot_headline.py` to force re-collection.

Outputs are written next to the script under `output/`. Figures already
present in the repo were generated from the canonical run set; re-running
should be a no-op unless you've changed code.

## Power caps

Most experiments are sensitive to GPU power caps. Apply them with `sudo
nvidia-smi -pl <watts> -i <gpu_ids>` before launching anything:

```bash
# p300_d300 (all GPUs uncapped at 300 W)
sudo nvidia-smi -i 0,1,2,3 -pl 300

# p300_d200 (decoder cap)
sudo nvidia-smi -i 0       -pl 300
sudo nvidia-smi -i 1,2,3   -pl 200

# p200_d200 (full cap)
sudo nvidia-smi -i 0,1,2,3 -pl 200
```

`run_sweep.sh` and `rerun_cell.sh` both call a `verify_caps()` helper that
reads `nvidia-smi --query-gpu=power.limit` and aborts on mismatch, so a
forgotten cap change will not silently mislabel output.

## Citation

If you use this code, please cite the ConServe paper:

```
TODO: BibTeX entry once the paper is published
```

## License

TODO
