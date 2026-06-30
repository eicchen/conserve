import os
try:
    import tomllib as _toml
except ImportError:
    import tomli as _toml
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_ENV_FILE = _REPO_ROOT / "config" / "config.env"


def _load_config_env():
    vals = {}
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                vals[k.strip()] = v.strip()
    return vals


_cfg = _load_config_env()


def _get(key, fallback=None):
    # config/config.env is primary — not the inherited shell environment
    return _cfg.get(key, fallback)


def _resolve_path(key, fallback):
    val = _get(key, fallback)
    p = Path(val)
    return p if p.is_absolute() else _REPO_ROOT / p


MODEL = _get("MODEL", "Qwen/Qwen3-0.6B")
MODEL_SHORT = MODEL.split("/")[-1]
MODEL_DIR = _resolve_path("MODEL_DIR", "models")
MODELS_ROOT = _resolve_path("MODELS_ROOT", "model_outputs")
MODEL_DATA_DIR = MODELS_ROOT / MODEL_SHORT
GPU_TYPE = _get("GPU_TYPE", "A40")
GPU_MON_ROOT = _resolve_path("GPU_MON_ROOT", f"profiling/gpu_profiling/{GPU_TYPE}")
TENSOR_PARALLEL_SIZE = int(_get("TENSOR_PARALLEL_SIZE", "1"))
BENCHMARK = _get("BENCHMARK", "SWE-bench_bm25_13K")
BENCHMARK_TRACE_DIR = MODEL_DATA_DIR / "benchmarks" / BENCHMARK

# Stamp the values back into os.environ so subprocesses (e.g. bash scripts
# launched from a notebook) see the file-derived values, not stale shell vars.
os.environ["MODEL"] = MODEL
os.environ["MODEL_DIR"] = str(MODEL_DIR)
os.environ["MODELS_ROOT"] = str(MODELS_ROOT)
os.environ["MODEL_DATA_DIR"] = str(MODEL_DATA_DIR)
os.environ["GPU_TYPE"] = str(GPU_TYPE)
os.environ["GPU_MON_ROOT"] = str(GPU_MON_ROOT)
os.environ["TENSOR_PARALLEL_SIZE"] = str(TENSOR_PARALLEL_SIZE)
os.environ["BENCHMARK"] = BENCHMARK
os.environ["BENCHMARK_TRACE_DIR"] = str(BENCHMARK_TRACE_DIR)


# ---------------------------------------------------------------------------
# Model profile registry
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    native_ctx: int
    eos_token_ids: list
    vllm_serve_flags: list = field(default_factory=list)


_PROFILES_FILE = _REPO_ROOT / "config" / "model_specific_configs.toml"
with open(_PROFILES_FILE, "rb") as _f:
    _PROFILES = {k: ModelProfile(**v) for k, v in _toml.load(_f)["models"].items()}

PROFILE = _PROFILES[MODEL]


if __name__ == "__main__":
    import shlex
    import sys
    if "--sh-vars" in sys.argv:
        flags = " ".join(shlex.quote(f) for f in PROFILE.vllm_serve_flags)
        print(f"VLLM_SERVE_FLAGS=({flags})")
