import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_ENV_FILE = _REPO_ROOT / "config.env"


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
    val = os.environ.get(key)
    if val is not None:
        return val
    return _cfg.get(key, fallback)


def _resolve_path(key, fallback):
    val = _get(key, fallback)
    p = Path(val)
    return p if p.is_absolute() else _REPO_ROOT / p


MODEL = _get("MODEL", "Qwen/Qwen3-0.6B")
MODEL_DIR = _resolve_path("MODEL_DIR", "models")
PROFILING_DATA_DIR = _resolve_path("PROFILING_DATA_DIR", "data/profiling")
GPU_MON_ROOT = _resolve_path("GPU_MON_ROOT", "gpu_monitoring")
