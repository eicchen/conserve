import os
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/projects/AgentScaling/models")
PROFILING_DATA_DIR = os.environ.get("PROFILING_DATA_DIR", "/data/projects/AgentScaling/data/profiling")
GPU_MON_ROOT = Path(os.environ.get("GPU_MON_ROOT", "/data/projects/AgentScaling/gpu_monitoring"))
