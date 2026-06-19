#!/bin/bash
# Central path config. Override any variable before sourcing this file, or set
# the corresponding env var before launching any script.
MODEL_DIR="${MODEL_DIR:-/data/projects/AgentScaling/models}"
PROFILING_DATA_DIR="${PROFILING_DATA_DIR:-/data/projects/AgentScaling/data/profiling}"
GPU_MON_ROOT="${GPU_MON_ROOT:-/data/projects/AgentScaling/gpu_monitoring}"
export MODEL_DIR PROFILING_DATA_DIR GPU_MON_ROOT
