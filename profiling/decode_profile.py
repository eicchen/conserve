import os
# Force InprocClient so per-call core_log_file is threaded to EngineCore.step().
# SyncMPClient (the default) runs EngineCore in a subprocess and silently
# drops the core_log_file kwarg, leaving vllm_core_log.jsonl unwritten.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import signal
import sys
from vllm import LLM, SamplingParams
import json
import subprocess
import signal
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_DIR, MODEL_DATA_DIR, GPU_MON_ROOT, MODEL, TENSOR_PARALLEL_SIZE, PROFILE

import time

# MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"  # local override
MODEL_PATH = MODEL
MODEL_SUFFIX = MODEL_PATH.split("/")[-1]
MAX_MODEL_LEN = 65536  # 2× native 32K; sufficient for decode-grid batch sizes
LLM_ARGS = {
    'model': MODEL_PATH,
    'dtype': "auto",
    'trust_remote_code': True,
    'download_dir': str(MODEL_DIR),
    'max_num_batched_tokens': 16384*2,
    'max_num_seqs': 1024,
    'enforce_eager': True,
    'tensor_parallel_size': TENSOR_PARALLEL_SIZE,
    'max_model_len': MAX_MODEL_LEN,
    **({'rope_scaling': {'rope_type': 'dynamic', 'factor': MAX_MODEL_LEN / PROFILE.native_ctx}}
       if MAX_MODEL_LEN > PROFILE.native_ctx else {}),
}
SAMPLING_PARAMS_ARGS = {
    'temperature': 1.2,
    'top_p': 1.0,
    'max_tokens': 1,
    'logit_bias': {tid: -100 for tid in PROFILE.eos_token_ids},
}
DCGMI_CMD = [
    "bash", "-c",
    "dcgmi dmon -e 155,156,157,1130,1131,1132,1133,150,140,151,152,153,158,159,1110,1111,1112,858,100,101,102,110,111,1120,203,204,206,207,1100,1101,1102,1103,1104 -d 1 | ts '%Y-%m-%dT%H:%M:%.S' >> "
]


llm = LLM(
    **LLM_ARGS,
)

import argparse

parser = argparse.ArgumentParser(description="Profile prefill with different input token sizes.")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
args = parser.parse_args()

in_token_size = 8
total_out_token_size = 65536
batch_sizes = [args.batch_size]
request_count = 16
in_dir = MODEL_DATA_DIR / "long_prompts"
out_dir = Path(f"{GPU_MON_ROOT}/{MODEL_SUFFIX}/decode")


for batch_size in batch_sizes:
    out_token_size = total_out_token_size // batch_size
    prompt_file = in_dir / f"prompts_{in_token_size}x2048.json"
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    prompt_texts = [p["prompt"] for p in prompts]
    prompt_texts = [prompt_texts[i:i+batch_size] for i in range(0, len(prompt_texts), batch_size)][:request_count]

    out_sub_dir = out_dir / f"{batch_size}"
    out_sub_dir.mkdir(parents=True, exist_ok=True)
    out_dcgmi_file = out_sub_dir / "dcgmi_trace.tsv"
    out_engine_file = out_sub_dir / "vllm_engine_log.jsonl"
    out_core_file = out_sub_dir / "vllm_core_log.jsonl"
    
    dcgmi_cmd = DCGMI_CMD.copy()
    dcgmi_cmd[-1] += str(out_dcgmi_file)
    sampling_params = SamplingParams(
        **{
            **SAMPLING_PARAMS_ARGS,
            "max_tokens": out_token_size
        }
    )

    dcgmi_proc = subprocess.Popen(
        dcgmi_cmd,
        preexec_fn=os.setsid  # Start the process in a new session (process group)
    )
    time.sleep(5)

    try:
        results = llm.generate(
            fixed_batches = prompt_texts,
            sampling_params = sampling_params,
            engine_log_file = out_engine_file,
            core_log_file = out_core_file,
        )
    finally:
        print("Stopping GPU monitoring...")
        try:
            os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGTERM)  # Kill the whole process group
            dcgmi_proc.wait(timeout=5)
            print("GPU monitoring stopped")
        except Exception as e:
            print(f"Failed to terminate dcgmi process group: {e}")
            try:
                os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGKILL)
            except Exception as e2:
                print(f"Failed to kill dcgmi process group: {e2}")
    # print(results)
    time.sleep(1)