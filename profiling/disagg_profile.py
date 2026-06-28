import signal
import json
import os
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import GPU_MON_ROOT, MODEL_DATA_DIR, MODEL_DIR, MODEL, MODEL_SHORT

import time
import requests
import argparse

parser = argparse.ArgumentParser(description="Profile disagg prefill/decode for different input token sizes.")
parser.add_argument("--in-token-size", type=int, required=True, help="Number of input tokens")
args = parser.parse_args()

in_token_size = args.in_token_size
out_token_size = 2

in_dir = MODEL_DATA_DIR / "long_prompts"
_out_base = os.environ.get("DISAGG_OUT_BASE", str(GPU_MON_ROOT / MODEL_SHORT / "pd_disagg_300W"))
out_dir = Path(_out_base) / str(in_token_size)

DCGMI_CMD = [
    "bash", "-c",
    "dcgmi dmon -e 155,156,157,1130,1131,1132,1133,150,140,151,152,153,158,159,1110,1111,1112,858,100,101,102,110,111,1120,203,204,206,207,1100,1101,1102,1103,1104 -d 1 | ts '%Y-%m-%dT%H:%M:%.S' >> "
]

url = "http://localhost:9101/v1/completions"
payload = {
    "model": MODEL,
    "max_tokens": out_token_size,
    "temperature": 1.2,
    "top_p": 1.0,
    "stream": False,
    "logit_bias": {
        2: -100,
        13: -100,
        128001: -100,
        128009: -100,
    },
}

prompt_file = in_dir / f"prompts_{in_token_size}x2048.json"
if prompt_file.exists():
    with open(prompt_file) as f:
        prompts = json.load(f)
    prompt_texts = [p["prompt"] for p in prompts]
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(MODEL_DIR), trust_remote_code=True)
    src = in_dir / "prompts_65536x2048.json"
    print(f"[disagg_profile] prompts_{in_token_size}x2048.json missing; "
          f"truncating {src.name} to {in_token_size} tokens.", flush=True)
    with open(src) as f:
        prompts = json.load(f)
    prompt_texts = []
    for p in prompts:
        ids = tok.encode(p["prompt"], add_special_tokens=False)[:in_token_size]
        prompt_texts.append(tok.decode(ids, skip_special_tokens=True))

_nreq = int(os.environ.get("DISAGG_PROFILE_NREQ", str(len(prompt_texts))))
prompt_texts = prompt_texts[:_nreq]
print(f"[disagg_profile] sending {len(prompt_texts)} requests for L={in_token_size}", flush=True)

out_dcgmi_file = out_dir / "dcgmi_trace.tsv"
out_dir.mkdir(parents=True, exist_ok=True)
DCGMI_CMD[-1] += str(out_dcgmi_file)

dcgmi_proc = subprocess.Popen(DCGMI_CMD, preexec_fn=os.setsid)
time.sleep(5)

try:
    for i, prompt in enumerate(prompt_texts):
        payload["prompt"] = prompt
        payload["request_id"] = f"{i}"
        print(f"payload: {payload}", flush=True)
        resp = requests.post(url, json=payload)
        resp_json = json.loads(resp.text)
        output_text = resp_json["choices"][0]["text"]
finally:
    print("Stopping GPU monitoring...")
    try:
        os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGTERM)
        dcgmi_proc.wait(timeout=5)
        print("GPU monitoring stopped")
    except Exception as e:
        print(f"Failed to terminate dcgmi process group: {e}")
        try:
            os.killpg(os.getpgid(dcgmi_proc.pid), signal.SIGKILL)
        except Exception as e2:
            print(f"Failed to kill dcgmi process group: {e2}")

time.sleep(1)
