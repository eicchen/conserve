import asyncio
import csv
import os
import signal
import subprocess
import time

import input_loader as cfg
from scheduler import schedule_conversations


def parse_args():
    import argparse

    def csv_ints(s):
        return [int(x) for x in s.split(",")]

    def csv_strs(s):
        return [x.strip() for x in s.split(",")]

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[7100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[7200])
    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--proxy-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--proxy-port", type=csv_ints, default=[9101])
    parser.add_argument("--baseline", type=str, default="baseline",
                        choices=["baseline", "no_disagg_oracle", "all_disagg",
                                 "adaptive_disagg_prefiller", "adaptive_disagg_decoders",
                                 "adaptive_disagg_decoders_per_turn_kv",
                                 "adaptive_disagg_oracle",
                                 "per_turn_adaptive_disagg_decoders"])
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--kv-budget-tokens", type=int, default=None,
                        help="KV cache budget per engine in tokens (default: per KV_BUDGET_TOKENS in input_loader)")
    parser.add_argument("--mock", action="store_true",
                        help="Skip vLLM; return random text after 1 s (for scheduler testing)")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Cap iterations per conversation (default: use all from trace)")
    parser.add_argument("--rps", type=float, default=None,
                        help="Poisson arrival rate (default: per RPS in input_loader)")
    parser.add_argument("--arrival-trace", type=str, default=None,
                        help="Path to arrival-trace JSON (list of {conv_id, offset_sec}). "
                             "When set, overrides Poisson arrivals for adaptive_disagg_decoders.")
    parser.add_argument("--order-seed", type=int, default=None,
                        help="If set, shuffles conversation launch order with this seed "
                             "(currently honoured by adaptive_disagg_prefiller).")
    parser.add_argument("--wrong-pred-pct", type=float, default=0.0,
                        help="(per_turn_adaptive_disagg_decoders only) fraction of "
                             "turn-2+ requests that the policy 'wrongly' routes to a "
                             "virtual prefiller. The decoder waits the AMPD-modeled "
                             "disagg cost (per_turn_cost_model.py) before serving.")
    parser.add_argument("--wrong-pred-seed", type=int, default=42,
                        help="(per_turn_adaptive_disagg_decoders only) RNG seed for "
                             "picking which (conv, iter) pairs are wrong-predicted; "
                             "fixed once at policy startup so reruns are reproducible.")
    parser.add_argument("--prefiller-trace-dir", type=str, default=None,
                        help="(per_turn_adaptive_disagg_decoders only) path to the "
                             "matching adaptive_disagg_prefiller run directory. "
                             "Its per_step_latency.csv seeds the virtual prefiller "
                             "queue so wrong-predict blocks delay the right iter-0s. "
                             "When set, a synthetic prefiller trace is written to "
                             "<output_dir>/synthetic_prefiller_per_step_latency.csv "
                             "at end of run.")

    args = parser.parse_args()
    assert args.num_decoders == len(args.decoder_host)
    assert args.num_decoders == len(args.decoder_port)

    cfg.MODEL = args.model
    if args.kv_budget_tokens is not None:
        cfg.KV_BUDGET_TOKENS = args.kv_budget_tokens
    if args.rps is not None:
        cfg.RPS = args.rps
        cfg.REQUEST_INTERVAL = 1.0 / args.rps
    cfg.MOCK_INFERENCE = args.mock
    cfg.apply_iter_cap(args.max_iters)

    return args


def prep_outputs(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cfg.OUT_DCGMI_FILE = os.path.join(args.output_dir, "dcgmi_trace.tsv")
    cfg.LATENCY_LOG_FILE = os.path.join(args.output_dir, "per_step_latency.csv")

    for path in (cfg.OUT_DCGMI_FILE, cfg.LATENCY_LOG_FILE):
        if os.path.exists(path):
            os.remove(path)

    with open(cfg.LATENCY_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["conv_id", "step_id", "prompt_tokens", "max_tokens", "start_time", "end_time", "latency_sec"]
        )
    print(f"Writing conversation logs to: {cfg.LATENCY_LOG_FILE}")


def start_dcgmi():
    print("Starting GPU monitoring...")
    dcgmi_cmd = cfg.DCGMI_CMD.copy()
    dcgmi_cmd[-1] += str(cfg.OUT_DCGMI_FILE)
    proc = subprocess.Popen(dcgmi_cmd, preexec_fn=os.setsid)
    time.sleep(5)
    print(f"GPU monitoring started — writing to: {cfg.OUT_DCGMI_FILE}")
    return proc


def stop_dcgmi(proc):
    print("Stopping GPU monitoring...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
        print("GPU monitoring stopped")
    except Exception as e:
        print(f"Failed to terminate dcgmi process group: {e}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e2:
            print(f"Failed to kill dcgmi process group: {e2}")


async def main():
    args = parse_args()
    prep_outputs(args)

    dcgmi_proc = start_dcgmi()
    try:
        await schedule_conversations(args)
    finally:
        stop_dcgmi(dcgmi_proc)


if __name__ == "__main__":
    asyncio.run(main())
