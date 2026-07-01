"""Build mini_swe_agent_trace.json from per-problem swe_output JSONs.

Reads:
    BENCHMARK_TRACE_DIR/swe_output/*.json   (written by mini_agent_test.py)

Writes:
    BENCHMARK_TRACE_DIR/mini_swe_agent_trace.json
        fields: conv_id, iter_id, in_token_size, out_token_size

Then optionally plots Figure 1 (--plot flag or by default).

Usage
-----
    python conserve/input/build_agent_trace.py            # build trace + plot
    python conserve/input/build_agent_trace.py --no-plot  # trace only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "config"))
from config import BENCHMARK_TRACE_DIR  # noqa: E402

SWE_OUTPUT_DIR = BENCHMARK_TRACE_DIR / "swe_output"
TRACE_PATH     = BENCHMARK_TRACE_DIR / "mini_swe_agent_trace.json"
PLOT_SCRIPT    = REPO_ROOT / "paper" / "figures" / "section3" / "scripts" / "plot_trace.py"


def build_trace():
    usages, uncached_tokens_list = [], []
    files = sorted(SWE_OUTPUT_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No JSON files found in {SWE_OUTPUT_DIR}")

    skipped_incomplete = 0
    for file_path in files:
        with open(file_path) as fh:
            data = json.load(fh)
        messages = data.get("messages", [])
        if not messages or messages[-1].get("role") != "exit":
            # Trajectory hasn't reached a terminal state (e.g. the job was
            # killed by Slurm's time limit mid-turn) — its token counts
            # aren't representative of a finished conversation, so leave it
            # out until a later --resume run finishes it.
            skipped_incomplete += 1
            continue
        asst = [m for m in messages if m["role"] == "assistant"]
        usage = [m["extra"]["response"]["usage"] for m in asst]
        conv_id = int(file_path.stem.split("_")[-1])
        for i, u in enumerate(usage):
            u["conv_id"] = conv_id
            u["iter_id"] = i
        if not usage:
            continue
        usages += usage
        uncached_tokens_list += (
            [usage[0]["prompt_tokens"]]
            + [usage[i + 1]["prompt_tokens"] - usage[i]["total_tokens"]
               for i in range(len(usage) - 1)]
        )

    records = [
        {
            "conv_id":        int(u["conv_id"]),
            "iter_id":        int(u["iter_id"]),
            "in_token_size":  int(unc),
            "out_token_size": int(u["completion_tokens"]),
        }
        for u, unc in zip(usages, uncached_tokens_list)
    ]
    records.sort(key=lambda r: (r["conv_id"], r["iter_id"]))

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(records))
    n_convs = len({r["conv_id"] for r in records})
    skip_note = f" ({skipped_incomplete} incomplete trajectories skipped)" if skipped_incomplete else ""
    print(f"Saved {len(records)} records ({n_convs} conversations) → {TRACE_PATH}{skip_note}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot_trace.py after building the trace")
    args = p.parse_args()

    build_trace()

    if not args.no_plot:
        subprocess.run([sys.executable, str(PLOT_SCRIPT)], check=True)


if __name__ == "__main__":
    main()
