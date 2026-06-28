"""
Generate prompts_{L}x2048.json files for any target token length.

Each output file contains exactly 2048 prompts, each tokenizing to exactly L
tokens under the active model's tokenizer. Prompts are built by cyclically
concatenating the 8192-token seed file and trimming to L tokens.

Reads:  models/<MODEL_SHORT>/long_prompts/prompts_8192x2048.json  (seed file)
Writes: models/<MODEL_SHORT>/long_prompts/prompts_{L}x2048.json   (one per target length)

Run from the repo root or profiling/ directory:
    python profiling/long_prompts/generate_long_prompts.py

Override targets via --targets:
    python profiling/long_prompts/generate_long_prompts.py --targets 4096,8192,40960
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "profiling"))
from config import MODEL, MODEL_DIR, MODEL_SHORT, MODEL_DATA_DIR

# Mirrors the full set in the reference data/profiling/ directory.
# 8192 is excluded — it is the seed file, not generated.
DEFAULT_TARGETS = [
    1, 2, 4, 8, 16, 32, 64,
    128, 256, 512, 1024, 2048, 4096, 6144,
    10240, 12288, 16384, 20480, 24576, 28672, 32768,
    65536,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--targets", type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_TARGETS,
        metavar="L1,L2,...",
        help=f"Comma-separated token lengths to generate (default: {DEFAULT_TARGETS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL, cache_dir=str(MODEL_DIR), trust_remote_code=True,
    )

    out_dir = MODEL_DATA_DIR / "long_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_path = out_dir / "prompts_8192x2048.json"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {seed_path}\n"
            f"Ensure models/{MODEL_SHORT}/long_prompts/prompts_8192x2048.json exists."
        )

    seed = json.loads(seed_path.read_text())
    assert len(seed) == 2048, f"expected 2048 seed prompts, got {len(seed)}"

    seed_ids = [tok.encode(p["prompt"], add_special_tokens=False) for p in seed]
    lengths = [len(ids) for ids in seed_ids]
    print(f"[{MODEL_SHORT}] seed tokenized lengths — min={min(lengths)} max={max(lengths)} "
          f"median={sorted(lengths)[len(lengths)//2]}")

    for L in args.targets:
        print(f"[{MODEL_SHORT}] generating L={L} ({2048} prompts)…", flush=True)
        out = []
        for i in range(2048):
            if i % 256 == 0:
                print(f"  L={L}: {i}/2048", flush=True)
            ids: list[int] = []
            j = i
            while len(ids) < L:
                ids = ids + seed_ids[j % 2048]
                j += 1
            ids = ids[:L]
            text = tok.decode(ids, skip_special_tokens=True)
            re_len = len(tok.encode(text, add_special_tokens=False))
            out.append({
                "id": seed[i]["id"],
                "topic": seed[i]["topic"],
                "prompt": text,
                "tokenized_len": re_len,
            })
        path = out_dir / f"prompts_{L}x2048.json"
        path.write_text(json.dumps(out))
        lens = [p["tokenized_len"] for p in out]
        match_rate = sum(x == L for x in lens)
        print(f"  wrote {path.name}: min={min(lens)} max={max(lens)} "
              f"target={L}  exact_match={match_rate}/{len(lens)}", flush=True)


if __name__ == "__main__":
    main()
