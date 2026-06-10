"""Generate compound_prompts.json: N distinct prompts each >= TARGET_TOKENS tokens.

Concatenates iter-0 prompts from the trace until each compound clears the threshold.
Uses the longest iter-0 prompts first so few concatenations are needed per compound,
and ensures no source prompt is reused across compounds (each compound's content is
distinct, which is what we want for the prefix-cache simulation).
"""

import json
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())


INPUT  = Path(f'{REPO_ROOT}/conserve/input/mini_swe_agent_trace.json')
OUTPUT = Path(f'{REPO_ROOT}/conserve/input/compound_prompts.json')
TARGET_TOKENS = 25_000
N_COMPOUND = 8


def main():
    with open(INPUT) as f:
        raw = json.load(f)

    iter0 = sorted(
        [e for e in raw if e['iter_id'] == 0],
        key=lambda x: -x['in_token_size'],          # longest first
    )

    compounds = []
    cursor = 0
    for k in range(N_COMPOUND):
        pieces, sizes, src_ids = [], [], []
        total = 0
        while total < TARGET_TOKENS and cursor < len(iter0):
            e = iter0[cursor]; cursor += 1
            pieces.append(e['prompt'])
            sizes.append(e['in_token_size'])
            src_ids.append(e['conv_id'])
            total += e['in_token_size']
        if total < TARGET_TOKENS:
            raise SystemExit(f'Ran out of source prompts before reaching {TARGET_TOKENS} tokens for compound {k}')
        compounds.append({
            'compound_idx':     k,
            'estimated_tokens': total,
            'n_source_prompts': len(pieces),
            'source_conv_ids':  src_ids,
            'source_sizes':     sizes,
            'prompt':           '\n'.join(pieces),
        })

    with open(OUTPUT, 'w') as f:
        json.dump(compounds, f, indent=2)

    print(f'Wrote {len(compounds)} compound prompts to {OUTPUT}')
    for c in compounds:
        print(f'  compound {c["compound_idx"]}: ~{c["estimated_tokens"]:,} tokens '
              f'({c["n_source_prompts"]} concatenated, conv_ids={c["source_conv_ids"]})')


if __name__ == '__main__':
    main()
