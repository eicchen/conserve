"""Merge per-shard interference outputs into one server_core.jsonl + cells.json.

Each shard wrote <base>/shard<i>/{server_core.jsonl, cells.json}. Concatenating
the core logs is valid: each shard's log is an internally complete, alternating
step_start/step_end sequence, so the plot's positional start/end zip still
holds; and request_ids are globally unique (cell_idx = rep*CELLS_PER_REP+local),
so rid->cell matching never crosses shards.

Usage: merge_interference_shards.py <base_dir> <n_shards>
"""

import json
import sys
from pathlib import Path


def main():
    base = Path(sys.argv[1])
    n_shards = int(sys.argv[2])
    shard_dirs = [base / f"shard{i}" for i in range(n_shards)]

    out_core = base / "server_core.jsonl"
    with open(out_core, "w") as out:
        for d in shard_dirs:
            with open(d / "server_core.jsonl") as f:
                for line in f:
                    out.write(line)

    merged = None
    all_cells = []
    for d in shard_dirs:
        j = json.loads((d / "cells.json").read_text())
        if merged is None:
            merged = {k: v for k, v in j.items() if k != "cells"}
        all_cells.extend(j["cells"])
    all_cells.sort(key=lambda c: c["cell_idx"])
    merged["cells"] = all_cells
    (base / "cells.json").write_text(json.dumps(merged, indent=2))

    print(f"merged {n_shards} shards -> {out_core}  ({len(all_cells)} cells)")


if __name__ == "__main__":
    main()
