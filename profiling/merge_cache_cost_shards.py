"""Merge per-shard cache_cost outputs into one cache_cost_data dir.

Each shard wrote <base>/shard<i>/{cell_*.jsonl, plan.json} for a disjoint
subset of L. cell_idx is globally unique (global L index * N_REPLICATES + rep),
so cell filenames never collide; this just moves them up into <base> and
concatenates the per-shard plan cell lists.

Usage: merge_cache_cost_shards.py <base_dir> <n_shards>
"""

import json
import shutil
import sys
from pathlib import Path


def main():
    base = Path(sys.argv[1])
    n_shards = int(sys.argv[2])
    merged = None
    all_cells = []
    for i in range(n_shards):
        d = base / f"shard{i}"
        j = json.loads((d / "plan.json").read_text())
        if merged is None:
            merged = {k: v for k, v in j.items() if k != "cells"}
        all_cells.extend(j["cells"])
        for f in d.glob("cell_*.jsonl"):
            shutil.move(str(f), str(base / f.name))
    all_cells.sort(key=lambda c: c["cell_idx"])
    merged["cells"] = all_cells
    (base / "plan.json").write_text(json.dumps(merged, indent=2))
    print(f"merged {n_shards} shards -> {base}  ({len(all_cells)} cells)")


if __name__ == "__main__":
    main()
