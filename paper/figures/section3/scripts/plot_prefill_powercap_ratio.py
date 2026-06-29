"""
200W-vs-300W prefill comparison: TTFT ratio (200W / 300W) vs input tokens.

Source: cache_cost_table.csv at both power levels (same harness, same script;
only the GPU power cap differs). The cache_cost experiment measures prefill
latency for a cache MISS (cold prefill) and a cache HIT (prefix-cached) at
each L. Plotting the 200W/300W ratio isolates the power-cap effect:
  - miss  : compute-bound -> ratio climbs with L (200W throttles compute)
  - hit   : memory-bound  -> ratio ~1.0 (power cap barely matters)

Common L values only (200W cache_cost has the 6-L sweep; 300W has 19).
"""

import json
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_SHORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEC3 = (REPO_ROOT / "paper/figures/section3")
OUT = SEC3 / "output" / MODEL_SHORT


def recheck_hit_median(recheck_dir) -> float:
    """Median prefix-cache-hit latency from an isolated single-L recheck.

    iter 1, first step_end per cell is the cache-hit prefill.
    """
    recheck_dir = Path(recheck_dir)
    plan = json.loads((recheck_dir / "plan.json").read_text())
    hits = []
    for c in plan["cells"]:
        eng = pd.read_json(recheck_dir / c["engine_log"], lines=True)
        it = sip = -1
        for _, r in eng.iterrows():
            if r["event"] == "iteration_start":
                it += 1
                sip = -1
            elif r["event"] == "step_end":
                sip += 1
                if sip == 0 and it == 1:
                    hits.append(float(r["step_duration_ms"]))
    return float(np.median(hits))


def main():
    df200 = pd.read_csv(SEC3 / "output" / MODEL_SHORT / "200W" / "cache_cost_table.csv")
    df300 = pd.read_csv(SEC3 / "output" / MODEL_SHORT / "300W" / "cache_cost_table.csv")

    # --- outlier fixes (the original sweeps caught a few noisy cells) ---
    # L=128: the 300W cells were cold-start noise -> full warmup re-run row.
    ov128 = pd.read_csv(SEC3 / "output" / MODEL_SHORT / "300W" / "cache_cost_rerun_128_4096.csv")
    ov128 = ov128[ov128["L"] == 128]
    df300 = pd.concat([df300[df300["L"] != 128], ov128], ignore_index=True)
    # L=4096: only the prefix-cache HIT was noisy (miss was fine) -> override
    # just the hit with an isolated single-L recheck, at both power levels.
    df300.loc[df300["L"] == 4096, "hit_p50"] = recheck_hit_median(
        SEC3 / "output" / MODEL_SHORT / "300W" / "cache_cost_recheck_4096")
    df200.loc[df200["L"] == 4096, "hit_p50"] = recheck_hit_median(
        SEC3 / "output" / MODEL_SHORT / "200W" / "cache_cost_recheck_4096")

    m = df200.merge(df300, on="L", suffixes=("_200", "_300")).sort_values("L")
    m["miss_ratio"] = m["miss_p50_200"] / m["miss_p50_300"]
    m["hit_ratio"] = m["hit_p50_200"] / m["hit_p50_300"]
    # Percentage change of 200W relative to 300W.
    m["miss_pct"] = (m["miss_ratio"] - 1.0) * 100.0
    m["hit_pct"] = (m["hit_ratio"] - 1.0) * 100.0

    print(m[["L", "miss_p50_200", "miss_p50_300", "miss_pct",
             "hit_p50_200", "hit_p50_300", "hit_pct"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.plot(m["L"], m["miss_pct"], marker="o", ms=5, lw=1.8,
            color="#CC3311", label="prefill")
    ax.plot(m["L"], m["hit_pct"], marker="s", ms=5, lw=1.8,
            color="#117733", label="prefill w/ prefix-cache hit")

    ax.set_xscale("log")
    ax.set_xlabel("Input tokens", fontsize=10)
    ax.set_ylabel("TTFT change (%)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "prefill_powercap_ratio.pdf", dpi=200)
    fig.savefig(OUT / "prefill_powercap_ratio.png", dpi=200)
    print("\nSaved prefill_powercap_ratio.pdf / .png")


if __name__ == "__main__":
    main()
