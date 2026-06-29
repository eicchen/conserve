"""
Combined view: prefill latency vs L, on log y, with the prefix-cache-hit curve
overlaid on top of the cache-miss (= cold prefill) measurement from
plot_prefill_linearity.

This is the "same plot, two regimes" version that subsumes cache_cost.png:
  - miss curve = single-prompt prefill from prefill_profile_data (19 L, 100 reps)
  - hit  curve = from cache_cost_table.csv (6 L, 5 reps, second prefill at same L)

Also overlays the 6 cache-miss points from cache_cost_table.csv as a sanity
check that the two independent harnesses agree.
"""

import json
from pathlib import Path
import sys

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_SHORT, MODEL_DATA_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = MODEL_DATA_DIR / "paper" / "section3" / "fig2"
DATA = MODEL_DATA_DIR / "paper" / "section3" / "profiling" / "prefill_profile_data"
CACHE_DATA = MODEL_DATA_DIR / "paper" / "section3" / "profiling" / "cache_cost_data"
CACHE_TABLE = OUT / "cache_cost_table.csv"

WARMUP_DROP = 2
KNEE_LOW = 512


def list_L_values():
    return sorted(int(p.name) for p in DATA.iterdir() if p.is_dir() and p.name.isdigit())


def parse_L(L: int) -> np.ndarray:
    eng = pd.read_json(DATA / str(L) / "engine_log.jsonl", lines=True)
    rows = []
    sip = -1
    for _, row in eng.iterrows():
        if row["event"] == "iteration_start":
            sip = -1
        elif row["event"] == "step_end":
            sip += 1
            if sip == 0:
                rows.append(float(row["step_duration_ms"]))
    arr = np.array(rows, dtype=float)
    if len(arr) > WARMUP_DROP:
        arr = arr[WARMUP_DROP:]
    return arr


def load_cache_hit_raw() -> pd.DataFrame:
    """Per-rep cache-hit prefill latency (iter 1, first step_end per cell)."""
    plan = json.loads((CACHE_DATA / "plan.json").read_text())
    rows = []
    for cell in plan["cells"]:
        eng = pd.read_json(CACHE_DATA / cell["engine_log"], lines=True)
        iter_idx = -1
        sip = -1
        for _, row in eng.iterrows():
            if row["event"] == "iteration_start":
                iter_idx += 1
                sip = -1
            elif row["event"] == "step_end":
                sip += 1
                if sip == 0 and iter_idx == 1:
                    rows.append({"L": cell["L"], "rep": cell["rep"],
                                 "hit_ms": float(row["step_duration_ms"])})
    return pd.DataFrame(rows)


def main():
    L_values = list_L_values()
    per_L = {L: parse_L(L) for L in L_values}

    # Floor (L <= 512)
    floor_L = [L for L in L_values if L <= KNEE_LOW]
    y_f = np.concatenate([per_L[L] for L in floor_L]) if floor_L else np.array([])
    floor_mean = float(np.mean(y_f)) if len(y_f) else float("nan")

    # Quadratic with constant over L >= 1024
    Ls_full = [L for L in L_values if L >= 1024]
    x_full = np.concatenate([np.full(len(per_L[L]), L) for L in Ls_full]).astype(float)
    y_full = np.concatenate([per_L[L] for L in Ls_full])
    a_fq, b_fq, c_fq = np.polyfit(x_full, y_full, 2)
    y_fq_pred = a_fq * x_full**2 + b_fq * x_full + c_fq
    r2_full_q = 1.0 - float(np.sum((y_full - y_fq_pred)**2)
                            / np.sum((y_full - y_full.mean())**2))

    # Cache hit raw per-rep data; table kept only for the bracket annotation.
    hit_raw = load_cache_hit_raw()
    cache = pd.read_csv(CACHE_TABLE)
    hit_const = float(cache["hit_p50"].mean())

    # === Plot ===
    fig, ax = plt.subplots(figsize=(5, 3.0))

    # Miss scatter (per-prompt prefill latencies)
    for L in L_values:
        v = per_L[L]
        ax.scatter(np.full(len(v), L), v, s=5, alpha=0.25, color="#4477AA",
                   rasterized=True,
                   label="prefill" if L == L_values[0] else None)

    # Quadratic fit (L >= 1024)
    x_q_line = np.linspace(1024, max(L_values) * 1.02, 400)
    ax.plot(x_q_line, a_fq * x_q_line**2 + b_fq * x_q_line + c_fq,
            color="#CC3311", lw=1.8,
            label=f"prefill ($\\geq$1024 tokens):\n"
                  f"{a_fq*1e6:.2f} ns·L² + {b_fq*1000:.1f} µs·L + {c_fq:.1f} ms  "
                  f"($R^2$={r2_full_q:.3f})")

    # Cache-hit raw scatter (per-rep) + median connecting line.
    ax.scatter(hit_raw["L"], hit_raw["hit_ms"], s=5, alpha=0.55,
               color="#117733", rasterized=True,
               label="prefill w/ prefix-cache hit")
    ax.plot(cache["L"], cache["hit_p50"], color="#117733", lw=1.6,
            label=f"prefill w/ prefix-cache hit (median): $\\approx${hit_const:.0f} ms")

    ax.set_xscale("log")
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("TTFT (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "prefill_with_cache.pdf", dpi=200)
    fig.savefig(OUT / "prefill_with_cache.png", dpi=200)

    # Print the sanity-check residuals: cache_cost miss medians vs the quadratic.
    print("Cache-miss sanity check (cache_cost medians vs prefill_profile quadratic):")
    print(f"{'L':>6}  {'cache_cost':>11}  {'quad_pred':>10}  {'profile_med':>12}  "
          f"{'cc_resid':>9}")
    for _, row in cache.iterrows():
        L = row["L"]
        pred = a_fq * L**2 + b_fq * L + c_fq
        prof_med = float(np.median(per_L[L])) if L in per_L else float("nan")
        print(f"{int(L):>6}  {row['miss_p50']:>11.2f}  {pred:>10.2f}  "
              f"{prof_med:>12.2f}  {row['miss_p50']-pred:>+9.2f}")


if __name__ == "__main__":
    main()
