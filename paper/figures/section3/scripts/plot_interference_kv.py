"""
Plot the L_decoder interference sweep.

Mirrors plot_interference.py's layout: two side-by-side panels (miss / hit)
with shared y-axis. X-axis is L_decoder; one line per L_prefill.

The metric is the engine-step duration of the step that executed the prefill
request — i.e., the stall every concurrent decoder experiences while that
prefill goes through.
"""

import json
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = (REPO_ROOT / "paper/figures/section3/output/300W/interference_kv_data")
OUT = (REPO_ROOT / "paper/figures/section3/output/300W")


def _strip(rid: str) -> str:
    if rid.startswith("cmpl-"):
        rid = rid[len("cmpl-"):]
    if "-" in rid:
        rid = rid.rsplit("-", 1)[0]
    return rid


def parse_steps():
    df = pd.read_json(DATA / "server_core.jsonl", lines=True)
    df["ts"] = pd.to_datetime(df["timestamp"])
    starts = df[df["event"] == "step_start"][["ts"]].reset_index(drop=True)
    ends = df[df["event"] == "step_end"].reset_index(drop=True)
    assert len(starts) == len(ends), (len(starts), len(ends))
    rows = []
    for i in range(len(ends)):
        e = ends.iloc[i]
        exec_rids = set(_strip(r) for r in (e["executed_request_ids"] or []))
        fin_rids = set(_strip(r) for r in (e["finished_request_ids"] or []))
        rows.append({
            "ts_start": starts.iloc[i]["ts"],
            "ts_end": e["ts"],
            "duration_ms": (e["ts"] - starts.iloc[i]["ts"]).total_seconds() * 1000.0,
            "executed_rids": exec_rids,
            "finished_rids": fin_rids,
        })
    return pd.DataFrame(rows)


def main():
    cells_info = json.loads((DATA / "cells.json").read_text())
    cells = cells_info["cells"]
    print(f"Parsing {len(cells)} cells")

    steps = parse_steps()
    rows = []
    for cell in cells:
        prid = cell["prefill_rid"]
        prefill_steps = steps[steps["executed_rids"].apply(lambda s: prid in s)].copy()
        if prefill_steps.empty:
            continue
        completing = prefill_steps[prefill_steps["finished_rids"].apply(lambda s: prid in s)]
        if completing.empty:
            continue
        compute_steps = prefill_steps[prefill_steps["ts_end"] <= completing["ts_end"].iloc[0]]
        durations = compute_steps["duration_ms"].to_numpy()
        rows.append({
            "cell_idx": cell["cell_idx"],
            "rep": cell.get("rep", 0),
            "B_decode": cell["B_decode"],
            "L_decoder": cell["L_decoder"],
            "L_prefill": cell["L_prefill"],
            "cache_hit": cell["cache_hit"],
            "n_prefill_steps": int(len(durations)),
            "prefill_step_mean_ms": float(np.mean(durations)),
            "prefill_step_max_ms": float(np.max(durations)),
            "prefill_step_p50_ms": float(np.median(durations)),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["cache_hit", "L_prefill", "L_decoder", "rep"]).reset_index(drop=True)
    summary.to_csv(OUT / "interference_kv_summary.csv", index=False)
    print(summary.to_string(index=False))

    L_pref_vals = sorted(summary["L_prefill"].unique())
    cmap = plt.get_cmap("viridis")
    L_colors = {L: cmap(i / max(1, len(L_pref_vals) - 1))
                for i, L in enumerate(L_pref_vals)}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    step_col = "prefill_step_mean_ms"
    step_max = float(np.nanmax(summary[step_col]))
    step_min = float(np.nanmin(summary[step_col]))

    for col_idx, cache_hit in enumerate([False, True]):
        cond = "cache miss (fresh prefill)" if not cache_hit else "cache hit (prefix-cached prefill)"
        sub = summary[summary["cache_hit"] == cache_hit]
        ax = axes[col_idx]
        for L in L_pref_vals:
            s = sub[sub["L_prefill"] == L]
            # Raw per-replicate points + mean line.
            ax.scatter(s["L_decoder"], s[step_col], s=6, alpha=0.25,
                       color=L_colors[L], rasterized=True)
            g = s.groupby("L_decoder")[step_col].mean().sort_index()
            ax.plot(g.index, g.values, lw=1.8, color=L_colors[L],
                    label=f"L_prefill = {L}")
        ax.axhline(14.5, color="gray", ls=":", lw=0.9, alpha=0.6,
                   label="isolated decode baseline" if col_idx == 0 else None)
        ax.set_title(f"{cond}", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Per-decoder context length  L_decoder  (tokens)")
        ax.set_ylim(step_min * 0.7, step_max * 1.4)
        ax.grid(True, which="both", alpha=0.3, linestyle=":")
        if col_idx == 0:
            ax.set_ylabel("Prefill-step duration (ms)\n(= stall every active decoder experiences)")
            ax.legend(fontsize=8, loc="upper left", frameon=False)
        if col_idx == 1:
            ax.legend(fontsize=8, loc="upper left", frameon=False,
                      title_fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "interference_kv.pdf", dpi=200)
    fig.savefig(OUT / "interference_kv.png", dpi=200)
    print("Saved interference_kv.pdf / .png")


if __name__ == "__main__":
    main()
