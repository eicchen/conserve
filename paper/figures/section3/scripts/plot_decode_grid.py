"""
Section 3 figures from the controlled (B, L) decode grid.

For each cell we have K_REPS iterations × N_DECODE step_ends. The FIRST
step_end after each iteration_start is the prefill; the remaining N_DECODE-1
are decode steps with known per-request KV = L + step_idx_in_iter - 1 and
fixed batch_active = B.

Outputs (in this directory):
  decode_grid_panel_B_at_L.{pdf,png}    - latency vs B, lines per L
  decode_grid_panel_L_at_B.{pdf,png}    - latency vs L, lines per B
  decode_grid_collapse_BxL.{pdf,png}    - latency vs B*L (total active KV)
  decode_grid_heatmap.{pdf,png}         - 2D median-latency surface
  decode_grid_fit.txt                   - planar fit lat = a + b*B + c*B*L
  decode_grid_cell_summary.csv          - per-cell median/p25/p75/p99
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "config"))
from config import MODEL_SHORT, MODEL_DATA_DIR

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA = (MODEL_DATA_DIR / "paper" / "section3" / "profiling" / "decode_grid_data")
OUT = (MODEL_DATA_DIR / "paper" / "section3" / "fig4")


def parse_cell(cell_idx: int, B: int, L: int) -> pd.DataFrame:
    """Tag each step as pure-decode iff every one of the B requests has
    already produced output in some previous step within this iter.

    Signal source: core_log's `finished_request_ids` — vLLM only emits a
    request ID there when that request produces output for the step
    (= completed prefill for chunked-prefill last chunk, or running in
    steady-state decode). The very first time we see a request in
    finished_request_ids is its prefill-finishing step; subsequent steps
    are pure decode. A step where ALL B request ids have already been seen
    in a strictly earlier step is therefore a pure decode step.

    Engine_log step_ends and core_log step_ends are 1:1 aligned in order;
    we use the engine_log timestamps to track iteration boundaries and the
    core_log to determine the prefill/decode classification.
    """
    eng = pd.read_json(DATA / f"engine_cell_{cell_idx:03d}.jsonl", lines=True)
    core = pd.read_json(DATA / f"core_cell_{cell_idx:03d}.jsonl", lines=True)
    core_se = core[core["event"] == "step_end"].reset_index(drop=True)

    iter_idx = -1
    sip = -1
    eng_step_pos = 0
    seen_in_iter: set = set()
    decode_step_idx = -1
    rows = []

    for _, row in eng.iterrows():
        if row["event"] == "iteration_start":
            iter_idx += 1
            sip = -1
            seen_in_iter = set()
            decode_step_idx = -1
        elif row["event"] == "step_end":
            sip += 1
            # Pure decode iff every request has appeared in a strictly earlier step.
            is_pure_decode = len(seen_in_iter) == B
            if is_pure_decode:
                decode_step_idx += 1
            # Now consume this step's finished_request_ids (becomes "seen" for the next step).
            core_row = core_se.iloc[eng_step_pos]
            fin = core_row.get("finished_request_ids") or []
            if isinstance(fin, list):
                seen_in_iter |= set(fin)
            rows.append({
                "iter": iter_idx,
                "step_in_iter": sip,
                "latency_ms": row["step_duration_ms"],
                "is_pure_decode": is_pure_decode,
                "decode_step_idx": decode_step_idx if is_pure_decode else -1,
                "n_seen_after": len(seen_in_iter),
            })
            eng_step_pos += 1

    df = pd.DataFrame(rows)
    df["cell_idx"] = cell_idx
    df["B"] = B
    df["L"] = L
    df["is_prefill"] = ~df["is_pure_decode"]

    # Per-request KV at start of a pure-decode step = L + decode_step_idx.
    # (Prefill produced 1 token, leaving KV = L; first pure-decode step reads L tokens.)
    df["kv_per_req"] = np.where(df["is_pure_decode"], L + df["decode_step_idx"], np.nan)
    df["active_kv_total"] = B * df["kv_per_req"]
    return df


def main():
    plan = json.loads((DATA / "plan.json").read_text())
    cells = [(c["cell_idx"], c["B"], c["L"]) for c in plan["cells"]]

    parts = []
    for cell_idx, B, L in cells:
        parts.append(parse_cell(cell_idx, B, L))
    df = pd.concat(parts, ignore_index=True)
    decode = df[~df["is_prefill"]].copy()
    print(f"Total decode steps across all cells: {len(decode)}")

    # Per-cell summary
    summary = (decode.groupby(["cell_idx", "B", "L"])
               .agg(n=("latency_ms", "size"),
                    lat_p25=("latency_ms", lambda x: float(np.percentile(x, 25))),
                    lat_p50=("latency_ms", lambda x: float(np.percentile(x, 50))),
                    lat_p75=("latency_ms", lambda x: float(np.percentile(x, 75))),
                    lat_p99=("latency_ms", lambda x: float(np.percentile(x, 99))),
                    lat_mean=("latency_ms", "mean"),
                    lat_std=("latency_ms", "std"),
                    kv_per_req_mean=("kv_per_req", "mean"),
                    active_kv_mean=("active_kv_total", "mean"))
               .reset_index())
    summary.to_csv(OUT / "decode_grid_cell_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Planar fit lat = a + b*B + c*active_kv_total (over all decode steps)
    y = decode["latency_ms"].to_numpy()
    X = np.column_stack([
        np.ones(len(decode)),
        decode["B"].to_numpy(),
        decode["active_kv_total"].to_numpy(),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    yhat = X @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(((y - yhat) ** 2).mean()))

    fit_path = OUT / "decode_grid_fit.txt"
    with open(fit_path, "w") as f:
        f.write("Planar fit on decode-only samples\n")
        f.write(f"Total samples: {len(decode)}\n")
        f.write(f"lat (ms) = {a:.4f} + {b*1000:.3f} us/req * B + {c*1000:.4f} us/KV * sum_KV\n")
        f.write(f"  alpha (intercept)            = {a:.4f} ms\n")
        f.write(f"  beta  (per-request FFN cost) = {b*1000:.3f} us/req\n")
        f.write(f"  gamma (per-KV-token cost)    = {c*1000:.4f} us/KV-tok\n")
        f.write(f"  R^2                          = {r2:.4f}\n")
        f.write(f"  RMSE                         = {rmse:.3f} ms\n")
    print("\n" + fit_path.read_text())

    # Visual style
    L_values = sorted(decode["L"].unique())
    B_values = sorted(decode["B"].unique())
    cmap_L = plt.get_cmap("viridis")
    cmap_B = plt.get_cmap("plasma")
    L_colors = {L: cmap_L(i / max(1, len(L_values) - 1)) for i, L in enumerate(L_values)}
    B_colors = {B: cmap_B(i / max(1, len(B_values) - 1)) for i, B in enumerate(B_values)}

    # === Plot 1: latency vs B, one curve per L ===
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for L in L_values:
        sub = summary[summary["L"] == L].sort_values("B")
        if len(sub) >= 2:
            ax.plot(sub["B"], sub["lat_p50"], marker="o", lw=1.5, ms=4,
                    color=L_colors[L], label=f"L = {L}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size B")
    ax.set_ylabel("Decode step latency p50 (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="per-request KV", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decode_grid_panel_B_at_L.pdf", dpi=200)
    fig.savefig(OUT / "decode_grid_panel_B_at_L.png", dpi=200)

    # === Plot 2: latency vs L, one curve per B ===
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for B in B_values:
        sub = summary[summary["B"] == B].sort_values("L")
        if len(sub) >= 2:
            ax.plot(sub["L"], sub["lat_p50"], marker="o", lw=1.5, ms=4,
                    color=B_colors[B], label=f"B = {B}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Per-request KV length L (tokens)")
    ax.set_ylabel("Decode step latency p50 (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="batch size", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decode_grid_panel_L_at_B.pdf", dpi=200)
    fig.savefig(OUT / "decode_grid_panel_L_at_B.png", dpi=200)

    # === Plot 3: latency vs total active KV (B * L), colored by B ===
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for B in B_values:
        sub = summary[summary["B"] == B].sort_values("L")
        if not len(sub):
            continue
        ax.plot(sub["active_kv_mean"], sub["lat_p50"], marker="o", lw=1.4, ms=4,
                color=B_colors[B], label=f"B = {B}")
    ax.set_xscale("log")
    ax.set_xlabel("Total active KV cache (tokens) = B × L")
    ax.set_ylabel("Decode step latency p50 (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="batch size", title_fontsize=8)
    fit_text = (f"$\\mathrm{{lat}} \\approx {a:.1f} + {b*1000:.0f}\\,\\mu s\\cdot B"
                f" + {c*1000:.3f}\\,\\mu s\\cdot\\mathrm{{KV}}$  ($R^2$={r2:.2f})")
    ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    fig.tight_layout()
    fig.savefig(OUT / "decode_grid_collapse_BxL.pdf", dpi=200)
    fig.savefig(OUT / "decode_grid_collapse_BxL.png", dpi=200)

    # === Plot 4: 2D heatmap of MEAN latency over (B, L) ===
    pivot = summary.pivot(index="B", columns="L", values="lat_mean")
    pivot = pivot.sort_index(ascending=False)  # large B on top for intuitive reading
    # Keep only power-of-2 batch sizes on the y-axis for readability.
    pivot = pivot.loc[[b for b in pivot.index if b > 0 and (b & (b - 1)) == 0]]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    vmin = float(np.nanmin(pivot.values))
    vmax = float(np.nanmax(pivot.values))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.set_xlabel("Per-request context length L (tokens)", fontsize=12)
    ax.set_ylabel("Batch size  B", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean TBT (ms)", fontsize=12)

    # Per-cell annotation. Threshold below 0.5 so very-dark cells get white text.
    thresh = vmin + 0.40 * (vmax - vmin)
    n_cells = pivot.notna().sum().sum()
    # annot_fs = 8 if n_cells <= 40 else (7 if n_cells <= 80 else 6)
    annot_fs = 10
    for i, B in enumerate(pivot.index):
        for j, L in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                txt_color = "black" if v > thresh else "white"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=annot_fs, color=txt_color, fontweight="bold")

    # Iso-active-KV diagonals as visual guides for "constant total memory".
    # active_KV = B * L  =>  log2(B) = log2(KV) - log2(L)
    L_grid = np.array(pivot.columns, dtype=float)
    B_grid = np.array(pivot.index, dtype=float)
    L_idx = {L: i for i, L in enumerate(pivot.columns)}
    for kv_iso in [65536]:
        pts = []
        for L in pivot.columns:
            B_target = kv_iso / L
            if B_target < min(B_grid) or B_target > max(B_grid):
                continue
            log_Bs = np.log2(B_grid)
            row = float(np.interp(np.log2(B_target), log_Bs[::-1], np.arange(len(B_grid))[::-1]))
            pts.append((L_idx[L], row))
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="white", lw=1.8, ls="--", alpha=0.75)
            # Label to the LEFT of the topmost feasible point. Use a solid
            # dark background so white text reads clearly against the figure
            # background (outside the heatmap area, where it would otherwise
            # be white-on-white).
            kx, ky = xs[0], ys[0]
            label = f"B·L = {kv_iso//1024}k"
            ax.text(kx - 0.15, ky - 0.35, label, color="white", fontsize=10,
                    ha="left", va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="0.15",
                              ec="white", lw=0.4))
    fig.tight_layout()
    fig.savefig(OUT / "decode_grid_heatmap.pdf", dpi=200)
    fig.savefig(OUT / "decode_grid_heatmap.png", dpi=200)

    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
