"""
Section 3 figure: prefill latency vs input token size (Qwen3-0.6B, batch=1).

Reads engine logs from section3/{200W,300W}/prefill_profile_data/{L}/engine_log.jsonl
produced by run_prefill_profile.py. Each cell is N_PROMPTS single-prompt prefill
trials at that L value.

Two-regime fit:
  low  L : constant overhead floor  (mean of prefill latencies for L below the knee)
  high L : linear in L              (slope = µs/token)
"""

from pathlib import Path
import json
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

WARMUP_DROP = 2          # drop first 2 prompts per L (server warmup)
KNEE_LOW = 512           # floor line spans L <= 512 only; L=1024 sits clearly above it
KNEE_HIGH = 8192         # linear-to-quadratic transition for the segmented fit
QUAD_FROM_1024 = True    # also fit a single quadratic over the full L >= 1024 range


def list_L_values():
    return sorted(int(p.name) for p in DATA.iterdir() if p.is_dir() and p.name.isdigit())


def parse_L(L: int) -> np.ndarray:
    """Return per-prompt prefill latency (ms) at this L, after warmup drop.

    Each fixed_batches sub-list was a single-prompt batch with max_tokens=2,
    yielding (iteration_start, prefill_step_end, decode_step_end) in order.
    Take every prefill_step_end -- the first step_end of each iteration.
    """
    eng = pd.read_json(DATA / str(L) / "engine_log.jsonl", lines=True)
    # Walk events; first step_end after each iter_start is prefill.
    rows = []
    sip = -1
    for _, row in eng.iterrows():
        if row["event"] == "iteration_start":
            sip = -1
        elif row["event"] == "step_end":
            sip += 1
            if sip == 0:  # prefill
                rows.append(float(row["step_duration_ms"]))
    arr = np.array(rows, dtype=float)
    if len(arr) > WARMUP_DROP:
        arr = arr[WARMUP_DROP:]
    return arr


def linfit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def main():
    L_values = list_L_values()
    per_L = {L: parse_L(L) for L in L_values}

    rows = []
    for L in L_values:
        v = per_L[L]
        rows.append({
            "L": L, "n": int(len(v)),
            "mean_ms": float(np.mean(v)), "median_ms": float(np.median(v)),
            "p25_ms": float(np.percentile(v, 25)), "p75_ms": float(np.percentile(v, 75)),
            "p99_ms": float(np.percentile(v, 99)), "std_ms": float(np.std(v)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "prefill_linearity_table.csv", index=False)
    print(table.to_string(index=False))

    # Three-regime fits
    floor_L = [L for L in L_values if L <= KNEE_LOW]
    lin_L = [L for L in L_values if 1024 <= L <= KNEE_HIGH]
    quad_L = [L for L in L_values if L >= KNEE_HIGH]

    def stack(Ls):
        xs = np.concatenate([np.full(len(per_L[L]), L) for L in Ls]).astype(float)
        ys = np.concatenate([per_L[L] for L in Ls])
        return xs, ys

    x_f, y_f = stack(floor_L) if floor_L else (np.array([]), np.array([]))
    x_l, y_l = stack(lin_L)
    x_q, y_q = stack(quad_L)

    floor_mean = float(np.mean(y_f)) if len(y_f) else float("nan")
    floor_std = float(np.std(y_f)) if len(y_f) else float("nan")

    slope_lin, b_lin, r2_lin = linfit(x_l, y_l)

    # Quadratic regime: fit y = a*L^2 + b*L + c
    coef_q = np.polyfit(x_q, y_q, 2)
    a_q, b_q, c_q = coef_q
    y_q_pred = a_q * x_q**2 + b_q * x_q + c_q
    r2_q = 1.0 - float(np.sum((y_q - y_q_pred)**2) / np.sum((y_q - y_q.mean())**2))

    # Alternative: single quadratic over the full L >= 1024 range.
    Ls_full = [L for L in L_values if L >= 1024]
    x_full, y_full = stack(Ls_full)
    coef_full = np.polyfit(x_full, y_full, 2)
    a_fq, b_fq, c_fq = coef_full
    y_fq_pred = a_fq * x_full**2 + b_fq * x_full + c_fq
    r2_full_q = 1.0 - float(np.sum((y_full - y_fq_pred)**2) / np.sum((y_full - y_full.mean())**2))

    # No-constant quadratic (y = a*L^2 + b*L) over L >= 1024, so the curve
    # naturally meets the floor at L_cross > 0.
    X_nc = np.column_stack([x_full**2, x_full])
    (a_nc, b_nc), *_ = np.linalg.lstsq(X_nc, y_full, rcond=None)
    y_nc_pred = a_nc * x_full**2 + b_nc * x_full
    r2_nc = 1.0 - float(np.sum((y_full - y_nc_pred)**2) / np.sum((y_full - y_full.mean())**2))
    # Crossing: a*L^2 + b*L = floor_mean
    disc_nc = b_nc**2 + 4 * a_nc * floor_mean
    L_cross = (-b_nc + np.sqrt(disc_nc)) / (2 * a_nc)

    with open(OUT / "prefill_linearity_fit.txt", "w") as f:
        f.write(f"{MODEL_SHORT} single-prompt prefill: three-regime model\n")
        f.write(f"Source: {DATA}\n")
        f.write(f"Warmup drop: first {WARMUP_DROP} prompts per L\n")
        f.write(f"Floor regime    L < {KNEE_LOW}     : {floor_L}\n")
        f.write(f"Linear regime   {KNEE_LOW} <= L <= {KNEE_HIGH}: {lin_L}\n")
        f.write(f"Quadratic regime L > {KNEE_HIGH}    : {quad_L}\n\n")
        if len(y_f):
            f.write(f"Floor: const = {floor_mean:.3f} ms (std {floor_std:.3f}, n={len(y_f)})\n\n")
        f.write(f"Linear (L in [{KNEE_LOW}, {KNEE_HIGH}]): y = a*L + b\n")
        f.write(f"  slope a   = {slope_lin*1000:.4f} us/tok  ({1000.0/slope_lin:.0f} tok/s)\n")
        f.write(f"  intercept = {b_lin:.4f} ms\n")
        f.write(f"  R^2       = {r2_lin:.5f}    n = {len(y_l)}\n\n")
        f.write(f"Quadratic (L > {KNEE_HIGH}): y = a2*L^2 + b1*L + c\n")
        f.write(f"  a2 (L^2) = {a_q*1e6:.4f} ns/tok^2\n")
        f.write(f"  b1 (L)   = {b_q*1000:.4f} us/tok\n")
        f.write(f"  c        = {c_q:.4f} ms\n")
        f.write(f"  R^2      = {r2_q:.5f}    n = {len(y_q)}\n\n")
        f.write(f"Single quadratic over L >= 1024 (alternative model):\n")
        f.write(f"  a2 (L^2) = {a_fq*1e6:.4f} ns/tok^2\n")
        f.write(f"  b1 (L)   = {b_fq*1000:.4f} us/tok\n")
        f.write(f"  c        = {c_fq:.4f} ms\n")
        f.write(f"  R^2      = {r2_full_q:.5f}    n = {len(y_full)}\n")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for L in L_values:
        v = per_L[L]
        ax.scatter(np.full(len(v), L), v, s=5, alpha=0.25, color="#4477AA", rasterized=True)

    # Floor: drawn out to L=1024 to close the visual gap with the quadratic.
    if len(y_f):
        x_lo_line = np.linspace(min(L_values), 1024, 64)
        ax.plot(x_lo_line, np.full_like(x_lo_line, floor_mean, dtype=float),
                color="#EE7733", lw=1.8,
                label=f"floor (L $\\leq$ 512): const $\\approx$ {floor_mean:.1f} ms")

    # Quadratic with constant: y = a*L^2 + b*L + c, fit over L >= 1024.
    x_q_line = np.linspace(1024, max(L_values) * 1.02, 400)
    ax.plot(x_q_line, a_fq * x_q_line**2 + b_fq * x_q_line + c_fq,
            color="#CC3311", lw=1.8,
            label=f"quadratic (L $\\geq$ 1024): "
                  f"{a_fq*1e6:.2f} ns·L² + {b_fq*1000:.1f} µs·L + {c_fq:.1f} ms  "
                  f"($R^2$={r2_full_q:.3f})")

    ax.set_xscale("log")
    # Linear y-axis: makes the actual functional shape (a quadratic) visible
    # as a smooth curve through the data, instead of compressing the small-L
    # deviations into apparent gaps on log-y.
    ax.set_xlabel("Input tokens (prompt length L)")
    ax.set_ylabel("Prefill latency (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "prefill_linearity.pdf", dpi=200)
    fig.savefig(OUT / "prefill_linearity.png", dpi=200)
    print()
    print(open(OUT / "prefill_linearity_fit.txt").read())


if __name__ == "__main__":
    main()
