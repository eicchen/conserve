"""
Section 3 figure: per-batch-bucket decode latency in the *rising* regime.

For each bucket we
  1. Bin the data by active_kv_total (quantile bins on raw samples).
  2. Locate the "knee" = first KV bin whose median exceeds the bucket-floor
     by THRESHOLD_MS, with sustained elevation in the next bin too.
  3. Fit a linear model lat = a*KV + b on raw samples to the right of the knee.
  4. Plot binned medians (full range), knee marker, and the per-bucket fit line.

Outputs:
  decode_knee.pdf, decode_knee.png
  decode_knee_table.csv     - per-bucket binned medians and knee marker
  decode_knee_fits.txt      - per-bucket slope/intercept/R^2/sample-count
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / ".conserve_root").exists())
import sys; sys.path.insert(0, str(REPO_ROOT / "profiling"))
from config import GPU_MON_ROOT, MODEL_SHORT, MODEL_DATA_DIR


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = (GPU_MON_ROOT / MODEL_SHORT / "decode")
OUT = MODEL_DATA_DIR / "paper" / "section3" / "fig4"
OUT.mkdir(parents=True, exist_ok=True)
SUBMITTED = [1, 2, 4, 8, 16, 32, 64]
INPUT_LEN = 8
WARMUP_DROP = 32

BATCH_BUCKETS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
BUCKET_COLORS = ["#332288", "#117733", "#88CCEE", "#DDCC77", "#CC6677", "#882255"]

N_BINS = 24
THRESHOLD_MS = 0.5     # how far above the floor a bin median must sit to count as "rising"
MIN_BIN_SAMPLES = 200


def parse_decode(submitted_b):
    df = pd.read_json(BASE / str(submitted_b) / "vllm_core_log.jsonl", lines=True)
    df = df[df["event"] == "step_end"].reset_index(drop=True)
    df["latency_ms"] = df["timestamp"].diff().dt.total_seconds() * 1000.0

    pos = {}
    kv = np.zeros(len(df), dtype=np.int64)
    bact = np.zeros(len(df), dtype=np.int32)
    is_dec = np.zeros(len(df), dtype=bool)
    for i, rids in enumerate(df["executed_request_ids"].tolist()):
        if not isinstance(rids, list) or not rids:
            continue
        s = 0
        pref = False
        for r in rids:
            k = pos.get(r, 0)
            if k == 0:
                pref = True
            s += INPUT_LEN + max(k - 1, 0)
            pos[r] = k + 1
        bact[i] = len(rids)
        kv[i] = s
        is_dec[i] = not pref
    df["batch_active"] = bact
    df["active_kv_total"] = kv
    df["pure_decode"] = is_dec
    df = df.iloc[1:]
    df = df[df["pure_decode"]].iloc[WARMUP_DROP:]
    return df[["batch_active", "active_kv_total", "latency_ms"]].dropna()


def bucket_label(lo, hi):
    return f"B = {lo}" if lo == hi else f"B ∈ [{lo}, {hi}]"


def linfit(x, y):
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def find_knee(bin_df, threshold_ms):
    """Return knee KV (left edge) or None if no rising regime detected.

    Floor = median of lowest 1/3 of bin medians (robust to noise).
    Knee  = first bin where the median > floor + max(threshold, 1.5*MAD-of-floor) AND
            the *next two* bin medians are non-decreasing and remain above the threshold.
    Falls back to "first bin above threshold" if there is one and it's the last bin.
    """
    if len(bin_df) < 3:
        return None, None
    n_floor = max(3, len(bin_df) // 3)
    floor_bins = bin_df["lat_p50"].iloc[:n_floor]
    floor = float(floor_bins.median())
    mad = float(np.median(np.abs(floor_bins - floor)))
    delta = max(threshold_ms, 1.5 * mad)
    threshold = floor + delta
    medians = bin_df["lat_p50"].to_numpy()
    n = len(medians)
    for i in range(n - 2):
        if (medians[i] > threshold and medians[i + 1] > threshold and medians[i + 2] > threshold
                and medians[i + 1] >= medians[i] and medians[i + 2] >= medians[i + 1]):
            return float(bin_df["kv_lo"].iloc[i]), floor
    # Soft fallback: a single rising bin at the very end with a large jump
    if n >= 2 and medians[-1] > floor + 2 * delta:
        return float(bin_df["kv_lo"].iloc[-1]), floor
    return None, floor


def main():
    parts = [parse_decode(b) for b in SUBMITTED]
    data = pd.concat(parts, ignore_index=True)
    data = data[np.isfinite(data["latency_ms"])]

    bin_rows = []
    knees = {}
    fits = {}
    for lo, hi in BATCH_BUCKETS:
        sl = data[(data["batch_active"] >= lo) & (data["batch_active"] <= hi)]
        if not len(sl):
            continue
        kv = sl["active_kv_total"].to_numpy()
        lat = sl["latency_ms"].to_numpy()
        edges = np.unique(np.quantile(kv, np.linspace(0, 1, N_BINS + 1)))
        rows_b = []
        for i in range(len(edges) - 1):
            mask = (kv >= edges[i]) & (kv <= edges[i + 1])
            if mask.sum() < MIN_BIN_SAMPLES:
                continue
            rows_b.append({
                "B_lo": lo, "B_hi": hi,
                "kv_lo": int(edges[i]), "kv_hi": int(edges[i + 1]),
                "kv_mid": float((edges[i] + edges[i + 1]) / 2),
                "n": int(mask.sum()),
                "lat_p50": float(np.percentile(lat[mask], 50)),
                "lat_p25": float(np.percentile(lat[mask], 25)),
                "lat_p75": float(np.percentile(lat[mask], 75)),
            })
        bin_df = pd.DataFrame(rows_b)
        bin_rows.append(bin_df)
        knee_kv, floor = find_knee(bin_df, THRESHOLD_MS)
        knees[(lo, hi)] = (knee_kv, floor)
        if knee_kv is not None:
            mask = sl["active_kv_total"] >= knee_kv
            x = sl.loc[mask, "active_kv_total"].to_numpy(dtype=float)
            y = sl.loc[mask, "latency_ms"].to_numpy(dtype=float)
            fit = linfit(x, y)
            fits[(lo, hi)] = (fit, int(mask.sum()))
        else:
            fits[(lo, hi)] = (None, 0)

    table = pd.concat(bin_rows, ignore_index=True)
    table.to_csv(OUT / "decode_knee_table.csv", index=False)

    with open(OUT / "decode_knee_fits.txt", "w") as f:
        f.write("Per-batch-bucket decode latency: rising-regime fit\n")
        f.write(f"Bin count: {N_BINS} quantile bins per bucket; min {MIN_BIN_SAMPLES} samples per bin\n")
        f.write(f"Knee detection: bin median > floor + {THRESHOLD_MS} ms AND next bin too\n")
        f.write(f"  (floor = min bin median over lower half of KV range)\n\n")
        for lo, hi in BATCH_BUCKETS:
            knee_kv, floor = knees.get((lo, hi), (None, None))
            fit, n_above = fits.get((lo, hi), (None, 0))
            label = bucket_label(lo, hi)
            f.write(f"--- {label} ---\n")
            if floor is not None:
                f.write(f"  floor (low-KV median) = {floor:.3f} ms\n")
            if knee_kv is None:
                f.write(f"  No rising regime detected within measured KV range (essentially flat).\n\n")
                continue
            f.write(f"  knee KV cutoff        = {int(knee_kv):,} tokens\n")
            f.write(f"  samples above knee    = {n_above:,}\n")
            if fit is None:
                f.write(f"  Insufficient samples to fit.\n\n")
                continue
            slope, intercept, r2 = fit
            f.write(f"  fit  lat = a*KV + b\n")
            f.write(f"    a = {slope*1000:.4f} us per active KV token\n")
            f.write(f"    b = {intercept:.3f} ms\n")
            f.write(f"    R^2 = {r2:.4f}\n\n")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (lo, hi), color in zip(BATCH_BUCKETS, BUCKET_COLORS):
        sub = table[(table["B_lo"] == lo) & (table["B_hi"] == hi)]
        if not len(sub):
            continue
        ax.plot(sub["kv_mid"], sub["lat_p50"], color=color, lw=1.0, marker="o",
                ms=3, alpha=0.55, label=bucket_label(lo, hi))
        knee_kv, floor = knees.get((lo, hi), (None, None))
        fit, _ = fits.get((lo, hi), (None, 0))
        if knee_kv is not None and fit is not None:
            slope, intercept, r2 = fit
            xx = np.linspace(knee_kv, sub["kv_hi"].max(), 80)
            ax.plot(xx, slope * xx + intercept, color=color, lw=2.0, ls="--",
                    label=f"  fit: {slope*1000:.3f} µs/KV  ($R^2$={r2:.2f})")
            ax.axvline(knee_kv, color=color, lw=0.6, ls=":", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Total active KV cache (tokens)")
    ax.set_ylabel("Decode step latency p50 (ms)")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "decode_knee.pdf", dpi=200)
    fig.savefig(OUT / "decode_knee.png", dpi=200)

    print(open(OUT / "decode_knee_fits.txt").read())


if __name__ == "__main__":
    main()
