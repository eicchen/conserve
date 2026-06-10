"""Section-5 power-cap comparison: 4 policies x 2 power configs.

Same 3x4 grid as the headline (TTFET / Last-turn TBT / E2E in rows of gmean,
p95, SLO violation %, plus Tokens-per-Joule spanning all 3 rows), but every
policy is plotted twice: solid line = uncapped (p300_d300), dashed line =
decoder cap (p300_d200). Color encodes the policy.

Energy correction for AMPD (per_turn_adaptive_disagg_decoders_p10): the live
prefiller doesn't actually execute the wrong-predict turn 2+ prefills (they
are simulated via VirtualPrefiller blocks), so the recorded prefiller energy
under-counts. We add  avg_prefiller_power * total_wrong_predict_base_time,
where the base time per wrong-predict event is read from run.log.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import s5_metrics as s5

OUT = Path(__file__).parent.parent / "output"
RPS = [0.5, 0.75, 1, 1.25, 1.5, 1.634]

POLICIES = [
    ("no_disagg",                              "Collocated",  "#4477AA", "s"),
    ("all_disagg",                             "Full Disagg", "#CC3311", "^"),
    ("per_turn_adaptive_disagg_decoders_p10",  "AMPD",        "#AA4499", "v"),
    ("adaptive_3eng",                          "ConServe",   "#117733", "o"),
]
POWER_CFGS = [
    ("p300_d300", "300/300", "-"),
    ("p300_d200", "300/200", "--"),
]
COLS = [("TTFET", "TTFET"), ("TBT", "Last-turn TBT"), ("E2E", "E2E")]
P95 = lambda x: np.percentile(x, 95)


_BASE_MS_RE = re.compile(r"base ([\d.]+) ms")


def ampd_added_prefill_seconds(run_dir):
    """Sum of WRONG PREDICT `base` durations (the virtual prefiller occupancy
    that the live prefiller didn't actually compute). Read from run.log."""
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return 0.0
    total_ms = 0.0
    with open(log) as f:
        for line in f:
            if "WRONG PREDICT" not in line:
                continue
            m = _BASE_MS_RE.search(line)
            if m:
                total_ms += float(m.group(1))
    return total_ms / 1000.0


def workload_energy_corrected(cfg, policy, rps):
    """Same as s5.workload_energy, but for AMPD we additionally bill the
    prefiller for the virtual-prefill busy time at the prefiller's average
    measured power."""
    base = s5.workload_energy(cfg, policy, rps)
    if not policy.startswith("per_turn_adaptive_disagg_decoders"):
        return base
    pref_dir = s5.prefiller_dir_for(cfg, rps)
    ps_p = pd.read_csv(pref_dir / "per_step_latency.csv")
    t0p = float(ps_p["start_time"].min())
    t1p = float(ps_p["end_time"].max())
    span = t1p - t0p
    if span <= 0:
        return base
    e_pref = s5.run_energy_joules(str(pref_dir), t0p, t1p, gpus=["0"])
    avg_p_pref = e_pref / span
    added_sec = ampd_added_prefill_seconds(
        s5.RPS_SWEEP / cfg / policy / f"rps_{rps}")
    return base + avg_p_pref * added_sec


def main():
    base = s5.load_baseline()
    base4 = s5.baseline_lastturn_tbt()

    # series[(cfg, label)] = {col: {rps: array}}, {col: {rps: pct}}, {rps: tpj}
    norm, slo, tpj = {}, {}, {}
    for cfg, _, _ in POWER_CFGS:
        for pol, label, *_ in POLICIES:
            key = (cfg, label)
            norm[key] = {c: {} for c, _ in COLS}
            slo[key] = {c: {} for c, _ in COLS}
            tpj[key] = {}

    for cfg, _, _ in POWER_CFGS:
        for pol, label, *_ in POLICIES:
            key = (cfg, label)
            for rps in RPS:
                run_dir = s5.RPS_SWEEP / cfg / pol / f"rps_{rps}"
                if not (run_dir / "per_step_latency.csv").exists():
                    continue
                df = s5.load_run(cfg, pol, rps).set_index("conv_id").join(base)
                norm[key]["TTFET"][rps] = (df["ttfet"] / df["base_ttfet"]).dropna().to_numpy()
                norm[key]["E2E"][rps] = (df["e2e"] / df["base_e2e"]).dropna().to_numpy()
                norm[key]["TBT"][rps] = s5.lastturn_tbt_tokens(cfg, pol, rps)

                vt = df[["ttfet", "slo_ttfet"]].dropna()
                slo[key]["TTFET"][rps] = float((vt["ttfet"] > vt["slo_ttfet"]).mean() * 100)
                ve = df[["e2e", "slo_e2e"]].dropna()
                slo[key]["E2E"][rps] = float((ve["e2e"] > ve["slo_e2e"]).mean() * 100)
                gaps = s5._iter4_ordered_gaps(str(run_dir), pol)
                viol = tot = 0
                for conv, g in gaps.items():
                    if conv in base4 and g:
                        tot += 1
                        viol += np.mean(g) > 5.0 * base4[conv]
                slo[key]["TBT"][rps] = 100.0 * viol / tot if tot else float("nan")

                tokens = s5.workload_tokens_full_context(cfg, pol, rps)
                energy = workload_energy_corrected(cfg, pol, rps)
                tpj[key][rps] = tokens / energy

    fig = plt.figure(figsize=(14.0, 6.6))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.05],
                          wspace=0.28, hspace=0.18)

    qos_axes = [[None] * 3 for _ in range(3)]
    for j in range(3):
        qos_axes[0][j] = fig.add_subplot(gs[0, j])
        qos_axes[1][j] = fig.add_subplot(gs[1, j], sharex=qos_axes[0][j])
        qos_axes[2][j] = fig.add_subplot(gs[2, j], sharex=qos_axes[0][j])
    for j in range(1, 3):
        qos_axes[0][j].sharey(qos_axes[0][0])
        qos_axes[1][j].sharey(qos_axes[1][0])
        qos_axes[2][j].sharey(qos_axes[2][0])

    rowdefs = [("gmean", gmean), ("p95", P95), ("SLO violation (%)", None)]
    for i, (rlabel, stat) in enumerate(rowdefs):
        for j, (col, ctitle) in enumerate(COLS):
            ax = qos_axes[i][j]
            for cfg, cfg_label, ls in POWER_CFGS:
                for pol, plabel, color, mk in POLICIES:
                    key = (cfg, plabel)
                    if stat is not None:
                        xs = [r for r in RPS
                              if norm[key][col].get(r) is not None
                              and len(norm[key][col][r])]
                        ys = [stat(norm[key][col][r]) for r in xs]
                    else:
                        xs = [r for r in RPS if r in slo[key][col]]
                        ys = [slo[key][col][r] for r in xs]
                    ax.plot(xs, ys, color=color, marker=mk, ms=5, lw=1.6,
                            linestyle=ls, label=f"{plabel} {cfg_label}",
                            markerfacecolor=(color if ls == "-" else "white"),
                            markeredgecolor=color, markeredgewidth=1.2)
            if stat is not None:
                ax.axhline(5.0, color="#BB5566", ls="--", lw=1.0, alpha=0.8)
                ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.7)
                ax.set_yscale("symlog", linthresh=5.0, linscale=2.0)
                ax.set_yticks([1, 2, 3, 4, 5, 10, 20, 50, 100])
                ax.set_yticklabels(["1", "2", "3", "4", "5", "10", "20", "50", "100"],
                                   fontsize=8)
                ax.set_ylim(bottom=0.8)
            else:
                ax.set_yscale("symlog", linthresh=25.0, linscale=2.0)
                ax.set_yticks([0, 5, 10, 15, 20, 25, 50, 100])
                ax.set_yticklabels(["0", "5", "10", "15", "20", "25", "50", "100"],
                                   fontsize=8)
                ax.set_ylim(0, 110)
            ax.grid(True, which="both", alpha=0.22, ls=":")
            if i == 0:
                ax.set_title(ctitle, fontsize=10)
            if j == 0:
                ax.set_ylabel(rlabel + ("  (normalized)" if stat is not None else ""),
                              fontsize=9)
            if i == 2:
                ax.set_xlabel("Request rate (conv / s)", fontsize=9)
                ax.set_xticks(RPS)
                ax.set_xticklabels([str(r) for r in RPS], fontsize=8)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    eax = fig.add_subplot(gs[:, 3])
    for cfg, cfg_label, ls in POWER_CFGS:
        for pol, plabel, color, mk in POLICIES:
            key = (cfg, plabel)
            xs = [r for r in RPS if r in tpj[key]]
            ys = [tpj[key][r] for r in xs]
            eax.plot(xs, ys, color=color, marker=mk, ms=6, lw=1.8,
                     linestyle=ls,
                     markerfacecolor=(color if ls == "-" else "white"),
                     markeredgecolor=color, markeredgewidth=1.2)
    eax.set_title("Tokens per Joule", fontsize=10)
    eax.set_ylabel("tokens / J", fontsize=9)
    eax.set_xlabel("Request rate (conv / s)", fontsize=9)
    eax.set_xticks(RPS)
    eax.set_xticklabels([str(r) for r in RPS], fontsize=8)
    eax.tick_params(axis="y", labelsize=8)
    eax.grid(True, alpha=0.25, ls=":")

    # Build a two-row legend: top row = policy (color), bottom row = power cfg (linestyle).
    policy_handles = [Line2D([0], [0], color=c, marker=m, ms=5, lw=1.8, label=lab)
                      for _, lab, c, m in POLICIES]
    cfg_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-",  label="300W / 300W (uncapped)"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="300W / 200W (decoder cap)"),
    ]
    slo_handle = Line2D([0], [0], color="#BB5566", ls="--", lw=1.0,
                        label="SLO (5$\\times$ baseline)")
    fig.legend(handles=policy_handles + cfg_handles + [slo_handle],
               labels=[h.get_label() for h in policy_handles + cfg_handles + [slo_handle]],
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=7, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "powercap.pdf", dpi=200)
    fig.savefig(OUT / "powercap.png", dpi=200)
    print("Saved powercap.pdf / .png\n")

    # Textual dump for the report.
    cfgs_short = [(cfg, lab) for cfg, lab, _ in POWER_CFGS]
    for col, ctitle in COLS:
        print(f"=== {ctitle} ===")
        for rlabel, stat in rowdefs:
            print(f"  {rlabel}")
            header = "rps    " + "  ".join(
                f"{p:>5}@{cfg_lab:<7}" for _, p, *_ in POLICIES
                for cfg, cfg_lab in cfgs_short)
            print("    " + header)
            for rps in RPS:
                vals = []
                for pol, plabel, *_ in POLICIES:
                    for cfg, cfg_lab in cfgs_short:
                        key = (cfg, plabel)
                        if stat is not None:
                            v = norm[key][col].get(rps)
                            vals.append(f"{stat(v):.2f}" if v is not None and len(v) else "-")
                        else:
                            s = slo[key][col].get(rps)
                            vals.append(f"{s:.0f}%" if s is not None else "-")
                row = " ".join(f"{v:>13}" for v in vals)
                print(f"    rps {rps:>5}: {row}")
        print()

    print("=== Tokens per Joule (AMPD energy corrected) ===")
    header = "  ".join(f"{p:>5}@{cfg_lab:<7}" for _, p, *_ in POLICIES
                       for cfg, cfg_lab in cfgs_short)
    print("    " + header)
    for rps in RPS:
        vals = []
        for pol, plabel, *_ in POLICIES:
            for cfg, cfg_lab in cfgs_short:
                key = (cfg, plabel)
                v = tpj[key].get(rps)
                vals.append(f"{v:.2f}" if v is not None else "-")
        row = " ".join(f"{v:>13}" for v in vals)
        print(f"    rps {rps:>5}: {row}")


if __name__ == "__main__":
    main()
