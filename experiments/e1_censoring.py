"""
E1: Censoring-Aware Analysis
=============================
Kaplan-Meier estimator + log-rank test for time-to-detection.
Restricted Mean Survival Time (RMST) comparison.
No re-run — reanalyzes existing e1_metrics.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIGS = ROOT / "figures"

BUDGET = 700  # censoring point
DETECTORS = ["entropy_weighted", "uniform", "adwin"]
DET_LABELS = {"entropy_weighted": "C1 (ours)", "uniform": "B4: Uniform", "adwin": "B5: ADWIN"}
COLORS = {"entropy_weighted": "#2563EB", "uniform": "#9CA3AF", "adwin": "#F59E0B"}


def kaplan_meier(times, events, max_t=BUDGET):
    """Compute KM survival function for time-to-detection."""
    # times: detection delay (BUDGET if censored)
    # events: 1 if detected, 0 if censored
    unique_t = np.sort(np.unique(times[events == 1]))
    S = 1.0
    km_t = [0]
    km_S = [1.0]

    for t in unique_t:
        at_risk = ((times >= t)).sum()
        events_at_t = ((times == t) & (events == 1)).sum()
        S *= (1 - events_at_t / at_risk) if at_risk > 0 else 1.0
        km_t.append(t)
        km_S.append(S)

    # Extend to max_t
    km_t.append(max_t)
    km_S.append(km_S[-1])
    return np.array(km_t), np.array(km_S)


def log_rank_test(times1, events1, times2, events2):
    """Two-sample log-rank test."""
    all_times = np.sort(np.unique(np.concatenate([
        times1[events1 == 1], times2[events2 == 1]
    ])))

    O1, E1_sum = 0, 0
    for t in all_times:
        n1 = (times1 >= t).sum()
        n2 = (times2 >= t).sum()
        d1 = ((times1 == t) & (events1 == 1)).sum()
        d2 = ((times2 == t) & (events2 == 1)).sum()
        n = n1 + n2
        d = d1 + d2
        if n == 0:
            continue
        E1_t = n1 * d / n
        O1 += d1
        E1_sum += E1_t

    if E1_sum == 0:
        return 0, 1.0
    Z2 = (O1 - E1_sum) ** 2 / E1_sum
    p = 1 - chi2.cdf(Z2, df=1)
    return Z2, p


def auc_detection(times, events, tau=BUDGET):
    """Area under cumulative detection curve up to tau.

    = integral of (1-S(t)) from 0 to tau.
    Higher = more/faster detections within budget.
    NOT the standard RMST (which is integral of S(t)).
    """
    km_t, km_S = kaplan_meier(times, events, tau)
    area_under_S = 0.0
    for i in range(len(km_t) - 1):
        dt = km_t[i + 1] - km_t[i]
        area_under_S += km_S[i] * dt
    return tau - area_under_S  # = integral of (1 - S(t))


def main():
    df = pd.read_csv(OUT / "e1_metrics.csv")
    df["event"] = (df["delay"] < BUDGET).astype(int)
    df["time"] = df["delay"].clip(upper=BUDGET)

    lines = []
    lines.append("=== E1 Censoring-Aware Analysis (40 seeds) ===\n")

    # --- KM plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx, sev in enumerate(["moderate", "severe"]):
        ax = axes[idx]
        for det in DETECTORS:
            sub = df[(df.severity == sev) & (df.detector == det)]
            t = sub["time"].values
            e = sub["event"].values
            km_t, km_S = kaplan_meier(t, e)
            # Plot as 1-S (detection probability)
            ax.step(km_t, 1 - km_S, where="post",
                    label=DET_LABELS[det], color=COLORS[det], linewidth=2)

        ax.set_xlabel("Steps after shift", fontsize=11)
        ax.set_ylabel("Cumulative detection probability", fontsize=11)
        ax.set_title(f"({chr(97+idx)}) {sev.capitalize()} shift", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, BUDGET)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / "e1_km.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "e1_km.png", dpi=200, bbox_inches="tight")
    lines.append(f"KM plot saved\n")

    # --- Log-rank + RMST ---
    for sev in ["moderate", "severe"]:
        lines.append(f"--- {sev} ---")
        c1 = df[(df.severity == sev) & (df.detector == "entropy_weighted")]
        uni = df[(df.severity == sev) & (df.detector == "uniform")]
        adw = df[(df.severity == sev) & (df.detector == "adwin")]

        # Log-rank: C1 vs Uniform
        Z2, p = log_rank_test(
            c1["time"].values, c1["event"].values,
            uni["time"].values, uni["event"].values
        )
        lines.append(f"  Log-rank C1 vs Uniform: chi2={Z2:.3f}, p={p:.4f}")

        # Log-rank: C1 vs ADWIN
        Z2a, pa = log_rank_test(
            c1["time"].values, c1["event"].values,
            adw["time"].values, adw["event"].values
        )
        lines.append(f"  Log-rank C1 vs ADWIN:   chi2={Z2a:.3f}, p={pa:.4f}")

        # AUC of cumulative detection curve
        for det in DETECTORS:
            sub = df[(df.severity == sev) & (df.detector == det)]
            r = auc_detection(sub["time"].values, sub["event"].values)
            lines.append(f"  AUC_det {det:20s}: {r:.1f} steps")

        lines.append("")

    report = "\n".join(lines)
    with open(OUT / "e1_censoring_analysis.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
