"""
E1 Figure Generation
====================
Figure 2a: within-horizon detection rate by severity
Figure 2b: conditional detection delay (moderate + severe only)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

HORIZON_BUDGET = 700  # post-shift budget (1000 - 300)
DETECTOR_LABELS = {
    "entropy_weighted": "C1 (ours)",
    "uniform": "B4: Uniform",
    "adwin": "B5: ADWIN",
}
DETECTOR_ORDER = ["entropy_weighted", "uniform", "adwin"]
COLORS = {"entropy_weighted": "#2563EB", "uniform": "#9CA3AF", "adwin": "#F59E0B"}
SEV_ORDER = ["mild", "moderate", "severe"]


def bootstrap_ci(data, n_boot=5000, ci=0.95):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.default_rng(999)
    means = np.array([rng.choice(data, size=len(data), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, 100 * (1 - ci) / 2)
    hi = np.percentile(means, 100 * (1 + ci) / 2)
    return lo, hi


def main():
    df = pd.read_csv(OUT / "e1_metrics.csv")

    # Mark detected vs not detected
    df["detected"] = df["delay"] < HORIZON_BUDGET

    # =========================================
    # Figure 2a: Detection rate by severity
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    x_pos = np.arange(len(SEV_ORDER))
    width = 0.22
    offsets = [-width, 0, width]

    for i, det in enumerate(DETECTOR_ORDER):
        rates = []
        cis_lo, cis_hi = [], []
        for sev in SEV_ORDER:
            sub = df[(df.severity == sev) & (df.detector == det)]
            rate = sub["detected"].mean()
            rates.append(rate)
            # Wilson CI for proportions
            n = len(sub)
            z = 1.96
            denom = 1 + z**2 / n
            center = (rate + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
            cis_lo.append(max(0, center - margin))
            cis_hi.append(min(1, center + margin))

        rates = np.array(rates)
        err_lo = rates - np.array(cis_lo)
        err_hi = np.array(cis_hi) - rates
        ax.bar(x_pos + offsets[i], rates, width, label=DETECTOR_LABELS[det],
               color=COLORS[det], edgecolor="white", linewidth=0.5,
               yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.capitalize() for s in SEV_ORDER])
    ax.set_ylabel("Detection rate\n(within 700-step budget)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) Detection reliability", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =========================================
    # Figure 2b: Conditional delay (moderate + severe)
    # =========================================
    ax2 = axes[1]
    sevs_show = ["moderate", "severe"]
    x_pos2 = np.arange(len(sevs_show))

    for i, det in enumerate(DETECTOR_ORDER):
        means, cis = [], []
        for sev in sevs_show:
            sub = df[(df.severity == sev) & (df.detector == det) & (df.detected)]
            if len(sub) > 1:
                m = sub["delay"].mean()
                lo, hi = bootstrap_ci(sub["delay"].values)
                means.append(m)
                cis.append((m - lo, hi - m))
            elif len(sub) == 1:
                means.append(sub["delay"].values[0])
                cis.append((0, 0))
            else:
                means.append(0)
                cis.append((0, 0))

        means = np.array(means)
        errs = np.array(cis).T
        bars = ax2.bar(x_pos2 + offsets[i], means, width,
                       label=DETECTOR_LABELS[det],
                       color=COLORS[det], edgecolor="white", linewidth=0.5,
                       yerr=errs, capsize=3, error_kw={"linewidth": 1})

        # Annotate ND for cases with 0 detected
        for j, sev in enumerate(sevs_show):
            sub = df[(df.severity == sev) & (df.detector == det) & (df.detected)]
            if len(sub) == 0:
                ax2.text(x_pos2[j] + offsets[i], 5, "ND",
                         ha="center", va="bottom", fontsize=7, fontweight="bold",
                         color=COLORS[det])

    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels([s.capitalize() for s in sevs_show])
    ax2.set_ylabel("Conditional delay\n(steps, detected runs only)")
    ax2.set_title("(b) Conditional detection delay", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / "e1_detection.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "e1_detection.png", dpi=200, bbox_inches="tight")
    print(f"[E1 Fig] Saved → {FIGS / 'e1_detection.pdf'}")

    # =========================================
    # Print paired statistics
    # =========================================
    print("\n=== Paired Comparisons (C1 vs Uniform) ===")
    for sev in ["moderate", "severe"]:
        c1 = df[(df.severity == sev) & (df.detector == "entropy_weighted")].sort_values("seed")
        uni = df[(df.severity == sev) & (df.detector == "uniform")].sort_values("seed")
        diff = c1["delay"].values - uni["delay"].values
        t_stat, p_val = stats.ttest_rel(c1["delay"].values, uni["delay"].values)
        m = diff.mean()
        lo, hi = bootstrap_ci(diff)
        print(f"  {sev}: C1-Uniform delay diff = {m:.1f} [{lo:.1f}, {hi:.1f}], "
              f"paired t={t_stat:.2f}, p={p_val:.3f}")


if __name__ == "__main__":
    main()
