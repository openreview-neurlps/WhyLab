"""
E3a Figure Generation
======================
Main figure: V_true trajectory for B1/B2/B3/C3_proxy at h=0.3
Appendix: V_true vs V_hat correlation scatter
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

CTRL_LABELS = {
    "B1_fixed": "B1: Fixed step",
    "B2_cosine": "B2: Cosine decay",
    "B3_gradclip": "B3: Grad clipping",
    "C3_proxy": "C3: Adaptive (ours)",
}
CTRL_ORDER = ["B1_fixed", "B2_cosine", "B3_gradclip", "C3_proxy"]
COLORS = {
    "B1_fixed": "#9CA3AF",
    "B2_cosine": "#F59E0B",
    "B3_gradclip": "#10B981",
    "C3_proxy": "#2563EB",
}
LINESTYLES = {
    "B1_fixed": "--",
    "B2_cosine": "-.",
    "B3_gradclip": ":",
    "C3_proxy": "-",
}


def main():
    df_step = pd.read_parquet(OUT / "e3a_stationary_stepwise.parquet")
    df_met = pd.read_csv(OUT / "e3a_stationary_metrics.csv")

    # =======================================
    # Main Figure: V_true trajectory at h=0.3
    # =======================================
    h_show = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    sub = df_step[(df_step.h_rate == h_show) & (df_step.controller.isin(CTRL_ORDER))]

    for cn in CTRL_ORDER:
        cs = sub[sub.controller == cn]
        # Average across seeds
        avg = cs.groupby("t")["V_true"].agg(["mean", "std"]).reset_index()
        t = avg["t"].values
        m = avg["mean"].values
        s = avg["std"].values

        ax.plot(t, m, label=CTRL_LABELS[cn], color=COLORS[cn],
                linestyle=LINESTYLES[cn], linewidth=1.8, alpha=0.9)
        ax.fill_between(t, np.maximum(m - s, 0), m + s,
                        color=COLORS[cn], alpha=0.1)

    ax.set_xlabel("Step $t$", fontsize=11)
    ax.set_ylabel("True Lyapunov energy $V(\\theta_t)$", fontsize=11)
    ax.set_title(f"(a) $V_t$ trajectory (h = {h_show})", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e3)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)

    # =======================================
    # Panel b: Violation rate + Final V bar
    # =======================================
    ax2 = axes[1]
    h_rates = [0.0, 0.1, 0.3, 0.5]
    x = np.arange(len(h_rates))
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    for i, cn in enumerate(CTRL_ORDER):
        viol_rates = []
        for h in h_rates:
            sub_m = df_met[(df_met.h_rate == h) & (df_met.controller == cn)]
            viol_rates.append(sub_m["true_viol_rate"].mean())
        ax2.bar(x + offsets[i], viol_rates, width,
                label=CTRL_LABELS[cn], color=COLORS[cn],
                edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={h}" for h in h_rates], fontsize=9)
    ax2.set_ylabel("True violation rate $P(\\Delta V > 0)$", fontsize=10)
    ax2.set_title("(b) Violation rate by noise level", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / "e3a_stationary.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "e3a_stationary.png", dpi=200, bbox_inches="tight")
    print(f"[E3a Fig] Main → {FIGS / 'e3a_stationary.pdf'}")

    # =======================================
    # Appendix: V_true vs V_hat correlation
    # =======================================
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for i, cn in enumerate(CTRL_ORDER):
        ax3 = axes2[i]
        for h in [0.0, 0.3, 0.5]:
            cs = df_step[(df_step.h_rate == h) & (df_step.controller == cn)]
            ax3.scatter(cs["V_true"], cs["V_hat"],
                        alpha=0.05, s=2, label=f"h={h}")
        ax3.set_xlabel("True $V_t$", fontsize=9)
        if i == 0:
            ax3.set_ylabel("Proxy $\\hat{V}_t$", fontsize=9)
        ax3.set_title(CTRL_LABELS[cn], fontsize=9, fontweight="bold")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlim(1e-4, 1e3)
        ax3.set_ylim(1e-6, 1e4)
        ax3.legend(fontsize=6)

        # Annotate Pearson
        for j, h in enumerate([0.3]):
            sub_m = df_met[(df_met.h_rate == h) & (df_met.controller == cn)]
            pr = sub_m["pearson"].mean()
            ax3.text(0.05, 0.95, f"r={pr:.2f}",
                     transform=ax3.transAxes, fontsize=8,
                     va="top", fontweight="bold")

    plt.tight_layout()
    fig2.savefig(FIGS / "e3a_correlation.pdf", dpi=300, bbox_inches="tight")
    fig2.savefig(FIGS / "e3a_correlation.png", dpi=200, bbox_inches="tight")
    print(f"[E3a Fig] Correlation → {FIGS / 'e3a_correlation.pdf'}")


if __name__ == "__main__":
    main()
