"""
E2 Figure Generation
====================
FDR-Recall Pareto frontier for E-only / RV-only / E+RV filters.
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

COLORS = {"none": "#EF4444", "E_only": "#9CA3AF", "RV_only": "#F59E0B", "E+RV": "#2563EB"}
LABELS = {"none": "No filter", "E_only": "E-only", "RV_only": "RV-only", "E+RV": "C2: E+RV (ours)"}
MARKERS = {"none": "X", "E_only": "s", "RV_only": "^", "E+RV": "o"}


def main():
    df = pd.read_csv(OUT / "e2_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel a: FDR vs Recall ---
    ax = axes[0]
    for mode in ["E_only", "RV_only", "E+RV"]:
        sub = df[df["mode"] == mode]
        ax.scatter(sub["recall"], sub["fragile_rate"],
                   c=COLORS[mode], label=LABELS[mode],
                   marker=MARKERS[mode], s=30, alpha=0.5, edgecolors="none")

    # No-filter point
    nf = df[df["mode"] == "none"].iloc[0]
    ax.scatter(nf["recall"], nf["fragile_rate"], c=COLORS["none"],
               marker=MARKERS["none"], s=100, label=LABELS["none"],
               edgecolors="black", linewidths=0.5, zorder=5)

    # Highlight best E+RV
    erv = df[(df["mode"] == "E+RV") & (df["recall"] > 0.5)].sort_values("fragile_rate")
    if len(erv) > 0:
        best = erv.iloc[0]
        ax.scatter(best["recall"], best["fragile_rate"], c=COLORS["E+RV"],
                   marker="o", s=120, edgecolors="black", linewidths=1.5, zorder=6)
        ax.annotate(f"Fragile={best.fragile_rate:.3f}\nRecall={best.recall:.2f}",
                    (best.recall, best.fragile_rate), fontsize=8,
                    xytext=(15, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("Recall (true reliable retained)", fontsize=11)
    ax.set_ylabel("Fragile rate (among accepted positives)", fontsize=11)
    ax.set_title("(a) FDR-Recall Pareto", fontsize=12, fontweight="bold")
    ax.axhline(0.10, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(0.02, 0.11, "FDR = 10%", fontsize=8, color="gray")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.01, 0.50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel b: Fragile rejection vs Retention ---
    ax2 = axes[1]
    for mode in ["E_only", "RV_only", "E+RV"]:
        sub = df[df["mode"] == mode]
        ax2.scatter(sub["retention"], sub["fragile_rej"],
                    c=COLORS[mode], label=LABELS[mode],
                    marker=MARKERS[mode], s=30, alpha=0.5, edgecolors="none")

    ax2.scatter(nf["retention"], nf["fragile_rej"], c=COLORS["none"],
                marker=MARKERS["none"], s=100, label=LABELS["none"],
                edgecolors="black", linewidths=0.5, zorder=5)

    if len(erv) > 0:
        # Find same best threshold
        best_erv = df[(df["mode"] == "E+RV") &
                      (df["E_thresh"] == best.E_thresh) &
                      (df["RV_thresh"] == best.RV_thresh)].iloc[0]
        ax2.scatter(best_erv["retention"], best_erv["fragile_rej"],
                    c=COLORS["E+RV"], marker="o", s=120,
                    edgecolors="black", linewidths=1.5, zorder=6)
        ax2.annotate(f"Frag.rej={best_erv.fragile_rej:.0%}\nRetention={best_erv.retention:.0%}",
                     (best_erv.retention, best_erv.fragile_rej), fontsize=8,
                     xytext=(-60, -25), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax2.set_xlabel("Sample retention rate", fontsize=11)
    ax2.set_ylabel("Fragile positive rejection rate", fontsize=11)
    ax2.set_title("(b) Fragile rejection vs retention", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / "e2_filtering.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "e2_filtering.png", dpi=200, bbox_inches="tight")
    print(f"[E2 Fig] Saved -> {FIGS / 'e2_filtering.pdf'}")


if __name__ == "__main__":
    main()
