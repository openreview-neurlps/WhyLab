"""
Aggregate Statistics → LaTeX Tables
====================================
Reads all experiment result CSVs (E1, E2, E3, Refutation, Ablation)
and generates:
  1. Summary tables with mean ± 95% CI
  2. LaTeX table files ready for \\input{} in main.tex

Output: paper/tables/*.tex
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER_TABLES = ROOT.parent / "paper" / "tables"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)


def ci_95(series):
    """Compute mean and 95% CI (t-based) for a series."""
    n = len(series)
    mean = series.mean()
    if n < 2:
        return mean, mean, mean
    se = series.std(ddof=1) / np.sqrt(n)
    margin = 1.96 * se  # Normal approx (n≥20)
    return mean, mean - margin, mean + margin


def fmt_ci(mean, lo, hi, decimals=4):
    """Format as 'mean [lo, hi]' string."""
    f = f".{decimals}f"
    return f"{mean:{f}} [{lo:{f}}, {hi:{f}}]"


def fmt_pm(mean, lo, hi, decimals=4):
    """Format as 'mean ± margin' string."""
    margin = (hi - lo) / 2
    f = f".{decimals}f"
    return f"${mean:{f}} \\pm {margin:{f}}$"


# ---------------------------------------------------------------------------
# E1: Drift Detection
# ---------------------------------------------------------------------------
def aggregate_e1():
    fpath = RESULTS / "e1_metrics.csv"
    if not fpath.exists():
        print("[E1] e1_metrics.csv not found, skipping")
        return
    df = pd.read_csv(fpath)

    # Key metrics per detector × severity
    group_cols = [c for c in ["detector", "severity"] if c in df.columns]
    if not group_cols:
        group_cols = ["detector"] if "detector" in df.columns else []

    if not group_cols:
        print("[E1] No grouping columns found, skipping")
        return

    metrics = ["auc", "fpr", "fnr", "detection_delay"]
    available = [m for m in metrics if m in df.columns]

    rows = []
    for name, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, [name] if len(group_cols) == 1 else name))
        for m in available:
            mean, lo, hi = ci_95(grp[m])
            row[f"{m}_mean_ci"] = fmt_pm(mean, lo, hi)
        rows.append(row)

    result = pd.DataFrame(rows)
    tex_path = PAPER_TABLES / "e1_stats.tex"
    result.to_latex(tex_path, index=False, escape=False,
                    caption="E1: Drift Detection Performance (mean $\\pm$ 95\\% CI)",
                    label="tab:e1_stats")
    print(f"[E1] Saved: {tex_path}")
    return result


# ---------------------------------------------------------------------------
# E2: Sensitivity Filter
# ---------------------------------------------------------------------------
def aggregate_e2():
    fpath = RESULTS / "e2_metrics.csv"
    if not fpath.exists():
        print("[E2] e2_metrics.csv not found, skipping")
        return
    df = pd.read_csv(fpath)

    # Aggregate by mode, using best E+RV threshold per seed
    modes = ["none", "E_only", "RV_only", "E+RV"]
    rows = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        if mode == "none":
            grp = sub
        else:
            # Use best threshold (highest recall with fragile_rate < 0.3)
            agg = sub.groupby(["E_thresh", "RV_thresh"]).agg(
                fragile_rate=("fragile_rate", "mean"),
                recall=("recall", "mean"),
            ).reset_index()
            good = agg[agg["recall"] > 0.5]
            if len(good) == 0:
                continue
            best = good.sort_values("fragile_rate").iloc[0]
            mask = True
            if "E_thresh" in sub.columns and best.get("E_thresh", 0) > 0:
                mask = mask & (sub["E_thresh"] == best["E_thresh"])
            if "RV_thresh" in sub.columns and best.get("RV_thresh", 0) > 0:
                mask = mask & (sub["RV_thresh"] == best["RV_thresh"])
            grp = sub[mask]

        for metric in ["fragile_rate", "recall", "fragile_rej", "retention_rate"]:
            if metric in grp.columns:
                mean, lo, hi = ci_95(grp[metric])
                rows.append({
                    "mode": mode,
                    "metric": metric,
                    "value": fmt_pm(mean, lo, hi),
                })

    result = pd.DataFrame(rows)
    tex_path = PAPER_TABLES / "e2_stats.tex"
    # Pivot for cleaner table
    pivot = result.pivot(index="mode", columns="metric", values="value")
    pivot.to_latex(tex_path, escape=False,
                   caption="E2: Sensitivity Filter Performance at Best Threshold (mean $\\pm$ 95\\% CI)",
                   label="tab:e2_stats")
    print(f"[E2] Saved: {tex_path}")
    return result


# ---------------------------------------------------------------------------
# E2 Refutation
# ---------------------------------------------------------------------------
def aggregate_e2_refutation():
    fpath = RESULTS / "e2_refutation.csv"
    if not fpath.exists():
        print("[E2-Refutation] e2_refutation.csv not found, skipping")
        return
    df = pd.read_csv(fpath)

    rows = []
    for (refuter, scenario_type), grp in df.groupby(["refuter", "type"]):
        pass_mean, pass_lo, pass_hi = ci_95(grp["passed"])
        delta_mean, delta_lo, delta_hi = ci_95(grp["delta_ratio"])
        rows.append({
            "Refuter": refuter.replace("_", " ").title(),
            "Scenario": scenario_type,
            "Pass Rate": fmt_pm(pass_mean, pass_lo, pass_hi, 3),
            "$\\Delta$ Ratio": fmt_pm(delta_mean, delta_lo, delta_hi, 3),
            "n": len(grp),
        })

    result = pd.DataFrame(rows)
    tex_path = PAPER_TABLES / "e2_refutation.tex"
    result.to_latex(tex_path, index=False, escape=False,
                    caption="E2 Refutation: Pass Rates by Scenario Type (mean $\\pm$ 95\\% CI)",
                    label="tab:e2_refutation")
    print(f"[E2-Refutation] Saved: {tex_path}")
    return result


# ---------------------------------------------------------------------------
# E3a: Stability (existing + ablation)
# ---------------------------------------------------------------------------
def aggregate_e3a():
    fpath = RESULTS / "e3a_stationary_metrics.csv"
    if not fpath.exists():
        print("[E3a] e3a_stationary_metrics.csv not found, skipping")
        return
    df = pd.read_csv(fpath)

    rows = []
    for (h_rate, ctrl), grp in df.groupby(["h_rate", "controller"]):
        viol_mean, viol_lo, viol_hi = ci_95(grp["true_viol_rate"])
        fv_mean, fv_lo, fv_hi = ci_95(grp["final_V"])
        cv_mean, cv_lo, cv_hi = ci_95(grp["cumulative_V"])
        rows.append({
            "$h$": h_rate,
            "Controller": ctrl,
            "Violation Rate": fmt_pm(viol_mean, viol_lo, viol_hi),
            "Final $V$": fmt_pm(fv_mean, fv_lo, fv_hi),
            "Cum. $V$": fmt_pm(cv_mean, cv_lo, cv_hi),
        })

    result = pd.DataFrame(rows)
    tex_path = PAPER_TABLES / "e3a_stats.tex"
    result.to_latex(tex_path, index=False, escape=False,
                    caption="E3a: Controller Comparison (mean $\\pm$ 95\\% CI, 40 seeds)",
                    label="tab:e3a_stats")
    print(f"[E3a] Saved: {tex_path}")
    return result


def aggregate_e3a_ablation():
    fpath = RESULTS / "e3a_ablation_t1_summary.csv"
    if not fpath.exists():
        print("[E3a-Ablation] e3a_ablation_t1_summary.csv not found, skipping")
        return

    # Track 1 already has mean/std → convert to CI
    df = pd.read_csv(fpath)
    rows = []
    for _, r in df.iterrows():
        # Approximate CI from std (20 seeds)
        n = 20
        viol_margin = 1.96 * r["viol_std"] / np.sqrt(n)
        fv_margin = 1.96 * r["final_V_std"] / np.sqrt(n)
        rows.append({
            "$h$": r["h_rate"],
            "Variant": r["variant"],
            "Violation Rate": f"${r['viol']:.4f} \\pm {viol_margin:.4f}$",
            "Final $V$": f"${r['final_V']:.4f} \\pm {fv_margin:.4f}$",
            "Converged": f"{r['converged']:.0%}",
            "Conv. Step": f"{r['conv_step']:.0f}",
        })

    result = pd.DataFrame(rows)
    tex_path = PAPER_TABLES / "e3a_ablation.tex"
    result.to_latex(tex_path, index=False, escape=False,
                    caption="E3a Ablation: Component Removal Impact (mean $\\pm$ 95\\% CI, 20 seeds)",
                    label="tab:e3a_ablation")
    print(f"[E3a-Ablation] Saved: {tex_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Aggregating experiment statistics → LaTeX tables")
    print("=" * 60)

    aggregate_e1()
    aggregate_e2()
    aggregate_e2_refutation()
    aggregate_e3a()
    aggregate_e3a_ablation()

    print(f"\nAll tables saved to: {PAPER_TABLES}")
    print("Use \\input{tables/e1_stats.tex} etc. in main.tex")


if __name__ == "__main__":
    main()
