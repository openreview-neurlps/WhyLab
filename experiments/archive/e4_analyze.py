"""
E4 Analysis: Bootstrap CI + Paired Tests + LaTeX Table
=======================================================
Reads e4_metrics.csv and produces:
  1. Summary table with cluster-bootstrap 95% CIs
  2. Paired difference (Δ) table: full vs none
  3. LaTeX table output for paper insertion
  4. Pareto data (acceptance_rate vs regression_rate)

Usage:
  python experiments/e4_analyze.py --input experiments/results/e4_metrics.csv
  python experiments/e4_analyze.py --input experiments/results/e4_metrics.csv --emit_latex paper/tables/e4_main.tex
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Cluster Bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    cluster_key: str = "task_id",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute cluster-bootstrap CI for a metric.

    Resamples at the cluster level (problem_id) to account for
    within-problem correlation across seeds.

    Returns:
        dict with mean, ci_low, ci_high, se
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_key].unique()
    n_clusters = len(clusters)

    boot_means = []
    for _ in range(n_bootstrap):
        # Resample clusters with replacement
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([df[df[cluster_key] == c] for c in sampled], ignore_index=True)
        boot_means.append(boot_df[metric].mean())

    boot_means = np.array(boot_means)
    mean_val = df[metric].mean()
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_means))

    return {
        "mean": round(mean_val, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "se": round(se, 4),
    }


def paired_delta_bootstrap(
    df: pd.DataFrame,
    metric: str,
    abl_a: str,
    abl_b: str,
    cluster_key: str = "task_id",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for Δ = abl_a - abl_b (paired by episode).

    Pairs episodes by (task_id, seed) to control for problem difficulty.

    Returns:
        dict with delta_mean, ci_low, ci_high, p_value
    """
    rng = np.random.default_rng(seed)

    df_a = df[df["ablation"] == abl_a].set_index([cluster_key, "seed"])
    df_b = df[df["ablation"] == abl_b].set_index([cluster_key, "seed"])

    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        return {"delta_mean": 0, "ci_low": 0, "ci_high": 0, "p_value": 1.0}

    deltas = df_a.loc[common, metric].values - df_b.loc[common, metric].values
    observed_delta = float(np.mean(deltas))

    # Cluster-level bootstrap on deltas
    clusters = df_a.loc[common].reset_index()[cluster_key].unique()
    n_clusters = len(clusters)

    # Map cluster -> indices in deltas
    idx_map = df_a.loc[common].reset_index()
    cluster_indices = {
        c: idx_map[idx_map[cluster_key] == c].index.tolist()
        for c in clusters
    }

    boot_deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        boot_vals = np.concatenate([deltas[cluster_indices[c]] for c in sampled])
        boot_deltas.append(np.mean(boot_vals))

    boot_deltas = np.array(boot_deltas)
    ci_low = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of bootstrap samples on opposite side of zero
    if observed_delta >= 0:
        p_value = float(np.mean(boot_deltas <= 0)) * 2
    else:
        p_value = float(np.mean(boot_deltas >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "delta_mean": round(observed_delta, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_value, 4),
    }


# ---------------------------------------------------------------------------
# Summary Aggregation
# ---------------------------------------------------------------------------

METRICS = [
    ("final_passed", "pass_rate", True),      # higher is better
    ("safe_pass", "safe_pass", True),
    ("regression_count", "regression_rate", False),  # lower is better
    ("oscillation_index", "oscillation", False),
]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ablation summary with bootstrap CIs."""
    rows = []
    for abl, abl_df in df.groupby("ablation"):
        row = {"ablation": abl, "n_episodes": len(abl_df)}

        for col, label, _ in METRICS:
            ci = cluster_bootstrap_ci(abl_df, col)
            row[f"{label}_mean"] = ci["mean"]
            row[f"{label}_ci"] = f"[{ci['ci_low']}, {ci['ci_high']}]"

        # Acceptance rate (not bootstrap — deterministic given audit decisions)
        total_proposed = abl_df["updates_accepted"].sum() + abl_df["updates_rejected"].sum()
        if total_proposed > 0:
            row["acceptance_rate"] = round(abl_df["updates_accepted"].sum() / total_proposed, 4)
        else:
            row["acceptance_rate"] = "N/A"

        # First pass attempt
        ci_fpa = cluster_bootstrap_ci(abl_df, "first_pass_attempt")
        row["first_pass_mean"] = ci_fpa["mean"]
        row["first_pass_ci"] = f"[{ci_fpa['ci_low']}, {ci_fpa['ci_high']}]"

        rows.append(row)

    return pd.DataFrame(rows)


def compute_paired_deltas(df: pd.DataFrame, reference: str = "none") -> pd.DataFrame:
    """Compute paired Δ (ablation - reference) for all ablations."""
    ablations = [a for a in df["ablation"].unique() if a != reference]
    rows = []

    for abl in ablations:
        row = {"comparison": f"{abl} − {reference}"}
        for col, label, _ in METRICS:
            delta = paired_delta_bootstrap(df, col, abl, reference)
            row[f"Δ{label}"] = delta["delta_mean"]
            row[f"Δ{label}_ci"] = f"[{delta['ci_low']}, {delta['ci_high']}]"
            row[f"Δ{label}_p"] = delta["p_value"]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pareto Data
# ---------------------------------------------------------------------------

def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Extract (acceptance_rate, regression_rate) pairs for Pareto plot."""
    rows = []
    for abl, abl_df in df.groupby("ablation"):
        total_proposed = abl_df["updates_accepted"].sum() + abl_df["updates_rejected"].sum()
        acc = abl_df["updates_accepted"].sum() / max(total_proposed, 1)
        reg = abl_df["regression_count"].mean()
        rows.append({
            "ablation": abl,
            "acceptance_rate": round(acc, 4),
            "regression_rate": round(reg, 4),
            "pass_rate": round(abl_df["final_passed"].mean(), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX Output
# ---------------------------------------------------------------------------

def emit_latex_table(summary: pd.DataFrame, path: str, caption: str, label: str):
    """Generate a LaTeX table from the summary DataFrame."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    Ablation & Pass@1 & Safe & Osc. & Acc. & FPA \\",
        r"    \midrule",
    ]

    for _, row in summary.iterrows():
        abl = row["ablation"]
        pr = row.get("pass_rate_mean", "")
        sp = row.get("safe_pass_mean", "")
        osc = row.get("oscillation_mean", "")
        acc = row.get("acceptance_rate", "N/A")
        fpa = row.get("first_pass_mean", "")
        lines.append(f"    {abl} & {pr} & {sp} & {osc} & {acc} & {fpa} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[LaTeX] Written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E4 Analysis with Bootstrap CI")
    parser.add_argument("--input", required=True,
                        help="Path to e4_metrics.csv")
    parser.add_argument("--emit_latex", default=None,
                        help="Output LaTeX table path")
    parser.add_argument("--reference", default="none",
                        help="Reference ablation for paired tests")
    parser.add_argument("--n_bootstrap", type=int, default=10_000,
                        help="Number of bootstrap resamples")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[E4-Analyze] Loaded {len(df)} rows from {args.input}")
    print(f"[E4-Analyze] Ablations: {df['ablation'].unique().tolist()}")
    print(f"[E4-Analyze] Problems: {df['task_id'].nunique()}, Seeds: {df['seed'].nunique()}")

    # 1. Summary with bootstrap CI
    print("\n=== Summary (cluster bootstrap CI) ===")
    summary = compute_summary(df)
    print(summary.to_string(index=False))

    out_dir = Path(args.input).parent
    summary.to_csv(out_dir / "e4_summary_ci.csv", index=False)

    # 2. Paired deltas
    print(f"\n=== Paired Δ vs '{args.reference}' ===")
    deltas = compute_paired_deltas(df, reference=args.reference)
    print(deltas.to_string(index=False))
    deltas.to_csv(out_dir / "e4_paired_deltas.csv", index=False)

    # 3. Pareto data
    print("\n=== Pareto (acceptance vs regression) ===")
    pareto = compute_pareto(df)
    print(pareto.to_string(index=False))
    pareto.to_csv(out_dir / "e4_pareto.csv", index=False)

    # 4. LaTeX table
    if args.emit_latex:
        emit_latex_table(
            summary, args.emit_latex,
            caption="W5 agent benchmark results (cluster bootstrap 95\\% CI)",
            label="tab:e4",
        )

    print("\n[E4-Analyze] Done.")


if __name__ == "__main__":
    main()
