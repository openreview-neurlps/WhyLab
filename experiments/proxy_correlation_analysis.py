"""
Phase 3: Proxy-State Correlation Analysis
==========================================
Empirically justifies the gap between Theorem 1 (true V_t)
and the practical controller (proxy V̂_t).

Runs E3a environment under multiple noise levels and computes:
  1. Spearman rank correlation between V_true and V_hat
  2. Kendall tau correlation
  3. Non-stationary regime (θ* shifts at t=500)
  4. Summary table for paper inclusion

Output: experiments/results/proxy_correlation.csv
        experiments/results/proxy_correlation_summary.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

# Re-use E3a environment
from experiments.e3a_stability import (
    DriftingQuadraticEnv,
    generate_noise_sequence,
    CONTROLLERS,
    run_single,
)

ROOT = Path(__file__).resolve().parent


def run_proxy_analysis(
    seeds: int = 200,
    horizon: int = 1000,
    hallucination_rates: list = None,
    dim: int = 3,
    include_nonstationary: bool = True,
):
    """Run proxy-state correlation analysis.

    Args:
        seeds: Number of random seeds.
        horizon: Time horizon per episode.
        hallucination_rates: Noise levels to test.
        dim: Dimensionality.
        include_nonstationary: Add a regime where θ* jumps at t=500.
    """
    if hallucination_rates is None:
        hallucination_rates = [0.0, 0.1, 0.3, 0.5]

    A_diag = np.array([2.0, 1.0, 0.5])[:dim]
    R_star = 10.0
    base_seed = 42
    noise_sigma = 0.3
    hall_sigma = 2.0
    theta0_scale = 3.0

    regimes = [("stationary", 0.005)]
    if include_nonstationary:
        regimes.append(("nonstationary", 0.02))  # faster drift

    rows = []

    for regime_name, drift_rate in regimes:
        env = DriftingQuadraticEnv(dim, A_diag, R_star, drift_rate=drift_rate)

        for h_rate in hallucination_rates:
            for s_idx in range(seeds):
                seed = base_seed + s_idx
                rng = np.random.default_rng(seed)
                theta0 = rng.normal(0, theta0_scale, dim)
                noise, _ = generate_noise_sequence(
                    horizon, dim, h_rate, noise_sigma, hall_sigma, rng
                )

                # Run only the adaptive (C3) controller
                ctrl_fn, ctrl_params = CONTROLLERS["adaptive"]
                logs = run_single(env, theta0, noise, ctrl_fn, ctrl_params, horizon)

                V_trues = np.array([l["V_true"] for l in logs])
                V_hats = np.array([l["V_hat"] for l in logs])

                # Filter out constant segments
                if np.std(V_trues) < 1e-10 or np.std(V_hats) < 1e-10:
                    sp_r, kt_r = 0.0, 0.0
                else:
                    sp_r, _ = spearmanr(V_trues, V_hats)
                    kt_r, _ = kendalltau(V_trues, V_hats)

                rows.append({
                    "regime": regime_name,
                    "hallucination_rate": h_rate,
                    "seed": seed,
                    "spearman_rho": round(sp_r, 4),
                    "kendall_tau": round(kt_r, 4),
                    "final_V_true": round(float(V_trues[-1]), 4),
                    "final_V_hat": round(float(V_hats[-1]), 4),
                })

            print(f"  [{regime_name}] h_rate={h_rate}: "
                  f"{seeds} seeds done")

    # Save
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "proxy_correlation.csv", index=False)

    # Summary
    summary = df.groupby(["regime", "hallucination_rate"]).agg(
        spearman_mean=("spearman_rho", "mean"),
        spearman_std=("spearman_rho", "std"),
        kendall_mean=("kendall_tau", "mean"),
        kendall_std=("kendall_tau", "std"),
        n_seeds=("seed", "count"),
    ).round(4).reset_index()

    summary.to_csv(out / "proxy_correlation_summary.csv", index=False)
    print("\n=== Proxy Correlation Summary ===")
    print(summary.to_string(index=False))

    # Paper assertion check
    min_spearman = summary["spearman_mean"].min()
    print(f"\n[Check] Min Spearman ρ: {min_spearman:.4f}")
    if min_spearman > 0.70:
        print("[PASS] Proxy tracks true state with ρ > 0.70 across all regimes")
    else:
        print("[WARN] Proxy correlation below 0.70 in some regimes")

    return df


if __name__ == "__main__":
    run_proxy_analysis()
