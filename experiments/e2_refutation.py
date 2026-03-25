"""
E2-Refutation: Causal Claim Robustness Tests (DoWhy-aligned)
=============================================================
Extends E2 sensitivity filter with three standard refutation tests
to defend C2's causal claims against reviewer skepticism.

Refuters (following DoWhy conventions):
  1. Random Common Cause  — add Z~N(0,1) to OLS; ATE should be stable
  2. Placebo Treatment    — replace T with random T'; ATE should ≈ 0
  3. Data Subset          — re-estimate on 80% subsets; ATE should be stable

For each scenario × seed, compares:
  - Original ATE (from E2 DGP)
  - Refuted ATE (from each refuter)
  - Pass/fail criterion

Output: experiments/results/e2_refutation.csv
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]

SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]

# ----- Dataset size & scenario configs (aligned with e2_sensitivity_filter) -----
N_OBS = 50
N_SCENARIOS = 80

SCENARIO_CONFIGS = [
    ("reliable_strong", 1.0, 0.0, 1.0, False),
    ("reliable_weak",   0.5, 0.0, 1.0, False),
    ("fragile_mild",    0.0, 0.3, 3.0, True),
    ("fragile_strong",  0.0, 0.5, 3.0, True),
]
SCENARIO_PROBS = [0.25, 0.25, 0.25, 0.25]

# ----- Refutation thresholds -----
RANDOM_CC_DELTA_THRESH = 0.15    # |ΔATE|/|ATE_orig| < 15%
PLACEBO_RATIO_THRESH = 0.50      # |ATE_placebo| < 50% of |ATE_orig|
SUBSET_CV_THRESH = 0.50          # CV of subset ATEs < 0.5
SUBSET_N_TRIALS = 5              # number of 80% subsets
SUBSET_FRACTION = 0.80


# ---------------------------------------------------------------------------
# DGP (reused from e2_sensitivity_filter.py)
# ---------------------------------------------------------------------------
def generate_scenario(n, rng, tau_true, confounding_strength, noise_sd):
    """Generate one observational study scenario.

    Model:
        X ~ N(0, 1)                              # observed covariate
        U ~ N(0, 1)                              # hidden confounder
        T* = 0.3*X + gamma*U + noise             # latent treatment propensity
        T  = 1{T* > median(T*)}                  # binary treatment
        Y  = tau*T + 0.5*X + gamma*U + eps       # outcome
    """
    X = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)

    T_star = 0.3 * X + confounding_strength * U + rng.normal(0, 0.8, n)
    T = (T_star > np.median(T_star)).astype(float)

    Y = (tau_true * T + 0.5 * X
         + confounding_strength * U
         + rng.normal(0, noise_sd, n))

    return X, T, Y, U


def estimate_ate_ols(Y, T, *covariates):
    """Estimate ATE via OLS: Y ~ 1 + T + covariates.

    Returns: (ate, se)
    """
    n = len(Y)
    cols = [np.ones(n), T]
    for c in covariates:
        cols.append(c)
    A = np.column_stack(cols)

    try:
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 1e6

    residuals = Y - A @ beta
    sigma2 = np.sum(residuals ** 2) / max(n - A.shape[1], 1)

    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return beta[1], 1e6

    ate = float(beta[1])
    se = float(np.sqrt(max(cov[1, 1], 1e-12)))
    return ate, se


# ---------------------------------------------------------------------------
# Refuters
# ---------------------------------------------------------------------------
def refute_random_common_cause(Y, T, X, rng):
    """Refuter 1: Add random Z ~ N(0,1) as extra covariate.

    If ATE changes significantly, the estimate may be fragile.
    """
    Z = rng.normal(0, 1, len(Y))
    ate_orig, _ = estimate_ate_ols(Y, T, X)
    ate_refuted, _ = estimate_ate_ols(Y, T, X, Z)

    delta_ratio = abs(ate_refuted - ate_orig) / max(abs(ate_orig), 1e-8)
    passed = delta_ratio < RANDOM_CC_DELTA_THRESH

    return {
        "refuter": "random_common_cause",
        "ate_orig": ate_orig,
        "ate_refuted": ate_refuted,
        "delta_ratio": delta_ratio,
        "passed": int(passed),
    }


def refute_placebo_treatment(Y, T, X, rng):
    """Refuter 2: Replace T with random Bernoulli(0.5).

    Randomized treatment should yield ATE ≈ 0.
    """
    ate_orig, _ = estimate_ate_ols(Y, T, X)
    T_rand = rng.binomial(1, 0.5, len(Y)).astype(float)
    ate_placebo, _ = estimate_ate_ols(Y, T_rand, X)

    ratio = abs(ate_placebo) / max(abs(ate_orig), 1e-8)
    passed = ratio < PLACEBO_RATIO_THRESH

    return {
        "refuter": "placebo_treatment",
        "ate_orig": ate_orig,
        "ate_refuted": ate_placebo,
        "delta_ratio": ratio,
        "passed": int(passed),
    }


def refute_data_subset(Y, T, X, rng):
    """Refuter 3: Re-estimate on 80% random subsets.

    If ATE is stable across subsets, the estimate is robust.
    """
    ate_orig, _ = estimate_ate_ols(Y, T, X)
    n = len(Y)
    subset_size = int(n * SUBSET_FRACTION)
    subset_ates = []

    for _ in range(SUBSET_N_TRIALS):
        idx = rng.choice(n, size=subset_size, replace=False)
        ate_sub, _ = estimate_ate_ols(Y[idx], T[idx], X[idx])
        subset_ates.append(ate_sub)

    subset_ates = np.array(subset_ates)
    mean_ate = np.mean(subset_ates)
    std_ate = np.std(subset_ates, ddof=1) if len(subset_ates) > 1 else 0.0
    cv = std_ate / max(abs(mean_ate), 1e-8)
    passed = cv < SUBSET_CV_THRESH

    return {
        "refuter": "data_subset",
        "ate_orig": ate_orig,
        "ate_refuted": mean_ate,
        "delta_ratio": cv,
        "passed": int(passed),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_refutation():
    all_rows = []
    refuters = [refute_random_common_cause, refute_placebo_treatment, refute_data_subset]

    for seed_idx in range(SEEDS):
        seed = BASE_SEED + seed_idx
        rng = np.random.default_rng(seed)

        for sc_idx in range(N_SCENARIOS):
            cfg_idx = rng.choice(len(SCENARIO_CONFIGS), p=SCENARIO_PROBS)
            label, tau, gamma, noise_sd, is_fragile = SCENARIO_CONFIGS[cfg_idx]

            X, T, Y, U = generate_scenario(N_OBS, rng, tau, gamma, noise_sd)

            for refuter_fn in refuters:
                result = refuter_fn(Y, T, X, rng)
                result.update({
                    "seed": seed,
                    "scenario": sc_idx,
                    "type": label,
                    "tau_true": tau,
                    "gamma": gamma,
                    "is_fragile": int(is_fragile),
                })
                all_rows.append(result)

        if (seed_idx + 1) % 10 == 0:
            print(f"  seed {seed_idx + 1}/{SEEDS}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out / "e2_refutation.csv", index=False)
    print(f"\n[E2-Refutation] Saved: {len(df_all)} rows → results/e2_refutation.csv")

    # --- Summary ---
    print("\n=== Refutation Summary (pass rate by type × refuter) ===\n")
    summary = df_all.groupby(["refuter", "type"]).agg(
        n=("passed", "count"),
        pass_rate=("passed", "mean"),
        mean_delta=("delta_ratio", "mean"),
        std_delta=("delta_ratio", "std"),
    ).round(4).reset_index()
    print(summary.to_string(index=False))

    summary.to_csv(out / "e2_refutation_summary.csv", index=False)
    print(f"\n[E2-Refutation] Summary saved → results/e2_refutation_summary.csv")

    # --- Key hypothesis: reliable scenarios should PASS, fragile should FAIL ---
    print("\n=== Key Hypothesis Test ===")
    for refuter_name in ["random_common_cause", "placebo_treatment", "data_subset"]:
        sub = df_all[df_all["refuter"] == refuter_name]
        reliable = sub[sub["is_fragile"] == 0]["passed"].mean()
        fragile = sub[sub["is_fragile"] == 1]["passed"].mean()
        print(f"  {refuter_name:25s} | reliable pass={reliable:.3f}  fragile pass={fragile:.3f}  gap={reliable - fragile:+.3f}")

    return df_all


if __name__ == "__main__":
    run_refutation()
