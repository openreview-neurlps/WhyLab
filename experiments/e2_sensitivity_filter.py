"""
E2: Sensitivity Filtering — C2 Validation (v2: Real E-value + RV)
==================================================================
Tests whether the dual-threshold E-value + RV_q filter correctly
separates genuine from fragile causal estimates using ACTUAL
computation of E-values and robustness values from OLS regression.

Protocol:
- Generate synthetic observational data with known confounders
- Estimate ATE via OLS (naive, ignoring hidden confounder U)
- Compute E-value via VanderWeele & Ding (2017) conversion
- Compute RV_q via Cinelli & Hazlett (2020) formula
- Compare: E-only / RV-only / E+RV (C2) / no-filter
- Threshold sweep
- Metrics: fragile rate, recall, fragile rejection, retention

Key DGP design:
- "reliable" scenarios: true causal effect tau > 0, no confounding,
  low outcome noise → ATE ≈ tau, high E-value, high RV
- "fragile" scenarios: true causal effect tau = 0, moderate
  confounding creates spurious positive ATE, high outcome noise
  → positive ATE due to confounding + noise, but LOW E-value, LOW RV
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]

SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]

# Experiment parameters
N_OBS = 50            # observations per scenario (small → SE matters)
N_SCENARIOS = 80      # scenarios per seed
E_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
RV_THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# Scenario configurations: (label, tau_true, confounding_gamma, outcome_noise_sd, is_fragile)
SCENARIO_CONFIGS = [
    ("reliable_strong", 1.0, 0.0, 1.0, False),   # strong true effect, clean
    ("reliable_weak",   0.5, 0.0, 1.0, False),   # weaker true effect, clean
    ("fragile_mild",    0.0, 0.3, 3.0, True),    # no real effect, mild confounding + high noise
    ("fragile_strong",  0.0, 0.5, 3.0, True),    # no real effect, stronger confounding + high noise
]
SCENARIO_PROBS = [0.25, 0.25, 0.25, 0.25]


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------
def generate_scenario(n, rng, tau_true, confounding_strength, noise_sd):
    """Generate one observational study scenario.

    Model:
        X ~ N(0, 1)                              # observed covariate
        U ~ N(0, 1)                              # hidden confounder
        T* = 0.3*X + gamma*U + noise             # latent treatment propensity
        T  = 1{T* > median(T*)}                  # binary treatment
        Y  = tau*T + 0.5*X + gamma*U + eps       # outcome (eps ~ N(0, noise_sd))

    For fragile: tau=0, gamma>0, high noise → spurious positive ATE
    For reliable: tau>0, gamma=0, low noise → clean positive ATE
    """
    X = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)

    T_star = 0.3 * X + confounding_strength * U + rng.normal(0, 0.8, n)
    T = (T_star > np.median(T_star)).astype(float)

    Y = (tau_true * T + 0.5 * X
         + confounding_strength * U
         + rng.normal(0, noise_sd, n))

    return X, T, Y


def estimate_ate_ols(X, T, Y):
    """Estimate ATE via OLS: Y ~ 1 + T + X (does NOT control for U).

    Returns: (ate, se, t_stat, sigma_pooled)
        sigma_pooled = pooled SD of outcome (for E-value computation)
    """
    n = len(Y)
    A = np.column_stack([np.ones(n), T, X])

    # Pooled SD of outcome for E-value (Def. 2 in paper)
    sigma_pooled = float(np.std(Y, ddof=1))

    try:
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 1e6, 0.0, sigma_pooled

    residuals = Y - A @ beta
    sigma2 = np.sum(residuals**2) / max(n - A.shape[1], 1)

    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return beta[1], 1e6, 0.0, sigma_pooled

    ate = beta[1]
    se = np.sqrt(max(cov[1, 1], 1e-12))
    t_stat = ate / se

    return ate, se, t_stat, sigma_pooled


# ---------------------------------------------------------------------------
# E-value (VanderWeele & Ding 2017, Def. 2 in paper)
# ---------------------------------------------------------------------------
def compute_evalue(ate, sigma_pooled):
    """Compute approximate E-value for continuous outcomes.

    Following VanderWeele & Ding (2017) / paper Def. 2:
        d = |ATE| / sigma_pooled  (standardized effect size, NOT SE)
        RR = exp(0.91 * d)
        E = RR + sqrt(RR * (RR - 1))

    sigma_pooled is the pooled SD of the outcome, distinct from SE
    of the coefficient. This makes E-value an effect-size measure,
    while RV (based on t-stat = ATE/SE) captures statistical precision.
    """
    d = min(abs(ate) / max(sigma_pooled, 1e-8), 10.0)
    rr = np.exp(0.91 * d)
    return float(rr + np.sqrt(rr * (rr - 1.0))) if rr > 1.0 else 1.0


# ---------------------------------------------------------------------------
# Robustness Value (Cinelli & Hazlett 2020, Appendix C)
# ---------------------------------------------------------------------------
def compute_rv(ate, se, q=0.0):
    """Compute Robustness Value RV_q.

    f_q = (|hat_beta| - q) / SE ; RV_q = 0.5*(sqrt(f^4+4f^2) - f^2)
    """
    f_q = (abs(ate) - q) / max(se, 1e-8)
    if f_q <= 0:
        return 0.0
    return float(0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2))


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_filter(df, E_thresh, RV_thresh, mode="E+RV"):
    if mode == "none":
        return np.ones(len(df), dtype=bool)
    elif mode == "E_only":
        return df["E_value"].values >= E_thresh
    elif mode == "RV_only":
        return df["RV_q"].values >= RV_thresh
    elif mode == "E+RV":
        return ((df["E_value"].values >= E_thresh)
                & (df["RV_q"].values >= RV_thresh))
    raise ValueError(f"Unknown mode: {mode}")


def compute_metrics(df, mask):
    accepted = df[mask]
    n_total = len(df)
    n_accepted = mask.sum()

    # Among accepted with positive ATE, fraction that are fragile
    acc_pos = accepted[accepted["ate_positive"] == 1]
    fragile_rate = float(acc_pos["is_fragile"].mean() if len(acc_pos) > 0 else 0.0)

    # Recall: fraction of reliable positives kept
    n_reliable = df["is_reliable"].sum()
    recall = float(accepted["is_reliable"].sum() / n_reliable if n_reliable > 0 else 1.0)

    # Fragile rejection
    n_fragile = df["is_fragile"].sum()
    fragile_rej = float(1 - accepted["is_fragile"].sum() / n_fragile
                        if n_fragile > 0 else 1.0)

    return {
        "n_accepted": int(n_accepted),
        "retention_rate": float(n_accepted / max(n_total, 1)),
        "fragile_rate": fragile_rate,
        "recall": recall,
        "fragile_rej": fragile_rej,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_e2():
    all_rows = []

    for seed_idx in range(SEEDS):
        seed = BASE_SEED + seed_idx
        rng = np.random.default_rng(seed)

        rows = []
        for sc_idx in range(N_SCENARIOS):
            cfg_idx = rng.choice(len(SCENARIO_CONFIGS), p=SCENARIO_PROBS)
            label, tau, gamma, noise_sd, is_fragile = SCENARIO_CONFIGS[cfg_idx]

            X, T, Y = generate_scenario(N_OBS, rng, tau, gamma, noise_sd)
            ate, se, t_stat, sigma_pooled = estimate_ate_ols(X, T, Y)

            e_val = compute_evalue(ate, sigma_pooled)
            rv_q = compute_rv(ate, se)
            ate_pos = ate > 0

            rows.append({
                "seed": seed, "scenario": sc_idx, "type": label,
                "tau_true": tau, "gamma": gamma, "noise_sd": noise_sd,
                "ate": ate, "se": se, "t_stat": t_stat,
                "E_value": e_val, "RV_q": rv_q,
                "is_fragile": int(is_fragile),
                "ate_positive": int(ate_pos),
                "is_reliable": int(ate_pos and not is_fragile),
            })

        df = pd.DataFrame(rows)

        # No filter
        mask = apply_filter(df, 0, 0, "none")
        m = compute_metrics(df, mask)
        all_rows.append({"seed": seed, "mode": "none",
                         "E_thresh": 0, "RV_thresh": 0, **m})

        # Threshold sweeps
        for E_t in E_THRESHOLDS:
            mask = apply_filter(df, E_t, 0, "E_only")
            m = compute_metrics(df, mask)
            all_rows.append({"seed": seed, "mode": "E_only",
                             "E_thresh": E_t, "RV_thresh": 0, **m})

        for RV_t in RV_THRESHOLDS:
            mask = apply_filter(df, 0, RV_t, "RV_only")
            m = compute_metrics(df, mask)
            all_rows.append({"seed": seed, "mode": "RV_only",
                             "E_thresh": 0, "RV_thresh": RV_t, **m})

        for E_t, RV_t in product(E_THRESHOLDS, RV_THRESHOLDS):
            mask = apply_filter(df, E_t, RV_t, "E+RV")
            m = compute_metrics(df, mask)
            all_rows.append({"seed": seed, "mode": "E+RV",
                             "E_thresh": E_t, "RV_thresh": RV_t, **m})

        if (seed_idx + 1) % 10 == 0:
            print(f"  seed {seed_idx + 1}/{SEEDS}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out / "e2_metrics.csv", index=False)
    print(f"\n[E2] Saved: {len(df_all)} rows")

    # Pareto summary
    print("\n=== E2 Summary: Best Fragile-Rate / Recall Pareto ===")
    for mode in ["none", "E_only", "RV_only", "E+RV"]:
        sub = df_all[df_all["mode"] == mode]
        agg = sub.groupby(["E_thresh", "RV_thresh"]).agg(
            fragile_rate=("fragile_rate", "mean"),
            recall=("recall", "mean"),
            fragile_rej=("fragile_rej", "mean"),
            retention=("retention_rate", "mean"),
        ).reset_index()

        good = agg[(agg.recall > 0.5)]
        if len(good) > 0:
            best = good.sort_values("fragile_rate").iloc[0]
            print(f"\n  {mode:8s} | E>={best.E_thresh:.1f} RV>={best.RV_thresh:.2f}")
            print(f"           frag_rate={best.fragile_rate:.3f} recall={best.recall:.3f} "
                  f"fragile_rej={best.fragile_rej:.3f} retention={best.retention:.3f}")
        else:
            print(f"\n  {mode:8s} | No config with recall > 0.5")

    # Save summary
    agg_full = df_all.groupby(["mode", "E_thresh", "RV_thresh"]).agg(
        fragile_rate=("fragile_rate", "mean"),
        recall=("recall", "mean"),
        fragile_rej=("fragile_rej", "mean"),
        retention=("retention_rate", "mean"),
    ).round(4).reset_index()
    agg_full.to_csv(out / "e2_summary.csv", index=False)
    print(f"\n[E2] Summary saved: {len(agg_full)} rows")

    return df_all


if __name__ == "__main__":
    run_e2()
