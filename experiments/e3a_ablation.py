"""
E3a-Ablation: Lyapunov ζ_max Component Ablation Study
======================================================
Validates C3 design choices by systematically removing/modifying
components of the adaptive damping controller.

Track 1 (Paper-aligned): EMA, floor, ceiling, β sweep
Track 2 (Implementation-aligned): signal/noise decomposition

Ablation Variants:
  Track 1 — Controller design ablation
    full:       standard C3_proxy (EMA + floor + ceiling)
    no_ema:     raw ζ_max without EMA smoothing
    no_floor:   ε_floor = 0 (allows learning stop)
    no_ceiling: ceiling = ∞ (allows arbitrarily large ζ)
    beta_low:   β = 0.7 (more reactive EMA)
    beta_high:  β = 0.95 (more stable EMA)

  Track 2 — Signal/Noise decomposition (via LyapunovFilter interface)
    lyap_full:       |ATE|×confidence / (DI + ARES + ε)
    lyap_no_conf:    |ATE| only / (DI + ARES + ε)
    lyap_no_ares:    |ATE|×confidence / (DI + ε)
    lyap_no_di:      |ATE|×confidence / (ARES + ε)
    lyap_ares_alt:   |ATE|×confidence / (DI + (1-p̂) + ε)

Output: experiments/results/e3a_ablation.csv
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["e3a"]
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]

SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]
HORIZON = CFG["horizon"]
DIM = CFG["dim"]
A_DIAG = np.array(CFG["A_diag"])
R_STAR = CFG["R_star"]
THETA0_SCALE = CFG["theta0_scale"]
H_RATES = CFG["hallucination_rates"]
NOISE_SIGMA = CFG["noise_sigma"]
HALL_SIGMA = CFG["hallucination_sigma"]

# Reduced seeds for ablation (still statistically meaningful)
ABLATION_SEEDS = 20


# ---------------------------------------------------------------------------
# Environment (reused from e3a_stationary.py)
# ---------------------------------------------------------------------------
class Env:
    def __init__(self, dim, A_diag, R_star):
        self.dim = dim
        self.A = np.diag(A_diag[:dim])
        self.R_star = R_star
        self.theta_star = np.zeros(dim)

    def gradient(self, theta):
        return self.A @ theta

    def reward(self, theta, rng):
        loss = 0.5 * theta @ self.A @ theta
        return self.R_star - loss + rng.normal(0, 0.1)

    def true_V(self, theta):
        d = theta - self.theta_star
        return 0.5 * np.dot(d, d)


def gen_noise(T, dim, h_rate, noise_sigma, hall_sigma, rng):
    noise = np.zeros((T, dim))
    is_hall = rng.random(T) < h_rate
    for t in range(T):
        sigma = hall_sigma if is_hall[t] else noise_sigma
        noise[t] = rng.normal(0, sigma, dim)
    return noise


# ---------------------------------------------------------------------------
# Track 1: Controller Design Ablation
# ---------------------------------------------------------------------------
def ctrl_proxy_ablation(t, T, g_hat, state, env, theta,
                        use_ema=True, epsilon_floor=0.01, ceiling=0.8,
                        beta=0.9, **kw):
    """C3-proxy with configurable components for ablation."""
    g_sq = np.dot(g_hat, g_hat)

    if use_ema:
        if state["ema"] is None:
            state["ema"] = g_sq
        else:
            state["ema"] = (1 - beta) * g_sq + beta * state["ema"]
        denom = state["ema"]
    else:
        denom = g_sq  # raw, no smoothing

    g_norm = np.linalg.norm(g_hat)
    z_max = 2.0 * g_norm / (denom + 1e-10)
    return float(np.clip(z_max, epsilon_floor, ceiling))


# Track 1 ablation configurations
TRACK1_CONFIGS = {
    "full": {
        "use_ema": True, "epsilon_floor": 0.01, "ceiling": 0.8, "beta": 0.9,
    },
    "no_ema": {
        "use_ema": False, "epsilon_floor": 0.01, "ceiling": 0.8, "beta": 0.9,
    },
    "no_floor": {
        "use_ema": True, "epsilon_floor": 0.0, "ceiling": 0.8, "beta": 0.9,
    },
    "no_ceiling": {
        "use_ema": True, "epsilon_floor": 0.01, "ceiling": 1e6, "beta": 0.9,
    },
    "beta_low": {
        "use_ema": True, "epsilon_floor": 0.01, "ceiling": 0.8, "beta": 0.7,
    },
    "beta_high": {
        "use_ema": True, "epsilon_floor": 0.01, "ceiling": 0.8, "beta": 0.95,
    },
}


# ---------------------------------------------------------------------------
# Track 2: Signal/Noise Decomposition (LyapunovFilter-style)
# ---------------------------------------------------------------------------
def lyap_filter_ablation(proposed_zeta, ate, drift_index, ares_penalty,
                         confidence, mode="full"):
    """LyapunovFilter.clip() with configurable signal/noise decomposition."""
    min_zeta = 0.01
    max_zeta = 0.8
    eps = 0.01

    if mode == "full":
        signal = abs(ate) * confidence
        noise = drift_index + ares_penalty
    elif mode == "no_conf":
        signal = abs(ate)
        noise = drift_index + ares_penalty
    elif mode == "no_ares":
        signal = abs(ate) * confidence
        noise = drift_index
    elif mode == "no_di":
        signal = abs(ate) * confidence
        noise = ares_penalty
    elif mode == "ares_alt":
        # Alternative: use (1 - p_hat) directly instead of penalty
        signal = abs(ate) * confidence
        noise = drift_index + (1.0 - confidence)  # p_hat ≈ confidence as proxy
    else:
        raise ValueError(f"Unknown mode: {mode}")

    zeta_max = (2.0 * signal) / (noise + eps) if noise > eps else max_zeta
    zeta_max = max(min_zeta, min(max_zeta, zeta_max))
    safe_zeta = min(proposed_zeta, zeta_max)
    safe_zeta = max(min_zeta, safe_zeta)
    return safe_zeta, zeta_max


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------
def run_track1(env, theta0, noise, ctrl_params, T):
    """Run one trajectory with Track 1 controller ablation."""
    theta = theta0.copy()
    state = {"ema": None}
    V_history = []

    for t in range(T):
        V_true = env.true_V(theta)
        V_history.append(V_true)

        g_true = env.gradient(theta)
        g_hat = g_true + noise[t]

        zeta = ctrl_proxy_ablation(
            t=t, T=T, g_hat=g_hat, state=state,
            env=env, theta=theta, **ctrl_params,
        )

        theta_new = theta - zeta * g_hat
        if np.linalg.norm(theta_new) > 1e6:
            theta_new = theta

        theta = theta_new

    V_arr = np.array(V_history)
    dV = np.diff(V_arr)
    return {
        "final_V": float(V_arr[-1]),
        "cumulative_V": float(np.mean(V_arr)),
        "true_viol_rate": float((dV > 0).sum() / len(dV)),
        "max_V": float(np.max(V_arr)),
        "converged": int(V_arr[-1] < V_arr[0] * 0.1),
        "convergence_step": int(np.argmin(V_arr > V_arr[0] * 0.01))
            if (V_arr < V_arr[0] * 0.01).any() else T,
    }


def run_track2_synthetic():
    """Run Track 2 with synthetic audit signals to test filter decomposition.

    Generates synthetic (ATE, DI, ARES_penalty, confidence) draws
    and compares how each ablation mode modifies the effective ζ.
    """
    rng = np.random.default_rng(BASE_SEED)
    rows = []
    N_DRAWS = 1000

    for _ in range(N_DRAWS):
        # Simulate audit signals
        ate = rng.uniform(0, 2.0)
        drift_index = rng.uniform(0, 1.0)
        ares_penalty = rng.uniform(0, 0.5)
        confidence = rng.uniform(0.3, 1.0)
        proposed_zeta = rng.uniform(0.1, 0.8)

        for mode in ["full", "no_conf", "no_ares", "no_di", "ares_alt"]:
            safe_z, z_max = lyap_filter_ablation(
                proposed_zeta, ate, drift_index, ares_penalty,
                confidence, mode,
            )
            rows.append({
                "mode": mode,
                "ate": ate, "drift_index": drift_index,
                "ares_penalty": ares_penalty, "confidence": confidence,
                "proposed_zeta": proposed_zeta,
                "zeta_max": z_max,
                "safe_zeta": safe_z,
                "was_clipped": int(proposed_zeta > z_max),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env = Env(DIM, A_DIAG, R_STAR)

    # ===== Track 1: Controller Design Ablation =====
    print("=" * 60)
    print("Track 1: Controller Design Ablation")
    print("=" * 60)

    t1_rows = []
    for h in H_RATES:
        for si in range(ABLATION_SEEDS):
            seed = BASE_SEED + si
            rng = np.random.default_rng(seed)
            theta0 = rng.normal(0, THETA0_SCALE, DIM)
            noise = gen_noise(HORIZON, DIM, h, NOISE_SIGMA, HALL_SIGMA, rng)

            for variant, params in TRACK1_CONFIGS.items():
                result = run_track1(env, theta0, noise, params, HORIZON)
                result.update({
                    "track": "T1",
                    "seed": seed,
                    "h_rate": h,
                    "variant": variant,
                })
                t1_rows.append(result)

        print(f"  h={h} done")

    df_t1 = pd.DataFrame(t1_rows)

    # ===== Track 2: Signal/Noise Decomposition =====
    print("\n" + "=" * 60)
    print("Track 2: Signal/Noise Decomposition")
    print("=" * 60)

    df_t2 = run_track2_synthetic()

    # ===== Save results =====
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_t1.to_csv(out / "e3a_ablation_t1.csv", index=False)
    df_t2.to_csv(out / "e3a_ablation_t2.csv", index=False)
    print(f"\n[Ablation] Track 1: {len(df_t1)} rows → results/e3a_ablation_t1.csv")
    print(f"[Ablation] Track 2: {len(df_t2)} rows → results/e3a_ablation_t2.csv")

    # ===== Track 1 Summary =====
    print("\n=== Track 1: Ablation Summary (mean over seeds) ===\n")
    agg = df_t1.groupby(["h_rate", "variant"]).agg(
        final_V=("final_V", "mean"),
        final_V_std=("final_V", "std"),
        cum_V=("cumulative_V", "mean"),
        viol=("true_viol_rate", "mean"),
        viol_std=("true_viol_rate", "std"),
        max_V=("max_V", "mean"),
        converged=("converged", "mean"),
        conv_step=("convergence_step", "mean"),
    ).round(4).reset_index()
    print(agg.to_string(index=False))
    agg.to_csv(out / "e3a_ablation_t1_summary.csv", index=False)

    # ===== Track 2 Summary =====
    print("\n=== Track 2: Filter Decomposition Summary ===\n")
    agg2 = df_t2.groupby("mode").agg(
        mean_zeta_max=("zeta_max", "mean"),
        std_zeta_max=("zeta_max", "std"),
        mean_safe=("safe_zeta", "mean"),
        clip_rate=("was_clipped", "mean"),
    ).round(4).reset_index()
    print(agg2.to_string(index=False))
    agg2.to_csv(out / "e3a_ablation_t2_summary.csv", index=False)

    return df_t1, df_t2


if __name__ == "__main__":
    main()
