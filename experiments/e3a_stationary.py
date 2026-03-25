"""
E3a: Stationary Quadratic - Theorem Validation (v3)
====================================================
Fixed theta* = 0. True V_t as main metric.

Controllers:
- B1: no_damping (zeta=0.3, reasonable fixed step)
- B2: cosine (starts at 0.3, decays)
- B3: grad_clip (c=1.0, max 0.3)
- C3-oracle: uses true zeta_max from Theorem 1
- C3-proxy: uses observable proxy for zeta_max

Key: Baseline steps are set to 0.3 < 2/lambda_max = 0.4
to represent a "well-tuned practitioner" baseline.
C3 does NOT need to know lambda_max.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

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
EMA_ALPHA = CFG["ema_alpha"]

# Reasonable fixed step: 0.3 < 2/lambda_max(A) = 2/5 = 0.4
REASONABLE_ZETA = 0.3


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


# --- Controllers ---
def ctrl_fixed(t, T, g_hat, state, env=None, theta=None, **kw):
    return REASONABLE_ZETA

def ctrl_cosine(t, T, g_hat, state, env=None, theta=None, **kw):
    return REASONABLE_ZETA * np.cos(np.pi * t / (2 * T)) + 0.01

def ctrl_gradclip(t, T, g_hat, state, env=None, theta=None, c=1.0, **kw):
    return min(REASONABLE_ZETA, c / (np.linalg.norm(g_hat) + 1e-10))

def ctrl_oracle(t, T, g_hat, state, env=None, theta=None,
                epsilon_floor=0.01, ceiling=0.8, **kw):
    """C3-oracle: uses TRUE zeta_max from Theorem 1."""
    inner = np.dot(theta - env.theta_star, env.gradient(theta))
    g_sq = np.dot(g_hat, g_hat)
    if g_sq > 0 and inner > 0:
        z_max = 2 * inner / g_sq
    else:
        z_max = epsilon_floor
    return float(np.clip(z_max, epsilon_floor, ceiling))

def ctrl_proxy(t, T, g_hat, state, env=None, theta=None,
               epsilon_floor=0.01, ceiling=0.8, **kw):
    """C3-proxy: estimates zeta_max from observable ||g_hat||."""
    g_sq = np.dot(g_hat, g_hat)
    if state["ema"] is None:
        state["ema"] = g_sq
    else:
        state["ema"] = 0.1 * g_sq + 0.9 * state["ema"]
    g_norm = np.linalg.norm(g_hat)
    z_max = 2.0 * g_norm / (state["ema"] + 1e-10)
    return float(np.clip(z_max, epsilon_floor, ceiling))


CTRLS = {
    "B1_fixed": (ctrl_fixed, {}),
    "B2_cosine": (ctrl_cosine, {}),
    "B3_gradclip": (ctrl_gradclip, {"c": 1.0}),
    "C3_plugin": (ctrl_oracle, {"epsilon_floor": 0.01, "ceiling": 0.8}),
    "C3_proxy": (ctrl_proxy, {"epsilon_floor": 0.01, "ceiling": 0.8}),
}


def run_one(env, theta0, noise, ctrl_fn, ctrl_params, T):
    theta = theta0.copy()
    state = {"ema": None}
    rows = []
    R_max_ema = None

    for t in range(T):
        V_true = env.true_V(theta)
        R_t = np.clip(env.reward(theta, np.random.default_rng(42 + t)), -1e8, 1e8)

        if R_max_ema is None:
            R_max_ema = R_t
        else:
            R_max_ema = max(EMA_ALPHA * R_t + (1 - EMA_ALPHA) * R_max_ema, R_t)
        V_hat = 0.5 * min((R_max_ema - R_t) ** 2, 1e12)

        g_true = env.gradient(theta)
        g_hat = g_true + noise[t]

        # True zeta_max (for logging)
        inner = np.dot(theta - env.theta_star, g_true)
        g_sq = np.dot(g_hat, g_hat)
        z_max_true = (2 * inner / g_sq) if (g_sq > 0 and inner > 0) else 0.0

        zeta = ctrl_fn(t=t, T=T, g_hat=g_hat, state=state,
                       env=env, theta=theta, **ctrl_params)

        theta_new = theta - zeta * g_hat
        if np.linalg.norm(theta_new) > 1e6:
            theta_new = theta

        rows.append({
            "t": t,
            "V_true": float(V_true),
            "V_hat": float(V_hat),
            "zeta": float(zeta),
            "zeta_max": float(z_max_true),
            "bound_gap": float(max(0, z_max_true - zeta)),
            "R_t": float(R_t),
        })
        theta = theta_new
    return rows


def main():
    env = Env(DIM, A_DIAG, R_STAR)
    metrics = []
    steps = []

    for h in H_RATES:
        for si in range(SEEDS):
            seed = BASE_SEED + si
            rng = np.random.default_rng(seed)
            theta0 = rng.normal(0, THETA0_SCALE, DIM)
            noise = gen_noise(HORIZON, DIM, h, NOISE_SIGMA, HALL_SIGMA, rng)

            for cn, (cf, cp) in CTRLS.items():
                rows = run_one(env, theta0, noise, cf, cp, HORIZON)
                df = pd.DataFrame(rows)
                Vt = df["V_true"].values
                Vh = df["V_hat"].values

                dV = np.diff(Vt)
                true_viol = (dV > 0).sum() / len(dV)
                cum_V = np.mean(Vt)

                if np.std(Vt) > 1e-10 and np.std(Vh) > 1e-10:
                    pr, _ = pearsonr(Vt, Vh)
                    sr, _ = spearmanr(Vt, Vh)
                else:
                    pr, sr = 0.0, 0.0

                bg = df["bound_gap"].values
                bg_m = np.mean(bg[bg > 0]) if (bg > 0).any() else 0.0

                metrics.append({
                    "seed": seed, "h_rate": h, "controller": cn,
                    "final_V": float(Vt[-1]),
                    "cumulative_V": float(cum_V),
                    "true_viol_rate": true_viol,
                    "pearson": pr, "spearman": sr,
                    "bound_gap_mean": bg_m,
                })

                if si < 3:
                    for r in rows:
                        steps.append({"seed": seed, "h_rate": h, "controller": cn, **r})

        print(f"  h={h} done")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_m = pd.DataFrame(metrics)
    df_m.to_csv(out / "e3a_stationary_metrics.csv", index=False)
    print(f"\n[E3a] {len(df_m)} rows")

    df_s = pd.DataFrame(steps)
    df_s.to_parquet(out / "e3a_stationary_stepwise.parquet", index=False)
    print(f"[E3a] {len(df_s)} stepwise")

    agg = df_m.groupby(["h_rate", "controller"]).agg(
        final_V=("final_V", "mean"),
        cum_V=("cumulative_V", "mean"),
        viol=("true_viol_rate", "mean"),
        pearson=("pearson", "mean"),
        bg=("bound_gap_mean", "mean"),
    ).round(4).reset_index()
    agg.to_csv(out / "e3a_stationary_summary.csv", index=False)
    print("\n=== E3a Stationary Summary ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
