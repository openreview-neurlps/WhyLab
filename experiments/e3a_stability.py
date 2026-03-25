"""
E3a: Stability Validation via Observable Proxy (v2)
====================================================
Compares B1 (no damping) vs B2 (cosine) vs B3 (grad clip) vs C3 (adaptive)
on a controlled convex quadratic environment WITH drifting optimum.

Key design principles:
1. True V_t AND proxy V_hat_t both logged for correlation analysis
2. Same noise realization shared across controllers (paired comparison)
3. Drifting optimum makes fixed step sizes dangerous
4. Hallucination = biased + high-variance noise (not just zero-mean)
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


class DriftingQuadraticEnv:
    """Quadratic loss with a slowly drifting optimum.

    theta*(t) = theta*_0 + drift_rate * t * drift_dir
    This makes fixed step-size controllers dangerous because they
    can't adapt to the changing landscape.
    """

    def __init__(self, dim, A_diag, R_star, drift_rate=0.005):
        self.dim = dim
        self.A = np.diag(A_diag[:dim])
        self.R_star = R_star
        self.theta_star_0 = np.zeros(dim)
        self.drift_rate = drift_rate
        self.drift_dir = np.ones(dim) / np.sqrt(dim)  # unit direction

    def theta_star(self, t):
        return self.theta_star_0 + self.drift_rate * t * self.drift_dir

    def true_gradient(self, theta, t):
        """True gradient: g = A @ (theta - theta*(t))"""
        return self.A @ (theta - self.theta_star(t))

    def reward(self, theta, t, rng):
        diff = theta - self.theta_star(t)
        loss = 0.5 * diff @ self.A @ diff
        eta = rng.normal(0, 0.1)
        return self.R_star - loss + eta

    def true_V(self, theta, t):
        diff = theta - self.theta_star(t)
        return 0.5 * np.dot(diff, diff)


def generate_noise_sequence(T, dim, h_rate, noise_sigma, hall_sigma, rng):
    """Pre-generate noise for paired comparison.

    Hallucination = large noise + directional bias (not just zero-mean).
    """
    noise = np.zeros((T, dim))
    is_hall = rng.random(T) < h_rate

    for t in range(T):
        if is_hall[t]:
            # Hallucination: biased + high-variance
            bias = rng.choice([-1, 1], size=dim) * hall_sigma * 0.3
            noise[t] = rng.normal(bias, hall_sigma, dim)
        else:
            noise[t] = rng.normal(0, noise_sigma, dim)

    return noise, is_hall


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
def ctrl_no_damping(t, T, g_hat, state, **kw):
    return 0.5

def ctrl_cosine(t, T, g_hat, state, **kw):
    return 0.5 * np.cos(np.pi * t / (2 * T)) + 0.01

def ctrl_grad_clip(t, T, g_hat, state, c=1.0, **kw):
    gn = np.linalg.norm(g_hat)
    return min(0.5, c / (gn + 1e-10))

def ctrl_adaptive(t, T, g_hat, state, epsilon_floor=0.01, ceiling=0.8, **kw):
    """C3: Lyapunov-bounded adaptive damping."""
    g_sq = np.dot(g_hat, g_hat)
    alpha_ema = 0.1

    if state["g_sq_ema"] is None:
        state["g_sq_ema"] = g_sq
    else:
        state["g_sq_ema"] = alpha_ema * g_sq + (1 - alpha_ema) * state["g_sq_ema"]

    # Numerator proxy: use gradient norm as proxy for <theta-theta*, g>
    # Under quadratic loss: <theta-theta*, g> = theta^T A (theta-theta*) ~ ||g||
    g_norm = np.linalg.norm(g_hat)
    numerator = 2.0 * g_norm

    denominator = state["g_sq_ema"] + 1e-10
    zeta_max = numerator / denominator
    return float(np.clip(zeta_max, epsilon_floor, ceiling))


def ctrl_adam(t, T, g_hat, state, lr=0.1, beta1=0.9, beta2=0.999,
              eps=1e-8, **kw):
    """Adam optimizer baseline.

    Standard adaptive learning rate method. Returns effective step size
    based on Adam's bias-corrected first/second moment estimates.
    """
    if state.get("adam_m") is None:
        dim = len(g_hat)
        state["adam_m"] = np.zeros(dim)
        state["adam_v"] = np.zeros(dim)
        state["adam_t"] = 0

    state["adam_t"] += 1
    state["adam_m"] = beta1 * state["adam_m"] + (1 - beta1) * g_hat
    state["adam_v"] = beta2 * state["adam_v"] + (1 - beta2) * g_hat ** 2

    m_hat = state["adam_m"] / (1 - beta1 ** state["adam_t"])
    v_hat = state["adam_v"] / (1 - beta2 ** state["adam_t"])

    # Effective step size = lr * ||m_hat|| / ||sqrt(v_hat) + eps||
    effective = lr * np.linalg.norm(m_hat) / (np.linalg.norm(np.sqrt(v_hat) + eps) + 1e-10)
    return float(np.clip(effective, 0.01, 0.8))


CONTROLLERS = {
    "no_damping": (ctrl_no_damping, {}),
    "cosine": (ctrl_cosine, {}),
    "grad_clip": (ctrl_grad_clip, {"c": 1.0}),
    "adam": (ctrl_adam, {"lr": 0.1}),
    "adaptive": (ctrl_adaptive, {"epsilon_floor": 0.01, "ceiling": 0.8}),
}


def run_single(env, theta0, noise_seq, ctrl_fn, ctrl_params, T):
    theta = theta0.copy()
    logs = []
    R_max_ema = None
    state = {"g_sq_ema": None}

    for t in range(T):
        V_true = env.true_V(theta, t)
        r_rng = np.random.default_rng(42 + t)
        R_t = env.reward(theta, t, r_rng)
        R_t = np.clip(R_t, -1e8, 1e8)

        # Proxy V_hat: EMA-based R_max
        if R_max_ema is None:
            R_max_ema = R_t
        else:
            R_max_ema = max(EMA_ALPHA * R_t + (1 - EMA_ALPHA) * R_max_ema, R_t)
        V_hat = 0.5 * min((R_max_ema - R_t) ** 2, 1e12)

        # Gradient
        g_true = env.true_gradient(theta, t)
        g_hat = g_true + noise_seq[t]

        # True zeta_max
        inner = np.dot(theta - env.theta_star(t), g_true)
        g_hat_sq = np.dot(g_hat, g_hat)
        zeta_max_true = (2 * inner / g_hat_sq) if (g_hat_sq > 0 and inner > 0) else 0.0

        # Controller
        zeta = ctrl_fn(t=t, T=T, g_hat=g_hat, state=state, **ctrl_params)

        # Update
        theta_new = theta - zeta * g_hat
        if np.linalg.norm(theta_new) > 1e6:
            theta_new = theta  # divergence guard

        V_hat_prev = logs[-1]["V_hat"] if logs else V_hat
        logs.append({
            "t": t,
            "V_true": float(V_true),
            "V_hat": float(V_hat),
            "zeta": float(zeta),
            "zeta_max": float(zeta_max_true),
            "delta_V_hat": float(V_hat - V_hat_prev) if t > 0 else 0.0,
            "R_t": float(R_t),
        })
        theta = theta_new

    return logs


def run_e3a():
    env = DriftingQuadraticEnv(DIM, A_DIAG, R_STAR, drift_rate=0.005)
    metrics_rows = []
    stepwise_rows = []

    for h_rate in H_RATES:
        for s_idx in range(SEEDS):
            seed = BASE_SEED + s_idx
            rng = np.random.default_rng(seed)
            theta0 = rng.normal(0, THETA0_SCALE, DIM)

            # SHARED noise across controllers
            noise, _ = generate_noise_sequence(
                HORIZON, DIM, h_rate, NOISE_SIGMA, HALL_SIGMA, rng)

            for ctrl_name, (ctrl_fn, ctrl_params) in CONTROLLERS.items():
                logs = run_single(env, theta0, noise, ctrl_fn, ctrl_params, HORIZON)
                df = pd.DataFrame(logs)

                V_hats = df["V_hat"].values
                V_trues = df["V_true"].values
                dv = df["delta_V_hat"].values[1:]
                violations = (dv > 0).sum()
                viol_rate = violations / max(len(dv), 1)

                # Convergence: first t where V_true < 0.5
                conv_mask = V_trues < 0.5
                conv_steps = int(np.argmax(conv_mask)) if conv_mask.any() else HORIZON

                # Correlation
                if np.std(V_trues) > 1e-10 and np.std(V_hats) > 1e-10:
                    pr, _ = pearsonr(V_trues, V_hats)
                    sr, _ = spearmanr(V_trues, V_hats)
                else:
                    pr, sr = 0.0, 0.0

                metrics_rows.append({
                    "seed": seed, "hallucination_rate": h_rate,
                    "controller": ctrl_name,
                    "final_vhat": float(V_hats[-1]),
                    "final_vtrue": float(V_trues[-1]),
                    "viol_rate": viol_rate,
                    "conv_steps": conv_steps,
                    "pearson_r": pr, "spearman_r": sr,
                })

                if s_idx < 3:
                    for row in logs:
                        stepwise_rows.append({
                            "seed": seed, "hallucination_rate": h_rate,
                            "controller": ctrl_name, **row,
                        })

        print(f"  h_rate={h_rate} done")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_m = pd.DataFrame(metrics_rows)
    df_m.to_csv(out / "e3a_metrics.csv", index=False)
    print(f"\n[E3a] Saved: {len(df_m)} rows")

    df_s = pd.DataFrame(stepwise_rows)
    df_s.to_parquet(out / "e3a_stepwise.parquet", index=False)
    print(f"[E3a] Saved stepwise: {len(df_s)} rows")

    agg = df_m.groupby(["hallucination_rate", "controller"]).agg(
        final_vtrue=("final_vtrue", "mean"),
        final_vhat=("final_vhat", "mean"),
        viol_rate=("viol_rate", "mean"),
        conv_steps=("conv_steps", "mean"),
        pearson=("pearson_r", "mean"),
    ).round(4).reset_index()
    agg.to_csv(out / "e3a_summary.csv", index=False)
    print("\n=== E3a Summary ===")
    print(agg.to_string(index=False))

    return df_m


if __name__ == "__main__":
    run_e3a()
