"""
E3b Full Metrics: viol rate + Pearson PER noise distribution
============================================================
Fixes the issue where Gaussian/t(3) had identical values (were from
sanity check mean-across-h, not per-noise). Now computes all metrics
for each noise type independently.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]
SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

T = 1000
DIM = 5
A = np.diag(np.arange(1, DIM + 1, dtype=float))
THETA_STAR = np.zeros(DIM)
R_STAR = 10.0
FLOOR = 0.01
CEIL = 0.8
ALPHA_EMA = 0.1
DELTA = 1e-10
H_RATE = 0.5

NOISE_CONFIGS = [
    {"name": "Gaussian",
     "gen": lambda rng, dim: rng.normal(0, 1, dim)},
    {"name": "Student-t(3)",
     "gen": lambda rng, dim: rng.standard_t(3, dim)},
    {"name": "Student-t(1.5)",
     "gen": lambda rng, dim: rng.standard_t(1.5, dim)},
]


def gen_noise_arr(T, dim, h_rate, rng, base_gen):
    """Generate noise array using given base distribution."""
    noise = np.zeros((T, dim))
    is_hall = rng.random(T) < h_rate
    for t in range(T):
        sigma = 3.0 if is_hall[t] else 0.5
        noise[t] = base_gen(rng, dim) * sigma
    return noise


def run(theta0, noise_arr, use_ema=True):
    theta = theta0.copy()
    ema_g_sq = None
    V_true_list = []
    V_proxy_list = []
    zeta_list = []
    R_max_ema = None

    for t in range(T):
        g = A @ (theta - THETA_STAR)
        g_hat = g + noise_arr[t]
        g_sq = np.dot(g_hat, g_hat)
        g_norm = np.linalg.norm(g_hat)

        if use_ema:
            if ema_g_sq is None:
                ema_g_sq = g_sq
            else:
                ema_g_sq = ALPHA_EMA * g_sq + (1 - ALPHA_EMA) * ema_g_sq
            z_raw = 2.0 * g_norm / (ema_g_sq + DELTA)
        else:
            z_raw = 2.0 * g_norm / (g_sq + DELTA)

        zeta = float(np.clip(z_raw, FLOOR, CEIL))
        theta = theta - zeta * g_hat

        V_true = 0.5 * np.dot(theta - THETA_STAR, theta - THETA_STAR)
        V_true_list.append(V_true)

        loss = 0.5 * theta @ A @ theta
        R_t = R_STAR - loss
        if R_max_ema is None:
            R_max_ema = R_t
        else:
            R_max_ema = 0.1 * max(R_t, R_max_ema) + 0.9 * R_max_ema
        V_proxy = 0.5 * (R_max_ema - R_t) ** 2
        V_proxy_list.append(V_proxy)
        zeta_list.append(zeta)

    V_true_arr = np.array(V_true_list)
    V_proxy_arr = np.array(V_proxy_list)

    dV = np.diff(V_true_arr)
    viol_rate = np.mean(dV > 0)
    final_V = V_true_arr[-1]
    auc_V = np.mean(V_true_arr)
    converged = 1 if final_V < V_true_arr[0] else 0
    deadlock = np.mean(np.array(zeta_list) <= FLOOR * 1.01)
    nonzero = np.mean(np.array(zeta_list) > FLOOR * 1.01)

    if np.std(V_true_arr) > 1e-10 and np.std(V_proxy_arr) > 1e-10:
        pr, _ = pearsonr(V_true_arr, V_proxy_arr)
    else:
        pr = 0.0

    return {
        "viol_rate": viol_rate,
        "pearson": pr,
        "final_V": final_V,
        "auc_V": auc_V,
        "convergence": converged,
        "deadlock_rate": deadlock,
        "nonzero_frac": nonzero,
    }


def main():
    rows = []
    for ncfg in NOISE_CONFIGS:
        for use_ema in [True, False]:
            ctrl = "C3_EMA" if use_ema else "C3_raw"
            for si in range(SEEDS):
                rng = np.random.default_rng(BASE_SEED + si)
                theta0 = rng.normal(0, 2, DIM)
                noise = gen_noise_arr(T, DIM, H_RATE, rng, ncfg["gen"])
                m = run(theta0, noise, use_ema=use_ema)
                rows.append({"noise": ncfg["name"], "controller": ctrl, "seed": si, **m})
        print(f"  Done: {ncfg['name']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3b_full_metrics.csv", index=False)

    agg = df.groupby(["noise", "controller"]).agg(
        viol_rate=("viol_rate", "mean"),
        pearson=("pearson", "mean"),
        convergence=("convergence", "mean"),
        final_V=("final_V", "mean"),
        deadlock=("deadlock_rate", "mean"),
    ).round(4)

    lines = ["=== E3b Full Metrics (per-noise) ===\n"]
    for noise in ["Gaussian", "Student-t(3)", "Student-t(1.5)"]:
        lines.append(f"--- {noise} ---")
        for ctrl in ["C3_EMA", "C3_raw"]:
            r = agg.loc[(noise, ctrl)]
            lines.append(
                f"  {ctrl:10s} viol={r.viol_rate:.3f} "
                f"pearson={r.pearson:.3f} "
                f"conv={r.convergence:.3f} "
                f"final_V={r.final_V:.2f} "
                f"deadlock={r.deadlock:.3f}"
            )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(OUT / "e3b_full_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
