"""
Sanity Check: C3-raw vs C3-EMA on E3a core metrics
===================================================
Runs BOTH controllers through E3a's EXACT stationary setup,
then compares:
  - violation rate P(ΔV > 0)
  - proxy-state Pearson
  - final V
  - AUC V_t

This is the decisive test: if raw wins on E3a's headline metrics,
the main C3 controller must be changed.
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

# --- Same environment as E3a ---
T = 1000
DIM = 5
A_diag = np.arange(1, DIM + 1, dtype=float)
A = np.diag(A_diag)
THETA_STAR = np.zeros(DIM)
R_STAR = 10.0
FLOOR = 0.01
CEIL = 0.8
ALPHA_EMA = 0.1  # same as E3a ctrl_proxy
DELTA = 1e-10

H_RATES = [0.0, 0.1, 0.3, 0.5]
NOISE_SIGMA = 0.5
HALL_SIGMA = 3.0


def gen_noise(T, dim, h_rate, rng):
    """Same noise generation as E3a."""
    noise = np.zeros((T, dim))
    is_hall = rng.random(T) < h_rate
    for t in range(T):
        sigma = HALL_SIGMA if is_hall[t] else NOISE_SIGMA
        noise[t] = rng.normal(0, sigma, dim)
    return noise


def run(theta0, noise_arr, use_ema=True):
    """Run one trajectory. Returns stepwise data."""
    theta = theta0.copy()
    ema_g_sq = None

    V_true_list = []
    V_proxy_list = []
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

        # Observable proxy (same as E3a)
        loss = 0.5 * theta @ A @ theta
        R_t = R_STAR - loss
        if R_max_ema is None:
            R_max_ema = R_t
        else:
            R_max_ema = 0.1 * max(R_t, R_max_ema) + 0.9 * R_max_ema
        V_proxy = 0.5 * (R_max_ema - R_t) ** 2
        V_proxy_list.append(V_proxy)

    V_true_arr = np.array(V_true_list)
    V_proxy_arr = np.array(V_proxy_list)

    # Metrics
    dV = np.diff(V_true_arr)
    viol_rate = np.mean(dV > 0)
    final_V = V_true_arr[-1]
    auc_V = np.mean(V_true_arr)

    if np.std(V_true_arr) > 1e-10 and np.std(V_proxy_arr) > 1e-10:
        pearson_r, _ = pearsonr(V_true_arr, V_proxy_arr)
    else:
        pearson_r = 0.0

    return {
        "viol_rate": viol_rate,
        "final_V": final_V,
        "auc_V": auc_V,
        "pearson": pearson_r,
    }


def main():
    rows = []

    for h in H_RATES:
        for use_ema in [True, False]:
            ctrl = "C3_EMA" if use_ema else "C3_raw"
            for seed_idx in range(SEEDS):
                rng = np.random.default_rng(BASE_SEED + seed_idx)
                theta0 = rng.normal(0, 2, DIM)
                noise = gen_noise(T, DIM, h, rng)

                m = run(theta0, noise, use_ema=use_ema)
                rows.append({"h_rate": h, "controller": ctrl, "seed": seed_idx, **m})

        print(f"  Done h={h}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sanity_raw_vs_ema.csv", index=False)

    # Summary
    lines = ["=== SANITY CHECK: C3-raw vs C3-EMA on E3a metrics ===\n"]
    agg = df.groupby(["h_rate", "controller"]).agg(
        viol_rate=("viol_rate", "mean"),
        final_V=("final_V", "mean"),
        auc_V=("auc_V", "mean"),
        pearson=("pearson", "mean"),
    ).round(4)

    for h in H_RATES:
        lines.append(f"--- h={h} ---")
        for ctrl in ["C3_EMA", "C3_raw"]:
            r = agg.loc[(h, ctrl)]
            lines.append(
                f"  {ctrl:10s} viol={r.viol_rate:.3f} "
                f"final_V={r.final_V:.2f} "
                f"auc_V={r.auc_V:.2f} "
                f"pearson={r.pearson:.4f}"
            )
        lines.append("")

    # Mean across h
    lines.append("--- MEAN across h ---")
    for ctrl in ["C3_EMA", "C3_raw"]:
        sub = df[df.controller == ctrl]
        lines.append(
            f"  {ctrl:10s} mean_viol={sub.viol_rate.mean():.3f} "
            f"mean_pearson={sub.pearson.mean():.4f} "
            f"mean_final_V={sub.final_V.mean():.2f}"
        )

    report = "\n".join(lines)
    print(report)
    with open(OUT / "sanity_check_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
