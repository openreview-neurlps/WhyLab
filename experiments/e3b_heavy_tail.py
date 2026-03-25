"""
E3b: Heavy-Tail Stress Test (aligned with E3a)
===============================================
Same stationary quadratic as E3a. Only noise distribution changes.
Compares C3-raw (proxy, no EMA) vs C3-EMA (proxy + EMA smoothing).

Controller definitions (aligned with E3a):
- C3-EMA: zeta = clip(EMA(2*||g||/(EMA(||g||^2)+δ)), floor, ceil)
  → identical to E3a ctrl_proxy
- C3-raw: zeta = clip(2*||g||/(||g||^2+δ), floor, ceil)
  → same proxy formula WITHOUT EMA smoothing [NEW]

NOT included here:
- C3-oracle (true θ*-based bound) → E3a appendix ablation only
"""
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]
SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]

OUT = ROOT / "results"
FIGS = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# --- Same environment as E3a ---
T = 1000
DIM = 5
A = np.diag(np.arange(1, DIM + 1, dtype=float))
THETA_STAR = np.zeros(DIM)
R_STAR = 10.0

# Controller params (same as E3a ctrl_proxy)
BETA_EMA = 0.9   # for second-moment EMA
ALPHA_ZETA = 0.1  # for zeta EMA (same as E3a: state["ema"] = 0.1*g_sq + 0.9*state["ema"])
DELTA = 1e-10
FLOOR = 0.01
CEIL = 0.8

# Noise configs
NOISE_CONFIGS = [
    {"name": "Gaussian",
     "gen": lambda rng, dim, h_rate: rng.normal(0, 1, dim) * (3.0 if rng.random() < h_rate else 0.5)},
    {"name": "Student-t(3)",
     "gen": lambda rng, dim, h_rate: rng.standard_t(3, dim) * (3.0 if rng.random() < h_rate else 0.5)},
    {"name": "Student-t(1.5)",
     "gen": lambda rng, dim, h_rate: rng.standard_t(1.5, dim) * (3.0 if rng.random() < h_rate else 0.5)},
]

H_RATE = 0.5  # same as E3a worst case


def run_trajectory(theta0, noise_gen, rng, use_ema=True):
    """Run one trajectory. Controller uses SAME proxy as E3a ctrl_proxy."""
    theta = theta0.copy()
    ema_g_sq = None  # second-moment EMA (same init as E3a)

    V_trace = []
    zeta_trace = []
    deadlock_steps = 0

    for t in range(T):
        # True gradient (same as E3a env.gradient)
        g = A @ (theta - THETA_STAR)

        # Noisy gradient (same structure as E3a gen_noise)
        noise = noise_gen(rng, DIM, H_RATE)
        g_hat = g + noise

        # Proxy-based zeta (same formula as E3a ctrl_proxy)
        g_sq = np.dot(g_hat, g_hat)
        g_norm = np.linalg.norm(g_hat)

        if use_ema:
            # C3-EMA: identical to E3a ctrl_proxy
            if ema_g_sq is None:
                ema_g_sq = g_sq
            else:
                ema_g_sq = ALPHA_ZETA * g_sq + (1 - ALPHA_ZETA) * ema_g_sq
            z_raw = 2.0 * g_norm / (ema_g_sq + DELTA)
        else:
            # C3-raw: same proxy formula, NO smoothing
            z_raw = 2.0 * g_norm / (g_sq + DELTA)

        zeta = float(np.clip(z_raw, FLOOR, CEIL))

        # Update (same as E3a)
        theta = theta - zeta * g_hat

        V = 0.5 * np.dot(theta - THETA_STAR, theta - THETA_STAR)
        V_trace.append(V)
        zeta_trace.append(zeta)

        if zeta <= FLOOR * 1.01:
            deadlock_steps += 1

    return {
        "V_trace": np.array(V_trace),
        "zeta_trace": np.array(zeta_trace),
        "deadlock_steps": deadlock_steps,
        "final_V": V_trace[-1],
        "V_0": V_trace[0] if V_trace else 0,
    }


def main():
    rows = []

    for noise_cfg in NOISE_CONFIGS:
        for use_ema in [True, False]:
            ctrl_name = "C3_EMA" if use_ema else "C3_raw"
            for seed_idx in range(SEEDS):
                rng = np.random.default_rng(BASE_SEED + seed_idx)
                theta0 = rng.normal(0, 2, DIM)

                res = run_trajectory(theta0, noise_cfg["gen"], rng, use_ema=use_ema)

                nonzero_frac = np.mean(res["zeta_trace"] > FLOOR * 1.01)
                converged = 1 if res["final_V"] < res["V_0"] else 0

                rows.append({
                    "noise": noise_cfg["name"],
                    "controller": ctrl_name,
                    "seed": seed_idx,
                    "deadlock_steps": res["deadlock_steps"],
                    "deadlock_rate": res["deadlock_steps"] / T,
                    "nonzero_frac": nonzero_frac,
                    "convergence": converged,
                    "final_V": res["final_V"],
                    "V_0": res["V_0"],
                })

        print(f"  Done: {noise_cfg['name']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3b_heavy_tail.csv", index=False)

    # Summary
    lines = ["=== E3b Heavy-Tail Stress Test (aligned) ===\n"]
    agg = df.groupby(["noise", "controller"]).agg(
        deadlock_rate=("deadlock_rate", "mean"),
        nonzero_frac=("nonzero_frac", "mean"),
        convergence=("convergence", "mean"),
        final_V=("final_V", "mean"),
    ).round(4)

    for noise in ["Gaussian", "Student-t(3)", "Student-t(1.5)"]:
        lines.append(f"--- {noise} ---")
        for ctrl in ["C3_EMA", "C3_raw"]:
            r = agg.loc[(noise, ctrl)]
            lines.append(
                f"  {ctrl:10s} deadlock={r.deadlock_rate:.3f} "
                f"nonzero={r.nonzero_frac:.3f} "
                f"converged={r.convergence:.3f} "
                f"final_V={r.final_V:.2f}"
            )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(OUT / "e3b_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics_list = ["deadlock_rate", "nonzero_frac", "convergence"]
    titles = ["(a) Deadlock rate ↓", "(b) Nonzero-step fraction ↑", "(c) Convergence ratio ↑"]
    colors = {"C3_EMA": "#2563EB", "C3_raw": "#9CA3AF"}
    labels = {"C3_EMA": "C3-EMA (ours)", "C3_raw": "C3-raw (no smoothing)"}

    for idx, (met, title) in enumerate(zip(metrics_list, titles)):
        ax = axes[idx]
        noises = ["Gaussian", "Student-t(3)", "Student-t(1.5)"]
        x = np.arange(len(noises))
        w = 0.3

        for i, ctrl in enumerate(["C3_EMA", "C3_raw"]):
            vals = [agg.loc[(n, ctrl), met] for n in noises]
            ax.bar(x + i * w, vals, w, label=labels[ctrl],
                   color=colors[ctrl], alpha=0.85, edgecolor="white")

        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(noises, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / "e3b_heavy_tail.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "e3b_heavy_tail.png", dpi=200, bbox_inches="tight")
    print(f"\n[E3b] Figure saved")


if __name__ == "__main__":
    main()
