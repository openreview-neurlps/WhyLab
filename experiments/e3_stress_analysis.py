"""
E3-stress: Drifting Optimum Stress Test — Appendix Metrics
==========================================================
Reanalyzes the existing drifting-optimum results with appropriate
non-stationary metrics:
  - cumulative tracking error (time-averaged V_true)
  - recovery lag after drift onset
  - fraction of time in epsilon-tube
  - area under V_t curve
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

# Load existing drifting results
MET = pd.read_csv(OUT / "e3a_metrics.csv")  # from drifting run
STEP = pd.read_parquet(OUT / "e3a_stepwise.parquet")  # from drifting run

CTRL_ORDER = ["no_damping", "cosine", "grad_clip", "adaptive"]
EPS_TUBE = 5.0  # epsilon-tube around optimum


def compute_stress_metrics():
    rows = []

    for h in MET["hallucination_rate"].unique():
        for ctrl in CTRL_ORDER:
            sub_met = MET[(MET.hallucination_rate == h) & (MET.controller == ctrl)]
            sub_step = STEP[(STEP.hallucination_rate == h) & (STEP.controller == ctrl)]

            # Per-seed metrics from stepwise data
            for seed in sub_step.seed.unique():
                ss = sub_step[sub_step.seed == seed].sort_values("t")
                Vt = ss["V_true"].values
                T = len(Vt)

                # Cumulative tracking error = mean(V_true)
                cum_error = np.mean(Vt)

                # Area under V_t (same as cum_error * T)
                auc = np.sum(Vt)

                # Fraction of time in eps-tube
                in_tube = (Vt < EPS_TUBE).sum() / T

                # Recovery lag: after initial transient (first 50 steps),
                # how long until V_t first drops below eps_tube?
                if Vt[0] >= EPS_TUBE:
                    below = np.where(Vt[50:] < EPS_TUBE)[0]
                    recovery = int(below[0]) + 50 if len(below) > 0 else T
                else:
                    recovery = 0

                rows.append({
                    "h_rate": h, "controller": ctrl, "seed": seed,
                    "cum_tracking_error": cum_error,
                    "auc_V": auc,
                    "frac_in_tube": in_tube,
                    "recovery_lag": recovery,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3_stress_metrics.csv", index=False)

    # Summary
    agg = df.groupby(["h_rate", "controller"]).agg(
        cum_error=("cum_tracking_error", "mean"),
        frac_tube=("frac_in_tube", "mean"),
        recovery=("recovery_lag", "mean"),
    ).round(4).reset_index()
    agg.to_csv(OUT / "e3_stress_summary.csv", index=False)
    print("=== E3-Stress Summary ===")
    print(agg.to_string(index=False))
    return agg


if __name__ == "__main__":
    compute_stress_metrics()
