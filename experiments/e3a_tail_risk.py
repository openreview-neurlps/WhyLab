"""
E3a: Tail-Risk Metrics
======================
Extracts from existing stepwise logs:
  - 95th percentile of positive Delta V
  - Max overshoot
  - CVaR of positive Delta V (expected shortfall of upward jumps)
  - Mean Pearson across all h_rates (for accurate claim)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

CTRL_ORDER = ["B1_fixed", "B2_cosine", "B3_gradclip", "C3_proxy"]


def main():
    met = pd.read_csv(OUT / "e3a_stationary_metrics.csv")
    step = pd.read_parquet(OUT / "e3a_stationary_stepwise.parquet")

    # --- Tail-risk from stepwise ---
    risk_rows = []
    for h in [0.0, 0.1, 0.3, 0.5]:
        for ctrl in CTRL_ORDER:
            sub = step[(step.h_rate == h) & (step.controller == ctrl)]
            Vt = sub.sort_values(["seed", "t"])["V_true"].values
            # Compute per-seed, then average
            seeds = sub.seed.unique()
            p95s, maxos, cvars = [], [], []
            for s in seeds:
                ss = sub[sub.seed == s].sort_values("t")
                V = ss["V_true"].values
                dV = np.diff(V)
                pos_dV = dV[dV > 0]
                if len(pos_dV) > 0:
                    p95s.append(np.percentile(pos_dV, 95))
                    maxos.append(np.max(pos_dV))
                    # CVaR at 95%: expected value above 95th percentile
                    thresh = np.percentile(pos_dV, 95)
                    tail = pos_dV[pos_dV >= thresh]
                    cvars.append(np.mean(tail) if len(tail) > 0 else 0)
                else:
                    p95s.append(0)
                    maxos.append(0)
                    cvars.append(0)

            risk_rows.append({
                "h_rate": h, "controller": ctrl,
                "p95_dV_plus": np.mean(p95s),
                "max_overshoot": np.mean(maxos),
                "cvar95_dV_plus": np.mean(cvars),
            })

    df_risk = pd.DataFrame(risk_rows)
    df_risk.to_csv(OUT / "e3a_tail_risk.csv", index=False)

    # --- Mean Pearson across h_rates (for accurate claim) ---
    lines = []
    lines.append("=== E3a Mean Pearson across h_rates ===")
    for ctrl in CTRL_ORDER + ["C3_plugin"]:
        sub = met[met.controller == ctrl]
        if len(sub) == 0:
            continue
        avg_p = sub["pearson"].mean()
        lines.append(f"  {ctrl:15s} mean_pearson={avg_p:.4f}")

    lines.append("")
    lines.append("=== E3a Tail-Risk Metrics ===")
    for h in [0.0, 0.3, 0.5]:
        lines.append(f"--- h={h} ---")
        for ctrl in CTRL_ORDER:
            r = df_risk[(df_risk.h_rate == h) & (df_risk.controller == ctrl)].iloc[0]
            lines.append(f"  {ctrl:15s} p95={r.p95_dV_plus:.4f} "
                         f"max_over={r.max_overshoot:.4f} "
                         f"cvar95={r.cvar95_dV_plus:.4f}")
        lines.append("")

    report = "\n".join(lines)
    with open(OUT / "e3a_tail_risk_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
