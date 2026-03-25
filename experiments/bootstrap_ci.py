# -*- coding: utf-8 -*-
"""Bootstrap CI for oscillation index difference (none vs C2)."""
import pandas as pd
import numpy as np

df = pd.read_csv(r"d:\00.test\PAPER\WhyLab\experiments\results\e5_metrics.csv")

# Get oscillation values per episode 
# osc_index is per-episode metric
none_osc = df[df["ablation"] == "none"]["oscillation_index"].values
c2_osc = df[df["ablation"] == "C2_calibrated"]["oscillation_index"].values

print(f"N(none) = {len(none_osc)}, N(C2) = {len(c2_osc)}")
print(f"Mean osc none = {none_osc.mean():.4f}")
print(f"Mean osc C2   = {c2_osc.mean():.4f}")
print(f"Delta         = {none_osc.mean() - c2_osc.mean():.4f}")

# Bootstrap CI for difference of means
np.random.seed(42)
n_boot = 10000
deltas = np.zeros(n_boot)

for i in range(n_boot):
    boot_none = np.random.choice(none_osc, size=len(none_osc), replace=True)
    boot_c2 = np.random.choice(c2_osc, size=len(c2_osc), replace=True)
    deltas[i] = boot_none.mean() - boot_c2.mean()

ci_lo = np.percentile(deltas, 2.5)
ci_hi = np.percentile(deltas, 97.5)
p_value = np.mean(deltas <= 0)  # one-sided: is delta > 0?

print(f"\nBootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"p-value (one-sided): {p_value:.4f}")
print(f"Significant at 0.05: {p_value < 0.05}")

# Also test pass rate difference
none_pass = df[df["ablation"] == "none"]["final_passed"].values
c2_pass = df[df["ablation"] == "C2_calibrated"]["final_passed"].values
print(f"\nPass rate none = {none_pass.mean():.4f}")
print(f"Pass rate C2   = {c2_pass.mean():.4f}")
print(f"Delta pass     = {none_pass.mean() - c2_pass.mean():.4f}")

# Bootstrap for pass rate
deltas_pass = np.zeros(n_boot)
for i in range(n_boot):
    b_none = np.random.choice(none_pass, size=len(none_pass), replace=True)
    b_c2 = np.random.choice(c2_pass, size=len(c2_pass), replace=True)
    deltas_pass[i] = b_none.mean() - b_c2.mean()

ci_lo_p = np.percentile(deltas_pass, 2.5)
ci_hi_p = np.percentile(deltas_pass, 97.5)
print(f"Pass delta 95% CI: [{ci_lo_p:.4f}, {ci_hi_p:.4f}]")
