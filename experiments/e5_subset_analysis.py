# -*- coding: utf-8 -*-
"""Phase 6: E5 Deep Subset Analysis for Paper Narrative Restructuring."""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(r"d:\00.test\PAPER\WhyLab\experiments\results")
df = pd.read_csv(OUT / "e5_metrics.csv")

results = []

# ============================================================
# 1. Identify oscillating problems (from baseline 'none')
# ============================================================
none = df[df['ablation'] == 'none']
osc_instances = none[none['oscillation_count'] > 0]['instance_id'].unique()
non_osc_instances = none[none['oscillation_count'] == 0]['instance_id'].unique()
# Also: problems that required multiple attempts
multi_instances = none[none['total_attempts'] > 1]['instance_id'].unique()

results.append("=" * 60)
results.append("E5 SUBSET ANALYSIS")
results.append("=" * 60)
results.append(f"\nTotal unique problems: {none['instance_id'].nunique()}")
results.append(f"Problems with oscillation (baseline): {len(osc_instances)}")
results.append(f"Problems needing >1 attempt: {len(multi_instances)}")
results.append(f"Problems solved on 1st attempt: {len(non_osc_instances) - len(set(multi_instances) - set(osc_instances))}")

# ============================================================
# 2. Per-ablation metrics on OSCILLATING SUBSET only
# ============================================================
results.append("\n" + "=" * 60)
results.append("TABLE A: Oscillating Problems Only (45 problems)")
results.append("=" * 60)
results.append(f"{'Ablation':<20} {'Pass':<8} {'OscFree':<8} {'MeanOsc':<10} {'MeanAttempts':<12} {'Recovery'}")

for abl in ['none', 'simple_retry', 'C1_only', 'C2_calibrated', 'C3_only', 'full_calibrated']:
    sub = df[(df['ablation'] == abl) & (df['instance_id'].isin(osc_instances))]
    pass_rate = sub['final_passed'].mean()
    osc_free = (sub['oscillation_count'] == 0).mean()
    mean_osc = sub['oscillation_index'].mean()
    mean_att = sub['total_attempts'].mean()
    recovery = ((sub['first_pass_attempt'] > 1) & (sub['final_passed'] == 1)).sum()
    results.append(f"{abl:<20} {pass_rate:<8.3f} {osc_free:<8.3f} {mean_osc:<10.4f} {mean_att:<12.2f} {recovery}")

# ============================================================
# 3. Per-ablation metrics on NON-OSCILLATING problems
# ============================================================
results.append("\n" + "=" * 60)
results.append("TABLE B: Non-Oscillating Problems (255 problems)")
results.append("=" * 60)
results.append(f"{'Ablation':<20} {'Pass':<8} {'MeanAttempts':<12}")

non_osc_only = set(none['instance_id'].unique()) - set(osc_instances)
for abl in ['none', 'simple_retry', 'C1_only', 'C2_calibrated', 'C3_only', 'full_calibrated']:
    sub = df[(df['ablation'] == abl) & (df['instance_id'].isin(non_osc_only))]
    pass_rate = sub['final_passed'].mean()
    mean_att = sub['total_attempts'].mean()
    results.append(f"{abl:<20} {pass_rate:<8.3f} {mean_att:<12.2f}")

# ============================================================
# 4. New Metrics: Convergence Efficiency, Oscillation-Free Rate
# ============================================================
results.append("\n" + "=" * 60)
results.append("TABLE C: New Metrics (ALL 300 problems)")
results.append("=" * 60)
results.append(f"{'Ablation':<20} {'Pass@1':<8} {'OscFreeRate':<12} {'ConvEff':<10} {'SafePass':<10} {'TotalOscEp':<10}")

for abl in ['none', 'simple_retry', 'C1_only', 'C2_calibrated', 'C3_only', 'full_calibrated']:
    sub = df[df['ablation'] == abl]
    pass_rate = sub['final_passed'].mean()
    osc_free_rate = (sub['oscillation_count'] == 0).mean()
    # Convergence Efficiency: among solved problems, 1/attempts
    solved = sub[sub['final_passed'] == 1]
    conv_eff = (1.0 / solved['total_attempts']).mean() if len(solved) > 0 else 0
    safe_pass = sub['safe_pass'].mean()
    total_osc = (sub['oscillation_count'] > 0).sum()
    results.append(f"{abl:<20} {pass_rate:<8.3f} {osc_free_rate:<12.3f} {conv_eff:<10.4f} {safe_pass:<10.3f} {total_osc:<10}")

# ============================================================
# 5. Pareto Analysis: Sweep C2 threshold effect
# ============================================================
results.append("\n" + "=" * 60)
results.append("TABLE D: C2 vs Baselines Key Comparison")
results.append("=" * 60)

for abl in ['none', 'simple_retry', 'C2_calibrated', 'full_calibrated']:
    sub = df[df['ablation'] == abl]
    osc_sub = df[(df['ablation'] == abl) & (df['instance_id'].isin(osc_instances))]
    results.append(f"\n{abl}:")
    results.append(f"  ALL: pass={sub.final_passed.mean():.3f} osc_index={sub.oscillation_index.mean():.4f} osc_episodes={int((sub.oscillation_count>0).sum())}")
    results.append(f"  OSC-SUBSET: pass={osc_sub.final_passed.mean():.3f} osc_index={osc_sub.oscillation_index.mean():.4f} osc_episodes={int((osc_sub.oscillation_count>0).sum())}")

# Write output
output = "\n".join(results)
print(output)
(OUT / "e5_subset_analysis.txt").write_text(output, encoding="utf-8")
print(f"\nSaved to {OUT / 'e5_subset_analysis.txt'}")
