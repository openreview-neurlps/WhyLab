# -*- coding: utf-8 -*-
"""E6: Non-Stationary Online Learning Agent with Full Audit Validation.

This experiment validates all three WhyLab components (C1+C2+C3) in a
non-stationary environment where:
- C1 detects distributional drift when the target shifts
- C2 filters noisy/spurious gradient updates
- C3 bounds the step-size via Lyapunov damping

The agent optimizes continuous parameters theta in R^d to minimize
||theta - theta*(t)||^2, where theta*(t) shifts at known drift points.
"""
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Configuration ──────────────────────────────────────────────

@dataclass
class E6Config:
    d: int = 10                    # parameter dimension
    T: int = 600                   # total steps
    drift_points: Tuple = (200, 400)  # when theta* shifts
    drift_magnitude: float = 3.0   # how far theta* moves
    lr_base: float = 0.1           # base learning rate
    h_rate: float = 0.3            # hallucination noise rate
    noise_std: float = 1.0         # observation noise
    beta_ema: float = 0.9          # EMA smoothing for C3
    c2_threshold: float = 1.5      # E-value threshold for C2
    c2_rv_threshold: float = 0.05  # RV threshold for C2
    c1_window: int = 20            # C1 drift detection window
    c1_threshold: float = 2.0      # C1 drift alarm threshold
    floor: float = 0.01            # C3 floor
    ceiling: float = 0.8           # C3 ceiling
    seed: int = 0


# ── Environment ────────────────────────────────────────────────

class NonStationaryEnv:
    """Environment with shifting optimal parameters."""

    def __init__(self, cfg: E6Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # Generate theta* for each phase
        self.targets = []
        base = rng.standard_normal(cfg.d)
        self.targets.append(base.copy())
        for _ in cfg.drift_points:
            shift = rng.standard_normal(cfg.d)
            shift = shift / np.linalg.norm(shift) * cfg.drift_magnitude
            base = base + shift
            self.targets.append(base.copy())

    def get_target(self, t: int) -> np.ndarray:
        phase = 0
        for dp in self.cfg.drift_points:
            if t >= dp:
                phase += 1
        return self.targets[min(phase, len(self.targets) - 1)]

    def observe(self, theta: np.ndarray, t: int) -> Tuple[float, np.ndarray]:
        """Return (reward, noisy_gradient)."""
        target = self.get_target(t)
        diff = theta - target
        true_reward = -0.5 * np.sum(diff ** 2)

        # Noisy gradient: true gradient + noise + hallucination
        true_grad = -diff  # gradient of -0.5||theta-theta*||^2
        noise = self.rng.standard_normal(self.cfg.d) * self.cfg.noise_std

        # Hallucination: with prob h_rate, replace gradient with random
        if self.rng.random() < self.cfg.h_rate:
            halluc = self.rng.standard_normal(self.cfg.d) * 5.0
            noisy_grad = halluc
        else:
            noisy_grad = true_grad + noise

        obs_reward = true_reward + self.rng.standard_normal() * self.cfg.noise_std
        return obs_reward, noisy_grad


# ── Audit Components ───────────────────────────────────────────

class C1DriftDetector:
    """Information-theoretic drift detection using sliding window."""

    def __init__(self, window: int, threshold: float):
        self.window = window
        self.threshold = threshold
        self.reward_history: List[float] = []
        self.drift_detected = False
        self.drift_index = 0.0

    def update(self, reward: float) -> bool:
        self.reward_history.append(reward)
        if len(self.reward_history) < 2 * self.window:
            self.drift_detected = False
            self.drift_index = 0.0
            return False

        recent = self.reward_history[-self.window:]
        past = self.reward_history[-2 * self.window:-self.window]

        # KL-divergence approximation via mean/std shift
        mu_past, std_past = np.mean(past), max(np.std(past), 1e-8)
        mu_recent, std_recent = np.mean(recent), max(np.std(recent), 1e-8)

        # Entropy-weighted drift index
        entropy_past = 0.5 * np.log(2 * np.pi * np.e * std_past ** 2)
        entropy_recent = 0.5 * np.log(2 * np.pi * np.e * std_recent ** 2)
        weight = 1.0 / (min(entropy_past, entropy_recent) + 1e-8)

        mean_shift = abs(mu_recent - mu_past) / std_past
        self.drift_index = weight * mean_shift

        self.drift_detected = self.drift_index > self.threshold
        return self.drift_detected


class C2SensitivityFilter:
    """Sensitivity-aware effect filtering using E-value approximation."""

    def __init__(self, e_threshold: float, rv_threshold: float):
        self.e_threshold = e_threshold
        self.rv_threshold = rv_threshold
        self.reward_buffer: List[float] = []

    def should_accept(self, reward_before: float, reward_after: float,
                      grad_norm: float) -> bool:
        delta = reward_after - reward_before
        self.reward_buffer.append(delta)

        if len(self.reward_buffer) < 3:
            return True  # not enough data

        # E-value: how much would an unmeasured confounder need to explain delta?
        recent = self.reward_buffer[-5:]
        mu = np.mean(recent)
        se = max(np.std(recent) / np.sqrt(len(recent)), 1e-8)

        e_value = max(abs(mu) / se, 1e-10)

        # Partial R^2 bound (simplified)
        total_var = max(np.var(self.reward_buffer[-20:]), 1e-8) if len(self.reward_buffer) >= 20 else max(np.var(self.reward_buffer), 1e-8)
        rv = delta ** 2 / (total_var + delta ** 2)

        return e_value >= self.e_threshold and rv >= self.rv_threshold


class C3LyapunovDamper:
    """Lyapunov-bounded adaptive step-size controller."""

    def __init__(self, beta: float, floor: float, ceiling: float):
        self.beta = beta
        self.floor = floor
        self.ceiling = ceiling
        self.ema_second_moment = 1.0
        self.ema_zeta = 0.5

    def compute_zeta(self, grad: np.ndarray, drift_alert: bool) -> float:
        grad_norm_sq = np.sum(grad ** 2) + 1e-10
        self.ema_second_moment = (self.beta * self.ema_second_moment
                                  + (1 - self.beta) * grad_norm_sq)

        zeta_raw = (2.0 * np.sqrt(grad_norm_sq)) / (
            np.sqrt(self.ema_second_moment) + 1e-10 + grad_norm_sq)

        self.ema_zeta = self.beta * self.ema_zeta + (1 - self.beta) * zeta_raw

        # If drift detected, reduce ceiling
        effective_ceiling = self.ceiling * 0.3 if drift_alert else self.ceiling
        zeta = np.clip(self.ema_zeta, self.floor, effective_ceiling)
        return zeta


# ── Agent ──────────────────────────────────────────────────────

def run_episode(cfg: E6Config, ablation: str) -> dict:
    """Run one episode with given ablation config."""
    rng = np.random.default_rng(cfg.seed)
    env = NonStationaryEnv(cfg, rng)

    # Initialize
    theta = rng.standard_normal(cfg.d) * 0.5

    # Components
    use_c1 = "C1" in ablation or ablation == "full"
    use_c2 = "C2" in ablation or ablation == "full"
    use_c3 = "C3" in ablation or ablation == "full"

    c1 = C1DriftDetector(cfg.c1_window, cfg.c1_threshold) if use_c1 else None
    c2 = C2SensitivityFilter(cfg.c2_threshold, cfg.c2_rv_threshold) if use_c2 else None
    c3 = C3LyapunovDamper(cfg.beta_ema, cfg.floor, cfg.ceiling) if use_c3 else None

    # Tracking
    rewards = []
    energies = []
    drift_detected_steps = []
    c2_rejections = 0
    oscillations = 0
    prev_improving = None

    for t in range(cfg.T):
        target = env.get_target(t)
        v_true = 0.5 * np.sum((theta - target) ** 2)
        energies.append(v_true)

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # C1: drift detection
        drift_alert = False
        if c1 is not None:
            drift_alert = c1.update(reward)
            if drift_alert and t not in drift_detected_steps:
                drift_detected_steps.append(t)

        # C2: sensitivity filter
        if c2 is not None:
            reward_before = reward
            # Estimate reward after (lookahead)
            theta_candidate = theta + cfg.lr_base * grad
            reward_after, _ = env.observe(theta_candidate, t)
            accept = c2.should_accept(reward_before, reward_after,
                                       np.linalg.norm(grad))
            if not accept:
                c2_rejections += 1
                continue  # skip update

        # C3: step-size control
        if c3 is not None:
            zeta = c3.compute_zeta(grad, drift_alert)
        else:
            zeta = cfg.lr_base

        # Update
        effective_lr = zeta if use_c3 else cfg.lr_base
        theta = theta + effective_lr * grad

        # Track oscillation
        improving = reward > np.mean(rewards[-10:]) if len(rewards) > 10 else True
        if prev_improving is not None and improving != prev_improving:
            oscillations += 1
        prev_improving = improving

    # Compute metrics
    target_final = env.get_target(cfg.T - 1)
    final_energy = 0.5 * np.sum((theta - target_final) ** 2)

    # Drift detection accuracy
    actual_drifts = list(cfg.drift_points)
    detection_delays = []
    for dp in actual_drifts:
        detected = [d for d in drift_detected_steps if dp <= d <= dp + 50]
        if detected:
            detection_delays.append(detected[0] - dp)
        else:
            detection_delays.append(50)  # max (not detected)

    # Recovery: energy after drift
    recovery_energies = []
    for dp in actual_drifts:
        if dp + 50 < cfg.T:
            recovery_energies.append(energies[min(dp + 50, cfg.T - 1)])

    # Regret
    total_regret = sum(-r for r in rewards)  # lower reward = higher regret

    return {
        "seed": cfg.seed,
        "ablation": ablation,
        "h_rate": cfg.h_rate,
        "final_energy": final_energy,
        "mean_energy": np.mean(energies),
        "max_energy": np.max(energies),
        "total_regret": total_regret,
        "oscillation_count": oscillations,
        "oscillation_index": oscillations / cfg.T,
        "c2_rejections": c2_rejections,
        "c2_rejection_rate": c2_rejections / cfg.T,
        "drift_detections": len(drift_detected_steps),
        "mean_detection_delay": np.mean(detection_delays) if detection_delays else 50,
        "mean_recovery_energy": np.mean(recovery_energies) if recovery_energies else final_energy,
    }


# ── Main ───────────────────────────────────────────────────────

def main():
    ablations = ["none", "C1_only", "C2_only", "C3_only",
                 "C1+C3", "C2+C3", "full"]
    h_rates = [0.0, 0.3, 0.5]
    n_seeds = 20

    results = []
    lr_scenarios = [0.1, 0.5]  # conservative and aggressive learning rates

    for lr in lr_scenarios:
        for h_rate in h_rates:
            for ablation in ablations:
                for seed in range(n_seeds):
                    cfg = E6Config(seed=seed, h_rate=h_rate, lr_base=lr)
                    row = run_episode(cfg, ablation)
                    row["lr_base"] = lr
                    results.append(row)

            print(f"lr={lr}, h_rate={h_rate} done ({len(ablations)*n_seeds} episodes)")

    df = pd.DataFrame(results)
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "e6_metrics.csv"), index=False)

    # Summary
    summary = df.groupby(["lr_base", "h_rate", "ablation"]).agg(
        final_energy=("final_energy", "mean"),
        mean_energy=("mean_energy", "mean"),
        regret=("total_regret", "mean"),
        osc_index=("oscillation_index", "mean"),
        c2_rej_rate=("c2_rejection_rate", "mean"),
        detection_delay=("mean_detection_delay", "mean"),
        recovery_energy=("mean_recovery_energy", "mean"),
    ).round(4)

    print("\n" + "=" * 70)
    print("E6: NON-STATIONARY AGENT — FULL RESULTS")
    print("=" * 70)
    print(summary.to_string())

    summary.to_csv(os.path.join(out_dir, "e6_summary.csv"))


if __name__ == "__main__":
    main()
