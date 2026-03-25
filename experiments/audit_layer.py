"""
WhyLab C1-C3 Audit Layer for Agent Self-Improvement Loops
==========================================================
Wraps C1 (drift detection), C2 (sensitivity filter), and C3 (damping)
for use in Reflexion-style agent experiments.

Key design: mirrors the paper's definitions while adapting to
agent-specific signals (cheap eval → full eval calibration, etc.).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class AuditDecision:
    """Result of the audit layer's evaluation."""
    accept: bool               # whether the update should be accepted
    c1_alarm: bool = False     # C1: cheap↔full calibration drift detected
    c2_reject: bool = False    # C2: fragile improvement filtered
    c3_damped: float = 1.0     # C3: damping factor applied (0..1)
    details: Dict[str, Any] = field(default_factory=dict)


class DriftMonitor:
    """C1: Cheap↔Full eval calibration drift detector.

    Instead of monitoring pass rate directly (which would false-alarm on
    genuine improvement), we monitor the *calibration* between cheap eval
    scores and full eval outcomes. When the relationship breaks down
    (e.g., cheap says improved but full disagrees), we flag drift.

    Uses a windowed tracker of (cheap_score, full_pass) pairs and
    computes rolling agreement rate. When agreement drops below threshold,
    drift is flagged.
    """

    def __init__(self, window: int = 20, agreement_threshold: float = 0.6):
        self.window = window
        self.agreement_threshold = agreement_threshold
        self._history: list[tuple[float, bool]] = []  # (cheap_score, full_pass)

    def update(self, cheap_score: float, full_pass: bool) -> bool:
        """Record observation and return True if drift is detected."""
        self._history.append((cheap_score, full_pass))
        if len(self._history) < self.window:
            return False

        # Look at recent window
        recent = self._history[-self.window:]
        agreements = 0
        for cheap, full in recent:
            # Agreement: cheap > 0.5 and full pass, or cheap <= 0.5 and full fail
            cheap_predicts_pass = cheap > 0.5
            if cheap_predicts_pass == full:
                agreements += 1

        agreement_rate = agreements / self.window
        return agreement_rate < self.agreement_threshold

    def reset(self) -> None:
        """Reset history (e.g., after recalibration)."""
        self._history.clear()


class SensitivityGate:
    """C2: E-value + RV sensitivity filter for agent updates.

    Adapted from E2 (e2_sensitivity_filter.py) to work with
    agent update signals. Uses cheap eval scores as continuous
    'treatment effect' estimates.

    The E-value measures how strong an unobserved confounder must be
    to explain away the improvement (effect size / σ_pooled).
    The RV measures statistical precision (|β̂| / SE).

    "Fragile improvement" = cheap eval says improved, but the improvement
    is not robust to evaluation noise / partial testing bias.
    """

    def __init__(self, e_thresh: float = 2.0, rv_thresh: float = 0.1):
        self.e_thresh = e_thresh
        self.rv_thresh = rv_thresh
        self._history: List[Dict[str, Any]] = []  # track updates for stats

    def compute_evalue(self, delta: float, sigma_pooled: float) -> float:
        """E-value via VanderWeele & Ding (2017) continuous approximation.

        delta = |improvement| (cheap score difference)
        sigma_pooled = pooled SD of cheap scores
        """
        d = min(abs(delta) / max(sigma_pooled, 1e-8), 10.0)
        rr = np.exp(0.91 * d)
        return float(rr + np.sqrt(rr * (rr - 1.0))) if rr > 1.0 else 1.0

    def compute_rv(self, delta: float, se: float, q: float = 0.0) -> float:
        """Robustness Value via Cinelli & Hazlett (2020).

        f_q = (|delta| - q) / SE
        RV_q = 0.5 * (sqrt(f^4 + 4f^2) - f^2)
        """
        f_q = (abs(delta) - q) / max(se, 1e-8)
        if f_q <= 0:
            return 0.0
        return float(0.5 * (np.sqrt(f_q**4 + 4 * f_q**2) - f_q**2))

    def evaluate(
        self,
        scores_before: List[float],
        scores_after: List[float],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate whether improvement from before→after is robust.

        Args:
            scores_before: Cheap eval scores before update (per-problem).
            scores_after: Cheap eval scores after update (per-problem).

        Returns:
            (is_robust, details_dict)
        """
        before = np.array(scores_before, dtype=float)
        after = np.array(scores_after, dtype=float)

        delta = float(np.mean(after) - np.mean(before))
        sigma_pooled = float(np.sqrt(
            (np.var(before, ddof=1) + np.var(after, ddof=1)) / 2
        )) if len(before) > 1 and len(after) > 1 else 1.0

        n = min(len(before), len(after))
        se = sigma_pooled / np.sqrt(max(n, 1))

        e_val = self.compute_evalue(delta, sigma_pooled)
        rv_val = self.compute_rv(delta, se)

        is_robust = (e_val >= self.e_thresh) and (rv_val >= self.rv_thresh)

        details = {
            "delta": delta,
            "sigma_pooled": sigma_pooled,
            "se": se,
            "e_value": e_val,
            "rv_value": rv_val,
            "e_pass": e_val >= self.e_thresh,
            "rv_pass": rv_val >= self.rv_thresh,
        }
        self._history.append(details)
        return is_robust, details


class DampingController:
    """C3: Lyapunov-bounded adaptive damping for memory/prompt updates.

    Controls the 'step size' of reflection-based memory updates.
    Uses EMA-smoothed proxy (Eq. 7 in paper) adapted to
    edit-distance-based update magnitudes.

    Damping factor α ∈ [ε_floor, ceiling]:
      new_memory = (1 - α) * old_memory + α * proposed_memory
    """

    def __init__(
        self,
        epsilon_floor: float = 0.01,
        ceiling: float = 0.8,
        beta: float = 0.9,
    ):
        self.epsilon_floor = epsilon_floor
        self.ceiling = ceiling
        self.beta = beta
        self._ema_magnitude: Optional[float] = None
        self._ema_zeta: Optional[float] = None

    def compute_step_size(self, update_magnitude: float) -> float:
        """Compute damped step size based on update magnitude.

        Args:
            update_magnitude: Normalized edit distance of proposed update (0..1).

        Returns:
            Damping factor α ∈ [ε_floor, ceiling].
        """
        mag_sq = update_magnitude ** 2

        # EMA of magnitude squared (Eq. 6 in paper)
        if self._ema_magnitude is None:
            self._ema_magnitude = mag_sq
        else:
            self._ema_magnitude = self.beta * self._ema_magnitude + (1 - self.beta) * mag_sq

        # Raw zeta (Eq. 6): 2 * |g| / (ema_m2 + δ)
        zeta_raw = 2.0 * update_magnitude / (self._ema_magnitude + 1e-10)

        # EMA smoothing (Eq. 7)
        if self._ema_zeta is None:
            self._ema_zeta = zeta_raw
        else:
            self._ema_zeta = self.beta * self._ema_zeta + (1 - self.beta) * zeta_raw

        # Clip to [floor, ceiling]
        return float(np.clip(self._ema_zeta, self.epsilon_floor, self.ceiling))

    def reset(self):
        """Reset EMA state."""
        self._ema_magnitude = None
        self._ema_zeta = None


class AgentAuditLayer:
    """Unified C1-C3 audit layer for agent self-improvement.

    Configuration determines which components are active (for ablation).
    """

    def __init__(self, config: dict):
        self.enable_c1 = config.get("c1", False)
        self.enable_c2 = config.get("c2", False)
        self.enable_c3 = config.get("c3", False)

        # Initialize active components
        self.c1 = DriftMonitor(
            window=config.get("c1_window", 20),
            agreement_threshold=config.get("c1_agreement_threshold", 0.6),
        ) if self.enable_c1 else None

        self.c2 = SensitivityGate(
            e_thresh=config.get("c2_e_thresh", 2.0),
            rv_thresh=config.get("c2_rv_thresh", 0.1),
        ) if self.enable_c2 else None

        self.c3 = DampingController(
            epsilon_floor=config.get("c3_epsilon_floor", 0.01),
            ceiling=config.get("c3_ceiling", 0.8),
        ) if self.enable_c3 else None

    def evaluate_update(
        self,
        cheap_score: float,
        full_pass: bool,
        scores_before: list[float],
        scores_after: list[float],
        update_magnitude: float,
    ) -> AuditDecision:
        """Run all enabled audit gates on a proposed update.

        Args:
            cheap_score: Current cheap eval score for this problem.
            full_pass: Whether full eval passed for this problem.
            scores_before: Cheap eval scores before update (window).
            scores_after: Cheap eval scores after update (window).
            update_magnitude: Normalized edit distance of proposed update.

        Returns:
            AuditDecision with accept/reject and per-component details.
        """
        details = {}
        accept = True

        # C1: Drift alarm
        c1_alarm = False
        if self.c1 is not None:
            c1_alarm = self.c1.update(cheap_score, full_pass)
            details["c1_alarm"] = c1_alarm
            if c1_alarm:
                accept = False  # reject update during calibration drift

        # C2: Sensitivity filter
        c2_reject = False
        if self.c2 is not None and len(scores_before) > 0 and len(scores_after) > 0:
            is_robust, c2_details = self.c2.evaluate(scores_before, scores_after)
            c2_reject = not is_robust
            details["c2"] = c2_details
            if c2_reject:
                accept = False

        # C3: Damping (doesn't reject, but scales the update)
        damping = 1.0
        if self.c3 is not None:
            damping = self.c3.compute_step_size(update_magnitude)
            details["c3_damping"] = damping

        return AuditDecision(
            accept=accept,
            c1_alarm=c1_alarm,
            c2_reject=c2_reject,
            c3_damped=damping,
            details=details,
        )

    def get_config_label(self) -> str:
        """Return human-readable label for this audit configuration."""
        parts = []
        if self.enable_c1:
            parts.append("C1")
        if self.enable_c2:
            parts.append("C2")
        if self.enable_c3:
            parts.append("C3")
        return "+".join(parts) if parts else "none"
