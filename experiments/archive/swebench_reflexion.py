"""
SWE-bench Reflexion Loop
=========================
Implements the Reflexion self-improvement loop for SWE-bench tasks:
  solve (generate patch) → test → reflect → retry (+ WhyLab audit gating)

Mirrors reflexion_loop.py (HumanEval) with adaptations:
  - Output: unified diff patches (not function bodies)
  - Input: issue descriptions + repo context (not function signatures)
  - Scoring: test pass ratio (continuous) not binary pass/fail
  - Update magnitude: patch diff size (not edit distance)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experiments.swebench_loader import (
    SWEProblem,
    PatchResult,
    apply_and_test_patch,
    compute_patch_magnitude,
)
from experiments.llm_client import CachedLLMClient
from experiments.audit_layer import AgentAuditLayer, AuditDecision

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class SWEAttemptRecord:
    """Record of a single patch attempt."""
    attempt_idx: int
    patch: str
    passed: bool
    cheap_score: float
    reflection: str = ""
    audit_decision: Optional[AuditDecision] = None
    execution_time_ms: float = 0.0
    tests_total: int = 0
    tests_passed: int = 0


@dataclass
class SWEEpisodeResult:
    """Result of a full Reflexion episode on one SWE-bench problem."""
    instance_id: str
    attempts: list[SWEAttemptRecord] = field(default_factory=list)
    final_passed: bool = False
    total_attempts: int = 0
    regression_count: int = 0      # pass→fail transitions
    oscillation_count: int = 0     # total pass↔fail transitions
    updates_accepted: int = 0
    updates_rejected: int = 0

    @property
    def oscillation_index(self) -> float:
        """Oscillation = total sign changes / max possible."""
        if self.total_attempts <= 1:
            return 0.0
        return self.oscillation_count / (self.total_attempts - 1)

    @property
    def first_pass_attempt(self) -> int:
        """1-indexed attempt number of first pass (0 if never passed)."""
        for i, a in enumerate(self.attempts):
            if a.passed:
                return i + 1
        return 0

    @property
    def safe_pass(self) -> bool:
        """True if passed with zero regressions (no pass→fail)."""
        return self.final_passed and self.regression_count == 0


def _load_prompt(name: str) -> str:
    """Load prompt template from prompts/ directory."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def _build_swe_solve_prompt(
    problem: SWEProblem,
    memory: list[str],
) -> str:
    """Build the full solve prompt for SWE-bench."""
    parts = []

    # Include reflections from memory
    if memory:
        parts.append("## Previous Reflections (learn from these)")
        for i, ref in enumerate(memory[-3:], 1):
            parts.append(f"{i}. {ref}")
        parts.append("")

    # The actual task
    parts.append("## Issue Description")
    parts.append(problem.problem_statement[:3000])
    parts.append("")

    if problem.hints_text:
        parts.append("## Hints")
        parts.append(problem.hints_text[:1000])
        parts.append("")

    parts.append(f"## Repository: {problem.repo}")
    parts.append(f"## Version: {problem.version}")
    parts.append("")
    parts.append("Generate a unified diff patch (`git apply` format) to fix this issue.")

    return "\n".join(parts)


def _build_swe_reflect_prompt(
    problem: SWEProblem,
    patch: str,
    result: PatchResult,
) -> str:
    """Build the reflection prompt after a failed patch attempt."""
    parts = [
        "## Failed Patch Attempt",
        "```diff",
        patch[:2000],
        "```",
        "",
        "## Test Result",
        f"Status: {'PASS' if result.passed else 'FAIL'}",
        f"Tests: {result.tests_passed}/{result.tests_total} passed",
    ]

    if result.stderr:
        parts.append(f"Error output:\n{result.stderr[:500]}")
    if result.stdout:
        parts.append(f"Test output:\n{result.stdout[:500]}")

    parts.append("")
    parts.append("Analyze what went wrong and how to fix it in the next attempt.")
    parts.append("Be specific about the root cause and the corrective action.")

    return "\n".join(parts)


def _extract_patch(response: str) -> str:
    """Extract unified diff patch from LLM response.

    Handles responses wrapped in code fences or containing extra text.
    """
    text = response.strip()

    # Extract from code fences
    if "```diff" in text:
        start = text.index("```diff") + len("```diff")
        end_text = text[start:]
        end = start + end_text.index("```") if "```" in end_text else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        nl = text.find("\n", start)
        if nl != -1:
            start = nl + 1
        end_text = text[start:]
        end = start + end_text.index("```") if "```" in end_text else len(text)
        text = text[start:end].strip()

    # If text starts with diff headers, use as-is
    if text.startswith("diff ") or text.startswith("--- "):
        return text

    # Try to find diff content within the response
    lines = text.splitlines()
    diff_start = None
    for i, line in enumerate(lines):
        if line.startswith("diff ") or line.startswith("--- "):
            diff_start = i
            break

    if diff_start is not None:
        return "\n".join(lines[diff_start:])

    # Return raw text as fallback (will likely fail patch apply)
    return text


def run_swe_reflexion_episode(
    problem: SWEProblem,
    llm: CachedLLMClient,
    max_attempts: int = 7,
    audit: Optional[AgentAuditLayer] = None,
    seed: int = 0,
    eval_mode: str = "lightweight",
    disable_reflection: bool = False,
) -> SWEEpisodeResult:
    """Run a full Reflexion episode on one SWE-bench problem.

    Args:
        problem: The SWE-bench problem to solve.
        llm: Cached LLM client.
        max_attempts: Maximum number of patch attempts.
        audit: Optional WhyLab audit layer (None = no auditing).
        seed: Seed for LLM cache key differentiation.
        eval_mode: "lightweight" or "docker".

    Returns:
        SWEEpisodeResult with all attempts and aggregated metrics.
    """
    solver_system = _load_prompt("swe_solver.txt")
    reflector_system = _load_prompt("swe_reflector.txt")

    memory: list[str] = []
    attempts: list[SWEAttemptRecord] = []
    prev_patch = ""
    cheap_score_window: list[float] = []

    for attempt_idx in range(max_attempts):
        # 1. Solve: generate patch
        solve_prompt = _build_swe_solve_prompt(problem, memory)
        solve_resp = llm.generate(
            system_prompt=solver_system,
            user_prompt=solve_prompt,
            seed=seed * 1000 + attempt_idx,
        )

        patch = _extract_patch(solve_resp.text)

        # 2. Test: apply patch and run tests
        test_result = apply_and_test_patch(
            problem, patch, mode=eval_mode,
        )

        # 3. Record cheap score
        cheap_score = test_result.cheap_score
        cheap_score_window.append(cheap_score)

        # 4. Audit gate (if enabled)
        audit_decision = None
        if audit is not None and attempt_idx > 0:
            update_magnitude = compute_patch_magnitude(prev_patch, patch)
            scores_before = cheap_score_window[-4:-1] if len(cheap_score_window) > 1 else [0.0]
            scores_after = cheap_score_window[-1:]

            audit_decision = audit.evaluate_update(
                cheap_score=cheap_score,
                full_pass=test_result.passed,
                scores_before=scores_before,
                scores_after=scores_after,
                update_magnitude=update_magnitude,
            )

        # 5. Record attempt
        record = SWEAttemptRecord(
            attempt_idx=attempt_idx,
            patch=patch,
            passed=test_result.passed,
            cheap_score=cheap_score,
            audit_decision=audit_decision,
            execution_time_ms=test_result.execution_time_ms,
            tests_total=test_result.tests_total,
            tests_passed=test_result.tests_passed,
        )

        # If audit rejects, revert to previous patch
        if audit_decision is not None and not audit_decision.accept:
            if prev_patch:
                record.patch = prev_patch

        attempts.append(record)

        # 6. Early exit if passed
        if test_result.passed:
            break

        # 7. Reflect (only on failure, skip if simple_retry)
        if disable_reflection:
            # Simple retry: no reflection, just retry with empty memory
            prev_patch = patch
            continue

        reflect_prompt = _build_swe_reflect_prompt(problem, patch, test_result)
        reflect_resp = llm.generate(
            system_prompt=reflector_system,
            user_prompt=reflect_prompt,
            seed=seed * 1000 + attempt_idx + 500,
        )
        reflection = reflect_resp.text.strip()
        record.reflection = reflection

        # 8. Update memory (respecting audit)
        if audit_decision is None or audit_decision.accept:
            memory.append(reflection)
            prev_patch = patch

    # Compute aggregate metrics
    result = SWEEpisodeResult(instance_id=problem.instance_id)
    result.attempts = attempts
    result.total_attempts = len(attempts)
    result.final_passed = attempts[-1].passed if attempts else False

    # Count regressions and oscillations
    for i in range(1, len(attempts)):
        prev_pass = attempts[i - 1].passed
        curr_pass = attempts[i].passed
        if prev_pass != curr_pass:
            result.oscillation_count += 1
            if prev_pass and not curr_pass:
                result.regression_count += 1

    # Count audit decisions
    for a in attempts:
        if a.audit_decision is not None:
            if a.audit_decision.accept:
                result.updates_accepted += 1
            else:
                result.updates_rejected += 1

    return result
