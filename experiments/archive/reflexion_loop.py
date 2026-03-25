"""
Reflexion Loop Engine for HumanEval
====================================
Implements the Reflexion self-improvement loop:
  solve → test → reflect → retry (+ WhyLab audit gating)

Reference: Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning" (NeurIPS 2023).

The agent maintains an episodic memory of reflections that is
prepended to subsequent solve prompts, enabling learning from
past failures without weight updates.
"""
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experiments.humaneval_loader import (
    HumanEvalProblem,
    ExecutionResult,
    execute_solution,
)
from experiments.llm_client import CachedLLMClient, LLMResponse
from experiments.audit_layer import AgentAuditLayer, AuditDecision

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class AttemptRecord:
    """Record of a single solve attempt."""
    attempt_idx: int
    solution: str
    passed: bool
    cheap_score: float
    reflection: str = ""
    audit_decision: Optional[AuditDecision] = None
    execution_time_ms: float = 0.0


@dataclass
class EpisodeResult:
    """Result of a full Reflexion episode on one problem."""
    task_id: str
    attempts: list[AttemptRecord] = field(default_factory=list)
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


def _build_solve_prompt(
    problem: HumanEvalProblem,
    memory: list[str],
) -> str:
    """Build the full solve prompt with reflections prepended."""
    parts = []

    # Include reflections from memory
    if memory:
        parts.append("## Previous Reflections (learn from these)")
        for i, ref in enumerate(memory[-3:], 1):  # last 3 reflections
            parts.append(f"{i}. {ref}")
        parts.append("")

    # The actual task
    parts.append("## Task")
    parts.append("Complete the following Python function:")
    parts.append("```python")
    parts.append(problem.prompt)
    parts.append("```")
    parts.append("")
    parts.append("Write ONLY the function body (no signature, no docstring).")

    return "\n".join(parts)


def _build_reflect_prompt(
    problem: HumanEvalProblem,
    solution: str,
    exec_result: ExecutionResult,
) -> str:
    """Build the reflection prompt after a failed attempt."""
    parts = [
        "## Failed Solution",
        "```python",
        problem.prompt + solution,
        "```",
        "",
        "## Test Result",
        f"Status: {'PASS' if exec_result.passed else 'FAIL'}",
    ]

    if exec_result.stderr:
        parts.append(f"Error output:\n{exec_result.stderr[:500]}")
    if exec_result.stdout:
        parts.append(f"Stdout:\n{exec_result.stdout[:500]}")

    parts.append("")
    parts.append("Analyze what went wrong and how to fix it in the next attempt.")

    return "\n".join(parts)


def _compute_edit_distance_normalized(old: str, new: str) -> float:
    """Compute normalized edit distance between two strings.

    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    Normalized by max token length to avoid bias toward longer texts.
    """
    if not old and not new:
        return 0.0

    old_tokens = old.split()
    new_tokens = new.split()
    max_len = max(len(old_tokens), len(new_tokens), 1)

    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens)
    similarity = matcher.ratio()
    return 1.0 - similarity


def _extract_code(response: str, prompt: str = "") -> str:
    """Extract Python code from LLM response and normalize indentation.

    Steps:
    1. Remove markdown code fences if present.
    2. Dedent all code to 0-indent (remove any existing indentation).
    3. Re-indent to match the expected function body level from prompt.

    This ensures correct indentation regardless of how the LLM formats
    its output (0-space, 4-space, or mixed).
    """
    import textwrap

    text = response.strip()

    # Remove markdown code fences if present
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end_search = text[start:]
        end = start + end_search.index("```") if "```" in end_search else len(text)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        # Skip optional language identifier on same line
        nl = text.find("\n", start)
        if nl != -1:
            start = nl + 1
        end_search = text[start:]
        end = start + end_search.index("```") if "```" in end_search else len(text)
        text = text[start:end]

    # Strip leading/trailing blank lines (but preserve internal structure)
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    text = "\n".join(lines)

    # Dedent to 0-indent
    text = textwrap.dedent(text)

    # Re-indent to match expected body level
    target_indent = _detect_body_indent(prompt) if prompt else 4
    if target_indent > 0:
        indent_str = " " * target_indent
        lines = text.split("\n")
        text = "\n".join(
            (indent_str + line if line.strip() else line)
            for line in lines
        )

    return text


def _detect_body_indent(prompt: str) -> int:
    """Detect the expected indentation for function body from prompt.

    Looks at the last non-empty line of the prompt (typically a docstring
    closing or empty line after docstring) to determine the indentation.
    """
    lines = prompt.rstrip().split("\n")
    for line in reversed(lines):
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            return indent
    return 4  # default




def run_reflexion_episode(
    problem: HumanEvalProblem,
    llm: CachedLLMClient,
    max_attempts: int = 5,
    audit: Optional[AgentAuditLayer] = None,
    seed: int = 0,
) -> EpisodeResult:
    """Run a full Reflexion episode on one HumanEval problem.

    Args:
        problem: The HumanEval problem to solve.
        llm: Cached LLM client.
        max_attempts: Maximum number of solve attempts.
        audit: Optional WhyLab audit layer (None = no auditing).
        seed: Seed for LLM cache key differentiation.

    Returns:
        EpisodeResult with all attempts and aggregated metrics.
    """
    solver_system = _load_prompt("solver.txt")
    reflector_system = _load_prompt("reflector.txt")

    memory: list[str] = []
    attempts: list[AttemptRecord] = []
    prev_solution = ""
    cheap_score_window: list[float] = []  # for C2 evaluation

    for attempt_idx in range(max_attempts):
        # 1. Solve
        solve_prompt = _build_solve_prompt(problem, memory)
        solve_resp = llm.generate(
            system_prompt=solver_system,
            user_prompt=solve_prompt,
            seed=seed * 1000 + attempt_idx,
        )

        solution = _extract_code(solve_resp.text, prompt=problem.prompt)

        # 2. Test (full eval — deterministic)
        exec_result = execute_solution(problem, solution)

        # 3. Record cheap score
        cheap_score = exec_result.cheap_score
        cheap_score_window.append(cheap_score)

        # 4. Audit gate (if enabled)
        audit_decision = None
        if audit is not None and attempt_idx > 0:
            update_magnitude = _compute_edit_distance_normalized(
                prev_solution, solution
            )
            # Use cheap scores before/after this attempt (window of last 3 for sigma stability)
            scores_before = cheap_score_window[-4:-1] if len(cheap_score_window) > 1 else [0.0]
            scores_after = cheap_score_window[-1:]

            audit_decision = audit.evaluate_update(
                cheap_score=cheap_score,
                full_pass=exec_result.passed,
                scores_before=scores_before,
                scores_after=scores_after,
                update_magnitude=update_magnitude,
            )

        # 5. Record attempt
        record = AttemptRecord(
            attempt_idx=attempt_idx,
            solution=solution,
            passed=exec_result.passed,
            cheap_score=cheap_score,
            audit_decision=audit_decision,
            execution_time_ms=exec_result.execution_time_ms,
        )

        # If audit rejects, revert to previous solution
        if audit_decision is not None and not audit_decision.accept:
            # Apply C3 damping even on rejection (partial update)
            if audit_decision.c3_damped < 1.0 and prev_solution:
                # Keep old solution (damped to near-zero update)
                record.solution = prev_solution
            else:
                record.solution = prev_solution if prev_solution else solution

        attempts.append(record)

        # 6. Early exit if passed
        if exec_result.passed:
            break

        # 7. Reflect (only on failure)
        reflect_prompt = _build_reflect_prompt(problem, solution, exec_result)
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
            prev_solution = solution
        # else: memory stays unchanged, prev_solution stays

    # Compute aggregate metrics
    result = EpisodeResult(task_id=problem.task_id)
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
