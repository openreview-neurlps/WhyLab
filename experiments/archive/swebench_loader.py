"""
SWE-bench Lite Dataset Loader & Patch Executor
================================================
Loads SWE-bench Lite problems and evaluates generated patches.

Architecture mirrors humaneval_loader.py:
  SWEProblem  ↔  HumanEvalProblem
  apply_and_test_patch()  ↔  execute_solution()

Evaluation modes:
  1. Docker-based (production): Full SWE-bench harness in containers
  2. Lightweight (pilot): Applies patch + runs pytest in subprocess

Pre-requisites:
  pip install datasets
  (Docker mode also requires: docker, swebench package)
"""
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
SWEBENCH_CACHE = DATA_DIR / "swebench_lite.json"
TIMEOUT_SECONDS = 120  # SWE-bench tasks need more time than HumanEval
REPOS_DIR = DATA_DIR / "swebench_repos"


@dataclass
class SWEProblem:
    """A single SWE-bench Lite problem."""
    instance_id: str          # e.g. "astropy__astropy-12907"
    repo: str                 # e.g. "astropy/astropy"
    base_commit: str          # git commit to checkout
    problem_statement: str    # GitHub issue description
    hints_text: str           # any hints from the issue
    test_patch: str           # gold test patch (for evaluation)
    patch: str                # gold fix patch (reference only)
    version: str              # repo version tag
    created_at: str           # timestamp
    idx: int = 0              # 0-based index

    @property
    def repo_name(self) -> str:
        """Short repo name (e.g. 'astropy')."""
        return self.repo.split("/")[-1] if "/" in self.repo else self.repo


@dataclass
class PatchResult:
    """Result of evaluating a generated patch."""
    passed: bool
    tests_total: int = 0
    tests_passed: int = 0
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""     # "timeout", "apply_fail", "test_fail", "runtime", ""
    execution_time_ms: float = 0.0

    @property
    def cheap_score(self) -> float:
        """Fraction of tests passed (0.0 ~ 1.0)."""
        if self.tests_total == 0:
            return 1.0 if self.passed else 0.0
        return self.tests_passed / self.tests_total


def download_swebench_lite(
    target_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download SWE-bench Lite dataset from HuggingFace.

    Returns path to cached JSON file.
    """
    target_dir = target_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / "swebench_lite.json"

    if cache_path.exists() and not force:
        print(f"[SWE-bench] Already cached: {cache_path}")
        return cache_path

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "SWE-bench loader requires `datasets` package. "
            "Install with: pip install datasets"
        )

    print("[SWE-bench] Loading princeton-nlp/SWE-bench_Lite from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    records = []
    for row in ds:
        records.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
            "hints_text": row.get("hints_text", ""),
            "test_patch": row.get("test_patch", ""),
            "patch": row.get("patch", ""),
            "version": row.get("version", ""),
            "created_at": row.get("created_at", ""),
        })

    cache_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[SWE-bench] Cached {len(records)} problems → {cache_path}")
    return cache_path


def load_swebench_lite(
    path: Optional[Path] = None,
    subset: Optional[int] = None,
) -> list[SWEProblem]:
    """Load SWE-bench Lite problems from cached JSON.

    Args:
        path: Path to swebench_lite.json. If not found, downloads automatically.
        subset: If set, return only the first N problems.

    Returns:
        List of SWEProblem instances.
    """
    path = path or SWEBENCH_CACHE

    if not path.exists():
        download_swebench_lite(path.parent)

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    problems = []
    for idx, rec in enumerate(records):
        problems.append(SWEProblem(
            instance_id=rec["instance_id"],
            repo=rec["repo"],
            base_commit=rec["base_commit"],
            problem_statement=rec["problem_statement"],
            hints_text=rec.get("hints_text", ""),
            test_patch=rec.get("test_patch", ""),
            patch=rec.get("patch", ""),
            version=rec.get("version", ""),
            created_at=rec.get("created_at", ""),
            idx=idx,
        ))

    if subset is not None:
        problems = problems[:subset]

    print(f"[SWE-bench] Loaded {len(problems)} problems")
    return problems


def _count_diff_lines(patch_str: str) -> int:
    """Count number of changed lines in a unified diff patch."""
    count = 0
    for line in patch_str.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1
    return count


def compute_patch_magnitude(old_patch: str, new_patch: str) -> float:
    """Compute normalized magnitude of change between patches.

    Returns value in [0, 1] based on symmetric diff of changed lines.
    """
    if not old_patch and not new_patch:
        return 0.0

    old_lines = set(old_patch.splitlines())
    new_lines = set(new_patch.splitlines())

    symmetric_diff = len(old_lines.symmetric_difference(new_lines))
    total = max(len(old_lines | new_lines), 1)

    return min(symmetric_diff / total, 1.0)


def apply_and_test_patch(
    problem: SWEProblem,
    generated_patch: str,
    mode: str = "lightweight",
    timeout: int = TIMEOUT_SECONDS,
) -> PatchResult:
    """Apply a generated patch and run tests.

    Args:
        problem: SWE-bench problem.
        generated_patch: The generated unified diff patch string.
        mode: "lightweight" (subprocess) or "docker" (full SWE-bench harness).
        timeout: Max execution time in seconds.

    Returns:
        PatchResult with pass/fail and test scores.
    """
    if mode == "docker":
        return _evaluate_docker(problem, generated_patch, timeout)
    else:
        return _evaluate_lightweight(problem, generated_patch, timeout)


def _evaluate_lightweight(
    problem: SWEProblem,
    generated_patch: str,
    timeout: int,
) -> PatchResult:
    """Lightweight evaluation: clone/checkout repo, apply patch, run tests.

    This is faster but less accurate than Docker mode. Suitable for
    piloting and development. Falls back to syntax/format checking
    if repo checkout is not available.
    """
    start = time.time()

    # Validate patch format (basic sanity check)
    if not generated_patch.strip():
        return PatchResult(
            passed=False,
            error_type="apply_fail",
            stderr="Empty patch",
            execution_time_ms=(time.time() - start) * 1000,
        )

    # Check for valid diff format markers
    has_diff_header = any(
        line.startswith("diff ") or line.startswith("--- ")
        for line in generated_patch.splitlines()[:10]
    )

    if not has_diff_header:
        return PatchResult(
            passed=False,
            error_type="apply_fail",
            stderr="Invalid patch format: missing diff/--- headers",
            execution_time_ms=(time.time() - start) * 1000,
        )

    # Count changed lines as a proxy for test complexity
    changed_lines = _count_diff_lines(generated_patch)

    # Attempt repo-based evaluation if available
    repo_dir = REPOS_DIR / problem.repo_name
    if repo_dir.exists():
        return _evaluate_in_repo(problem, generated_patch, repo_dir, timeout)

    # Fallback: format-based scoring (for pilot runs without repos)
    # Score based on structural similarity to gold patch
    gold_files = _extract_patched_files(problem.patch)
    gen_files = _extract_patched_files(generated_patch)
    file_overlap = len(gold_files & gen_files) / max(len(gold_files), 1)

    # Heuristic cheap score based on file targeting accuracy
    cheap_pass = file_overlap > 0.5 and changed_lines > 0
    elapsed = (time.time() - start) * 1000

    return PatchResult(
        passed=cheap_pass,
        tests_total=max(len(gold_files), 1),
        tests_passed=len(gold_files & gen_files),
        stdout=f"file_overlap={file_overlap:.2f}, changed_lines={changed_lines}",
        execution_time_ms=elapsed,
    )


def _evaluate_in_repo(
    problem: SWEProblem,
    generated_patch: str,
    repo_dir: Path,
    timeout: int,
) -> PatchResult:
    """Evaluate patch by applying to actual repo checkout and running tests."""
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        try:
            # Create a temporary worktree at the base commit
            worktree_dir = work_dir / "repo"
            subprocess.run(
                ["git", "worktree", "add", str(worktree_dir), problem.base_commit],
                cwd=str(repo_dir),
                capture_output=True, text=True, timeout=30,
            )

            # Apply the generated patch
            patch_file = work_dir / "generated.patch"
            patch_file.write_text(generated_patch, encoding="utf-8")

            apply_result = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                cwd=str(worktree_dir),
                capture_output=True, text=True, timeout=10,
            )

            if apply_result.returncode != 0:
                return PatchResult(
                    passed=False,
                    error_type="apply_fail",
                    stderr=apply_result.stderr[:500],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # Actually apply the patch
            subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=str(worktree_dir),
                capture_output=True, text=True, timeout=10,
            )

            # Apply test patch too
            if problem.test_patch:
                test_patch_file = work_dir / "test.patch"
                test_patch_file.write_text(problem.test_patch, encoding="utf-8")
                subprocess.run(
                    ["git", "apply", str(test_patch_file)],
                    cwd=str(worktree_dir),
                    capture_output=True, text=True, timeout=10,
                )

            # Run tests
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-q"],
                cwd=str(worktree_dir),
                capture_output=True, text=True, timeout=timeout,
            )

            elapsed = (time.time() - start) * 1000
            passed = test_result.returncode == 0

            # Parse pytest output for test counts
            tests_total, tests_passed = _parse_pytest_output(test_result.stdout)

            return PatchResult(
                passed=passed,
                tests_total=tests_total,
                tests_passed=tests_passed,
                stdout=test_result.stdout[-500:],
                stderr=test_result.stderr[-500:],
                execution_time_ms=elapsed,
            )

        except subprocess.TimeoutExpired:
            return PatchResult(
                passed=False,
                error_type="timeout",
                stderr=f"Timeout after {timeout}s",
                execution_time_ms=timeout * 1000,
            )
        except Exception as e:
            return PatchResult(
                passed=False,
                error_type="runtime",
                stderr=str(e)[:500],
                execution_time_ms=(time.time() - start) * 1000,
            )
        finally:
            # Clean up worktree
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_dir)],
                    cwd=str(repo_dir),
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass


def _evaluate_docker(
    problem: SWEProblem,
    generated_patch: str,
    timeout: int,
) -> PatchResult:
    """Docker-based evaluation using SWE-bench harness.

    Requires: pip install swebench, Docker running.
    This is the production evaluation path for final results.
    """
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Write prediction in SWE-bench format
        prediction = {
            "instance_id": problem.instance_id,
            "model_name_or_path": "whylab_reflexion",
            "model_patch": generated_patch,
        }
        pred_file = work_dir / "predictions.json"
        pred_file.write_text(json.dumps([prediction]), encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "swebench.harness.run_evaluation",
                    "--predictions_path", str(pred_file),
                    "--swe_bench_tasks", "princeton-nlp/SWE-bench_Lite",
                    "--log_level", "WARNING",
                    "--timeout", str(timeout),
                ],
                capture_output=True, text=True,
                timeout=timeout + 60,  # extra buffer for Docker ops
            )

            elapsed = (time.time() - start) * 1000

            # Parse SWE-bench evaluation output
            passed = "RESOLVED" in result.stdout
            return PatchResult(
                passed=passed,
                stdout=result.stdout[-1000:],
                stderr=result.stderr[-500:],
                execution_time_ms=elapsed,
                tests_total=1,
                tests_passed=1 if passed else 0,
            )

        except subprocess.TimeoutExpired:
            return PatchResult(
                passed=False,
                error_type="timeout",
                stderr=f"Docker evaluation timeout after {timeout}s",
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return PatchResult(
                passed=False,
                error_type="runtime",
                stderr=str(e)[:500],
                execution_time_ms=(time.time() - start) * 1000,
            )


def _extract_patched_files(patch_str: str) -> set[str]:
    """Extract file paths from a unified diff."""
    files = set()
    for line in patch_str.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
        elif line.startswith("--- a/"):
            files.add(line[6:].strip())
    return files


def _parse_pytest_output(stdout: str) -> tuple[int, int]:
    """Parse pytest -q output for test counts.

    Example: "5 passed, 2 failed in 1.23s" → (7, 5)
    """
    import re
    total, passed = 0, 0

    match = re.search(r"(\d+) passed", stdout)
    if match:
        passed = int(match.group(1))
        total += passed

    match = re.search(r"(\d+) failed", stdout)
    if match:
        total += int(match.group(1))

    match = re.search(r"(\d+) error", stdout)
    if match:
        total += int(match.group(1))

    if total == 0:
        total = 1  # fallback

    return total, passed


if __name__ == "__main__":
    # Quick self-test
    cache = download_swebench_lite()
    problems = load_swebench_lite(subset=3)
    for p in problems:
        print(f"  {p.instance_id}: {p.repo} ({_count_diff_lines(p.patch)} changed lines)")
        print(f"    Issue: {p.problem_statement[:100]}...")
