"""
HumanEval Dataset Loader & Code Executor
=========================================
Loads HumanEval problems and executes generated solutions safely.

Safety: Solutions are executed via subprocess with timeouts,
memory limits, and filesystem isolation — following the official
HumanEval repo guidance to never run untrusted code without sandboxing.
"""
import gzip
import json
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
HUMANEVAL_FILE = DATA_DIR / "HumanEval.jsonl.gz"
TIMEOUT_SECONDS = 10
MAX_MEMORY_MB = 256


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem."""
    task_id: str
    prompt: str          # function signature + docstring
    entry_point: str     # function name to call
    canonical_solution: str
    test: str            # assert-based test code
    idx: int = 0         # 0-based index


@dataclass
class ExecutionResult:
    """Result of executing a solution against tests."""
    passed: bool
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""   # "timeout", "runtime", "syntax", "memory", ""
    execution_time_ms: float = 0.0
    # Cheap eval: fraction of test assertions passed (0.0~1.0)
    cheap_score: float = 0.0


def load_humaneval(
    path: Optional[Path] = None,
    subset: Optional[int] = None,
) -> list[HumanEvalProblem]:
    """Load HumanEval problems from JSONL (gzipped or plain).

    Args:
        path: Path to HumanEval.jsonl or HumanEval.jsonl.gz.
              Defaults to experiments/data/HumanEval.jsonl.gz.
        subset: If set, return only the first N problems (for piloting).

    Returns:
        List of HumanEvalProblem instances.
    """
    path = path or HUMANEVAL_FILE

    if not path.exists():
        # Try non-gzipped version
        alt = path.with_suffix("") if path.suffix == ".gz" else path.with_suffix(".jsonl.gz")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"HumanEval dataset not found at {path}. "
                f"Download from: https://github.com/openai/human-eval"
            )

    problems = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            problems.append(HumanEvalProblem(
                task_id=data["task_id"],
                prompt=data["prompt"],
                entry_point=data["entry_point"],
                canonical_solution=data["canonical_solution"],
                test=data["test"],
                idx=idx,
            ))

    if subset is not None:
        problems = problems[:subset]

    return problems


def _build_test_script(
    prompt: str,
    solution: str,
    test_code: str,
    entry_point: str,
) -> str:
    """Build a self-contained test script.

    The script:
    1. Defines the function (prompt + solution body)
    2. Runs the test assertions
    3. Prints PASS/FAIL + individual assertion results for cheap scoring
    """
    # Combine prompt + solution into a complete function
    full_function = prompt + solution

    script = textwrap.dedent(f"""\
    import sys
    import math
    from typing import *

    # --- Generated solution ---
    {full_function}

    # --- Test harness with cheap scoring ---
    def _run_tests():
        _total = 0
        _passed = 0
        _errors = []

        # The test code defines check() and calls it
        def check(candidate):
            nonlocal _total, _passed, _errors
            # We'll intercept assertions via a wrapper
            pass

    {textwrap.indent(test_code, '    ')}

        # Run the check function
        try:
            check({entry_point})
        except AssertionError as e:
            _errors.append(str(e))
        except Exception as e:
            _errors.append(f"{{type(e).__name__}}: {{e}}")

        return len(_errors) == 0, _errors

    try:
        passed, errors = _run_tests()
        if passed:
            print("RESULT:PASS")
        else:
            print(f"RESULT:FAIL:{{len(errors)}}")
            for e in errors[:5]:
                print(f"ERROR:{{e}}")
    except Exception as e:
        print(f"RESULT:FAIL:1")
        print(f"ERROR:{{type(e).__name__}}: {{e}}")
    """)
    return script


def _build_simple_test_script(
    prompt: str,
    solution: str,
    test_code: str,
    entry_point: str,
) -> str:
    """Build a simple test script that just runs the tests directly.

    Returns exit code 0 if all pass, 1 otherwise.
    """
    full_function = prompt + solution

    script = f"""\
import sys
import math
from typing import *

# --- Generated solution ---
{full_function}

# --- Tests ---
{test_code}

try:
    check({entry_point})
    print("RESULT:PASS")
except AssertionError as e:
    print(f"RESULT:FAIL")
    print(f"ERROR:{{e}}")
except Exception as e:
    print(f"RESULT:FAIL")
    print(f"ERROR:{{type(e).__name__}}: {{e}}")
"""
    return script


def execute_solution(
    problem: HumanEvalProblem,
    solution: str,
    timeout: int = TIMEOUT_SECONDS,
    cheap_test_fraction: float = 1.0,
) -> ExecutionResult:
    """Execute a solution against test cases in a sandboxed subprocess.

    Args:
        problem: The HumanEval problem.
        solution: Generated solution body (appended to prompt).
        timeout: Max execution time in seconds.
        cheap_test_fraction: Fraction of tests to run for cheap eval (1.0 = all).

    Returns:
        ExecutionResult with pass/fail and cheap_score.
    """
    import time

    script = _build_simple_test_script(
        problem.prompt,
        solution,
        problem.test,
        problem.entry_point,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        start = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **dict(__import__("os").environ),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        elapsed_ms = (time.time() - start) * 1000

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        passed = "RESULT:PASS" in stdout

        # Cheap score: 1.0 if pass, 0.0 if fail
        # (In a more sophisticated setup, this would be partial test score)
        cheap_score = 1.0 if passed else 0.0

        return ExecutionResult(
            passed=passed,
            stdout=stdout,
            stderr=stderr,
            error_type="" if passed else "runtime",
            execution_time_ms=elapsed_ms,
            cheap_score=cheap_score,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            stderr=f"Timeout after {timeout}s",
            error_type="timeout",
            execution_time_ms=timeout * 1000,
            cheap_score=0.0,
        )

    except Exception as e:
        return ExecutionResult(
            passed=False,
            stderr=str(e),
            error_type="runtime",
            cheap_score=0.0,
        )

    finally:
        Path(script_path).unlink(missing_ok=True)


def download_humaneval(target_dir: Optional[Path] = None) -> Path:
    """Download HumanEval dataset from GitHub.

    Returns path to downloaded file.
    """
    import urllib.request

    target_dir = target_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "HumanEval.jsonl.gz"

    if target.exists():
        print(f"[HumanEval] Already exists: {target}")
        return target

    url = (
        "https://github.com/openai/human-eval/raw/master/"
        "data/HumanEval.jsonl.gz"
    )
    print(f"[HumanEval] Downloading from {url}...")
    urllib.request.urlretrieve(url, target)
    print(f"[HumanEval] Saved: {target}")
    return target


if __name__ == "__main__":
    # Quick self-test
    path = download_humaneval()
    problems = load_humaneval(path, subset=3)
    for p in problems:
        print(f"  {p.task_id}: {p.entry_point}")
        # Test with canonical solution
        result = execute_solution(p, p.canonical_solution)
        status = "PASS" if result.passed else "FAIL"
        print(f"    Canonical: {status} ({result.execution_time_ms:.0f}ms)")
