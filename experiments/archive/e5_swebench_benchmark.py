"""
E5: SWE-bench Lite Benchmark — Reflexion + WhyLab Audit
========================================================
Tests whether the WhyLab audit layer (C1-C3) reduces oscillation
and regression in a Reflexion-style self-improvement loop on
SWE-bench Lite (real-world software engineering tasks).

This is the primary validation experiment for the paper.

Protocol:
- Benchmark: SWE-bench Lite (300 real-world GitHub issues)
- Agent: Reflexion (solve→test→reflect→retry, max 7 attempts)
- Audit: 7 ablation configs (none / simple_retry / C1 / C2 / C3 / full)
- Seeds: 2 (pilot) or 5 (final)
- LLM: Gemini Flash, temperature=0.7

Metrics:
  1. Final pass@1 — fraction of problems resolved at last attempt
  2. Regression count — pass→fail transitions across attempts
  3. Oscillation index — total sign changes / (attempts - 1)
  4. Update acceptance rate — fraction of updates passing audit

Output:
  experiments/results/e5_metrics.csv    — per-seed × ablation × problem
  experiments/results/e5_summary.csv    — aggregated by ablation
"""
import argparse
import time

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG_FILE = ROOT / "config.yaml"

# Load .env from project root (safe import)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on env vars


def load_config():
    """Load experiment configuration."""
    full = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8"))
    return full["experiment"], full["e5"]


def run_e5(
    seeds: int = None,
    problems_subset: int = None,
    mode: str = None,
    split: str = None,
    eval_mode: str = None,
):
    """Run the E5 SWE-bench benchmark experiment.

    Args:
        seeds: Override number of seeds (default: from config).
        problems_subset: Override problem count (default: from config).
        mode: Override LLM cache mode (default: from config).
        split: Named split from config.splits (pilot/main/full).
        eval_mode: "lightweight" or "docker" (default: from config).
    """
    # Lazy imports
    from experiments.swebench_loader import load_swebench_lite, download_swebench_lite
    from experiments.llm_client import CachedLLMClient
    from experiments.swebench_reflexion import run_swe_reflexion_episode
    from experiments.audit_layer import AgentAuditLayer

    exp_cfg, e5_cfg = load_config()

    base_seed = exp_cfg["rng_base_seed"]
    max_attempts = e5_cfg["max_attempts"]
    llm_cfg = e5_cfg["llm"]
    audit_cfg = e5_cfg["audit"]
    ablations = e5_cfg["ablations"]

    # Resolve split-based seeds/problems
    splits_cfg = e5_cfg.get("splits", {})
    if split and split in splits_cfg:
        s = splits_cfg[split]
        n_seeds = seeds or s["n_seeds"]
        subset = problems_subset or s["n_problems"]
    else:
        n_seeds = seeds or exp_cfg["seeds"]
        subset = problems_subset or e5_cfg.get("problems_subset")

    # LLM cache mode
    cache_mode = mode or llm_cfg.get("cache_mode", "hybrid")

    # Eval mode
    eval_m = eval_mode or e5_cfg.get("eval_mode", "lightweight")

    # Load SWE-bench Lite
    print("[E5] Loading SWE-bench Lite dataset...")
    download_swebench_lite()
    all_problems = load_swebench_lite(subset=None)

    # Apply subset limit
    if subset is not None:
        problems = all_problems[:subset]
    else:
        problems = all_problems
    print(f"[E5] Loaded {len(problems)} problems")

    # Initialize LLM client
    llm = CachedLLMClient(
        model=llm_cfg["model"],
        cache_dir=ROOT / "cache",
        mode=cache_mode,
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        prompt_version=llm_cfg.get("prompt_version", "v1"),
    )

    # Cost guardrail
    max_tokens_per_seed = e5_cfg.get("max_tokens_per_seed", 1_000_000)

    all_rows = []
    total_start = time.time()

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        seed_token_start = llm.stats["total_tokens"]

        for abl in ablations:
            abl_name = abl["name"]
            abl_max_attempts = abl.get("max_attempts", max_attempts)

            # Build audit config
            audit_config = {
                **{k: v for k, v in audit_cfg.items()},
                "c1": abl.get("c1", False),
                "c2": abl.get("c2", False),
                "c3": abl.get("c3", False),
            }
            for override_key in ("c2_e_thresh", "c2_rv_thresh",
                                 "c3_epsilon_floor", "c3_ceiling"):
                if override_key in abl:
                    audit_config[override_key] = abl[override_key]
            audit = AgentAuditLayer(audit_config)

            use_audit = (abl.get("c1", False) or
                         abl.get("c2", False) or
                         abl.get("c3", False))

            for problem in problems:
                # Reset audit state for each problem to prevent cross-problem leakage
                audit = AgentAuditLayer(audit_config)

                # Determine if this is a simple_retry ablation (no reflection)
                is_simple_retry = abl_name == "simple_retry"

                episode = run_swe_reflexion_episode(
                    problem=problem,
                    llm=llm,
                    max_attempts=abl_max_attempts,
                    audit=audit if use_audit else None,
                    seed=seed,
                    eval_mode=eval_m,
                    disable_reflection=is_simple_retry,
                )

                all_rows.append({
                    "seed": seed,
                    "ablation": abl_name,
                    "instance_id": episode.instance_id,
                    "final_passed": int(episode.final_passed),
                    "safe_pass": int(episode.safe_pass),
                    "first_pass_attempt": episode.first_pass_attempt,
                    "total_attempts": episode.total_attempts,
                    "regression_count": episode.regression_count,
                    "oscillation_count": episode.oscillation_count,
                    "oscillation_index": round(episode.oscillation_index, 4),
                    "updates_accepted": episode.updates_accepted,
                    "updates_rejected": episode.updates_rejected,
                })

            n_passed = sum(
                r["final_passed"]
                for r in all_rows
                if r["seed"] == seed and r["ablation"] == abl_name
            )
            print(f"  seed {seed_idx + 1}/{n_seeds}, ablation={abl_name}: "
                  f"{n_passed}/{len(problems)} passed")

        # Seed-level cost check
        seed_tokens = llm.stats["total_tokens"] - seed_token_start
        print(f"  [Cost] seed {seed_idx + 1} tokens: {seed_tokens:,} "
              f"(cumulative: {llm.stats['total_tokens']:,})")
        if seed_tokens > max_tokens_per_seed:
            print(f"  [WARN] Seed {seed_idx + 1} exceeded token budget")

    elapsed = time.time() - total_start
    print(f"\n[E5] Total time: {elapsed:.1f}s")

    # Save results
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "e5_metrics.csv", index=False)
    print(f"[E5] Saved: {len(df)} rows → e5_metrics.csv")

    # Generate summary
    summary = df.groupby("ablation").agg(
        pass_rate=("final_passed", "mean"),
        safe_pass_rate=("safe_pass", "mean"),
        mean_first_pass=("first_pass_attempt", "mean"),
        mean_attempts=("total_attempts", "mean"),
        mean_regressions=("regression_count", "mean"),
        mean_oscillation=("oscillation_index", "mean"),
        total_accepted=("updates_accepted", "sum"),
        total_rejected=("updates_rejected", "sum"),
    ).round(4).reset_index()

    # Add acceptance rate
    summary["acceptance_rate"] = (
        summary["total_accepted"] /
        (summary["total_accepted"] + summary["total_rejected"]).replace(0, 1)
    ).round(4)

    summary.to_csv(out / "e5_summary.csv", index=False)
    print(f"[E5] Summary saved: {len(summary)} rows → e5_summary.csv")

    print("\n=== E5 Summary ===")
    print(summary.to_string(index=False))

    # Print LLM stats
    stats = llm.get_stats()
    print(f"\n[LLM] Calls: {stats['calls']}, "
          f"Cache hits: {stats['cache_hits']}, "
          f"API calls: {stats['api_calls']}, "
          f"Total tokens: {stats['total_tokens']}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E5: SWE-bench Benchmark")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of seeds")
    parser.add_argument("--problems", type=int, default=None,
                        help="Subset of problems")
    parser.add_argument("--mode", choices=["online", "replay", "hybrid"],
                        default=None, help="LLM cache mode")
    parser.add_argument("--split", choices=["pilot", "main", "full"],
                        default=None, help="Named split from config")
    parser.add_argument("--eval-mode", choices=["lightweight", "docker"],
                        default=None, help="Evaluation mode")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot: 2 seeds, 30 problems")
    args = parser.parse_args()

    if args.pilot:
        args.split = args.split or "pilot"

    run_e5(
        seeds=args.seeds,
        problems_subset=args.problems,
        mode=args.mode,
        split=args.split,
        eval_mode=args.eval_mode,
    )
