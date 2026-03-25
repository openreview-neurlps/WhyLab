"""
E4: Agent Benchmark — HumanEval + Reflexion + WhyLab Audit
============================================================
Tests whether the WhyLab audit layer (C1-C3) improves stability
and reduces regressions in a Reflexion-style self-improvement loop
on the HumanEval code generation benchmark.

Protocol:
- Benchmark: HumanEval (164 functional correctness problems)
- Agent: Reflexion (solve→test→reflect→retry, max 5 attempts)
- Audit: 5 ablation configs (none / C1 / C2 / C3 / full)
- Seeds: 20 (dev) or 40 (final)
- LLM: Single model (Gemini Flash), temperature=0.0

Metrics:
  1. Final pass@1 — fraction of problems solved at last attempt
  2. Regression count — pass→fail transitions across attempts
  3. Oscillation index — total sign changes / (attempts - 1)
  4. Update acceptance rate — fraction of updates passing audit

Output:
  experiments/results/e4_metrics.csv    — per-seed × ablation × problem
  experiments/results/e4_summary.csv    — aggregated by ablation
"""
import argparse
import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CFG_FILE = ROOT / "config.yaml"

# Load .env from project root (for GEMINI_API_KEY)
load_dotenv(ROOT.parent / ".env")


def load_config():
    """Load experiment configuration."""
    full = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8"))
    return full["experiment"], full["e4"]


def run_e4(
    seeds: int = None,
    problems_subset: int = None,
    mode: str = None,
    split: str = None,
    holdout_exclude: str = None,
):
    """Run the E4 agent benchmark experiment.

    Args:
        seeds: Override number of seeds (default: from config).
        problems_subset: Override problem count (default: from config).
        mode: Override LLM cache mode (default: from config).
        split: Named split from config.splits (pilot/main/full).
        holdout_exclude: Exclude problems used in this split (e.g. 'pilot').
    """
    # Lazy imports to avoid circular dependencies
    from experiments.humaneval_loader import load_humaneval, download_humaneval
    from experiments.llm_client import CachedLLMClient
    from experiments.reflexion_loop import run_reflexion_episode
    from experiments.audit_layer import AgentAuditLayer

    exp_cfg, e4_cfg = load_config()

    base_seed = exp_cfg["rng_base_seed"]
    max_attempts = e4_cfg["max_attempts"]
    llm_cfg = e4_cfg["llm"]
    audit_cfg = e4_cfg["audit"]
    ablations = e4_cfg["ablations"]

    # Resolve split-based seeds/problems
    splits_cfg = e4_cfg.get("splits", {})
    if split and split in splits_cfg:
        s = splits_cfg[split]
        n_seeds = seeds or s["n_seeds"]
        subset = problems_subset or s["n_problems"]
    else:
        n_seeds = seeds or exp_cfg["seeds"]
        subset = problems_subset or e4_cfg.get("problems_subset")

    # LLM cache mode
    cache_mode = mode or llm_cfg.get("cache_mode", "hybrid")

    # Load HumanEval
    print("[E4] Loading HumanEval dataset...")
    download_humaneval()
    all_problems = load_humaneval(subset=None)  # load all first

    # Holdout exclusion: skip problems used in exclude split
    exclude_n = 0
    if holdout_exclude and holdout_exclude in splits_cfg:
        exclude_n = splits_cfg[holdout_exclude]["n_problems"]

    if exclude_n > 0:
        all_problems = all_problems[exclude_n:]  # skip first N (used by pilot)

    # Apply subset limit
    if subset is not None:
        problems = all_problems[:subset]
    else:
        problems = all_problems
    print(f"[E4] Loaded {len(problems)} problems")

    # Initialize LLM client
    llm = CachedLLMClient(
        model=llm_cfg["model"],
        cache_dir=ROOT / "cache",
        mode=cache_mode,
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        prompt_version=llm_cfg["prompt_version"],
    )

    # Cost guardrail: max tokens per seed (configurable)
    max_tokens_per_seed = e4_cfg.get("max_tokens_per_seed", 500_000)

    all_rows = []
    total_start = time.time()

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        seed_token_start = llm.stats["total_tokens"]

        for abl in ablations:
            abl_name = abl["name"]

            # Stress ablations can override max_attempts
            abl_max_attempts = abl.get("max_attempts", max_attempts)

            # Build audit config: global defaults first, then ablation overrides
            audit_config = {
                **{k: v for k, v in audit_cfg.items()},  # global defaults
                "c1": abl.get("c1", False),
                "c2": abl.get("c2", False),
                "c3": abl.get("c3", False),
            }
            # Per-ablation threshold overrides (e.g. C2_default vs C2_calibrated)
            for override_key in ("c2_e_thresh", "c2_rv_thresh", "c3_epsilon_floor", "c3_ceiling"):
                if override_key in abl:
                    audit_config[override_key] = abl[override_key]
            audit = AgentAuditLayer(audit_config)

            # Determine if audit layer is active
            use_audit = abl.get("c1", False) or abl.get("c2", False) or abl.get("c3", False)

            for problem in problems:
                episode = run_reflexion_episode(
                    problem=problem,
                    llm=llm,
                    max_attempts=abl_max_attempts,
                    audit=audit if use_audit else None,
                    seed=seed,
                )

                all_rows.append({
                    "seed": seed,
                    "ablation": abl_name,
                    "task_id": episode.task_id,
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

            print(f"  seed {seed_idx + 1}/{n_seeds}, ablation={abl_name}: "
                  f"{sum(r['final_passed'] for r in all_rows if r['seed'] == seed and r['ablation'] == abl_name)}"
                  f"/{len(problems)} passed")

        # Seed-level cost check
        seed_tokens = llm.stats["total_tokens"] - seed_token_start
        print(f"  [Cost] seed {seed_idx + 1} tokens: {seed_tokens:,} "
              f"(cumulative: {llm.stats['total_tokens']:,})")
        if seed_tokens > max_tokens_per_seed:
            print(f"  [WARN] Seed {seed_idx + 1} exceeded token budget "
                  f"({seed_tokens:,} > {max_tokens_per_seed:,})")

    elapsed = time.time() - total_start
    print(f"\n[E4] Total time: {elapsed:.1f}s")


    # Save results
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "e4_metrics.csv", index=False)
    print(f"[E4] Saved: {len(df)} rows → e4_metrics.csv")

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

    summary.to_csv(out / "e4_summary.csv", index=False)
    print(f"[E4] Summary saved: {len(summary)} rows → e4_summary.csv")

    print("\n=== E4 Summary ===")
    print(summary.to_string(index=False))

    # Print LLM stats
    stats = llm.get_stats()
    print(f"\n[LLM] Calls: {stats['calls']}, "
          f"Cache hits: {stats['cache_hits']}, "
          f"API calls: {stats['api_calls']}, "
          f"Total tokens: {stats['total_tokens']}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E4: Agent Benchmark")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of seeds")
    parser.add_argument("--problems", type=int, default=None,
                        help="Subset of problems (for piloting)")
    parser.add_argument("--mode", choices=["online", "replay", "hybrid"],
                        default=None, help="LLM cache mode")
    parser.add_argument("--split", choices=["pilot", "main", "full"],
                        default=None, help="Named split from config.splits")
    parser.add_argument("--holdout_exclude", type=str, default=None,
                        help="Exclude problems from this split (e.g. pilot)")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot: 2 seeds, 5 problems")
    args = parser.parse_args()

    if args.pilot:
        args.split = args.split or "pilot"

    run_e4(
        seeds=args.seeds,
        problems_subset=args.problems,
        mode=args.mode,
        split=args.split,
        holdout_exclude=args.holdout_exclude,
    )
