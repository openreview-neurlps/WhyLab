# -*- coding: utf-8 -*-
"""
E5-Multi: Multi-LLM Cross-Family Experiment
=============================================
Runs the Reflexion + WhyLab audit loop across multiple LLM families
to validate universality of oscillation patterns.

Models: GPT-5 mini, Claude Sonnet 4.6, Gemini 2.0 Flash (+ flagship subset)
Protocol: 100 problems × 3 seeds × 3 ablations per model

Usage:
    python -m experiments.e5_multi_llm                     # full run
    python -m experiments.e5_multi_llm --pilot              # 3 problems × 1 seed
    python -m experiments.e5_multi_llm --models gpt-5-mini  # single model
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CFG_FILE = ROOT / "config.yaml"

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass


# ── Config ──────────────────────────────────────────────────────────────
DEFAULT_MODELS = [
    {
        "name": "gemini-2.0-flash",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    {
        "name": "gpt-5.4-nano",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    {
        "name": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
]

# Reduced ablation set for multi-LLM (cost control)
MULTI_ABLATIONS = [
    {"name": "none", "c1": False, "c2": False, "c3": False},
    {"name": "C2_calibrated", "c1": False, "c2": True, "c3": False,
     "c2_e_thresh": 1.5, "c2_rv_thresh": 0.05},
    {"name": "full_calibrated", "c1": True, "c2": True, "c3": True,
     "c2_e_thresh": 1.5, "c2_rv_thresh": 0.05},
]


def run_multi_llm(
    models: list[dict] = None,
    n_problems: int = 100,
    n_seeds: int = 3,
    mode: str = "online",
    eval_mode: str = "lightweight",
    checkpoint: bool = True,
):
    """Run the multi-LLM experiment.

    Args:
        models: List of model configs [{name, temperature, max_tokens}].
        n_problems: Number of SWE-bench problems per model.
        n_seeds: Number of random seeds per model.
        mode: LLM cache mode (online/replay/hybrid).
        eval_mode: "lightweight" or "docker".
        checkpoint: If True, save after each problem for resume.
    """
    # Lazy imports
    from experiments.swebench_loader import load_swebench_lite, download_swebench_lite
    from experiments.llm_client import CachedLLMClient
    from experiments.swebench_reflexion import run_swe_reflexion_episode
    from experiments.audit_layer import AgentAuditLayer

    if models is None:
        models = DEFAULT_MODELS

    # Load config for audit defaults
    full_cfg = yaml.safe_load(CFG_FILE.read_text(encoding="utf-8"))
    exp_cfg = full_cfg["experiment"]
    e5_cfg = full_cfg["e5"]
    audit_cfg = e5_cfg["audit"]
    base_seed = exp_cfg["rng_base_seed"]
    max_attempts = e5_cfg["max_attempts"]

    # Load SWE-bench
    print("[Multi-LLM] Loading SWE-bench Lite...")
    download_swebench_lite()
    all_problems = load_swebench_lite(subset=None)
    problems = all_problems[:n_problems]
    print(f"[Multi-LLM] {len(problems)} problems × {n_seeds} seeds × "
          f"{len(MULTI_ABLATIONS)} ablations × {len(models)} models")

    out_dir = ROOT / "results" / "multi_llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_start = time.time()

    for model_cfg in models:
        model_name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"[Model] {model_name}")
        print(f"{'='*60}")

        # Check for checkpoint
        model_csv = out_dir / f"e5_{model_name.replace('/', '_')}_metrics.csv"
        done_keys = set()
        model_rows = []
        if checkpoint and model_csv.exists():
            try:
                existing = pd.read_csv(model_csv)
                if len(existing) > 0:
                    for _, row in existing.iterrows():
                        done_keys.add(f"{row['seed']}_{row['ablation']}_{row['instance_id']}")
                    print(f"  [Checkpoint] {len(done_keys)} episodes already done, resuming...")
                    model_rows = existing.to_dict("records")
            except Exception:
                pass

        # Create LLM client
        llm = CachedLLMClient(
            model=model_name,
            cache_dir=ROOT / "cache",
            mode=mode,
            temperature=model_cfg.get("temperature", 0.7),
            max_tokens=model_cfg.get("max_tokens", 4096),
            prompt_version="v1",
        )

        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx

            for abl in MULTI_ABLATIONS:
                abl_name = abl["name"]

                # Build audit config
                ac = {**audit_cfg, **{k: v for k, v in abl.items() if k != "name"}}
                audit = AgentAuditLayer(ac)
                use_audit = abl.get("c1", False) or abl.get("c2", False) or abl.get("c3", False)

                for prob_idx, problem in enumerate(problems):
                    key = f"{seed}_{abl_name}_{problem.instance_id}"
                    if key in done_keys:
                        continue

                    try:
                        episode = run_swe_reflexion_episode(
                            problem=problem,
                            llm=llm,
                            max_attempts=max_attempts,
                            audit=audit if use_audit else None,
                            seed=seed,
                            eval_mode=eval_mode,
                            disable_reflection=False,
                        )

                        row = {
                            "model": model_name,
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
                        }
                        model_rows.append(row)
                        done_keys.add(key)

                        # Checkpoint: save after each problem
                        if checkpoint and (prob_idx + 1) % 5 == 0:
                            pd.DataFrame(model_rows).to_csv(model_csv, index=False)

                    except Exception as e:
                        print(f"  [ERROR] {problem.instance_id}: {e}")
                        continue

                n_passed = sum(
                    r["final_passed"]
                    for r in model_rows
                    if r["seed"] == seed and r["ablation"] == abl_name
                )
                print(f"  seed={seed_idx+1}/{n_seeds} abl={abl_name}: "
                      f"{n_passed}/{len(problems)} passed")

        # Final save for this model
        model_df = pd.DataFrame(model_rows)
        model_df.to_csv(model_csv, index=False)
        print(f"  Saved: {len(model_df)} rows -> {model_csv.name}")

        stats = llm.get_stats()
        print(f"  [LLM] calls={stats['calls']}, cache_hits={stats['cache_hits']}, "
              f"api_calls={stats['api_calls']}, tokens={stats['total_tokens']:,}")

        all_results.extend(model_rows)

    # Combined summary
    elapsed = time.time() - total_start
    print(f"\n[Multi-LLM] Total time: {elapsed:.1f}s")

    if all_results:
        combined = pd.DataFrame(all_results)
        combined.to_csv(out_dir / "e5_multi_metrics.csv", index=False)

        summary = combined.groupby(["model", "ablation"]).agg(
            pass_rate=("final_passed", "mean"),
            safe_pass_rate=("safe_pass", "mean"),
            mean_regressions=("regression_count", "mean"),
            mean_oscillation=("oscillation_index", "mean"),
        ).round(4).reset_index()

        summary.to_csv(out_dir / "e5_multi_summary.csv", index=False)
        print("\n=== Multi-LLM Summary ===")
        print(summary.to_string(index=False))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E5-Multi: Multi-LLM Experiment")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to test (default: all 3)")
    parser.add_argument("--problems", type=int, default=100,
                        help="Number of problems per model")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per model")
    parser.add_argument("--mode", choices=["online", "replay", "hybrid"],
                        default="online", help="LLM cache mode")
    parser.add_argument("--eval-mode", choices=["lightweight", "docker"],
                        default="lightweight", help="SWE-bench evaluation mode")
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot: 3 problems, 1 seed")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpointing")
    args = parser.parse_args()

    if args.pilot:
        args.problems = 3
        args.seeds = 1

    # Filter models if specified
    selected_models = None
    if args.models:
        selected_models = [m for m in DEFAULT_MODELS if m["name"] in args.models]
        if not selected_models:
            # Try partial match
            selected_models = [
                m for m in DEFAULT_MODELS
                if any(a.lower() in m["name"].lower() for a in args.models)
            ]
        if not selected_models:
            # Ad-hoc: create model configs for unrecognized names
            selected_models = [
                {"name": name, "temperature": 0.7, "max_tokens": 4096}
                for name in args.models
            ]

    run_multi_llm(
        models=selected_models,
        n_problems=args.problems,
        n_seeds=args.seeds,
        mode=args.mode,
        eval_mode=args.eval_mode,
        checkpoint=not args.no_checkpoint,
    )
