"""
Evaluate all candidates from a learned policy file multiple times.

Creates a folder structure like:
  evaluations/trial3_.../
    summary.json                    # Overview of all candidates
    cand0_id33000/
      test_eval_run1.json
      test_eval_run2.json
      test_eval_run3.json
    cand1_id44000/
      test_eval_run1.json
      test_eval_run2.json
      test_eval_run3.json
    ...

Usage:
    python evaluate_all_candidates.py \
        --prompt-path learned_policies/8traj_12iter_10cand/rpe_ega_result_trial3_8traj_12iter_10cand.json \
        --n-runs 3 \
        --max-concurrency 10 \
        --output-dir evaluations/trial3_8traj_12iter_10cand
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import RunConfig


def load_candidates(prompt_path: str) -> List[Dict]:
    """Load all candidates from a learned policy file."""
    with open(prompt_path, 'r') as f:
        data = json.load(f)

    if "final_population" in data:
        return data["final_population"]
    else:
        raise ValueError(f"No final_population found in {prompt_path}")


def load_test_task_ids(test_path: str) -> List[int]:
    """Load test task IDs."""
    with open(test_path, 'r') as f:
        data = json.load(f)
    return [t["task_id"] for t in data]


def evaluate_on_task(
    wiki: str,
    task_id: int,
    config: RunConfig,
) -> Dict[str, Any]:
    """Run agent with given wiki on a specific test task."""
    try:
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            task_index=task_id,
        )

        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )

        result = agent.solve(env=env, task_index=task_id)

        # Extract actions
        actions_taken = []
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    actions_taken.append(tc["function"]["name"])

        return {
            "task_id": task_id,
            "success": result.reward > 0,
            "reward": result.reward,
            "actions": actions_taken,
            "n_messages": len(result.messages),
            "error": None,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "reward": 0.0,
            "actions": [],
            "n_messages": 0,
            "error": str(e),
        }


def evaluate_candidate_once(
    wiki: str,
    task_ids: List[int],
    config: RunConfig,
    max_concurrency: int = 1,
    run_num: int = 1,
    total_runs: int = 1,
    cand_idx: int = 0,
) -> Dict[str, Any]:
    """Evaluate a candidate on all test tasks once."""

    completed = [0]
    total = len(task_ids)

    def _run(tid):
        result = evaluate_on_task(wiki, tid, config)
        completed[0] += 1
        status = "SUCCESS" if result["success"] else "FAIL"
        print(f"    [{completed[0]}/{total}] Task {tid}: {status}")
        return result

    if max_concurrency <= 1:
        results = [_run(tid) for tid in task_ids]
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            results = list(executor.map(_run, task_ids))

    successes = sum(1 for r in results if r["success"])
    success_rate = successes / len(task_ids)

    return {
        "n_tasks": len(task_ids),
        "n_successes": successes,
        "success_rate": success_rate,
        "mean_reward": success_rate,
        "task_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all candidates multiple times")
    parser.add_argument(
        "--prompt-path",
        type=str,
        required=True,
        help="Path to learned policy JSON with final_population",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json",
        help="Path to test trajectories JSON",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of times to evaluate each candidate",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent task evaluations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for agent",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--candidate-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific candidate indices to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Load candidates
    print("=" * 70)
    print("MULTI-CANDIDATE EVALUATION")
    print("=" * 70)
    print(f"Loading candidates from: {args.prompt_path}")

    candidates = load_candidates(args.prompt_path)
    print(f"Found {len(candidates)} candidates")

    # Filter candidates if specified
    if args.candidate_indices:
        candidates_to_eval = [(i, candidates[i]) for i in args.candidate_indices]
    else:
        candidates_to_eval = list(enumerate(candidates))

    print(f"Will evaluate {len(candidates_to_eval)} candidates, {args.n_runs} runs each")

    # Load test task IDs
    task_ids = load_test_task_ids(args.test_path)
    print(f"Test set: {len(task_ids)} tasks")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = RunConfig(
        env="retail",
        model=args.model,
        model_provider="openai",
        user_model="gpt-4o",
        user_model_provider="openai",
        agent_strategy="tool-calling",
        task_split="test",
        user_strategy="llm",
        temperature=0.0,
        seed=42,
        start_index=0,
        end_index=-1,
        log_dir="./eval_logs",
        max_concurrency=args.max_concurrency,
        shuffle=False,
        num_trials=1,
        task_ids=[],
    )

    # Results summary
    all_results = []

    for idx, candidate in candidates_to_eval:
        cand_id = candidate.get("id", idx)
        cand_gen = candidate.get("generation", "?")
        wiki = candidate["wiki"]

        # Create candidate folder
        cand_folder = output_dir / f"cand{idx}_id{cand_id}"
        cand_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CANDIDATE {idx} (id={cand_id}, gen={cand_gen})")
        print(f"Saving to: {cand_folder}")
        print(f"{'='*60}")

        run_scores = []

        for run in range(args.n_runs):
            print(f"\n  Run {run+1}/{args.n_runs}:")

            result = evaluate_candidate_once(
                wiki=wiki,
                task_ids=task_ids,
                config=config,
                max_concurrency=args.max_concurrency,
                run_num=run+1,
                total_runs=args.n_runs,
                cand_idx=idx,
            )

            run_scores.append(result["success_rate"])

            # Save individual run result
            run_output = {
                "source": {
                    "type": "rpe_output",
                    "path": args.prompt_path,
                    "candidate_index": idx,
                    "candidate_id": cand_id,
                    "generation": cand_gen,
                    "train_score": candidate.get("raw_score"),
                    "lcb_score": candidate.get("lcb_score"),
                },
                "config": {
                    "test_path": args.test_path,
                    "model": args.model,
                    "run_number": run + 1,
                },
                "results": {
                    "n_tasks": result["n_tasks"],
                    "n_successes": result["n_successes"],
                    "success_rate": result["success_rate"],
                    "mean_reward": result["mean_reward"],
                },
                "task_results": result["task_results"],
                "wiki": wiki,
                "timestamp": datetime.now().isoformat(),
            }

            run_file = cand_folder / f"test_eval_run{run+1}.json"
            with open(run_file, 'w') as f:
                json.dump(run_output, f, indent=2)

            print(f"  → {result['success_rate']:.1%} ({result['n_successes']}/{result['n_tasks']}) saved to {run_file.name}")

        avg_score = sum(run_scores) / len(run_scores)
        min_score = min(run_scores)
        max_score = max(run_scores)

        print(f"\n  Summary: avg={avg_score:.1%}, min={min_score:.1%}, max={max_score:.1%}")

        all_results.append({
            "candidate_index": idx,
            "candidate_id": cand_id,
            "generation": cand_gen,
            "train_score": candidate.get("raw_score"),
            "lcb_score": candidate.get("lcb_score"),
            "run_scores": run_scores,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "folder": str(cand_folder),
        })

    # Save summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Idx':<4} {'ID':<8} {'Gen':<4} {'Train':<8} {'LCB':<8} {'Test Avg':<10} {'Min':<8} {'Max':<8}")
    print("-" * 70)

    for r in all_results:
        train = f"{r['train_score']:.1%}" if r['train_score'] else "?"
        lcb = f"{r['lcb_score']:.1%}" if r['lcb_score'] else "?"
        print(f"{r['candidate_index']:<4} {r['candidate_id']:<8} {r['generation']:<4} {train:<8} {lcb:<8} {r['avg_score']:.1%}      {r['min_score']:.1%}    {r['max_score']:.1%}")

    # Save summary file
    summary = {
        "config": {
            "prompt_path": args.prompt_path,
            "test_path": args.test_path,
            "n_runs": args.n_runs,
            "model": args.model,
            "n_tasks": len(task_ids),
            "n_candidates": len(candidates_to_eval),
        },
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
