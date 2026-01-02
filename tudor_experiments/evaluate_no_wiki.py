"""
Evaluate with no wiki (empty string) to establish baseline.

Usage:
    python evaluate_no_wiki.py --n-runs 10 --max-concurrency 10 --output-dir evaluations/no_wiki_baseline
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


def evaluate_once(
    wiki: str,
    task_ids: List[int],
    config: RunConfig,
    max_concurrency: int = 1,
    run_num: int = 1,
) -> Dict[str, Any]:
    """Evaluate on all test tasks once."""

    completed = [0]
    total = len(task_ids)

    def _run(tid):
        result = evaluate_on_task(wiki, tid, config)
        completed[0] += 1
        status = "SUCCESS" if result["success"] else "FAIL"
        print(f"  [{completed[0]}/{total}] Task {tid}: {status}")
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
        "task_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate with no wiki (baseline)")
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json",
        help="Path to test trajectories JSON",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of evaluation runs",
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
        default="evaluations/no_wiki_baseline",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NO-WIKI BASELINE EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Wiki: (empty string)")
    print(f"Runs: {args.n_runs}")

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

    # Empty wiki
    wiki = ""

    run_scores = []

    for run in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{args.n_runs}")
        print(f"{'='*60}")

        result = evaluate_once(
            wiki=wiki,
            task_ids=task_ids,
            config=config,
            max_concurrency=args.max_concurrency,
            run_num=run+1,
        )

        run_scores.append(result["success_rate"])

        # Save individual run result
        run_output = {
            "config": {
                "test_path": args.test_path,
                "model": args.model,
                "wiki": "(empty)",
                "run_number": run + 1,
            },
            "results": {
                "n_tasks": result["n_tasks"],
                "n_successes": result["n_successes"],
                "success_rate": result["success_rate"],
            },
            "task_results": result["task_results"],
            "timestamp": datetime.now().isoformat(),
        }

        run_file = output_dir / f"run{run+1}.json"
        with open(run_file, 'w') as f:
            json.dump(run_output, f, indent=2)

        print(f"→ {result['success_rate']:.1%} ({result['n_successes']}/{result['n_tasks']}) saved to {run_file.name}")

    # Summary
    avg_score = sum(run_scores) / len(run_scores)
    min_score = min(run_scores)
    max_score = max(run_scores)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Runs: {args.n_runs}")
    print(f"Scores: {[f'{s:.1%}' for s in run_scores]}")
    print(f"Average: {avg_score:.1%}")
    print(f"Min: {min_score:.1%}")
    print(f"Max: {max_score:.1%}")

    # Save summary
    summary = {
        "config": {
            "test_path": args.test_path,
            "model": args.model,
            "wiki": "(empty)",
            "n_runs": args.n_runs,
            "n_tasks": len(task_ids),
        },
        "results": {
            "run_scores": run_scores,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
