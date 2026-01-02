"""
Evaluate a learned prompt/wiki on the test set (115 tasks).

This script takes the output from rpe_iterative.py (or any saved wiki) and evaluates
it on the test set to get final performance metrics.

Usage:
    # Evaluate best candidate from RPE-EGA output
    python evaluate_prompt.py --prompt-path learned_policies/rpe_ega_result_XXXX.json

    # Evaluate a specific candidate by index (0 = best, 1 = second best, etc.)
    python evaluate_prompt.py --prompt-path learned_policies/rpe_ega_result_XXXX.json --candidate-index 2

    # Evaluate a standalone wiki file
    python evaluate_prompt.py --wiki-path learned_policies/my_policy.json

    # Evaluate on specific task IDs only
    python evaluate_prompt.py --prompt-path ... --task-ids 0 1 2 3 4
"""

import sys
from pathlib import Path
# Add parent directory to path so tau_bench can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import RunConfig


def load_wiki_from_rpe_output(
    prompt_path: str,
    candidate_index: int = 0,
) -> tuple[str, Dict[str, Any]]:
    """
    Load wiki from RPE-EGA output file.

    Args:
        prompt_path: Path to RPE-EGA output JSON
        candidate_index: Index in final_population (0 = best)

    Returns:
        (wiki_text, candidate_info)
    """
    with open(prompt_path, 'r') as f:
        data = json.load(f)

    if "final_population" in data:
        # RPE-EGA output format
        if candidate_index >= len(data["final_population"]):
            raise ValueError(f"candidate_index {candidate_index} out of range (only {len(data['final_population'])} candidates)")

        candidate = data["final_population"][candidate_index]
        wiki = candidate["wiki"]
        return wiki, candidate
    elif "wiki" in data:
        # Simple format with just a wiki field
        return data["wiki"], data
    else:
        raise ValueError(f"Could not find wiki in {prompt_path}")


def load_wiki_from_file(wiki_path: str) -> tuple[str, Dict[str, Any]]:
    """
    Load wiki from a standalone wiki JSON file.

    Args:
        wiki_path: Path to wiki JSON file

    Returns:
        (wiki_text, metadata)
    """
    with open(wiki_path, 'r') as f:
        data = json.load(f)

    if "wiki" in data:
        return data["wiki"], data
    elif isinstance(data, str):
        return data, {"wiki": data}
    else:
        raise ValueError(f"Could not find wiki in {wiki_path}")


def load_test_trajectories(test_path: str) -> List[Dict]:
    """Load test trajectories for reference."""
    with open(test_path, 'r') as f:
        return json.load(f)


def evaluate_on_task(
    wiki: str,
    task_id: int,
    config: RunConfig,
    save_conversations: bool = False,
) -> Dict[str, Any]:
    """Run agent with given wiki on a specific test task."""

    try:
        # Create environment
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            task_index=task_id,
        )

        # Create agent with wiki
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )

        # Run
        result = agent.solve(env=env, task_index=task_id)

        # Extract actions taken
        actions_taken = []
        for msg in result.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    actions_taken.append(tc["function"]["name"])

        task_result = {
            "task_id": task_id,
            "success": result.reward > 0,
            "reward": result.reward,
            "actions": actions_taken,
            "n_messages": len(result.messages),
            "error": None,
        }

        # Optionally include full conversation
        if save_conversations:
            task_result["messages"] = result.messages

        return task_result

    except Exception as e:
        print(f"  Error on task {task_id}: {e}")
        return {
            "task_id": task_id,
            "success": False,
            "reward": 0.0,
            "actions": [],
            "n_messages": 0,
            "error": str(e),
        }


def evaluate_wiki(
    wiki: str,
    task_ids: Optional[List[int]] = None,
    test_path: str = "data/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json",
    model: str = "gpt-4o",
    seed: int = 42,
    max_concurrency: int = 1,
    save_conversations: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a wiki on the test set.

    Args:
        wiki: The wiki/policy text to evaluate
        task_ids: Specific task IDs to evaluate (None = all 115)
        test_path: Path to test trajectories JSON
        model: Model to use for agent
        seed: Random seed
        max_concurrency: Max concurrent evaluations
        save_conversations: If True, include full message history in results

    Returns:
        Dict with evaluation results
    """
    # Load test trajectories to get task IDs
    test_trajectories = load_test_trajectories(test_path)

    if task_ids is None:
        task_ids = [t["task_id"] for t in test_trajectories]

    print(f"Evaluating on {len(task_ids)} test tasks...")

    # Config for test tasks
    config = RunConfig(
        env="retail",
        model=model,
        model_provider="openai",
        user_model="gpt-4o",
        user_model_provider="openai",
        agent_strategy="tool-calling",
        task_split="test",  # TEST split
        user_strategy="llm",
        temperature=0.0,
        seed=seed,
        start_index=0,
        end_index=-1,
        log_dir="./eval_logs",
        max_concurrency=max_concurrency,
        shuffle=False,
        num_trials=1,
        task_ids=[],
    )

    # Run evaluation
    results = []
    successes = 0

    if max_concurrency <= 1:
        # Sequential execution
        for i, task_id in enumerate(task_ids):
            print(f"  [{i+1}/{len(task_ids)}] Task {task_id}...", end=" ")
            result = evaluate_on_task(wiki, task_id, config, save_conversations)
            results.append(result)

            if result["success"]:
                successes += 1
                print("SUCCESS")
            else:
                print(f"FAIL (error: {result['error']})" if result["error"] else "FAIL")
    else:
        # Parallel execution with progress tracking
        print(f"Running with {max_concurrency} parallel workers...")
        completed = [0]  # Use list to allow modification in nested function
        total = len(task_ids)

        def _run_task(task_id: int) -> Dict[str, Any]:
            result = evaluate_on_task(wiki, task_id, config, save_conversations)
            completed[0] += 1
            status = "SUCCESS" if result["success"] else "FAIL"
            print(f"  [{completed[0]}/{total}] Task {task_id}: {status}")
            return result

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            results = list(executor.map(_run_task, task_ids))

        # Count successes
        for result in results:
            if result["success"]:
                successes += 1

    # Calculate metrics
    success_rate = successes / len(task_ids) if task_ids else 0.0
    mean_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0

    return {
        "n_tasks": len(task_ids),
        "n_successes": successes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "task_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a wiki on the test set")

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt-path",
        type=str,
        help="Path to RPE-EGA output JSON (contains final_population with wikis)",
    )
    input_group.add_argument(
        "--wiki-path",
        type=str,
        help="Path to standalone wiki JSON file",
    )

    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Which candidate to evaluate from final_population (0 = best, default)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json",
        help="Path to test trajectories JSON (115 tasks)",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific task IDs to evaluate (default: all 115)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for agent",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent evaluations",
    )
    parser.add_argument(
        "--save-conversations",
        action="store_true",
        help="Save full conversation messages in output (warning: large files)",
    )

    args = parser.parse_args()

    # Load wiki
    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    if args.prompt_path:
        print(f"\nLoading wiki from: {args.prompt_path}")
        print(f"Candidate index: {args.candidate_index}")
        wiki, candidate_info = load_wiki_from_rpe_output(args.prompt_path, args.candidate_index)
        source_info = {
            "type": "rpe_output",
            "path": args.prompt_path,
            "candidate_index": args.candidate_index,
            "candidate_id": candidate_info.get("id"),
            # Support both old (train_score) and new (posterior_mean, lcb_score) formats
            "train_score": candidate_info.get("train_score") or candidate_info.get("posterior_mean"),
            "posterior_mean": candidate_info.get("posterior_mean"),
            "lcb_score": candidate_info.get("lcb_score"),
        }
    else:
        print(f"\nLoading wiki from: {args.wiki_path}")
        wiki, candidate_info = load_wiki_from_file(args.wiki_path)
        source_info = {
            "type": "wiki_file",
            "path": args.wiki_path,
        }

    print(f"Wiki length: {len(wiki)} characters")
    print(f"Wiki preview: {wiki[:200]}...")

    # Run evaluation
    print(f"\n{'='*60}")
    eval_results = evaluate_wiki(
        wiki=wiki,
        task_ids=args.task_ids,
        test_path=args.test_path,
        model=args.model,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        save_conversations=args.save_conversations,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Tasks evaluated: {eval_results['n_tasks']}")
    print(f"Successes: {eval_results['n_successes']}")
    print(f"Success rate: {eval_results['success_rate']:.2%}")
    print(f"Mean reward: {eval_results['mean_reward']:.4f}")

    if source_info.get("train_score") is not None:
        print(f"\nTrain score was: {source_info['train_score']:.2%}")
        print(f"Test score is: {eval_results['success_rate']:.2%}")

    # Save results
    output = {
        "source": source_info,
        "config": {
            "test_path": args.test_path,
            "task_ids": args.task_ids,
            "model": args.model,
            "seed": args.seed,
        },
        "results": {
            "n_tasks": eval_results["n_tasks"],
            "n_successes": eval_results["n_successes"],
            "success_rate": eval_results["success_rate"],
            "mean_reward": eval_results["mean_reward"],
        },
        "task_results": eval_results["task_results"],
        "wiki": wiki,
        "timestamp": datetime.now().isoformat(),
    }

    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"evaluations/test_eval_{timestamp}.json"

    # Auto-create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")

    return output


if __name__ == "__main__":
    main()
