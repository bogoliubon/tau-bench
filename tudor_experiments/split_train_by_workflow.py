"""
One-time script to split training trajectories into workflow-specific files.

Run once:
    python split_train_by_workflow.py

Creates:
    results/train_cancel.json
    results/train_exchange.json
    results/train_return.json
    results/train_modify.json
"""

import json
from collections import defaultdict
from typing import List, Dict

# Paths relative to tudor_experiments/ directory
TRAIN_PATH = "data/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1210141628-train.json"
OUTPUT_DIR = "data"

WORKFLOW_TOOLS = {
    "cancel": ["cancel_pending_order"],
    "exchange": ["exchange_delivered_order_items"],
    "return": ["return_delivered_order_items"],
    "modify": ["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"],
}


def classify_workflows(trajectory: Dict) -> List[str]:
    """Classify a trajectory by ALL workflow types it contains (non-exclusive)."""
    try:
        actions = trajectory["info"]["task"]["actions"]
        action_names = [a["name"] for a in actions]
    except (KeyError, TypeError):
        return []

    workflows = []
    for workflow, tools in WORKFLOW_TOOLS.items():
        if any(tool in action_names for tool in tools):
            workflows.append(workflow)
    return workflows


def main():
    # Load all training trajectories
    print(f"Loading training data from: {TRAIN_PATH}")
    with open(TRAIN_PATH, 'r') as f:
        all_trajectories = json.load(f)

    print(f"Total trajectories: {len(all_trajectories)}")

    # Filter for successful only
    successful = [t for t in all_trajectories if t.get("reward", 0) > 0]
    print(f"Successful trajectories: {len(successful)}")

    # Split by workflow (non-exclusive)
    by_workflow = defaultdict(list)
    for traj in successful:
        workflows = classify_workflows(traj)
        for workflow in workflows:
            by_workflow[workflow].append(traj)

    # Save each workflow to a separate file
    print("\nSaving workflow-specific files:")
    for workflow, trajs in sorted(by_workflow.items()):
        output_path = f"{OUTPUT_DIR}/train_{workflow}.json"
        with open(output_path, 'w') as f:
            json.dump(trajs, f, indent=2)

        # Get unique task IDs
        task_ids = sorted(set(t["task_id"] for t in trajs))
        print(f"  {workflow}: {len(trajs)} trajectories, {len(task_ids)} unique task_ids -> {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_slots = sum(len(trajs) for trajs in by_workflow.values())
    print(f"Total unique trajectories: {len(successful)}")
    print(f"Total slots across all groups: {total_slots}")
    print(f"Overlap (multi-workflow tasks): {total_slots - len(successful)}")

    print("\nFiles created:")
    for workflow in sorted(by_workflow.keys()):
        print(f"  results/train_{workflow}.json")


if __name__ == "__main__":
    main()
