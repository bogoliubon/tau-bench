import json
import os
import argparse
import random
from typing import Optional, List, Dict

# Import utility functions
from utils import filter_tasks_by_tool, call_llm, tool_name_to_description, prepare_trajectory_with_ground_truth, extract_ground_truth_actions

def get_flow_types() -> List[str]:
    """Get all flow types we want to cover."""
    return [
        'exchange_delivered_order_items',
        'return_delivered_order_items', 
        'cancel_pending_order',
        'modify'  # Aggregates modify_pending_order_address/items/payment
    ]


def sample_trajectories_per_flow(
    results_path: str,
    n_successful_per_flow: int,
    n_failed_per_flow: int,
    seed: Optional[int] = None,
) -> tuple[List[int], Dict[str, Dict[str, List[int]]]]:
    """
    Sample trajectories ensuring balanced coverage across flow types.
    
    Args:
        results_path: Path to results.json
        n_successful_per_flow: Number of successful trajectories per flow
        n_failed_per_flow: Number of failed trajectories per flow
        seed: Random seed for sampling
        
    Returns:
        Tuple of (selected_task_ids, flow_breakdown)
        where flow_breakdown maps flow_name -> {'successful': [...], 'failed': [...]}
    """
    if seed is not None:
        random.seed(seed)
    
    flow_types = get_flow_types()
    
    # Load all data to check success/failure
    with open(results_path, "r") as f:
        all_data = json.load(f)
    
    # Create lookup for reward by task_id
    task_id_to_reward = {result["task_id"]: result.get("reward", 0.0) for result in all_data}
    
    selected_task_ids = []
    flow_breakdown = {}
    
    for flow in flow_types:
        print(f"\nSampling for flow: {flow}")
        
        # Get all task_ids for this flow (both successful and failed)
        if flow == 'modify':
            # Special handling for modify (aggregates 3 tools)
            all_flow_task_ids = []
            modify_tools = [
                'modify_pending_order_address',
                'modify_pending_order_items',
                'modify_pending_order_payment'
            ]
            for tool in modify_tools:
                task_ids, _ = filter_tasks_by_tool(results_path, tool)
                all_flow_task_ids.extend(task_ids)
            all_flow_task_ids = list(set(all_flow_task_ids))  # Remove duplicates
        else:
            all_flow_task_ids, _ = filter_tasks_by_tool(results_path, flow)
        
        # Split into successful and failed
        successful_ids = [tid for tid in all_flow_task_ids if task_id_to_reward.get(tid, 0.0) > 0.0]
        failed_ids = [tid for tid in all_flow_task_ids if task_id_to_reward.get(tid, 0.0) == 0.0]
        
        print(f"  Available: {len(successful_ids)} successful, {len(failed_ids)} failed")
        
        # Sample successful
        n_success_to_sample = min(n_successful_per_flow, len(successful_ids))
        if n_success_to_sample < n_successful_per_flow:
            print(f"  [warning] Only {n_success_to_sample} successful available, requested {n_successful_per_flow}")
        
        sampled_successful = random.sample(successful_ids, n_success_to_sample) if n_success_to_sample > 0 else []
        
        # Sample failed
        n_failed_to_sample = min(n_failed_per_flow, len(failed_ids))
        if n_failed_to_sample < n_failed_per_flow:
            print(f"  [warning] Only {n_failed_to_sample} failed available, requested {n_failed_per_flow}")
        
        sampled_failed = random.sample(failed_ids, n_failed_to_sample) if n_failed_to_sample > 0 else []
        
        # Store breakdown for this flow
        flow_breakdown[flow] = {
            'successful': sampled_successful,
            'failed': sampled_failed
        }
        
        # Add to selected list: successful first, then failed (for this flow)
        selected_task_ids.extend(sampled_successful)
        selected_task_ids.extend(sampled_failed)
        
        print(f"  Sampled: {len(sampled_successful)} successful + {len(sampled_failed)} failed = {len(sampled_successful) + len(sampled_failed)} total")
    
    # NOTE: We do NOT shuffle here - we keep flow-by-flow, successful-first order
    
    return selected_task_ids, flow_breakdown


def iterative_policy_refinement(
    results_path: str,
    n_successful_per_flow: int = 5,
    n_failed_per_flow: int = 0,
    model_name: str = "gpt-4o",
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Iteratively refine agent policy with balanced sampling across flow types.
    
    Args:
        results_path: Path to results.json with trajectories
        n_successful_per_flow: Number of successful trajectories per flow type
        n_failed_per_flow: Number of failed trajectories per flow type
        model_name: Specific model name
        output_path: Path to save output JSON (if None, auto-generate)
        seed: Random seed for trajectory selection
    
    Returns:
        Dictionary with refinement history
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Sample trajectories per flow
    print(f"Sampling trajectories: {n_successful_per_flow} successful + {n_failed_per_flow} failed per flow...")
    selected_task_ids, flow_breakdown = sample_trajectories_per_flow(
        results_path,
        n_successful_per_flow,
        n_failed_per_flow,
        seed
    )
    
    print(f"\nTotal trajectories selected: {len(selected_task_ids)}")
    print(f"Selected task IDs: {selected_task_ids}")
    
    # Initialize refinement history
    refinement_history = {
        "config": {
            "n_successful_per_flow": n_successful_per_flow,
            "n_failed_per_flow": n_failed_per_flow,
            "model_name": model_name,
            "results_path": results_path,
            "seed": seed,
            "flow_breakdown": flow_breakdown,
            "selected_task_ids": selected_task_ids,
        },
        "iterations": []
    }
    
    current_policy = None
    
    # Prompts - use general extraction since we're covering all flows
    INITIAL_PROMPT_SUCCESS = """You are analyzing successful customer service agent conversations to extract the agent's policy for handling different workflows.

Here is a conversation between an agent and a user:

{trajectory}

Please extract and describe the agent's policy - the rules, guidelines, and strategies the agent follows to successfully handle customer requests. Focus on:
1. How the agent gathers information
2. When and how the agent uses tools
3. How the agent communicates with the user
4. Any specific constraints or requirements the agent follows

Provide a clear, structured policy description."""

    INITIAL_PROMPT_FAILURE = """You are analyzing customer service conversations to extract the agent's policy.

Here is a FAILED conversation with the ground truth of what should have been done:

{trajectory}

The conversation shows what went wrong, and the ground truth shows what should have been done.
Please extract the agent's policy based on the correct behavior shown in the ground truth. Focus on:
1. How the agent should gather information
2. When and how the agent should use tools
3. How the agent should communicate with the user
4. Any specific constraints or requirements the agent should follow

Provide a clear, structured policy description that would lead to the correct behavior shown in the ground truth."""

    UPDATE_PROMPT_SUCCESS = """You previously extracted the following agent policy:

{current_policy}

Here is another successful conversation:

{trajectory}

Based on this new conversation, does the policy need to be updated? When considering updates, focus on:
1. How the agent gathers information
2. When and how the agent uses tools
3. How the agent communicates with the user
4. Any specific constraints or requirements the agent follows

Consider:
- Are there new patterns or rules to add?
- Are there existing rules that need refinement?
- Does the policy need any corrections?

If NO update is needed (the current policy already covers this conversation well), respond with ONLY the word:
No

If an update IS needed, provide the COMPLETE updated policy (do not just describe the changes, provide the full policy text)."""

    UPDATE_PROMPT_FAILURE = """You previously extracted the following agent policy:

{current_policy}

Here is a FAILED conversation with the ground truth of what should have been done:

{trajectory}

The conversation shows what went wrong, and the ground truth shows what should have been done.
Based on this failure and the correct behavior, does the policy need to be updated? Focus on:
1. Missing requirements that led to the failure
2. Incorrect procedures that need correction
3. Additional checks or validations needed
4. Clarifications to prevent similar mistakes

Consider:
- Are there gaps in the current policy that this failure reveals?
- Do existing rules need refinement to prevent this type of error?
- Are there new constraints or requirements to add?

If NO update is needed (the current policy already covers what the ground truth shows), respond with ONLY the word:
No

If an update IS needed, provide the COMPLETE updated policy (do not just describe the changes, provide the full policy text)."""

    # Iterative refinement
    n_traj = len(selected_task_ids)
    
    # Create a mapping to know which flow each task belongs to (for logging)
    task_to_flow = {}
    for flow, breakdown in flow_breakdown.items():
        for tid in breakdown['successful']:
            task_to_flow[tid] = (flow, 'successful')
        for tid in breakdown['failed']:
            task_to_flow[tid] = (flow, 'failed')
    
    for i in range(n_traj):
        task_id = selected_task_ids[i]
        trial = 0  # Assuming trial 0 for now
        
        # Get flow and type for logging
        flow_name, expected_type = task_to_flow.get(task_id, ('unknown', 'unknown'))
        
        print(f"\n[Iteration {i+1}/{n_traj}] Processing task_id={task_id} (flow: {flow_name}, expected: {expected_type})")
        
        # Extract trajectory with ground truth if failed
        trajectory, is_success = prepare_trajectory_with_ground_truth(
            results_path, 
            task_id, 
            trial,
            include_instruction=False
        )
        
        if not trajectory:
            print(f"[warning] Could not extract trajectory for task_id={task_id}")
            continue
        
        traj_type = "SUCCESS" if is_success else "FAILED"
        print(f"  Trajectory type: {traj_type}")
        
        # Sanity check
        actual_type = 'successful' if is_success else 'failed'
        if actual_type != expected_type:
            print(f"  [warning] Expected {expected_type} but got {actual_type}")
        
        # Build prompt based on success vs failure
        if i == 0:
            # Initial extraction
            if is_success:
                prompt = INITIAL_PROMPT_SUCCESS.format(trajectory=trajectory)
            else:
                prompt = INITIAL_PROMPT_FAILURE.format(trajectory=trajectory)
        else:
            # Update existing policy
            if is_success:
                prompt = UPDATE_PROMPT_SUCCESS.format(
                    current_policy=current_policy,
                    trajectory=trajectory
                )
            else:
                prompt = UPDATE_PROMPT_FAILURE.format(
                    current_policy=current_policy,
                    trajectory=trajectory
                )
        
        # Call LLM
        print(f"Calling {model_name}...")
        response = call_llm(prompt, model_name)
        
        # Check if update was made
        if response.strip() == "No":
            print(f"No update needed - policy remains unchanged")
            updated = False
        else:
            print(f"Policy updated (length: {len(response)} chars)")
            current_policy = response
            updated = True
        
        # Record iteration
        refinement_history["iterations"].append({
            "iteration": i + 1,
            "task_id": task_id,
            "flow": flow_name,
            "is_success": is_success,
            "updated": updated,
            "policy": current_policy
        })
    
    # Save results
    if output_path is None:
        output_path = f"policy_refinement_{n_successful_per_flow}success_{n_failed_per_flow}failed_per_flow_{model_name}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refinement_history, f, indent=2)
    
    print(f"\n[Done] Saved refinement history to {output_path}")
    print(f"Final policy length: {len(current_policy)} chars")
    
    # Print statistics
    num_success = sum(1 for it in refinement_history["iterations"] if it.get("is_success", True))
    num_failed = n_traj - num_success
    print(f"Trajectories processed: {num_success} successful, {num_failed} failed")
    
    return refinement_history


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively refine agent policy with balanced sampling across flow types"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--n_successful_per_flow",
        type=int,
        default=5,
        help="Number of successful trajectories per flow type"
    )
    parser.add_argument(
        "--n_failed_per_flow",
        type=int,
        default=0,
        help="Number of failed trajectories per flow type"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["claude", "gpt", "llama"],
        default="claude",
        help="Model family to use"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specific model name (e.g., 'claude-sonnet-4-5-20250929', 'gpt-4o')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (auto-generated if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trajectory selection"
    )
    
    args = parser.parse_args()
    
    # Run refinement
    iterative_policy_refinement(
        results_path=args.results_path,
        n_successful_per_flow=args.n_successful_per_flow,
        n_failed_per_flow=args.n_failed_per_flow,
        model_name=args.model_name,
        output_path=args.output_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()