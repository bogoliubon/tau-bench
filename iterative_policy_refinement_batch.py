import json
import os
import argparse
import random
from typing import Optional, List, Dict

# Import utility functions
from utils import filter_tasks_by_tool, extract_conversation_text, call_llm


def tool_name_to_description(tool_name: str) -> str:
    """Convert tool name to human-readable description."""
    return tool_name.replace("_", " ")


def extract_ground_truth_actions(results_path: str, task_id: int, trial: int = 0) -> str:
    """
    Extract ground truth actions from the trajectory's info field.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        
    Returns:
        Formatted string of ground truth actions, or empty string if not found
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Find the trajectory
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            try:
                actions = result["info"]["task"]["actions"]
                if not actions:
                    return ""
                
                # Format actions nicely
                formatted_actions = []
                for i, action in enumerate(actions, 1):
                    name = action.get("name", "unknown_action")
                    kwargs = action.get("kwargs", {})
                    kwargs_str = json.dumps(kwargs, indent=2)
                    formatted_actions.append(f"{i}. {name}({kwargs_str})")
                
                return "\n".join(formatted_actions)
            except (KeyError, TypeError):
                return ""
    
    return ""


def prepare_trajectory_with_ground_truth(
    results_path: str,
    task_id: int,
    trial: int,
    include_instruction: bool = False
) -> tuple[str, bool]:
    """
    Prepare trajectory text, adding ground truth for failed trajectories.
    
    Args:
        results_path: Path to results.json
        task_id: Task ID
        trial: Trial number
        include_instruction: Whether to include instruction
        
    Returns:
        Tuple of (prepared_trajectory_text, is_success)
    """
    # Get conversation text
    conversation = extract_conversation_text(
        results_path,
        task_id,
        trial,
        include_instruction=include_instruction
    )
    
    if not conversation:
        return "", False
    
    # Check if successful
    with open(results_path, "r") as f:
        data = json.load(f)
    
    is_success = False
    for result in data:
        if result["task_id"] == task_id and result["trial"] == trial:
            is_success = (result.get("reward", 0.0) == 1.0)
            break
    
    # For successful trajectories, return as-is
    if is_success:
        prepared = f"=== SUCCESSFUL TRAJECTORY ===\n\n{conversation}\n"
        return prepared, True
    
    # For failed trajectories, add ground truth
    ground_truth = extract_ground_truth_actions(results_path, task_id, trial)
    
    if ground_truth:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH (What should have been done) ---\n{ground_truth}\n"
    else:
        prepared = f"=== FAILED TRAJECTORY ===\n\n{conversation}\n\n"
        prepared += f"--- GROUND TRUTH NOT AVAILABLE ---\n"
    
    return prepared, False


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


def prepare_batch_of_trajectories(
    results_path: str,
    task_ids: List[int],
    trial: int = 0
) -> tuple[str, int, int]:
    """
    Prepare a batch of trajectories for policy learning.
    
    Args:
        results_path: Path to results.json
        task_ids: List of task IDs to include in batch
        trial: Trial number
        
    Returns:
        Tuple of (batch_text, num_successful, num_failed)
    """
    batch_parts = []
    num_successful = 0
    num_failed = 0
    
    for task_id in task_ids:
        prepared, is_success = prepare_trajectory_with_ground_truth(
            results_path,
            task_id,
            trial,
            include_instruction=False
        )
        
        if prepared:
            batch_parts.append(prepared)
            if is_success:
                num_successful += 1
            else:
                num_failed += 1
    
    # Create summary header
    header = f"""
{'='*80}
TRAJECTORY BATCH FOR POLICY LEARNING
{'='*80}
Total trajectories: {len(batch_parts)}
- Successful: {num_successful}
- Failed (with ground truth): {num_failed}
{'='*80}

"""
    
    batch_text = header + "\n\n".join(batch_parts)
    
    return batch_text, num_successful, num_failed


def batch_policy_refinement(
    results_path: str,
    n_successful_per_flow: int = 5,
    n_failed_per_flow: int = 0,
    batch_size: int = None,
    model_name: str = "gpt-4o",
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Iteratively refine agent policy using batches of trajectories, organized flow-by-flow.
    
    Args:
        results_path: Path to results.json with trajectories
        n_successful_per_flow: Number of successful trajectories per flow type
        n_failed_per_flow: Number of failed trajectories per flow type
        batch_size: Number of trajectories per batch (None = all at once per flow)
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
    
    # Initialize refinement history
    refinement_history = {
        "config": {
            "n_successful_per_flow": n_successful_per_flow,
            "n_failed_per_flow": n_failed_per_flow,
            "batch_size": batch_size,
            "model_name": model_name,
            "results_path": results_path,
            "seed": seed,
            "flow_breakdown": flow_breakdown,
        },
        "iterations": []
    }
    
    current_policy = None
    
    # Prompts for batch extraction
    INITIAL_PROMPT = """You are analyzing customer service agent conversations to extract a comprehensive agent policy.

Here is a batch of conversations between agents and users:

{batch}

The batch includes both SUCCESSFUL conversations (showing what works well) and FAILED conversations with ground truth (showing what should have been done).

Please extract a comprehensive agent policy that captures:
1. How the agent gathers information and authenticates users
2. When and how the agent uses different tools for various workflows
3. How the agent communicates with users
4. Any specific constraints or requirements the agent must follow

For successful conversations: identify what they did right
For failed conversations: learn from the ground truth what should have been done

Provide a clear, structured policy description that covers all workflows."""

    UPDATE_PROMPT = """You previously extracted the following agent policy:

{current_policy}

Now consider this new batch of conversations:

{batch}

The batch includes both successful conversations and failed conversations with ground truth.

Based on this batch, does the policy need to be updated? Consider:
1. New patterns or rules revealed by these conversations
2. Gaps in the current policy (especially shown by failures)
3. Edge cases that need to be addressed
4. Clarifications or refinements needed

If NO update is needed (the current policy already covers everything well), respond with ONLY the word:
No

If an update IS needed, provide the COMPLETE updated policy (do not just describe the changes, provide the full policy text)."""

    # Process flow by flow
    flow_types = get_flow_types()
    batch_counter = 0
    
    for flow_idx, flow in enumerate(flow_types):
        print(f"\n{'='*70}")
        print(f"Processing flow: {flow} ({flow_idx+1}/{len(flow_types)})")
        print(f"{'='*70}")
        
        # Get task IDs for this flow (successful first, then failed)
        flow_task_ids = flow_breakdown[flow]['successful'] + flow_breakdown[flow]['failed']
        
        if not flow_task_ids:
            print(f"No trajectories for flow {flow}, skipping...")
            continue
        
        # If batch_size is None, process entire flow at once
        if batch_size is None:
            batches_for_flow = [flow_task_ids]
        else:
            # Split into batches
            batches_for_flow = [
                flow_task_ids[i:i+batch_size] 
                for i in range(0, len(flow_task_ids), batch_size)
            ]
        
        print(f"Processing {len(flow_task_ids)} trajectories in {len(batches_for_flow)} batch(es)")
        
        for batch_idx, batch_task_ids in enumerate(batches_for_flow):
            batch_counter += 1
            print(f"\n[Batch {batch_counter}] Flow: {flow}, Batch {batch_idx+1}/{len(batches_for_flow)}")
            print(f"  Processing {len(batch_task_ids)} trajectories: {batch_task_ids}")
            
            # Prepare batch
            batch_text, num_success, num_failed = prepare_batch_of_trajectories(
                results_path,
                batch_task_ids,
                trial=0
            )
            
            print(f"  Batch composition: {num_success} successful, {num_failed} failed")
            
            # Build prompt
            if batch_counter == 1:
                # First batch overall - initial extraction
                prompt = INITIAL_PROMPT.format(batch=batch_text)
            else:
                # Update existing policy
                prompt = UPDATE_PROMPT.format(
                    current_policy=current_policy,
                    batch=batch_text
                )
            
            # Call LLM
            print(f"  Calling {model_name}...")
            response = call_llm(prompt, model_name)
            
            # Check if update was made
            if response.strip() == "No":
                print(f"  No update needed - policy remains unchanged")
                updated = False
            else:
                print(f"  Policy updated (length: {len(response)} chars)")
                current_policy = response
                updated = True
            # Record iteration
            refinement_history["iterations"].append({
                "batch_number": batch_counter,
                "flow": flow,
                "task_ids": batch_task_ids,
                "num_successful": num_success,
                "num_failed": num_failed,
                "updated": updated,
                "policy": current_policy
            })
    
    # Save results
    if output_path is None:
        batch_str = f"batch{batch_size}" if batch_size else "flow_at_once"
        output_path = f"policy_refinement_{n_successful_per_flow}success_{n_failed_per_flow}failed_per_flow_{batch_str}_{model_name}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refinement_history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"[Done] Saved refinement history to {output_path}")
    print(f"Final policy length: {len(current_policy) if current_policy else 0} chars")
    print(f"Total batches processed: {batch_counter}")
    print(f"{'='*70}")
    
    return refinement_history


def main():
    parser = argparse.ArgumentParser(
        description="Batch-based iterative policy refinement with balanced sampling across flow types"
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
        "--batch_size",
        type=int,
        default=None,
        help="Number of trajectories per batch (None = process entire flow at once)"
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
    batch_policy_refinement(
        results_path=args.results_path,
        n_successful_per_flow=args.n_successful_per_flow,
        n_failed_per_flow=args.n_failed_per_flow,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_path=args.output_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()