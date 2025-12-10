import json
import os
import argparse
import random
from typing import Optional, List, Dict
# import anthropic
import openai

# Import utility functions
from utils import filter_tasks_by_tool, extract_conversation_text


def tool_name_to_description(tool_name: str) -> str:
    """Convert tool name to human-readable description."""
    return tool_name.replace("_", " ")


def call_llm(
    prompt: str,
    model: str = "claude",
    model_name: Optional[str] = None
) -> str:
    """
    Call LLM API to generate response.
    
    Args:
        prompt: The prompt to send
        model: Model family ("claude", "gpt", "llama")
        model_name: Specific model name (e.g., "claude-sonnet-4-5-20250929")
    
    Returns:
        Generated text response
    """
    if model == "claude":
        client = anthropic.Anthropic()
        model_name = model_name or "claude-sonnet-4-5-20250929"
        
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    elif model == "gpt":
        client = openai.OpenAI()
        model_name = model_name or "gpt-4o"
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    elif model == "llama":
        # TODO: Add llama implementation (via together.ai, replicate, or local)
        raise NotImplementedError("Llama support coming soon")
    
    else:
        raise ValueError(f"Unknown model: {model}")


def iterative_policy_refinement(
    results_path: str,
    n_traj: int,
    tool_name: Optional[str] = None,
    model: str = "claude",
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    success_only: bool = True,
) -> Dict:
    """
    Iteratively refine agent policy from successful trajectories.
    
    Args:
        results_path: Path to results.json with trajectories
        n_traj: Number of trajectories to use for refinement
        tool_name: Tool name to filter by (None for all tools)
        model: Model family to use ("claude", "gpt", "llama")
        model_name: Specific model name
        output_path: Path to save output JSON (if None, auto-generate)
        seed: Random seed for trajectory selection
        success_only: If True, only use successful trajectories
    
    Returns:
        Dictionary with refinement history
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Filter tasks (always success_only=True)
    print(f"Filtering tasks (tool={tool_name}, success_only=True)...")
    # for non-modify tools, we just filter by tool_name
    if tool_name != "modify":
        task_ids_by_tool, task_ids_by_tool_success = filter_tasks_by_tool(results_path, tool_name)
        task_ids = task_ids_by_tool_success if success_only else task_ids_by_tool
    else:
        modify_task_ids = []
        modify_task_ids_success = []
        # for modify, we take all modify tasks
        modify_tools=['modify_pending_order_address',
            'modify_pending_order_items',
            'modify_pending_order_payment']
        for task in modify_tools:
            t_ids, t_ids_success = filter_tasks_by_tool(results_path, task)
            modify_task_ids.extend(t_ids)
            modify_task_ids_success.extend(t_ids_success)
        task_ids = list(set(modify_task_ids_success)) if success_only else list(set(modify_task_ids))   
            
    if len(task_ids) < n_traj:
        print(f"[warning] Only {len(task_ids)} tasks available, requested {n_traj}")
        n_traj = len(task_ids)
    
    # Randomly sample tasks
    selected_task_ids = random.sample(task_ids, n_traj)
    print(f"Found {len(task_ids)} matching tasks, randomly selected {n_traj}")
    print(f"Selected task IDs: {selected_task_ids}")
    
    # Initialize refinement history
    refinement_history = {
        "config": {
            "tool_name": tool_name,
            "n_traj": n_traj,
            "model": model,
            "model_name": model_name,
            "results_path": results_path,
            "seed": seed,
            "selected_task_ids": selected_task_ids,
        },
        "iterations": []
    }
    
    current_policy = None
    
    # Prompts for flow-specific extraction
    INITIAL_PROMPT_FLOW_SPECIFIC = """You are analyzing successful customer service agent conversations to extract the agent's policy for handling {flow_description}.

Here is a conversation between an agent and a user:

{trajectory}

Please extract and describe the agent's policy specifically for {flow_description}. Focus on:
1. How the agent gathers information needed for {flow_description}
2. When and how the agent uses the {tool_name} tool
3. How the agent communicates with the user during this process
4. Any specific constraints or requirements for {flow_description}

Provide a clear, structured policy description focused on this specific workflow."""

    UPDATE_PROMPT_FLOW_SPECIFIC = """You previously extracted the following agent policy for {flow_description}:

{current_policy}

Here is another successful conversation involving {flow_description}:

{trajectory}

Based on this new conversation, does the policy need to be updated? When considering updates, focus on:
1. How the agent gathers information needed for {flow_description}
2. When and how the agent uses the {tool_name} tool
3. How the agent communicates with the user during this process
4. Any specific constraints or requirements for {flow_description}

Consider:
- Are there new patterns or rules to add?
- Are there existing rules that need refinement?
- Does the policy need any corrections?

If NO update is needed (the current policy already covers this conversation well), respond with ONLY the word:
No

If an update IS needed, provide the COMPLETE updated policy (do not just describe the changes, provide the full policy text)."""

    # Prompts for general extraction (all flows)
    INITIAL_PROMPT_GENERAL = """You are analyzing successful customer service agent conversations to extract the agent's policy.

Here is a conversation between an agent and a user:

{trajectory}

Please extract and describe the agent's policy - the rules, guidelines, and strategies the agent follows to successfully handle customer requests. Focus on:
1. How the agent gathers information
2. When and how the agent uses tools
3. How the agent communicates with the user
4. Any specific constraints or requirements the agent follows

Provide a clear, structured policy description."""

    UPDATE_PROMPT_GENERAL = """You previously extracted the following agent policy:

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

    # Iterative refinement
    # import pdb; pdb.set_trace()
    for i in range(n_traj):
        task_id = selected_task_ids[i]
        trial = 0  # Assuming trial 0 for now
        
        print(f"\n[Iteration {i+1}/{n_traj}] Processing task_id={task_id}")
        
        # Extract trajectory
        trajectory = extract_conversation_text(
            results_path, 
            task_id, 
            trial,
            include_instruction=False
        )
        
        if not trajectory:
            print(f"[warning] Could not extract trajectory for task_id={task_id}")
            continue
        
        # Build prompt based on flow-specific vs general
        if tool_name is not None:
            flow_desc = tool_name_to_description(tool_name)
            if i == 0:
                prompt = INITIAL_PROMPT_FLOW_SPECIFIC.format(
                    trajectory=trajectory,
                    flow_description=flow_desc,
                    tool_name=tool_name
                )
            else:
                prompt = UPDATE_PROMPT_FLOW_SPECIFIC.format(
                    current_policy=current_policy,
                    trajectory=trajectory,
                    flow_description=flow_desc,
                    tool_name=tool_name
                )
        else:
            # General extraction (all flows)
            if i == 0:
                prompt = INITIAL_PROMPT_GENERAL.format(trajectory=trajectory)
            else:
                prompt = UPDATE_PROMPT_GENERAL.format(
                    current_policy=current_policy,
                    trajectory=trajectory
                )
        
        # Call LLM
        print(f"Calling {model}...")
        response = call_llm(prompt, model, model_name)
        
        # Check if update was made
        # import pdb; pdb.set_trace()
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
            "updated": updated,
            "policy": current_policy
        })
    
    # Save results
    if output_path is None:
        tool_str = tool_name if tool_name else "all_tools"
        output_path = f"policy_refinement_{tool_str}_{n_traj}traj_{model}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refinement_history, f, indent=2)
    
    print(f"\n[Done] Saved refinement history to {output_path}")
    print(f"Final policy length: {len(current_policy)} chars")
    
    return refinement_history


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively refine agent policy from successful trajectories"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--n_traj",
        type=int,
        required=True,
        help="Number of trajectories to use for refinement"
    )
    parser.add_argument(
        "--tool_name",
        type=str,
        default=None,
        help="Tool name to filter by (e.g., 'exchange_delivered_order_items'). If not specified, use all tools."
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
        n_traj=args.n_traj,
        tool_name=args.tool_name,
        model=args.model,
        model_name=args.model_name,
        output_path=args.output_path,
        seed=args.seed,
        success_only=True,
    )


if __name__ == "__main__":
    main()