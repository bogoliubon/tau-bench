"""
RPE-Style Wiki Induction for Tau-Bench

Implements Kilian's feedback:
1. Use multiple trajectories at once
2. Iterate with different orderings to check stability
3. Generate multiple candidate wikis
4. Evaluate each by re-running agent on source tasks
5. Select best candidate based on success rate

Based on: "Reverse Prompt Engineering" (Li & Klabjan, 2025)
Adapted for tau-bench where metric = task success rate (not prompt similarity)
"""

import json
import random
import os
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from statistics import mean, stdev

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import RunConfig


# ============================================================================
# Evaluation Mode
# ============================================================================

class EvalMode(Enum):
    """Evaluation mode for candidate scoring"""
    REWARD = "reward"              # Final state matching (tau-bench reward)
    ACTION_SEQUENCE = "action"     # Action sequence similarity
    BOTH = "both"                  # Both metrics (for analysis)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WikiCandidate:
    """A candidate wiki with metadata about how it was generated"""
    wiki: str
    source_task_ids: List[int]
    ordering_idx: int
    candidate_idx: int
    generation_history: List[Dict]  # Refinement steps

    # Filled in during evaluation - reward-based
    source_task_scores: Optional[Dict[int, float]] = None
    source_task_success_rate: Optional[float] = None

    # Filled in during evaluation - action sequence-based
    action_similarity_scores: Optional[Dict[int, float]] = None
    mean_action_similarity: Optional[float] = None

    # Store trajectories for action comparison
    generated_trajectories: Optional[Dict[int, List[Dict]]] = None


@dataclass
class RPEResult:
    """Result of RPE-style wiki induction"""
    best_candidate: WikiCandidate
    all_candidates: List[WikiCandidate]
    ordering_stability: Dict[str, Any]
    evaluation_results: Dict[str, Any]


# ============================================================================
# Trajectory Loading & Filtering
# ============================================================================

def load_trajectories_from_file(
    filepath: str,
    success_only: bool = True,
    tool_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Load trajectories from JSON file.

    Args:
        filepath: Path to trajectories JSON
        success_only: Only return successful trajectories (reward >= 0.99)
        tool_filter: Only return trajectories using this tool
    """
    with open(filepath, 'r') as f:
        trajectories = json.load(f)

    results = []
    for traj in trajectories:
        # Filter by success
        if success_only and traj.get('reward', 0) < 0.99:
            continue

        # Filter by tool if specified
        if tool_filter:
            actions = traj.get('info', {}).get('task', {}).get('actions', [])
            tool_names = [a.get('name', '') for a in actions]
            if tool_filter not in tool_names:
                continue

        results.append(traj)

    return results


def extract_conversation_text(traj: Dict, include_system: bool = False) -> str:
    """Extract conversation as formatted text for LLM analysis"""
    messages = traj.get('traj', [])
    formatted = []

    start_idx = 0 if include_system else 1  # Skip system message by default

    for msg in messages[start_idx:]:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'user':
            formatted.append(f"User: {content}")
        elif role == 'assistant':
            if msg.get('tool_calls'):
                for tool_call in msg['tool_calls']:
                    func_name = tool_call['function']['name']
                    func_args = tool_call['function']['arguments']
                    # Truncate long args
                    if len(func_args) > 200:
                        func_args = func_args[:200] + "..."
                    formatted.append(f"Agent [tool]: {func_name}({func_args})")
            elif content:
                formatted.append(f"Agent: {content}")
        elif role == 'tool':
            truncated = content[:150] + "..." if len(content) > 150 else content
            formatted.append(f"Tool result: {truncated}")

    return "\n".join(formatted)


# ============================================================================
# Action Sequence Extraction & Similarity
# ============================================================================

def extract_tool_calls(messages: List[Dict]) -> List[Tuple[str, Dict]]:
    """
    Extract tool calls from a conversation as (tool_name, arguments) tuples.

    Args:
        messages: List of message dicts from a trajectory

    Returns:
        List of (tool_name, parsed_arguments) tuples
    """
    tool_calls = []

    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                func = tc.get('function', {})
                name = func.get('name', '')
                args_str = func.get('arguments', '{}')

                # Parse arguments
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}

                tool_calls.append((name, args))

    return tool_calls


def extract_tool_calls_from_traj(traj: Dict) -> List[Tuple[str, Dict]]:
    """Extract tool calls from a trajectory dict"""
    messages = traj.get('traj', traj.get('messages', []))
    return extract_tool_calls(messages)


def action_sequence_similarity(
    original_actions: List[Tuple[str, Dict]],
    new_actions: List[Tuple[str, Dict]],
    mode: Literal["exact", "name_only", "jaccard"] = "exact",
) -> float:
    """
    Compare two action sequences and return similarity score.

    Args:
        original_actions: Ground truth actions [(name, args), ...]
        new_actions: Generated actions [(name, args), ...]
        mode: Comparison mode
            - "exact": Exact match of (name, args) pairs in order
            - "name_only": Match tool names only (ignore args)
            - "jaccard": Set-based Jaccard similarity of tool names

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not original_actions and not new_actions:
        return 1.0
    if not original_actions or not new_actions:
        return 0.0

    if mode == "exact":
        # Exact sequence match with edit distance-like scoring
        matches = 0
        max_len = max(len(original_actions), len(new_actions))
        min_len = min(len(original_actions), len(new_actions))

        for i in range(min_len):
            orig_name, orig_args = original_actions[i]
            new_name, new_args = new_actions[i]

            if orig_name == new_name:
                # Name matches, check args
                if orig_args == new_args:
                    matches += 1.0
                else:
                    # Partial credit for same tool with different args
                    matches += 0.5

        return matches / max_len

    elif mode == "name_only":
        # Just compare tool names in sequence
        matches = 0
        max_len = max(len(original_actions), len(new_actions))
        min_len = min(len(original_actions), len(new_actions))

        for i in range(min_len):
            if original_actions[i][0] == new_actions[i][0]:
                matches += 1

        return matches / max_len

    elif mode == "jaccard":
        # Set-based comparison of tool names (order-independent)
        orig_names = set(a[0] for a in original_actions)
        new_names = set(a[0] for a in new_actions)

        intersection = len(orig_names & new_names)
        union = len(orig_names | new_names)

        return intersection / union if union > 0 else 0.0

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_ground_truth_actions(traj: Dict) -> List[Tuple[str, Dict]]:
    """Extract ground truth actions from trajectory info"""
    task_info = traj.get('info', {}).get('task', {})
    actions = task_info.get('actions', [])

    result = []
    for action in actions:
        name = action.get('name', '')
        kwargs = action.get('kwargs', {})
        result.append((name, kwargs))

    return result


# ============================================================================
# Wiki Generation (Simple LLM-based, no DSPy dependency)
# ============================================================================

def call_llm(
    messages: List[Dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """Call LLM and return response text"""
    from litellm import completion

    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_initial_wiki(
    conversations: List[str],
    tools_summary: str,
    task_domain: str = "retail customer service",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, str]:
    """
    Generate initial wiki from multiple conversations at once.
    Returns (wiki, reasoning).
    """
    conversations_text = "\n\n---\n\n".join([
        f"Example {i+1}:\n{conv}"
        for i, conv in enumerate(conversations)
    ])

    prompt = f"""You are analyzing successful agent conversations to extract a system prompt (wiki) that would guide an agent to behave similarly.

Task Domain: {task_domain}
Available Tools: {tools_summary}

Here are {len(conversations)} successful conversations showing correct agent behavior:

{conversations_text}

Based on ALL these examples, extract:
1. The key policies and rules the agent follows
2. The workflow patterns (what steps, in what order)
3. Important constraints or edge cases

Write a concise system prompt that would guide an agent to handle similar tasks correctly.

Format your response as:
REASONING: [Your analysis of patterns across examples]
WIKI: [The system prompt]"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, model=model, temperature=temperature)

    # Parse response
    if "WIKI:" in response:
        parts = response.split("WIKI:", 1)
        reasoning = parts[0].replace("REASONING:", "").strip()
        wiki = parts[1].strip()
    else:
        reasoning = ""
        wiki = response

    return wiki, reasoning


def refine_wiki(
    current_wiki: str,
    new_conversations: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, bool, str]:
    """
    Refine wiki based on additional conversations.
    Returns (refined_wiki, needs_revision, reasoning).
    """
    conversations_text = "\n\n---\n\n".join([
        f"Example {i+1}:\n{conv}"
        for i, conv in enumerate(new_conversations)
    ])

    prompt = f"""You have a current system prompt (wiki) for a customer service agent:

CURRENT WIKI:
{current_wiki}

Here are additional successful conversations:

{conversations_text}

Analyze whether the current wiki adequately covers the policies and behaviors shown in these new examples.

If the wiki is missing important policies or could be improved, revise it.
If the wiki already covers everything, keep it as is.

Format your response as:
NEEDS_REVISION: [yes/no]
REASONING: [What's missing or why no changes needed]
REVISED_WIKI: [The updated wiki, or the original if no changes]"""

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages, model=model, temperature=temperature)

    # Parse response
    needs_revision = "yes" in response.lower().split("NEEDS_REVISION:")[-1].split("\n")[0].lower() if "NEEDS_REVISION:" in response else False

    if "REVISED_WIKI:" in response:
        parts = response.split("REVISED_WIKI:", 1)
        reasoning_part = parts[0]
        wiki = parts[1].strip()
        if "REASONING:" in reasoning_part:
            reasoning = reasoning_part.split("REASONING:", 1)[1].split("REVISED_WIKI:")[0].strip()
        else:
            reasoning = reasoning_part.strip()
    else:
        reasoning = ""
        wiki = current_wiki

    return wiki, needs_revision, reasoning


def generate_wiki_from_trajectories(
    conversations: List[str],
    tools_summary: str,
    task_domain: str = "retail customer service",
    batch_size: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, List[Dict]]:
    """
    Generate wiki using iterative refinement with batches of conversations.

    Args:
        conversations: List of formatted conversation texts
        tools_summary: Available tools description
        task_domain: Domain description
        batch_size: How many conversations to process at once
        model: LLM model to use
        temperature: Sampling temperature

    Returns:
        (final_wiki, generation_history)
    """
    history = []

    # Step 1: Generate initial wiki from first batch
    first_batch = conversations[:batch_size]
    wiki, reasoning = generate_initial_wiki(
        first_batch, tools_summary, task_domain, model, temperature
    )

    history.append({
        "step": 0,
        "type": "initial",
        "num_examples": len(first_batch),
        "reasoning": reasoning,
        "wiki": wiki
    })

    # Step 2: Iteratively refine with remaining batches
    remaining = conversations[batch_size:]
    step = 1

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        if not batch:
            break

        wiki, needs_revision, reasoning = refine_wiki(
            wiki, batch, model, temperature
        )

        history.append({
            "step": step,
            "type": "refinement",
            "num_examples": len(batch),
            "needs_revision": needs_revision,
            "reasoning": reasoning,
            "wiki": wiki
        })
        step += 1

    return wiki, history


# ============================================================================
# Multi-Candidate Generation with Ordering Variation
# ============================================================================

def generate_candidates_with_ordering(
    trajectories: List[Dict],
    tools_summary: str,
    num_orderings: int = 3,
    candidates_per_ordering: int = 2,
    batch_size: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    seed: int = 42,
    verbose: bool = True,
) -> List[WikiCandidate]:
    """
    Generate multiple wiki candidates with trajectory ordering variation.

    This implements Kilian's feedback:
    - Use multiple trajectories at once
    - Iterate with different orderings
    - Generate multiple candidates

    Args:
        trajectories: List of trajectory dicts
        tools_summary: Available tools description
        num_orderings: Number of different trajectory orderings to try
        candidates_per_ordering: Candidates to generate per ordering
        batch_size: Conversations per batch in iterative refinement
        model: LLM model
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        List of WikiCandidate objects
    """
    random.seed(seed)
    candidates = []

    # Extract conversations and task IDs
    all_conversations = [extract_conversation_text(t) for t in trajectories]
    all_task_ids = [t['task_id'] for t in trajectories]

    total_candidates = num_orderings * candidates_per_ordering

    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING {total_candidates} WIKI CANDIDATES")
        print(f"{'='*60}")
        print(f"  Trajectories: {len(trajectories)}")
        print(f"  Orderings: {num_orderings}")
        print(f"  Candidates per ordering: {candidates_per_ordering}")
        print(f"  Batch size: {batch_size}")

    for ordering_idx in range(num_orderings):
        # Create shuffled ordering
        indices = list(range(len(all_conversations)))
        random.shuffle(indices)

        shuffled_conversations = [all_conversations[i] for i in indices]
        shuffled_task_ids = [all_task_ids[i] for i in indices]

        if verbose:
            print(f"\n  Ordering {ordering_idx + 1}/{num_orderings}: {shuffled_task_ids[:5]}...")

        for candidate_idx in range(candidates_per_ordering):
            if verbose:
                print(f"    Generating candidate {candidate_idx + 1}/{candidates_per_ordering}...", end=" ")

            wiki, history = generate_wiki_from_trajectories(
                conversations=shuffled_conversations,
                tools_summary=tools_summary,
                batch_size=batch_size,
                model=model,
                temperature=temperature,
            )

            candidate = WikiCandidate(
                wiki=wiki,
                source_task_ids=shuffled_task_ids,
                ordering_idx=ordering_idx,
                candidate_idx=candidate_idx,
                generation_history=history,
            )
            candidates.append(candidate)

            if verbose:
                print(f"done ({len(wiki)} chars)")

    return candidates


# ============================================================================
# Candidate Evaluation (RPE-style: re-run on source tasks)
# ============================================================================

def evaluate_candidate_on_task(
    wiki: str,
    task_id: int,
    config: RunConfig,
    task_split: str = "test",
    original_traj: Optional[Dict] = None,
    action_similarity_mode: Literal["exact", "name_only", "jaccard"] = "exact",
) -> Tuple[int, float, Optional[float], List[Dict], Dict]:
    """
    Evaluate a wiki on a single task.

    Args:
        wiki: The wiki/system prompt to evaluate
        task_id: Task ID to run
        config: RunConfig
        task_split: Task split
        original_traj: Original trajectory for action comparison (optional)
        action_similarity_mode: How to compare action sequences

    Returns:
        (task_id, reward, action_similarity, messages, info)
    """
    try:
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=task_split,
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

        # Calculate action similarity if original trajectory provided
        action_sim = None
        if original_traj is not None:
            original_actions = get_ground_truth_actions(original_traj)
            new_actions = extract_tool_calls(result.messages)
            action_sim = action_sequence_similarity(
                original_actions, new_actions, mode=action_similarity_mode
            )

        return (
            task_id,
            result.reward,
            action_sim,
            result.messages,
            {"success": result.reward >= 0.99}
        )

    except Exception as e:
        return task_id, 0.0, None, [], {"error": str(e)}


def evaluate_candidate(
    candidate: WikiCandidate,
    config: RunConfig,
    source_trajectories: Optional[Dict[int, Dict]] = None,
    eval_mode: EvalMode = EvalMode.REWARD,
    task_split: str = "test",
    max_tasks: Optional[int] = None,
    action_similarity_mode: Literal["exact", "name_only", "jaccard"] = "exact",
    verbose: bool = True,
) -> WikiCandidate:
    """
    Evaluate a wiki candidate by re-running agent on source tasks.

    This is the RPE sanity check: does the wiki explain the behavior
    it was derived from?

    Args:
        candidate: WikiCandidate to evaluate
        config: RunConfig for agent/env
        source_trajectories: Dict mapping task_id -> original trajectory (for action comparison)
        eval_mode: Which evaluation metric to use
            - REWARD: Final state matching (tau-bench reward)
            - ACTION_SEQUENCE: Action sequence similarity
            - BOTH: Calculate both metrics
        task_split: Which task split to use
        max_tasks: Limit evaluation to first N tasks (for speed)
        action_similarity_mode: How to compare action sequences
        verbose: Print progress

    Returns:
        Updated WikiCandidate with scores filled in
    """
    task_ids = candidate.source_task_ids
    if max_tasks:
        task_ids = task_ids[:max_tasks]

    reward_scores = {}
    action_scores = {}
    trajectories = {}

    for task_id in task_ids:
        if verbose:
            print(f"      Task {task_id}...", end=" ")

        # Get original trajectory if available
        original_traj = None
        if source_trajectories and task_id in source_trajectories:
            original_traj = source_trajectories[task_id]

        # Run evaluation
        _, reward, action_sim, messages, _ = evaluate_candidate_on_task(
            wiki=candidate.wiki,
            task_id=task_id,
            config=config,
            task_split=task_split,
            original_traj=original_traj if eval_mode in [EvalMode.ACTION_SEQUENCE, EvalMode.BOTH] else None,
            action_similarity_mode=action_similarity_mode,
        )

        reward_scores[task_id] = reward
        trajectories[task_id] = messages

        if action_sim is not None:
            action_scores[task_id] = action_sim

        if verbose:
            status_parts = []
            if eval_mode in [EvalMode.REWARD, EvalMode.BOTH]:
                status_parts.append(f"reward={'OK' if reward >= 0.99 else 'FAIL'}")
            if eval_mode in [EvalMode.ACTION_SEQUENCE, EvalMode.BOTH] and action_sim is not None:
                status_parts.append(f"action_sim={action_sim:.2f}")
            print(" | ".join(status_parts))

    # Store results based on eval mode
    candidate.source_task_scores = reward_scores
    candidate.source_task_success_rate = sum(reward_scores.values()) / len(reward_scores) if reward_scores else 0.0
    candidate.generated_trajectories = trajectories

    if action_scores:
        candidate.action_similarity_scores = action_scores
        candidate.mean_action_similarity = sum(action_scores.values()) / len(action_scores)

    return candidate


def evaluate_all_candidates(
    candidates: List[WikiCandidate],
    config: RunConfig,
    source_trajectories: Optional[Dict[int, Dict]] = None,
    eval_mode: EvalMode = EvalMode.REWARD,
    task_split: str = "test",
    max_tasks_per_candidate: Optional[int] = None,
    action_similarity_mode: Literal["exact", "name_only", "jaccard"] = "exact",
    verbose: bool = True,
) -> List[WikiCandidate]:
    """
    Evaluate all candidates on their source tasks.

    Args:
        candidates: List of WikiCandidate to evaluate
        config: RunConfig for agent/env
        source_trajectories: Dict mapping task_id -> original trajectory
        eval_mode: Evaluation mode (REWARD, ACTION_SEQUENCE, or BOTH)
        task_split: Task split to use
        max_tasks_per_candidate: Limit tasks per candidate
        action_similarity_mode: How to compare actions
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATING {len(candidates)} CANDIDATES")
        print(f"  Mode: {eval_mode.value}")
        print(f"{'='*60}")

    for i, candidate in enumerate(candidates):
        if verbose:
            print(f"\n  Candidate {i+1}/{len(candidates)} (ordering={candidate.ordering_idx}):")

        evaluate_candidate(
            candidate=candidate,
            config=config,
            source_trajectories=source_trajectories,
            eval_mode=eval_mode,
            task_split=task_split,
            max_tasks=max_tasks_per_candidate,
            action_similarity_mode=action_similarity_mode,
            verbose=verbose,
        )

        if verbose:
            print(f"    Success rate: {candidate.source_task_success_rate:.1%}", end="")
            if candidate.mean_action_similarity is not None:
                print(f" | Action similarity: {candidate.mean_action_similarity:.2f}", end="")
            print()

    return candidates


# ============================================================================
# Candidate Selection
# ============================================================================

def select_best_candidate(
    candidates: List[WikiCandidate],
    selection_mode: EvalMode = EvalMode.REWARD,
    verbose: bool = True,
) -> WikiCandidate:
    """
    Select the best candidate based on the specified metric.

    Args:
        candidates: List of evaluated WikiCandidate objects
        selection_mode: Which metric to use for selection
            - REWARD: Select by success rate
            - ACTION_SEQUENCE: Select by action similarity
            - BOTH: Select by combined score (average of both)
        verbose: Print ranking
    """
    def get_score(c: WikiCandidate) -> float:
        if selection_mode == EvalMode.REWARD:
            return c.source_task_success_rate or 0.0
        elif selection_mode == EvalMode.ACTION_SEQUENCE:
            return c.mean_action_similarity or 0.0
        else:  # BOTH
            reward = c.source_task_success_rate or 0.0
            action = c.mean_action_similarity or 0.0
            # If action similarity not available, just use reward
            if c.mean_action_similarity is None:
                return reward
            return (reward + action) / 2

    # Sort by score descending
    sorted_candidates = sorted(candidates, key=get_score, reverse=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"CANDIDATE RANKING (by {selection_mode.value})")
        print(f"{'='*60}")
        for i, c in enumerate(sorted_candidates):
            score_str = f"{get_score(c):.2f}"
            detail_parts = [f"reward={c.source_task_success_rate:.1%}"]
            if c.mean_action_similarity is not None:
                detail_parts.append(f"action={c.mean_action_similarity:.2f}")
            print(f"  {i+1}. Ordering {c.ordering_idx}, Candidate {c.candidate_idx}: "
                  f"score={score_str} ({', '.join(detail_parts)})")

    return sorted_candidates[0]


# ============================================================================
# Ordering Stability Analysis
# ============================================================================

def _safe_mean(values: List[float]) -> float:
    """Calculate mean, returning 0.0 for empty lists"""
    return mean(values) if values else 0.0


def _safe_stdev(values: List[float]) -> float:
    """Calculate stdev, returning 0.0 for lists with < 2 elements"""
    return stdev(values) if len(values) >= 2 else 0.0


def analyze_ordering_stability(
    candidates: List[WikiCandidate],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze whether wikis are stable across different trajectory orderings.

    Kilian's feedback: check that updates are stable under reordering.
    """
    # Group candidates by ordering
    by_ordering = defaultdict(list)
    for c in candidates:
        by_ordering[c.ordering_idx].append(c)

    # Calculate stats per ordering
    ordering_stats = {}
    for ordering_idx, order_candidates in by_ordering.items():
        scores = [c.source_task_success_rate or 0.0 for c in order_candidates]
        ordering_stats[ordering_idx] = {
            "mean_success_rate": _safe_mean(scores),
            "std_success_rate": _safe_stdev(scores),
            "num_candidates": len(order_candidates),
            "wiki_lengths": [len(c.wiki) for c in order_candidates],
        }

    # Overall stability metrics
    all_scores = [c.source_task_success_rate or 0.0 for c in candidates]
    mean_scores_per_ordering = [s["mean_success_rate"] for s in ordering_stats.values()]

    cross_ordering_std = _safe_stdev(mean_scores_per_ordering)

    stability = {
        "ordering_stats": ordering_stats,
        "overall_mean": _safe_mean(all_scores),
        "overall_std": _safe_stdev(all_scores),
        "cross_ordering_std": cross_ordering_std,
        "is_stable": cross_ordering_std < 0.1,  # <10% variance
    }

    if verbose:
        print(f"\n{'='*60}")
        print("ORDERING STABILITY ANALYSIS")
        print(f"{'='*60}")
        print(f"  Overall mean success rate: {stability['overall_mean']:.1%}")
        print(f"  Overall std: {stability['overall_std']:.1%}")
        print(f"  Cross-ordering std: {stability['cross_ordering_std']:.1%}")
        print(f"  Stable: {'Yes' if stability['is_stable'] else 'No (>10% variance)'}")
        print(f"\n  Per-ordering breakdown:")
        for ordering_idx, stats in ordering_stats.items():
            print(f"    Ordering {ordering_idx}: {stats['mean_success_rate']:.1%} "
                  f"(std={stats['std_success_rate']:.1%})")

    return stability


# ============================================================================
# Main Pipeline
# ============================================================================

def run_rpe_wiki_induction(
    trajectories_path: str,
    config: RunConfig,
    tool_filter: Optional[str] = None,
    num_orderings: int = 3,
    candidates_per_ordering: int = 2,
    batch_size: int = 3,
    max_trajectories: int = 20,
    max_eval_tasks: int = 10,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    eval_mode: EvalMode = EvalMode.REWARD,
    action_similarity_mode: Literal["exact", "name_only", "jaccard"] = "exact",
    task_split: str = "test",
    seed: int = 42,
    verbose: bool = True,
) -> RPEResult:
    """
    Full RPE-style wiki induction pipeline.

    Args:
        trajectories_path: Path to trajectories JSON
        config: RunConfig for evaluation
        tool_filter: Filter trajectories by tool name
        num_orderings: Number of trajectory orderings to try
        candidates_per_ordering: Candidates per ordering
        batch_size: Conversations per batch in generation
        max_trajectories: Max trajectories to use
        max_eval_tasks: Max tasks to evaluate per candidate
        model: LLM for generation
        temperature: Sampling temperature
        eval_mode: Evaluation mode
            - REWARD: Final state matching (tau-bench default)
            - ACTION_SEQUENCE: Action sequence similarity
            - BOTH: Calculate both metrics
        action_similarity_mode: How to compare action sequences
            - "exact": Match (name, args) pairs in order
            - "name_only": Match tool names only
            - "jaccard": Set-based similarity
        task_split: Task split for evaluation
        seed: Random seed
        verbose: Print progress

    Returns:
        RPEResult with best candidate and analysis
    """

    if verbose:
        print(f"\n{'='*60}")
        print("RPE-STYLE WIKI INDUCTION")
        print(f"{'='*60}")
        print(f"  Trajectories: {trajectories_path}")
        print(f"  Tool filter: {tool_filter or 'None'}")
        print(f"  Orderings: {num_orderings}")
        print(f"  Candidates per ordering: {candidates_per_ordering}")
        print(f"  Eval mode: {eval_mode.value}")
        if eval_mode in [EvalMode.ACTION_SEQUENCE, EvalMode.BOTH]:
            print(f"  Action similarity mode: {action_similarity_mode}")

    # Step 1: Load trajectories
    if verbose:
        print(f"\n[1/5] Loading trajectories...")

    trajectories = load_trajectories_from_file(
        trajectories_path,
        success_only=True,
        tool_filter=tool_filter,
    )

    if len(trajectories) > max_trajectories:
        random.seed(seed)
        trajectories = random.sample(trajectories, max_trajectories)

    if verbose:
        print(f"  Loaded {len(trajectories)} successful trajectories")

    # Build source trajectories dict for action comparison
    source_trajectories = {t['task_id']: t for t in trajectories}

    # Get tools summary from environment
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        task_split=task_split,
        user_provider=config.user_model_provider,
    )
    tools_summary = ", ".join([
        t.get('function', {}).get('name', 'unknown')
        for t in env.tools_info
    ])

    # Step 2: Generate candidates with ordering variation
    if verbose:
        print(f"\n[2/5] Generating candidates...")

    candidates = generate_candidates_with_ordering(
        trajectories=trajectories,
        tools_summary=tools_summary,
        num_orderings=num_orderings,
        candidates_per_ordering=candidates_per_ordering,
        batch_size=batch_size,
        model=model,
        temperature=temperature,
        seed=seed,
        verbose=verbose,
    )

    # Step 3: Evaluate candidates on source tasks
    if verbose:
        print(f"\n[3/5] Evaluating candidates on source tasks...")

    candidates = evaluate_all_candidates(
        candidates=candidates,
        config=config,
        source_trajectories=source_trajectories,
        eval_mode=eval_mode,
        task_split=task_split,
        max_tasks_per_candidate=max_eval_tasks,
        action_similarity_mode=action_similarity_mode,
        verbose=verbose,
    )

    # Step 4: Analyze ordering stability
    if verbose:
        print(f"\n[4/5] Analyzing ordering stability...")

    stability = analyze_ordering_stability(candidates, verbose=verbose)

    # Step 5: Select best candidate
    if verbose:
        print(f"\n[5/5] Selecting best candidate...")

    best = select_best_candidate(candidates, selection_mode=eval_mode, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print("BEST WIKI")
        print(f"{'='*60}")
        print(best.wiki[:1000] + "..." if len(best.wiki) > 1000 else best.wiki)

    return RPEResult(
        best_candidate=best,
        all_candidates=candidates,
        ordering_stability=stability,
        evaluation_results={
            "num_candidates": len(candidates),
            "best_success_rate": best.source_task_success_rate,
            "best_action_similarity": best.mean_action_similarity,
            "eval_mode": eval_mode.value,
        }
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RPE-style wiki induction")
    parser.add_argument("--trajectories", type=str,
                       default="historical_trajectories/gpt-4o-retail.json",
                       help="Path to trajectories JSON")
    parser.add_argument("--tool-filter", type=str, default=None,
                       help="Filter trajectories by tool name")
    parser.add_argument("--num-orderings", type=int, default=3,
                       help="Number of trajectory orderings")
    parser.add_argument("--candidates-per-ordering", type=int, default=2,
                       help="Candidates per ordering")
    parser.add_argument("--max-trajectories", type=int, default=15,
                       help="Max trajectories to use")
    parser.add_argument("--max-eval-tasks", type=int, default=5,
                       help="Max tasks to evaluate per candidate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model for generation")
    parser.add_argument("--agent-model", type=str, default="gpt-4o-mini",
                       help="Model for agent evaluation")
    parser.add_argument("--eval-mode", type=str, default="reward",
                       choices=["reward", "action", "both"],
                       help="Evaluation mode: reward (final state), action (sequence similarity), both")
    parser.add_argument("--action-similarity-mode", type=str, default="exact",
                       choices=["exact", "name_only", "jaccard"],
                       help="How to compare action sequences")
    parser.add_argument("--output", type=str, default="rpe_wiki_result.json",
                       help="Output file for results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Map eval mode string to enum
    eval_mode_map = {
        "reward": EvalMode.REWARD,
        "action": EvalMode.ACTION_SEQUENCE,
        "both": EvalMode.BOTH,
    }
    eval_mode = eval_mode_map[args.eval_mode]

    # Configure
    config = RunConfig(
        env="retail",
        model=args.agent_model,
        model_provider="openai",
        user_model="gpt-4o-mini",
        user_model_provider="openai",
        agent_strategy="tool-calling",
        task_split="test",
        user_strategy="llm",
        temperature=0.0,
        seed=args.seed,
        start_index=0,
        end_index=-1,
        log_dir="./logs",
        max_concurrency=1,
        shuffle=False,
        num_trials=1,
        task_ids=[],
        few_shot_displays_path=None,
    )

    # Run pipeline
    result = run_rpe_wiki_induction(
        trajectories_path=args.trajectories,
        config=config,
        tool_filter=args.tool_filter,
        num_orderings=args.num_orderings,
        candidates_per_ordering=args.candidates_per_ordering,
        max_trajectories=args.max_trajectories,
        max_eval_tasks=args.max_eval_tasks,
        model=args.model,
        eval_mode=eval_mode,
        action_similarity_mode=args.action_similarity_mode,
        seed=args.seed,
        verbose=True,
    )

    # Save results
    output_data = {
        "best_wiki": result.best_candidate.wiki,
        "best_success_rate": result.best_candidate.source_task_success_rate,
        "best_action_similarity": result.best_candidate.mean_action_similarity,
        "best_source_task_ids": result.best_candidate.source_task_ids,
        "eval_mode": args.eval_mode,
        "action_similarity_mode": args.action_similarity_mode,
        "ordering_stability": {
            "is_stable": result.ordering_stability["is_stable"],
            "cross_ordering_std": result.ordering_stability["cross_ordering_std"],
        },
        "all_candidates": [
            {
                "ordering_idx": c.ordering_idx,
                "candidate_idx": c.candidate_idx,
                "success_rate": c.source_task_success_rate,
                "action_similarity": c.mean_action_similarity,
                "wiki_length": len(c.wiki),
            }
            for c in result.all_candidates
        ]
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")
