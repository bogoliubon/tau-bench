"""
RPE-EGA Style Iterative Wiki Optimization for Tau-Bench

Implements true RPE Figure 6 loop:
1. Maintain population of K candidate prompts
2. Each iteration:
   - Sample trajectories (balanced across workflow types)
   - Run agent with each candidate → get new outputs A'
   - Compare A' to reference outputs A → summarize diffs
   - Mutate each candidate based on diffs → get K new candidates
   - Select best K from 2K (parents + children)
3. Repeat for k iterations

Key features:
- Stratified sampling: equal representation of cancel/exchange/modify/return
- Failure-driven mutation: diffs guide improvements
- Diversity preservation: prevent convergence to single solution
- Guardrails: constrained mutations to prevent drift
"""

import sys
from pathlib import Path
# Add parent directory to path so tau_bench can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

import openai
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import RunConfig

# ============================================================================
# TauBench RPE Implementation
# ============================================================================

class TauBenchRPE:
    """
    RPE (Reverse Prompt Engineering) implementation for tau-bench.

    Based on the algorithm from RPE_ga.py but adapted for trajectory-based optimization:
    - Uses trajectory diffs instead of answer diffs
    - Evaluation uses task success rate, not ROUGE
    - Uses call_llm for API calls (handles GPT-5 compatibility)
    """

    def __init__(self, model: str = "gpt-4o"):
        """Initialize without spacy/rouge (we don't need them)."""
        self.llm = openai.OpenAI()
        self.model = model
        self.answers = []  # Reference trajectories (as text)
        self.candidate_prompts = []  # Wiki/policies
        self.candidate_answers = []  # Candidate trajectories (as text)
        self.candidate_scores = []

    def chat(self, messages, model=None, n=1, t=0.5):
        """
        Override chat to use our call_llm logic that handles GPT-5.
        """
        if model is None:
            model = self.model

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if role == "system":
                prompt_parts.append(f"[System]: {content}")
            else:
                prompt_parts.append(content)
        full_prompt = "\n\n".join(prompt_parts)

        # Use call_llm which handles GPT-5 compatibility
        responses = []
        for _ in range(n):
            response = call_llm(full_prompt, model=model, temperature=t, max_tokens=2000)
            responses.append(response)
        return responses

    def mutate_policy(
        self,
        old_policy: str,
        trajectory_pairs: List[Tuple[str, str, int]],  # (candidate_text, reference_text, task_id)
    ) -> str:
        """
        Mutate a policy using the RPE 4-step cross_mutation process.

        This is the core algorithm from RPE_ga.py cross_mutation():
        1. For each task, get difference between candidate and reference (PER-TASK)
        2. Summarize all differences
        3. Generate advice on how to revise
        4. Revise the policy

        Args:
            old_policy: Current wiki/policy text
            trajectory_pairs: List of (candidate_text, reference_text, task_id) tuples

        Returns:
            New revised policy
        """
        suggest_list = []

        # Step 1: Get difference for EACH TASK (candidate_i vs reference_i)
        for cand_traj, ref_traj, task_id in trajectory_pairs:
            prompt_e = ("You should answer concisely with no redundant information. "
                       "Given two agent trajectories for the SAME task below, how is the candidate trajectory different from "
                       "the reference trajectory? If there is no significant difference, you can answer 'There is no difference'.")
            message_e = [
                {"role": "user", "content": prompt_e},
                {"role": "user", "content": f"Task {task_id} - Candidate Trajectory:\n" + cand_traj},
                {"role": "user", "content": f"Task {task_id} - Reference Trajectory:\n" + ref_traj}
            ]
            suggest_e = self.chat(message_e)
            suggest_list.append(f"Task {task_id}: {suggest_e[0]}")

        # Step 2: Summarize all differences (RPE lines 224-228)
        prompt_s = "Given a list of responses, summarize them into one concise response."
        message_s = [
            {"role": "user", "content": prompt_s},
            {"role": "user", "content": "List of responses: " + str(suggest_list)}
        ]
        summary_s = self.chat(message_s)[0]

        # Step 3: Generate advice to revise policy (RPE lines 230-235)
        prompt_p = ("According to the difference above, how should you change the agent policy that "
                   "generated the candidate trajectory to make the candidate trajectory more similar to the reference trajectory? Answer concisely.")
        message_p = [
            {"role": "user", "content": "Difference: " + summary_s},
            {"role": "user", "content": prompt_p}
        ]
        advice = self.chat(message_p)[0]

        # Step 4: Revise the policy (RPE lines 237-249)
        message_r = [
            {"role": "user", "content": f"""You are a policy revision assistant. Your ONLY job is to output a revised policy document.

Suggestion: {advice}

Current Policy: {old_policy}

INSTRUCTIONS:
1. Output the COMPLETE revised policy wrapped in <POLICY>...</POLICY> tags
2. DO NOT ask questions, request clarification, or say you need more information
3. DO NOT include any conversational text before or after the policy
4. Just output <POLICY>revised policy here</POLICY> and nothing else

<POLICY>"""}
        ]
        child_r = self.chat(message_r)[0]

        # Since we primed with <POLICY>, response should start with content
        if '</POLICY>' in child_r:
            return child_r.split('</POLICY>')[0].strip()

        # Fallback: try the old parsing method
        splited_child = re.split(r'<POLICY>|</POLICY>', child_r)
        if len(splited_child) >= 3:
            return splited_child[1].strip()

        # Last resort: return the whole response
        return child_r.strip()


# ============================================================================
# Data Structures
# ============================================================================

# ============================================================================
# Beta-Binomial Hyperparameters
# ============================================================================

# Prior pseudo-counts (based on ~70% baseline success rate, 12 pseudo-trials)
ALPHA_0 = 8.0  # prior successes
BETA_0 = 4.0   # prior failures

# Inheritance weight for children (higher = more like parent)
INHERITANCE_W = 0.7

# LCB z-score (1.0 = mild, 1.64 = 90% bound)
LCB_Z = 1.0

# Sanity threshold - if current batch < this, force penalty
BATCH_SANITY_THRESHOLD = 0.5


@dataclass
class CandidatePrompt:
    """A candidate wiki/policy prompt"""
    wiki: str
    id: int
    generation: int  # Which iteration it was created
    parent_id: Optional[int] = None  # ID of parent (if mutated)

    # Beta-Binomial tracking (pseudo-counts)
    alpha: float = ALPHA_0  # successes + prior
    beta: float = BETA_0    # failures + prior

    # History
    mutation_history: List[str] = field(default_factory=list)

    @property
    def posterior_mean(self) -> float:
        """μ = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def lcb_score(self) -> float:
        """Lower Confidence Bound: μ - z * sqrt(μ(1-μ) / (α+β+1))"""
        mu = self.posterior_mean
        n = self.alpha + self.beta
        std_err = (mu * (1 - mu) / (n + 1)) ** 0.5
        return mu - LCB_Z * std_err

    @property
    def total_trials(self) -> float:
        """Total pseudo-trials (α + β)"""
        return self.alpha + self.beta

    @property
    def raw_successes(self) -> float:
        """Actual successes (excluding prior)"""
        return self.alpha - ALPHA_0

    @property
    def raw_trials(self) -> float:
        """Actual trials (excluding prior)"""
        return (self.alpha - ALPHA_0) + (self.beta - BETA_0)

    @property
    def raw_score(self) -> float:
        """Raw success rate (excluding prior). Returns 0 if no trials yet."""
        if self.raw_trials <= 0:
            return 0.0
        return self.raw_successes / self.raw_trials

    def update(self, successes: int, failures: int):
        """Update alpha/beta with new evidence."""
        self.alpha += successes
        self.beta += failures


@dataclass
class TrajectoryResult:
    """Result of running agent on a task"""
    task_id: int
    workflow_type: str
    success: bool
    reward: float
    actions_taken: List[Dict]  # Tool calls made
    conversation: List[Dict]  # Full trajectory


@dataclass
class DiffSummary:
    """Summary of differences between candidate output and reference"""
    task_id: int
    workflow_type: str
    reference_actions: List[str]
    candidate_actions: List[str]
    reference_success: bool
    candidate_success: bool
    key_differences: str  # LLM-generated summary


# ============================================================================
# Workflow Classification
# ============================================================================

WORKFLOW_TOOLS = {
    "cancel": ["cancel_pending_order"],
    "exchange": ["exchange_delivered_order_items"],
    "return": ["return_delivered_order_items"],
    "modify": ["modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"],
}


def classify_workflows(trajectory: Dict) -> List[str]:
    """
    Classify a trajectory by ALL workflow types it contains.
    A trajectory can belong to multiple workflows (non-exclusive).
    """
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


def split_trajectories_by_workflow(trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Split trajectories into workflow categories (non-exclusive).
    A trajectory can appear in multiple workflow groups if it contains multiple workflow types.
    """
    by_workflow = defaultdict(list)

    for traj in trajectories:
        workflows = classify_workflows(traj)
        for workflow in workflows:
            by_workflow[workflow].append(traj)

    return dict(by_workflow)


# ============================================================================
# Data Loading
# ============================================================================

# Pre-split training files (created by split_train_by_workflow.py)
# Paths relative to tudor_experiments/ directory
TRAIN_FILES = {
    "cancel": "data/train_cancel.json",
    "exchange": "data/train_exchange.json",
    "modify": "data/train_modify.json",
    "return": "data/train_return.json",
}


def load_train_data_from_splits(
    train_dir: str = "data",
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Load pre-split training trajectories by workflow type.

    These files are created once by running: python split_train_by_workflow.py

    Args:
        train_dir: Directory containing train_*.json files
        seed: Random seed

    Returns:
        Dict mapping workflow type to list of trajectories
    """
    random.seed(seed)

    by_workflow = {}
    total = 0

    print("Loading pre-split training data:")
    for workflow, filepath in TRAIN_FILES.items():
        with open(filepath, 'r') as f:
            trajs = json.load(f)
        by_workflow[workflow] = trajs
        total += len(trajs)
        print(f"  {workflow}: {len(trajs)} trajectories")

    print(f"Total slots: {total}")
    return by_workflow


def sample_balanced_trajectories(
    train_by_workflow: Dict[str, List[Dict]],
    n_per_workflow: int = 3,
) -> List[Dict]:
    """
    Sample n trajectories from each workflow type, deduplicating by task_id.

    Since tasks can appear in multiple workflows (non-exclusive), we need to
    ensure the same task_id doesn't appear twice in the final sample.
    """
    target_total = n_per_workflow * len(train_by_workflow)  # e.g., 3 * 4 = 12
    sampled = []
    seen_task_ids = set()

    # First pass: sample from each workflow, skipping duplicates
    for workflow, trajs in train_by_workflow.items():
        # Filter out already-seen task_ids
        available = [t for t in trajs if t["task_id"] not in seen_task_ids]
        n = min(n_per_workflow, len(available))
        chosen = random.sample(available, n)
        for t in chosen:
            sampled.append(t)
            seen_task_ids.add(t["task_id"])

    # Second pass: if we're short due to dedup, fill from any workflow
    if len(sampled) < target_total:
        all_trajs = []
        for trajs in train_by_workflow.values():
            all_trajs.extend(trajs)
        available = [t for t in all_trajs if t["task_id"] not in seen_task_ids]
        needed = target_total - len(sampled)
        if available and needed > 0:
            extra = random.sample(available, min(needed, len(available)))
            for t in extra:
                sampled.append(t)
                seen_task_ids.add(t["task_id"])

    random.shuffle(sampled)
    return sampled


# ============================================================================
# LLM Utilities
# ============================================================================

def call_llm(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call LLM and return response text."""
    client = openai.OpenAI()

    # GPT-5/o1/o3 models use max_completion_tokens and don't support custom temperature
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return response.choices[0].message.content


# ============================================================================
# Initial Population Generation
# ============================================================================

def generate_initial_population(
    train_by_workflow: Dict[str, List[Dict]],
    n_candidates: int = 10,
    n_trajectories_per_candidate: int = 12,
    model: str = "gpt-4o",
) -> List[CandidatePrompt]:
    """Generate initial population of candidate prompts."""

    candidates = []

    for i in range(n_candidates):
        print(f"Generating initial candidate {i+1}/{n_candidates}...")

        # Sample different trajectories for each candidate (diversity)
        sampled = sample_balanced_trajectories(
            train_by_workflow,
            n_per_workflow=n_trajectories_per_candidate // 4
        )

        # Format trajectories for prompt
        traj_text = format_trajectories_for_prompt(sampled)

        # Generate wiki - no domain-specific hints, LLM must infer everything from trajectories
        prompt = f"""You are a policy extraction assistant. Your ONLY job is to output a policy document.

Analyze these {len(sampled)} successful agent conversations and extract the agent's policy:

{traj_text}

Extract the rules, guidelines, and strategies the agent follows. Focus on:
1. How the agent gathers information
2. When and how the agent uses tools
3. How the agent communicates with the user
4. Any specific constraints or requirements the agent follows

INSTRUCTIONS:
- Output ONLY the policy document (at least 500 words)
- DO NOT ask questions, request clarification, or say you need more information
- DO NOT include any conversational text before or after the policy
- Just write the policy directly based on the conversations above"""

        wiki = call_llm(prompt, model=model, temperature=0.7)

        candidate = CandidatePrompt(
            wiki=wiki,
            id=i,
            generation=0,
        )
        candidates.append(candidate)

    return candidates


def format_trajectories_for_prompt(trajectories: List[Dict], max_per_traj: int = 1500) -> str:
    """Format trajectories into text for LLM prompt. No domain-specific labels."""
    parts = []

    for i, traj in enumerate(trajectories):
        # No workflow labels - LLM infers from content
        conv_text = format_conversation(traj.get("traj", []))
        if len(conv_text) > max_per_traj:
            conv_text = conv_text[:max_per_traj] + "...[truncated]"

        parts.append(f"=== Conversation {i+1} ===\n{conv_text}")

    return "\n\n".join(parts)


def format_conversation(messages: List[Dict]) -> str:
    """Format conversation messages into readable text."""
    lines = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # Skip system message
        elif role == "user":
            lines.append(f"[User]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    # Truncate long args
                    if len(args) > 200:
                        args = args[:200] + "..."
                    lines.append(f"[Agent] → {name}({args})")
            elif content:
                lines.append(f"[Agent]: {content}")
        elif role == "tool":
            # Truncate tool results
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append(f"[Tool Result]: {content}")

    return "\n".join(lines)


# ============================================================================
# Agent Evaluation
# ============================================================================

def run_agent_on_task(
    wiki: str,
    task_id: int,
    config: RunConfig,
    reference_trajectory: Dict,
) -> TrajectoryResult:
    """Run agent with given wiki on a specific task."""

    workflows = classify_workflows(reference_trajectory)
    workflow = "+".join(workflows) if workflows else "unknown"

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

        # Create agent with candidate wiki
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
                    actions_taken.append({
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", "{}"),
                    })

        return TrajectoryResult(
            task_id=task_id,
            workflow_type=workflow or "unknown",
            success=result.reward > 0,
            reward=result.reward,
            actions_taken=actions_taken,
            conversation=result.messages,
        )

    except Exception as e:
        print(f"    Error on task {task_id}: {e}")
        return TrajectoryResult(
            task_id=task_id,
            workflow_type=workflow or "unknown",
            success=False,
            reward=0.0,
            actions_taken=[],
            conversation=[],
        )


def evaluate_candidate(
    candidate: CandidatePrompt,
    trajectories: List[Dict],
    config: RunConfig,
    max_concurrency: int = 1,
) -> Tuple[List[TrajectoryResult], int, int]:
    """
    Evaluate a candidate on a set of trajectories.

    Args:
        candidate: The candidate prompt to evaluate
        trajectories: List of reference trajectories
        config: RunConfig for agent
        max_concurrency: Number of parallel evaluations (1 = sequential)

    Returns:
        results: List of TrajectoryResult
        successes: Number of successful tasks
        failures: Number of failed tasks
    """

    def _run_single(traj: Dict) -> TrajectoryResult:
        task_id = traj["task_id"]
        return run_agent_on_task(
            wiki=candidate.wiki,
            task_id=task_id,
            config=config,
            reference_trajectory=traj,
        )

    if max_concurrency <= 1:
        # Sequential execution
        results = [_run_single(traj) for traj in trajectories]
    else:
        # Parallel execution with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            results = list(executor.map(_run_single, trajectories))

    # Count successes/failures
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes

    return results, successes, failures


# ============================================================================
# Diff Analysis & Mutation
# ============================================================================

def extract_actions_from_trajectory(trajectory: Dict) -> List[str]:
    """Extract action names from a trajectory."""
    actions = []
    for msg in trajectory.get("traj", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                actions.append(tc["function"]["name"])
    return actions


def extract_ground_truth_actions(trajectory: Dict) -> List[Dict]:
    """Extract ground-truth actions with kwargs from info.task.actions."""
    try:
        actions = trajectory["info"]["task"]["actions"]
        return [{"name": a["name"], "kwargs": a.get("kwargs", {})} for a in actions]
    except (KeyError, TypeError):
        return []


def extract_tool_calls_with_args(messages: List[Dict]) -> List[Dict]:
    """Extract all tool calls with their parsed arguments from trajectory messages."""
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"_raw": args_str}
                tool_calls.append({"name": name, "args": args})
    return tool_calls


def extract_tool_outputs(messages: List[Dict], max_len: int = 200) -> List[Dict]:
    """Extract tool outputs from trajectory messages."""
    outputs = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > max_len:
                content = content[:max_len] + "..."
            outputs.append({
                "tool_name": msg.get("name", "unknown"),
                "output": content,
            })
    return outputs


def format_tool_calls_summary(tool_calls: List[Dict], max_args_len: int = 150) -> str:
    """Format tool calls into a readable summary."""
    lines = []
    for tc in tool_calls:
        args_str = json.dumps(tc["args"], ensure_ascii=False)
        if len(args_str) > max_args_len:
            args_str = args_str[:max_args_len] + "..."
        lines.append(f"  - {tc['name']}({args_str})")
    return "\n".join(lines) if lines else "  (none)"


def format_tool_outputs_summary(outputs: List[Dict], max_outputs: int = 5) -> str:
    """Format tool outputs into a readable summary."""
    lines = []
    for out in outputs[:max_outputs]:
        lines.append(f"  - {out['tool_name']}: {out['output']}")
    if len(outputs) > max_outputs:
        lines.append(f"  ... and {len(outputs) - max_outputs} more")
    return "\n".join(lines) if lines else "  (none)"


def trajectory_to_text(traj_data: Dict, result: Optional[TrajectoryResult] = None) -> str:
    """Convert a trajectory to text representation for comparison.

    Works with either reference trajectory (dict with 'traj') or candidate result.
    """
    if result is not None:
        # Candidate result
        tool_calls = extract_tool_calls_with_args(result.conversation)
        calls_summary = format_tool_calls_summary(tool_calls)
        return f"Tool calls:\n{calls_summary}\nSuccess: {result.success}"
    else:
        # Reference trajectory
        tool_calls = extract_tool_calls_with_args(traj_data.get("traj", []))
        calls_summary = format_tool_calls_summary(tool_calls)
        success = traj_data.get("reward", 0) > 0
        return f"Tool calls:\n{calls_summary}\nSuccess: {success}"


def rpe_get_difference(candidate_answer: str, reference_answer: str, model: str = "gpt-4o") -> str:
    """
    Step 1 of RPE cross_mutation: Get difference between candidate and reference.

    Adapted from RPE_ga.py lines 215-222.
    """
    prompt = ("You should answer concisely with no redundant information. "
              "Given two agent trajectories below, how is the candidate trajectory different from "
              "the reference trajectory? Focus on: tool calls, arguments, order of operations. "
              "If there is no significant difference, answer 'There is no difference'.")

    full_prompt = f"""{prompt}

Candidate Trajectory:
{candidate_answer}

Reference Trajectory:
{reference_answer}"""

    return call_llm(full_prompt, model=model, temperature=0.3, max_tokens=300)


def rpe_summarize_differences(suggest_list: List[str], model: str = "gpt-4o") -> str:
    """
    Step 2 of RPE cross_mutation: Summarize all differences into one.

    Adapted from RPE_ga.py lines 224-228.
    """
    prompt = "Given a list of observations about trajectory differences, summarize them into one concise response."

    full_prompt = f"""{prompt}

List of observations:
{json.dumps(suggest_list, indent=2)}"""

    return call_llm(full_prompt, model=model, temperature=0.3, max_tokens=400)


def rpe_get_revision_advice(difference_summary: str, model: str = "gpt-4o") -> str:
    """
    Step 3 of RPE cross_mutation: Generate advice on how to revise the policy.

    Adapted from RPE_ga.py lines 230-235.
    """
    prompt = ("According to the differences above, how should you change the agent policy "
              "to make the candidate trajectory more similar to the reference trajectory? "
              "Answer concisely with specific, actionable rules.")

    full_prompt = f"""Difference Summary:
{difference_summary}

{prompt}"""

    return call_llm(full_prompt, model=model, temperature=0.3, max_tokens=400)


def rpe_revise_policy(suggestion: str, old_policy: str, max_tokens: int = 3000, model: str = "gpt-4o") -> str:
    """
    Step 4 of RPE cross_mutation: Revise the policy using the suggestion.

    Adapted from RPE_ga.py lines 237-249.
    """
    prompt = f"""You are a policy revision assistant. Your ONLY job is to output a revised policy document.

TASK: Revise the policy below using the given suggestion.

Suggestion:
{suggestion}

Current Policy:
{old_policy}

INSTRUCTIONS:
1. Output the COMPLETE revised policy wrapped in <POLICY>...</POLICY> tags
2. Keep it under {max_tokens} tokens
3. DO NOT ask questions, request clarification, or say you need more information
4. DO NOT include any conversational text before or after the policy
5. Just output <POLICY>revised policy here</POLICY> and nothing else

<POLICY>"""

    response = call_llm(prompt, model=model, temperature=0.5, max_tokens=max_tokens)

    # Since we primed with <POLICY>, response should start with content and end with </POLICY>
    # Try to extract up to </POLICY>
    if '</POLICY>' in response:
        return response.split('</POLICY>')[0].strip()

    # Fallback: try the old parsing method
    parts = re.split(r'<POLICY>|</POLICY>', response)
    if len(parts) >= 3:
        return parts[1].strip()

    # Last resort: return the whole response
    return response.strip()


def mutate_candidate(
    candidate: CandidatePrompt,
    references: List[Dict],
    results: List[TrajectoryResult],
    model: str = "gpt-4o",
    generation: int = 1,
    parent_alpha: float = None,  # Pre-batch alpha (to avoid leakage)
    parent_beta: float = None,   # Pre-batch beta (to avoid leakage)
) -> CandidatePrompt:
    """
    Mutate a candidate using TauBenchRPE (which inherits from RPE_ga.py).

    Uses the RPE 4-step cross_mutation process via the TauBenchRPE class.

    Args:
        parent_alpha/beta: Pre-batch values to avoid inheritance leakage.
                          If None, uses candidate.alpha/beta (legacy behavior).
    """
    # Use pre-batch alpha/beta if provided (avoids leakage)
    base_alpha = parent_alpha if parent_alpha is not None else candidate.alpha
    base_beta = parent_beta if parent_beta is not None else candidate.beta

    # Child inherits damped α, β from parent (Bayesian prior)
    w = INHERITANCE_W
    child_alpha = w * base_alpha + (1 - w) * ALPHA_0
    child_beta = w * base_beta + (1 - w) * BETA_0

    # Count failures
    n_failures = sum(1 for r in results if not r.success)

    # If 0 failures, skip mutation - just return parent wiki unchanged
    if n_failures == 0:
        print(f"      No failures - keeping parent wiki unchanged")
        return CandidatePrompt(
            wiki=candidate.wiki,  # Keep parent wiki
            id=candidate.id + 1000 * generation,
            generation=generation,
            parent_id=candidate.id,
            alpha=child_alpha,
            beta=child_beta,
            mutation_history=candidate.mutation_history + [f"Gen {generation}: No failures, kept unchanged"],
        )

    # Build per-task trajectory pairs: (candidate_text, reference_text, task_id)
    trajectory_pairs = []
    for ref, result in zip(references, results):
        ref_text = trajectory_to_text(ref)
        cand_text = trajectory_to_text(None, result)
        task_id = result.task_id
        trajectory_pairs.append((cand_text, ref_text, task_id))

    # Use TauBenchRPE to mutate (uses RPE's 4-step process)
    rpe = TauBenchRPE(model=model)
    print(f"      Running RPE 4-step mutation ({n_failures} failures)...")

    new_wiki = rpe.mutate_policy(
        old_policy=candidate.wiki,
        trajectory_pairs=trajectory_pairs,
    )

    # Validation: if new_wiki is garbage, fall back to parent
    if (not new_wiki or
        len(new_wiki.strip()) < 100 or
        "I don't have" in new_wiki or
        "I don't have" in new_wiki or
        "please provide" in new_wiki.lower()):
        print(f"      WARNING: Invalid wiki returned, keeping parent")
        print(f"      Raw response (first 300 chars): {new_wiki[:300] if new_wiki else '(empty)'}...")
        new_wiki = candidate.wiki

    return CandidatePrompt(
        wiki=new_wiki,
        id=candidate.id + 1000 * generation,
        generation=generation,
        parent_id=candidate.id,
        alpha=child_alpha,
        beta=child_beta,
        mutation_history=candidate.mutation_history + [f"Gen {generation}: RPE mutation ({n_failures} failures)"],
    )


# ============================================================================
# Selection
# ============================================================================

def select_top_k(
    candidates: List[CandidatePrompt],
    k: int = 10,
    diversity_weight: float = 0.1,
) -> List[CandidatePrompt]:
    """Select top k candidates using LCB (Lower Confidence Bound) ranking."""

    # Sort by LCB score (penalizes low-sample candidates)
    sorted_candidates = sorted(candidates, key=lambda c: c.lcb_score, reverse=True)

    if diversity_weight == 0:
        return sorted_candidates[:k]

    # Simple diversity: don't take too many from same parent
    selected = []
    parent_counts = defaultdict(int)
    max_per_parent = max(2, k // 3)

    for c in sorted_candidates:
        parent = c.parent_id if c.parent_id is not None else c.id
        if parent_counts[parent] < max_per_parent:
            selected.append(c)
            parent_counts[parent] += 1

        if len(selected) >= k:
            break

    # Fill remaining with best available
    if len(selected) < k:
        for c in sorted_candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= k:
                break

    return selected[:k]


# ============================================================================
# Main RPE-EGA Loop
# ============================================================================

def rpe_ega_optimize(
    n_iterations: int = 5,
    n_candidates: int = 10,
    n_per_workflow: int = 3,  # 3 per workflow = 12 total per iteration
    model: str = "gpt-4o",
    gen_model: Optional[str] = None,  # Model for generation/mutation (defaults to model)
    output_path: Optional[str] = None,
    seed: int = 42,
    max_concurrency: int = 1,
) -> Dict[str, Any]:
    """
    Run RPE-EGA optimization loop.

    Uses pre-split training data from results/train_{workflow}.json files.
    Create these files by running: python split_train_by_workflow.py

    Args:
        n_iterations: Number of evolution iterations
        n_candidates: Population size to maintain
        n_per_workflow: Trajectories to sample per workflow type each iteration
        model: LLM model to use for agent evaluation
        gen_model: LLM model to use for wiki generation/mutation (defaults to model)
        output_path: Where to save results
        seed: Random seed
        max_concurrency: Number of parallel task evaluations (1 = sequential)

    Returns:
        Dict with optimization results (all final candidates with scores)
    """
    # Default gen_model to model if not specified
    if gen_model is None:
        gen_model = model
    random.seed(seed)

    # Setup detailed run log file (derive from output_path if provided)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path:
        # Derive run log path from output path (e.g., foo.json -> foo_runlog.json)
        base = output_path.rsplit('.json', 1)[0]
        run_log_path = f"{base}_runlog.json"
    else:
        run_log_path = f"learned_policies/rpe_run_log_{timestamp}.json"

    # Initialize run log structure
    run_log = {
        "started_at": datetime.now().isoformat(),
        "config": {
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_per_workflow": n_per_workflow,
            "model": model,
            "gen_model": gen_model,
            "seed": seed,
            "max_concurrency": max_concurrency,
        },
        "initial_population": [],
        "iterations": [],
    }

    def save_run_log():
        """Save run log to file (called incrementally)."""
        with open(run_log_path, 'w') as f:
            json.dump(run_log, f, indent=2)

    # Config for agent runs on TRAIN tasks
    train_config = RunConfig(
        env="retail",
        model=model,  # Use the passed eval model (not hardcoded)
        model_provider="openai",
        user_model="gpt-4o",
        user_model_provider="openai",
        agent_strategy="tool-calling",
        task_split="train",  # Train split for iteration
        user_strategy="llm",
        temperature=0.0,
        seed=seed,
        start_index=0,
        end_index=-1,
        log_dir="./rpe_logs",
        max_concurrency=1,
        shuffle=False,
        num_trials=1,
        task_ids=[],
    )

    print("=" * 70)
    print("RPE-EGA WIKI OPTIMIZATION")
    print("=" * 70)

    # Load pre-split train data (from results/train_{workflow}.json)
    print("\n[1/3] Loading pre-split training data...")
    train_by_workflow = load_train_data_from_splits(seed=seed)

    # Generate initial population
    print(f"\n[2/3] Generating initial population of {n_candidates} candidates (using {gen_model})...")
    population = generate_initial_population(
        train_by_workflow,
        n_candidates=n_candidates,
        n_trajectories_per_candidate=n_per_workflow * 4,
        model=gen_model,
    )

    # Log initial population
    for c in population:
        run_log["initial_population"].append({
            "id": c.id,
            "wiki": c.wiki,
        })
    save_run_log()
    print(f"Run log: {run_log_path}")

    # Track history
    history = {
        "iterations": [],
        "best_scores": [],
        "mean_scores": [],
    }

    # Main loop
    print(f"\n[3/3] Running {n_iterations} evolution iterations...")

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # Initialize iteration log
        iter_log = {
            "iteration": iteration + 1,
            "sampled_task_ids": [],
            "parent_evaluations": [],
            "mutations": [],
            "child_evaluations": [],
            "selection": {
                "all_candidates": [],
                "selected_ids": [],
            },
        }

        # Sample trajectories for this iteration
        sampled = sample_balanced_trajectories(train_by_workflow, n_per_workflow)
        n_tasks = len(sampled)
        iter_log["sampled_task_ids"] = [t["task_id"] for t in sampled]
        print(f"Sampled {n_tasks} trajectories (balanced)")

        # Evaluate all candidates on TRAIN tasks
        print(f"Evaluating {len(population)} candidates...")
        all_results = {}
        batch_scores = {}  # Track current batch performance for sanity check
        pre_batch_priors = {}  # Snapshot alpha/beta BEFORE update (to avoid leakage)

        for i, candidate in enumerate(population):
            print(f"  Candidate {i+1}/{len(population)} (id={candidate.id})...", end=" ")

            # Snapshot BEFORE batch update (to avoid inheritance leakage)
            pre_batch_priors[candidate.id] = (candidate.alpha, candidate.beta)

            results, successes, failures = evaluate_candidate(candidate, sampled, train_config, max_concurrency)
            candidate.update(successes, failures)
            all_results[candidate.id] = results
            batch_score = successes / n_tasks
            batch_scores[candidate.id] = batch_score
            print(f"batch={batch_score:.0%} ({successes}/{n_tasks}), cumul={candidate.raw_score:.0%} ({int(candidate.raw_successes)}/{int(candidate.raw_trials)}), LCB={candidate.lcb_score:.1%}")

            # Log parent evaluation
            iter_log["parent_evaluations"].append({
                "id": candidate.id,
                "batch_score": batch_score,
                "batch_successes": successes,
                "batch_failures": failures,
                "cumul_score": candidate.raw_score,
                "cumul_successes": int(candidate.raw_successes),
                "cumul_trials": int(candidate.raw_trials),
                "lcb_score": candidate.lcb_score,
            })

        # Generate mutations using RPE 4-step process
        print(f"Generating mutations (RPE 4-step)...")
        children = []
        mutation_logs = []

        for candidate in population:
            print(f"  Mutating candidate {candidate.id}...")

            # Sanity check: if current batch was terrible, note it
            if batch_scores[candidate.id] < BATCH_SANITY_THRESHOLD:
                print(f"    [!] Below sanity threshold ({batch_scores[candidate.id]:.0%})")

            # Collect results for this candidate (matching sampled trajectories)
            candidate_results = []
            for traj in sampled:
                task_id = traj["task_id"]
                result = next((r for r in all_results[candidate.id] if r.task_id == task_id), None)
                if result:
                    candidate_results.append(result)

            # Get pre-batch priors for this candidate (avoids inheritance leakage)
            parent_alpha, parent_beta = pre_batch_priors[candidate.id]

            # Mutate using RPE 4-step process
            child = mutate_candidate(
                candidate=candidate,
                references=sampled,  # Reference trajectories
                results=candidate_results,  # Candidate's results
                model=gen_model,
                generation=iteration + 1,
                parent_alpha=parent_alpha,  # Pre-batch alpha
                parent_beta=parent_beta,    # Pre-batch beta
            )
            children.append(child)

            # Log mutation
            n_failures = sum(1 for r in candidate_results if not r.success)
            n_successes = sum(1 for r in candidate_results if r.success)
            mutation_logs.append({
                "parent_id": candidate.id,
                "child_id": child.id,
                "parent_wiki": candidate.wiki,
                "child_wiki": child.wiki,
                "n_failures": n_failures,
                "n_successes": n_successes,
            })

        iter_log["mutations"] = mutation_logs

        # Evaluate children on TRAIN tasks
        print(f"Evaluating {len(children)} children...")
        for i, child in enumerate(children):
            print(f"  Child {i+1}/{len(children)}...", end=" ")
            results, successes, failures = evaluate_candidate(child, sampled, train_config, max_concurrency)
            child.update(successes, failures)
            batch_score = successes / n_tasks
            print(f"batch={batch_score:.0%} ({successes}/{n_tasks}), cumul={child.raw_score:.0%} ({int(child.raw_successes)}/{int(child.raw_trials)}), LCB={child.lcb_score:.1%}")

            # Log child evaluation
            iter_log["child_evaluations"].append({
                "id": child.id,
                "parent_id": child.parent_id,
                "batch_score": batch_score,
                "batch_successes": successes,
                "batch_failures": failures,
                "cumul_score": child.raw_score,
                "cumul_successes": int(child.raw_successes),
                "cumul_trials": int(child.raw_trials),
                "lcb_score": child.lcb_score,
            })

        # Select top k from parents + children using LCB
        all_candidates = population + children

        # Log all candidates before selection
        for c in all_candidates:
            iter_log["selection"]["all_candidates"].append({
                "id": c.id,
                "generation": c.generation,
                "parent_id": c.parent_id,
                "lcb_score": c.lcb_score,
                "cumul_score": c.raw_score,
            })

        population = select_top_k(all_candidates, k=n_candidates)
        selected_ids = set(c.id for c in population)
        iter_log["selection"]["selected_ids"] = list(selected_ids)

        # Mark which candidates were selected in the log
        for entry in iter_log["selection"]["all_candidates"]:
            entry["selected"] = entry["id"] in selected_ids

        # Track stats (now using posterior_mean and lcb_score)
        best_lcb = max(c.lcb_score for c in population)
        mean_lcb = sum(c.lcb_score for c in population) / len(population)
        best_mu = max(c.posterior_mean for c in population)

        history["iterations"].append(iteration + 1)
        history["best_scores"].append(best_mu)
        history["mean_scores"].append(mean_lcb)

        # Add summary to iteration log
        iter_log["summary"] = {
            "best_lcb": best_lcb,
            "mean_lcb": mean_lcb,
            "best_mu": best_mu,
            "selected_population_ids": [c.id for c in population],
        }

        # Save iteration to run log
        run_log["iterations"].append(iter_log)
        save_run_log()

        print(f"\nIteration {iteration + 1} summary:")
        print(f"  Best μ: {best_mu:.2%}, Best LCB: {best_lcb:.2%}")
        print(f"  Mean LCB: {mean_lcb:.2%}")
        print(f"  Population: {[c.id for c in population]}")

    # Sort final population by LCB score
    population = sorted(population, key=lambda c: c.lcb_score, reverse=True)
    best_candidate = population[0]

    print(f"\n{'='*60}")
    print("FINAL POPULATION (sorted by LCB score)")
    print(f"{'='*60}")
    for i, c in enumerate(population):
        print(f"  {i+1}. id={c.id}, gen={c.generation}, cumul={c.raw_score:.0%} ({int(c.raw_successes)}/{int(c.raw_trials)}), LCB={c.lcb_score:.1%}")

    # Prepare output - save ALL candidates with full wikis
    output = {
        "config": {
            "train_files": TRAIN_FILES,
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_per_workflow": n_per_workflow,
            "model": model,
            "gen_model": gen_model,
            "seed": seed,
            "max_concurrency": max_concurrency,
            "beta_binomial": {
                "alpha_0": ALPHA_0,
                "beta_0": BETA_0,
                "inheritance_w": INHERITANCE_W,
                "lcb_z": LCB_Z,
            },
        },
        "history": history,
        "best_candidate": {
            "id": best_candidate.id,
            "generation": best_candidate.generation,
            "wiki": best_candidate.wiki,
            "raw_score": best_candidate.raw_score,
            "raw_successes": int(best_candidate.raw_successes),
            "raw_trials": int(best_candidate.raw_trials),
            "posterior_mean": best_candidate.posterior_mean,
            "lcb_score": best_candidate.lcb_score,
            "alpha": best_candidate.alpha,
            "beta": best_candidate.beta,
            "mutation_history": best_candidate.mutation_history,
        },
        # Save ALL final candidates with their full wikis
        "final_population": [
            {
                "id": c.id,
                "generation": c.generation,
                "raw_score": c.raw_score,
                "raw_successes": int(c.raw_successes),
                "raw_trials": int(c.raw_trials),
                "posterior_mean": c.posterior_mean,
                "lcb_score": c.lcb_score,
                "alpha": c.alpha,
                "beta": c.beta,
                "wiki": c.wiki,
                "mutation_history": c.mutation_history,
            }
            for c in population
        ],
    }

    # Save
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"learned_policies/rpe_ega_result_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Final update to run log
    run_log["completed_at"] = datetime.now().isoformat()
    run_log["final_result_path"] = output_path
    run_log["best_candidate_id"] = best_candidate.id
    run_log["best_candidate_score"] = best_candidate.raw_score
    save_run_log()

    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"Run log saved to: {run_log_path}")
    print(f"Best candidate: cumul={best_candidate.raw_score:.1%} ({int(best_candidate.raw_successes)}/{int(best_candidate.raw_trials)}), LCB={best_candidate.lcb_score:.2%}")
    print(f"\nTo evaluate on test set, run:")
    print(f"  python evaluate_prompt.py --prompt-path {output_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RPE-EGA Wiki Optimization. Uses pre-split training data from results/train_{workflow}.json"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of evolution iterations",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help="Population size (number of candidate prompts to maintain)",
    )
    parser.add_argument(
        "--n-per-workflow",
        type=int,
        default=3,
        help="Trajectories per workflow type per iteration (3 = 12 total)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model for agent evaluation",
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default=None,
        help="LLM model for wiki generation/mutation (defaults to --model)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for results",
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
        help="Number of parallel task evaluations (default 1 = sequential)",
    )

    args = parser.parse_args()

    rpe_ega_optimize(
        n_iterations=args.n_iterations,
        n_candidates=args.n_candidates,
        n_per_workflow=args.n_per_workflow,
        model=args.model,
        gen_model=args.gen_model,
        output_path=args.output_path,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
    )
