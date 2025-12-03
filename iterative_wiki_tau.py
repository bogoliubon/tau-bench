import dspy
from dspy.teleprompt import MIPROv2
from tau_bench.envs import get_env
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.types import RunConfig
from typing import List, Dict, Any
import json

EXCHANGE_TASK_IDS=[0,1,6,7,8,9,18,19,27,45,49,52,58,70,75,77,80,93,94,95,96,100,106,107]
# ============================================================================
# 1. DSPy LM Setup
# ============================================================================

def setup_dspy_lm(model: str = "gpt-4o-mini", provider: str = "openai"):
    """Configure DSPy's default language model"""
    lm = dspy.LM(f"{provider}/{model}", cache=False)
    dspy.configure(lm=lm)
    print(f"✅ DSPy configured with {provider}/{model}")
    return lm


# ============================================================================
# 2. Conversation Formatting
# ============================================================================

def format_conversation_for_analysis(conversation: List[Dict]) -> str:
    """Format conversation for WikiGenerator to analyze"""
    formatted = []
    
    # Skip the system message (that's the wiki we're trying to infer/refine)
    for msg in conversation[1:]:  # Start from index 1 to skip system message
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted.append(f"User: {content}")
        elif role == 'assistant':
            if msg.get('tool_calls'):
                # Show tool calls
                for tool_call in msg['tool_calls']:
                    func_name = tool_call['function']['name']
                    func_args = tool_call['function']['arguments']
                    formatted.append(f"Agent (tool): {func_name}({func_args})")
            elif content:
                formatted.append(f"Agent: {content}")
        elif role == 'tool':
            # Truncate long tool results
            truncated = content[:200] + "..." if len(content) > 200 else content
            formatted.append(f"Tool result: {truncated}")
    
    return "\n".join(formatted)


# ============================================================================
# 3. Iterative Wiki Generator (DSPy Signatures & Module)
# ============================================================================

class InitialWikiGenerator(dspy.Signature):
    """Analyze a successful agent conversation for handling order exchanges and generate an initial system prompt section that focuses specifically on handling exchanging delivered order items."""
    
    available_tools = dspy.InputField(desc="Available tools for the agent")
    example_conversation = dspy.InputField(desc="A successful exchange conversation showing agent behavior")
    
    system_wiki = dspy.OutputField(desc="System prompt section focused on exchange workflow policies")
    reasoning = dspy.OutputField(desc="Key patterns observed in the conversation that inform the wiki")


class WikiRefiner(dspy.Signature):
    """Given a current wiki section for order exchanges and another successful exchange conversation, determine if the wiki adequately covers the exchange policies demonstrated, and refine it if needed."""
    
    current_wiki = dspy.InputField(desc="Current system prompt section for exchanges")
    new_conversation = dspy.InputField(desc="Another successful exchange conversation")
    
    needs_revision = dspy.OutputField(desc="Does the wiki need revision to better handle exchanges? (yes/no)")
    revision_reasoning = dspy.OutputField(desc="What exchange-related aspects are missing or need clarification?")
    revised_wiki = dspy.OutputField(desc="Revised system prompt section (or original if adequate)")


class IterativeWikiGenerator(dspy.Module):
    """Generate exchange-focused wiki by iteratively refining based on multiple successful exchange examples"""
    
    def __init__(self, num_examples: int = 5):
        super().__init__()
        self.num_examples = num_examples
        self.initial_generator = dspy.ChainOfThought(InitialWikiGenerator)
        self.refiner = dspy.ChainOfThought(WikiRefiner)
    
    def forward(self, task_domain: str, available_tools: str, example_conversations: List[str]):
        """
        Args:
            task_domain: "retail customer service - exchange"
            available_tools: Tool descriptions
            example_conversations: List of formatted exchange conversations
        """
        # Step 1: Generate initial wiki from first exchange example
        print(f"  Step 0: Generating initial wiki from first exchange example...")
        initial_result = self.initial_generator(
            available_tools=available_tools,
            example_conversation=example_conversations[0]
        )
        
        current_wiki = initial_result.system_wiki
        refinement_history = [
            {
                "step": 0,
                "wiki": current_wiki,
                "reasoning": initial_result.reasoning
            }
        ]
        
        # Steps 2-N: Iteratively refine based on additional exchange examples
        num_to_use = min(self.num_examples, len(example_conversations))
        for i, conversation in enumerate(example_conversations[1:num_to_use], start=1):
            print(f"  Step {i}: Refining wiki with exchange example {i+1}...")
            refinement_result = self.refiner(
                current_wiki=current_wiki,
                new_conversation=conversation,
            )
            
            refinement_history.append({
                "step": i,
                "needs_revision": refinement_result.needs_revision,
                "reasoning": refinement_result.revision_reasoning,
                "wiki": refinement_result.revised_wiki
            })
            
            # Update current wiki
            current_wiki = refinement_result.revised_wiki
        
        return dspy.Prediction(
            system_wiki=current_wiki,
            refinement_history=refinement_history
        )


# ============================================================================
# 4. Episode Runner
# ============================================================================

class TauBenchEpisodeRunner(dspy.Module):
    """Run tau-bench episodes with iteratively generated exchange wiki"""
    
    def __init__(self, config: RunConfig, wiki_module: IterativeWikiGenerator):
        super().__init__()
        self.config = config
        self.wiki_module = wiki_module
    
    def forward(self, task_id: int, task_domain: str, available_tools: str, 
                example_conversations: List[str]):
        """Generate wiki iteratively from examples, then run episode"""
        
        # Generate wiki using iterative refinement
        wiki_pred = self.wiki_module(
            task_domain=task_domain,
            available_tools=available_tools,
            example_conversations=example_conversations
        )
        generated_wiki = wiki_pred.system_wiki
        
        # Create environment
        env = get_env(
            self.config.env,
            user_strategy=self.config.user_strategy,
            user_model=self.config.user_model,
            task_split=self.config.task_split,
            user_provider=self.config.user_model_provider,
            task_index=task_id,
        )
        
        # Create agent with generated wiki
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=generated_wiki,
            model=self.config.model,
            provider=self.config.model_provider,
            temperature=self.config.temperature,
        )
        
        # Run episode
        try:
            result = agent.solve(env=env, task_index=task_id)
            reward = result.reward
            info = result.info
        except Exception as e:
            reward = 0.0
            info = {"error": str(e)}
        
        return dspy.Prediction(
            reward=reward,
            info=info,
            generated_wiki=generated_wiki,
            refinement_history=wiki_pred.refinement_history
        )


# ============================================================================
# 5. Data Loading (Simplified - No Filtering)
# ============================================================================

def extract_tools_summary(tools_info: List[Dict[str, Any]]) -> str:
    """Extract concise summary of available tools"""
    tool_names = [tool.get('function', {}).get('name', 'unknown') 
                  for tool in tools_info]
    return ", ".join(tool_names)


def load_exchange_trajectories(
    trajectories_path: str,
    config: RunConfig,
    verbose: bool = True
) -> List[dspy.Example]:
    """
    Load pre-filtered exchange trajectories.
    Assumes the JSON file contains only exchange-related trajectories.
    """
    
    with open(trajectories_path, 'r') as f:
        exchange_trajectories = json.load(f)
    
    if verbose:
        print(f"📊 Loading trajectories from: {trajectories_path}")
        print(f"  Total trajectories: {len(exchange_trajectories)}")
    
    # Filter for successful ones
    successful_exchanges = [t for t in exchange_trajectories if t['reward'] >= 0.99]
    
    if verbose:
        print(f"  ✅ Successful: {len(successful_exchanges)}")
        print(f"  ❌ Failed: {len(exchange_trajectories) - len(successful_exchanges)}")
        success_rate = len(successful_exchanges) / len(exchange_trajectories) if exchange_trajectories else 0
        print(f"  Success rate: {success_rate:.1%}")
    
    if len(successful_exchanges) == 0:
        raise ValueError("No successful exchange trajectories found!")
    
    # Get environment info
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        task_split=config.task_split,
        user_provider=config.user_model_provider,
    )
    
    available_tools = extract_tools_summary(env.tools_info)
    
    # Extract all successful conversations
    all_conversations = []
    task_ids = []
    
    for traj in successful_exchanges:
        conversation = traj.get('traj', [])
        conversation_text = format_conversation_for_analysis(conversation)
        all_conversations.append(conversation_text)
        task_ids.append(traj['task_id'])
    
    # Create DSPy examples
    # Each example gets access to ALL successful exchange conversations
    examples = []
    for task_id in task_ids:
        example = dspy.Example(
            task_id=task_id,
            task_domain="retail customer service - exchange",
            available_tools=available_tools,
            example_conversations=all_conversations,  # All conversations available
            reward=1.0  # All are successful
        ).with_inputs('task_id', 'task_domain', 'available_tools', 'example_conversations')
        
        examples.append(example)
    
    if verbose:
        print(f"\n📚 Created {len(examples)} DSPy examples")
        print(f"   Each example has access to all {len(all_conversations)} successful exchange conversations")
        if all_conversations:
            print(f"   Sample conversation length: {len(all_conversations[0])} chars")
    
    return examples


# ============================================================================
# 6. Metric
# ============================================================================

def episode_reward_metric(example, pred, trace=None):
    """Higher reward = better wiki"""
    return pred.reward


# ============================================================================
# 7. Optimization
# ============================================================================

def optimize_exchange_wiki_with_mipro(
    config: RunConfig,
    trajectories_path: str,
    train_size: int = 20,
    val_size: int = 10,
    num_refinement_examples: int = 5
):
    """
    Optimize iterative wiki generation for exchange workflow.
    """
    
    # Load exchange trajectories
    print("="*80)
    print("LOADING EXCHANGE TRAJECTORIES")
    print("="*80)
    all_examples = load_exchange_trajectories(trajectories_path, config)
    
    total_needed = train_size + val_size
    if len(all_examples) < total_needed:
        raise ValueError(f"Need at least {total_needed} successful exchange trajectories, only have {len(all_examples)}")
    
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size + val_size]
    
    print(f"\n✅ Data split:")
    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")
    print(f"   Refinement examples per generation: {num_refinement_examples}")
    
    # Initialize iterative wiki generator
    wiki_module = IterativeWikiGenerator(num_examples=num_refinement_examples)
    
    # Wrap with episode runner
    episode_runner = TauBenchEpisodeRunner(config, wiki_module)
    
    # Configure MIPRO
    print("\n" + "="*80)
    print("INITIALIZING MIPRO")
    print("="*80)
    print("⚠️  MIPRO will optimize instructions for:")
    print("   1. Initial wiki generation from first exchange example")
    print("   2. Iterative refinement based on additional examples")
    
    mipro_optimizer = MIPROv2(
        metric=episode_reward_metric,
        prompt_model=None,
        task_model=None,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_candidates=3,
        auto=None,
        num_threads=1,
        seed=42,
        init_temperature=1.0,
        verbose=True,
        track_stats=True,
        log_dir="./mipro_logs_exchange",
    )
    
    # Run optimization
    print("\n" + "="*80)
    print("MIPRO OPTIMIZATION")
    print("="*80)
    print("🚀 Starting optimization of iterative exchange wiki generation...\n")
    
    optimized_program = mipro_optimizer.compile(
        episode_runner,
        trainset=train_examples,
        valset=val_examples,
        num_trials=3,
        minibatch_size=5,
    )
    
    print("\n✅ Optimization complete!")
    
    # Generate a sample wiki using optimized process
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        task_split=config.task_split,
        user_provider=config.user_model_provider,
    )
    
    # Use first N training conversations for sample generation
    sample_conversations = train_examples[0].example_conversations[:num_refinement_examples]
    
    print(f"\n📝 Generating sample exchange wiki using {len(sample_conversations)} examples...")
    sample_output = optimized_program.wiki_module(
        task_domain="retail customer service - exchange",
        available_tools=extract_tools_summary(env.tools_info),
        example_conversations=sample_conversations
    )
    
    optimized_wiki = sample_output.system_wiki
    
    print(f"\n" + "="*80)
    print("GENERATED EXCHANGE WIKI")
    print("="*80)
    print(optimized_wiki)
    
    print(f"\n" + "="*80)
    print("REFINEMENT HISTORY")
    print("="*80)
    for step_info in sample_output.refinement_history:
        step_num = step_info['step']
        if step_num == 0:
            print(f"\nStep 0 (Initial):")
            print(f"  Reasoning: {step_info['reasoning'][:200]}...")
        else:
            print(f"\nStep {step_num}:")
            print(f"  Needs revision: {step_info.get('needs_revision', 'N/A')}")
            print(f"  Reasoning: {step_info['reasoning'][:200]}...")
    
    return optimized_wiki, optimized_program


# ============================================================================
# 8. Evaluation
# ============================================================================

def evaluate_wiki(
    wiki_text: str,
    config: RunConfig,
    test_task_ids: List[int],
    label: str = "optimized",
    task_split: str = "test"
):
    """Evaluate a wiki on test tasks"""
    results = []
    
    print(f"\n📊 Evaluating {label} wiki on {len(test_task_ids)} tasks from '{task_split}' split...")
    
    for i, task_id in enumerate(test_task_ids):
        print(f"[{i+1}/{len(test_task_ids)}] Running task {task_id}...", end=" ")
        
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
                wiki=wiki_text,
                model=config.model,
                provider=config.model_provider,
                temperature=config.temperature,
            )
            
            result = agent.solve(env=env, task_index=task_id)
            results.append({
                'task_id': task_id,
                'reward': result.reward,
                'info': result.info
            })
            print("✅" if result.reward >= 0.99 else "❌", f"reward={result.reward}")
            
        except Exception as e:
            results.append({
                'task_id': task_id,
                'reward': 0.0,
                'info': {'error': str(e)}
            })
            print(f"❌ error={str(e)[:100]}")
    
    avg_reward = sum(r['reward'] for r in results) / len(results) if results else 0
    success_rate = sum(1 for r in results if r['reward'] >= 0.99) / len(results) if results else 0
    
    print(f"\n{label.upper()} RESULTS:")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Success rate: {success_rate:.1%}")
    
    return results, avg_reward, success_rate


# ============================================================================
# 9. Main Pipeline
# ============================================================================

if __name__ == "__main__":
    # Setup DSPy
    print("Setting up DSPy...")
    setup_dspy_lm(model="gpt-4o-mini", provider="openai")

    # Configure tau-bench
    config = RunConfig(
        env="retail",
        model="gpt-4o",
        model_provider="openai",
        user_model="gpt-4o",
        user_model_provider="openai",
        agent_strategy="tool-calling",
        task_split="train",
        user_strategy="llm",
        temperature=0.0,
        seed=42,
        start_index=0,
        end_index=-1,
        log_dir="./logs",
        max_concurrency=1,
        shuffle=False,
        num_trials=1,
        task_ids=[],
        few_shot_displays_path=None
    )
    
    # Path to your PRE-FILTERED exchange trajectories JSON
    trajectories_path = "exchange_delivered_order_items.json"  # ← UPDATE THIS
    
    print(f"\n🎯 EXCHANGE WORKFLOW OPTIMIZATION")
    print(f"   Goal: Learn exchange-specific wiki from successful trajectories")
    print(f"   Method: Iterative refinement (1 initial + 4 refinements)")
    
    # Optimize exchange wiki
    optimized_wiki, optimized_program = optimize_exchange_wiki_with_mipro(
        config=config,
        trajectories_path=trajectories_path,
        train_size=10,
        val_size=5,
        num_refinement_examples=5
    )
    
    # Get baseline wiki for comparison
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        task_split=config.task_split,
        user_provider=config.user_model_provider,
    )
    baseline_wiki = env.wiki
    
    # Test on exchange tasks
    test_task_ids = EXCHANGE_TASK_IDS
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Evaluate baseline
    baseline_results, baseline_reward, baseline_success = evaluate_wiki(
        baseline_wiki, config, test_task_ids, label="baseline", task_split="test"
    )
    
    # Evaluate optimized
    optimized_results, optimized_reward, optimized_success = evaluate_wiki(
        optimized_wiki, config, test_task_ids, label="optimized", task_split="test"
    )
    
    # Compare
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Baseline:  {baseline_reward:.3f} reward, {baseline_success:.1%} success")
    print(f"Optimized: {optimized_reward:.3f} reward, {optimized_success:.1%} success")
    print(f"Improvement: {(optimized_reward - baseline_reward):+.3f} reward, {(optimized_success - baseline_success):+.1%} success rate")
    
    # Save results
    with open("exchange_wiki_optimization_results.json", "w") as f:
        json.dump({
            'workflow': 'exchange',
            'method': 'iterative_refinement',
            'num_refinement_examples': 5,
            'baseline_results': baseline_results,
            'optimized_results': optimized_results,
            'optimized_wiki': optimized_wiki,
            'baseline_wiki': baseline_wiki
        }, f, indent=2)
    
    print(f"\n💾 Results saved to exchange_wiki_optimization_results.json")