# Copyright Sierra

from pathlib import Path
import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def _load_policy_override_text(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"policy_override_path does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"policy_override_path is not a file: {p}")
    return p.read_text(encoding="utf-8").strip()


def _get_repo_root() -> Path:
    # This file is: <repo_root>/tau_bench/run.py
    # parents[0] = tau_bench/, parents[1] = repo_root/
    return Path(__file__).resolve().parents[1]


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)

    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = (
        f"{config.log_dir}/"
        f"{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}"
        f"_range_{config.start_index}-{config.end_index}"
        f"_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    )
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )

    # ----------------------------
    # Choose base wiki
    # ----------------------------
    if config.wikipath:
        with open(config.wikipath, "r", encoding="utf-8") as f:
            data = json.load(f)
            wiki = data["iterations"][-1]["policy"]

    elif config.concatenate_from_model:
        repo_root = _get_repo_root()
        wiki_folder = repo_root / "iterative_policy_refinement"  # e.g. ~/Desktop/agent/tau-bench/iterative_policy_refinement
        if not wiki_folder.exists():
            raise ValueError(f"iterative_policy_refinement folder not found: {wiki_folder}")

        wiki_files = [
            p for p in wiki_folder.iterdir()
            if p.is_file() and config.concatenate_from_model in p.name
        ]
        if not wiki_files:
            raise ValueError(f"No wiki files found for model: {config.concatenate_from_model} in {wiki_folder}")

        wiki = ""
        for p in sorted(wiki_files):
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                wiki += data["iterations"][-1]["policy"]

    elif config.summarize_from_model:
        repo_root = _get_repo_root()
        wiki_folder = repo_root / "iterative_policy_refinement"
        if not wiki_folder.exists():
            raise ValueError(f"iterative_policy_refinement folder not found: {wiki_folder}")

        wiki_files = [
            p for p in wiki_folder.iterdir()
            if p.is_file() and config.summarize_from_model in p.name
        ]
        if not wiki_files:
            raise ValueError(f"No wiki files found for model: {config.summarize_from_model} in {wiki_folder}")

        combined_wiki = []
        for p in sorted(wiki_files):
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                combined_wiki.append(data["iterations"][-1]["policy"])

        from utils import summarize_wiki
        wiki = summarize_wiki(combined_wiki)

    else:
        wiki = env.wiki

    # ----------------------------
    # Apply prompt-policy override
    # ----------------------------
    if getattr(config, "policy_override_path", None):
        override_text = _load_policy_override_text(config.policy_override_path)

        # Option A (recommended): append to existing wiki
        wiki = (
            wiki
            + "\n\n"
            + "================ PROMPT POLICY OVERRIDE ================\n"
            + override_text
            + "\n"
            + "========================================================\n"
        )

        # Option B: override completely (uncomment if you prefer)
        # wiki = override_text

    # ----------------------------
    # Create agent
    # ----------------------------
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=wiki,
        config=config,
    )

    end_index = len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))

    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()

    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")

    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))

        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(env=isolated_env, task_index=idx)
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )

            print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
            print("-----")

            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)

            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")

    return results


def agent_factory(tools_info: List[Dict[str, Any]], wiki, config: RunConfig) -> Agent:
    if config.agent_strategy == "tool-calling":
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        from tau_bench.agents.chat_react_agent import ChatReActAgent
        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        from tau_bench.agents.chat_react_agent import ChatReActAgent
        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r", encoding="utf-8") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]
        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)

    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0

    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)

    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
