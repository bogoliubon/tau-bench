# scripts/train_infer_policy_prompt_small.py
"""
Train/compile a SMALL DSPy program that infers a prompt-policy TEXT (sentences)
from dialogue only, and evaluates candidates using tau-bench binary reward (0/1).

Manual splits:
- You will manually choose:
  - 10 task_ids for training
  - 5 task_ids for testing
(Optionally a dev set too, but for now we keep it simple.)

Usage (example):
  export OPENAI_API_KEY="sk-..."
  export DSPY_MODEL="openai/gpt-5"
  export LITELLM_MAX_PARALLEL_REQUESTS=1

  PYTHONPATH=. python3 -m scripts.train_infer_policy_prompt_small \
    --data_dir data/tau2/human_trajectories/retail \
    --out_dir artifacts/infer_policy_prompt_small
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy

from policies.infer_prompt_policy_text_small import InferPromptPolicyTextSmall

import subprocess
import tempfile



# ----------------------------
# 0) Manual splits (will edit these later)
# ----------------------------

TRAIN_TASK_IDS: List[str] = [str(i) for i in range(1, 11)]
TEST_TASK_IDS: List[str]  = [str(i) for i in range(11, 16)]

# Optional: if you want a small dev set distinct from train/test, fill this later.
DEV_TASK_IDS: List[str] = [
    # "..."
]


# ----------------------------
# 1) Trajectory loading + dialogue serialization
# ----------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _iter_cases_from_file(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    A) big files: list of cases with case["traj"]
    B) single human trajectory object: {task_id, messages:[...]}
    C) list of human trajectory objects
    """
    obj = _read_json(path)

    # A) list of big cases
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "traj" in obj[0]:
        return obj

    # C) list of human trajectories
    if isinstance(obj, list) and (not obj or (isinstance(obj[0], dict) and "messages" in obj[0] and "task_id" in obj[0])):
        return obj

    # B) single human trajectory object
    if isinstance(obj, dict) and "messages" in obj and "task_id" in obj:
        return [obj]

    # Common containers
    if isinstance(obj, dict):
        for k in ("cases", "records", "tasks", "trajectories", "data", "examples"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]

    raise ValueError(f"Unrecognized JSON structure in {path}")

def serialize_dialogue(messages: List[Dict[str, Any]], keep_last_lines: int = 120) -> str:
    """
    Must preserve tool calls when assistant content is null.
    Output is a newline-separated transcript with role tags.
    """
    lines: List[str] = []

    for m in messages or []:
        role = m.get("role", "unknown")

        # Skip system (we are inferring system prompt)
        if role == "system":
            continue

        if role == "user":
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[user] {content}")
            continue

        if role == "assistant":
            content = (m.get("content") or "")
            content = content.strip() if isinstance(content, str) else ""
            if content:
                lines.append(f"[assistant] {content}")
            else:
                tool_calls = m.get("tool_calls") or []
                for tc in tool_calls:
                    # human trajectories often store name/arguments at top-level
                    name = tc.get("name") or (tc.get("function") or {}).get("name") or "unknown_tool"
                    args = tc.get("arguments", "")
                    # args may be dict (human traj) or string (openai-style)
                    if isinstance(args, dict):
                        args_str = json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = str(args)
                    if len(args_str) > 240:
                        args_str = args_str[:240] + "..."
                    lines.append(f"[assistant][tool_call] {name} {args_str}")
            continue

        if role == "tool":
            tool_name = m.get("name", "tool")
            content = (m.get("content") or "").strip()
            if content:
                if len(content) > 320:
                    content = content[:320] + "..."
                lines.append(f"[tool:{tool_name}] {content}")
            continue

        # fallback
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"[{role}] {content}")

    # keep last N lines
    lines = lines[-keep_last_lines:]
    return "\n".join(lines)

def extract_task_id_and_messages(case: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Handles:
    - big schema: case["info"]["task"]["task_id"] or case["info"]["task"]["id"] sometimes; messages in case["traj"]
    - human schema: case["task_id"], messages in case["messages"]
    """
    if "messages" in case and "task_id" in case:
        return str(case["task_id"]), case["messages"]

    traj = case.get("traj", [])
    # task id inference
    info_task = ((case.get("info") or {}).get("task") or {})
    # different datasets use different keys; try a few
    task_id = info_task.get("task_id") or info_task.get("id") or case.get("task_id") or case.get("id")
    task_id = str(task_id) if task_id is not None else ""
    return task_id, traj


# ----------------------------
# 2) Tau-bench binary reward hook (YOU wire this to your wrapper)
# ----------------------------

def evaluate_prompt_policy_binary(
    *,
    task_id: str,
    prompt_policy: str,
    tau_bench_root: Optional[Path] = None,
    timeout_s: int = 120,
) -> int:
    """
    Runs tau-bench for a single task_id with an injected prompt policy.
    Returns 1 if reward==1 else 0.
    """
    if tau_bench_root is None:
        # Adjust if your DSPy script is run from repo root with PYTHONPATH=.
        tau_bench_root = Path(__file__).resolve().parents[1]  # scripts/.. -> repo root

    tau_bench_root = tau_bench_root.expanduser().resolve()

    # 🔒 SANITY CHECK 
    run_py = tau_bench_root / "run.py"
    if not run_py.exists():
        raise RuntimeError(f"Cannot find run.py at: {run_py}")

    task_id_int = int(task_id)

    # 1) Write policy to a temp file
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        policy_path = td_path / "policy_override.txt"
        policy_path.write_text(prompt_policy, encoding="utf-8")

        # 2) Run tau-bench in an isolated log dir so we can find the output deterministically
        log_dir = td_path / "tau_bench_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: call the CLI entrypoint you actually use.
        # If your repo root has "run.py" (the CLI), use: python run.py ...
        # If you run as module, use: python -m run ...
        cmd = [
            "python3",
            "run.py",
            "--env", "retail",
            "--agent-strategy", "tool-calling",
            "--task-split", os.environ.get("TAU_BENCH_SPLIT", "train"),
            "--task-ids", str(task_id_int),
            "--num-trials", "1",
            "--max-concurrency", "1",
            "--shuffle", "0",
            "--log-dir", str(log_dir),
            "--policy-override-path", str(policy_path),

            # You must also provide your model/provider arguments:
            "--model", os.environ.get("TAU_BENCH_MODEL", "gpt-4o"),
            "--model-provider", os.environ.get("TAU_BENCH_PROVIDER", "openai"),
            "--user-model", os.environ.get("TAU_BENCH_USER_MODEL", "gpt-4o"),
            "--user-model-provider", os.environ.get("TAU_BENCH_USER_PROVIDER", "openai"),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(tau_bench_root)

        proc = subprocess.run(
            cmd,
            cwd=str(tau_bench_root),
            env=env,  
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )

        if proc.returncode != 0:
            # Treat any crash as failure
            return 0

        # 3) Find the json results file in log_dir (tau-bench writes exactly one file per run call here)
        json_files = sorted(log_dir.glob("*.json"))
        if not json_files:
            return 0

        results = json.loads(json_files[-1].read_text(encoding="utf-8"))
        # results is a list; for single task, take last/first entry
        if not results:
            return 0

        reward = results[-1].get("reward", 0.0)
        return 1 if abs(float(reward) - 1.0) < 1e-6 else 0
    


# ----------------------------
# 3) DSPy metric + compile
# ----------------------------



def configure_lm():
    model_name = os.getenv("DSPY_MODEL", "openai/gpt-5")

    # You already observed DSPy constraints for GPT-5 reasoning models.
    # Keep consistent with your repo: temperature=1.0, max_tokens>=16000.
    lm = dspy.LM(model_name, temperature=1.0, max_tokens=16000)
    dspy.settings.configure(lm=lm)
    print("[small-train] LM:", model_name)

def binary_reward_metric(ex, pred: dspy.Prediction, trace=None) -> float:
    prompt_policy = (getattr(pred, "prompt_policy", "") or "").strip()
    if not prompt_policy:
        return 0.0
    try:
        r = evaluate_prompt_policy_binary(task_id=str(ex["task_id"]), prompt_policy=prompt_policy)
        return float(1.0 if int(r) == 1 else 0.0)
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing per-task trajectory JSON files")
    ap.add_argument("--out_dir", default="artifacts/infer_policy_prompt_small", help="Where to save compiled program")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_last_lines", type=int, default=120)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Set it in your shell first.")

    random.seed(args.seed)
    configure_lm()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect cases for selected task_ids by scanning JSONs in data_dir
    # Expect filenames like task_83_human.json but do not rely on naming.
    all_jsons = sorted([p for p in data_dir.rglob("*.json") if p.is_file()])
    if not all_jsons:
        raise SystemExit(f"No .json files found under {data_dir}")

    def collect_examples(target_task_ids: List[str]) -> List[dspy.Example]:
        target = set(str(x) for x in target_task_ids if str(x).strip())
        out: List[dspy.Example] = []
        if not target:
            return out

        for p in all_jsons:
            try:
                cases = _iter_cases_from_file(p)
            except Exception:
                continue
            for c in cases:
                task_id, msgs = extract_task_id_and_messages(c)
                if task_id in target:
                    dialogue = serialize_dialogue(msgs, keep_last_lines=args.keep_last_lines)
                    # dspy.Example must be dict-like. Also mark inputs.
                    ex = dspy.Example(task_id=task_id, dialogue=dialogue).with_inputs("dialogue")
                    out.append(dspy.Example(task_id=task_id, dialogue=dialogue).with_inputs("dialogue"))

        # de-dup: keep first per task_id
        seen = set()
        uniq = []
        for ex in out:
            if ex["task_id"] not in seen:
                seen.add(ex["task_id"])
                uniq.append(ex)
        return uniq

      

    train_examples = collect_examples(TRAIN_TASK_IDS)
    dev_examples = collect_examples(DEV_TASK_IDS) if DEV_TASK_IDS else []
    test_examples = collect_examples(TEST_TASK_IDS)

    print("[small-train] train tasks:", TRAIN_TASK_IDS)
    print("[small-train] test  tasks:", TEST_TASK_IDS)
    print("[small-train] found train/dev/test examples:", len(train_examples), len(dev_examples), len(test_examples))

    if len(train_examples) == 0:
        raise SystemExit(
            "No training examples found. Fill TRAIN_TASK_IDS with your 10 selected task_ids "
            "and ensure data_dir points to the matching trajectory JSON files."
        )
    if len(test_examples) == 0:
        print("[small-train][warn] No test examples found yet. Fill TEST_TASK_IDS later.")

    student = InferPromptPolicyTextSmall()

    # We compile with InferRules but WITHOUT demos (zero-shot final program).
    infer = dspy.InferRules(
        metric=lambda ex, pr, trace=None: binary_reward_metric(ex, pr, trace),
        num_candidates=6,
        num_rules=8,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    compiled = infer.compile(student, trainset=train_examples, valset=(dev_examples or train_examples))

    # Hard guarantee: remove any demos if they exist
    try:
        for pred in compiled.predictors():
            pred.demos = []
    except Exception:
        pass

    # Save compiled program
    saved = False
    if hasattr(compiled, "save"):
        compiled.save(str(out_dir), save_program=True)
        saved = True
        print(f"[small-train] Saved compiled program to: {out_dir}/")
    if not saved:
        import pickle
        with open(out_dir / "compiled.pkl", "wb") as f:
            pickle.dump(compiled, f)
        print(f"[small-train] Saved compiled program to: {out_dir}/compiled.pkl (pickle fallback)")

    # Optional final eval on test set (binary reward)
    if test_examples:
        scores = []
        for ex in test_examples:
            pr = compiled(dialogue=ex.dialogue)
            s = binary_reward_metric(ex, pr)
            scores.append(s)
            print(f"[small-train][test] task_id={ex.task_id} reward={int(s)}")
        print(f"[small-train] Test mean reward: {sum(scores)/len(scores):.3f}")

if __name__ == "__main__":
    main()
