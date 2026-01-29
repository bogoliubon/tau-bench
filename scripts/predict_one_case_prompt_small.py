# scripts/predict_one_case_prompt_small.py
"""
Predict prompt-policy TEXT for ONE trajectory file (single-case debug).

Usage:
  export OPENAI_API_KEY="sk-..."
  export DSPY_MODEL="openai/gpt-5"

  PYTHONPATH=. python3 -m scripts.predict_one_case_prompt_small \
    --data data/tau2/human_trajectories/retail/task_83_human.json \
    --case_idx 0 \
    --out_dir artifacts/infer_policy_prompt_small \
    --run_eval 0
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dspy

from scripts.train_infer_policy_prompt_small import (
    _iter_cases_from_file,
    extract_task_id_and_messages,
    serialize_dialogue,
    evaluate_prompt_policy_binary,
)

def configure_lm():
    model_name = os.getenv("DSPY_MODEL", "openai/gpt-5")
    lm = dspy.LM(model_name, temperature=1.0, max_tokens=16000)
    dspy.settings.configure(lm=lm)

def load_compiled_program(out_dir: Path):
    try:
        return dspy.load(str(out_dir))
    except Exception:
        pass

    pkl = out_dir / "compiled.pkl"
    if pkl.exists():
        import pickle
        with pkl.open("rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(f"Could not load compiled program from {out_dir} (dir save or compiled.pkl).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to a trajectory JSON file (human or big schema)")
    ap.add_argument("--case_idx", type=int, default=0, help="Index within that file if it contains a list")
    ap.add_argument("--out_dir", default="artifacts/infer_policy_prompt_small")
    ap.add_argument("--keep_last_lines", type=int, default=120)
    ap.add_argument("--run_eval", type=int, default=0, help="1 to run tau-bench binary eval hook")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Set it in your shell first.")

    configure_lm()

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    compiled = load_compiled_program(out_dir)

    cases = _iter_cases_from_file(data_path)
    if args.case_idx < 0 or args.case_idx >= len(cases):
        raise SystemExit(f"case_idx out of range: {args.case_idx} (0..{len(cases)-1})")

    case = cases[args.case_idx]
    task_id, messages = extract_task_id_and_messages(case)
    dialogue = serialize_dialogue(messages, keep_last_lines=args.keep_last_lines)

    pred = compiled(dialogue=dialogue)
    prompt_policy = (getattr(pred, "prompt_policy", "") or "").strip()

    print("DATA FILE:", str(data_path))
    print("CASE IDX :", args.case_idx)
    print("TASK ID  :", task_id)

    print("\n--- SERIALIZED DIALOGUE (tail) ---\n")
    print(dialogue)

    print("\n--- PREDICTED PROMPT POLICY (TEXT) ---\n")
    print(prompt_policy if prompt_policy else "(empty)")

    if args.run_eval == 1:
        r = evaluate_prompt_policy_binary(task_id=task_id, prompt_policy=prompt_policy)
        print("\n--- TAU-BENCH BINARY REWARD ---\n")
        print(int(r))

if __name__ == "__main__":
    main()
