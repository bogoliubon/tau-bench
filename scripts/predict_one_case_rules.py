# scripts/predict_one_case_rules.py
"""
Predict policy-rules JSON for ONE trajectory case (case-by-case debug).

This script:
- loads a compiled DSPy program from --out_dir (saved by train_infer_policy_rules.py)
- loads one tau-bench trajectory file (list of cases)
- extracts (user_instruction, gold prompt policy, dialogue turns) for --case_idx
- runs the compiled program to predict policy_rules JSON
- prints gold prompt policy (truncated) + predicted JSON

Usage:
  export OPENAI_API_KEY="sk-..."
  export DSPY_MODEL="openai/gpt-5"   # optional
  python -m scripts.predict_one_case_rules \
    --data historical_trajectories/tool-calling-....json \
    --case_idx 0 \
    --out_dir artifacts/infer_policy_rules_small
"""

import json
import os
from pathlib import Path
import argparse

import dspy


# ----------------------------
# Helpers
# ----------------------------

def serialize_dialogue(turns, keep_last: int = 80) -> str:
    """
    Tau-bench traj contains assistant messages with content=None but tool_calls!=None.
    We must include those tool call events in the serialized dialogue.

    keep_last counts *lines* (after serialization), not raw turns.
    """
    lines = []

    for t in turns or []:
        role = t.get("role", "unknown")

        # skip system message in dialogue
        if role == "system":
            continue

        # user normal text
        if role == "user":
            content = (t.get("content") or "").strip()
            if content:
                lines.append(f"[user] {content}")
            continue

        # assistant: either content OR tool_calls
        if role == "assistant":
            content = (t.get("content") or "").strip()
            if content:
                lines.append(f"[assistant] {content}")
            else:
                tool_calls = t.get("tool_calls") or []
                # include tool call lines even if content is null
                for tc in tool_calls:
                    fn = (tc.get("function") or {})
                    name = fn.get("name", "unknown_tool")
                    args = fn.get("arguments", "")
                    # keep args short so prompts don’t explode
                    if isinstance(args, str) and len(args) > 200:
                        args = args[:200] + "..."
                    lines.append(f"[assistant][tool_call] {name} {args}")
            continue

        # tool: include name + content
        if role == "tool":
            tool_name = t.get("name", "tool")
            content = (t.get("content") or "").strip()
            if content:
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"[tool:{tool_name}] {content}")
            continue

        # fallback
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"[{role}] {content}")

    # keep last N lines
    lines = lines[-keep_last:]
    return "\n".join(lines)


def load_cases(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # try common containers
        for k in ("cases", "records", "tasks", "trajectories", "data", "examples"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unrecognized JSON structure in {path}")


def extract_fields_tau_bench_case(case: dict):
    """
    Supports your current tau-bench historical trajectory schema:
      - user_instruction: case["info"]["task"]["instruction"]
      - dialogue turns:   case["traj"]
      - gold policy:      first system message in case["traj"]
    """
    traj = case.get("traj", [])
    instr = (case.get("info", {}) or {}).get("task", {}) or {}
    user_instruction = instr.get("instruction", "")

    gold_policy = next((m.get("content", "") for m in traj if m.get("role") == "system"), "")

    dialogue = serialize_dialogue(traj)
    return user_instruction, gold_policy, dialogue


def configure_lm():
    """
    Configure DSPy LM for inference.

    Uses DSPY_MODEL if set; otherwise defaults to openai/gpt-5.
    Keep max_tokens small because we output JSON rules only.
    """
    model_name = os.getenv("DSPY_MODEL", "openai/gpt-5")
    # You can adjust max_tokens/temperature as desired
    lm = dspy.LM(model_name, max_tokens=256, temperature=0.0)
    dspy.settings.configure(lm=lm)


def load_compiled_program(out_dir: Path):
    """
    Load compiled program saved by train_infer_policy_rules.py.
    - Prefer dspy.load(out_dir) if DSPy .save() was used.
    - Otherwise fall back to pickle at out_dir/compiled.pkl
    """
    # If you used compiled.save(out_dir, save_program=True), DSPy can load directory
    try:
        return dspy.load(str(out_dir))
    except Exception:
        pass

    # Pickle fallback
    pkl = out_dir / "compiled.pkl"
    if pkl.exists():
        import pickle
        with pkl.open("rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Could not load compiled program from {out_dir}. "
        f"Expected DSPy save dir or {pkl}."
    )


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to a tau-bench historical trajectory JSON file")
    ap.add_argument("--case_idx", type=int, default=0, help="Index within that JSON file")
    ap.add_argument("--out_dir", default="artifacts/infer_policy_rules", help="Directory containing compiled program")
    args = ap.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Set it in your shell first.")

    configure_lm()

    compiled = load_compiled_program(out_dir)

    cases = load_cases(data_path)
    if args.case_idx < 0 or args.case_idx >= len(cases):
        raise SystemExit(f"case_idx out of range: {args.case_idx} (0..{len(cases)-1})")

    case = cases[args.case_idx]
    user_instruction, gold_policy, dialogue = extract_fields_tau_bench_case(case)

    if not user_instruction:
        print("[warn] user_instruction is empty for this case")
    if not dialogue:
        print("[warn] dialogue is empty for this case")

    pred = compiled(user_instruction=user_instruction, dialogue=dialogue)

    print("DATA FILE:", str(data_path))
    print("CASE IDX :", args.case_idx)

    print("\n--- GOLD PROMPT POLICY (from trajectory system message, truncated) ---\n")
    print((gold_policy or "(none)")[:1500], "\n...")

    print("\n--- PREDICTED POLICY RULES JSON ---\n")
    print(getattr(pred, "policy_rules", ""))


if __name__ == "__main__":
    main()
