import json
import random
import re
from pathlib import Path
import os

import dspy
from policies.infer_prompt_policy_rules import InferPromptPolicyRules, RULE_KEYS

# ----------------------------
# 1) Data loading / parsing
# ----------------------------

def _first_present(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None

def load_cases(json_path: str):
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("cases", "records", "tasks", "trajectories", "data", "examples"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unrecognized JSON structure in {json_path}. Top-level type: {type(obj)}")

def serialize_dialogue(turns, keep_last=40):
    lines = []
    turns = turns[-keep_last:]
    for t in turns:
        role = t.get("role", "unknown")
        content = (t.get("content") or "").strip()
        # include tool name to preserve meaning
        if role == "tool":
            name = t.get("name", "tool")
            content = f"{name}: {content}"
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def record_to_example(case: dict):
    traj = case.get("traj", [])
    if not traj:
        raise KeyError("case.traj is missing/empty")

    # user instruction
    user_instruction = (case.get("info", {}) or {}).get("task", {}).get("instruction", "")
    if not user_instruction:
        raise KeyError("case.info.task.instruction missing")

    # gold promptpolicy = first system message
    gold_policy = next((m.get("content", "") for m in traj if m.get("role") == "system"), "")
    if not gold_policy:
        raise KeyError("no system message found in traj")

    # dialogue = everything except the system message
    dialogue_turns = [m for m in traj if m.get("role") != "system"]

    return dspy.Example(
        user_instruction=user_instruction,
        dialogue=serialize_dialogue(dialogue_turns),
        prompt_policy=gold_policy,   # label for scoring/training
    ).with_inputs("user_instruction", "dialogue")


# ----------------------------
# 2) Rule extraction metric
# ----------------------------

_RULE_PATTERNS = {
    "auth_required": [
        r"authenticat", r"locat(e|ing).*(user )?id", r"\bemail\b", r"\bzip\b", r"name.*zip"
    ],
    "one_user_per_conversation": [
        r"one user per conversation", r"only help one user", r"one customer at a time",
        r"do not (handle|help) other users"
    ],
    "explicit_confirmation_before_db_write": [
        r"explicit( |-)confirm", r"confirm .* before", r"obtain explicit .* confirmation", r"user confirmation"
    ],
    "one_tool_call_at_a_time": [
        r"one tool call", r"at a time.*tool call"
    ],
    "modify_or_cancel_only_if_pending": [
        r"modify.*pending", r"cancel.*pending", r"only .*modif(y|ied).* pending"
    ],
    "return_only_if_delivered": [
        r"return.*delivered", r"only .*return .* if .* delivered"
    ],
    "exchange_only_if_delivered": [
        r"exchang(e|es|ed).*delivered", r"only .*exchang(e|es).* if .* delivered"
    ],
    "no_hallucination_or_subjective_advice": [
        r"should not make up", r"do not make up", r"not make up", r"subjective recommendation", r"subjective comments"
    ],
}

def gold_policy_to_rule_dict(policy_text: str):
    t = (policy_text or "").lower()
    out = {}

    # auth_required: if it talks about authentication at all
    out["auth_required"] = any(re.search(p, t) for p in _RULE_PATTERNS["auth_required"])

    # auth_methods: more specific
    methods = []
    if re.search(r"\bemail\b", t):
        methods.append("email")
    if re.search(r"\bzip\b", t) and re.search(r"\bname\b", t):
        methods.append("name_zip")
    out["auth_methods"] = methods

    # rest booleans
    for k in RULE_KEYS:
        if k in ("auth_required", "auth_methods"):
            continue
        pats = _RULE_PATTERNS.get(k, [])
        out[k] = any(re.search(p, t) for p in pats)

    # Ensure all keys exist
    for k in RULE_KEYS:
        out.setdefault(k, False if k != "auth_methods" else [])
    return out

def parse_predicted_rule_json(policy_rules_json: str):
    """
    Returns dict or None if invalid.
    """
    try:
        obj = json.loads(policy_rules_json)
        if not isinstance(obj, dict):
            return None

        # Normalize booleans and list
        norm = {}
        for k in RULE_KEYS:
            if k == "auth_methods":
                v = obj.get(k, [])
                if isinstance(v, list):
                    v = [str(x).strip() for x in v]
                else:
                    v = []
                # keep only known values
                v = [x for x in v if x in ("email", "name_zip")]
                norm[k] = sorted(set(v))
            else:
                v = obj.get(k, False)
                norm[k] = bool(v)
        return norm
    except Exception:
        return None

def rule_accuracy_metric(example, pred):
    gold = gold_policy_to_rule_dict(example.prompt_policy)
    got = parse_predicted_rule_json(getattr(pred, "policy_rules", "") or "")
    if got is None:
        return 0.0

    correct = 0
    total = 0

    for k in RULE_KEYS:
        total += 1
        if k == "auth_methods":
            if sorted(set(gold[k])) == sorted(set(got[k])):
                correct += 1
        else:
            if bool(gold[k]) == bool(got[k]):
                correct += 1

    return correct / total

def eval_program(program, dataset):
    scores = []
    for ex in dataset:
        p = program(user_instruction=ex.user_instruction, dialogue=ex.dialogue)
        scores.append(rule_accuracy_metric(ex, p))
    return sum(scores) / max(len(scores), 1)

# ----------------------------
# 3) Train / compile / test
# ----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to historical_trajectories JSON file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--max_cases", type=int, default=0, help="0 = use all")
    ap.add_argument("--out_dir", default="artifacts/infer_policy_rules")
    args = ap.parse_args()

    random.seed(args.seed)

    cases = load_cases(args.data)
    if args.max_cases and args.max_cases > 0:
        cases = cases[:args.max_cases]

    examples = [record_to_example(r) for r in cases]
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * args.train_frac)
    n_dev = int(n * args.dev_frac)
    trainset = examples[:n_train]
    devset = examples[n_train:n_train + n_dev]
    testset = examples[n_train + n_dev:]
    
    print("[train] total examples:", len(examples))
    print("[train] train/dev/test:", len(trainset), len(devset), len(testset))
    print("[train] sample instruction:", trainset[0].user_instruction[:80] if trainset else "(empty)")
    # IMPORTANT: configure your LM exactly how you already do in your repo.
    # Keep output small, so max_tokens can be small too.
    # Example (adjust to your setup):
    #
    # lm = dspy.LM("openai/gpt-4o-mini", max_tokens=256, temperature=0.0)
    # dspy.settings.configure(lm=lm)
    #
    # If you already configured dspy.settings elsewhere, you can omit this.

    student = InferPromptPolicyRules()
    model_name = os.getenv("DSPY_MODEL", "openai/gpt-5")

    # GPT-5 (reasoning model) constraints in DSPy:
    lm = dspy.LM(model_name, temperature=1.0, max_tokens=16000)

    dspy.settings.configure(lm=lm)
    print("[train] LM:", model_name)



    # STEP 1: infer rules -> improves the "initial prompt" (no demos kept)
    # InferRules induces natural language rules and updates instructions, then searches candidates. :contentReference[oaicite:4]{index=4}
    # We force demos to 0 so the final program is zero-shot (no few-shot examples). :contentReference[oaicite:5]{index=5}
    infer = dspy.InferRules(
        metric=lambda ex, pr, trace=None: rule_accuracy_metric(ex, pr),
        num_candidates=6,      # keep small for speed; increase later
        num_rules=8,           # keep short; increase later
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    compiled = infer.compile(student, trainset=trainset, valset=devset)

    # Hard guarantee: remove any demos if they exist
    try:
        for pred in compiled.predictors():
            pred.demos = []
    except Exception:
        pass

    dev_score = eval_program(compiled, devset) if devset else float("nan")
    test_score = eval_program(compiled, testset) if testset else float("nan")

    print(f"Dev rule-accuracy:  {dev_score:.3f}")
    print(f"Test rule-accuracy: {test_score:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preferred save/load method (DSPy docs) :contentReference[oaicite:6]{index=6}
    saved = False
    if hasattr(compiled, "save"):
        compiled.save(str(out_dir), save_program=True)
        saved = True
        print(f"Saved compiled program to: {out_dir}/")
    if not saved:
        # fallback: pickle
        import pickle
        with open(out_dir / "compiled.pkl", "wb") as f:
            pickle.dump(compiled, f)
        print(f"Saved compiled program to: {out_dir}/compiled.pkl (pickle fallback)")

if __name__ == "__main__":
    main()
