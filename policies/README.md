# README — PromptPolicy Reconstruction (Tau-Bench) via DSPy (Rules JSON, Zero-Shot)
# Goal

Given tau-bench historical trajectories (dialogue logs), reconstruct the hidden per-task “prompt policy” that the agent followed.

# Core idea

Instead of predicting a long policy paragraph, we predict a small JSON rule vector that captures key constraints (auth, confirmations, tool usage constraints, etc.). This avoids long outputs, truncation, and brittle exact string matching.

# Important constraints
# 1) wiki.md is NOT provided to the reconstructor

tau_bench/envs/retail/wiki.md is a fixed global system policy.

In this project, we do not pass wiki.md into the reconstructor during training or inference (to match “wiki cannot be seen at training stage”).

The reconstructor only sees:

task instruction (info.task.instruction)

observed dialogue (traj, excluding the system message)

# 2) No few-shot examples in the final compiled program

We compile with DSPy using:

max_bootstrapped_demos=0

max_labeled_demos=0

This keeps the compiled program zero-shot (no demo examples embedded).

# What is the “gold prompt policy”?

For each case, the gold prompt policy is stored as the first system message inside case["traj"]:

# gold_policy = first message where role == "system"

We use gold policy only as a label for metrics/training. The reconstructor does not see it as input.

# Output format (what the model predicts)

The reconstructor predicts JSON:

{
  "auth_required": true/false,
  "auth_methods": ["email", "name_zip"],
  "one_user_per_conversation": true/false,
  "explicit_confirmation_before_db_write": true/false,
  "one_tool_call_at_a_time": true/false,
  "modify_or_cancel_only_if_pending": true/false,
  "return_only_if_delivered": true/false,
  "exchange_only_if_delivered": true/false,
  "no_hallucination_or_subjective_advice": true/false
}

# Repo structure
tau-bench/
├─ tau_bench/
├─ historical_trajectories/
├─ policies/
│  ├─ __init__.py
│  ├─ infer_prompt_policy_rules.py
├─ scripts/
│  ├─ train_infer_policy_rules.py
│  ├─ predict_one_case_rules.py
├─ artifacts/
│  ├─ infer_policy_rules_small/   (created after training)
└─ ...

Tau-bench schema used by these scripts

# Each JSON file under historical_trajectories/ is a list of cases.

For each case:

# 1) Task instruction (input)

case["info"]["task"]["instruction"]

# 2) Trajectory messages

case["traj"] is a list of messages with roles:

system, user, assistant, tool

# 3) Gold prompt policy (label)

first message in case["traj"] where role == "system"

# 4) Dialogue fed to the model (input)

all messages in case["traj"] except the system message

Important: tool calls in the dialogue

Tau-bench often contains assistant messages where:

content == null

tool_calls != null

Inference-time (predict_one_case_rules.py) behavior

The single-case predictor script preserves tool calls by serializing them as:

[assistant][tool_call] <tool_name> <arguments>

Tool outputs are serialized as:

[tool:<tool_name>] <tool_output>

Training-time (train_infer_policy_rules.py) behavior

The training script uses a simpler serializer that primarily prints [role] <content>.

Because tau-bench tool calls are sometimes stored in tool_calls with content=null, training dialogue serialization may omit some tool-call lines.

This does NOT stop training from running, because the training target is derived from the gold policy text (system message) and the model output is a compact JSON.

# Files overview
# 1) policies/infer_prompt_policy_rules.py

Defines the DSPy module that predicts policy_rules JSON from:

user_instruction

dialogue

No wiki input.

# 2) scripts/train_infer_policy_rules.py

loads cases from one trajectory JSON file

extracts:

user_instruction from info.task.instruction

gold prompt policy from the first system message in traj

dialogue from the rest of traj

splits into train/dev/test

compiles a zero-shot program using dspy.InferRules

evaluates dev/test rule-accuracy

saves compiled program under --out_dir (DSPy save or compiled.pkl fallback)

# 3) scripts/predict_one_case_rules.py

loads the compiled program from --out_dir

runs one case (--case_idx)

prints:

gold policy (truncated)

predicted JSON rules

# Setup
1) Use python3 (macOS)

Use python3 because python may not exist in your shell.

2) Set API key
export OPENAI_API_KEY="sk-..."
export LITELLM_MAX_PARALLEL_REQUESTS=1

3) Model selection (GPT-5)
export DSPY_MODEL="openai/gpt-5"

Important: DSPy constraint for GPT-5

DSPy treats GPT-5 as a reasoning model and requires:

temperature=1.0 (or None)

max_tokens>=16000 (or None)

The training script configures GPT-5 accordingly.

# Run a tiny sanity training (5 cases)
cd /Users/ooooo/Desktop/agent/tau-bench

export OPENAI_API_KEY="sk-..."
export DSPY_MODEL="openai/gpt-5"
export LITELLM_MAX_PARALLEL_REQUESTS=1

PYTHONPATH=. python3 -m scripts.train_infer_policy_rules \
  --data historical_trajectories/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --max_cases 5 \
  --train_frac 0.6 \
  --dev_frac 0.2 \
  --out_dir artifacts/infer_policy_rules_small


With 5 cases, train/dev/test is usually 3/1/1.

Note about “100%”

With dev/test size 1, “100%” means it matched that single example. Increase --max_cases for meaningful estimates.

# Predict one case (case-by-case)
cd /Users/ooooo/Desktop/agent/tau-bench
export OPENAI_API_KEY="sk-..."
export DSPY_MODEL="openai/gpt-5"

PYTHONPATH=. python3 -m scripts.predict_one_case_rules \
  --data historical_trajectories/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --case_idx 0 \
  --out_dir artifacts/infer_policy_rules_small

# Common issues
# 1) ModuleNotFoundError: policies

Run with:

PYTHONPATH=. python3 -m scripts.train_infer_policy_rules ...


and ensure:

ls policies/__init__.py

# 2) GPT-5 ValueError about temperature/max_tokens

Use GPT-5 with temperature=1.0 and max_tokens=16000 (already set in training script).

# 3) Data schema mismatch

If your trajectory file format changes, update:

record_to_example() in scripts/train_infer_policy_rules.py

extract_fields_tau_bench_case() in scripts/predict_one_case_rules.py

# example run
ooooo@ooooodeAir tau-bench % python3 -m scripts.train_infer_policy_rules \
  --data historical_trajectories/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --max_cases 5 \
  --train_frac 0.6 \
  --dev_frac 0.2 \
  --out_dir artifacts/infer_policy_rules_small
[train] total examples: 5
[train] train/dev/test: 3 1 1
[train] sample instruction: You are Yusuf Rossi in 19122. You want to know how many tshirt options are avail
[train] LM: openai/gpt-5
  0%|                                                     | 0/3 [00:00<?, ?it/s]
Bootstrapped 0 full traces after 0 examples for up to 1 rounds, amounting to 0 attempts.
Average Metric: 1.00 / 1 (100.0%): 100%|█████████| 1/1 [00:00<00:00, 250.38it/s]
2026/01/03 21:28:47 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:28:47 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 1 with score 100.0. Current best score: 100.0
Average Metric: 1.00 / 1 (100.0%): 100%|█████████| 1/1 [00:00<00:00, 652.81it/s]
2026/01/03 21:28:47 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:28:47 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 2 with score 100.0. Current best score: 100.0
Average Metric: 1.00 / 1 (100.0%): 100%|██████████| 1/1 [00:51<00:00, 51.33s/it]
2026/01/03 21:30:26 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:30:26 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 3 with score 100.0. Current best score: 100.0
Average Metric: 1.00 / 1 (100.0%): 100%|██████████| 1/1 [00:35<00:00, 35.50s/it]
2026/01/03 21:31:54 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:31:54 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 4 with score 100.0. Current best score: 100.0
Average Metric: 1.00 / 1 (100.0%): 100%|██████████| 1/1 [00:24<00:00, 24.89s/it]
2026/01/03 21:33:06 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:33:06 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 5 with score 100.0. Current best score: 100.0
Average Metric: 1.00 / 1 (100.0%): 100%|██████████| 1/1 [00:25<00:00, 25.37s/it]
2026/01/03 21:36:13 INFO dspy.evaluate.evaluate: Average Metric: 1.0 / 1 (100.0%)
2026/01/03 21:36:13 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 6 with score 100.0. Current best score: 100.0
2026/01/03 21:36:13 INFO dspy.teleprompt.infer_rules: Final best score: 100.0
Dev rule-accuracy:  1.000
Test rule-accuracy: 1.000
Saved compiled program to: artifacts/infer_policy_rules_small/
