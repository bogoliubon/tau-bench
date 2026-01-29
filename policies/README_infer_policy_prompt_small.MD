# README — PromptPolicy Reconstruction (Tau-Bench) via DSPy (Prompt Text, Small)

Goal
Reconstruct the hidden per-task “prompt policy” as a short natural-language system/policy prompt, using ONLY the observed dialogue.

What this “small” variant changes vs the JSON-rules version
1) Input: dialogue only
- No user_instruction is passed to the reconstructor.

2) Output: prompt-policy text
- The model outputs short policy sentences (e.g., “Require authentication before accessing orders.”),
  not a structured JSON vector.

3) Evaluation/metric: tau-bench binary reward (0/1)
- Candidates are evaluated by running tau-bench with the inferred prompt policy.
- 1 = success on the task, 0 = failure.

4) Manual splits by task_id
- You will manually pick:
  - 10 tasks for training
  - 5 tasks for testing
- These should collectively cover most policy constraints.

Wiki is NOT provided
As with your setup, tau_bench/envs/retail/wiki.md is not provided to the reconstructor.

Files
- policies/infer_prompt_policy_text_small.py
  DSPy module: dialogue -> prompt_policy (text)

- scripts/train_infer_policy_prompt_small.py
  Compiles a zero-shot DSPy program with InferRules using a binary reward metric.
  You must:
  (a) Fill TRAIN_TASK_IDS (10 tasks) and TEST_TASK_IDS (5 tasks)
  (b) Implement evaluate_prompt_policy_binary() by calling your tau-bench wrapper.

- scripts/predict_one_case_prompt_small.py
  Loads compiled program, runs one trajectory file, prints inferred prompt policy, optional binary eval.

Supported trajectory schemas
A) Big historical files:
- list[case], where case["traj"] is a list of messages (system/user/assistant/tool)

B) Human trajectories:
- single object or list of objects with:
  { "task_id": "...", "messages": [ ... ] }

Setup
export OPENAI_API_KEY="sk-..."
export DSPY_MODEL="openai/gpt-5"
export LITELLM_MAX_PARALLEL_REQUESTS=1
cd /path/to/tau-bench
PYTHONPATH=. python3 -m scripts.train_infer_policy_prompt_small --data_dir data/tau2/human_trajectories/retail

Important TODO (required)
Implement evaluate_prompt_policy_binary() in:
- scripts/train_infer_policy_prompt_small.py

It must:
- Run tau-bench for a single task_id with the predicted prompt policy as the system prompt
- Return 1 if the agent succeeds, else 0

After you add these files
1) Add your 10 training task_ids and 5 testing task_ids.
2) Wire tau-bench wrapper into evaluate_prompt_policy_binary().
3) Run training and then debug individual cases with predict_one_case_prompt_small.py.
