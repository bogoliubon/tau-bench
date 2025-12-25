# run the exchange tasks w/ exchange prompt
command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-4o \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 5 \
--task-split test \
--num-trials 1 \
--wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_modify_30traj_gpt-4o.json" \
--task-ids 3 4 15 17 20 21 22 34 36 37 40 41 42 44 56 60 61 63 71 72 79 85 86 87 97 98 102 110 111 112 113 \
--log-dir modify"
echo $command
eval $command

command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-4o \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 5 \
--task-split test \
--num-trials 1 \
--wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_modify_30traj_gpt-5.json" \
--task-ids 3 4 15 17 20 21 22 34 36 37 40 41 42 44 56 60 61 63 71 72 79 85 86 87 97 98 102 110 111 112 113 \
--log-dir modify"
echo $command
eval $command