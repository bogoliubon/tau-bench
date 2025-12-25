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
--wiki-path "/home/cw862/tau-bench/policy_refinement_all_tools_50traj_batch5_gpt-4o.json" \
--log-dir general_policy_gpt-4o-n_traj50_batch5"
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
--wiki-path "/home/cw862/tau-bench/policy_refinement_all_tools_50traj_batch5_gpt-5.json" \
--log-dir general_policy_gpt-5-n_traj50_batch5"
echo $command
eval $command