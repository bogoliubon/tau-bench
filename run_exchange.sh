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
--wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_exchange_delivered_order_items_30traj_gpt-4o.json" \
--task-ids 0 1 6 7 8 9 27 45 49 52 58 70 75 77 80 93 94 95 106 107 \
--log-dir exchange"
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
--wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_exchange_delivered_order_items_30traj_gpt-5.json" \
--task-ids 0 1 6 7 8 9 27 45 49 52 58 70 75 77 80 93 94 95 106 107 \
--log-dir exchange"
echo $command
eval $command