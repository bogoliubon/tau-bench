# run the exchange tasks w/ exchange prompt
# command="python run.py \
# # --agent-strategy tool-calling \
# # --env retail \
# # --model gpt-4o \
# # --model-provider openai \
# # --user-model gpt-4o \
# # --user-model-provider openai \
# # --user-strategy llm \
# # --max-concurrency 5 \
# # --task-split test \
# # --num-trials 1 \
# # --wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_return_delivered_order_items_30traj_gpt-4o.json" \
# # --task-ids 2 5 18 19 26 46 47 48 51 53 73 82 83 84 89 109 \
# # --log-dir return"
# echo $command
# eval $command

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
--wiki-path "/home/cw862/tau-bench/iterative_policy_refinement/policy_refinement_return_delivered_order_items_30traj_gpt-5.json" \
--task-ids 2 5 18 19 26 46 47 48 51 53 73 82 83 84 89 109 \
--log-dir return"
echo $command
eval $command