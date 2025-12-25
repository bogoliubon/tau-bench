command="python iterative_policy_refinement.py \
  --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --n_traj 50 \
  --model_name 'gpt-4o' \
  --output_path general_policy_gpt-4o.json \
  --seed 42"
echo $command
eval $command

command="python iterative_policy_refinement.py \
  --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --n_traj 50 \
  --model_name 'gpt-5' \
  --output_path general_policy_gpt-5.json \
  --seed 42"
echo $command
eval $command