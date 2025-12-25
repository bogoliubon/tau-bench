# command="python batch_refine.py \
#   --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
#   --n_traj 50 \
#   --model_name 'gpt-4o' \
#   --seed 42 \
#   --batch_size 5"
# echo $command
# eval $command

# command="python batch_refine.py \
#   --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
#   --n_traj 50 \
#   --model_name 'gpt-5' \
#   --seed 42 \
#   --batch_size 5"
# echo $command
# eval $command

# command="python batch_refine.py \
#   --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
#   --n_traj 50 \
#   --model_name 'gpt-4o' \
#   --seed 42 \
#   --batch_size 10"
# echo $command
# eval $command

# command="python batch_refine.py \
#   --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
#   --n_traj 50 \
#   --model_name 'gpt-5' \
#   --seed 42 \
#   --batch_size 10"
# echo $command
# eval $command

command="python batch_refine.py \
  --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --n_traj 50 \
  --model_name 'gpt-4o' \
  --seed 42 \
  --batch_size 3"
echo $command
eval $command

command="python batch_refine.py \
  --results_path /home/cw862/tau-bench/results/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_1118113344.json \
  --n_traj 50 \
  --model_name 'gpt-5' \
  --seed 42 \
  --batch_size 3"
echo $command
eval $command