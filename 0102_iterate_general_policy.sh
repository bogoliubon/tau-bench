SEED=42
RESULTS_PATH=/home/cw862/tau-bench/train/tool-calling-gpt-4o-0.0_range_0--1_user-gpt-4o-llm_0101012317.json
MODEL=gpt-5

# Baseline
cmd="python iterative_policy_refinement.py \
    --results_path $RESULTS_PATH \
    --n_successful_per_flow 12 --n_failed_per_flow 0 \
    --model_name $MODEL --seed $SEED"
echo $cmd
eval $cmd

# Setting 1: Mostly successful
cmd="python iterative_policy_refinement.py \
    --results_path $RESULTS_PATH \
    --n_successful_per_flow 10 --n_failed_per_flow 2 \
    --model_name $MODEL --seed $SEED"
echo $cmd
eval $cmd

# Setting 2: Balanced
cmd="python iterative_policy_refinement.py \
    --results_path $RESULTS_PATH \
    --n_successful_per_flow 6 --n_failed_per_flow 6 \
    --model_name $MODEL --seed $SEED"
echo $cmd
eval $cmd

# Setting 3: More successful
cmd="python iterative_policy_refinement.py \
    --results_path $RESULTS_PATH \
    --n_successful_per_flow 8 --n_failed_per_flow 4 \
    --model_name $MODEL --seed $SEED"
echo $cmd
eval $cmd

# Setting 4: More failed
cmd="python iterative_policy_refinement.py \
    --results_path $RESULTS_PATH \
    --n_successful_per_flow 4 --n_failed_per_flow 8 \
    --model_name $MODEL --seed $SEED"
echo $cmd
eval $cmd