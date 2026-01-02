# use j,k tuple in (12,0) (10,2), (8,4) (6,6) (4,8) for wiki-path and log-dir
for j in 12 10 8 6 4
do
    for i in {1..3}
    do
        k=$((12-j))
        log-dir="policy_refinement_${j}success_${k}failed_per_flow_run${i}_gpt-5"
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
        --wiki-path "/home/cw862/tau-bench/policy_refinement_${j}success_${k}failed_per_flow_gpt-5.json" \
        --log-dir ${log-dir}"
        echo $command
        eval $command
    done
done