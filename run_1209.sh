# command="python run.py \
# --agent-strategy tool-calling \
# --env retail \
# --model gpt-4o \
# --model-provider openai \
# --user-model gpt-4o \
# --user-model-provider openai \
# --user-strategy llm \
# --max-concurrency 1 \
# --task-split train \
# --num-trials 1"
# echo $command
# eval $command

# command="python run.py \
# --agent-strategy tool-calling \
# --env retail \
# --model gpt-5 \
# --model-provider openai \
# --user-model gpt-4o \
# --user-model-provider openai \
# --user-strategy llm \
# --max-concurrency 1 \
# --task-split train \
# --num-trials 1"
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
--task-split train \
--num-trials 1"
echo $command
eval $command

# command="python run.py \
# --agent-strategy tool-calling \
# --env airline \
# --model gpt-4o \
# --model-provider openai \
# --user-model gpt-4o \
# --user-model-provider openai \
# --user-strategy llm \
# --max-concurrency 1 \
# --task-split train \
# --num-trials 1"
# echo $command
# eval $command