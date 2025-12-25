command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-4o \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 1 \
--num-trials 1 \
--task-ids 54 106 108"
echo $command
eval $command


command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-5 \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 1 \
--num-trials 1 \
--task-ids 54 106 108"
echo $command
eval $command