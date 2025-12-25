
command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-4o \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 10 \
--task-split test \
--num-trials 1 \
--summarize-from-model gpt-4o \
--log-dir summarize"
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
--summarize-from-model gpt-5 \
--log-dir summarize"
echo $command
eval $command