command="python run.py \
--agent-strategy tool-calling \
--env retail \
--model gpt-4o \
--model-provider openai \
--user-model gpt-5 \
--user-model-provider openai \
--user-strategy llm \
--max-concurrency 1 \
--num-trials 1"
echo $command
eval $command