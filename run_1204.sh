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
--task-ids 0 1 6 7 8 9 18 19 27 45 49 52 58 70 75 77 80 93 94 95 96 100 106 107"
echo $command
eval $command