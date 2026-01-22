model_path=/models/Qwen3-235B-A22B-Instruct-2507-MXFP4
PROFILE=""
if [ $# -ge 1 ] && [ "$1" = "p" ]; then
    PROFILE="--profile"
fi

input_tokens=24000
output_tokens=5
max_concurrency=64
num_prompts=128

vllm bench serve ${PROFILE} --host localhost --port 8000 --model ${model_path} --dataset-name random --random-input-len ${input_tokens} --random-output-len ${output_tokens} --max-concurrency ${max_concurrency} --num-prompts ${num_prompts} --percentile-metrics ttft,tpot --ignore-eos
