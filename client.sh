model_path=/models/Qwen3-Coder-480B-A35B-Instruct-FP8
input_tokens=8000
output_tokens=1000
max_concurrency=64
num_prompts=192

/workdir/kunlun-benchmark/kunlun-benchmark vllm server \
    --port 8000 \
    --work_mode manual \
    --max_input_len ${input_tokens} \
    --min_input_len ${input_tokens} \
    --max_output_len ${output_tokens} \
    --min_output_len ${output_tokens}  \
    --concurrency  ${max_concurrency}\
    --query_num ${num_prompts} \
    --result_dir /workdir/log/ \
    --model_path ${model_path} \
    --is_sla True \
    --sla_decode 50 \
    --sla_prefill 3000
# vllm bench serve --host localhost --port 8000 --model ${model_path} --dataset-name random --random-input-len ${input_tokens} --random-output-len ${output_tokens} --max-concurrency ${max_concurrency} --num-prompts ${num_prompts} --percentile-metrics ttft,tpot --ignore-eos
#profile
# vllm bench serve --profile --host localhost --port 8000 --model ${model_path} --dataset-name random --random-input-len ${input_tokens} --random-output-len ${output_tokens} --max-concurrency ${max_concurrency} --num-prompts ${num_prompts} --percentile-metrics ttft,tpot --ignore-eos


