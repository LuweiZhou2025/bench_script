model_path=/models/Qwen3-Coder-480B-A35B-Instruct-FP8
max_model_len=22000 # Must be >= the input + the output token lengths.
max_num_seqs=2048 # max seqs in one schedule iteration
max_num_batched_tokens=72K #max tokens in all the sequence. 
tensor_parallel_size=8
# data_parallel_size=8

#SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1  vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} --tensor-parallel-size ${tensor_parallel_size} --max-num-seqs ${max_num_seqs} --enable-expert-parallel --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --quantization fp8 --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling'



SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1  vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} -tp ${tensor_parallel_size} --enable-expert-parallel --max-num-seqs ${max_num_seqs} \
                                                                                                                            --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --quantization fp8 --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling

# SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1  vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} -dp ${data_parallel_size} --enable-expert-parallel --max-num-seqs ${max_num_seqs} \
#                                                                                                                             --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --quantization fp8 --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling
