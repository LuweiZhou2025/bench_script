model_path=/models/Qwen3-235B-A22B-Instruct-2507-MXFP4
if [ $# -ge 1 ] && [ "$1" = "p" ]; then
        echo "profiling data in ./profiler/"
        export VLLM_TORCH_PROFILER_DIR=./profiler/
fi

#hugging face HUB format
# model_path=../huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/xxxxxxxxxxxxxxxxxxx/
max_model_len=22000 # Must be >= the input + the output token lengths.
max_num_seqs=2048 # max seqs in one schedule iteration
max_num_batched_tokens=73728 #max tokens in all the sequence.

tensor_parallel_size=8
# TORCH_NCCL_BLOCKING_WAIT=true
TORCH_NCCL_BLOCKING_WAIT=true VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4  SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1  \
        vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} \
                          -tp ${tensor_parallel_size} --max-num-seqs ${max_num_seqs} --kv-cache-dtype fp8 --gpu-memory-utilization 0.73 \
                            --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling
#PROFILE VLLM
# VLLM_TORCH_PROFILER_DIR=xxx SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_USE_AITER_MOE=1  vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} -dp ${data_parallel_size} --enable-expert-parallel --max-num-seqs ${max_num_seqs} \
#                                                                                                                             --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --quantization fp8 --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling
