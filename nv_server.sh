model_path="/models--nvidia--Qwen3-235B-A22B-NVFP4/snapshots/21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2"
if [ $# -ge 1 ] && [ "$1" = "p" ]; then
        echo "profiling data in ./profiler/"
        export VLLM_TORCH_PROFILER_DIR=./profiler/
fi

max_model_len=22000 # Must be >= the input + the output token lengths.
max_num_seqs=2048 # max seqs in one schedule iteration
max_num_batched_tokens=73728 #max tokens in all the sequence.


export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
# export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=4190208
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve ${model_path} --host localhost --port 8000 --swap-space 64 --max-model-len ${max_model_len} \
                            -tp 2 -dp 1  --kv-cache-dtype fp8  --max-num-seqs ${max_num_seqs}  --gpu-memory-utilization 0.9 \
                            --max-num-batched-tokens ${max_num_batched_tokens} --no-enable-prefix-caching --async-scheduling
