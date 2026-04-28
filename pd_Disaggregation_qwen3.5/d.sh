
##Qwen3-235B
# export GLOO_SOCKET_IFNAME=enp193s0f1np1
# export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
# export NCCL_IB_TC=104
# export NCCL_IB_FIFO_TC=184
# export NCCL_IB_GID_INDEX=1
# export NCCL_CROSS_NIC=0
# export SGLANG_HOST_IP=10.2.224.5
# python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  \
#     --load-format dummy --disaggregation-mode decode --disaggregation-ib-device ${NCCL_IB_HCA} \
#     --host ${SGLANG_HOST_IP} --port 3002 --trust-remote-code --tp-size 8 --ep-size 8 --dp-size 8 --decode-log-interval 1 --watchdog-timeout 3600 \
#     --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --disaggregation-transfer-backend mori --moe-a2a-backend mori \
#     --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --max-running-requests 24 --chunked-prefill-size 16384 --cuda-graph-bs 1 2 3 --disable-radix-cache \
#     --mem-fraction-static 0.25

###Qwen3.5

export SGLANG_USE_AITER=1

# These env vars should adjust according to your environment
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export SGLANG_HOST_IP=10.2.224.5
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_TC=104
export NCCL_IB_FIFO_TC=184
export NCCL_IB_GID_INDEX=1
export NCCL_CROSS_NIC=0
export SGLANG_TORCH_PROFILER_DIR=/mywork/pd_Disaggregation/prof_decoder/

# export AMD_SERIALIZE_KERNEL=1
# export HIP_LAUNCH_BLOCKING=1
# export TORCH_USE_HIP_DSA=1


# export model_name=Qwen/Qwen3.5-397B-A17B
export model_name=Qwen/Qwen3.5-397B-A17B-FP8
# export model_name=amd/Qwen3.5-397B-A17B-MXFP4

#TP8 + EP1/EP8 + without DP attention
python3 -m sglang.launch_server --model-path ${model_name} --disaggregation-mode decode \
        --disaggregation-ib-device ${NCCL_IB_HCA} \
        --host ${SGLANG_HOST_IP} --port 3002 --trust-remote-code \
        --tp-size 8 --ep-size 1\
        --decode-log-interval 10 --watchdog-timeout 3600 \
        --load-balance-method round_robin  --attention-backend aiter \
        --disaggregation-transfer-backend mori  --mem-fraction-static 0.85 \
        --max-running-requests 2048 --chunked-prefill-size 16384 --disable-radix-cache  --cuda-graph-max-bs 1024 2>&1 | tee decoder.log

# DP attention + EP MOE

# CUDA_VISIBLE_DEVICES=4,5,6,7
# python3 -m sglang.launch_server --model-path ${model_name} --disaggregation-mode decode \
#         --disaggregation-ib-device ${NCCL_IB_HCA} \
#         --host ${SGLANG_HOST_IP} --port 3002 --trust-remote-code \
#         --tp-size 8 --ep-size 8  --dp-size 8 --enable-dp-attention --moe-a2a-backend mori --moe-dense-tp-size 1 \
#         --decode-log-interval 10 --watchdog-timeout 3600 \
#         --load-balance-method round_robin  --attention-backend aiter \
#         --disaggregation-transfer-backend mori  --mem-fraction-static 0.85 \
#         --max-running-requests 2048 --chunked-prefill-size 16384 --disable-radix-cache  --cuda-graph-max-bs 1024 >&1 | tee decoder.log