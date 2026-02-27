export GLOO_SOCKET_IFNAME=enp193s0f1np1
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_TC=104
export NCCL_IB_FIFO_TC=184
export NCCL_IB_GID_INDEX=1
export NCCL_CROSS_NIC=0
 
size=8
python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  \
    --load-format dummy \
    --host 10.2.224.6 --port 30003 --trust-remote-code --tp-size 8 --ep-size 8 --dp-size 8 --decode-log-interval 1 --watchdog-timeout 3600 \
    --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --attention-backend aiter --moe-a2a-backend mori \
    --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --max-running-requests 24 --chunked-prefill-size 16384 --cuda-graph-bs 1 2 3 --disable-radix-cache \
    --mem-fraction-static 0.25 2>&1 | tee debug.log