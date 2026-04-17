# step by step  to verify PD disaggregation:
a2abackend should only be valid in DP attention + EP MOE case.

## First ensure most of none-PD case can run, single node can also use DP attention + MOE EP + a2a backend.
1, one node no PD, TP=8 EP=1 without DP attention
2, one node no PD, TP=8 EP=8 without DP attention
3, one node no PD, TP=8 EP=8 with DP attention , a2abakcend = none
4, one node no PD, TP=8 EP=8 with DP attention , a2abakcend = mori

## Second try PD case, First try  --disaggregation-transfer-backend mooncake then mori.
1, 1P1D, TP=8 EP=1 without DP attention
2, 1P1D, TP=8 EP=8 without DP attention
3, 1P1D, TP=8 EP=8 with DP attention , a2abakcend = none
4, 1P1D, TP=8 EP=8 with DP attention , a2abakcend = mori


# If suspect a2a backendissue for DP attention, use --moe-a2a-backend none. 
# If suspect mori-io PD problem, use --disaggregation-transfer-backend mooncake
# --moe-a2a-backend is decoupled with --disaggregation-transfer-backend. no need to be same.

# launch.json to use python debugger. very smooth to debugging.

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
        "configurations": [
        {
            "name": "Python Debugger: Module",
            "type": "debugpy",
            "request": "launch",
            "module": "sglang.launch_server",
            "justMyCode": false,
            "args": [
                "--model-path",
                "Qwen/Qwen3.5-397B-A17B-FP8",
                "--port",
                "9700",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--context-length",
                "262144",
                "--reasoning-parser=qwen3",
                "--attention-backend=triton",
                "--disable-radix-cache",
                "--disable-cuda-graph",
                "--watchdog-timeout=3600"
                //"--cuda-graph-max-bs=64"
            ],
            "env": {
                "OPTFLAG": "w8a8_gemm,moe",
                "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
                "CUDA_VISIBLE_DEVICES": "6,7", //"4,5,6,7",
                //"HIP_VISIBLE_DEVICES":"6,7", //,6,7"
            }
        },
        {
            "name": "PD: Module",
            "type": "debugpy",
            "request": "launch",
            "module": "sglang.launch_server",
            "args": [
                "--model-path", "Qwen/Qwen3.5-397B-A17B-FP8",
                "--load-format", "dummy",
                "--disaggregation-mode", "prefill",
                "--disaggregation-ib-device", "ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7",
                "--host", "10.2.224.9",
                "--port", "30002",
                "--trust-remote-code",
                "--tp-size", "8",
                "--ep-size", "1",
                "--decode-log-interval", "1",
                "--watchdog-timeout", "3600",
                "--ep-dispatch-algorithm", "fake",
                "--load-balance-method", "round_robin",
                "--attention-backend", "triton",
                "--disaggregation-transfer-backend", "mori",
                // "--moe-a2a-backend", "mori",
                // "--enable-dp-attention",
                "--moe-dense-tp-size", "1",
                "--enable-dp-lm-head",
                //"--mem-fraction-static", "0.8",
                "--max-running-requests", "24",
                "--chunked-prefill-size", "16384",
                "--cuda-graph-bs", "1", "2", "3",
                "--mem-fraction-static", "0.8",
                "--disable-radix-cache"],
            "env": {
                "GLOO_SOCKET_IFNAME": "enp193s0f1np1",
                "NCCL_IB_HCA": "ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7",
                "NCCL_IB_TC": "104",
                "NCCL_IB_FIFO_TC": "184",
                "NCCL_IB_GID_INDEX": "1",
                "NCCL_CROSS_NIC": "0",
                "SGLANG_HOST_IP": "10.2.224.9"
            }
        },
    ]
}
```