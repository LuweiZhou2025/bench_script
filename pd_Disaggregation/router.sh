set -ex


export GLOO_SOCKET_IFNAME=enp193s0f1np1
export SGLANG_HOST_IP=10.2.224.6
# python -m sglang_router.launch_router \
#   --pd-disaggregation --mini-lb \
#   --prefill http://10.2.224.6:10002 \
#   --decode http://10.2.224.5:3002 \
#   --host 0.0.0.0 --port 8000

python -m sglang_router.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill http://10.2.224.6:10002 \
  --decode http://10.2.224.6:3002 \
  --host 0.0.0.0 --port 8000