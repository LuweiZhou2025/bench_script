# python3 -m sglang.bench_serving \
#   --backend sglang \
#   --base-url http://127.0.0.1:8000 \
#   --dataset-name random \
#   --num-prompts 1 \
#   --random-input 256 \
#   --random-output 2

python3 -m sglang.bench_serving \
  --backend sglang \
  --host 10.2.224.6 \
  --port 30003 \
  --num-prompts 1 \
  --random-input 32 \
  --random-output 2