
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "python如何获取一个list的升序排列的对应元素的下标列表？"}
    ],
    "max_tokens": 10240,
    "temperature": 0.8
  }'