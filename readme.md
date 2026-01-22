
## docker run command MI355
```
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
   -it \
    -v  /mnt/nfs/huggingface:/root/.cache/huggingface \
    -v  /mnt/nfs:/nfs \
    -v /mnt/nfs/luwei/:/mywork  --workdir /mywork \
    -e "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    -p 8000:8000 \
    --ipc=host \
    --name mi355_alipoc \
    --entrypoint /bin/bash \
   rocm/ali-private:ubuntu22.04_rocm7.0.1_gfx950_cp312_vllm_e3c0995_aiter_5f4c65e_20251210
```

## podman run command MI355
```
podman run -it --name alipoc_355  --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add keep-groups --ipc=host -v /mnt/nfs/huggingface/:/models -v /mnt/nfs:/nfs - -v /mnt/nfs/luwei:/workdir --workdir /workdir rocm/ali-private:ubuntu22.04_rocm7.0.1_gfx950_cp312_vllm_e3c0995_aiter_5f4c65e_20251210  bash
```

## docker start
```
docker start -i mi355_alipoc
docker exec -it mi355_alipoc bash
```

## build kunlun with 3.12 version python. kunlun version would introduce some perf regression. So ensure right version matched with python verison
```
git clone https://github.com/LuweiZhou2025/bench_script.git
cd ./bench_script
tar -zxvf ./kunlun-benchmark.tar.gz
cd kunlun-benchmark
./build.sh
```
#some time would have pypi fetch issue. Just need to appliy follow changes in build.sh
```
#pip install poetry -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install poetry -i  https://pypi.org/simple/
```

## applied the vllm fp8 limiation on MI355.
https://github.com/ROCm/vllm/blob/dev/perf/vllm/_aiter_ops.py#L702
## run command
```
./server.sh
python ./run_client.py
```


# nv docker

docker run --runtime nvidia --gpus all \
   -it \
    -v /data/huggingface/models--nvidia--Qwen3-235B-A22B-NVFP4:/models--nvidia--Qwen3-235B-A22B-NVFP4 \
    -v /data/huggingface:/models \
    -v $HOME:/workdir --workdir /workdir \
    -p 8006:8006 \
    --ipc=host \
   --name luwei_vllm_cuda13 \
    --entrypoint /bin/bash \
    nvcr.io/nvidia/vllm:25.12.post1-py3




