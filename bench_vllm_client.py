import json

mean_ttft_ms
mean_tpot_ms
total_token_throughput
output_throughput
max_concurrent_requests
with open('openai-infqps-concurrency16-Qwen3-Coder-480B-A35B-Instruct-FP8-20251215-080259.json') as f:
    d = json.load(f)
    print(type(d))
    print(list(d.keys()))
    print(type(d.keys()))
    # for key, value in d.items():
    #     print(key)
    

import os
import glob


model_path="/models/Qwen3-Coder-480B-A35B-Instruct-FP8"
model="Qwen3-Coder-480B-A35B-Instruct-FP8"
metrics=["output_throughput", "total_token_throughput", "max_concurrent_requests", "mean_ttft_ms" , "mean_tpot_ms"]
# (intputmin, inputmax, outputmin, outputmax)
token_list=[# 3.0~3.6k/0.3~0.5k
            # (3000, 3600, 300, 500, [64, 128,256]),
            # # 0.8~1k/1.6~2k
            # (800, 1000, 1600, 2000, [128,256,512]),
            # 3.6~4.4k/1.8~2.2k
            (3600, 4400, 1800, 2200, [128, 256, 512]),
            # 11~15k/2.5~2.9k
            (11000, 15000, 2500, 2900, [128,256]),
            # # 16~20k/0.3~0.5k
            (16000, 20000, 300, 500, [16, 32])
    ]


num_prompts=0
dir="log"
if os.path.exists(dir):
    os.system(f'rm -rf ./{dir}')
os.makedirs(dir, exist_ok=True)
summary=open(f'{dir}/summary.md','w')
for iopair in token_list:
    imin,imax,omin,omax, concurency_list=iopair
    assert omin<=omax and imin<=imax, f'Invalid token range: {iopair}'
    summary.write(f'\ninput: [{imin}-{imax}] / output: [{omin}-{omax}]: \n')
    summary.write(f'--------------------------------------------------------------------------\n')
    summary.write(f'GTPS(tokens/s)   TPS(tokens/s)    QPS(reqs/s)    Concurrency       TTFT(ms)      TPOT(ms)      Requestd done\n')
    summary.flush()

    logdir =f'./{dir}/{imin}_{imax}+{omin}_{omax}'
    if not os.path.exists(logdir):
        os.makedirs(logdir,  exist_ok=True)
    for filename in os.listdir(logdir):
        file_path = os.path.join(logdir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    for concurrency in concurency_list:
        # save some timing
        if (concurrency < 512):
            num_prompts=concurrency*3
        else:
            num_prompts = concurrency*2
        # no need to run big concurrency with big input.
        # if imin == 16000 and concurrency >=64:
        #     continue
        print(f"Running test with input tokens [{imin},{imax}], output tokens [{omin},{omax}], concurrency {concurrency}")
        client_cmd = f'''/workdir/kunlun-benchmark/kunlun-benchmark vllm server \
            --port 8000 \
            --work_mode manual \
            --max_input_len {imax} \
            --min_input_len {imin} \
            --max_output_len {omax} \
            --min_output_len {omin}  \
            --concurrency  {concurrency}\
            --query_num {num_prompts} \
            --result_dir {logdir} \
            --model_path {model_path} \
            '''
            # --is_sla True \
            # --sla_decode 50 \
            # --sla_prefill 3000'''
        print(f'{client_cmd}\n, concurrency:{concurrency}, input:{imin}-{imax}, output:{omin}-{omax}\n')
        ret=os.system(client_cmd)
        if ret != 0:
            print(f"!!!!!!!!!!! input: [{imin}-{imax}] / output: [{omin}-{omax}]  concurrency:{concurrency} failed with return code {ret}")
            continue
        
        pattern = f'{model}_normal_distribution_unknown_server_vllm_tp-1_{imax}*{imin}_{omax}*{omin}_{concurrency}_{num_prompts}*_ai_perf_benchmark.md'
        filter_file = glob.glob(f'{logdir}/{pattern}')
        assert len(filter_file) == 1, f"Expected one result file, found {len(filter_file)} files. list is {filter_file}"
        pref_data=''
        with open(f'{filter_file[0]}') as f:
            perf_data = f.read()
        i0 = info(perf_data)
        summary.write(i0.__str__())
        summary.flush()
        
    summary.write(f'--------------------------------------------------------------------------\n')
    summary.flush()


summary.close()

