
import os
import glob
import re


class info:
    def pattern_int(keyword):
        keyword=keyword.replace('(', r'\(')
        keyword=keyword.replace(')', r'\)')
        return r'\|\s*'+keyword+r'\s*\|\s*(\d*)\s*'

    def pattern_float(keyword):
        keyword=keyword.replace("(", "\(")
        keyword=keyword.replace(")", "\)")
        return r'\|\s*'+keyword+r'\s*\|\s*(\d*.\d\d\d\d)\s*'

    pat = {
        "test":re.compile("\|\s*test\s*\|\s*(\d*)\s*"),
        "Requests":re.compile(pattern_int("Success Requests         ")),
        "GTPS":re.compile(pattern_float("Generate Tokens  Per Second(GTPS)")),
        "TPS":re.compile(pattern_float("Total    Tokens  Per Second(TPS)")),
        "QPS":re.compile(pattern_float("Queries Per Second(QPS)(All requests finished)")),
        "Concurrency":re.compile(pattern_int("Concurrency        ")),
        "TTFT":re.compile(pattern_float("Average  Prefill Time (TTFT) ")),
        "TPOT":re.compile(pattern_float("Average  Decode Time (TPOT) ")),
    }
    def __init__(self, log_text) -> None:
        self.build = []
        for k,v in info.pat.items():
            setattr(self, k, float ('nan'))
        for line in log_text.splitlines():
            for k,v in info.pat.items():
                m = v.match(line)
                if (m):
                    setattr(self, k, float(m.group(1)))

    def __repr__(self) -> str:
        # return f"successfuly_prompts={self.successfuly_prompts} GTPS={self.GTPS}, TPS={self.TPS} QPS={self.QPS}  Concurrency={self.Concurrency} TTFT={self.TTFT} TPOT={self.TPOT}\n"
        return f"{self.GTPS}            {self.TPS}          {self.QPS}          {self.Concurrency}          {self.TTFT}             {self.TPOT}         {self.Requests}\n"

model_path="/models/Qwen3-235B-A22B-Instruct-2507-MXFP4"
model="Qwen3-235B-A22B-Instruct-2507-MXFP4"

# use the huggingface format model.
# model_path=../huggingface/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/xxxxxxxxxxxxxxxxxxx/
# !!!!!{model} is needs to be updated because kunlun-benchmark would use {model} as result file name '{model}_normal_distribution_unknown_server_vllm_tp-1....'
# model="xxxxxxxxxxxxxxxxxxx"

token_list=[# 3.0~3.6k/0.3~0.5k
            (3000, 3600, 300, 500, [128])
            # # 16~20k/0.3~0.5k
            # (16000, 20000, 300, 500, [8, 16,32]),
            # # 0.8~1k/1.6~2k
            # (800, 1000, 1600, 2000, [16, 32, 64]),
            # # 3.6~4.4k/1.8~2.2k
            # (3600, 4400, 1800, 2200, [16, 32,64]),
            # # 11~15k/2.5~2.9k
            # (11000, 15000, 2500, 2900, [8, 16, 32])
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
        if (concurrency < 256):
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
