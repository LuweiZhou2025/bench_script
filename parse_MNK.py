import re

def extract_MNK_from_logfile(log_file_path):
    """
    从日志文件中提取所有 M、N、K 参数
    """
    results = []
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # 移除ANSI颜色代码
            ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
            clean_line = ansi_escape.sub('', line)
            
            # 查找 M:N:K 模式
            pattern = r'M:(\d+),\s*N:(\d+),\s*K:(\d+), not found tuned config'
            match = re.search(pattern, clean_line)
            
            if match:
                M, N, K = match.groups()
                if [M,N,K] not in results:
                    results.append([
    return results

# 使用示例
log_results = extract_MNK_from_logfile('./bench_script/server.log')

# nk_filter = [['1792','6144'],['6144','1536'],['160','6144']], qkv and out projeciton
nk_filter = [['1792','6144'],['6144','1536'],['160','6144']]

mlist = []
# print(nk_filter)
for result in log_results:
    # print(f"{result[0]}, {result[1]}, {result[2]}")
    nk_item = [result[1],  result[2]]
    if nk_item in nk_filter:
        print(f"{result[0]}, {result[1]}, {result[2]}")
        if int(result[0]) not in mlist:
            mlist.append(int(result[0]))
# print('----------------------------------------')
# print(mlist)