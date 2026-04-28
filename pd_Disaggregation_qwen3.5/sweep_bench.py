#!/usr/bin/env python3
"""
Sweep sglang bench_serving over a grid of parameters, then parse all results
into a single CSV file.

Usage:
    python sweep_bench.py [options]

Examples:
    python sweep_bench.py
    python sweep_bench.py --base-url http://127.0.0.1:8000 --num-prompts 256 -o my_results.csv
"""

import re
import csv
import sys
import argparse
import subprocess
from itertools import product
from pathlib import Path

# ── sweep grid ────────────────────────────────────────────────────────────────
# DEFAULT_MAX_CONCURRENCY = [16]
DEFAULT_MAX_CONCURRENCY = [8, 16, 32, 64, 128, 256, 512, 768, 1024]
DEFAULT_IO_PAIRS       = [(1024, 1024), (8192, 1024)]

# DEFAULT_IO_PAIRS       = [(8192, 1024), ]
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    "Request throughput (req/s)":     r"Request throughput \(req/s\):\s+([\d.]+)",
    "Input token throughput (tok/s)": r"Input token throughput \(tok/s\):\s+([\d.]+)",
    "Output token throughput (tok/s)":r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Mean E2E Latency (ms)":          r"Mean E2E Latency \(ms\):\s+([\d.]+)",
    "Mean TTFT (ms)":                 r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Mean TPOT (ms)":                 r"Mean TPOT \(ms\):\s+([\d.]+)",
}

FIELDS = [
    "max_concurrency",
    "random_input",
    "random_output",
    "Request throughput (req/s)",
    "Input token throughput (tok/s)",
    "Output token throughput (tok/s)",
    "Mean E2E Latency (ms)",
    "Mean TTFT (ms)",
    "Mean TPOT (ms)",
]


def parse_log(text: str, concurrency: int, input_len: int, output_len: int) -> dict:
    row = {
        "max_concurrency": concurrency,
        "random_input":    input_len,
        "random_output":   output_len,
    }
    for field, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        row[field] = m.group(1) if m else ""
    return row


def run_bench(base_url: str, concurrency: int,
              input_len: int, output_len: int, log_dir: Path) -> str:
    logfile = log_dir / f"c{concurrency}_i{input_len}_o{output_len}.log"
    # if concurrency <=16:
    #     num_prompts = 32
    # elif concurrency <= 64:
    #     num_prompts = 128
    # elif concurrency <= 512:
    #     num_prompts = concurrency*2
    # else:
    #     num_prompts = concurrency
        
    num_prompts = max(concurrency, 64)
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend",         "sglang",
        "--base-url",        base_url,
        "--dataset-name",    "random",
        "--max-concurrency", str(concurrency),
        "--num-prompts",     str(num_prompts),
        "--random-input",    str(input_len),
        "--random-output",   str(output_len),
        "--output-file", f'{log_dir}/tp8_isl{input_len}_osl{output_len}_conc{concurrency}.json',
    ]
    lines = []
    with open(logfile, "w", buffering=1) as lf:   # line-buffered
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()                             # flush each line to disk
            lines.append(line)
        proc.wait()
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Sweep sglang bench and collect results.")
    parser.add_argument("--base-url",       default="http://127.0.0.1:8000")
    parser.add_argument("--max-concurrency", type=int, nargs="+",
                        default=DEFAULT_MAX_CONCURRENCY,
                        metavar="N", help="List of max-concurrency values")
    parser.add_argument("--io-pairs", type=int, nargs="+",
                        default=[v for pair in DEFAULT_IO_PAIRS for v in pair],
                        metavar="N",
                        help="Flat list of (input, output) pairs, e.g. 1024 1024 8192 1024")
    parser.add_argument("--log-dir",        default="./mylog",
                        help="Directory to store per-run logs (default: ./mylog)")
    parser.add_argument("-o", "--output",   default="sweep_results.csv")
    args = parser.parse_args()

    flat = args.io_pairs
    if len(flat) % 2 != 0:
        parser.error("--io-pairs must have an even number of values (input output ...)")
    io_pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logs -> {log_dir}")

    grid = list(product(io_pairs, args.max_concurrency))
    total = len(grid)
    rows = []

    csv_file = open(args.output, "w", newline="", buffering=1)
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()
    csv_file.flush()

    try:
        for idx, ((input_len, output_len), concurrency) in enumerate(grid, 1):
            print(f"\n[{idx}/{total}] max-concurrency={concurrency}  input={input_len}  output={output_len}")
            output = run_bench(args.base_url, 
                               concurrency, input_len, output_len, log_dir)
            row = parse_log(output, concurrency, input_len, output_len)
            rows.append(row)

            writer.writerow(row)
            csv_file.flush()                       # flush CSV row to disk immediately
    finally:
        csv_file.close()

    print(f"\nDone. {total} runs -> {args.output}")


if __name__ == "__main__":
    main()
