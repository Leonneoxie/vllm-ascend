#!/usr/bin/env python3
"""小样本快速验证评测脚本（不跑全量，仅验证正确性）。

用法: python run_limited_eval.py <port> <work_dir> <dataset> <limit>
示例: python run_limited_eval.py 8001 ./outputs/rtn_quick mmlu_pro 10
      python run_limited_eval.py 8001 ./outputs/fq_quick live_code_bench 5

参数:
    port     目标服务端口
    work_dir 输出目录
    dataset  mmlu_pro / live_code_bench / math_500
    limit    每个子集的样本数（evalscope 语义: per-subset）

说明:
    - MMLU-Pro 有 14 个学科子集, limit=10 即 140 题
    - LiveCodeBench 需要 docker sandbox (python:3.11-slim)
    - 生成配置与全量脚本 (run_math500 / run_mmlu_pro / run_livecodebench)
      完全一致, 仅限样本数, 便于全量前的快速回归

环境要求:
  python 需安装 evalscope, 可选设置 MODELSCOPE_CACHE 指向数据集缓存
"""

import sys

from evalscope import TaskConfig, run_task

port, work_dir, dataset, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

if dataset == "live_code_bench":
    dataset_args = {"live_code_bench": {"subset_list": ["release_latest"], "extra_params": {}}}
    batch_size = 8
    use_sandbox = True
    sandbox = {"enabled": True, "engine": "docker", "default_config": {"image": "python:3.11-slim"}}
else:
    dataset_args = {dataset: {}}
    batch_size = 32
    use_sandbox = False
    sandbox = None

task = TaskConfig(
    model="qwen3.5",
    model_id="qwen3.5",
    api_url=f"http://localhost:{port}/v1/chat/completions",
    api_key="EMPTY",
    datasets=[dataset],
    dataset_args=dataset_args,
    eval_type="openai_api",
    eval_batch_size=batch_size,
    limit=limit,
    generation_config={
        "batch_size": batch_size,
        "max_tokens": 8192,
        "n": 1,
        "stream": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "do_sample": True,
        "timeout": 600,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    use_sandbox=use_sandbox,
    sandbox=sandbox,
    work_dir=work_dir,
    seed=42,
)
run_task(task)
