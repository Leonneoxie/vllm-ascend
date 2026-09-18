# W4A4 量化精度基线（Qwen3.5-9B）

> 16 场景 × 4 指标的正式评测矩阵（2026-09 收官），供新算法接入者对照参考。
> 接入新算法后，请在**相同契约**下复测并对表，判断精度是否达标。

## 一、评测契约（复现前提）

| 项 | 值 |
|---|---|
| 数据集 | MATH-500（500 题）/ MMLU-Pro（12032 题）/ LiveCodeBench release_latest（1055 题） |
| PPL | WikiText-2，AMCT 标准化评测 |
| 解码 | `do_sample=true, temperature=1.0, top_k=20, top_p=0.95, seed=42, max_tokens=8192, enable_thinking=False` |
| 工具 | EvalScope 1.10.0，`gpu_memory_utilization=0.90` |
| 量化 | W4A4 MXFP4 `group_size=32`；RHT `matrix_size=128`（Rademacher signs seed=0 + 归一化 FWHT） |
| 硬件 | Ascend 910，TP=1，BF16 serve（`--quantization ascend`） |

## 二、完整精度矩阵

| 场景 | PPL ↓ | MATH-500 ↑ | MMLU-Pro ↑ | LiveCodeBench ↑ |
|---|---:|---:|---:|---:|
| **BF16（基线）** | 8.171 | 94.0% | 79.31% | 58.39% |
| RTN attn-only | 9.297 | 86.4% | 74.38% | 40.19% |
| RTN mlp-only | 8.509 | 92.0% | 76.46% | 50.05% |
| RTN attn-mlp | 9.559 | 79.0% | 69.21% | 31.00% |
| FlatQuant attn-only | 7.812 | 92.4% | 77.95% | 51.28% |
| FlatQuant mlp-only | 8.239 | 94.0% | 78.92% | 58.67% |
| FlatQuant attn-mlp | 7.972 | 91.0% | 77.53% | 53.18% |
| OmniQuant attn-only | 8.105 | 93.0% | 76.37% | 46.26% |
| OmniQuant mlp-only | 8.774 | 92.6% | 77.90% | 57.63% |
| OmniQuant attn-mlp | 8.700 | 91.6% | 74.98% | 46.64% |
| LHT attn-only | 7.801 | 88.4% | 75.40% | 45.40% |
| LHT mlp-only | 8.397 | 93.2% | 78.13% | 54.31% |
| LHT attn-mlp | 8.256 | 88.0% | 73.94% | 42.84% |
| RHT attn-only | 8.029 | 86.6% | 73.53% | 39.91% |
| RHT mlp-only | 8.539 | 93.2% | 76.50% | 51.37% |
| RHT attn-mlp | 8.648 | 80.6% | 67.84% | 30.52% |

注：`w4a4-w8a8-mixed` 配置未包含在本矩阵中，待补。

## 三、关键结论

1. **FlatQuant 数据集综合最优**：attn-mlp 场景 MMLU-Pro 77.53% / LCB 53.18% 为五算法最佳；mlp-only 场景 MATH-500 94.0% 追平 BF16、LCB 58.67% 反超 BF16（58.39%）
2. **学习型变换 PPL 可低于 BF16**：LHT attn 7.801、FlatQuant attn 7.812 / attn-mlp 7.972，均优于 BF16 的 8.171
3. **RHT 的价值是零训练成本抑制离群值**：vs RTN，PPL attn 9.297→8.029、attn-mlp 9.559→8.648；mlp-only 数据集净收益（MATH-500 +1.2pt、LCB +1.3pt），attn 场景数据集与 RTN 基本持平
4. **attn 是 W4A4 主要损失源**：RTN vs BF16，attn-only MATH-500 -7.6pt / LCB -18.2pt，远大于 mlp-only 的 -2.0pt / -8.3pt
5. **学习型变换全量化收益 +12pt 级**：attn-mlp MATH-500，RTN 79.0% → OmniQuant 91.6% / FlatQuant 91.0%
6. **重度量化的数据集收益需要学习型变换**：RHT attn-mlp vs RTN 仅 MATH-500 +1.6pt，MMLU-Pro -1.4pt / LCB -0.5pt

## 四、数据版本

数据产自与本分支 `fake-mx-lite` 等价的实现，等价性由 `test_fake_mx_amct_crosscheck.py` 守护。

## 五、新算法接入验证路径

1. **单元测试**：在 `tests/ut/quantization/methods/` 补充新算法单测（对照 `test_fake_mx_expert_transforms.py` 惯例）
2. **快验**：按 [quick-verify-guide.md](./quick-verify-guide.md) 跑 MATH-500 Level 3 子集（105 题，~10min/场景）
3. **小样本回归**：`scripts/eval/run_limited_eval.py <port> <work_dir> <dataset> <limit>`（生成配置与全量脚本完全一致）
4. **全量对基线**：`scripts/eval/run_three_datasets.sh` + PPL，同契约复测本矩阵；任一指标偏离 >2% 需排查
