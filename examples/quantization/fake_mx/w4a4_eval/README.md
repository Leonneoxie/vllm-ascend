# Qwen3.5-9B W4A4 伪量化评测指南

当前分支仅保留 RTN、FlatQuant、OmniQuant、LHT、RHT（Linear 与 MoE 均支持）。实现结构与接入步骤见 [fake_mx README](../README.md)。历史扩展文档中的 AutoRound/LWC/LAC 不再适用于此分支。

本文档说明如何使用 vllm-ascend 的 fake_mx 伪量化方案对 Qwen3.5-9B 进行 W4A4 量化评测。
所有配置文件、脚本和转换工具均在本目录中，与 `vllm_ascend/quantization/methods/fake_mx_algorithms/` 算法包（`methods/fake_mx.py` 为 scheme 注册 shim）保持一致。

## 目录结构

```
w4a4_eval/
├── README.md                          # 本文档
├── baseline-results.md                # W4A4 精度基线（64 单元完整矩阵）
├── quick-verify-guide.md              # 快速验证方案（Level 子集）
├── configs/                           # 19 个量化配置文件
│   ├── qwen3_5_9b_rtn_attn-only_w4a4.json
│   ├── qwen3_5_9b_rtn_mlp-only_w4a4.json
│   ├── qwen3_5_9b_rtn_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_rtn_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_rht_attn-only_w4a4.json
│   ├── qwen3_5_9b_rht_mlp-only_w4a4.json
│   ├── qwen3_5_9b_rht_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_rht_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_flatquant_attn-only_w4a4.json
│   ├── qwen3_5_9b_flatquant_mlp-only_w4a4.json
│   ├── qwen3_5_9b_flatquant_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_flatquant_attn-only_w4a4-w8a8-mixed.json
│   ├── qwen3_5_9b_omniquant_attn-only_w4a4.json
│   ├── qwen3_5_9b_omniquant_mlp-only_w4a4.json
│   ├── qwen3_5_9b_omniquant_attn-mlp_w4a4.json
│   ├── qwen3_5_9b_lht_attn-only_w4a4.json
│   ├── qwen3_5_9b_lht_mlp-only_w4a4.json
│   ├── qwen3_5_9b_lht_attn-mlp_w4a4.json
│   └── qwen3_5_9b_lht_attn-only_w4a4-w8a8-mixed.json
├── scripts/
│   ├── serve/vllm_serve.sh            # vllm serve 启动脚本
│   ├── eval/run_math500.py            # MATH-500 评测脚本
│   ├── eval/run_limited_eval.py       # 小样本快速验证（limit per-subset）
│   ├── ptq/
│   │   ├── extract_ptq_data.sh        # 校准数据提取
│   │   └── run_amct_ptq.sh            # PTQ 多卡训练
│   └── convert/convert_ptq_to_vllm.py # PTQ 参数转换工具
```

## 配置文件命名规则

```
qwen3_5_9b_{algo}_{scope}_{precision}.json
```

| 字段 | 含义 | 取值 |
|------|------|------|
| algo | 量化算法 | rtn, rht, flatquant, omniquant, lht |
| scope | 量化模块范围 | attn-only, mlp-only, attn-mlp |
| precision | 精度配置 | w4a4, w4a4-w8a8-mixed |

**w4a4**: 所有量化的 linear 层权重和激活均为 W4A4 MXFP4。
**w4a4-w8a8-mixed**: attention input 投影（qkv_proj/in_proj）为 W4A4 MXFP4，attention output 投影（o_proj/out_proj）为 W8A8 MXFP8，MLP 保持浮点。

## 量化算法说明

### RTN（Round-To-Nearest）

最简单的量化方式，直接对权重和激活做 MXFP4 QDQ（量化-反量化），无任何变换。

- 配置项：无额外配置
- scheme 名：`W4A4_MXFP4_FAKE`
- 代码位置：`fake_mx_algorithms/linear.py` `FakeMXLinearMethod`
- 无需 PTQ 训练，无需外部参数

### RHT（Randomized Hadamard Transform）

在量化前对权重和激活施加随机 Hadamard 变换，打散离群值，降低量化误差。

- 配置项：
  - `rht_seed` — 随机 sign 序列的种子（默认 0，与 AMCT 一致）
  - `rht_matrix_size` — Hadamard 分块大小（默认 128，兼容回退 `rht_group_size`）
  - `rht_params_path` — 可选；AMCT 导出的 `rht_signs` sidecar，设置后将逐层比特级校验运行时生成的 signs 与 AMCT 一致，不一致即报错
- scheme 名：`W4A4_MXFP4_RHT_FAKE` / `W8A8_MXFP8_RHT_FAKE`
- 代码位置：`fake_mx_algorithms/rht.py` `RHTLinearMethod`
- 无需 PTQ 训练，用 seed 生成随机 signs
- 权重变换：加载时自动旋转权重（`prepare_weight` 中调用 `randomized_hadamard_transform`）
- 激活变换：`randomized_hadamard_transform(x, signs, matrix_size)`

### FlatQuant

学习型 Kronecker 变换，通过 PTQ 训练优化 left/right 变换矩阵和 diag_scale。

- 配置项：
  - `flatquant_params_path` — 外部参数文件路径（相对于模型目录）
  - `max_supported_tp` — 最大支持的 TP 数（默认 4）
  - `flatquant_matrix_size` — AMCT 分解的矩阵大小 K（默认 128）
  - `flatquant_use_diag` — 是否使用 diag_scale（默认 true）
  - `group_size` — MX 分块大小（默认 32）
- scheme 名：`W4A4_MXFP4_FLATQUANT_FAKE` / `W8A8_MXFP8_FLATQUANT_FAKE`
- 代码位置：`fake_mx_algorithms/flatquant.py` `FlatQuantLinearMethod`
- 需要 PTQ 训练生成参数
- 权重变换：`W' = inv(left) @ reshape(W) @ inv(right).T / diag_scale`
- 激活变换：`x' = reshape(left.T @ reshape(x) @ right) * diag_scale`
- 参数文件格式（safetensors）：
  ```
  {prefix}.weight           — 变换后的 FP 权重 [out, in]
  {prefix}.left_trans       — 左变换矩阵 [L, L]
  {prefix}.right_trans      — 右变换矩阵 [R, R]
  {prefix}.clip_ratio       — 裁剪比例 [1] (float32, 值=1.0)
  {prefix}.diag_scale       — 对角缩放 [L*R] (float32, 可选)
  ```
  其中 L*R = in_features，L 和 R 由 AMCT 训练确定

### LHT（Learnable Hadamard Transform）

学习型 Hadamard 变换，通过 PTQ 训练优化一个 K×K 的正交变换矩阵。

- 配置项：
  - `lht_params_path` — 外部参数文件路径（相对于模型目录，**必填**）
  - `hadamard_learning_matrix_size` — 变换矩阵大小 K（默认 128）
  - `group_size` — MX 分块大小（默认 32）
- scheme 名：`W4A4_MXFP4_HADAMARD_LEARNING_FAKE` / `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`
- 代码位置：`fake_mx_algorithms/lht.py` `LHTLinearMethod`
- 需要 PTQ 训练生成参数
- 权重变换：`W' = reshape(W, -1, K) @ inv(T).T`
- 激活变换：`x' = reshape(x, -1, K) @ T`
- 参数文件格式（safetensors）：
  ```
  {prefix}.transform_weight — K×K 变换矩阵 [K, K] (bfloat16)
  ```
  其中 K = hadamard_learning_matrix_size（默认 128），in_features 必须能被 K 整除

## 模块覆盖说明

配置通过 `module_quant_overrides` 的 glob 模式匹配模块路径。匹配规则使用 `fnmatchcase`，按字典序遍历，**先匹配到的 pattern 生效**。

Qwen3.5-9B 的关键模块路径：
```
model.language_model.layers.{N}.self_attn.qkv_proj.weight      # attention input (fused QKV)
model.language_model.layers.{N}.self_attn.o_proj.weight         # attention output
model.language_memory.layers.{N}.linear_attn.in_proj_qkvz.weight  # GDN input (fused QKVZ)
model.language_memory.layers.{N}.linear_attn.in_proj_ba.weight    # GDN input (fused BA)
model.language_memory.layers.{N}.linear_attn.out_proj.weight      # GDN output
model.language_memory.layers.{N}.mlp.gate_proj.weight           # MLP gate
model.language_memory.layers.{N}.mlp.up_proj.weight             # MLP up
model.language_memory.layers.{N}.mlp.down_proj.weight           # MLP down
```

> **注意**：`self_attn` 和 `linear_attn` 在不同层交替出现（Qwen3.5 混合架构），`*self_attn*` 和 `*linear_attn*` 的 glob 会分别匹配。

所有配置都有 `"*": "FLOAT"` 作为兜底，确保未显式指定的模块（embed_tokens、lm_head、visual、MTP 等）保持浮点。

## 使用步骤

### 1. RTN / RHT 评测（无需 PTQ 训练）

```bash
# 复制配置到模型目录
cp configs/qwen3_5_9b_rtn_attn-only_w4a4.json /path/to/model/quant_model_description.json

# 启动 vllm serve
# 第五参数 eager（默认）用于对照；decode_graph 用于逐算法验证图模式性能
./scripts/serve/vllm_serve.sh /path/to/model 0 8001 configs/qwen3_5_9b_rtn_attn-only_w4a4.json

# 运行评测
python scripts/eval/run_math500.py 8001 ./outputs/rtn_attn-only_w4a4

# 三数据集串行评测（smoke -> MATH-500 -> MMLU-Pro -> LiveCodeBench）
SCENARIO=rtn_attn-only_w4a4 PORT=8001 WORK_DIR=./outputs bash scripts/eval/run_three_datasets.sh

# 小样本快速验证（全量前的回归，limit 为每个子集的样本数）
python scripts/eval/run_limited_eval.py 8001 ./outputs/rtn_quick mmlu_pro 10
```

### 2. FlatQuant / LHT 评测（需要 PTQ 训练）

#### 2.1 提取校准数据

```bash
# attn 层校准数据
./scripts/ptq/extract_ptq_data.sh /path/to/model /data/ptq_data attn-linear 0

# mlp 层校准数据
./scripts/ptq/extract_ptq_data.sh /path/to/model /data/ptq_data mlp 1
```

#### 2.2 PTQ 训练（8 卡并行）

```bash
# FlatQuant attn 训练
./scripts/ptq/run_amct_ptq.sh flatquant attn-linear /path/to/model /data/ptq_data /data/ptq_fq_attn

# LHT attn 训练
./scripts/ptq/run_amct_ptq.sh learnable_had attn-linear /path/to/model /data/ptq_data /data/ptq_lht_attn

# FlatQuant mlp 训练
./scripts/ptq/run_amct_ptq.sh flatquant mlp /path/to/model /data/ptq_data /data/ptq_fq_mlp

# LHT mlp 训练
./scripts/ptq/run_amct_ptq.sh learnable_had mlp /path/to/model /data/ptq_data /data/ptq_lht_mlp
```

#### 2.3 转换参数

```bash
# FlatQuant attn 参数（需要模型权重做权重变换）
python scripts/convert/convert_ptq_to_vllm.py \
  --algo flatquant --target attn-linear \
  --ptq_dir /data/ptq_fq_attn/ptq_params/qwen3_5/attn-linear \
  --model_dir /path/to/model \
  --output /data/flatquant_attn_params.safetensors

# FlatQuant mlp 参数
python scripts/convert/convert_ptq_to_vllm.py \
  --algo flatquant --target mlp \
  --ptq_dir /data/ptq_fq_mlp/ptq_params/qwen3_5/mlp \
  --model_dir /path/to/model \
  --output /data/flatquant_mlp_params.safetensors

# 合并 attn + mlp 参数（用于 attn-mlp 场景）
python scripts/convert/convert_ptq_to_vllm.py \
  --merge /data/flatquant_attn_params.safetensors \
          /data/flatquant_mlp_params.safetensors \
          /data/flatquant_attn_mlp_params.safetensors

# LHT attn 参数（不需要模型权重）
python scripts/convert/convert_ptq_to_vllm.py \
  --algo lht --target attn-linear \
  --ptq_dir /data/ptq_lht_attn/ptq_params/qwen3_5/attn-linear \
  --output /data/lht_attn_params.safetensors

# LHT mlp 参数
python scripts/convert/convert_ptq_to_vllm.py \
  --algo lht --target mlp \
  --ptq_dir /data/ptq_lht_mlp/ptq_params/qwen3_5/mlp \
  --output /data/lht_mlp_params.safetensors

# 合并 LHT attn + mlp
python scripts/convert/convert_ptq_to_vllm.py \
  --merge /data/lht_attn_params.safetensors \
          /data/lht_mlp_params.safetensors \
          /data/lht_attn_mlp_params.safetensors
```

#### 2.4 部署参数并评测

```bash
# 将参数文件放到模型目录（或配置中指定的路径）
cp /data/flatquant_attn_params.safetensors /path/to/model/flatquant_params.safetensors

# 启动 eager 对照；FlatQuant/LHT 的 decode_graph 需先通过一致性验证
./scripts/serve/vllm_serve.sh /path/to/model 0 8001 \
  configs/qwen3_5_9b_flatquant_attn-only_w4a4.json --enforce-eager

# 评测
python scripts/eval/run_math500.py 8001 ./outputs/flatquant_attn-only_w4a4
```

## 配置项与代码对照表

| 配置 JSON key | 代码读取位置 | 说明 |
|--------------|------------|------|
| `default_quant_type` | `modelslim_config.py` | 默认 scheme 名 |
| `module_quant_overrides` | `modelslim_config.py` | glob 模式覆盖 |
| `group_size` | `fake_mx_algorithms/linear.py` `FakeMXLinearMethod.__init__` | MX 分块大小（默认 32） |
| `rht_seed` | `fake_mx_algorithms/rht.py` `RHTLinearMethod.__init__` | RHT 随机种子（默认 0） |
| `rht_matrix_size` | `fake_mx_algorithms/rht.py` `RHTLinearMethod.__init__` | RHT Hadamard 分块大小（默认 128，兼容 `rht_group_size` 回退） |
| `rht_params_path` | `fake_mx_algorithms/rht.py` `RHTLinearMethod.__init__` | 可选；AMCT `rht_signs` sidecar，逐层比特级校验 |
| `flatquant_params_path` | `fake_mx_algorithms/flatquant.py` `FlatQuantLinearMethod.__init__` | FlatQuant 参数文件路径 |
| `max_supported_tp` | `fake_mx_algorithms/flatquant.py` `FlatQuantLinearMethod.__init__` | FlatQuant 最大 TP |
| `flatquant_matrix_size` | `fake_mx_algorithms/flatquant.py` `FlatQuantLinearMethod.__init__` | FlatQuant AMCT 矩阵大小 K |
| `flatquant_use_diag` | `fake_mx_algorithms/flatquant.py` `FlatQuantLinearMethod.__init__` | 是否使用 diag_scale |
| `lht_params_path` | `fake_mx_algorithms/lht.py` `LHTLinearMethod.__init__` | LHT 参数文件路径（必填） |
| `hadamard_learning_matrix_size` | `fake_mx_algorithms/lht.py` `LHTLinearMethod.__init__` | LHT 矩阵大小 K |
| `omniquant_params_path` | `fake_mx_algorithms/omniquant.py` `OmniQuantLinearMethod.__init__` | OmniQuant `log_scale` 参数文件路径 |

> **注意**：重构后 `fake_mx_weight_state`、`auto_transform`/`auto_rotate`、`fake_mx_quant_targets`（含 `attn-cache`）系列配置项已移除。非 Linear 节点不再纳入 fake-MX；变换始终在加载时自动执行，无需手动开关。

## 评测结果（Qwen3.5-9B）

完整 64 单元精度基线（16 场景 × 4 指标：PPL / MATH-500 / MMLU-Pro / LiveCodeBench）见 [baseline-results.md](./baseline-results.md)。

### MATH-500 摘要（全 W4A4，采样解码契约）

| 算法 | attn-only | mlp-only | attn-mlp |
|------|-----------|----------|----------|
| RTN | 86.4% | 92.0% | 79.0% |
| RHT | 86.6% | 93.2% | 80.6% |
| FlatQuant | 92.4% | 94.0% | 91.0% |
| OmniQuant | 93.0% | 92.6% | 91.6% |
| LHT | 88.4% | 93.2% | 88.0% |

BF16 基线：94.0%

> 解码契约：`do_sample=true, temperature=1.0, top_k=20, top_p=0.95, seed=42`（与 `scripts/eval/` 全量脚本一致）。历史文档中的确定性解码数字（如 BF16 93.8%）及 bug 时代 RHT 数据已作废，不可与本表比较。

### 关键结论

1. **FlatQuant 数据集综合最优**：attn-mlp 场景 MMLU-Pro 77.5% / LCB 53.2% 为五算法最佳；mlp-only MATH-500 94.0% 追平 BF16、LCB 58.7% 反超 BF16
2. **学习型变换 PPL 可低于 BF16**：LHT attn 7.801、FlatQuant attn 7.812 / attn-mlp 7.972（BF16 为 8.171）
3. **RHT 零训练成本抑制离群值**：vs RTN，PPL attn 9.30→8.03、attn-mlp 9.56→8.65；mlp-only 数据集净收益（MATH-500 +1.2pt、LCB +1.3pt）
4. **attn 是 W4A4 主要损失源**：RTN vs BF16，attn-only MATH-500 -7.6pt / LCB -18.2pt（mlp-only 仅 -2.0pt / -8.3pt）
5. **学习型变换全量化收益 +12pt 级**：attn-mlp MATH-500，RTN 79.0% → OmniQuant 91.6% / FlatQuant 91.0%

Level3 快速验证方案见 [quick-verify-guide.md](./quick-verify-guide.md)。

## 环境要求

- vllm-ascend 分支：`fake-mx-lite`
- CANN >= 9.0.1
- Ascend 910 NPU
- AMCT PTQ 训练需独立 conda 环境（amct_pytorch）
- evalscope >= 1.9.1
