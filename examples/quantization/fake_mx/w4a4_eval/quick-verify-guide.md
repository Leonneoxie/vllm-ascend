# W4A4 量化修改快速验证方案

> 目标：代码/配置修改后，快速验证是否掉点（无需跑全量 500 题）

## 一、验证原理

evalscope 的 math_500 数据集支持 `subset_list` 指定只跑某个 Level 子集：

| Level | 题数 | 难度 | 各算法分数范围 | 推荐用途 |
|-------|------|------|----------------|---------|
| Level 1 | 43 | 最简单 | 95%~100% | 验证基本功能不崩 |
| Level 3 | 105 | 中等 | 91%~99% | **推荐用于检测掉点** |

**推荐 Level 3**：题量适中（105题，~10min/场景），区分度高，对量化误差敏感。

## 二、验证脚本

```bash
python scripts/eval/run_level3_verify.py <port> <work_dir>
# 示例: python scripts/eval/run_level3_verify.py 8008 ./outputs/verify_fq_attn
```

生成配置与全量脚本完全一致（`do_sample=true, temperature=1.0, top_k=20, top_p=0.95, seed=42`），仅限 Level 3 子集。

## 三、并行验证流程

### 关键原则

1. **不需要模型副本**——同一模型目录，端口区分实例
2. **不需要在 model name 里加 port**——统一用 `--served-model-name qwen3.5`
3. **vllm 加载完配置后不再读文件**——可以安全替换配置文件
4. **时序错开是必须的**：必须等前一个 vllm 完全加载完配置（日志出现 "Loaded N fake-MX transform params"）后，才能替换配置文件启动下一个
5. **先以 eager 建立对照，再逐算法验证 decode_graph**：FlatQuant/LHT 图模式兼容性尚需实测；启动脚本第五参数可传 `decode_graph`。不能同时传 `--enforce-eager` 和图配置。通过输出一致性和性能验证后，再在完整测评使用图模式。
6. **不要用 heredoc 创建 JSON 配置**——shell 会吃掉引号。用 python json.dump 或 scp 传输

### 步骤

```bash
# 1. 部署 FlatQuant 配置并启动（卡5端口8008）
cp configs/qwen3_5_9b_flatquant_attn-only_w4a4.json /path/to/model/quant_model_description.json
cp flatquant_params.safetensors /path/to/model/
ASCEND_RT_VISIBLE_DEVICES=5 vllm serve /path/to/model \
  --port 8008 --served-model-name qwen3.5 --enforce-eager ... &

# 2. 等待 FlatQuant 就绪并确认参数加载
curl -s http://localhost:8008/v1/models | head -1  # 返回 JSON 即就绪
grep "Loaded.*fake-MX.*transform" serve.log  # 必须有 "Loaded N fake-MX transform params" 日志

# 3. 启动 FlatQuant 评测
python scripts/eval/run_level3_verify.py 8008 ./outputs/verify_fq &

# 4. 等 FlatQuant 评测开始跑（确认有进度），替换配置为 LHT
cp configs/qwen3_5_9b_lht_attn-only_w4a4.json /path/to/model/quant_model_description.json
rm -f /path/to/model/flatquant_params.safetensors
cp lht_params.safetensors /path/to/model/
ASCEND_RT_VISIBLE_DEVICES=7 vllm serve /path/to/model \
  --port 8009 --served-model-name qwen3.5 --enforce-eager ... &

# 5. 等 LHT 就绪并确认参数加载
grep "Loaded.*fake-MX.*transform" serve.log  # "Loaded 88 fake-MX transform params"

# 6. 启动 LHT 评测
python scripts/eval/run_level3_verify.py 8009 ./outputs/verify_lht &
```

### 验证参数加载（必须检查）

```bash
# 统一日志格式（重构后）：
grep "Loaded.*fake-MX.*transform" serve.log
# FlatQuant: "Loaded 264 fake-MX transform params from flatquant_params.safetensors"
# LHT: "Loaded 88 fake-MX transform params from lht_params.safetensors"
```

**如果没有 loaded 日志，说明参数文件没加载，结果无效！** 常见原因：
- 参数文件路径错误（配置中 `params_path` 与实际文件名不匹配）
- 配置文件 JSON 格式错误（heredoc 创建时引号被吃）
- 配置文件被其他 vllm 实例覆盖（时序错开没做好）

## 四、对比标准

| 算法 | 全量 Level3 基线 | 允许误差 |
|------|----------------|---------|
| RTN | 92.38% | ±1% |
| FlatQuant | 97.14% | ±2% |
| LHT | 97.14% | ±1% |

差异 >2% 说明修改有问题。

## 五、卡选择和进程管理

### HBM 检查
```bash
npu-smi info | grep HBM-Usage
# 空闲卡 HBM < 5GB，被占用卡 HBM > 55GB
```

### 进程清理
```bash
# 只杀自己端口的进程
pkill -9 -f "port 8008" 2>/dev/null
# 不要用 pkill -9 -f VLLM::EngineCore（会杀掉所有进程包括他人的）
# 用 pid 精确杀
kill -9 <pid>
```
