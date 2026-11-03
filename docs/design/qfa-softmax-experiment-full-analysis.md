# QFA Softmax FP32→FP16 实验：完整方案分析与执行计划

> 终极目标：全 MXFP8 量化管线（极限形态）中，softmax 从 FP32 降到 FP16 对数据集精度的影响。
> 掉点小 → 极限管线还有性能空间；掉点大 → softmax 是守门员，必须保 FP32。
>
> 本文档自包含：涵盖所有方案的分析、最终选择、详细执行计划、msmodelslim 量化指导。

---

## 1. QFA 全路径覆盖的四种方案

### 核心问题

QFA 算子接口要求 k/v 为 `float8_e4m3fn + e8m0 descale`——它是量化 attention 算子，只吃量化输入。而 vllm 默认的 KV cache 是 BF16。**格式不匹配是所有方案差异的根源。**

```
FIA：接受 BF16 → 读 BF16 cache → 天然匹配
QFA：只接受 fp8+descale → BF16 cache 喂不进去 → 需要某种"桥接"
```

### 方案 A：QFA 在线量化（非 L2，仅 prefill）

**原理**：prefill 时 K/V 是当步投影输出（小张量，在内存中），直接在线量化喂 QFA。

```python
# prefill 时 wrapper 收到：
#   query = 当步 Q 投影 (4096, N, D) BF16  ← 小，几 MB
#   key   = 当步 K 投影 (4096, N, D) BF16  ← 小，几 MB
#   value = 当步 V 投影 (4096, N, D) BF16  ← 小
#
# 在线量化：
k8, k_ds = npu_dynamic_mx_quant(key)    ← 量化几 MB，毫秒级 ✅
v8, v_ds = npu_dynamic_mx_quant(value)
quant_flash_attn(q8, k8, v8, q_ds, k_ds, v_ds, mode, ...)
```

**decode 为何回退**：

```python
# decode 时 wrapper 收到（_get_fia_params :1211-1219）：
#   key = self.key_cache.view(num_block, block_size, -1)
#         = 整个 KV cache 池 (10000, 128, N, D) BF16  ← 2.6 GB ！
#   value = 同上
#
# "在线量化 key" = 量化 2.6 GB → 每层每步 → 完全不可行
#
# 回退代码（device_op.py :1395-1396）：
if kwargs.get("block_table") is not None:    # decode/前缀续算有 block_table
    return None                              # "我不处理" → FIA 兜底
```

**覆盖矩阵**：

| 状态 | 覆盖 | 原因 |
|---|---|---|
| 全新 prefill | ✅ | K/V 是小张量，随手量化 |
| decode | ❌ 回退 FIA | K/V 是整个 cache 池（GB 级），量化不可行 |
| 前缀续算 | ❌ 回退 FIA | 同上 |
| chunked prefill | ❌ 回退 FIA | 后续块有 block_table |
| 图模式 | ❌ | V 量化 tolist host 陷阱 |

**适用**：prefill-only 的 softmax 快速实验；作为动态 V 消融探针（B1 异常时对比 V 量化方式）。

---

### 方案 B：QFA + gather + 在线量化

**原理**：decode 时用 block_table 从 cache 中只提取当前请求的 K/V 历史（MB 级），在线量化后喂 QFA。

```python
# gather：从 10000 块中挑出当前请求的 30 块
req_blocks = block_table[b, :num_blocks]        # e.g., [15, 42, 87, 203, ...]
k_req = key_cache[req_blocks].reshape(-1, N, D)[:L]  ← gather ~4 MB ✅

# 在线量化（同 prefill 数学）
k8, k_ds = npu_dynamic_mx_quant(k_req)

# 喂 QFA（TND 模式）
quant_flash_attn(q8, k8, v8, q_ds, k_ds, v_ds, ...)
```

**覆盖矩阵**：

| 状态 | 覆盖 | 原因 |
|---|---|---|
| 全新 prefill | ✅ | 同方案 A（在线量化） |
| decode | ✅ | gather 提取当前请求 K/V → 量化 → QFA |
| 前缀续算 | ✅ | 同上 |
| chunked prefill | ✅ | 同上 |
| 图模式 | ❌ | V 量化 tolist host 陷阱 |

**优缺点**：

| 优点 | 缺点 |
|---|---|
| 全路径覆盖 ✅ | O(seq_len) gather per layer per step |
| 不改 cache 格式 | 同样的历史 K/V 每步重新量化（重复劳动） |
| 代码简单（v30-fa 已实现） | decode 性能 ~100x 慢于 L2 |
| 教育价值：清晰展示"为什么非 L2 回退" | 仅 eager |

**适用**：功能验证、教育参考、正确性基线。

---

### 方案 C：QFA + L2（cache MxFP8 化）

**原理**：改变 cache 存储格式——写入时量化（每个 token 只做一次），attention 时直读（零拷贝零重量化）。

```
写入时（token 首次到来）：
  K 投影(BF16) → mx_quant → fp8+descale → 写入 cache 的对应 slot
  V 投影(BF16) → 静态scale量化 → fp8 → 写入 cache

attention 时（所有状态统一）：
  Q 投影(BF16) → mx_quant → fp8+descale       ← 唯一保留的在线量化
  K/V → 直接从 fp8 cache 读（block_table 寻址）  ← 零拷贝，零重量化
  quant_flash_attn(q8, k_cache, v_cache, ..., block_table, PA layout)
```

**V 尾块问题（方案 A 版 L2 的复杂度来源）**：

```
QFA v_descale = per-64-token 共享块（接口定义）

增量写入冲突：
  64-token 块未满时 → scale 依赖组内 max
  新 token 到来 → max 变大 → scale 需更新
  → 已写入的前序 token 量化值作废（按旧 scale 除的）
  → 数据/scale 不一致 = 数值错误（不是精度，是正确性）

解法：尾块机制
  不完备的组 → BF16 尾块暂存
  满 64 时 → 一次性量化写入主 cache + descale → 尾块清零复用
```

**方案 B'（L2 静态 V scale）**：如果 V scale 不依赖运行时数据（静态校准/neutral default），则无组依赖 → 逐 token 独立量化 → **零尾块**。

**覆盖矩阵**：

| 状态 | 覆盖 | 原因 |
|---|---|---|
| 全新 prefill | ✅ | cache 直读（写入时已量化） |
| decode | ✅ | cache 直读（PA 布局原生支持） |
| 前缀续算 | ✅ | 同上 |
| chunked prefill | ⚠️ | 部分调用点未传 _l2_kv_cache（可补） |
| 图模式 | ❌ | 写入路径 host 逻辑（tolist） |

**优缺点**：

| 优点 | 缺点 |
|---|---|
| 全路径覆盖 ✅ | cache 分配/写入/attention 三处改造 |
| decode 性能最优（零拷贝零重量化） | model_runner 侵入（cache 分配改动） |
| 量化一次永久存储 | 依赖 wrapper 代码（v30-fa 分支） |
| V 静态 scale 版无尾块复杂度 | 仅 eager |

---

### 方案 D：w8a8c8-mxfp8 + 生产 C8-MXFP backend（最终选择 ✅）

**原理**：v0.30.0 已内建完整的生产级 QFA 集成（`attention_c8_mxfp.py`，929 行），由量化产物自动激活。

```
激活链（零代码）：
  产物：quant_type = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
  启动：--kv-cache-dtype mxfp8
  → AscendC8MXFPKVCacheAttentionMethod 安装
  → cache 分配：packet-packed PA_NZ（[K|K_scale|V|V_scale] 单块布局）
  → 写入：K 动态量化 + V 静态校准量化 + scatter
  → attention：每 step 单次 QFA 调用（prefill+decode 统一）
  → 图模式原生（npugraph_ex，GRAPH_PATH=7 上板验证）
```

**覆盖矩阵**：

| 状态 | 覆盖 | 原因 |
|---|---|---|
| 全新 prefill | ✅ | 生产路径 |
| decode | ✅ | 生产路径（PA_NZ 直读） |
| 前缀续算 | ✅ | 生产路径 |
| chunked prefill | ✅ | 生产路径 |
| **图模式** | ✅ | **原生支持**（npugraph_ex 上板验证） |
| MTP/投机解码 | ✅ | 同一派生链 |

**优缺点**：

| 优点 | 缺点 |
|---|---|
| 零代码改动 | 需要量化产物带 C8-MXFP recipe |
| 生产级（上板验证、图模式、性能优化） | V scale 依赖校准产物 |
| 全路径 + 全状态覆盖 | 模型必须能被 msmodelslim 量化 |
| softmax 归因最干净 | — |

---

## 2. 四方案对比总结

| 维度 | A（在线量化） | B（gather） | C（L2 wrapper） | **D（生产 backend）** |
|---|---|---|---|---|
| prefill | ✅ | ✅ | ✅ | ✅ |
| decode | ❌ | ✅ | ✅ | ✅ |
| 图模式 | ❌ | ❌ | ❌ | **✅** |
| 代码量 | v30-fa 已有 | v30-fa 已有 | v30-fa 已有 | **零** |
| decode 性能 | — | ~100x 慢 | 最优 | **生产级** |
| cache 改动 | 无 | 无 | fp8+descale | **packet PA_NZ** |
| V scale | 动态 | 动态 | 动态/静态可选 | **静态校准** |
| 适用场景 | 消融探针 | 教育/功能验证 | wrapper 参考 | **主实验** ✅ |

---

## 3. FP16 softmax 的 D=128 约束与方案选择

### 问题

实验版 QFA mode 3（MxFP8 + Softmax FP16）约束：

| 约束 | 实验版 mode 3 | 主线 QFA | 差距 |
|---|---|---|---|
| D | **128 only** | 64/72/128/256 | 主线已有 D=256 |
| PA | **不支持** | PA_BNBD/PA_BBND/PA_NZ | 主线已有 PA |
| softmax | **FP16** ✅ | FP32 only | 主线缺 FP16 |

### 解法：kernel 合体（用户负责）

```
合体 = 主线 QFA kernel（D=128 + PA + 全布局）
     + 实验版 mode 3 的 softmax FP16 计算路径

产出 = 一个 kernel 同时支持：
  mode 1（FP32 softmax）— B1 基线
  新 mode（FP16 softmax）— B2 实验
  两者共享 D=128 + PA + 全布局
```

### D=128 约束下的模型选择

| 模型 | D | 类型 | w8a8c8 可量化 | 适配度 |
|---|---|---|---|---|
| Qwen3.6-27B | **256** | hybrid（GDN+full-attn） | ✅ | D=256 超出 mode 3 约束 |
| Qwen3.5-9B | 256? | hybrid | ✅ | 同上 |
| **Qwen2.5-7B** | **128** | **Dense（28 层全 full-attn）** | ✅ | **完美匹配** ✅✅✅ |
| Llama-3-8B | 128 | Dense | ✅ | 同样匹配 |

**选择 Qwen2.5-7B**：
- D=128 → mode 3 约束满足
- Dense（无 GDN）→ 所有层都是 full-attention，softmax 实验信号最大化
- msmodelslim 完整支持 w8a8c8-mxfp8 量化
- 中文能力强，与实验场景匹配

### 关键认知（纠正之前的误区）

```
之前的误区：
  "mode 3 不支持 PA → 生产 backend 用不了 → 需要 gather 路径"
  
纠正：
  mode 3 的"不支持 PA"是实验 kernel 的约束
  合体后（FP16 移入主线 kernel）→ 主线的 PA 支持全部保留
  → 生产 backend 的 PA_NZ 路径完全可用
  → 不需要 gather，不需要 wrapper，直接生产路径
```

---

## 4. 最终实验方案（Qwen2.5-7B + w8a8c8-mxfp8 + 生产 backend）

```
模型：Qwen2.5-7B → msmodelslim 量化为 w8a8c8-mxfp8
环境：A5/950 系，CANN 9.2.0，vllm-ascend v0.30.0

B1（FP32 基线）：
  vllm serve Qwen2.5-7B-w8a8c8-mxfp8 \
      --quantization ascend \
      --kv-cache-dtype mxfp8 \
      --trust-remote-code
  → 生产 C8-MXFP backend 自动激活
  → QFA mode 1（MxFP8 + softmax FP32）
  → 零代码改动 ✅

B2（FP16 实验）：
  前提：kernel 合体完成（FP16 softmax → 主线 QFA kernel）
  改动：attention_c8_mxfp.py:450
        QFA_QUANT_MODE_MXFP8 = 1 → 新 FP16 mode 值
  → 同一个生产 backend，同一个 PA_NZ 路径
  → 唯一差异 = softmax 精度 ✅

B1 vs B2 = softmax FP32 vs FP16（唯一变量，最干净对照）
```

---

## 5. msmodelslim 量化 Qwen2.5-7B → w8a8c8-mxfp8 详细指导

### 5.1 前置条件

```bash
# 硬件：Atlas A5/950 系（msmodelslim 校准需 NPU）
# 软件：msmodelslim（与 vllm-ascend v0.30.0 兼容版本）
# 原始模型：Qwen/Qwen2.5-7B（从 ModelScope 或 HuggingFace 下载 BF16 原版）
```

### 5.2 下载原始模型

```bash
# ModelScope 下载
modelscope download --model Qwen/Qwen2.5-7B --local_dir ./Qwen2.5-7B

# 或使用 modelscope Python API
python -c "
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B')
print(f'Downloaded to: {model_dir}')
"
```

### 5.3 准备量化配置 YAML

创建量化配置文件（参考 Qwen3.6-27B-w8a8c8-mxfp8 的产物逆向推导配方）：

```yaml
# qwen2.5_7b_w8a8c8_mxfp8.yaml
apiversion: multimodal_vlm_modelslim_v1
spec:
  process:
  - type: group
    configs:
    # GEMM 量化：所有 Linear 层 W8A8 MXFP8
    - type: linear_quant
      qconfig:
        act:
          scope: per_block
          dtype: mxfp8
          symmetric: true
          method: minmax
        weight:
          scope: per_block
          dtype: mxfp8
          symmetric: true
          method: minmax
      include:
      - '*self_attn*'
      - '*mlp*'
      exclude:
      - mtp*
      - '*norm*'
      - '*embed*'

  # KV cache 量化：K 动态 + V 静态校准
  - type: linear_quant
    qconfig:
      act:
        scope: per_block
        dtype: mxfp8
        symmetric: true
        method: minmax
      weight:
        scope: per_block
        dtype: mxfp8
        symmetric: true
        method: minmax
    include:
    - '*self_attn*'
    exclude:
    - '*self_attn.o_proj*'
    - mtp*

  save:
  - type: ascendv1_saver
    part_file_size: 4
  dataset: calibImages

metadata:
  config_id: Qwen2_5_7B_w8a8c8-mxfp8-v1
  score: 90.0
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: true
  verified_model_types:
  - Qwen2.5-7B
```

> **注意**：以上 YAML 为参考模板。实际配方参数（分组、校准方法、量化粒度）请参考 msmodelslim 官方文档或已有 w8a8c8-mxfp8 产物的 `best_practice.yaml`。

### 5.4 执行量化

```bash
msmodelslim quant \
    --model_path ./Qwen2.5-7B \
    --save_path ./Qwen2.5-7B-w8a8c8-mxfp8 \
    --model_type Qwen2.5-7B \
    --config qwen2.5_7b_w8a8c8_mxfp8.yaml \
    --trust_remote_code true \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7
```

**参数说明**：

| 参数 | 值 | 说明 |
|---|---|---|
| `--model_path` | ./Qwen2.5-7B | 原始 BF16 模型路径 |
| `--save_path` | ./Qwen2.5-7B-w8a8c8-mxfp8 | 量化产物输出路径 |
| `--model_type` | Qwen2.5-7B | 模型架构标识（msmodelslim 内部路由） |
| `--config` | qwen2.5_7b_w8a8c8_mxfp8.yaml | 量化配方配置 |
| `--trust_remote_code` | true | 允许自定义模型代码 |
| `--device` | npu | 在 NPU 上校准 |
| `--device_id` | 0 1 2 3 4 5 6 7 | 使用的 NPU 卡（8 卡校准加速） |

### 5.5 验证量化产物

量化完成后，检查输出目录的关键文件：

```bash
ls Qwen2.5-7B-w8a8c8-mxfp8/
# 应包含：
#   quant_model_description.json   ← 量化描述（激活关键）
#   quant_model_weights-*.safetensors  ← 量化权重
#   config.json                   ← 模型配置
#   tokenizer.json / vocab.json   ← 分词器
```

**检查 quant_model_description.json 的关键字段**：

```bash
python -c "
import json
with open('Qwen2.5-7B-w8a8c8-mxfp8/quant_model_description.json') as f:
    desc = json.load(f)

# 1. 检查 GEMM 量化（应全为 W8A8_MXFP8）
for k, v in desc.items():
    if 'self_attn.q_proj' in k and 'scale' not in k:
        print(f'{k}: {v}')  # 应为 W8A8_MXFP8
        break

# 2. 检查 attention 量化类型（激活生产 backend 的关键字段）
for k, v in desc.items():
    if 'quant_type' in k and 'self_attn' in k:
        print(f'{k}: {v}')  # 应为 QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL
        break

# 3. 检查 V 静态校准 scale（checkpoint 中的 fa_v.scale）
for k, v in desc.items():
    if 'fa_v.scale' in k:
        print(f'{k}: {v}')  # 应为 FAQuant
        break

# 4. 检查 KV cache 量化标签
print('kv_cache_type:', desc.get('kv_cache_type', 'NOT FOUND'))
"
```

**预期输出**：
```
model.language_model.layers.0.self_attn.q_proj.weight: W8A8_MXFP8
model.language_model.layers.0.self_attn.quant_type: QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL
model.language_model.layers.0.self_attn.fa_v.scale: FAQuant
```

### 5.6 部署与验证

```bash
# 部署到 A5 环境
vllm serve Qwen2.5-7B-w8a8c8-mxfp8 \
    --quantization ascend \
    --kv-cache-dtype mxfp8 \
    --trust-remote-code \
    --enforce-eager

# 检查激活日志（应看到）：
# [quantization] C8 MXFP8 KV cache enabled (--kv-cache-dtype mxfp8)

# 快速功能测试
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen2.5-7B-w8a8c8-mxfp8", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 5.7 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `quant_type` 字段缺失 | YAML 配方中未启用 KV cache 量化 | 检查 YAML 中 kv_cache 相关配置 |
| `fa_v.scale` 缺失 | 校准数据不足或 V 校准未执行 | 增加 `--device_id` 数量或更换校准数据 |
| vllm 启动报 dtype mismatch | 产物中权重 dtype 与预期不符 | 检查 msmodelslim 版本与 vllm-ascend 兼容性 |
| `--kv-cache-dtype mxfp8` 不生效 | vllm-ascend 版本不支持 | 确认使用 v0.30.0+ |

---

## 6. 执行计划（最终版）

| # | 任务 | 负责 | 依赖 | 产出 |
|---|---|---|---|---|
| 1 | 下载 Qwen2.5-7B 原始模型（BF16） | 执行者 | — | 原始权重就位 |
| 2 | msmodelslim 量化 → w8a8c8-mxfp8 | 执行者 | 1 | 量化产物（§5 验证通过） |
| 3 | A5 环境准备（v0.30.0 镜像 + cann_ops_transformer） | 执行者 | — | 环境就绪 |
| 4 | **B1 评测**（生产 backend mode 1，FP32 softmax） | 执行者 | 2+3 | B1 基线分数 |
| 5 | **kernel 合体**（FP16 softmax → 主线 QFA kernel） | **模型侧负责人** | 可与 4 并行 | 合体 kernel |
| 6 | mode 值切换（attention_c8_mxfp.py:450 一行改动） | 模型侧 | 5 | FP16 模式就绪 |
| 7 | **B2 评测**（生产 backend FP16 mode） | 执行者 | 4+6 | B2 分数 |
| 8 | **B1 vs B2 → softmax FP16 结论** | — | 7 | 最终报告 |

### 风险与降级

| 风险 | 影响 | 降级方案 |
|---|---|---|
| Qwen2.5-7B w8a8c8 量化产物不完美 | B1 基线异常 | 调整 YAML 配方重量化；或改用 w8a8-mxfp8（仅 GEMM）+ wrapper 验证 |
| kernel 合体进度 | B2 延后 | B1 独立成立（FP32 基线先行） |
| 合体 kernel FP16 掉分（FA-FP16 教训） | softmax 是守门员 | 已有结论，省下后续推广投资 |
| 合体 kernel FP16 不掉分 | 好消息 | 极限管线可降档 → 推广到 Qwen3.6-27B |

---

## 附：v30-fa 分支代码资产定位

| 代码 | 定位 | 在本实验中的角色 |
|---|---|---|
| wrapper FA 分支（66bb2943e） | 阶段一 FIA→FA 工具 | 已完成使命，FA softmax 验证的复现工具 |
| wrapper QFA 在线量化分支 | 方案 A 实现 | 消融探针（B1 异常时对比 V 量化方式） |
| gather QFA 分支（bb1d477cb + 37fad713f） | 方案 B 实现 | 教育/参考（展示"为什么需要 L2/生产路径"） |
| L2 QFA 分支（b75c44a4b） | 方案 C 实现 | wrapper 参考（动态 V 量化变体） |
| 图模式 bypass（b75c44a4b 内） | 阶段一图模式支持 | 可选（阶段一图模式性能数据） |
| 设计文档 ×2 | 方案分析与蓝图 | 本文档的前置参考 |
| **生产 backend（attention_c8_mxfp.py）** | **方案 D = 最终选择** | **B1/B2 的实验载体，零代码改动** ✅ |
