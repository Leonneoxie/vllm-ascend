# QFA 全 MXFP8 极限管线实验：设计文档与执行计划（终稿）

> **文档目的**：自包含交接文档。读者（其他 agent 或工程师）无需对话上下文即可理解：实验目标是什么、为何如此设计、具体如何执行、需要验证什么。
> **代码基线**：vllm-ascend v0.30.0（server127 rel 分支 32c31d8ad）+ 本地分支 `v30-fa`（66bb2943e，阶段一成果）
> **硬件环境**：实验执行环境 = **Atlas A5/950 系（待就绪，阶段一实测所用 A5 环境的同类机型；CANN 9.2.0）**。**server127 为 Atlas A3**，仅作代码仓库中转（v30-fa 已推送至此），**非实验执行环境**（A3 不支持 MXFP8）。当前**无 A5 环境可供 agent 自动验证**——本设计所有结论为理论验证（代码/文档/产物字段实证），实测验证全部依赖人工在 A5 环境执行（见 §6 验证清单）。
> **任务档案**：`.do/task/20261029-fia-fa-replacement-bench/`（本文档为其 artifacts 终稿）

---

## 0. 一页摘要

**科学问题**：全 MXFP8 量化管线（极限形态）中，softmax 从 FP32 降到 FP16 对数据集精度的影响是多少？掉点小 → 管线还有性能空间；掉点大 → softmax 是极限形态的守门员。

**核心决策**：
- 阶段一（FIA→FA，BF16 attention 背景）：已完成，载体 = `v30-fa` 分支的 wrapper 开关（**仅 eager 有效**，见 §3.4 遗漏分析）
- 阶段二三（FA→QFA，量化 attention 背景）：**切换模型到 `Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8` + `--kv-cache-dtype mxfp8`**，走 v0.30.0 内建的生产 C8-MXFP backend（QFA 双算子），**不采用** wrapper+QFA 自建改造路线（该路线必然要求实施 L2 cache 改造大工程，见 §4.2 论证）

**softmax 归因的关键洞察**：QFA-FP32 vs QFA-FP16 是**组内同模型对照**——切换模型发生在组间（阶段一→阶段二三），不污染 softmax 归因。

**分工**：FP16 算子（QFA mode 3 合体 kernel）由模型侧负责人处理；环境/基线/冒烟验证按 §7 计划执行。

---

## 1. 终极目标与实验矩阵

### 1.1 极限形态定义（全 MXFP8 管线）

```
full_attn 层 ×16（Qwen3.6-27B，64 层中每 4 层一个）
├─ GEMM（q/k/v/o_proj）      MXFP8     （w8a8-mxfp8 / w8a8c8-mxfp8 产物均已含）
├─ attention QKV/P           MXFP8     ← QFA（quant_flash_attn mode 1）
├─ KV cache                  MxFP8     ← C8-MXFP backend 自动处理
└─ softmax                   FP32→FP16 ← 实验变量
linear_attn 层 ×48（GDN）
├─ MLP                       MXFP8     （产物已含）
└─ GDN 本体（state/in_proj/out_proj/conv） FLOAT  ← 极限的边界（递推精度结构性不可降）
vision tower / lm_head       FLOAT      ← 极限的边界
```

### 1.2 实验矩阵（两个背景层）

**背景层 1：BF16 attention（模型 `Eco-Tech/Qwen3.6-27B-w8a8-mxfp8`，阶段一，已完成 ✓）**

| 组 | GEMM | attention | softmax | 状态 |
|---|---|---|---|---|
| FIA-FP32 基线 | MXFP8 | BF16 (FIA) | FP32 | ✓ |
| FA-FP32 | MXFP8 | BF16 (FA) | FP32 | ✓（功能验证，应≈FIA） |
| FA-FP16 | MXFP8 | BF16 (FA) | **FP16** | ✓ **已完成验证** |

→ 结论：softmax 在 BF16 attention 背景下的影响（已有数据）

**背景层 2：MxFP8 全路径 attention（模型 `Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8`，阶段二三，待执行）**

| 组 | 启动配置 | attention | softmax | 回答的问题 |
|---|---|---|---|---|
| B0 | 不加 `--kv-cache-dtype` | BF16 (FIA) | FP32 | 基线（*待冒烟确认可行性，见 §6.V1*） |
| B1 | `--kv-cache-dtype mxfp8` | **MxFP8 全路径 (QFA-FP32)** | FP32 | QFA 基线；B0 vs B1 = **极限形态代价**（attention 量化掉多少） |
| B2 | B1 + FP16 kernel | MxFP8 全路径 (QFA) | **FP16** | **终极目标**：B1 vs B2 = softmax 在量化背景下的影响（唯一变量） |

跨背景层对比（FA-FP16 vs B2）：softmax 敏感度是否随 attention 量化背景变化——注意两模型存在 GDN in_proj/vision 量化差异（w8a8-mxfp8 为 FLOAT，w8a8c8-mxfp8 为 MXFP8），报告需注明此边界。

---

## 2. 背景：现状与资产

### 2.1 阶段一成果（v30-fa 分支，66bb2943e，+338 行，已推送 GitHub 与 server127 代码仓）

`vllm_ascend/device/device_op.py` 的 `A5DeviceAdaptor.npu_fused_infer_attention_score` wrapper 内实现硬编码开关：

```python
A5_ATTENTION_IMPL = "FIA"   # "FIA" | "FA" | "QFA"，位于 class A5DeviceAdaptor 前
```

- **"FA"**：替换为 ops-transformer 主线 `cann_ops_transformer.flash_attn`（BF16）。prefill 走 TND 布局；decode/前缀缓存批直读 KV cache 为 PA_BBND `(num_blocks, block_size, N, D)` + block_table，无 gather
- **"QFA"**：替换为 `cann_ops_transformer.quant_flash_attn` mode 1（MxFP8 + softmax FP32）。**仅全新 prefill**（block_table is None 时在线量化 Q/K/V）；decode/前缀续算批**回退 FIA**（BF16 cache 无法喂 QFA，见 §4.1）
- **"FIA"**：原路径逐字节保留（默认行为零变化）
- FA/QFA 分支在 FIA 之前 return，返回 None 自动落回 FIA

### 2.2 阶段一已完成的验证

1. **用户实测**（eager 模式）：wrapper 打印确认 prefill+decode 全部收口（A5+无MTP 配置下 PA 分支被否决，全走 FIA 路径→wrapper）；FA 路径跑通（decode 用 cache PA_BBND 直传 + block_table）
2. **softmax FP32→FP16 在 FA 上已验证完成**——证明 ops-transformer kernel 的 softmax 精度改造链路已打通（FA kernel 已有双版本），该经验直接复用于 QFA 的 FP16 kernel（kernel 结构同源）
3. **接口逐参数审计**（已归档 work-04）：FA/QFA 签名对照官方文档逐项核对；mask 编码经 tests 源码 ground truth 确认（`triu(ones, diagonal=1)` int8，1=遮挡）；descale 配对方向经 golden 源码确认；`npu_dynamic_mx_quant` 的输出形状/dtype 经 op-plugin 官方文档确认（**修复过一个 CRITICAL bug**：op 实际返回 uint8 dtype）

### 2.3 v0.30.0 内建的生产 QFA 集成（阶段二三的载体，发现于 work-05）

`vllm_ascend/attention/attention_c8_mxfp.py`（929 行）`AscendC8MXFPAttentionBackend`：

- QFA 双算子（`quant_flash_attn_metadata` + `quant_flash_attn`），`QFA_QUANT_MODE_MXFP8 = 1`（MxFP8 + softmax **FP32**）
- Q/K 动态量化（`npu_dynamic_mx_quant`，与 v30-fa wrapper 同款调用形态）；**V 静态校准 scale**（checkpoint 的 `fa_v.scale`，msmodelslim 校准产物）
- cache 布局：packet-packed PA_NZ（单物理块 `[K | K_scale | V | V_scale]` 四 strided view）
- **图模式原生**（npugraph_ex，golden 上板验证 GRAPH_PATH=7）；每 step 单次 QFA 调用（prefill+decode 统一，无按状态分支）；NO_MASK decode 优化；N2TGD decode q-scale 布局
- 激活链（**已双向实证验证**）：`--kv-cache-dtype mxfp8` + 产物 quant 描述 full-attn 层 `quant_type = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"`（w8a8c8-mxfp8 产物已抓取确认逐字匹配）→ `AscendC8MXFPKVCacheAttentionMethod`（quantization/methods/kv_cache/mxfp_c8.py）安装 → impl 类替换接管全部 attention
- 代码内含 CANN 9.2.0 上板实测签名勘误——**v30-fa wrapper 的 QFA 传参经对照全部吻合**（交叉验证了 wrapper 正确性）

### 2.4 两候选模型量化矩阵（产物字段实证）

实验只在以下两个量化产物中选择（Eco-Tech 系不存在其他候选）：

| 模块 | **w8a8-mxfp8**（阶段一基线） | **w8a8c8-mxfp8**（阶段二三载体） |
|---|---|---|
| full_attn GEMM（q/k/v/o_proj） | W8A8_MXFP8 | W8A8_MXFP8 |
| MLP | W8A8_MXFP8 | W8A8_MXFP8 |
| GDN in_proj（linear_attn） | FLOAT（不量化） | W8A8_MXFP8 |
| vision tower | FLOAT（不量化） | W8A8_MXFP8 |
| KV cache | 无量化（BF16） | C8-MXFP（K 动态 + V 静态校准） |
| attention QKV/P | 无量化（BF16 attention） | MXFP8（QFA mode 1） |

两模型的 GEMM（full_attn + MLP）完全相同（W8A8_MXFP8）——差异集中在：GDN in_proj / vision 的量化、以及 attention 段（KV/QKV 量化 + QFA）。这正是阶段二三切换模型后跨背景层对比需注明的边界（GDN/vision 量化差异）。

---

## 3. 阶段一审核：FIA→FA wrapper 的正确性与覆盖边界

### 3.1 改动满足要求的确认

**需求**：w8a8-mxfp8 权重下，通过 wrapper（device_op.py 单点）将 eager attention 从 FIA 切换到 FA，支持 FA 的 softmax FP32/FP16 验证。

**确认结论：满足**。依据：
1. 单点收口：v0.30.0 attention_v1.py 的 eager causal 全部调用汇入 `:994`/`:1045`/`:1076` 的 `DeviceOperator.npu_fused_infer_attention_score`（A5 实现 = 我们的 wrapper）——一处改动覆盖全部 eager 路径
2. 用户实测通过（prefill+decode 打印验证 + FA 跑通 + softmax 双版本验证完成）
3. 接口审计归档（work-04）：签名/mask/descale 逐项核对，含官方文档与 tests/golden 源码双重 ground truth
4. FIA 默认路径零变化（diff 为纯插入，328 insertions 0 deletions）——基线不受污染

### 3.2 eager 模式全覆盖的证据链

v0.30.0 attention_v1.py 的 FIA 调用点全量审计（行号已核对）：

| 调用点 | 场景 | 开关覆盖 |
|---|---|---|
| :994 wrapper | eager causal 主路径（全新 prefill/前缀续算/decode） | ✅ |
| :1045/:1076 wrapper | A5/batch-invariant chunked prefill 拆分 | ✅ |
| :921 v2 直调 | sinks（Gemma 类模型） | ❌ 不经 wrapper，但 Qwen3.6 无 sinks，死代码 |
| :941 v1 直调 | 非因果 | ❌ 死代码（因果 LM） |
| :956 v1 直调 | 滑窗 | ❌ 死代码（Qwen3.6 full-attn 无滑窗） |
| :1552+ C8 路径 | C8 int8 KV 量化产物 | ❌ 死代码（w8a8-mxfp8 无 KV 量化字段） |
| mm_encoder_attention.py | VL vision tower | ❌ 纯文本数据集不触发；**多模态数据集会绕过开关**（边界条件） |
| :688/:760 full_graph_fia/v2 | **图捕获** | ❌ **真缺口**，见 3.4 |

结论：**eager + 纯文本 + 无 MTP 配置下，wrapper 100% 覆盖实际会触发的 attention 节点**（MTP 开启时 spec draft/verify 亦汇入 causal wrapper，同样覆盖）。

### 3.3 非 eager（图模式）的遗漏分析

**遗漏点**：图捕获期间（`_EXTRA_CTX.capturing=True`），attention 分流到 `full_graph_fia`（:688，v1 `.out` 直调）/`full_graph_fia_v2`（:760）——**完全绕过 wrapper**；图回放期间 Python 不重新执行，graph task 机制只更新 FIA 算子入参。

**造成的具体问题**：

```
图模式 + A5_ATTENTION_IMPL="FA" 运行时：
  prefill 段：eager（图只捕 decode）→ wrapper → FA ✓ 生效
  decode 段：图捕获走 full_graph_fia → FIA；回放重放 FIA
             → FA 开关静默失效，decode 段跑的是 FIA
```

**危害性质**：静默失效（不报错、结果数值正确但是 FIA 的）——实验数据看起来"跑通了"，但 decode 段的 FA/softmax 数据是假的。**这正是 softmax 实验不可接受的失败模式**：若 softmax FP16 kernel 的差异主要体现在 decode 段，图模式数据将完全测不到。

**范围界定**：
- 阶段一已完成（eager 验证 + softmax 结论在 eager 下取得）——**不受此遗漏影响，结论有效**
- 若阶段一需要图模式性能数据：遗漏使 FA 在图模式下不可用；补救 = capture 放行改造（~20 行：capturing 分支放行 FA + `seq_lens` 改从 `attn_metadata.seq_lens` 设备张量取以避免回放值冻结 + 条件性跳过 FIA padding），**已分析未实施**
- QFA 分支的 wrapper 路线即使做了图放行也不可用：V 量化的 `cu_seqlens_kv.tolist()` 是 host 侧取值，回放期不重跑 Python → 分段边界冻结为捕获值 → **静默算错**（比失效更危险）

---

## 4. 阶段二方案选择：为何弃 wrapper+QFA 改造，选 w8a8c8 模型切换

### 4.1 QFA 全路径覆盖的技术必然：cache 数据格式

QFA 算子接口定义（不可协商）：k/v 必须是 `float8_e4m3fn` + e8m0 descale——**它是量化 attention 算子，只吃量化 KV**，无 BF16 选项（dtype 校验直接拦截）。

量化时机决定了路径可行性：

| KV 来源 | 可行时机 | wrapper 现状 |
|---|---|---|
| prefill 的 K/V（当步投影输出，数据在手） | **喂算子前量化**（在线） | ✅ 已实现（QFA 分支的 TND 路径） |
| decode/前缀续算的 KV（历史数据，存储于 cache） | **只能写入时量化**（读取时数据格式已定） | ❌ cache 是 BF16 → 回退 FIA |

读后量化（gather→quant→QFA）被排除：每层每步全 KV 拷贝+量化（性能倒退）、V 分段 tolist 的 host 同步、**图模式永久不可用**（host 逻辑回放不重跑）。

**结论**：wrapper 路线要实现 QFA 全路径（decode/前缀续算/chunked 全走 QFA），**必然要求 cache 改存 MxFP8（L2）**——没有绕开方案。

### 4.2 L2 的工程构成论证（wrapper 路线的真实成本）

L2 = cache 分配 + 写入路径 + attention 统一路径 + swap 四块改造（详见 `artifacts/l2-qfa-full-coverage-design.md`）：

1. **cache 分配**：fp8 数据 cache ×2 + k_descale `(Bn,Bs,N,⌈D/64⌉,2)` + v_descale `(Bn,Bs/64,N,D,2)` + V 尾块池；按 `A5_ATTENTION_IMPL` 开关分配（与 FA/FIA 的 BF16 cache 互斥，切模式需重启）
2. **写入路径（最复杂）**：K 逐 token 量化 scatter（可复用 `npu_dynamic_mx_quant`）；**V 必须尾块机制**——QFA 的 v_descale 是 **per-64-token 共享块**（接口定义），decode 每步追加 token 会改变所在 64 块的 max → 已按旧 scale 量化的前序 token 全部失效 → 必须 BF16 尾块暂存 + 满 64 固化（量化写入主 cache + descale 行）
3. **swap/PD**：descale + 尾块随块迁移（尾块不在 block_table 内，需专门处理）
4. **组合无先例**：`npu_dynamic_mx_quant × quant_flash_attn` 的动态 V 组合没有任何现成用例（生产用静态 V）——golden 对拍是必做且沉重的验证负担

**工作量**：数天开发 + 调试 + 验证；且覆盖正确性（基线可靠性）需要自证。

### 4.3 生产路径的能力（为何它是更优解）

w8a8c8-mxfp8 + `--kv-cache-dtype mxfp8` 激活的 C8-MXFP backend（§2.3）：

- **覆盖正确性现成**：impl 类替换结构性接管全部 attention（prefill/decode/chunked/MTP/图模式），上板验证（golden GRAPH_PATH=7）
- **V 量化的生产解法**：静态校准 scale（checkpoint `fa_v.scale`）——增量写入问题直接消失（校准数据来自 msmodelslim，w8a8c8 产物自带）
- **零代码**：激活链已双向实证（产物字段 ↔ 代码触发条件逐字匹配）
- **图模式原生**：阶段一的 eager-only 限制在此路径不存在

### 4.4 方案对比与决策

| 维度 | wrapper+QFA（需 L2） | **w8a8c8 模型切换（生产路径）** |
|---|---|---|
| QFA-FP32 基线覆盖正确性 | 需自证（L2 自建 + golden 对拍） | **现成上板验证** |
| 工程量 | L2 全套（数天） | **零**（环境准备即基线） |
| V 量化 | 动态 per-64（需尾块机制） | 静态校准（产物自带） |
| 图模式 | 永久不可用（tolist 陷阱） | 原生支持 |
| softmax 归因（B1 vs B2） | 组内同模型 ✓ | **组内同模型 ✓（同等干净）** |
| 实验背景语义 | 自定义管线（动态 V 无生产对应） | **真实部署极限形态**（校准 V + KV 量化 + GDN/vision 量化——正是极限形态的完整定义） |
| 与阶段一的连续性 | 同模型连续 | 断裂（需 B0 补跑，成本一次基线评测） |

**决策：w8a8c8 模型切换**。理由排序：① 基线覆盖正确性（实验零点）由生产代码保证；② softmax 归因同等干净（组内对照）；③ 工程量从数天降为零；④ 实验背景恰好是"极限形态"的真实定义（本实验的科学语境）；⑤ 图模式限制消失。

**v30-fa 分支的最终定位**：阶段一成果归档（FIA/FA 切换工具 + FA softmax 验证载体，eager 有效，图模式限制见 §3.4）；wrapper 的 QFA 分支保留为"动态 V 量化对照变体"——若后续需消融"动态 per-64 vs 静态校准 V"的影响，它是现成载体。

---

## 5. 实验矩阵终稿

见 §1.2。执行顺序：B0（基线）→ B1（QFA-FP32 基线，极限形态）→ B2（QFA-FP16，终极目标）。

**归因规则**（报告必须注明）：
- B0 vs B1：attention 量化代价（同模型，唯一变量 = KV+QKV attention 量化）——前提 B0 可行（§6.V1）
- B1 vs B2：softmax 精度（同模型同管线，唯一变量）——**终极目标，归因不受任何模型切换影响**
- FA-FP16（阶段一）vs B2（阶段二三）：跨背景层对比（softmax 敏感度 vs attention 量化背景），**跨模型**——GDN in_proj/vision 量化差异是边界，注明
- 标准实验配置：`--enforce-eager`（阶段二三可开图模式对照性能，但精度主实验建议 eager 统一条件）+ 无 MTP（接受率低性能差已实证；MTP num=1 与图模式互斥待三组对比，均与精度归因无关，纯性能维度）+ `--enable-prefix-caching false` + `--max-num-batched-tokens ≥ 最长 prompt`（消除前缀/切块对覆盖分析的干扰）

---

## 6. 验证清单（准确性保障）

### V1. B0 可行性冒烟（优先级最高，决定 B0 是否成立）
- 操作：w8a8c8-mxfp8 **不加** `--kv-cache-dtype` 启动
- 判据：正常跑通（FIA-BF16 attention + checkpoint 的 fa_v/fa_q/fa_k scale 被容忍加载）→ B0 成立
- 若失败（产物 recipe 无开关报错）：B0 退化为 w8a8-mxfp8 的 FIA 基线近似，"极限形态代价"对比注明跨模型边界（GDN/vision 量化差异混入）
- 依据：modelslim_config.py:465-470 有"有 recipe 未开开关"的处理分支（具体行为需实测）

### V2. B1 激活与覆盖确认
- 启动日志确认：`[quantization] C8 MXFP8 KV cache enabled (--kv-cache-dtype mxfp8)`
- GDN 层兼容确认：Qwen3.6-27B 是 hybrid（48 GDN + 16 full-attn）——代码注释声明 hybrid 支持（`_is_full_attention_layer` 判断 + linear-attention 层不 reach backend），**需实测确认 16 个 full-attn 层激活、48 个 GDN 层不受影响**
- 覆盖确认（**主证据，方案详见 §8 Profiling 操作指南**）：msprof 抓稳态，确认 `QuantFlashAttn` kernel 在 prefill+decode 均出现、无 `FusedInferAttentionScore` 残留（16 层全走 QFA）；GDN kernel 正常共存

### V3. 精度合理性检查（每级基线）
- B0 vs B1 的数据集分数差应 ≈ attention 量化量级（MXFP8 相对误差 ~6% 背景 → 数据集分数掉点通常 <1-2%）——**量级偏离（异常大或 B1 反常升高）即报警排查**
- B1 vs BF16 官方基线的差异合理（w8a8c8 全量量化 vs 原始 BF16，掉点幅度参照 Eco-Tech README 的精度表）

### V4. FP16 kernel 就位后（B2 前置）
- 算子级 golden：QFA-FP16 vs QFA-FP32 差异量级符合预期（softmax FP16 相对误差 ~0.1%，远小于 MXFP8 背景 ~6%）
- B1 复跑确认（kernel 更换后 FP32 侧不变——同一合体 kernel 的 mode 1 路径）

### V5. 实验条件一致性
- 同数据集、同评测框架（evalscope/lm_eval，参照 cann-recipes-infer 的 baseline 组织方式）、同温度/采样参数
- 多次运行取波动（README 提示"精度存在波动，建议多次测试"）

---

## 7. 执行计划

### 阶段二三主体（本计划）

| # | 任务 | 负责 | 依赖 | 产出 |
|---|---|---|---|---|
| 1 | A5 执行环境确认（环境信息待模型侧负责人提供：机型/地址）：v0.30.0 镜像、CANN 9.2.0、`cann_ops_transformer` 扩展包（版本配套：custom opp 包 + torch 扩展必须同版本安装） | 环境执行者 | — | 环境就绪 |
| 2 | 下载 `Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8` 权重至 A5 执行环境（~54GB，10 分片） | 环境执行者 | 1 | 权重就位 |
| 3 | **V1 冒烟**（B0 可行性）+ **V2**（B1 激活/覆盖确认） | 环境执行者 | 2 | B0/B1 可行性结论 |
| 4 | **B0 基线评测**（数据集精度） | 实验执行者 | 3 | B0 分数 |
| 5 | **B1 基线评测**（QFA-FP32，极限形态） | 实验执行者 | 3 | B1 分数 + B0/B1 差（极限形态代价） |
| 6 | FP16 kernel：experimental mode 3（MxFP8+Softmax FP16）合入主线框架，获得 D=256 + PA 支持（合体 kernel）；集成点 = `attention_c8_mxfp.py:450` 的 `QFA_QUANT_MODE_MXFP8` 常量（1→FP16 模式值） | **模型侧负责人**（FA kernel 改造经验直接复用） | 可与 3-5 并行 | 合体 kernel + backend 切换 |
| 7 | **V4 验证** + **B2 评测**（QFA-FP16，终极目标） | 实验执行者 | 5, 6 | B2 分数 + **B1/B2 差（softmax 结论）** |
| 8 | 报告：实验矩阵全表 + 归因边界注明（§5 规则）+ 背景层对比（FA-FP16 vs B2） | 实验执行者 | 7 | 最终报告 |

### 可选项（非实验必需）

- 阶段一图模式性能数据：v30-fa 的 capture 放行改造（~20 行，方案已分析：capturing 分支放行 FA + seq_lens 设备化 + 条件 padding 跳过）
- 动态 V vs 静态校准 V 的消融：v30-fa wrapper 的 QFA 分支（仅 prefill，eager）作为对照变体
- MTP num=1 vs 关闭 vs 图模式的三组性能对比（与精度实验正交）

### 关键风险与降级路径

| 风险 | 影响 | 降级 |
|---|---|---|
| B0 不可行（V1 失败） | 极限形态代价对比跨模型 | 用 w8a8-mxfp8 FIA 基线近似 + 注明边界；softmax 归因（B1 vs B2）不受影响 |
| C8-MXFP backend 对 hybrid 模型不兼容（V2 失败） | 阶段二三载体失效 | 回退 wrapper+L2 路线（§4.2，数天）或修复兼容问题 |
| FP16 合体 kernel 进度 | B2 延后 | B0/B1 独立成立（极限形态代价结论先行） |
| B1 精度异常（V3 报警） | 基线可疑 | 覆盖确认（V2 的 profiler）+ 激活日志复查 + 单层 golden + **启用 v30-fa QFA 探针（动态 V vs 静态校准 V 消融）** |

---

## 8. Profiling 操作指南（路径确认工具）

### 8.1 为什么需要 profiling（原理）

```
日志打印 = host 侧 Python 执行时触发 → 图回放期 Python 不重跑 → 图模式下永远打不出来
profiling = 设备侧 kernel 执行记录 → 图回放期 kernel 照常在 AICore 上执行 → 照样被记录
```

图模式下确认实际调用路径，profiling 是**正确且几乎唯一**的可靠手段。

### 8.2 工具选择（关键分水岭：图模式）

| | torch_npu.profiler | **msprof（系统级）** |
|---|---|---|
| 采集机制 | 框架层 hook（host 侧逐算子记录） | **直接采 AICore 硬件事件** |
| eager 路径 | ✓ 逐算子 kernel 名 | ✓ |
| **图回放路径** | **风险：torchair 整图下发，host 无逐算子调度——图内 kernel 可能记录不到或聚合成整段** | **✓ 图内每个 kernel 照常记录**（设备上真实执行的每个 AICore kernel 均被硬件事件采集覆盖） |

**规则：eager 路径两者皆可；图模式路径确认必须用 msprof。**

### 8.3 torch_npu.profiler 用法（v0.30.0 内置支持）

vllm-ascend v0.30.0 内置 `TorchNPUProfilerWrapper`（`vllm_ascend/profiler/torch_npu_profiler.py`：CPU+NPU activities，Level1 采样，tensorboard_trace_handler 输出 Chrome trace）。

**注意：旧环境变量 `VLLM_TORCH_PROFILER_DIR` 在 v0.30.0 已移除**，必须用 `--profiler-config` 传参。

```bash
# 方式 A：在线服务 + API 触发（推荐，可控采样窗口）
vllm serve <model> ... \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/path/to/prof_out
# 预热完成后（确保图捕获/缓存就绪），抓稳态：
curl -X POST http://127.0.0.1:8000/start_profile
#   ← 保持请求在跑（最好有正在 decode 的长输出请求），3-5 秒即可（trace 很大别贪长）
curl -X POST http://127.0.0.1:8000/stop_profile

# 方式 B：离线 benchmark 自动化（最省事：自动 warmup → start_profile → generate → stop_profile）
vllm bench latency -m <model> --output-len 256 --profile \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/path/to/prof_out
```

输出：每个 worker 一个 Chrome trace（.json），在 torch_profiler_dir 下。

### 8.4 msprof 用法（图模式必用）

```bash
# 系统级采集：--delay 等服务启动+图捕获/预热完成后，再采稳态窗口
msprof --output=./prof_graph \
    --application="vllm serve <model> ... <图模式配置>" \
    --delay=180 --duration=15
```

- 输出：kernel 明细 CSV（可直接 grep）+ .npu-rep（可视化报告，可用 MindStudio Insight 打开）
- 开销约 3-5%，只抓稳态窗口，避免把图捕获期/预热期混进样本
- 不依赖任何 Python hook，对 torchair 图回放/ACL graph 均有效

### 8.5 Kernel 名对照表（trace 中搜索这些字符串）

| trace 中的 kernel | 含义 | 对应验证 |
|---|---|---|
| `aclnnFusedInferAttentionScore` / `...V2` | FIA 在跑（含图内直调） | B0（FIA-BF16 基线）；图模式下 FA 开关失效检测 |
| `FlashAttn...`（ops-transformer flash_attn 算子族） | **FA 在跑** | 阶段一图模式放行改造后的确认 |
| `QuantFlashAttn...` | **QFA 在跑** | **V2：B1/B2 覆盖确认（核心）** |
| `quant_flash_attn_metadata` / `flash_attn_metadata` | QFA/FA 的 metadata 前置算子 | 佐证 QFA/FA 链路完整 |
| `npu_dynamic_mx_quant` | Q/K/V 在线 MxFP8 量化 | QFA 链路完整性的佐证 |
| `causal_conv1d` / `recurrent_gated_delta` / chunk 系 | GDN（linear attention）kernel | 确认 GDN 层不受 QFA backend 影响 |

### 8.6 各验证项的 profiling 方案

| 验证项 | 工具 | 检查点 |
|---|---|---|
| **V2-B1 覆盖确认**（最重要） | msprof（图模式）/ 两者皆可（eager） | prefill 窗口 + decode 窗口**均见** `QuantFlashAttn`；**无** `FusedInferAttentionScore` 残留（16 个 full-attn 层全走 QFA）；GDN kernel 正常共存（48 层不受影响） |
| 图模式 FA 开关确认（若做放行改造） | **msprof** | 图回放段：`FlashAttn` 出现 = 放行生效；`FusedInferAttentionScore` 出现 = 静默失效（即 §3.4 所述问题的实测判据） |
| B2 切换确认（FP16 kernel 就位后） | msprof | 见 8.7 限制①——建议合体 kernel 用可区分算子名，一步解决 |
| 性能数据（顺带） | 两者 | kernel 执行时长/占比（QFA vs FIA 的算子级对比） |

### 8.7 限制与注意事项（诚实边界）

1. **算子内部模式不可直接区分**：QFA mode 1（softmax FP32）vs mode 3/合体（softmax FP16）若编译后**同名**，trace 无法从名字区分。两种解法：
   - **首选**：FP16 合体 kernel 编译时**用不同的算子名**（如 `QuantFlashAttnFp16`）——把确认问题变成"搜字符串"
   - 次选：执行时长间接判断（FP16 softmax 的 kernel 时长应明显短于 FP32 版）
2. **采样窗口**：profiling 有 3-5% 开销，只抓稳态（预热/图捕获完成后），窗口 10-15 秒足够
3. **trace 体量**：长序列高并发下单次 trace 可达数百 MB——用 **Perfetto**（ui.perfetto.dev）或 MindStudio Insight 打开后**直接搜 kernel 名字符串**，不要肉眼翻
4. **多 worker**：TP>1 时每个 rank 各出一个 trace 文件，检查任一 rank 即可（同构）

---

## 附：证据索引

| 结论 | 证据 |
|---|---|
| wrapper eager 全覆盖 | v0.30.0 attention_v1.py :994/:1045/:1076 调用点审计 + 用户实测打印 |
| wrapper 图模式失效 | attention_v1.py :688/:760 `full_graph_fia` 直调绕过 wrapper（代码事实） |
| QFA 只吃量化 KV | quant_flash_attn 文档参数表（k/v dtype 强制 fp8/e8m0）；v30-fa wrapper 回退设计归档 |
| V 尾块必然性 | v_descale 接口定义 per-64-token 共享（文档 shape 公式 + golden 的 per_channel_group 量化数学） |
| 生产 backend 能力 | attention_c8_mxfp.py 通读（work-05 归档）+ GRAPH_PATH=7 上板验证注释 |
| 激活链匹配 | w8a8c8-mxfp8 产物 quant_model_description.json 实抓（full-attn 层 quant_type 逐字匹配）+ modelslim_config.py:730-743/:828 触发逻辑 |
| 三模型量化矩阵 | 三个模型的 quant_model_description.json 实抓对比 |
| npu_dynamic_mx_quant 输出 | op-plugin 官方文档（uint8 dtype + shape 公式 rank+1/尾维2/⌈n/64⌉） |
| FA/QFA 签名 | 官方 torchapi 文档 + attention_c8_mxfp.py 内 CANN 9.2.0 上板勘误（与 v30-fa 传参对照吻合） |
| profiling 工具差异（图模式） | torchair 整图下发（host 无逐算子调度）vs msprof AICore 硬件事件采集——原理性差异；vllm-ascend v0.30.0 profiler 支持实证（TorchNPUProfilerWrapper + --profiler-config 参数，VLLM_TORCH_PROFILER_DIR 已移除） |
