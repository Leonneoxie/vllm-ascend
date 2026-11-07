# Attention 张量布局参考：TND / BNSD / BSND / PA_NZ / PA_BNBD / PA_BBND

> 本文档总结 attention 算子的主流张量布局：字母含义、形状、内存示意图、用途和兼容性。

---

## 1. 布局名称的字母含义

| 布局 | 字母拆解 | 含义 |
|---|---|---|
| **TND** | **T**oken - **N**umber of heads - **D**imension | Token 优先：所有请求的 token 打包成一条序列 |
| **BNSD** | **B**atch - **N**umber of heads - **S**equence - **D**imension | Batch 优先：每个请求独立一个 batch 元素 |
| **BSND** | **B**atch - **S**equence - **N**umber of heads - **D**imension | Batch 优先，序列在头之前 |
| **PA_NZ** | **P**aged **A**ttention - **N**on-**Z**ero | 分页存储 + Ascend 硬件碎片格式 |
| **PA_BNBD** | **P**aged **A**ttention - **B**lock - **N** - **B**lock - **D** | 分页存储，块-头-块-维 |
| **PA_BBND** | **P**aged **A**ttention - **B**lock - **B**lock - **N** - **D** | 分页存储，块-块-头-维 |

---

## 2. 布局形状与含义

### 2.1 TND：紧凑变长打包

```
形状：(T, N, D)

T = 所有请求的 token 总数（变长打包）
N = 注意力头数
D = 每头维度（如 256）

例：2 个请求（长度 100 和 200）
    q shape = (300, 24, 256)
    cu_seqlens_q = [0, 100, 300]   ← 定义每个请求的起止位置
```

### 2.2 BNSD：按 batch 展开（pad 到最大）

```
形状：(B, N, S_max, D)

B = batch 数（每个请求一个元素）
N = 注意力头数
S_max = 序列长度（pad 到 batch 内最大值）
D = 每头维度

例：2 个请求（长度 100 和 200）
    q shape = (2, 24, 200, 256)    ← pad 到 200
    seqused_q = [100, 200]         ← 告诉 kernel 每个请求的实际长度
```

### 2.3 BSND：按 batch 展开，序列在前

```
形状：(B, S_max, N, D)

与 BNSD 类似，但 S 和 N 的顺序交换。
部分算子偏好此布局（如 vLLM 内部部分路径）。
```

### 2.4 PA_NZ：分页存储 + 硬件碎片格式

```
K/V 数据形状（5D）：(Bn, N, D//32, Bs, 32)

Bn = 物理 cache 块数（如 10000）
N = KV 头数
D//32 = D 维分成 32 元素碎片
Bs = 每块 token 数（如 128）
32 = NZ 碎片大小（Ascend 硬件优化的内存布局）

K descale 形状（6D）：(Bn, N, Bs//16, D//64, 16, 2)
V descale 形状（6D）：(Bn, N, D//16, Bs//64, 16, 2)

例：cache 10000 块，4 KV 头，D=256，Bs=128
    k_cache shape = (10000, 4, 8, 128, 32)
    block_table = [[15, 42, 87...], ...]  ← 每个请求用哪些块
```

### 2.5 PA_BNBD / PA_BBND：分页存储（简化布局）

```
PA_BNBD：(Bn, N, Bs, D) — 4D
    块 → 头 → 块内序列 → 维度

PA_BBND：(Bn, Bs, N, D) — 4D
    块 → 块内序列 → 头 → 维度

两种都是 QFA 支持的 paged 布局（比 PA_NZ 简单，无碎片化）。
```

---

## 3. 内存示意图

### 3.1 TND：零浪费的紧凑打包

```
┌─────────────── request 0 (100 tokens) ───────────────┐┌────── request 1 (200 tokens) ──────┐
[t_0, t_1, t_2, ..., t_99, t_100, t_101, ..., t_199, ..., t_299]
└────────────────────────────────────────────────────────┘└─────────────────────────────────────┘
 ← cu_seqlens = [0, 100, 300] 定义边界 →

特点：零 padding 浪费，所有 token 连续存储。
缺点：变长信息需要额外的 cu_seqlens 张量。
```

### 3.2 BNSD：按请求独立存储（有 padding）

```
request 0:  [t_0, t_1, ..., t_99, PAD, PAD, PAD, ..., PAD]  ← 200 slots（只用 100）
request 1:  [t_0, t_1, ..., t_199]                            ← 200 slots（全用）

特点：每个请求独立、shape 规整、适合标准 batch 操作。
缺点：短序列 pad 浪费（本例 request 0 浪费 50%）。
seqused_q = [100, 200] 告诉 kernel 跳过 PAD。
```

### 3.3 PA_NZ / PA_BNBD / PA_BBND：分页存储

```
物理内存：
  块 0:     [128 tokens]  ← 可能属于任何请求
  块 1:     [128 tokens]  ← 可能属于任何请求
  块 2:     [128 tokens]
  ...
  块 9999:  [128 tokens]

block_table（请求 → 块映射）：
  request 0 → [块15, 块42, 块87, 块203, ...]
  request 1 → [块3, 块203, 块567, ...]
                ↑ 两个请求可以共享块（prefix cache）

特点：KV cache 的标准存储方式（vLLM 底层）。
  块可以动态分配/释放/共享，无外部碎片。
  请求的 KV 数据散布在多个非连续的物理块中。
  需要 block_table 导航。

PA_NZ 额外特点：每个块内部再按 32 元素碎片化，
  是 Ascend Cube 硬件的最优内存布局（矩阵乘效率最高），
  但形状最复杂（5D/6D）。

PA_BNBD/PA_BBND：块内不碎片化（4D），形状简单，
  是 QFA 支持的简化 paged 布局。
```

---

## 4. Descale（量化 scale）布局对比

| 布局 | K descale 形状 | V descale 形状 | 说明 |
|---|---|---|---|
| TND | (T, N, ⌈D/64⌉, 2) | (⌈T/64⌉, N, D, 2) | K per-token，V per-64-token |
| BNSD | (B, N, S_max, ⌈D/64⌉, 2) | (B, N, ⌈S_max/64⌉, D, 2) | 同上，按 batch 展开 |
| PA_NZ | (Bn, N, Bs//16, D//64, **16, 2**) — **6D** | (Bn, N, D//16, Bs//64, **16, 2**) — **6D** | 碎片化，最复杂 |
| PA_BNBD | (Bn, N, Bs, ⌈D/64⌉, 2) | (Bn, N, Bs//64, D, 2) | 简化 paged |
| PA_BBND | (Bn, Bs, N, ⌈D/64⌉, 2) | (Bn, Bs//64, N, D, 2) | 简化 paged |

注：尾维 `2` 表示 MX 块量化的 even/odd 配对（2 个 32-elem 块共享一个 64-elem 对）。

---

## 5. 各布局的用途与兼容性

### 5.1 按场景

| 场景 | 最佳布局 | 原因 |
|---|---|---|
| **Prefill**（K/V 在内存中） | **TND** | 当步投影输出是连续张量，零浪费 |
| **Decode**（K/V 在分页 cache 中） | **PA_NZ** / PA_BNBD / PA_BBND | 直读 cache，零拷贝 |
| **实验 QFA mode 3 (FP16)** | **BNSD only** | 实验版 kernel 限制 |
| **主线 QFA mode 1 (FP32)** | TND / BNSD / BSND / PA 均可 | 主线 kernel 全布局支持 |
| **生产 C8-MXFP backend** | **PA_NZ only** | 深度绑定 packet-packed cache |

### 5.2 按算子

| 算子 | TND | BNSD | BSND | PA_BNBD | PA_BBND | PA_NZ |
|---|---|---|---|---|---|---|
| **主线 QFA mode 1** (FP32) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **实验 QFA mode 3** (FP16) | ❌ | **✅ 唯一** | ❌ | ❌ | ❌ | ❌ |
| **主线 flash_attn** (BF16) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **FIA v1/v2** | ✅ | ✅ | ✅ | — | — | — |

### 5.3 按代码路径

| 代码路径 | 布局 | 场景 |
|---|---|---|
| wrapper QFA prefill（非 L2） | **TND** | 在线量化，K/V 是当步张量 |
| wrapper QFA prefill mode 3 | **BNSD** | TND → BNSD 转换（wrapper 内） |
| wrapper QFA gather | **TND** → BNSD | gather 后拼 TND，mode 3 转 BNSD |
| wrapper QFA L2 | **PA_BBND** | fp8 cache 直读 |
| wrapper FA prefill | **TND** | BF16 连续张量 |
| wrapper FA decode | **PA_BBND** | BF16 cache 直读 |
| 生产 C8-MXFP backend | **PA_NZ** | packet-packed cache 直读 |

---

## 6. 布局转换

| 从 → 到 | 转换方式 | 开销 |
|---|---|---|
| TND → BNSD | split by cu_seqlens → pad → stack | pad 浪费 + permute 拷贝 |
| TND → PA_BNBD | scatter by block_table | 需要知道块映射 |
| BNSD → TND | extract valid tokens (用 seqused) → cat | 提取有效部分 |
| PA_NZ → BNSD | gather blocks → reshape → pad | 最复杂（5D→4D + 碎片重组） |
| PA_BNBD → BNSD | gather blocks → reshape → pad | 较简单（4D→4D） |

### B=1 时的特殊情况

推荐实验配置（`--max-num-batched-tokens ≥ 最长 prompt`）下每次只有一个请求（B=1）：

- TND → BNSD = 简单 permute（(S,N,D) → (1,N,S,D)），零 padding 浪费
- 开销可忽略

---

## 7. 布局选择的决策树

```
K/V 在哪里？
  ├─ 内存中（当步投影输出）
  │   └─ QFA mode?
  │       ├─ mode 1 (FP32) → TND（直接用）
  │       └─ mode 3 (FP16) → BNSD（wrapper 转）
  │
  └─ KV cache 中（分页存储）
      └─ Cache 格式？
          ├─ BF16 → gather 出来 → TND → mode 3 时转 BNSD
          ├─ fp8 (L2) → PA_BBND 直读（mode 1）
          │            └─ mode 3 需 gather → TND → BNSD
          └─ fp8 (生产 PA_NZ) → PA_NZ 直读（mode 1 only）
                       └─ mode 3 不兼容（需 gather + 复杂转换）
```
