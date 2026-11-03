# L2 补齐：wrapper QFA + cache MxFP8 化 完整修改点地图

> 状态：**方案 B（静态 V scale）已选定落地**。方案 A（动态 per-64 + 尾块）保留为设计参考。
> 默认关闭 = `A5_ATTENTION_IMPL = "FIA"` 时零行为变化。
> 分支：v30-fa 基础上新增 commit。
> 参考：生产实现 `attention_c8_mxfp.py`（已上板验证，修改点结构同构）。

---

## 方案 A vs B 摘要

| | 方案 A（动态 per-64 + 尾块） | **方案 B（静态 V scale，已选定）** |
|---|---|---|
| V scale 来源 | 运行时逐组计算 max | neutral default（或自校准），分配期广播 |
| 尾块 | **必须**（增量写入的不完备问题） | **不需要**（scale 数据无关 → 逐 token 独立） |
| 写入复杂度 | 最高（per-request 循环 + 尾块状态 + 固化） | 中（fill descale + npu_quantize + scatter） |
| 图模式 | 永久不可用（tolist host 陷阱） | 理论可行（零 host 逻辑） |
| 精度 | 理论上限（动态自适应） | 接近（FP8 背景噪声 ~6% 下差距 <1%） |
| 工作量 | ~600 行 | **~400 行** |

**尾块的本质**：动态 scale 的"数据依赖性"与增量写入的"时序性"冲突。B 的 scale 不依赖运行时数据 → 每个 token 独立量化写入 → 无组依赖 → 无尾块。

---

## 修改点总览（方案 B，5 个）

```
① A5_ATTENTION_IMPL_L2 开关 + cache 布局工具函数     device_op.py（新增）
② cache 分配分流                                      model_runner_v1.py（:6352 附近）
③ 写入路径（K 量化 scatter + V 尾块追加/固化）          attention_v1.py reshape_and_cache
④ attention 统一路径（PA + v_tail 三件套）             device_op.py wrapper QFA 分支重写
⑤ swap 分流                                           model_runner_v1.py swap_blocks 附近
```

---

## 修改点 ① ：开关 + cache 布局工具（device_op.py 新增）

```python
# =====================================================================
# A5 L2 cache MxFP8 switch (experimental, wrapper-QFA full-path).
# Set both A5_ATTENTION_IMPL = "QFA" and A5_ATTENTION_L2 = True to
# enable the MxFP8 KV cache (fp8 data + e8m0 descale buffers + v_tail
# pool). Default False = BF16 cache, current behavior unchanged.
# =====================================================================
A5_ATTENTION_L2 = False  # ← 主开关：True 时 cache 分配/写入/attention 全链切换

# QFA L2 cache 布局常量（packet 独立张量版，比生产的 packet-packed 简化）
_L2_K_CACHE_DTYPE = torch.float8_e4m3fn
_L2_SCALE_DTYPE = torch.float8_e8m0fnu  # 存储时用 uint8 view
_L2_MX_GROUP_SIZE = 32

def l2_k_descale_shape(num_blocks, block_size, num_kv_heads, head_dim):
    """(num_blocks, block_size, N, ceil(D/64), 2) — K per-token descale."""
    return (num_blocks, block_size, num_kv_heads, (head_dim + 63) // 64, 2)

def l2_v_descale_shape(num_blocks, block_size, num_kv_heads, head_dim):
    """(num_blocks, block_size // 64, N, D, 2) — V per-64-token descale."""
    return (num_blocks, block_size // 64, num_kv_heads, head_dim, 2)

def l2_v_tail_shape(max_num_seqs, block_size, num_kv_heads, head_dim):
    """(max_num_seqs, block_size, N, D) BF16 — V 高精尾块池."""
    return (max_num_seqs, block_size, num_kv_heads, head_dim)
```

---

## 修改点 ② ：cache 分配分流（model_runner_v1.py :6352-6357 附近）

**位置**：`initialize_kv_caches` 中 `isinstance(attn_module, Attention)` 分支

```python
# 现状（:6352-6357）：
elif isinstance(attn_module, Attention):
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        if self._is_c8_mxfp_kv_cache(spec):
            spec = mxfp_cache_spec(spec)
        kv_cache_spec[layer_name] = spec
        attn_layer_names.add(layer_name)

# 修改：加 L2 分流
elif isinstance(attn_module, Attention):
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        if self._is_c8_mxfp_kv_cache(spec):
            spec = mxfp_cache_spec(spec)
        elif A5_ATTENTION_L2 and A5_ATTENTION_IMPL == "QFA" and <is_A5>:
            spec = l2_mxfp_cache_spec(spec)  # ← 新函数：改 dtype + 加 descale/tail buffer
        kv_cache_spec[layer_name] = spec
        attn_layer_names.add(layer_name)
```

**新函数 `l2_mxfp_cache_spec`**（放 device_op.py 或独立 utils）：

```python
def l2_mxfp_cache_spec(spec):
    """Replace FullAttentionSpec with L2 MxFP8 version.

    Changes dtype to float8_e4m3fn and adds descale/tail buffer shapes
    so the cache allocator creates the full L2 tuple.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    return replace(
        spec,
        dtype=torch.float8_e4m3fn,
        # 额外：page_size_bytes 需包含 descale bytes（参考 mxfp_cache_spec 的做法）
    )
```

**分配处**（model_runner_v1.py :5514+ 附近，`_adjust_kv_layout` 之前）需要分流：

```python
# 在 kv_cache_shape 组装处加 L2 分支：
if A5_ATTENTION_L2 and A5_ATTENTION_IMPL == "QFA":
    # 返回多元组 cache：(k_fp8, v_fp8, k_descale, v_descale, v_tail_pool)
    k_shape = (num_blocks, block_size, N, D)
    v_shape = (num_blocks, block_size, N, D)
    k_ds_shape = l2_k_descale_shape(num_blocks, block_size, N, D)
    v_ds_shape = l2_v_descale_shape(num_blocks, block_size, N, D)
    v_tail_shape = l2_v_tail_shape(max_num_seqs, block_size, N, D)
    # 分配 5 个 tensor：
    kv_caches[layer_name] = (
        torch.empty(k_shape, dtype=torch.float8_e4m3fn, device=...),
        torch.empty(v_shape, dtype=torch.float8_e4m3fn, device=...),
        torch.empty(k_ds_shape, dtype=torch.uint8, device=...),
        torch.empty(v_ds_shape, dtype=torch.uint8, device=...),
        torch.zeros(v_tail_shape, dtype=torch.bfloat16, device=...),
    )
```

> **注意**：这需要绕过 vllm 的 `kv_cache_raw_tensors` 统一分配（它只支持单 dtype/tuple strides）——这是 L2 侵入 model_runner 最深的一刀。生产的 packet-packed 方案（mxfp_cache_spec）绕开了这个问题：单 tensor + strided views。**建议借鉴**。

---

## 修改点 ③ ：写入路径（attention_v1.py `reshape_and_cache` / device_op.py）

**位置**：`AscendAttentionBackendImpl.reshape_and_cache`（attention_v1.py）

```python
# 现状：
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) > 1:
        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        slots = attn_metadata.slot_mapping
        DeviceOperator.reshape_and_cache(key=..., value=..., key_cache=..., value_cache=..., slot_mapping=...)

# 修改：加 L2 分流
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if len(kv_cache) >= 5:  # L2: (k, v, k_ds, v_ds, v_tail) 五元组
        self._l2_reshape_and_cache(key, value, kv_cache, attn_metadata)
        return query, key, value, output
    # ... 原路径不变
```

**L2 写入的核心逻辑**（最复杂的部分）：

```python
def _l2_reshape_and_cache(self, key, value, kv_cache, attn_metadata):
    """L2 MxFP8 写入：K 量化 scatter + V 尾块追加/固化."""
    k_cache, v_cache, k_ds_cache, v_ds_cache, v_tail_pool = kv_cache
    num_tokens = key.shape[0]
    slot_mapping = attn_metadata.slot_mapping[:num_tokens]

    # ---- K：per-token 动态量化 + scatter（无共享问题）----
    k8, k_scale = torch_npu.npu_dynamic_mx_quant(
        key, dst_type=torch.float8_e4m3fn, axis=-1)
    # k_scale: (num_tokens, N, ceil(D/64), 2) — per-token 行
    # scatter k8 → k_cache（现有 npu_scatter_pa_kv_cache 可用，dtype fp8）
    # scatter k_scale → k_ds_cache（per-token 行写入，index_put_ 或自定义 op）

    # ---- V：尾块机制（最复杂）----
    # 逐请求处理：
    #   1. 尾块未满的 token → 追加到 v_tail_pool[req_idx]（BF16）
    #   2. 尾块满 64 时 → mx_quant(axis=0) 该 64 token → 写入主 cache + v_descale
    #   3. 重排剩余尾块到位置 0
    #
    # 需要维护 per-request 状态：固化边界 S_b = floor(L_b / 64) * 64
    # 可从 attn_metadata.seq_lens 推导（无需持久状态）：
    #   S_b = seq_lens[b] - (seq_lens[b] % 64)
    #   尾块长度 = seq_lens[b] % 64
    #   尾块内容 = v_tail_pool[req_slot_idx]  ← 需要知道请求的 slot 索引
    #   （从 attn_metadata.req_pool_slots 或 block_table 首块推导）

    seq_lens = attn_metadata.seq_lens.cpu().tolist()  # host 取值（eager OK，图模式不安全）
    block_tables = attn_metadata.block_tables
    block_size = k_cache.shape[1]  # (=128)

    for b, L in enumerate(seq_lens):
        # 本步新增的 token 范围（从 slot_mapping 推导）
        # ...（复杂度集中在 per-request 的 slot 映射 + 尾块状态管理）

        # 固化：满 64 的组量化写入
        # seg8, seg_scale = torch_npu.npu_dynamic_mx_quant(
        #     v_segment_64, dst_type=torch.float8_e4m3fn, axis=0)
        # v_cache[slots] = seg8
        # v_ds_cache[block_id, group_id] = seg_scale
```

> **写入是 L2 最复杂的部分**：per-request host 循环 + 尾块状态 + 固化时机。生产实现（attention_c8_mxfp.py）用**静态校准 V scale**绕开了整个尾块机制（`fill_mxfp_v_scale_cache` 分配期广播 + `npu_quantize` 逐 token 写入，零尾块逻辑）。**如果 L2 补齐也用静态 V scale**（比如用 neutral default 或运行时统计），写入复杂度骤降 80%。

---

## 修改点 ④ ：attention 统一路径（device_op.py wrapper QFA 分支重写）

**现状**（`_a5_quant_flash_attn`）：仅全新 prefill（block_table is None），在线量化。
**L2 版**：统一 PA + v_tail，所有状态走同一条路。

```python
@classmethod
def _a5_quant_flash_attn_l2(cls, query, key, value, attn_metadata, *, num_heads,
                            num_key_value_heads, head_size, scale, **kwargs):
    """L2 MxFP8 统一路径：所有 attention 状态走 PA + v_tail."""
    cann_ops = cls._a5_import_cann_ops("QFA")
    from vllm_ascend.device.device_op import A5_ATTENTION_L2
    if not A5_ATTENTION_L2:
        return None  # L2 未开 → 回退到现有 prefill-only 路径或 FIA

    # Q 在线量化（唯一保留的在线量化）
    q8, q_descale = cls._a5_mx_quant_qk(query)

    # cache 五元组
    # kv_cache 从 wrapper kwargs 获取或从 layer 获取
    k_cache, v_cache, k_ds, v_ds, v_tail_pool = <kv_cache_tuple>

    # PA 路径（所有状态统一）
    cu_seqlens_q = cls._a5_cu_seqlens(kwargs.get("actual_seq_lengths"), query.device)
    seqused_kv = cls._a5_seqused_from_metadata(attn_metadata,
                                                kwargs.get("actual_seq_lengths_kv"),
                                                query.device)
    block_table = kwargs.get("block_table")

    # v_tail 三件套
    seqused_v_tail = seqused_kv % 64                                # (B,) int32
    block_table_tail = <per-request tail block indices>             # (B, 1) int32
    # 静态分配：tail block index = request slot index（无需动态管理）

    metadata = cann_ops.quant_flash_attn_metadata(
        num_heads, num_key_value_heads, head_size, 1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        mask_mode=3,
        layout_q="TND", layout_q_descale="TND",
        layout_kv="PA_BBND", layout_out="TND",
    )
    attn_out, _ = cann_ops.quant_flash_attn(
        q8, k_cache, v_cache, q_descale, k_ds, v_ds, 1,
        block_table=block_table.to(torch.int32),
        seqused_kv=seqused_kv,
        v_tail=v_tail_pool,
        block_table_tail=block_table_tail,
        seqused_v_tail=seqused_v_tail,
        attn_mask=cls._a5_causal_mask_int8(kwargs.get("atten_mask")),
        metadata=metadata,
        softmax_scale=scale,
        mask_mode=3,
        layout_q="TND", layout_q_descale="TND",
        layout_kv="PA_BBND", layout_out="TND",
        return_softmax_lse=False,
    )
    return attn_out, None
```

---

## 修改点 ⑤ ：swap 分流

**位置**：model_runner_v1.py `swap_blocks` 附近

```python
# 现状：v0.30.0 的 swap_blocks 已支持 tuple cache（generic zip）
# L2 五元组 cache 的 swap 需要确认：
#   k_cache/v_cache/k_ds/v_ds：按 block 索引搬运 → 现有机制覆盖
#   v_tail_pool：不在 block_table 索引内 → **swap 不覆盖**
# 限制标注：v_tail 内容在 swap-out 时丢失（请求重新计算时重建）
# 对于单机 eager 实验场景：无 swap → 无影响
# 若需 PD/swap 支持：v_tail 需要随请求迁移（额外工程）
```

---

## 修改点之间的依赖顺序

```
① 开关+布局工具  →  ② cache 分配  →  ③ 写入路径  →  ④ attention 统一路径
                                              ↘  ⑤ swap（可选，单机 eager 可跳过）
```

---

## 复杂度评估

| 修改点 | 复杂度 | 难点 |
|---|---|---|
| ① 开关+工具 | 低 | 纯新增 |
| ② cache 分配 | **高** | 侵入 model_runner 深处，绕过 vllm 统一分配机制 |
| ③ 写入路径 | **最高** | per-request host 循环 + 尾块状态管理 + 固化时机 |
| ④ attention 统一路径 | 中 | 接口组装（QFA 签名已验证），v_tail 三件套的 block_table_tail 推导 |
| ⑤ swap | 低（可跳过） | 单机 eager 场景无 swap |

**总量估计**：400-600 行新代码 + model_runner 侵入 ~50 行修改。是 v30-fa 现有改动（391 行）的 1.5-2 倍规模。

---

## 关键设计决策（供你定夺）

| 决策点 | 选项 A | 选项 B | 建议 |
|---|---|---|---|
| V 量化方式 | 动态 per-64（尾块机制） | **静态校准/neutral**（广播填充，零尾块） | **B**：写入复杂度骤降 80%，QFA 算子不关心 descale 的来源 |
| cache 布局 | 独立五元组 | **packet-packed 单 tensor + views**（借鉴生产） | **B**：绕开 model_runner 统一分配的侵入 |
| 尾块管理 | per-request 动态 | **无尾块（静态 V 下不需要）** | 跟 V 量化方式联动 |

**如果选 B+B+B（静态 V + packet + 无尾块）**：L2 的实现形态趋近于生产 `attention_c8_mxfp.py` 的 wrapper 版移植——差异只剩激活方式（A5_ATTENTION_IMPL 开关 vs 量化产物）和 V scale 来源（wrapper 自己算/中性 vs checkpoint 校准）。
