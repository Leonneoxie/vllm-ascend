# 基线复测参数归档（params/）

本目录归档各场景**基线复测参数**（vLLM 侧 sidecar）。参数文件名与 `../configs/`
配置文件一一同名；复测流程与验收标准见 [`../baseline-guide.md`](../baseline-guide.md)。

当前提供下表中的 FlatQuant MLP 与 LHT 三场景参数。其余算法和 scope 的归档后续补充，不能仅凭配置文件认定已有可复现参数。

## 参数清单

| 场景 | 文件 | 键数 | 大小 | SHA256 |
|---|---|---|---|---|
| flatquant mlp | `flatquant/qwen3_5_9b_flatquant_mlp-only_w4a4.safetensors` | 288 | 9.91 MB | `6d47223d8431f525d0bbcbf9e4588ee39a8b482bc91607aef27ca845abd8bb6e` |
| lht attn | `lht/qwen3_5_9b_lht_attn-only_w4a4.safetensors` | 88 | 5.51 MB | `d8ed373839690f4250f5d008c7ff358839302b027a46d9ed29c3a51ef16b2691` |
| lht mlp | `lht/qwen3_5_9b_lht_mlp-only_w4a4.safetensors` | 96 | 6.01 MB | `90ab1d04debd5dbab60aca10452f63bc804231dac1e9b8f310c43c159bdecb24` |
| lht attn-mlp | `lht/qwen3_5_9b_lht_attn-mlp_w4a4.safetensors` | 184 | 11.52 MB | `ccb02d8cc6ef54babe536a132633bcb9411e05c729c37db9353ea692d983ca97` |

## 文件格式（lht 示例）

- 88 键 = 24 层 linear_attn × {in_proj_qkvz, in_proj_ba, out_proj} + 8 层 self_attn × {qkv_proj, o_proj}；96 键 = 32 层 mlp × {gate_proj, up_proj, down_proj}；attn-mlp 为两者合并 184 键
- 形状：128×128；dtype：float32；in_proj_qkvz/in_proj_ba 共享同一输入变换（值相同），gate/up 同理
- 正交性：导出即有效正交矩阵（max ‖QQᵀ−I‖ ≈ 1.07e-06）；加载期由 `validate_lht_matrix` 守卫（容差 1e-4）拒收非正交产物
- 由 `scripts/convert/convert_ptq_to_vllm.py --algo lht` 从 AMCT PTQ 产物转换（本批参数源：AMCT lht 分支 9c11a28，Qwen3.5-9B W4A4，训练超参对齐 16 场景基线）
- 复测参考：MATH500 全量 0.900 / 0.930 / 0.856（attn / mlp / attn-mlp），WikiText-2 PPL 7.878 / 8.283 / 8.217

## 文件格式（flatquant mlp 示例）

- 288 键 = 32 层 × {gate_proj, up_proj, down_proj} × {left_trans, right_trans, diag_scale}
- 形状：gate/up `left (32,32) right (128,128) diag (4096,)`；down `left (96,96) right (128,128) diag (12288,)`
- dtype：float32；gate/up 共享同一输入变换（值相同）
- 由 `scripts/convert/convert_ptq_to_vllm.py` 从 AMCT PTQ 产物转换：映射参数键名并统一保存为 FP32，不导出变换后的模型权重

## 维护（新增参数入库）

仓库已配置这些参数的 LFS 跟踪规则，无需重复修改 `.gitattributes`。

```bash
# 仓库根目录（一次性）
git lfs install

# 新增文件：params/<algo>/<与 configs 同名>.safetensors + 上表加一行（含完整 SHA256）
git add examples/quantization/fake_mx/w4a4_eval/params/
git commit -s -m "docs(quantization): archive <algo> <scope> baseline params"
```

> 若仓库未启用 LFS：参数挂 release 资产，本表记 SHA256 与下载链接，不直接 commit。

## 边界说明

本目录只收 vLLM 侧复测所需算法 sidecar；复测仍需原始 BF16 模型、对应配置和评测数据。AMCT 侧训练产物
（`layer_*.pt`）不入仓，由训练侧环境留存。
