# 基线复测参数归档（params/）

本目录归档各场景**基线复测参数**（vLLM 侧 sidecar）。参数文件名与 `../configs/`
配置文件一一同名；复测流程与验收标准见 [`../baseline-guide.md`](../baseline-guide.md)。

当前仅提供下表中的 FlatQuant MLP 参数。其他算法和 scope 的归档后续补充，不能仅凭配置文件认定已有可复现参数。

## 参数清单

| 场景 | 文件 | 键数 | 大小 | SHA256 |
|---|---|---|---|---|
| flatquant mlp | `flatquant/qwen3_5_9b_flatquant_mlp-only_w4a4.safetensors` | 288 | 9.91 MB | `6d47223d8431f525d0bbcbf9e4588ee39a8b482bc91607aef27ca845abd8bb6e` |

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
