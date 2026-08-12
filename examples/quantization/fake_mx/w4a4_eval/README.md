# Qwen3.5-9B Fake-MX 评测资产

本目录只保存配置和辅助脚本，不再重复解释算法或维护另一份结果结论。

唯一有效的使用说明是：

- `docs/source/developer_guide/qwen3_5_9b_fake_mx_online_algorithms_guide_v023.md`

## 目录

```text
w4a4_eval/
├── configs/                 # RTN/RHT/LHT/FlatQuant 等实验配置
└── scripts/
    ├── serve/               # vllm serve 辅助脚本
    ├── eval/                # 数据集评测脚本
    ├── ptq/                 # AMCT 参数训练/提取脚本
    └── convert/             # AMCT 参数到 safetensors 的转换工具
```

## 使用原则

1. 以 `configs/` 中 JSON 和当前代码支持的字段为准；
2. 模型始终加载原始 BF16/FP16 weight；RHT 无 PTQ 参数，LHT/FlatQuant 加载
   外部变换参数并在线处理 weight；
3. `module_quant_overrides` 按插入顺序首个匹配项生效，建议以 `"*": "FLOAT"`
   收尾；
4. `attn-linear`、MLP-only、attn+MLP 和 mixed 必须作为不同 scope 记录；
5. mixed W4A4/W8A8 是敏感层保护实验，不是 AMCT uniform W4A4 默认策略；
6. 运行结果必须同时记录配置文件内容、代码 commit、模型、数据集、上下文长度、
   TP、seed、sampling 和 `max_tokens`；
7. 目录内脚本是辅助工具，不是正确性证明。参数转换完成后仍需检查 key、shape、
   正逆变换等价性和运行时 scheme 命中日志。

## 当前结果口径

只在主指南维护已确认结果。目前可引用的身份是：RHT mixed 91.6%（无 PTQ）、
FlatQuant mixed 92.6%（PTQ）、LHT mixed 93.0%（`learnable_had` PTQ）。这些
mixed 结果不能与 uniform RTN 直接做纯算法归因。
