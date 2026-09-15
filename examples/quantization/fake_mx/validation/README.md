# Fake MX validation tools

This directory contains reproducible Qwen3.5-9B smoke and kernel-parity tools.
The source checkpoint is never modified: the smoke runner creates a temporary
model view, symlinks checkpoint files, and installs the selected config as
`quant_model_description.json`.

## Configurations

- `configs/qwen3_5_9b_attn_linear_w4a4.json`: Full Attention and GDN projection
  Linear modules use MXFP4 W4A4.
- `configs/qwen3_5_9b_attn_linear_w8a8.json`: the same projections use MXFP8
  W8A8.
- `configs/qwen3_5_9b_attn_linear_mixed.json`: qkv/GDN input projections use
  W4A4; o/GDN output projections use W8A8.
- `configs/qwen3_5_9b_mlp_w4a4.json`: only dense MLP gate/up/down projections
  use MXFP4 W4A4.
- `configs/qwen3_5_9b_mlp_w8a8.json`: only dense MLP gate/up/down projections
  use MXFP8 W8A8.
- `configs/qwen3_5_9b_mlp_mixed.json`: only dense MLP is quantized; fused
  gate/up uses W4A4 and down uses W8A8.

All other modules are explicitly FLOAT. The MLP-only configs simply do not
select Attention/GDN projection Linear modules in `module_quant_overrides`
(the non-Linear `attn-cache` target and its `fake_mx_quant_targets` key were
removed in the fake_mx_algorithms refactor).
The two legacy root-level JSON names remain as compatibility aliases for the
all-W4A4 and all-W8A8 attention configurations.

## Smoke test

```bash
python examples/quantization/fake_mx/validation/run_fake_mx_smoke.py \
  /path/to/Qwen3.5-9B \
  --quant-config examples/quantization/fake_mx/validation/configs/qwen3_5_9b_attn_linear_mixed.json \
  --output-json /tmp/qwen3_5_9b_fake_mx.json
```

This proves model loading, weight QDQ, prefill, and decode. It is not a
perplexity test or a kernel benchmark.

## Remote wrapper

The wrapper has no machine-specific overlay, NPU, or virtual-environment path:

```bash
export VLLM_ASCEND_ROOT=/home/xyj/dev/cann/vllm-ascend
export VLLM_ROOT=/home/xyj/dev/cann/vllm
export CANN_ENV_SCRIPT=/data/Ascend/cann-9.0.0/set_env.sh
export VIRTUAL_ENV_ACTIVATE=/path/to/venv/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0

bash examples/quantization/fake_mx/validation/run_remote_validation.sh \
  /path/to/Qwen3.5-9B \
  examples/quantization/fake_mx/validation/configs/qwen3_5_9b_attn_linear_mixed.json \
  /tmp/result.json
```

`run_remote_fake_mx.sh` remains as a compatibility alias to the new wrapper.
