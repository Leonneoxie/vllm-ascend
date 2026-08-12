# Fake MXFP4/MXFP8 validation

> **Qwen3.5-9B guide:** use
> `docs/source/developer_guide/qwen3_5_9b_fake_mx_online_algorithms_guide_v023.md`
> for current Qwen3.5-9B configuration, online transforms, algorithm identity,
> results, and limitations.

Kernel 替换和新算法接入见
`docs/source/developer_guide/fake_mx_kernel_and_algorithm_extension_guide_v023.md`。

This path emulates MX quantization error on devices without native MXFP4 or
MXFP8 support. It never creates packed MX tensors. The default `reference`
backend uses ordinary floating-point operators, while an optional fused QDQ
kernel can replace that numerical implementation without moving quantization
boundaries:

1. Load the original FP16/BF16 checkpoint.
2. Round weights to the selected MX grid once in
   `process_weights_after_loading`.
3. Round activations to the selected MX grid on every forward.
4. Execute the ordinary floating-point linear or grouped-matmul kernel.

The stored parameter and activation dtypes remain FP16/BF16. The values include
the error from E2M1/E4M3 element rounding and one E8M0 power-of-two scale per
group.

## Configuration

Copy one of the sample files to the model directory as
`quant_model_description.json`, then run:

```bash
vllm serve /path/to/qwen3.5-model --quantization ascend
```

Supported scheme names:

- `W4A4_MXFP4_FAKE`: fake MXFP4 weights and activations.
- `W8A8_MXFP8_FAKE`: fake MXFP8 weights and activations.
- `W4A4_MXFP4_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP4.
- `W8A8_MXFP8_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP8.
- `W4A4_MXFP4_OMNIQUANT_FAKE` / `W8A8_MXFP8_OMNIQUANT_FAKE`.
- `W4A4_MXFP4_RHT_FAKE` / `W8A8_MXFP8_RHT_FAKE`.
- `W4A4_MXFP4_HADAMARD_LEARNING_FAKE` /
  `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`: AMCT-Q learnable block transform
  followed by fake MX QDQ.
- `W4A4_MXFP4_AUTOROUND_FAKE` / `W8A8_MXFP8_AUTOROUND_FAKE`.

`default_quant_type` applies to modules without an explicit `*.weight` entry.
`module_quant_overrides` is evaluated in JSON insertion order; the first glob
matching the vLLM module prefix wins. An explicit per-weight entry has the
highest priority and can use `FLOAT` to skip a module.

### QDQ execution backend

`fake_mx_backend` controls how every enabled Fake MX node computes the same
quantize-dequantize contract. It is independent of `fake_mx_quant_targets` and
module precision overrides:

- `reference` (default): use the AMCT-compatible PyTorch golden implementation.
- `kernel`: require the optional fused QDQ kernel and fail immediately if it
  cannot execute the request.
- `auto`: use the fused kernel when supported, otherwise log once and fall back
  to `reference`.

```json
{
  "fake_mx_backend": "reference",
  "fake_mx_quant_targets": ["attn-linear"]
}
```

The optional operator adapter is
`vllm_ascend/quantization/kernels/fake_mx.py`. When the external kernel is
delivered, update only `_load_external_fake_mx_kernel()` to import its actual
single-input/single-output QDQ entry point and add finalized capability checks
to `fake_mx_kernel_support_reason()`. Existing Linear, Attention, GDN, and MoE
insertion points continue to call the stable `fake_mx_quantize()` wrapper.

### AMCT-compatible attention targets

`fake_mx_quant_targets` independently controls Qwen3.5 attention injection
boundaries. It defaults to `["attn-linear"]`:

- `attn-linear`: quantize operands and weights of Linear modules selected by
  `module_quant_overrides`; it does not quantize Q/K/V or the GDN core input.
- `attn-cache`: additionally fake-QDQ normalized/RoPE-applied Q/K/V immediately
  before the fused attention/cache boundary. This matches AMCT's Q/K/V operand
  placement, but the fused vLLM attention backend does not expose AMCT's
  post-softmax probability (P) fake-QDQ point.
- `gdn-core`: experimental ablation that fake-QDQs projected `mixed_qkv`
  before the recurrent GDN core. It is disabled by default, is not part of
  AMCT `attn-linear`, and is generally precision-sensitive without an expected
  quantization benefit.
The MX element/shared-exponent math follows AMCT-Q: 32-element blocks by
default, shared exponent carry at mantissa `> 1.75`, minimum E8M0 exponent
`-127`, and half-away-from-zero element rounding. The last tensor dimension
must be divisible by `group_size`, as required by AMCT's `unflatten` contract.

The Qwen direct-conversion samples use W4A4 MXFP4 for attention/GDN
projections, Dense MLP, shared experts, and routed experts. Embeddings, the
visual tower, router gates, and LM head remain floating point. Full-attention
Q/K/V remain floating point unless `attn-cache` is explicitly enabled. The
projected GDN `mixed_qkv` also remains floating point unless the experimental
`gdn-core` target is explicitly enabled.

## FlatQuant parameter contract

Fake FlatQuant currently starts from the original FP16/BF16 model weight. Each
enabled Linear supplies `left_trans`, `right_trans`, and, by default,
`diag_scale` through an external safetensors file. The runtime transforms and
QDQs weight once after loading, then applies:

```text
x -> left_trans.T @ reshape(x) @ right_trans
  -> optional diagonal scaling
  -> fake MX QDQ
  -> floating-point GEMM with online inverse-transformed, QDQ weight
```

Do not feed an already transformed weight checkpoint to this path; that would
apply the transform twice. Routed MoE FlatQuant is outside the current
Qwen3.5-9B delivery scope.

## Algorithm checkpoint contracts

The algorithm examples are not drop-in configs for an untouched pretrained
checkpoint. Each algorithm expects specific calibration artifacts:

- **FlatQuant**: `flatquant_params_path` must point to a safetensors file
  containing `left_trans`, `right_trans`, and `diag_scale` (required when
  `flatquant_use_diag=true`) per enabled layer. The runtime applies the inverse
  transform to the weight, and applies the forward transform to activations.
- **RHT**: No external params needed. The runtime generates random signs from
  `rht_seed` and rotates both weight and activations at load/forward time.
- **Hadamard Learning (LHT)**: `lht_params_path` must point to a safetensors
  file containing `transform_weight` (K×K matrix) per enabled layer. The
  runtime loads the matrix, applies `inv(T).T` to the weight, and applies
  `x @ T` to activations.
- **OmniQuant**: loads `log_scale` and applies matching online activation and
  weight rescaling before QDQ.
- **AutoRound**: loads rounding/clipping parameters and applies them online.
  Both schemes exist in code but are outside the current RHT/LHT/FlatQuant
  acceptance scope.

See the Qwen3.5-9B guide linked at the top for the runtime boundary and exact
insertion points.

## Scope and limitations

- This validates numerical accuracy, perplexity, and task metrics, not MX
  kernel performance or packed-checkpoint memory use.
- Router logits are computed from the original activation. Expert FC1 input,
  expert weights, and the post-activation FC2 input receive fake-MX error.
- Fake-MX routed MoE requires the split dispatch/GMM1/SwiGLU/GMM2/combine
  path. Monolithic FusedMC2 is rejected because it has no GMM1/GMM2
  intermediate QDQ insertion point.
- The implementation uses only ordinary floating-point tensors and arithmetic,
  so it does not require `torch.float4`, `torch.float8`, or E8M0 dtypes.
- `group_size` defaults to 32, matching OCP MX formats.
