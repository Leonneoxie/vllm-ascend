# Fake MXFP4/MXFP8 validation

This path emulates MX quantization error on devices without native MXFP4 or
MXFP8 support. It never creates packed MX tensors and never invokes native MX
operators:

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

The Qwen direct-conversion samples use W4A4 MXFP4 for attention/GDN
projections, Dense MLP, shared experts, and routed experts. Embeddings, the
visual tower, router gates, and LM head remain floating point. Full-attention
Q/K/V and the projected GDN Q/K/V input are also QDQ'd before their internal
attention operators.

## FlatQuant checkpoint contract

Fake FlatQuant is a linear-only validation path. Each enabled linear layer must
provide floating-point, FlatQuant-transformed `weight` plus `left_trans`,
`right_trans`, and `clip_ratio` tensors from calibration. The runtime applies:

```text
x -> left_trans @ reshape(x) @ right_trans
  -> block clipping
  -> fake MX QDQ
  -> floating-point GEMM with the fake-MX-QDQ transformed weight
```

Do not select a FlatQuant fake scheme for a plain pretrained checkpoint that
does not contain these transform parameters. Routed MoE FlatQuant is not
implemented because the current Ascend FlatQuant contract is linear-only.

## Algorithm checkpoint contracts

The algorithm examples are not drop-in configs for an untouched pretrained
checkpoint. Their `fake_mx_weight_state` value describes what the offline
calibration/export step must write:

- `flatquant_transformed_fp`: transformed FP weight plus `left_trans`,
  `right_trans`, and `clip_ratio`.
- `rht_rotated_fp`: FP weight whose input dimension was rotated using the same
  `rht_seed` and `rht_group_size` as the runtime.
- `hadamard_learning_transformed_fp`: FP weight transformed offline with
  `inv(transform_weight).T`; runtime loads AMCT-Q's `transform_weight`, applies
  `x @ transform_weight` blockwise, then injects fake-MX error.
- `prequantized_qdq`: FP16/BF16 weight that already contains the final
  OmniQuant LWC/LET or AutoRound MX QDQ error. The runtime deliberately skips a
  second weight QDQ.

The runtime rejects an algorithm scheme when this marker is missing or does
not match. See
`docs/source/developer_guide/fake_mx_algorithm_adaptation_v023.md` for the
offline/runtime boundary and exact insertion points.

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

See
`docs/source/developer_guide/qwen3_5_a4w4_mxfp_hadamard_flatquant_validation_v023.md`
for Qwen3.5 attention, prefill/decode, Hadamard, FlatQuant, and MoE details.
Hadamard Learning 的训练语义、导出映射和逐 expert 插入点见
`docs/source/developer_guide/qwen3_5_hadamard_learning_fake_mx_v023.md`。
