# Fake MXFP4/MXFP8 validation

This path emulates MX quantization error on devices without native MXFP4 or
MXFP8 support. It never creates packed MX tensors. The QDQ math runs on
ordinary floating-point operators (the single reference path; the
fused-kernel adapter seam was removed in the fake_mx_algorithms refactor):

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

Supported scheme names (five validated algorithms, Linear):

- `W4A4_MXFP4_FAKE`: fake MXFP4 weights and activations.
- `W8A8_MXFP8_FAKE`: fake MXFP8 weights and activations.
- `W4A4_MXFP4_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP4.
- `W8A8_MXFP8_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP8.
- `W4A4_MXFP4_OMNIQUANT_FAKE` / `W8A8_MXFP8_OMNIQUANT_FAKE`.
- `W4A4_MXFP4_RHT_FAKE` / `W8A8_MXFP8_RHT_FAKE`.
- `W4A4_MXFP4_HADAMARD_LEARNING_FAKE` /
  `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`: AMCT-Q learnable block transform
  followed by fake MX QDQ.

MoE (routed experts) supports all five algorithms with the same scheme names
(RTN/RHT/LHT/OmniQuant/FlatQuant); per-expert transforms are loaded from the
algorithm sidecar (`--target moe` in the converter). Shared experts keep the
Linear path.

`default_quant_type` applies to modules without an explicit `*.weight` entry.
`module_quant_overrides` is evaluated in JSON insertion order; the first glob
matching the vLLM module prefix wins. An explicit per-weight entry has the
highest priority and can use `FLOAT` to skip a module.

The MX element/shared-exponent math follows AMCT-Q: 32-element blocks by
default, shared exponent carry at mantissa `> 1.75`, minimum E8M0 exponent
`-127`, and half-away-from-zero element rounding. The last tensor dimension
must be divisible by `group_size`, as required by AMCT's `unflatten` contract.
The reference implementation in `vllm_ascend/quantization/fake_mx.py` is the
only QDQ path; the optional fused-kernel adapter seam has been removed.

## Adding a new algorithm

The implementation lives in `vllm_ascend/quantization/methods/fake_mx_algorithms/`
(`linear.py` holds the shared Linear entry, `moe.py` the MoE adapters with the
`_fake_mx_fc1_transform` / `_fake_mx_fc2_transform` post-dispatch slots):

1. Add a module under `fake_mx_algorithms` inheriting `FakeMXLinearMethod`.
2. Read the algorithm config once in `__init__` from `self.config`; declare
   loadable parameters via `get_weight` / `get_pertensor_param` when needed.
3. `prepare_weight(layer)` loads parameters and transforms the weight; it must
   NOT apply weight QDQ or maintain a processed flag — the shared entry point
   handles both.
4. `transform_activation(layer, x)` returns the transformed activation; the
   default `quantize_activation` applies the shared QDQ and `apply` only calls
   it plus `F.linear`. Override `quantize_activation` only for
   algorithm-specific clipping (e.g. FlatQuant); the shared executor never
   reads algorithm-private fields.
5. Register the two format classes in `methods/fake_mx.py`, export them in
   `methods/__init__.py`, and add the scheme strings to the
   `FAKE_MX_QUANT_TYPES` whitelist in `modelslim_config.py`. No model or
   executor changes are required.
6. Add regression tests (missing/shape parameter checks, paired transforms,
   weight idempotency, FP32/BF16) plus an NPU smoke run before dataset
   evaluation.

Algorithm modules reuse the vLLM scheme lifecycle; they are not standalone
training frameworks. Math functions are unit-testable on their own.

## FlatQuant checkpoint contract

Fake FlatQuant is a validation path for Linear layers and routed MoE experts.
The checkpoint always supplies
the **original** BF16/FP16 `weight`; the calibration artifact
(`flatquant_params_path`, safetensors) provides `left_trans` [L, L],
`right_trans` [R, R] and optional `diag_scale` [L*R] per enabled layer
(`L * R == in_features`). At load time the runtime applies the inverse weight
transform `W' = inv(left) @ (W / diag) @ inv(right).T`, and on every forward
the activation transform `x' = left.T @ (reshape(x) * diag) @ right` followed
by block clipping and fake MX QDQ:

```text
x -> left_trans.T @ reshape(x) @ right_trans
  -> block clipping
  -> fake MX QDQ
  -> floating-point GEMM with the inverse-transformed weight
```

Do not select a FlatQuant fake scheme for a plain pretrained checkpoint that
does not have the transform artifact.

## Algorithm checkpoint contracts

The algorithm examples are not drop-in configs for an untouched pretrained
checkpoint. Each algorithm expects specific calibration artifacts:

- **FlatQuant**: `flatquant_params_path` must point to a safetensors file
  containing `left_trans`, `right_trans`, and optional `diag_scale` per
  enabled layer. The runtime applies the inverse transform to the weight and
  the forward transform to activations.
- **RHT**: No external params needed. The runtime deterministically generates
  Rademacher signs with `rht_seed` (default 0, matching AMCT's
  `_HadamardTransform`) and applies the same normalized FWHT
  (`randomized_hadamard_transform`) to both weight (at load time) and
  activations (every forward). `rht_matrix_size` defaults to 128.
- **Hadamard Learning (LHT)**: `lht_params_path` must point to a safetensors
  file containing `transform_weight` (K×K matrix) per enabled layer. The
  runtime loads the matrix, applies the paired `W @ inv(T).T` block transform
  to the weight (invertible, not restricted to orthogonal matrices), and
  applies `x @ T` to activations. MoE LHT loads per-expert
  `w13/w2_transform_weight` from the same sidecar (`--target moe`).
- **OmniQuant**: `omniquant_params_path` must point to a safetensors file
  containing per-dimension `log_scale`. The runtime scales the weight up by
  `exp(log_scale)` (clamped to [1e-4, 1e4]) and the activation down, pairing
  with MX QDQ.

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
