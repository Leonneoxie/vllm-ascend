"""Core algorithm contract tests; also runnable with the CPU isolation harness."""

from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.quantization.fake_mx import fake_mx_quantize, randomized_hadamard_transform
from vllm_ascend.quantization.methods.fake_mx import (
    AscendW4A4MXFP4FakeFlatQuantLinearMethod,
    AscendW4A4MXFP4FakeFusedMoEMethod,
    AscendW4A4MXFP4FakeLinearMethod,
    AscendW4A4MXFP4HadamardLearningFakeLinearMethod,
    AscendW4A4MXFP4OmniQuantFakeLinearMethod,
    AscendW4A4MXFP4RHTFakeLinearMethod,
)
from vllm_ascend.quantization.methods.fake_mx_algorithms import common, flatquant, linear, moe, omniquant

CASES = (
    ("rtn", AscendW4A4MXFP4FakeLinearMethod),
    ("rht", AscendW4A4MXFP4RHTFakeLinearMethod),
    ("lht", AscendW4A4MXFP4HadamardLearningFakeLinearMethod),
    ("omniquant", AscendW4A4MXFP4OmniQuantFakeLinearMethod),
    ("flatquant", AscendW4A4MXFP4FakeFlatQuantLinearMethod),
)


@pytest.mark.parametrize("name,cls", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_core_weight_and_activation_contract(name, cls, dtype):
    config = dict(
        group_size=4,
        rht_matrix_size=4,
        rht_seed=0,
        hadamard_learning_matrix_size=4,
        flatquant_matrix_size=2,
        lht_params_path="unused",
        omniquant_params_path="unused",
        flatquant_params_path="unused",
    )
    # Hand-written orthogonal 4x4 matrix (H4 / 2) used as the LHT transform.
    q = torch.tensor(
        [
            [0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5, 0.5],
        ]
    )
    params = {
        "layer.transform_weight": q,
        "layer.log_scale": torch.tensor([[0.1, -0.2, 0.3, -0.4]]),
        "layer.left_trans": torch.tensor([[1.0, 0.2], [0.0, 1.0]]),
        "layer.right_trans": torch.tensor([[1.0, 0.0], [0.1, 1.0]]),
        "layer.diag_scale": torch.tensor([1.0, 1.1, 0.9, 1.2]),
    }
    with (
        patch.object(common, "_quant_description", return_value=config),
        patch("vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description", return_value=config),
        patch.object(common, "_load_safetensors", return_value=params),
        patch.object(common, "_resolve_model_artifact", side_effect=lambda p: p),
        patch.object(flatquant, "get_tensor_model_parallel_world_size", return_value=1),
    ):
        method = cls()
        layer = torch.nn.Module()
        layer.prefix = "layer"
        for key, value in method.get_weight(4, 4, dtype).items():
            layer.register_parameter(key, torch.nn.Parameter(value, requires_grad=False))
        if getattr(method, "supports_pertensor_layer_type", False):
            for key, value in method.get_pertensor_param(dtype).items():
                layer.register_parameter(key, torch.nn.Parameter(value, requires_grad=False))
        w = torch.arange(-8, 8).reshape(4, 4).to(dtype) / 5
        x = torch.tensor([[0.5, -1.0, 2.0, 3.0]], dtype=dtype)
        layer.weight.data.copy_(w)
        if name == "rht":
            generator = torch.Generator(device="cpu").manual_seed(config["rht_seed"])
            signs = torch.randint(0, 2, (4,), generator=generator, dtype=torch.int8) * 2 - 1
            expected_w, expected_x = (
                randomized_hadamard_transform(w, signs, 4),
                randomized_hadamard_transform(x, signs, 4),
            )
        elif name == "lht":
            expected_w = (w.float() @ q).to(dtype)
            expected_x = (x.float() @ q).to(dtype)
        elif name == "omniquant":
            scale = params["layer.log_scale"].to(dtype).float().exp().clamp(1e-4, 1e4)
            expected_w = (w.float() * scale).to(dtype)
            expected_x = (x.float() / scale).to(dtype)
        elif name == "flatquant":
            left, right, diag = (params["layer." + k] for k in ("left_trans", "right_trans", "diag_scale"))
            expected_w = (
                (
                    torch.linalg.inv(left)
                    @ (w.float().reshape(-1, 2, 2) / diag.reshape(2, 2))
                    @ torch.linalg.inv(right).T
                )
                .reshape(4, 4)
                .to(dtype)
            )
            expected_x = (
                left.to(dtype).T @ (x.reshape(-1, 2, 2) * diag.to(dtype).reshape(2, 2)) @ right.to(dtype)
            ).reshape_as(x)
        else:
            expected_w, expected_x = w, x
        method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight, fake_mx_quantize(expected_w, "mxfp4", 4))
        expected = torch.nn.functional.linear(fake_mx_quantize(expected_x, "mxfp4", 4), layer.weight)
        torch.testing.assert_close(method.apply(layer, x), expected)
        if name == "flatquant":
            layer.aclnn_clip_ratio = 0.5
            clipped = fake_mx_quantize(expected_x, "mxfp4", 4, clip_ratio=0.5)
            torch.testing.assert_close(method.apply(layer, x), torch.nn.functional.linear(clipped, layer.weight))
        saved = {k: v.clone() for k, v in layer.state_dict().items()}
        method.process_weights_after_loading(layer)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_missing_or_wrong_shape_parameter_fails():
    layer = torch.nn.Module()
    layer.prefix = "layer"
    with pytest.raises(KeyError):
        common._copy_transform_param(torch.empty(4), {}, layer, "missing")
    with pytest.raises(ValueError, match="shape"):
        common._copy_transform_param(torch.empty(4), {"layer.x": torch.ones(2)}, layer, "x")


def test_lht_moe_requires_params_path():
    with (
        patch.object(moe, "_quant_description", return_value={"group_size": 4}),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
        pytest.raises(ValueError, match="lht_params_path"),
    ):
        moe.LHTMoEMethod()


def test_lht_moe_ignores_weight_state_marker():
    """New LHT MoE contract: transforms at load time, no pre-transformed checkpoint required."""
    config = {
        "group_size": 4,
        "hadamard_learning_matrix_size": 4,
        "lht_params_path": "dummy.safetensors",
        "fake_mx_weight_state": "hadamard_learning_transformed_fp",  # unrelated marker
    }
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.LHTMoEMethod()
    assert method.required_weight_state is None


def test_lht_moe_parameters_and_deferred_activation_qdq():
    config = {
        "group_size": 4,
        "hadamard_learning_matrix_size": 4,
        "lht_params_path": "dummy.safetensors",
    }
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.LHTMoEMethod()
        baseline = moe.FakeMXMoEMethod()
    method.mx_format = baseline.mx_format = "mxfp4"
    layer = torch.nn.Module()
    weights = method.get_weight(2, 4, 8, torch.float32)
    assert weights["w13_transform_weight"].shape == weights["w2_transform_weight"].shape == (2, 4, 4)
    # Identity initialization: experts missing from the sidecar fall back to
    # a mathematically paired no-op transform.
    torch.testing.assert_close(weights["w13_transform_weight"], torch.eye(4).repeat(2, 1, 1))
    for key, value in weights.items():
        init = value if "transform_weight" in key else torch.ones_like(value)
        layer.register_parameter(key, torch.nn.Parameter(init.clone(), requires_grad=False))
    assert set(baseline.get_weight(2, 4, 8, torch.float32)) == {"w13_weight", "w2_weight"}
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    x = torch.tensor([[1.1, 2.2, 3.3, 4.4]])
    assert method.quantize_input(x) is x
    torch.testing.assert_close(baseline.quantize_input(x), fake_mx_quantize(x, "mxfp4", 4))
    identity_sidecar = {
        f"layers.0.experts.{e}.{fc}.transform_weight": torch.eye(4)
        for e in range(2)
        for fc in ("w13", "w2")
    }
    with (
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "_load_transform_params", return_value=identity_sidecar),
    ):
        method.process_weights_after_loading(layer)
        saved = {key: value.clone() for key, value in layer.state_dict().items()}
        method.process_weights_after_loading(layer)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    assert layer.w13_weight.shape == (2, 8, 8)
    assert layer.w2_weight.shape == (2, 4, 8)


def test_lht_weight_formula_uses_inv_t_transpose():
    """transform_lht_weight must compute W @ inv(T).T, not W @ T."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transform = torch.tensor([[2.0, 0.5], [0.25, 1.5]])  # non-orthogonal

    result = moe.transform_lht_weight(weight, transform, 2)

    inv_t_t = common._inverse_fp32(transform, transpose=True)
    expected = weight.to(torch.float32).reshape(-1, 2) @ inv_t_t
    expected = expected.reshape(weight.shape)
    torch.testing.assert_close(result, expected)

    # Verify it differs from the old orthogonal-assumption formula W @ T.
    old = weight.to(torch.float32).reshape(-1, 2) @ transform.to(torch.float32)
    old = old.reshape(weight.shape)
    assert not torch.allclose(result, old), "Formula should differ from W@T for non-orthogonal T"


def test_lht_moe_transforms_weights_per_expert_at_load_time():
    config = {
        "group_size": 2,
        "hadamard_learning_matrix_size": 2,
        "lht_params_path": "dummy.safetensors",
    }
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.LHTMoEMethod()
    method.mx_format = "mxfp4"

    transforms = [
        torch.tensor([[2.0, 0.5], [0.25, 1.5]]),
        torch.tensor([[1.5, 0.25], [0.5, 2.0]]),
    ]
    sidecar = {
        f"layers.0.experts.{e}.{fc}.transform_weight": transforms[e]
        for e in range(2)
        for fc in ("w13", "w2")
    }
    layer = torch.nn.Module()
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    layer.w13_weight = torch.nn.Parameter(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32),
        requires_grad=False,
    )
    layer.w13_transform_weight = torch.nn.Parameter(torch.eye(2).repeat(2, 1, 1), requires_grad=False)
    layer.w2_transform_weight = torch.nn.Parameter(torch.eye(2).repeat(2, 1, 1), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    with (
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size: weight),
    ):
        method.process_weights_after_loading(layer)

    for e in range(2):
        expected_w13_e = moe.transform_lht_weight(original_w13[e], transforms[e], 2).transpose(0, 1).contiguous()
        expected_w2_e = moe.transform_lht_weight(original_w2[e], transforms[e], 2).transpose(0, 1).contiguous()
        torch.testing.assert_close(layer.w13_weight.data[e], expected_w13_e)
        torch.testing.assert_close(layer.w2_weight.data[e], expected_w2_e)


def test_rht_moe_rotates_weights_at_load_time():
    """RHT MoE rotates w13/w2 with the same deterministic signs as the FC1 activation."""
    config = {"group_size": 4, "rht_group_size": 4, "rht_seed": 0}
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.RHTMoEMethod()
    method.mx_format = "mxfp4"

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    with (
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size: weight),
    ):
        method.process_weights_after_loading(layer)

    signs = common._make_rademacher_signs(4, 0)
    expected_w13 = randomized_hadamard_transform(original_w13, signs, 4).transpose(1, 2).contiguous()
    expected_w2 = randomized_hadamard_transform(original_w2, signs, 4).transpose(1, 2).contiguous()
    torch.testing.assert_close(layer.w13_weight, expected_w13)
    torch.testing.assert_close(layer.w2_weight, expected_w2)
    # FC2 transform slot is populated (FC1 runs pre-dispatch in quantize_input).
    assert callable(layer._fake_mx_fc2_transform)
    assert getattr(layer, "_fake_mx_fc1_transform", None) is None


def test_rht_moe_weight_and_activation_are_mathematically_paired():
    """(x@R) @ (W@R).T == x @ W.T for the shared signs R (R is orthogonal)."""
    weight = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.25, -2.0, 1.0, 3.0]])
    activation = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    signs = common._make_rademacher_signs(4, 0)

    transformed_weight = randomized_hadamard_transform(weight, signs, 4)
    transformed_activation = randomized_hadamard_transform(activation, signs, 4)

    torch.testing.assert_close(
        torch.nn.functional.linear(transformed_activation, transformed_weight),
        torch.nn.functional.linear(activation, weight),
    )


def test_omniquant_moe_requires_params_path():
    with (
        patch.object(moe, "_quant_description", return_value={"group_size": 4}),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
        pytest.raises(ValueError, match="omniquant_params_path"),
    ):
        moe.OmniQuantMoEMethod()


def test_omniquant_moe_scales_weights_at_load_time():
    """weight' = weight * exp(log_scale); activation divided by the same scale in moe_mlp."""
    fc1_log_scale = torch.tensor([[0.6931, 0.0, 0.0, 0.0], [0.0, 0.6931, 0.0, 0.0]], dtype=torch.float32)
    fc2_log_scale = torch.zeros(2, 4, dtype=torch.float32)
    config = {"group_size": 4, "omniquant_params_path": "/fake/path.pt"}
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.OmniQuantMoEMethod()
    method.mx_format = "mxfp4"

    layer = torch.nn.Module()
    layer.prefix = "test"
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer.w13_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    layer.w2_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()

    sidecar = {"test.w13_log_scale": fc1_log_scale, "test.w2_log_scale": fc2_log_scale}
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size: weight),
    ):
        method.process_weights_after_loading(layer)

    # weight' = weight * scale  (broadcasting [E, 2*inter, hidden] * [E, 1, hidden]),
    # then the shared entry transposes to the [E, K, N] GMM layout.
    expected_fc1_scale = torch.exp(fc1_log_scale).clamp(min=1e-4, max=1e4)
    expected_w13 = (original_w13 * expected_fc1_scale.unsqueeze(1)).transpose(1, 2).contiguous()
    torch.testing.assert_close(layer.w13_weight.data, expected_w13)
    # The transform slots close over the per-expert scales; a spy executor
    # verifies both the bound values and the shared calling convention.
    captured = {}

    def spy_executor(x, scale, group_list, group_list_type):
        captured.setdefault("scale", scale)
        return x

    with patch.object(moe, "_apply_expert_omniquant", side_effect=spy_executor):
        layer._fake_mx_fc1_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
        layer._fake_mx_fc2_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
    torch.testing.assert_close(captured["scale"], expected_fc1_scale)
    assert layer._fake_mx_fc1_transform is not layer._fake_mx_fc2_transform


def test_omniquant_moe_maps_global_scales_to_local_experts():
    fc1_log_scale = torch.log(torch.tensor([2.0, 3.0, 5.0, 7.0])).unsqueeze(1).repeat(1, 4)
    fc2_log_scale = torch.log(torch.tensor([11.0, 13.0, 17.0, 19.0])).unsqueeze(1).repeat(1, 4)
    config = {"group_size": 4, "omniquant_params_path": "/fake/path.pt"}
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.OmniQuantMoEMethod()
    method.mx_format = "mxfp4"

    layer = torch.nn.Module()
    layer.prefix = "test"
    layer._expert_map = torch.tensor([1, -1, 0, -1])
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    layer.w13_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    layer.w2_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)

    sidecar = {"test.w13_log_scale": fc1_log_scale, "test.w2_log_scale": fc2_log_scale}
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
    ):
        method.process_weights_after_loading(layer)

    captured = {}

    def spy_executor(x, scale, group_list, group_list_type):
        captured.setdefault("scale", scale)
        return x

    with patch.object(moe, "_apply_expert_omniquant", side_effect=spy_executor):
        layer._fake_mx_fc1_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
    # Global scales [2,3,5,7] mapped through _expert_map [1,-1,0,-1]:
    # local slot 0 <- logical 2 (scale 5), slot 1 <- logical 0 (scale 2).
    torch.testing.assert_close(captured["scale"], torch.tensor([[5.0] * 4, [2.0] * 4]))


def test_omniquant_moe_weight_and_activation_are_mathematically_paired():
    """(x / scale) @ (weight * scale).T == x @ weight.T per expert."""
    weight = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.25, -2.0, 1.0, 3.0]])
    activation = torch.tensor([[0.5, -1.0, 2.0, 1.0]])
    scale = torch.tensor([2.0, 0.5, 1.0, 4.0])

    torch.testing.assert_close(
        torch.nn.functional.linear(activation / scale, weight * scale),
        torch.nn.functional.linear(activation, weight),
    )


def test_flatquant_moe_requires_params_path():
    with (
        patch.object(moe, "_quant_description", return_value={"group_size": 4}),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
        pytest.raises(ValueError, match="flatquant_params_path"),
    ):
        moe.FlatQuantMoEMethod()


def test_flatquant_moe_inverse_transforms_weights_and_packs_state():
    """Identity sidecar => weight unchanged (mod QDQ), fc states packed on the layer."""
    config = {"group_size": 4, "flatquant_params_path": "/fake/path.pt", "flatquant_matrix_size": 4}
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = moe.FlatQuantMoEMethod()
    method.mx_format = "mxfp4"

    layer = torch.nn.Module()
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    for key, value in method.get_weight(2, 4, 4, torch.float32).items():
        layer.register_parameter(key, torch.nn.Parameter(torch.ones_like(value), requires_grad=False))

    # Identity transforms in the sidecar for every expert and component.
    sidecar = {}
    for e in range(2):
        for fc, (left_dim, right_dim) in (("fc1", (1, 4)), ("fc2", (1, 4))):
            sidecar[f"layers.0.experts.{e}.{fc}.left_trans"] = torch.eye(left_dim)
            sidecar[f"layers.0.experts.{e}.{fc}.right_trans"] = torch.eye(right_dim)
            sidecar[f"layers.0.experts.{e}.{fc}.diag"] = torch.ones(4)

    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
    ):
        method.process_weights_after_loading(layer)

    # Identity transform => weight only goes through QDQ + GMM transpose.
    expected_w13 = fake_mx_quantize(torch.ones(2, 8, 4), "mxfp4", 4).transpose(1, 2).contiguous()
    torch.testing.assert_close(layer.w13_weight.data, expected_w13)
    # Transform slots close over the per-expert state; a spy executor verifies
    # the packed components and the shared calling convention.
    captured = {}

    def spy_executor(x, fc_state, group_list, group_list_type):
        captured.setdefault("fc_state", fc_state)
        return x

    with patch.object(moe, "_apply_expert_flatquant", side_effect=spy_executor):
        layer._fake_mx_fc1_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
    assert set(captured["fc_state"]) == {"left_trans", "right_trans", "diag_scale"}


def test_flatquant_moe_inverse_transform_matches_shared_formula():
    """Per-expert inverse transform must match the shared FlatQuant math."""
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    left = torch.tensor([[2.0]])
    right = torch.tensor([[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.2, 0.0], [0.0, 0.0, 1.0, 0.3], [0.1, 0.0, 0.0, 1.0]])
    diag = torch.ones(4)

    activation = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    transformed_weight = flatquant.transform_flatquant_weight(weight, left, right, diag, 1, 4)
    transformed_activation = flatquant.transform_flatquant_activation(activation, left, right, diag, 1, 4)
    # Paired transforms preserve the linear output: A(x) @ W'.T == x @ W.T.
    torch.testing.assert_close(
        torch.nn.functional.linear(transformed_activation, transformed_weight),
        torch.nn.functional.linear(activation, weight),
        rtol=1e-4,
        atol=1e-4,
    )


def test_moe_qdq_layout_and_idempotency():
    with (
        patch.object(moe, "_quant_description", return_value={"group_size": 4}),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        method = AscendW4A4MXFP4FakeFusedMoEMethod()
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    expected = [
        fake_mx_quantize(w, "mxfp4", 4).transpose(1, 2).contiguous() for w in (layer.w13_weight, layer.w2_weight)
    ]
    with patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w):
        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(layer)
    for actual, target in zip((layer.w13_weight, layer.w2_weight), expected):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    with (
        patch.object(moe, "_EXTRA_CTX", Mock(moe_comm_type=moe.MoECommType.FUSED_MC2)),
        pytest.raises(NotImplementedError, match="split dispatch"),
    ):
        method._validate_execution_path()


def test_flatquant_row_parallel_dense_transform_gathers_full_tp_input_and_weight():
    """Dense row-parallel transforms must be applied before TP partitioning."""
    config = {
        "group_size": 1,
        "flatquant_params_path": "/fake/path.pt",
        "flatquant_use_diag": True,
    }
    left = torch.tensor([[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.2, 0.0], [0.0, 0.0, 1.0, 0.3], [0.1, 0.0, 0.0, 1.0]])
    right = torch.ones(1, 1)
    diag = torch.ones(4)
    sidecar = {
        "test.left_trans": left,
        "test.right_trans": right,
        "test.diag_scale": diag,
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(flatquant, "_load_transform_params", return_value=sidecar),
        patch.object(flatquant, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(flatquant, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
        patch.object(linear, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()
        layer = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.prefix = "test"
        layer.tp_size = 2
        layer.tp_rank = 1
        layer.left_trans = torch.nn.Parameter(torch.eye(4), requires_grad=False)
        layer.right_trans = torch.nn.Parameter(torch.eye(1), requires_grad=False)
        layer.diag_scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        layer.clip_ratio = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        layer.weight = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]), requires_grad=False)

        gather_calls = 0

        def gather(value, dim=-1):
            nonlocal gather_calls
            gather_calls += 1
            prefix = torch.tensor([[1.0, 2.0]]) if gather_calls == 1 else torch.tensor([[5.0, 6.0]])
            return torch.cat((prefix, value), dim=dim)

        with patch.object(flatquant, "tensor_model_parallel_all_gather", side_effect=gather):
            method.process_weights_after_loading(layer)
            layer.aclnn_clip_ratio = 1.0
            actual = method.apply(layer, torch.tensor([[7.0, 8.0]]))

    full_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    full_activation = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    expected_weight = flatquant.transform_flatquant_weight(full_weight, left, right, diag, 4, 1)
    expected_activation = flatquant.transform_flatquant_activation(full_activation, left, right, diag, 4, 1)
    expected = expected_activation[:, 2:] @ expected_weight[:, 2:].t()
    torch.testing.assert_close(actual, expected)
    assert layer._fake_mx_flatquant_tp_full_transform is True
    assert layer.left_trans.shape == (4, 4)
    assert gather_calls == 2


def test_omniquant_row_parallel_loads_full_scale_then_selects_tp_rank():
    """Row-parallel layers load the full-width log_scale, then slice per TP rank."""
    full_log_scale = torch.log(torch.tensor([[2.0, 3.0, 5.0, 7.0]]))
    config = {"group_size": 2, "omniquant_params_path": "/fake/path.pt"}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(omniquant, "_load_transform_params", return_value={"test.log_scale": full_log_scale}),
        patch.object(omniquant, "get_tensor_model_parallel_world_size", return_value=2),
    ):
        method = AscendW4A4MXFP4OmniQuantFakeLinearMethod()
        method.get_weight(2, 1, torch.float32)
        # Row layers allocate the full-width placeholder (input_size * tp_size).
        pertensor = method.get_pertensor_param(torch.float32, layer_type="row")
        assert pertensor["log_scale"].shape == (1, 4)

        layer = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.prefix = "test"
        layer.tp_size = 2
        layer.tp_rank = 1
        layer.log_scale = torch.nn.Parameter(pertensor["log_scale"], requires_grad=False)
        layer.weight = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
        method.prepare_weight(layer)

    # Full scale loaded, then this rank's slice applied to the weight.
    torch.testing.assert_close(layer.log_scale.data, full_log_scale[:, 2:])
    torch.testing.assert_close(layer.weight, torch.tensor([[5.0, 7.0]]))


def test_omniquant_non_row_keeps_local_scale_without_slicing():
    config = {"group_size": 2, "omniquant_params_path": "/fake/path.pt"}
    log_scale = torch.log(torch.tensor([[2.0, 3.0]]))
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(omniquant, "_load_transform_params", return_value={"test.log_scale": log_scale}),
        patch.object(omniquant, "get_tensor_model_parallel_world_size", return_value=2),
    ):
        method = AscendW4A4MXFP4OmniQuantFakeLinearMethod()
        method.get_weight(2, 1, torch.float32)
        pertensor = method.get_pertensor_param(torch.float32, layer_type="others")
        assert pertensor["log_scale"].shape == (1, 2)

        layer = torch.nn.Module()
        layer.prefix = "test"
        layer.log_scale = torch.nn.Parameter(pertensor["log_scale"], requires_grad=False)
        layer.weight = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
        method.prepare_weight(layer)

    torch.testing.assert_close(layer.log_scale.data, log_scale)
    torch.testing.assert_close(layer.weight, torch.tensor([[2.0, 3.0]]))


@pytest.mark.parametrize(
    ("params", "target_shape", "expert_map", "expected_error", "message"),
    [
        pytest.param({}, (2, 4), [0, 1], KeyError, "Missing", id="missing-key"),
        pytest.param(
            {"test.w13_log_scale": torch.zeros(2, 3)},
            (2, 4),
            [0, 1],
            ValueError,
            "shape",
            id="feature-shape-mismatch",
        ),
        pytest.param(
            {"test.w13_log_scale": torch.zeros(3, 4)},
            (2, 4),
            [0, 1],
            ValueError,
            "has 3 experts",
            id="expert-count-mismatch",
        ),
        pytest.param(
            {"test.w13_log_scale": torch.zeros(2, 4)},
            (1, 4),
            [0, 1],
            ValueError,
            "selects 2",
            id="local-count-mismatch",
        ),
        pytest.param(
            {"test.w13_log_scale": torch.zeros(2, 4)},
            (2, 4),
            [0, 2],
            ValueError,
            "invalid local slot",
            id="invalid-local-slot",
        ),
    ],
)
def test_omniquant_moe_rejects_invalid_expert_scale_metadata(params, target_shape, expert_map, expected_error, message):
    layer = torch.nn.Module()
    layer.prefix = "test"
    layer._expert_map = torch.tensor(expert_map)
    target = torch.zeros(*target_shape)

    with pytest.raises(expected_error, match=message):
        common._copy_expert_transform_param(target, params, layer, "w13_log_scale")


def _lht_method(config_overrides=None):
    config = {"group_size": 2, "hadamard_learning_matrix_size": 2, "lht_params_path": "dummy.safetensors"}
    config.update(config_overrides or {})
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        return moe.LHTMoEMethod()


def test_lht_moe_rejects_missing_sidecar_key():
    """A missing per-expert transform key must fail, not fall back to identity."""
    method = _lht_method()
    method.mx_format = "mxfp4"
    layer = torch.nn.Module()
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 4, 2), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 2, 2), requires_grad=False)
    layer.w13_transform_weight = torch.nn.Parameter(torch.eye(2).repeat(2, 1, 1), requires_grad=False)
    layer.w2_transform_weight = torch.nn.Parameter(torch.eye(2).repeat(2, 1, 1), requires_grad=False)

    # Sidecar covers expert 0 only; expert 1 is missing.
    partial_sidecar = {
        "layers.0.experts.0.w13.transform_weight": torch.eye(2),
        "layers.0.experts.0.w2.transform_weight": torch.eye(2),
    }
    with (
        patch.object(moe, "_load_transform_params", return_value=partial_sidecar),
        pytest.raises(KeyError, match="experts.1.w13.transform_weight"),
    ):
        method.process_weights_after_loading(layer)


def _flatquant_moe_method(config_overrides=None):
    config = {"group_size": 4, "flatquant_params_path": "/fake/path.pt", "flatquant_matrix_size": 4}
    config.update(config_overrides or {})
    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
    ):
        return moe.FlatQuantMoEMethod()


def _flatquant_layer(method):
    layer = torch.nn.Module()
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
    for key, value in method.get_weight(2, 4, 4, torch.float32).items():
        layer.register_parameter(key, torch.nn.Parameter(torch.ones_like(value), requires_grad=False))
    return layer


def _flatquant_identity_sidecar(with_diag=True, diag_value=None, drop_key=None):
    sidecar = {}
    for e in range(2):
        for fc in ("fc1", "fc2"):
            sidecar[f"layers.0.experts.{e}.{fc}.left_trans"] = torch.eye(1)
            sidecar[f"layers.0.experts.{e}.{fc}.right_trans"] = torch.eye(4)
            if with_diag:
                diag = torch.ones(4) if diag_value is None else diag_value
                sidecar[f"layers.0.experts.{e}.{fc}.diag"] = diag
    if drop_key is not None:
        sidecar.pop(drop_key)
    return sidecar


def test_flatquant_moe_rejects_missing_sidecar_key():
    """A missing per-expert transform key must fail, not fall back to identity."""
    method = _flatquant_moe_method()
    method.mx_format = "mxfp4"
    layer = _flatquant_layer(method)

    sidecar = _flatquant_identity_sidecar(drop_key="layers.0.experts.1.fc2.left_trans")
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        pytest.raises(KeyError, match="experts.1.fc2.left_trans"),
    ):
        method.process_weights_after_loading(layer)


def test_flatquant_moe_disabled_diag_rejects_non_identity_sidecar_diag():
    """flatquant_use_diag=False plus a non-identity sidecar diag must fail:
    honouring the switch would break the weight/activation pairing."""
    method = _flatquant_moe_method({"flatquant_use_diag": False})
    method.mx_format = "mxfp4"
    layer = _flatquant_layer(method)

    sidecar = _flatquant_identity_sidecar(with_diag=True, diag_value=torch.tensor([2.0, 1.0, 1.0, 1.0]))
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        pytest.raises(ValueError, match="non-identity diag_scale"),
    ):
        method.process_weights_after_loading(layer)


def test_flatquant_moe_disabled_diag_skips_diag_on_both_sides():
    """use_diag=False + no sidecar diag: weight transform and packed activation
    state must both skip diag (mirroring the Linear contract)."""
    method = _flatquant_moe_method({"flatquant_use_diag": False})
    method.mx_format = "mxfp4"
    layer = _flatquant_layer(method)

    sidecar = _flatquant_identity_sidecar(with_diag=False)
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size: weight),
    ):
        method.process_weights_after_loading(layer)

    # Identity left/right and no diag division => weight only transposed.
    torch.testing.assert_close(layer.w13_weight.data, torch.ones(2, 8, 4).transpose(1, 2).contiguous())
    # Packed runtime state omits diag entirely on both FCs.
    captured = {}

    def spy_executor(x, fc_state, group_list, group_list_type):
        captured.setdefault("fc_state", fc_state)
        return x

    with patch.object(moe, "_apply_expert_flatquant", side_effect=spy_executor):
        layer._fake_mx_fc1_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
        layer._fake_mx_fc2_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
    assert set(captured["fc_state"]) == {"left_trans", "right_trans"}


def test_flatquant_moe_disabled_diag_accepts_identity_sidecar_diag():
    """use_diag=False plus an all-ones sidecar diag is a no-op, not a conflict."""
    method = _flatquant_moe_method({"flatquant_use_diag": False})
    method.mx_format = "mxfp4"
    layer = _flatquant_layer(method)

    sidecar = _flatquant_identity_sidecar(with_diag=True)  # all-ones diag
    with (
        patch.object(moe, "_load_transform_params", return_value=sidecar),
        patch.object(moe, "maybe_trans_nz", side_effect=lambda w: w),
        patch.object(moe, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size: weight),
    ):
        method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight.data, torch.ones(2, 8, 4).transpose(1, 2).contiguous())
    captured = {}

    def spy_executor(x, fc_state, group_list, group_list_type):
        captured.setdefault("fc_state", fc_state)
        return x

    with patch.object(moe, "_apply_expert_flatquant", side_effect=spy_executor):
        layer._fake_mx_fc1_transform(torch.ones(1, 4), torch.tensor([1, 1]), 1)
    assert set(captured["fc_state"]) == {"left_trans", "right_trans"}


def test_fake_mx_moe_apply_passes_generic_transform_slots():
    """The shared apply() only forwards the generic transform slots.

    A hypothetical new algorithm that populates ``_fake_mx_fc1_transform`` /
    ``_fake_mx_fc2_transform`` flows through the common execution path with
    no algorithm-specific field anywhere in build_fused_experts_input's
    kwargs - adding an algorithm must not touch the shared chain.
    """
    config = {"group_size": 4}
    output = torch.ones(2, 4)
    captured_kwargs = {}

    def fake_build_fused_experts_input(**kwargs):
        captured_kwargs.update(kwargs)
        return "fused-input"

    with (
        patch.object(moe, "_quant_description", return_value=config),
        patch.object(moe, "get_ascend_config", return_value=Mock(eplb_config=Mock(dynamic_eplb=False))),
        patch.object(moe, "get_moe_num_logical_experts", return_value=2),
        patch.object(moe, "select_experts", return_value=(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64))),
        patch.object(
            moe,
            "_EXTRA_CTX",
            Mock(
                moe_comm_method=Mock(fused_experts=Mock(return_value=output)),
                moe_comm_type=object(),  # any non-FUSED_MC2 value
            ),
        ),
        patch.object(moe, "build_fused_experts_input", side_effect=fake_build_fused_experts_input),
    ):
        method = moe.FakeMXMoEMethod()
        method.mx_format = "mxfp4"
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
        # A brand-new algorithm: only the generic slots are populated.
        layer._fake_mx_fc1_transform = lambda x, gl, glt: x * 2.0
        layer._fake_mx_fc2_transform = lambda x, gl, glt: x * 3.0
        actual = method.apply(layer, torch.ones(1, 4), torch.ones(1, 2), top_k=2, renormalize=True)

    assert actual is output
    assert captured_kwargs["fake_mx_format"] == "mxfp4"
    assert captured_kwargs["fake_mx_group_size"] == 4
    assert captured_kwargs["fake_mx_fc1_transform"] is layer._fake_mx_fc1_transform
    assert captured_kwargs["fake_mx_fc2_transform"] is layer._fake_mx_fc2_transform
    # No algorithm-specific leakage into the shared parameter contract.
    forbidden = [k for k in captured_kwargs if k.startswith("fake_mx_") and k not in (
        "fake_mx_format", "fake_mx_group_size", "fake_mx_fc1_transform", "fake_mx_fc2_transform",
    )]
    assert not forbidden, f"algorithm-specific fields leaked into the shared chain: {forbidden}"


def _flatquant_linear_layer():
    layer = torch.nn.Module()
    layer.prefix = "test"
    layer.left_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    layer.right_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    layer.diag_scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    layer.clip_ratio = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.25, -2.0, 1.0, 3.0]]), requires_grad=False
    )
    return layer


def test_flatquant_linear_disabled_diag_rejects_non_identity_sidecar_diag():
    """use_diag=False plus a non-identity sidecar diag must fail (Linear, mirroring MoE)."""
    config = {
        "group_size": 4,
        "flatquant_params_path": "/fake/path.pt",
        "flatquant_use_diag": False,
        "flatquant_matrix_size": 2,
    }
    sidecar = {
        "test.left_trans": torch.eye(2),
        "test.right_trans": torch.eye(2),
        "test.diag_scale": torch.tensor([2.0, 1.0, 1.0, 1.0]),
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(flatquant, "_load_transform_params", return_value=sidecar),
        patch.object(flatquant, "get_tensor_model_parallel_world_size", return_value=1),
        pytest.raises(ValueError, match="non-identity"),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()
        layer = _flatquant_linear_layer()
        method.prepare_weight(layer)


def test_flatquant_linear_disabled_diag_skips_diag_on_both_sides():
    """use_diag=False + no sidecar diag: weight and activation both skip diag
    while left/right keep the transform paired."""
    config = {
        "group_size": 4,
        "flatquant_params_path": "/fake/path.pt",
        "flatquant_use_diag": False,
        "flatquant_matrix_size": 2,
    }
    left = torch.tensor([[2.0, 0.5], [0.25, 1.5]])
    right = torch.tensor([[1.0, 0.2], [0.0, 1.0]])
    sidecar = {"test.left_trans": left, "test.right_trans": right}
    layer = _flatquant_linear_layer()
    original_weight = layer.weight.detach().clone()
    activation = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(flatquant, "_load_transform_params", return_value=sidecar),
        patch.object(flatquant, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(flatquant, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
        patch.object(linear, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()
        layer = _flatquant_linear_layer()
        method.process_weights_after_loading(layer)
        layer.aclnn_clip_ratio = 1.0
        actual = method.apply(layer, activation)

    # The diag placeholder stays untouched (never loaded, never applied) and
    # the left/right pairing preserves the linear output exactly.
    torch.testing.assert_close(layer.diag_scale.data, torch.ones(4))
    torch.testing.assert_close(actual, torch.nn.functional.linear(activation, original_weight), rtol=1e-4, atol=1e-4)


def test_flatquant_linear_disabled_diag_accepts_identity_sidecar_diag():
    """use_diag=False plus an all-ones sidecar diag is a no-op, not a conflict."""
    config = {
        "group_size": 4,
        "flatquant_params_path": "/fake/path.pt",
        "flatquant_use_diag": False,
        "flatquant_matrix_size": 2,
    }
    sidecar = {
        "test.left_trans": torch.eye(2),
        "test.right_trans": torch.eye(2),
        "test.diag_scale": torch.ones(4),
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description",
            return_value=config,
        ),
        patch.object(flatquant, "_load_transform_params", return_value=sidecar),
        patch.object(flatquant, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(flatquant, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
        patch.object(linear, "fake_mx_quantize", side_effect=lambda weight, _format, _group_size, **_: weight),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()
        layer = _flatquant_linear_layer()
        method.process_weights_after_loading(layer)

    # All-ones diag never reaches the weight transform; identity left/right
    # keep the weight unchanged apart from layout.
    torch.testing.assert_close(layer.weight.data, _flatquant_linear_layer().weight.data)
    torch.testing.assert_close(layer.diag_scale.data, torch.ones(4))
