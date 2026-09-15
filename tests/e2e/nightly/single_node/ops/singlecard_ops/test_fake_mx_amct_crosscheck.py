# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""AMCT-vs-vLLM single-layer transform cross-checks (evaluation gate 3).

Compares the production fake-MX transforms against the AMCT calibration
source (feat/lht-rht-final) as ground truth:

- RHT: bit-identical Rademacher signs (seed=0) and matching transforms for
  both activations and weights.
- FlatQuant: activation ``L.T @ (x*d) @ R`` and weight
  ``inv(L) @ (W/d) @ inv(R).T`` against AMCT's ``_InvFlatDecomposeTransform``
  using parameters from a real converted sidecar.

RHT uses the same FP32 butterfly path on both sides and must match exactly.
FlatQuant currently has different rounding paths; its tolerance-based
transform checks do NOT certify full QDQ or Linear output parity.

Without an explicit AMCT source root these optional tests skip. Once the
root is supplied, missing prerequisites fail rather than silently skip.
Run with pytest -s to retain source paths, commits and worktree status:

- VLLM_ASCEND_AMCT_SOURCE_ROOT (required for validation)
- VLLM_ASCEND_AMCT_FQ_SIDECAR (required for FlatQuant validation)
"""

import inspect
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.quantization.methods.fake_mx_algorithms import common

AMCT_SOURCE_ROOT = os.environ.get("VLLM_ASCEND_AMCT_SOURCE_ROOT")
FQ_SIDECAR_PATH = os.environ.get("VLLM_ASCEND_AMCT_FQ_SIDECAR")

RHT_MATRIX_SIZE = 128
K_FEATURES = 512


def _amct_source_available() -> bool:
    return bool(AMCT_SOURCE_ROOT) and (Path(AMCT_SOURCE_ROOT) / "amct_pytorch" / "algorithms" / "quant").is_dir()


def _fq_sidecar_available() -> bool:
    return bool(FQ_SIDECAR_PATH) and Path(FQ_SIDECAR_PATH).is_file()


def _npu_available() -> bool:
    # Import torch_npu lazily: on hosts without it (CI runners, laptops) the
    # collection itself must not fail, the tests should skip instead.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    if not torch.npu.is_available():
        return False
    try:
        torch.npu.set_device(0)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not AMCT_SOURCE_ROOT, reason="set VLLM_ASCEND_AMCT_SOURCE_ROOT to run validation"),
]


@pytest.fixture(scope="module", autouse=True)
def validation_environment():
    assert _amct_source_available(), f"AMCT source tree not found at {AMCT_SOURCE_ROOT}"
    assert _npu_available(), "Configured validation requires a real NPU device"
    from vllm_ascend.quantization import fake_mx

    vllm_root = Path(__file__).resolve().parents[6]
    imported_source = Path(fake_mx.__file__).resolve()
    assert imported_source.is_relative_to(vllm_root), f"Wrong vLLM-Ascend import: {imported_source}"
    print(f"vLLM-Ascend fake_mx: {imported_source}")
    for root in (Path(AMCT_SOURCE_ROOT).resolve(), vllm_root):
        print(f"Validation source: {root}")
        for args in (("rev-parse", "HEAD"), ("branch", "--show-current"), ("status", "--short")):
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
            print(f"git {' '.join(args)}: {result.stdout.strip()}")


def _load_amct_symbols():
    root = str(Path(AMCT_SOURCE_ROOT).resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    from amct_pytorch.algorithms.quant.flatquant import _InvFlatDecomposeTransform
    from amct_pytorch.algorithms.quant.random_hadamard import RandomHadamard

    for symbol in (RandomHadamard, _InvFlatDecomposeTransform):
        source = Path(inspect.getfile(symbol)).resolve()
        assert source.is_relative_to(Path(root)), f"Wrong AMCT import: {source}; expected source under {root}"
        print(f"AMCT {symbol.__name__}: {source}")
    return RandomHadamard, _InvFlatDecomposeTransform


def _load_fq_sidecar():
    from safetensors.torch import load_file

    return load_file(str(FQ_SIDECAR_PATH))


def test_amct_vllm_rht_signs_bit_identical():
    """Same Rademacher signs on both sides (the rht_params_path check in prod)."""
    from vllm_ascend.quantization.methods.fake_mx_algorithms.common import _make_rademacher_signs

    RandomHadamard, _ = _load_amct_symbols()
    amct = RandomHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=RHT_MATRIX_SIZE))
    exported = amct.export_ptq_params()
    assert exported["rht_size"] == RHT_MATRIX_SIZE
    # AMCT exports the Rademacher signs directly (int8 +/-1, see
    # RandomHadamard.export_ptq_params); vLLM must regenerate the identical
    # signs at runtime (seed=0) -- this is exactly the rht_params_path check.
    exported_signs = exported["rht_signs"].detach().cpu().to(torch.int8).reshape(-1)
    assert torch.equal(exported_signs, _make_rademacher_signs(RHT_MATRIX_SIZE, 0).cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [8, 257])
def test_amct_vllm_rht_transform_matches(dtype, rows):
    """Compare AMCT's public structure-transform interface on both sides."""
    from vllm_ascend.quantization.fake_mx import randomized_hadamard_transform
    from vllm_ascend.quantization.methods.fake_mx_algorithms.common import _make_rademacher_signs

    RandomHadamard, _ = _load_amct_symbols()
    amct = RandomHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=RHT_MATRIX_SIZE))
    torch.manual_seed(42)
    K = RHT_MATRIX_SIZE * 4
    x = torch.randn(rows, K, device="npu", dtype=dtype)
    W = torch.randn(64, K, device="npu", dtype=dtype)
    signs = _make_rademacher_signs(RHT_MATRIX_SIZE, 0).to("npu")

    amct_x = amct(x, name="test.linear")
    ours_x = randomized_hadamard_transform(x, signs, RHT_MATRIX_SIZE)
    amct_W = amct(W, inv_t=True, name="test.linear")
    ours_W = randomized_hadamard_transform(W, signs, RHT_MATRIX_SIZE)

    # fp32: agree to rounding of op order (matmul vs FWHT butterfly differ in
    # the last fp32 bits, ~1e-7). bf16: one-ULP rounding-path noise.
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=0.07)
    torch.testing.assert_close(ours_x, amct_x, **tol)
    torch.testing.assert_close(ours_W, amct_W, **tol)

    if dtype == torch.float32:
        torch.testing.assert_close(F.linear(ours_x, ours_W), F.linear(x, W), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_amct_vllm_flatquant_matches(dtype):
    """FlatQuant transforms agree with AMCT using real sidecar parameters."""
    from vllm_ascend.quantization.methods.fake_mx_algorithms.flatquant import (
        transform_flatquant_activation,
        transform_flatquant_weight,
    )

    assert _fq_sidecar_available(), f"Set VLLM_ASCEND_AMCT_FQ_SIDECAR to an existing sidecar: {FQ_SIDECAR_PATH}"
    _, _InvFlatDecomposeTransform = _load_amct_symbols()
    sidecar = _load_fq_sidecar()
    prefix = "model.language_model.layers.0.linear_attn.in_proj_qkvz"
    L = sidecar[f"{prefix}.left_trans"].float()
    R = sidecar[f"{prefix}.right_trans"].float()
    D = sidecar[f"{prefix}.diag_scale"].float()
    left_dim, right_dim = L.shape[0], R.shape[0]

    amct = _InvFlatDecomposeTransform(left_dim, right_dim, add_diag=True)
    amct.linear_left.weight.data.copy_(L)
    amct.linear_right.weight.data.copy_(R)
    amct.diag_scale.data.copy_(D)

    torch.manual_seed(42)
    K = left_dim * right_dim
    x = torch.randn(16, K, device="npu", dtype=dtype)
    W = torch.randn(256, K, device="npu", dtype=dtype)

    amct_act = amct(x)
    ours_act = transform_flatquant_activation(x, L.to(x.device), R.to(x.device), D.to(x.device), left_dim, right_dim)
    amct_w = amct(W, inv_t=True)
    ours_w = transform_flatquant_weight(W, L.to(x.device), R.to(x.device), D.to(x.device), left_dim, right_dim)

    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=0.07)
    torch.testing.assert_close(ours_act, amct_act, **tol)
    torch.testing.assert_close(ours_w.to(dtype), amct_w, **tol)

    if dtype == torch.float32:
        torch.testing.assert_close(F.linear(ours_act, ours_w), F.linear(x, W), rtol=2e-2, atol=2e-2)


def _load_amct_lht_omniquant():
    """Extend the AMCT source loading to the LHT/OmniQuant algorithms."""
    _load_amct_symbols()  # performs the sys.path injection and source checks
    from amct_pytorch.algorithms.quant.learnable_hadamard import LearnableHadamard
    from amct_pytorch.algorithms.quant.omniquant import OmniQuant

    root = Path(AMCT_SOURCE_ROOT).resolve()
    for symbol in (LearnableHadamard, OmniQuant):
        source = Path(inspect.getfile(symbol)).resolve()
        assert source.is_relative_to(root), f"Wrong AMCT import: {source}; expected source under {root}"
        print(f"AMCT {symbol.__name__}: {source}")
    return LearnableHadamard, OmniQuant


def _linear_method_patches(config, sidecar):
    """Patch the vLLM config/artifact lookups for the whole method lifetime.

    The patches must stay active through prepare_weight/transform_activation:
    the artifact loader reaches into get_current_vllm_config(), which has no
    test context here.
    """
    return (
        patch("vllm_ascend.quantization.methods.fake_mx_algorithms.linear._quant_description", return_value=config),
        patch.object(common, "_load_safetensors", return_value=sidecar),
        patch.object(common, "_resolve_model_artifact", side_effect=lambda p: p),
        # Single-process tests have no initialized TP group; OmniQuant reads
        # the TP size in __init__ (FlatQuant tests patch the same symbol).
        patch(
            "vllm_ascend.quantization.methods.fake_mx_algorithms.omniquant.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    )


def _build_linear_method(method_cls, config, sidecar):
    """Instantiate a real vLLM fake-MX Linear method with mocked artifacts."""
    layer = torch.nn.Module()
    layer.prefix = "layer"
    method = method_cls()
    # Parameters live on the device in production; building them on the NPU
    # keeps scale buffers and weight transforms on one device.
    for key, value in method.get_weight(K_FEATURES, 256, torch.float32).items():
        layer.register_parameter(key, torch.nn.Parameter(value.to("npu"), requires_grad=False))
    if getattr(method, "supports_pertensor_layer_type", False):
        for key, value in method.get_pertensor_param(torch.float32).items():
            layer.register_parameter(key, torch.nn.Parameter(value.to("npu"), requires_grad=False))
    return method, layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_amct_vllm_lht_matches(dtype):
    """AMCT's orthogonal Cayley Q vs the vLLM paired LHT transforms.

    vLLM uses the general inverse formula W @ inv(T).T (solve-based); for the
    orthogonal Q that AMCT exports this must reduce to AMCT's W @ Q.
    """
    LearnableHadamard, _ = _load_amct_lht_omniquant()
    from vllm_ascend.quantization.methods.fake_mx import LHTLinearMethod

    amct = LearnableHadamard(SimpleNamespace(), SimpleNamespace(matrix_size=RHT_MATRIX_SIZE))
    Q = amct.export_ptq_params()["transform_weight"].float()
    assert tuple(Q.shape) == (RHT_MATRIX_SIZE, RHT_MATRIX_SIZE)

    config = dict(group_size=32, hadamard_learning_matrix_size=RHT_MATRIX_SIZE, lht_params_path="unused")

    torch.manual_seed(42)
    x = torch.randn(16, K_FEATURES, device="npu", dtype=dtype)
    W = torch.randn(256, K_FEATURES, device="npu", dtype=dtype)

    amct_act = amct(x)
    amct_w = amct(W)
    with ExitStack() as stack:
        for cm in _linear_method_patches(config, {"layer.transform_weight": Q}):
            stack.enter_context(cm)
        method, layer = _build_linear_method(LHTLinearMethod, config, {"layer.transform_weight": Q})
        layer.weight.data.copy_(W)
        method.prepare_weight(layer)
        ours_act = method.transform_activation(layer, x)
        ours_w = layer.weight.data

    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=0.07)
    torch.testing.assert_close(ours_act, amct_act, **tol)
    torch.testing.assert_close(ours_w.to(device=amct_w.device, dtype=dtype), amct_w.to(dtype), **tol)
    if dtype == torch.float32:
        # Paired transforms must reconstruct the original linear output.
        torch.testing.assert_close(F.linear(ours_act, ours_w.to(ours_act.device)), F.linear(x, W), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_amct_vllm_omniquant_matches(dtype):
    """AMCT's paired log-scale rescaling vs the vLLM OmniQuant method path."""
    _, OmniQuant = _load_amct_lht_omniquant()
    from vllm_ascend.quantization.methods.fake_mx import OmniQuantLinearMethod

    torch.manual_seed(7)
    log_scale = torch.randn(1, K_FEATURES) * 0.5

    config = dict(group_size=32, omniquant_params_path="unused")

    amct = OmniQuant(SimpleNamespace(), SimpleNamespace(dim_size=K_FEATURES))
    with torch.no_grad():
        amct.log_scale.copy_(log_scale)

    torch.manual_seed(42)
    x = torch.randn(16, K_FEATURES, device="npu", dtype=dtype)
    W = torch.randn(256, K_FEATURES, device="npu", dtype=dtype)

    amct_act = amct(x)
    amct_w = amct(W, inv_t=True)
    with ExitStack() as stack:
        for cm in _linear_method_patches(config, {"layer.log_scale": log_scale}):
            stack.enter_context(cm)
        method, layer = _build_linear_method(OmniQuantLinearMethod, config, {"layer.log_scale": log_scale})
        layer.weight.data.copy_(W)
        method.prepare_weight(layer)
        ours_act = method.transform_activation(layer, x)
        ours_w = layer.weight.data

    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=0.07)
    torch.testing.assert_close(ours_act, amct_act, **tol)
    torch.testing.assert_close(ours_w.to(device=amct_w.device, dtype=dtype), amct_w.to(dtype), **tol)
    if dtype == torch.float32:
        torch.testing.assert_close(F.linear(ours_act, ours_w.to(ours_act.device)), F.linear(x, W), rtol=2e-2, atol=2e-2)
