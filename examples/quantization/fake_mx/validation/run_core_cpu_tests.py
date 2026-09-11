"""Run real core tensor/registry tests with framework-only CPU stubs.

This isolated process does not validate vLLM loading, TP, MoE dispatch or NPU graphs.
"""

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest
import torch


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    sys.modules[name] = value
    return value


def main():
    root = Path(__file__).resolve().parents[4]
    for name, path in (
        ("vllm_ascend", root / "vllm_ascend"),
        ("vllm_ascend.quantization", root / "vllm_ascend/quantization"),
        ("vllm_ascend.quantization.methods", root / "vllm_ascend/quantization/methods"),
    ):
        module(name, __path__=[str(path)])
    module("vllm", __path__=[])
    module("vllm.config", get_current_vllm_config=lambda: None, get_current_vllm_config_or_none=lambda: None)
    cpu_logger = logging.getLogger("cpu-test")
    cpu_logger.warning_once = lambda msg, *args, **kwargs: cpu_logger.warning(msg, *args, **kwargs)
    module("vllm.logger", logger=cpu_logger)
    module(
        "vllm.distributed",
        get_tensor_model_parallel_world_size=lambda: 1,
        tensor_model_parallel_all_gather=lambda value, dim=-1: value,
    )
    module("vllm.model_executor", __path__=[])
    module("vllm.model_executor.layers", __path__=[])
    module("vllm.model_executor.layers.linear", RowParallelLinear=type("RowParallelLinear", (torch.nn.Module,), {}))
    module(
        "vllm_ascend.quantization.methods.base",
        AscendLinearScheme=type("LinearScheme", (), {}),
        AscendMoEScheme=type("MoEScheme", (), {}),
        QuantType=types.SimpleNamespace(NONE=None),
        get_moe_num_logical_experts=lambda *a, **kw: 1,
    )
    module("vllm_ascend.ascend_config", get_ascend_config=lambda: None)
    module("vllm_ascend.ascend_forward_context", _EXTRA_CTX=None, MoECommType=types.SimpleNamespace(FUSED_MC2=1))
    module("vllm_ascend.ops.fused_moe.experts_selector", select_experts=None)
    # Placeholder stand-ins: the real per-expert executors live in moe_mlp.py
    # (imports torch_npu, unavailable on CPU). The transform factories in
    # fake_mx_algorithms.moe only close over them; the math itself is covered
    # by tests/ut/ops/test_moe_mlp.py on a real environment.
    module(
        "vllm_ascend.ops.fused_moe.moe_mlp",
        _apply_expert_flatquant=None,
        _apply_expert_learned_hadamard=None,
        _apply_expert_omniquant=None,
    )
    module("vllm_ascend.ops.fused_moe.moe_runtime_args", build_fused_experts_input=None)
    module("vllm_ascend.utils", maybe_trans_nz=lambda x: x)
    importlib.import_module("vllm_ascend.quantization.methods.fake_mx")
    return pytest.main(
        [
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(root / "tests/ut/quantization/methods/test_fake_mx_algorithms.py"),
            str(root / "tests/ut/quantization/test_fake_mx_registry.py"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
