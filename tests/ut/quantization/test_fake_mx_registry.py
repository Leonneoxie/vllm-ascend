import pytest

from vllm_ascend.quantization.methods.registry import get_scheme_class


@pytest.mark.parametrize("fmt", ["W4A4_MXFP4", "W8A8_MXFP8"])
def test_core_registry(fmt):
    for algo in ("", "_FLATQUANT", "_OMNIQUANT", "_RHT", "_HADAMARD_LEARNING"):
        assert get_scheme_class(fmt + algo + "_FAKE", "linear") is not None
    for algo in ("_AUTOROUND", "_LWC", "_LAC"):
        for layer in ("linear", "moe"):
            assert get_scheme_class(fmt + algo + "_FAKE", layer) is None
    for algo in ("", "_FLATQUANT", "_OMNIQUANT", "_RHT", "_HADAMARD_LEARNING"):
        assert get_scheme_class(fmt + algo + "_FAKE", "moe") is not None
