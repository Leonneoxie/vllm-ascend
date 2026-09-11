#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""RHT: deterministic Rademacher signs (seed=0) + normalized FWHT, matching AMCT."""

import torch
from vllm.logger import logger

from vllm_ascend.quantization.fake_mx import randomized_hadamard_transform

from .common import _find_transform_param, _load_transform_params, _make_rademacher_signs
from .linear import FakeMXLinearMethod


def _validated_rht_signs(
    signs: torch.Tensor,
    seed: int,
    matrix_size: int,
    params_path: str | None,
    layer: torch.nn.Module,
) -> torch.Tensor:
    """Return the runtime-generated signs, validated against AMCT if possible.

    Trusting the default seed alone is not enough: only a direct comparison
    proves that vLLM and AMCT run the same transform. When an optional
    ``rht_params_path`` sidecar carries the AMCT-exported ``rht_signs``, the
    generated signs must match bit-for-bit, otherwise the run fails. Without
    a sidecar the signs summary is logged so it can be compared against the
    AMCT side manually before long evaluations.
    """
    if params_path:
        ext_params = _load_transform_params(params_path)
        match = _find_transform_param(ext_params, layer, "rht_signs")
        if match is None:
            raise KeyError(
                f"rht_params_path is set but no 'rht_signs' parameter was found for layer "
                f"{getattr(layer, 'prefix', '') or getattr(layer, 'layer_name', '')!r}. Export "
                "the AMCT signs or drop rht_params_path to use runtime generation."
            )
        key, expected = match
        if expected.shape != signs.shape or not torch.equal(expected.cpu().to(torch.int8), signs):
            raise ValueError(
                f"RHT signs mismatch for layer {key!r}: the runtime-generated signs "
                f"(seed={seed}, matrix_size={matrix_size}) differ from the AMCT sidecar. "
                "Align rht_seed / rht_matrix_size with the AMCT calibration or the "
                "evaluation would compare different transforms."
            )
        logger.info("RHT signs validated against AMCT sidecar %s (seed=%d, size=%d)", key, seed, matrix_size)
    else:
        logger.info(
            "RHT signs generated at runtime (seed=%d, matrix_size=%d, head=%s); "
            "verify these against AMCT before long evaluations or provide rht_params_path",
            seed,
            matrix_size,
            signs[:8].tolist(),
        )
    return signs


class RHTLinearMethod(FakeMXLinearMethod):
    """RHT for Linear: deterministic signs (seed=0) + normalized FWHT.

    Generates Rademacher signs with a fixed seed at load time, matching
    AMCT's ``_HadamardTransform``. No external file needed: both AMCT and
    vLLM produce identical signs independently; an optional
    ``rht_params_path`` sidecar enables a bit-for-bit startup check.
    """

    algorithm = "rht"

    def __init__(self):
        super().__init__()
        quant_description = self.config
        self.rht_matrix_size = int(
            quant_description.get("rht_matrix_size", quant_description.get("rht_group_size", self.group_size))
        )
        self.rht_seed = int(quant_description.get("rht_seed", 0))
        self.params_path = quant_description.get("rht_params_path")

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        # Signs are fixed-size (rht_matrix_size, e.g. 128), matching AMCT's
        # _HadamardTransform which generates signs of length=matrix_size.
        # randomized_hadamard_transform broadcasts signs to all blocks.
        signs = _validated_rht_signs(
            _make_rademacher_signs(self.rht_matrix_size, self.rht_seed),
            self.rht_seed,
            self.rht_matrix_size,
            self.params_path,
            layer,
        )
        layer.fake_mx_rht_signs = signs
        layer.weight.data.copy_(
            randomized_hadamard_transform(layer.weight.data, layer.fake_mx_rht_signs, self.rht_matrix_size)
        )

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return randomized_hadamard_transform(x, layer.fake_mx_rht_signs, self.rht_matrix_size)
