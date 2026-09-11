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

from vllm_ascend.quantization.fake_mx import randomized_hadamard_transform

from .linear import FakeMXLinearMethod


class RHTLinearMethod(FakeMXLinearMethod):
    """RHT for Linear: deterministic signs (seed=0) + normalized FWHT.

    Generates Rademacher signs with a fixed seed=0 at load time, matching
    AMCT's ``_HadamardTransform`` (also seed=0).  No external file needed:
    both AMCT and vLLM produce identical signs independently.
    """

    algorithm = "rht"

    def __init__(self):
        super().__init__()
        quant_description = self.config
        self.rht_matrix_size = int(
            quant_description.get("rht_matrix_size", quant_description.get("rht_group_size", self.group_size))
        )
        self.rht_seed = int(quant_description.get("rht_seed", 0))

    @staticmethod
    def _make_signs(size: int, seed: int) -> torch.Tensor:
        """Generate Rademacher signs (+/-1) with fixed seed, matching AMCT."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
        return signs.mul_(2).sub_(1)

    def prepare_weight(self, layer: torch.nn.Module) -> None:
        # Signs are fixed-size (rht_matrix_size, e.g. 128), matching AMCT's
        # _HadamardTransform which generates signs of length=matrix_size.
        # randomized_hadamard_transform broadcasts signs to all blocks.
        layer.fake_mx_rht_signs = self._make_signs(self.rht_matrix_size, self.rht_seed)
        layer.weight.data.copy_(
            randomized_hadamard_transform(layer.weight.data, layer.fake_mx_rht_signs, self.rht_matrix_size)
        )

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return randomized_hadamard_transform(x, layer.fake_mx_rht_signs, self.rht_matrix_size)
