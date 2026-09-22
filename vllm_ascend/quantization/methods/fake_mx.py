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
"""Compatibility scheme names for the five core fake-MX algorithms."""

from vllm_ascend.quantization.fake_mx import FakeMXFormat

from .fake_mx_algorithms.flatquant import FlatQuantLinearMethod
from .fake_mx_algorithms.lht import LHTLinearMethod
from .fake_mx_algorithms.linear import FakeMXLinearMethod
from .fake_mx_algorithms.moe import FakeMXMoEMethod, FlatQuantMoEMethod, LHTMoEMethod, OmniQuantMoEMethod, RHTMoEMethod
from .fake_mx_algorithms.omniquant import OmniQuantLinearMethod
from .fake_mx_algorithms.rht import RHTLinearMethod
from .registry import register_scheme


@register_scheme("W4A4_MXFP4_FAKE", "linear")
class AscendW4A4MXFP4FakeLinearMethod(FakeMXLinearMethod):
    """MXFP4 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "linear")
class AscendW8A8MXFP8FakeLinearMethod(FakeMXLinearMethod):
    """MXFP8 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "linear")
class AscendW4A4MXFP4OmniQuantFakeLinearMethod(OmniQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "linear")
class AscendW8A8MXFP8OmniQuantFakeLinearMethod(OmniQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_RHT_FAKE", "linear")
class AscendW4A4MXFP4RHTFakeLinearMethod(RHTLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "linear")
class AscendW8A8MXFP8RHTFakeLinearMethod(RHTLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "linear")
class AscendW4A4MXFP4HadamardLearningFakeLinearMethod(LHTLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "linear")
class AscendW8A8MXFP8HadamardLearningFakeLinearMethod(LHTLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_FLATQUANT_FAKE", "linear")
class AscendW4A4MXFP4FakeFlatQuantLinearMethod(FlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FLATQUANT_FAKE", "linear")
class AscendW8A8MXFP8FakeFlatQuantLinearMethod(FlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_FAKE", "moe")
class AscendW4A4MXFP4FakeFusedMoEMethod(FakeMXMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "moe")
class AscendW8A8MXFP8FakeFusedMoEMethod(FakeMXMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "moe")
class AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod(LHTMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "moe")
class AscendW8A8MXFP8HadamardLearningFakeFusedMoEMethod(LHTMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_RHT_FAKE", "moe")
class AscendW4A4MXFP4RHTFakeFusedMoEMethod(RHTMoEMethod):
    """RHT for MoE: rotates weights at load time using deterministic signs
    (seed=0) + normalized FWHT, same as Linear RHT."""

    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "moe")
class AscendW8A8MXFP8RHTFakeFusedMoEMethod(RHTMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "moe")
class AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod(OmniQuantMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "moe")
class AscendW8A8MXFP8OmniQuantFakeFusedMoEMethod(OmniQuantMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_FLATQUANT_FAKE", "moe")
class AscendW4A4MXFP4FakeFlatQuantFusedMoEMethod(FlatQuantMoEMethod):
    """W4A4 MXFP4 FlatQuant fake-QDQ for FusedMoE (routed experts only)."""

    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FLATQUANT_FAKE", "moe")
class AscendW8A8MXFP8FakeFlatQuantFusedMoEMethod(FlatQuantMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
