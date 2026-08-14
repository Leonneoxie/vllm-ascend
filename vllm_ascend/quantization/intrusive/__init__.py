#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Intrusive injection adapters for Fake-MX A/B verification.

Each adapter manages the weight and activation lifecycle for one
algorithm (RTN, FlatQuant, LHT) by calling shared math functions.
Adapters do NOT reimplement quantization formulas — they only
orchestrate parameter loading, transform timing, and audit events.
"""

# Explicit imports trigger @register_intrusive_adapter decorators.
from . import (
    flatquant,  # noqa: F401
    lht,  # noqa: F401
    rtn,  # noqa: F401
)
from .base import IntrusiveLinearAdapter, maybe_create_intrusive_linear_adapter

__all__ = [
    "IntrusiveLinearAdapter",
    "maybe_create_intrusive_linear_adapter",
]
