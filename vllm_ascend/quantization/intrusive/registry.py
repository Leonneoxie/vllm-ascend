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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import IntrusiveLinearAdapter

_INTRUSIVE_ADAPTERS: dict[str, type["IntrusiveLinearAdapter"]] = {}


def register_intrusive_adapter(algorithm: str):
    """Decorator to register an intrusive adapter for *algorithm*.

    Raises ``ValueError`` on duplicate registration.
    """

    def decorator(cls: type["IntrusiveLinearAdapter"]) -> type["IntrusiveLinearAdapter"]:
        if algorithm in _INTRUSIVE_ADAPTERS:
            raise ValueError(f"Duplicate intrusive adapter for algorithm {algorithm!r}")
        _INTRUSIVE_ADAPTERS[algorithm] = cls
        return cls

    return decorator


def get_intrusive_adapter_class(algorithm: str) -> type["IntrusiveLinearAdapter"]:
    """Return the adapter class registered for *algorithm*.

    Raises ``NotImplementedError`` if no adapter is registered.
    """
    cls = _INTRUSIVE_ADAPTERS.get(algorithm)
    if cls is None:
        raise NotImplementedError(
            f"No intrusive adapter registered for algorithm {algorithm!r}. "
            f"Available: {sorted(_INTRUSIVE_ADAPTERS)}"
        )
    return cls
