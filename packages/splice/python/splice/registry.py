# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Op registry: maps a string name to an op class.

Ops self-register via `@register_op("name")`. The loader looks ops up by name.
"""

_REGISTRY: dict[str, type] = {}


def register_op(name: str):
    def decorator(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"op name {name!r} already registered to {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        cls._OP_NAME = name
        return cls

    return decorator


def get_op(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"op {name!r} not registered. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY)
