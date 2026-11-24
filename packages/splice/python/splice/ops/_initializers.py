# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initializers for scaled and directly parameterized channel-mixing weights."""

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

SCALED_INITS = frozenset({"normal", "identity"})
DIRECT_INITS = frozenset(
    {
        "fan_in_normal",
        "fan_in_uniform",
        "xavier_normal",
        "xavier_uniform",
        "orthogonal",
        "identity",
    }
)


def initialize_linear(
    key: Array,
    shape: tuple[int, int],
    dtype: Any,
    *,
    init: str,
    weight_scaling: bool,
) -> Array:
    """Initialize a ``[fan_in, fan_out]`` matrix for one parameterization.

    Scaled weights retain the historical SPLICE convention: unit-normal stored
    weights and a runtime ``1/sqrt(fan_in)`` factor. Direct weights carry their
    complete variance in the parameter and receive no runtime fan-in factor.
    """
    fan_in, fan_out = shape
    allowed = SCALED_INITS if weight_scaling else DIRECT_INITS
    if init not in allowed:
        mode = "scaled" if weight_scaling else "direct"
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{mode} linear init must be one of {choices}; got {init!r}")

    if init == "identity":
        if fan_in != fan_out:
            raise ValueError(
                f"identity linear init requires fan_in == fan_out, got {fan_in} and {fan_out}"
            )
        factor = math.sqrt(fan_in) if weight_scaling else 1.0
        return jnp.eye(fan_in, fan_out, dtype=dtype) * factor
    if init == "normal":
        return jax.random.normal(key, shape, dtype)
    if init == "fan_in_normal":
        return jax.random.normal(key, shape, dtype) / math.sqrt(fan_in)
    if init == "fan_in_uniform":
        bound = math.sqrt(3.0 / fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    if init == "xavier_normal":
        return jax.random.normal(key, shape, dtype) * math.sqrt(2.0 / (fan_in + fan_out))
    if init == "xavier_uniform":
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    if init == "orthogonal":
        return jax.nn.initializers.orthogonal()(key, shape, dtype)
    raise AssertionError(f"unhandled linear initializer {init!r}")
