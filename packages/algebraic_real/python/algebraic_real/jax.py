# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import lax

from algebraic_real import AlgebraicReal, _value

_COMPOSITE = "algebraic_real.exact_constant"


@partial(lax.composite, name=_COMPOSITE)
def _exact_constant(value, /, *, algebraic_real):
    del algebraic_real
    return value


def _data(value: AlgebraicReal) -> tuple:
    return value._algebraic_real_data()


def exact(values: Any, *, dtype: Any) -> jnp.ndarray:
    """Carry integers, fractions, or algebraic values exactly into JAX compiler IR."""
    dtype = np.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"exact dtype must be floating, got {dtype}")
    source = np.asarray(values, dtype=object)
    exact_values = [_value(value) for value in source.reshape(-1)]
    numeric = np.asarray([float(value) for value in exact_values], dtype=dtype).reshape(
        source.shape
    )
    payload = json.dumps([1, [_data(value) for value in exact_values]], separators=(",", ":"))
    return _exact_constant(jnp.asarray(numeric), algebraic_real=payload)


__all__ = ["exact"]
