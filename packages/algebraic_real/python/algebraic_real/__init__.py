# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact real algebraic numbers and rational powers, without array dependencies."""

from fractions import Fraction
from operator import index
from typing import Any

from algebraic_real._native import AlgebraicReal


def _value(value: Any) -> AlgebraicReal:
    if isinstance(value, AlgebraicReal):
        return value
    if isinstance(value, Fraction):
        return AlgebraicReal(value)
    try:
        return AlgebraicReal(index(value))
    except TypeError:
        raise TypeError(
            f"exact values must be AlgebraicReal, int, or Fraction, got {type(value).__name__}"
        ) from None


def pow(base: AlgebraicReal | int | Fraction, exponent: int | Fraction) -> AlgebraicReal:
    """Raise an exact real value to an exact rational power."""
    if not isinstance(exponent, Fraction):
        try:
            exponent = index(exponent)
        except TypeError:
            raise TypeError("pow(): exponent must be int or Fraction") from None
    return _value(base) ** exponent


__all__ = ["AlgebraicReal", "pow"]
