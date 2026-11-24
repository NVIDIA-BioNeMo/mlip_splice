# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every elementwise operation Loom accepts, against JAX on the same inputs.

These are the operations exact rewriting cannot rewrite: it recognizes the field
operations and square roots, so each function here is an atom that bounds the
island a rewrite works inside. What is checked is therefore only that the value
Loom computes is the value JAX computes, to the precision of the hardware
approximation the kernel is built from.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import lax

from _support import _call, _sds

pytestmark = pytest.mark.capped_autotune

# The approximate instructions carry a few units in the last place, so the
# tolerance is looser than the exact arithmetic elsewhere in this suite.
_APPROX = dict(rtol=3e-6, atol=3e-6)

_POSITIVE = np.asarray([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 7.0, 11.0], np.float32)
_SIGNED = np.asarray([-3.5, -2.0, -0.75, -0.25, 0.25, 0.75, 2.0, 3.5], np.float32)
_ROUNDING = np.asarray([-2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5], np.float32)


def _shaped(values):
    return jnp.asarray(values.reshape(2, 1, 4))


@pytest.mark.parametrize(
    ("name", "function", "values"),
    [
        ("exp", jnp.exp, _SIGNED),
        ("expm1", jnp.expm1, _SIGNED),
        ("log", jnp.log, _POSITIVE),
        ("log1p", jnp.log1p, _POSITIVE),
        ("logistic", lambda v: 1.0 / (1.0 + jnp.exp(-v)), _SIGNED),
        ("tanh", jnp.tanh, _SIGNED),
        ("sin", jnp.sin, _SIGNED),
        ("cos", jnp.cos, _SIGNED),
        ("tan", jnp.tan, _SIGNED),
        ("cbrt", jnp.cbrt, _SIGNED),
        ("floor", jnp.floor, _ROUNDING),
        ("ceil", jnp.ceil, _ROUNDING),
        ("round", jnp.round, _ROUNDING),
        (
            "round-afz",
            lambda v: lax.round(v, lax.RoundingMethod.AWAY_FROM_ZERO),
            _ROUNDING,
        ),
        ("sqrt", jnp.sqrt, _POSITIVE),
        ("rsqrt", lax.rsqrt, _POSITIVE),
        ("pow17", lambda v: jnp.power(v, 17.0), _POSITIVE),
        ("pow-half", lambda v: jnp.power(v, 1.5), _POSITIVE),
    ],
)
def test_an_elementwise_function_matches_jax(device, name, function, values):
    x = _shaped(values)
    y = _call(function, x, _sds(x.shape, x.dtype))
    npt.assert_allclose(y, function(x), **_APPROX)


def test_clamp_bounds_from_both_sides(device):
    x = _shaped(_SIGNED)
    y = _call(lambda value: jnp.clip(value, -1.0, 1.0), x, _sds(x.shape, x.dtype))
    npt.assert_allclose(y, jnp.clip(x, -1.0, 1.0), **_APPROX)


def test_is_finite_separates_infinities_and_nan(device):
    values = np.asarray([0.0, 1.0, -1.0, 1e30, np.inf, -np.inf, np.nan, 2.0], np.float32)
    x = _shaped(values)
    y = _call(lambda value: jnp.isfinite(value).astype(jnp.float32), x, _sds(x.shape, x.dtype))
    npt.assert_array_equal(np.asarray(y).reshape(-1), np.isfinite(values).astype(np.float32))


def test_a_transcendental_composes_with_the_algebraic_core(device):
    """A rewrite island bounded by two atoms: the algebra between them still
    collapses, and the atoms still evaluate."""
    x = _shaped(_POSITIVE)
    y = _call(
        lambda value: jnp.exp(value) * (value / value) + jnp.log(value) * 0.0,
        x,
        _sds(x.shape, x.dtype),
    )
    npt.assert_allclose(y, jnp.exp(x), **_APPROX)


@pytest.mark.parametrize(
    ("name", "integer_dtype"),
    [("clz", jnp.int32), ("popc", jnp.int32), ("clz8", jnp.int8), ("popc8", jnp.uint8)],
)
def test_a_bit_count_counts_the_operand_width(device, name, integer_dtype):
    """PTX counts a whole register, so a narrower type has to say its own width."""
    info = np.iinfo(integer_dtype)
    values = np.asarray([0, 1, 2, 7, 8, info.max, info.min, -1], np.int64).astype(integer_dtype)
    x = jnp.asarray(values.reshape(2, 1, 4))
    counting = lax.clz if name.startswith("clz") else lax.population_count
    y = _call(counting, x, _sds(x.shape, x.dtype))
    npt.assert_array_equal(np.asarray(y).reshape(-1), np.asarray(counting(x)).reshape(-1))


@pytest.mark.parametrize(
    "integer_dtype", [jnp.int8, jnp.uint8, jnp.int16, jnp.uint16, jnp.int32, jnp.uint32, jnp.int64]
)
def test_not_preserves_each_integer_bit(device, integer_dtype):
    """A sub-word integer rides a wider register, so a complement has to leave
    the bits above its own width as its sign or its zeros."""
    info = np.iinfo(integer_dtype)
    values = np.asarray([0, 1, 2, 7, 8, info.max, info.min, -1], np.int64).astype(integer_dtype)
    x = jnp.asarray(values.reshape(2, 1, 4))
    y = _call(jnp.bitwise_not, x, _sds(x.shape, x.dtype))
    npt.assert_array_equal(y, jnp.bitwise_not(x))


def test_atan2_matches_jax_in_every_quadrant(device):
    """PTX has no arc tangent, so this one is a reduction and a polynomial. What
    matters is the quadrant, over both signs of both operands and on the axes.

    The two operands ride the function's own axis: element `v[0]` is the
    numerator and `v[1]` the denominator, so one call covers every pair.
    """
    grid = np.asarray([-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0], np.float32)
    numerator, denominator = (a.reshape(-1) for a in np.meshgrid(grid, grid))
    lanes = 4
    pairs = np.stack([numerator, denominator], axis=1)[:, :, None]
    pairs = np.repeat(pairs, lanes, axis=2).astype(np.float32)
    x = jnp.asarray(pairs)

    y = _call(
        lambda v: jnp.arctan2(v[0], v[1]),
        x,
        _sds((pairs.shape[0], lanes), jnp.float32),
    )
    expected = np.arctan2(numerator, denominator)
    npt.assert_allclose(np.asarray(y)[:, 0], expected, rtol=2e-6, atol=2e-6)


def test_a_double_transcendental_is_refused_rather_than_narrowed(device):
    import jax

    with jax.enable_x64(True):
        x = jnp.asarray(_POSITIVE.astype(np.float64).reshape(2, 1, 4))
        if device.platform == "cpu":
            npt.assert_allclose(_call(jnp.exp, x, _sds(x.shape, jnp.float64)), jnp.exp(x))
        else:
            with pytest.raises(Exception, match="double-precision"):
                _call(jnp.exp, x, _sds(x.shape, jnp.float64))
