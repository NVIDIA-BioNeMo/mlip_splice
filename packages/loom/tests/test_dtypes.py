# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import loom
import numpy as np
import numpy.testing as npt
import pytest
from loom import Compile, DomainGraph, IndexMap, domains

from _support import _INTEGER_DTYPES, _call, _sds
from marks import gpu_only

pytestmark = pytest.mark.capped_autotune


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_identity_and_overflow(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        info = np.iinfo(integer_dtype)
        values = np.asarray(
            [info.max, info.min, 0, 1, 2, 7, info.max - 1, info.min + 1],
            dtype=integer_dtype,
        ).reshape(2, 1, 4)
        x = jnp.asarray(values)
        one = jnp.asarray(1, dtype=integer_dtype)
        y = _call(lambda value: value + one, x, _sds(x.shape, integer_dtype))
        expected = np.add(values, np.asarray(1, dtype=integer_dtype))
        npt.assert_array_equal(y, expected)


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_compare_select(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        values = np.arange(8, dtype=integer_dtype).reshape(2, 1, 4)
        x = jnp.asarray(values)
        threshold = jnp.asarray(3, dtype=integer_dtype)
        replacement = jnp.asarray(11, dtype=integer_dtype)
        y = _call(
            lambda value: jnp.where(value > threshold, value, replacement),
            x,
            _sds(x.shape, integer_dtype),
        )
        npt.assert_array_equal(y, np.where(values > 3, values, 11))


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_extrema(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        info = np.iinfo(integer_dtype)
        values = np.asarray(
            [info.min, info.max, 0, 1, 2, 7, info.max - 1, info.min + 1],
            dtype=integer_dtype,
        ).reshape(2, 1, 4)
        x = jnp.asarray(values)
        lower = jnp.asarray(2, dtype=integer_dtype)
        upper = jnp.asarray(7, dtype=integer_dtype)
        maximum, minimum = loom.call(
            lambda value: (
                jnp.maximum(value, lower),
                jnp.minimum(value, upper),
            ),
            x,
            domain_graph=DomainGraph({"n": x.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )
        npt.assert_array_equal(maximum, np.maximum(values, 2))
        npt.assert_array_equal(minimum, np.minimum(values, 7))


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_to_float_conversion(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        values = np.arange(8, dtype=integer_dtype).reshape(2, 1, 4)
        x = jnp.asarray(values)
        y = _call(
            lambda value: value.astype(jnp.float32),
            x,
            _sds(x.shape, jnp.float32),
        )
        npt.assert_array_equal(y, values.astype(np.float32))


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_width_one_output_wraps_sum(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        info = np.iinfo(integer_dtype)
        values = np.asarray(
            [info.max, 1, 2, 3, info.min, info.min, 7, 9], dtype=integer_dtype
        ).reshape(2, 1, 4)
        x = jnp.asarray(values)
        y = _call(
            lambda value: value,
            x,
            _sds((2, 1, 1), integer_dtype),
        )
        expected = np.add.reduce(values, axis=-1, dtype=integer_dtype, keepdims=True)
        npt.assert_array_equal(y, expected)


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_division_and_remainder(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        if jnp.issubdtype(integer_dtype, jnp.signedinteger):
            values = np.asarray([-17, -8, -1, 0, 1, 8, 17, 29], dtype=integer_dtype)
        else:
            values = np.asarray([0, 1, 2, 7, 8, 17, 29, 41], dtype=integer_dtype)
        values = values.reshape(2, 1, 4)
        x = jnp.asarray(values)
        divisor = jnp.asarray(5, dtype=integer_dtype)
        quotient, remainder = loom.call(
            lambda value: (jax.lax.div(value, divisor), jax.lax.rem(value, divisor)),
            x,
            domain_graph=DomainGraph({"n": x.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )
        expected_quotient = np.trunc(values.astype(object) / 5).astype(integer_dtype)
        expected_remainder = values - expected_quotient * np.asarray(5, integer_dtype)
        npt.assert_array_equal(quotient, expected_quotient)
        npt.assert_array_equal(remainder, expected_remainder)


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_bitwise_and_shifts(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        info = np.iinfo(integer_dtype)
        values = np.asarray([info.min, info.max, 0, 1, 2, 7, 17, 31], dtype=integer_dtype).reshape(
            2, 1, 4
        )
        x = jnp.asarray(values)
        one = jnp.asarray(1, dtype=integer_dtype)
        mask = jnp.asarray(0x55, dtype=integer_dtype)
        left, logical, bits = loom.call(
            lambda value: (
                jax.lax.shift_left(value, one),
                jax.lax.shift_right_logical(value, one),
                (value & mask) ^ one,
            ),
            x,
            domain_graph=DomainGraph({"n": x.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )
        unsigned_dtype = np.dtype(f"u{np.dtype(integer_dtype).itemsize}")
        expected_logical = np.right_shift(values.view(unsigned_dtype), 1).view(integer_dtype)
        npt.assert_array_equal(left, np.left_shift(values, 1))
        npt.assert_array_equal(logical, expected_logical)
        npt.assert_array_equal(bits, np.bitwise_xor(np.bitwise_and(values, 0x55), 1))


@pytest.mark.parametrize("integer_dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.int64])
def test_integer_arithmetic_shift(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        values = np.asarray([-128, -17, -2, -1, 0, 1, 17, 63], dtype=integer_dtype)
        values = values.reshape(2, 1, 4)
        x = jnp.asarray(values)
        one = jnp.asarray(1, dtype=integer_dtype)
        y = _call(
            lambda value: jax.lax.shift_right_arithmetic(value, one),
            x,
            _sds(x.shape, integer_dtype),
        )
        npt.assert_array_equal(y, np.right_shift(values, 1))


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_float_to_integer_conversion(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        values = np.asarray([0.0, 1.9, 2.1, 7.8, 8.0, 17.2, 29.9, 41.0], np.float32)
        if jnp.issubdtype(integer_dtype, jnp.signedinteger):
            values[0] = -7.9
        values = values.reshape(2, 1, 4)
        x = jnp.asarray(values)
        y = _call(
            lambda value: value.astype(integer_dtype),
            x,
            _sds(x.shape, integer_dtype),
        )
        npt.assert_array_equal(y, values.astype(integer_dtype))


def test_output_storage_narrows_after_pointwise_math(device):
    x = jnp.linspace(0.1, 1.0, 32, dtype=jnp.float16).reshape(4, 2, 4)
    y = loom.call(
        lambda value: value.astype(jnp.float32) * 1.125,
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains=domains["n"],
        output_dtypes=jnp.float16,
    )
    assert y.shape == x.shape
    assert y.dtype == jnp.float16
    npt.assert_array_equal(y, (x.astype(jnp.float32) * 1.125).astype(jnp.float16))


def test_output_storage_narrows_after_domain_and_lane_reductions(device):
    edges, u = 1024, 4
    x = jnp.full((edges, 1, u), 0.1, jnp.float16)
    y = loom.call(
        lambda value: value.astype(jnp.float32),
        x,
        domain_graph=DomainGraph({"edge": edges}),
        input_domains=domains["edge"],
        output_domains=domains[:],
        reduce_lanes=True,
        output_dtypes=jnp.float16,
    )
    expected = x.astype(jnp.float32).sum().astype(jnp.float16)
    assert y.shape == (1, 1, 1)
    assert y.dtype == jnp.float16
    npt.assert_array_equal(y[0, 0, 0], expected)


@gpu_only
def test_a_widened_accumulator_avoids_math_dtype_overflow(device):
    """256 ones summed in int8 wrap to 0; the accumulator widens to the output's
    own int32, so the sum is the 256 the program asked for.
    """
    x = jnp.ones((256, 1), jnp.int8)
    y = loom.call(
        lambda value: value,
        x,
        u=1,
        domain_graph=DomainGraph({"edge": x.shape[0]}),
        input_domains=domains["edge"],
        output_domains=domains[:],
        output_dtypes=jnp.int32,
    )
    npt.assert_array_equal(y, jnp.asarray([[256]], jnp.int32))


@gpu_only
def test_output_storage_narrows_after_continuation_launches(device):
    x = jnp.linspace(0.01, 0.1, 96, dtype=jnp.float32).reshape(3, 8, 4)

    def pointwise(value):
        result = value
        for _ in range(60):
            result = result * value + value
        return result

    y = loom.call(
        pointwise,
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains=domains["n"],
        output_dtypes=jnp.float16,
        compile_options=Compile(max_kernel_statements=16),
    )
    npt.assert_array_equal(y, pointwise(x).astype(jnp.float16))


@gpu_only
def test_output_storage_narrows_after_split_xir_reduction(device):
    x = jnp.linspace(0.01, 0.1, 3 * 128 * 4, dtype=jnp.float32).reshape(3, 128, 4)
    y = loom.call(
        lambda value: value.sum(axis=0),
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains=domains["n"],
        output_dtypes=jnp.float16,
        compile_options=Compile(max_kernel_statements=80),
    )
    npt.assert_array_equal(y, x.sum(axis=1).astype(jnp.float16))


def test_boolean_physical_buffers(device):
    values = np.asarray([False, True, True, False, True, False, False, True]).reshape(2, 1, 4)
    x = jnp.asarray(values)
    y = _call(
        lambda value: value,
        x,
        _sds(x.shape, jnp.bool_),
    )
    npt.assert_array_equal(y, values)


def test_boolean_hidden_lane_reduction_is_rejected(device):
    values = jnp.asarray([False, True, True, False]).reshape(1, 1, 4)
    with pytest.raises(ValueError, match="boolean hidden-lane reduction"):
        _call(lambda value: value, values, _sds((1, 1, 1), jnp.bool_))


def test_boolean_domain_reduction_is_rejected(device):
    edges, nodes, u = 4, 1, 4
    values = jnp.asarray(np.arange(edges * u).reshape(edges, 1, u) % 2 == 0)
    target_indices = np.zeros(edges, np.int32)
    domain_graph = DomainGraph(
        {"edge": edges, "node": nodes},
        {("edge", "node"): IndexMap(target_indices)},
    )
    with pytest.raises(ValueError, match="boolean DomainIndex reduction"):
        loom.call(
            lambda value: value,
            values,
            domain_graph=domain_graph,
            input_domains=domains["edge"],
            output_domains=domains["node"],
        )


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_compare_to_boolean_buffer(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        values = np.arange(8, dtype=integer_dtype).reshape(2, 1, 4)
        x = jnp.asarray(values)
        threshold = jnp.asarray(3, dtype=integer_dtype)
        y = _call(lambda value: value > threshold, x, _sds(x.shape, jnp.bool_))
        npt.assert_array_equal(y, values > 3)


def test_boolean_input_selects_integer_values(device):
    values = np.asarray([False, True, True, False, True, False, False, True]).reshape(2, 1, 4)
    x = jnp.asarray(values)
    yes = jnp.asarray(11, jnp.int32)
    no = jnp.asarray(-7, jnp.int32)
    y = _call(lambda value: jnp.where(value, yes, no), x, _sds(x.shape, jnp.int32))
    npt.assert_array_equal(y, np.where(values, 11, -7))


@pytest.mark.parametrize("loom_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_v1_dtypes(device, loom_dtype):
    x = jnp.arange(16, dtype=loom_dtype).reshape(2, 2, 4)
    y = _call(lambda v: v * 2, x, _sds(x.shape, loom_dtype))
    npt.assert_allclose(y, x * 2, rtol=1e-2, atol=1e-2)


def test_v1_f64(device):
    with jax.enable_x64(True):
        x = jnp.arange(16, dtype=jnp.float64).reshape(2, 2, 4)
        y = _call(lambda v: v * 2, x, _sds(x.shape, jnp.float64))
        npt.assert_allclose(y, x * 2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("storage_dtype", "hidden_width"),
    [
        pytest.param(jnp.float16, 128, id="f16-vec4"),
        pytest.param(jnp.bfloat16, 256, id="bf16-vec8"),
        pytest.param(jnp.bfloat16, 260, id="bf16-tail"),
    ],
)
@pytest.mark.capped_autotune(search_budget_ms=250.0)
@gpu_only
def test_half_vector_io_forward_and_backward(device, storage_dtype, hidden_width):
    positions, segments = 257, 3
    x = jnp.linspace(
        0.125,
        0.875,
        positions * segments * hidden_width,
        dtype=jnp.float32,
    ).astype(storage_dtype)
    x = x.reshape(positions, segments, hidden_width)
    scale = jnp.linspace(
        0.75,
        1.25,
        positions * segments,
        dtype=jnp.float32,
    ).astype(storage_dtype)
    scale = scale.reshape(positions, segments, 1)

    def pointwise(values):
        value, factor = (operand.astype(jnp.float32) for operand in values)
        return (value * value + value * factor).astype(storage_dtype)

    def run(value, factor):
        return loom.call(
            pointwise,
            (value, factor),
            u=hidden_width,
            domain_graph=DomainGraph({"n": positions}),
            input_domains=(domains["n"], domains["n"]),
            output_domains=domains["n"],
            compile_options=Compile(),
        )

    tolerance = 2e-2 if storage_dtype == jnp.float16 else 3e-2
    actual = run(x, scale)
    expected = pointwise((x, scale))
    npt.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    def loss(fn, value, factor):
        return fn(value, factor).astype(jnp.float32).sum()

    actual_grad = jax.grad(lambda value, factor: loss(run, value, factor), argnums=(0, 1))(x, scale)
    expected_grad = jax.grad(
        lambda value, factor: loss(lambda left, right: pointwise((left, right)), value, factor),
        argnums=(0, 1),
    )(x, scale)
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        npt.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.capped_autotune(search_budget_ms=250.0)
@gpu_only
def test_half_vector_self_division_preserves_special_values(device):
    positions, segments, hidden_width = 257, 2, 256
    x = jnp.linspace(
        0.125,
        0.875,
        positions * segments * hidden_width,
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    x = x.reshape(positions, segments, hidden_width)
    x = x.at[0, :, :4].set(jnp.asarray([0.0, -0.0, jnp.inf, jnp.nan], dtype=x.dtype))

    def pointwise(value):
        ratio = value / value
        return ratio, ratio

    wide, folded = loom.call(
        pointwise,
        x,
        u=hidden_width,
        domain_graph=DomainGraph({"n": positions}),
        input_domains=domains["n"],
        output_domains=(domains["n"], domains["n"]),
        reduce_lanes=(False, True),
        compile_options=Compile(),
    )
    expected = x / x
    npt.assert_array_equal(wide.astype(jnp.float32), expected.astype(jnp.float32))
    npt.assert_array_equal(
        folded.astype(jnp.float32), expected.sum(axis=-1, keepdims=True).astype(jnp.float32)
    )
