# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle

import jax
import jax.numpy as jnp
import loom
import numpy.testing as npt
import pytest
from loom import Compile, DomainGraph, domains

from _support import _call, _sds
from marks import gpu_only

pytestmark = pytest.mark.capped_autotune


def test_hidden_u_elementwise(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    y = _call(lambda v: v * v + 2 * v, x, _sds(x.shape, dtype))
    npt.assert_allclose(y, x * x + 2 * x, rtol=1e-5, atol=1e-5)


def test_zero_domain_axes_are_the_default(device, dtype):
    x = jnp.arange(4, dtype=dtype)
    y = loom.call(lambda value: value * value + 2 * value, x)
    npt.assert_allclose(y, x * x + 2 * x, rtol=1e-5, atol=1e-5)


def test_split_visible_program_preserves_ad(device, dtype):
    x = jnp.linspace(0.01, 0.1, 96, dtype=dtype).reshape(3, 8, 4)

    def pointwise(value):
        result = value
        for _ in range(20):
            result = result * value + value
        return result

    def call(value):
        return _call(
            pointwise,
            value,
            _sds(value.shape, dtype),
            compile_options=Compile(max_kernel_statements=256),
        )

    expected = pointwise(x)
    npt.assert_allclose(call(x), expected, rtol=3e-5, atol=3e-5)

    tangent = jnp.ones_like(x)
    got_jvp = jax.jvp(call, (x,), (tangent,))[1]
    expected_jvp = jax.jvp(pointwise, (x,), (tangent,))[1]
    npt.assert_allclose(got_jvp, expected_jvp, rtol=5e-5, atol=5e-5)

    got_grad = jax.grad(lambda value: jnp.sum(call(value)))(x)
    expected_grad = jax.grad(lambda value: jnp.sum(pointwise(value)))(x)
    npt.assert_allclose(got_grad, expected_grad, rtol=7e-5, atol=7e-5)


def test_max_kernel_statements_is_validated_before_lowering():
    x = jnp.ones((1, 1, 1), jnp.float32)
    with pytest.raises(ValueError, match="max_kernel_statements"):
        _call(
            lambda value: value,
            x,
            _sds(x.shape, x.dtype),
            compile_options=Compile(max_kernel_statements=0),
        )


def test_max_workspace_bytes_is_validated_before_lowering():
    x = jnp.ones((1, 1, 1), jnp.float32)
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        _call(
            lambda value: value,
            x,
            _sds(x.shape, x.dtype),
            compile_options=Compile(max_workspace_bytes=-1),
        )
    with pytest.raises(TypeError, match="max_workspace_bytes"):
        _call(
            lambda value: value,
            x,
            _sds(x.shape, x.dtype),
            compile_options=Compile(max_workspace_bytes=True),
        )


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_search_budget_must_be_finite_and_positive(budget):
    x = jnp.ones((1, 1, 1), jnp.float32)
    with pytest.raises(ValueError, match="search_budget_ms"):
        _call(
            lambda value: value,
            x,
            _sds(x.shape, x.dtype),
            compile_options=Compile(search_budget_ms=budget),
        )


def test_search_budget_rejects_boolean():
    x = jnp.ones((1, 1, 1), jnp.float32)
    with pytest.raises(TypeError, match="search_budget_ms"):
        _call(
            lambda value: value,
            x,
            _sds(x.shape, x.dtype),
            compile_options=Compile(search_budget_ms=True),
        )


@pytest.mark.parametrize("name", ["search_budget_ms", "shortlist_budget_ms"])
@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_rule_budget_must_be_finite_and_positive(name, budget):
    with pytest.raises(ValueError, match=name):
        Compile(**{name: {1: budget}})


@pytest.mark.parametrize("name", ["search_budget_ms", "shortlist_budget_ms"])
def test_rule_budget_rejects_boolean(name):
    with pytest.raises(TypeError, match=name):
        Compile(**{name: {1: True}})


@pytest.mark.parametrize("rule", [True, 1.5, "1"])
def test_rule_budget_requires_integer_index(rule):
    with pytest.raises(TypeError, match="rule indices"):
        Compile(search_budget_ms={rule: 1})


def test_rule_budget_requires_nonnegative_index():
    with pytest.raises(ValueError, match="rule indices"):
        Compile(search_budget_ms={-1: 1})


def test_none_rule_budget_resets_inherited_override():
    options = Compile(search_budget_ms={0: 20, 1: None, 2: 1})

    assert options.search_budget_ms == {0: 20.0, 1: None, 2: 1.0}
    assert options.advance_transpose().current_search_budget_ms is None
    assert options.advance_transpose().advance_transpose().current_search_budget_ms == 1.0


def test_none_rule_budget_is_inherited_by_later_omissions():
    options = Compile(search_budget_ms={0: 20, 1: None})

    assert options.advance_transpose().current_search_budget_ms is None
    assert options.advance_transpose().advance_transpose().current_search_budget_ms is None


def test_empty_rule_budget_inherits():
    options = Compile(search_budget_ms={})

    assert options.search_budget_ms is ...
    assert options == Compile()


def test_rule_budgets_are_canonical_hashable_and_immutable():
    left = Compile(search_budget_ms={2: 1, 0: 20})
    right = Compile(search_budget_ms={0: 20, 2: 1})

    assert left == right
    assert hash(left) == hash(right)
    restored = pickle.loads(pickle.dumps(left))
    assert restored == left
    assert hash(restored) == hash(left)
    assert copy.copy(left) == left
    with pytest.raises(AttributeError, match="immutable"):
        left.search_budget_ms._items = ((0, 1.0),)
    with pytest.raises(AttributeError, match="immutable"):
        del left.search_budget_ms._items


def test_a_polynomial_executes(device):
    x = jnp.linspace(0.25, 2.0, 8, dtype=jnp.float32).reshape(2, 1, 4)
    y = _call(
        lambda value: value * value + value,
        x,
        _sds(x.shape, x.dtype),
    )
    npt.assert_allclose(y, x * x + x, rtol=1e-6, atol=1e-6)


@gpu_only
def test_exact_rewriting_cancels_multiplication_by_zero(device):
    infinities = jnp.full((1, 1, 4), jnp.inf, jnp.float32)
    product = _call(lambda value: value * 0, infinities, _sds(infinities.shape, infinities.dtype))
    npt.assert_array_equal(product, jnp.zeros_like(product))


@gpu_only
@pytest.mark.parametrize("floating_dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_self_division_preserves_special_values(device, floating_dtype):
    with jax.enable_x64(floating_dtype == jnp.float64):
        x = jnp.asarray(
            [0.0, -0.0, jnp.inf, -jnp.inf, jnp.nan, 1.0, -2.0, 0.5],
            dtype=floating_dtype,
        ).reshape(2, 1, 4)
        quotient = _call(lambda value: value / value, x, _sds(x.shape, x.dtype))
        npt.assert_array_equal(quotient.astype(jnp.float32), (x / x).astype(jnp.float32))


def test_width_one_input_broadcasts_and_output_sums(device, dtype):
    x = jnp.arange(6, dtype=dtype).reshape(3, 2, 1)
    y = _call(lambda v: v + 2, x, _sds((3, 2, 1), dtype), u=8)
    npt.assert_allclose(y, (x + 2) * 8, rtol=1e-5, atol=1e-5)


def test_width_one_output_uses_standard_sum(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    y = _call(lambda v: v + 3, x, _sds((3, 2, 1), dtype))
    npt.assert_allclose(y, (x + 3).sum(-1, keepdims=True), rtol=1e-5, atol=1e-5)


def test_visible_reduction(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    y = _call(lambda v: jnp.sum(v), x, _sds((3, 4), dtype))
    npt.assert_allclose(y, x.sum(axis=1), rtol=1e-5, atol=1e-5)


def test_visible_pad(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    y = _call(lambda value: jnp.pad(value, ((1, 2),)), x, _sds((3, 5, 4), dtype))
    npt.assert_array_equal(y, jnp.pad(x, ((0, 0), (1, 2), (0, 0))))


def test_output_configuration_broadcasts_over_inferred_pytree(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    result = loom.call(
        lambda value: {"wide": value + 1, "sum": jnp.sum(value)},
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains={"wide": domains["n"], "sum": domains[:]},
        reduce_lanes={"wide": False, "sum": True},
        output_dtypes={"wide": None, "sum": jnp.float16},
    )
    assert result["wide"].shape == x.shape
    assert result["wide"].dtype == x.dtype
    assert result["sum"].shape == (1, 1)
    assert result["sum"].dtype == jnp.float16
    npt.assert_array_equal(result["wide"], x + 1)
    npt.assert_array_equal(result["sum"][0, 0], x.sum().astype(jnp.float16))


def test_exact_zero_gate(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4) - 12
    y = _call(lambda v: jnp.where(v > 0, v * v, 0), x, _sds(x.shape, dtype))
    npt.assert_allclose(y, jnp.where(x > 0, x * x, 0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("hidden_width", [1, 5, 32, 64], ids=lambda u: f"u{u}")
def test_exact_zero_gate_preserves_jvp_and_reverse_shapes(device, dtype, hidden_width):
    x = jnp.linspace(-1, 1, 2 * hidden_width, dtype=dtype).reshape(2, 1, hidden_width)

    def pointwise(value):
        return jnp.where(value > 0, value * value, 0)

    def call(value):
        return _call(
            pointwise,
            value,
            _sds(value.shape, dtype),
            u=hidden_width,
            compile_options=Compile(),
        )

    expected = pointwise(x)
    result = call(x)
    npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    tangent = jnp.ones_like(x)
    got_jvp = jax.jvp(call, (x,), (tangent,))[1]
    expected_jvp = jax.jvp(pointwise, (x,), (tangent,))[1]
    npt.assert_allclose(got_jvp, expected_jvp, rtol=2e-5, atol=2e-5)

    got_grad = jax.grad(lambda value: jnp.sum(call(value)))(x)
    expected_grad = jax.grad(lambda value: jnp.sum(pointwise(value)))(x)
    assert got_grad.shape == x.shape
    npt.assert_allclose(got_grad, expected_grad, rtol=2e-5, atol=2e-5)


def test_float_extrema(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4) - 12
    lower = jnp.asarray(-3, dtype=dtype)
    upper = jnp.asarray(5, dtype=dtype)
    maximum, minimum = loom.call(
        lambda value: (jnp.maximum(value, lower), jnp.minimum(value, upper)),
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains=domains["n"],
    )
    npt.assert_array_equal(maximum, jnp.maximum(x, lower))
    npt.assert_array_equal(minimum, jnp.minimum(x, upper))


def test_u_is_required_when_all_widths_are_one(device):
    x = jnp.ones((2, 3, 1), jnp.float32)
    with pytest.raises(ValueError, match="pass u explicitly"):
        _call(lambda v: v, x, _sds(x.shape, x.dtype))


def test_captured_array(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)
    scale = jnp.asarray([2, 3], dtype=dtype)
    y = _call(lambda v: v * scale, x, _sds(x.shape, dtype))
    npt.assert_allclose(y, x * scale[..., None], rtol=1e-5, atol=1e-5)


def test_vmap(device, dtype):
    x = jnp.arange(48, dtype=dtype).reshape(2, 3, 2, 4)

    def one(value):
        return _call(lambda v: v * 2, value, _sds(value.shape, dtype))

    npt.assert_allclose(jax.vmap(one)(x), x * 2, rtol=1e-5, atol=1e-5)


def test_dce_unused_output(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4)

    def one(value):
        _, square = loom.call(
            lambda v: (v + 1, v * v),
            value,
            domain_graph=DomainGraph({"n": value.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )
        return square

    npt.assert_allclose(jax.jit(one)(x), x * x, rtol=1e-5, atol=1e-5)


def test_public_call_uses_only_jax_for_mixed_static_metadata(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    result = loom.call(
        lambda value: value + 1,
        x,
        domain_graph=DomainGraph({"n": 2}),
        input_domains=domains["n"],
        output_domains=domains["n"],
    )
    npt.assert_array_equal(result, x + 1)


def test_scalar_predicate_selects_across_wider_arms(device, dtype):
    """A one-element predicate governs every element of the arms it chooses.

    This is what a compact-support cutoff is: one radius decides a whole path's
    worth of terms. The predicate is computed from one component and the arms
    span all of them, so the select has to broadcast the predicate rather than
    refuse the shapes. Every math policy has to agree, because only `algebraic`
    can prove the predicate away and the others must still compile it.
    """
    x = jnp.linspace(-1.0, 1.0, 24, dtype=dtype).reshape(3, 2, 4)

    def cutoff(value):
        # value[0] is one component; the result spans all of them, so the
        # comparison is a single element against a wider pair of arms.
        inside = value[0] > 0
        return jnp.where(inside, value * value, jnp.zeros_like(value))

    y = _call(
        cutoff,
        x,
        _sds(x.shape, dtype),
        compile_options=Compile(),
    )
    expected = jnp.where(x[:, :1] > 0, x * x, jnp.zeros_like(x))
    npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


def test_search_memory_budget_transpose_and_cache_key():
    for budget in (None, 0, 128 << 20):
        options = Compile(search_memory_budget_bytes=budget)
        assert options.advance_transpose().search_memory_budget_bytes == budget
        assert pickle.loads(pickle.dumps(options)) == options
        assert hash(copy.deepcopy(options)) == hash(options)
    assert Compile(search_memory_budget_bytes=128 << 20) != Compile(search_memory_budget_bytes=None)


@pytest.mark.parametrize("budget", [True, 1.5, "1024"])
def test_search_memory_budget_rejects_nonintegers(budget):
    with pytest.raises(TypeError, match="search_memory_budget_bytes"):
        Compile(search_memory_budget_bytes=budget)


@pytest.mark.parametrize("budget", [-1, 1 << 64])
def test_search_memory_budget_rejects_values_outside_u64(budget):
    with pytest.raises(ValueError, match="search_memory_budget_bytes"):
        Compile(search_memory_budget_bytes=budget)
