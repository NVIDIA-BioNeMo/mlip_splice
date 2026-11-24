# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import loom
import numpy as np
import numpy.testing as npt
import pytest
from loom import DomainGraph, domains
from loom._src.primitive.options import Compile

from _support import _call, _sds

pytestmark = pytest.mark.capped_autotune


def _loom_compile_options(closed_jaxpr):
    found = []
    for equation in closed_jaxpr.jaxpr.eqns:
        if equation.primitive.name == "loom":
            found.append((equation.params["ad_depth"], equation.params["options"]))
        for value in equation.params.values():
            if hasattr(value, "jaxpr"):
                found.extend(_loom_compile_options(value))
    return found


def _budgeted_call(program, value):
    return loom.call(
        program,
        value,
        domain_graph=DomainGraph({"n": 1}),
        input_domains=domains["n"],
        output_domains=domains["n"],
        u=1,
        compile_options=Compile(
            search_budget_ms={0: 20.0, 1: 10.0, 2: 1.0},
            shortlist_budget_ms={0: 30.0, 1: 15.0, 2: 2.0},
        ),
    )


def test_rule_indexed_budgets_advance_by_transpose_order():
    x = jnp.ones((1, 1, 1), jnp.float32)

    def loss(value):
        return _budgeted_call(lambda visible: visible * visible, value).sum()

    first = _loom_compile_options(jax.make_jaxpr(jax.grad(loss))(x))
    second = _loom_compile_options(
        jax.make_jaxpr(jax.grad(lambda value: jax.grad(loss)(value).sum()))(x)
    )

    assert any(depth >= 2 for depth, _ in first)
    assert all(
        options.current_search_budget_ms == (10.0 if depth >= 2 else 20.0)
        and options.current_shortlist_budget_ms == (15.0 if depth >= 2 else 30.0)
        for depth, options in first
    )
    assert any(depth >= 4 for depth, _ in second)
    assert all(
        options.current_search_budget_ms == (1.0 if depth >= 4 else 10.0 if depth >= 2 else 20.0)
        and options.current_shortlist_budget_ms
        == (2.0 if depth >= 4 else 15.0 if depth >= 2 else 30.0)
        for depth, options in second
    )


def test_nested_forward_jvps_do_not_advance_transpose_budget():
    x = jnp.ones((1, 1, 1), jnp.float32)
    tangent = jnp.ones_like(x)

    def square(value):
        return _budgeted_call(lambda visible: visible * visible, value)

    def second_jvp(value):
        def first_jvp(primal):
            return jax.jvp(square, (primal,), (tangent,))[1]

        return jax.jvp(first_jvp, (value,), (tangent,))[1]

    found = _loom_compile_options(jax.make_jaxpr(second_jvp)(x))

    assert any(depth >= 2 for depth, _ in found)
    assert all(options.current_search_budget_ms == 20.0 for _, options in found)


def test_linear_transpose_advances_budget_once():
    x = jnp.ones((1, 1, 1), jnp.float32)

    def linear(value):
        return _budgeted_call(lambda visible: 2 * visible, value)

    def transposed(cotangent):
        return jax.linear_transpose(linear, x)(cotangent)[0]

    found = _loom_compile_options(jax.make_jaxpr(transposed)(jnp.ones_like(x)))

    assert found
    assert all(options.current_search_budget_ms == 10.0 for _, options in found)


@pytest.mark.parametrize("mode", ["forward_over_reverse", "reverse_over_forward"])
def test_mixed_autodiff_advances_only_for_actual_transpose(mode):
    x = jnp.ones((1, 1, 1), jnp.float32)
    tangent = jnp.ones_like(x)

    def square(value):
        return _budgeted_call(lambda visible: visible * visible, value)

    if mode == "forward_over_reverse":

        def differentiated(value):
            def gradient(primal):
                return jax.grad(lambda v: square(v).sum())(primal)

            return jax.jvp(gradient, (value,), (tangent,))[1]

    else:

        def differentiated(value):
            def tangent_sum(primal):
                return jax.jvp(square, (primal,), (tangent,))[1].sum()

            return jax.grad(tangent_sum)(value)

    found = _loom_compile_options(jax.make_jaxpr(differentiated)(x))
    budgets = {options.current_search_budget_ms for _, options in found}

    assert 10.0 in budgets
    assert budgets <= {20.0, 10.0}


def test_compile_advance_transpose_advances_rule_budgets():
    c0 = Compile(
        search_budget_ms={0: 2000, 1: 1600, 2: 200},
        shortlist_budget_ms={0: 2000, 1: 1600, 2: 200},
    )
    c1 = c0.advance_transpose()
    c2 = c1.advance_transpose()

    # The rule-indexed mapping is unchanged data; advancing raises the depth that
    # reads it. The observable budget at each depth is what the schedule promises.
    assert c0.search_budget_ms == {0: 2000.0, 1: 1600.0, 2: 200.0}
    assert (c0.transpose_depth, c1.transpose_depth, c2.transpose_depth) == (0, 1, 2)
    assert c1.search_budget_ms == c0.search_budget_ms
    assert c2.search_budget_ms == c0.search_budget_ms
    assert c0.current_search_budget_ms == 2000.0
    assert c1.current_search_budget_ms == 1600.0
    assert c1.current_shortlist_budget_ms == 1600.0
    assert c2.current_search_budget_ms == 200.0


def test_compile_advance_transpose_inherits_missing_rule_budget():
    options = Compile(search_budget_ms={0: 20, 2: 1})

    assert options.advance_transpose().current_search_budget_ms == 20.0
    assert options.advance_transpose().advance_transpose().current_search_budget_ms == 1.0


def test_jvp_and_grad(device, dtype):
    x = jnp.arange(24, dtype=dtype).reshape(3, 2, 4) / 10

    def fn(value):
        return _call(lambda v: v * v, value, _sds(value.shape, dtype))

    tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))[1]
    grad = jax.grad(lambda value: fn(value).sum())(x)
    npt.assert_allclose(tangent, 2 * x, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(grad, 2 * x, rtol=1e-5, atol=1e-5)


def test_output_storage_cast_is_differentiated(device):
    x = jnp.linspace(0.1, 1.0, 24, dtype=jnp.float32).reshape(3, 2, 4)

    def fn(value):
        return loom.call(
            lambda visible: visible * visible,
            value,
            domain_graph=DomainGraph({"n": value.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
            output_dtypes=jnp.float16,
        )

    tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))[1]
    grad = jax.grad(lambda value: fn(value).sum())(x)
    assert tangent.dtype == jnp.float16
    npt.assert_allclose(tangent, (2 * x).astype(jnp.float16), rtol=2e-3)
    npt.assert_allclose(grad, 2 * x, rtol=2e-3)


def test_mixed_discrete_jvp_uses_float0(device):
    values = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 4) / 10
    indices = jnp.arange(8, dtype=jnp.int32).reshape(2, 1, 4)
    mask = indices % 2 == 0

    def run(value, index, predicate):
        return loom.call(
            lambda args: (
                jnp.where(
                    args[2],
                    args[0] * (args[1].astype(jnp.float32) + 1),
                    -args[0],
                ),
                args[1] + 1,
                args[2],
            ),
            (value, index, predicate),
            domain_graph=DomainGraph({"n": value.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )

    discrete_index_tangent = np.zeros(indices.shape, dtype=jax.dtypes.float0)
    discrete_mask_tangent = np.zeros(mask.shape, dtype=jax.dtypes.float0)
    primal, tangent = jax.jvp(
        run,
        (values, indices, mask),
        (jnp.ones_like(values), discrete_index_tangent, discrete_mask_tangent),
    )
    factor = indices.astype(jnp.float32) + 1
    npt.assert_array_equal(primal[0], jnp.where(mask, values * factor, -values))
    npt.assert_array_equal(primal[1], indices + 1)
    npt.assert_array_equal(primal[2], mask)
    npt.assert_array_equal(tangent[0], jnp.where(mask, factor, -1))
    assert tangent[1].dtype == jax.dtypes.float0
    assert tangent[2].dtype == jax.dtypes.float0


def test_discrete_only_jvp_is_zero(device):
    indices = jnp.arange(8, dtype=jnp.int32).reshape(2, 1, 4)

    def run(index):
        return loom.call(
            lambda value: (value.astype(jnp.float32), value + 1),
            index,
            domain_graph=DomainGraph({"n": index.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )

    discrete_tangent = np.zeros(indices.shape, dtype=jax.dtypes.float0)
    primal, tangent = jax.jvp(run, (indices,), (discrete_tangent,))
    npt.assert_array_equal(primal[0], indices.astype(jnp.float32))
    npt.assert_array_equal(primal[1], indices + 1)
    npt.assert_array_equal(tangent[0], jnp.zeros(indices.shape, jnp.float32))
    assert tangent[1].dtype == jax.dtypes.float0


def test_mixed_discrete_transpose_preserves_primals(device):
    values = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 4) / 10
    indices = jnp.arange(8, dtype=jnp.int32).reshape(2, 1, 4)
    mask = indices % 2 == 0

    def loss(value, index, predicate):
        result = loom.call(
            lambda args: jnp.where(
                args[2],
                args[0] * (args[1].astype(jnp.float32) + 1),
                -args[0],
            ),
            (value, index, predicate),
            domain_graph=DomainGraph({"n": value.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
        )
        return result.sum()

    value_grad, index_grad, mask_grad = jax.grad(loss, argnums=(0, 1, 2), allow_int=True)(
        values, indices, mask
    )
    factor = indices.astype(jnp.float32) + 1
    npt.assert_array_equal(value_grad, jnp.where(mask, factor, -1))
    assert index_grad.dtype == jax.dtypes.float0
    assert mask_grad.dtype == jax.dtypes.float0


def test_grad_width_one_counts_every_lane(device, dtype):
    u = 8
    x = jnp.arange(6, dtype=dtype).reshape(3, 2, 1) / 10

    def loss(value):
        result = _call(lambda v: v * v, value, _sds(value.shape, dtype), u=u)
        return result.sum()

    npt.assert_allclose(jax.grad(loss)(x), 2 * u * x, rtol=1e-5, atol=1e-5)


def test_aot_ambient_budget_resolves_at_transpose_depth(monkeypatch):
    """An `aot` mapping budget varies with the consuming program's transpose depth,
    exactly as a per-call mapping does — the unified indexing rule.

    Exercises the ambient path in `compile_recipe`/`run_recipe` directly with a stub
    forge, so it is deterministic and needs no device: the resolved `search_budget_ms`
    and `shortlist_budget_ms` must follow the floor-indexed schedule at each depth.
    """
    from loom import aot
    from loom._src.runtime import recipes

    monkeypatch.setattr(recipes, "_device_fingerprint", lambda: ("source-test", 100))
    seen_compile = []

    def fake_compile(*args, search_budget_ms=None, **kwargs):
        seen_compile.append(search_budget_ms)
        return b"recipe"

    schedule = {0: 2000.0, 1: 1600.0, 2: 200.0}
    with aot(Compile(search_budget_ms=schedule, shortlist_budget_ms=schedule)):
        for depth in (0, 1, 2, 3):
            seen_compile.clear()
            recipes.compile_recipe(
                fake_compile, "prog", depth, compile_options=Compile(transpose_depth=depth)
            )
            expected = {0: 2000.0, 1: 1600.0, 2: 200.0, 3: 200.0}[depth]
            assert seen_compile == [expected], (depth, seen_compile)

    # A call's own budget still wins over the ambient one, at any depth.
    with aot(Compile(search_budget_ms=schedule)):
        seen_compile.clear()
        recipes.compile_recipe(
            fake_compile,
            "prog",
            9,
            compile_options=Compile(search_budget_ms=5.0, transpose_depth=2),
        )
        assert seen_compile == [5.0]


def test_aot_ambient_scalar_budget_ignores_transpose_depth(monkeypatch):
    """A scalar `aot` budget is depth-independent — the mapping is opt-in."""
    from loom import aot
    from loom._src.runtime import recipes

    monkeypatch.setattr(recipes, "_device_fingerprint", lambda: ("source-test", 100))
    seen = []

    def fake_compile(*args, search_budget_ms=None, **kwargs):
        seen.append(search_budget_ms)
        return b"recipe"

    with aot(Compile(search_budget_ms=100.0)):
        for depth in (0, 1, 2):
            seen.clear()
            recipes.compile_recipe(
                fake_compile, "prog", depth, compile_options=Compile(transpose_depth=depth)
            )
            assert seen == [100.0]
