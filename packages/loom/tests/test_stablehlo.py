# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction

import algebraic_real
import algebraic_real.jax
import jax
import jax.numpy as jnp
import loom
import numpy as np
import numpy.testing as npt
import pytest
import splice
from loom import _native as _engine
from loom._src.primitive.stablehlo import lower_module

from _support import _call, _sds
from marks import gpu_only

pytestmark = pytest.mark.capped_autotune


def test_indexed_reduction_reaches_existing_xir_ops():
    factors = jnp.asarray([[0, 0], [0, 1], [1, 1]], jnp.int32)
    monomials = jnp.asarray([0, 1, 1, 2], jnp.int32)
    outputs = jnp.asarray([0, 0, 1, 1], jnp.int32)

    def program(x):
        products = jnp.prod(x[factors], axis=1)
        return jnp.zeros(2, x.dtype).at[outputs].add(products[monomials])

    lowered = jax.jit(program).lower(_sds((2,), jnp.float32)).compiler_ir(dialect="stablehlo")
    module = lower_module(lowered, _engine.XirBuilder())
    text = module.text()

    assert "gather" in text
    assert "reduce" in text
    assert "scatter" in text


def test_integer_not_reaches_xir_builder():
    lowered = jax.jit(jnp.bitwise_not).lower(_sds((3,), jnp.int32)).compiler_ir(dialect="stablehlo")
    module = lower_module(lowered, _engine.XirBuilder())
    assert "not" in module.operation_names(0)


def test_exact_constant_reaches_xir_builder():
    class Builder:
        def begin_func(self, name):
            assert name == "main"

        def exact_constant(self, dtype, dims, data, terms):
            self.constant = (dtype, dims, data, terms)
            return 0

        def constant(self, dtype, dims, data):
            return 1

        def end_func(self, results):
            assert results == [0]
            return 0

        def finish(self, entry):
            assert entry == 0
            return self

    def program():
        return algebraic_real.jax.exact(
            [algebraic_real.pow(2, Fraction(1, 2)), 1], dtype=jnp.float32
        )

    lowered = jax.jit(program).lower().compiler_ir(dialect="stablehlo")
    stablehlo = str(lowered)
    assert 'stablehlo.composite "algebraic_real.exact_constant"' in stablehlo
    assert "mhlo.frontend_attributes" not in stablehlo
    npt.assert_array_equal(jax.jit(program)(), program())
    builder = lower_module(lowered, Builder())

    dtype, dims, data, terms = builder.constant
    assert dtype == "f32"
    assert dims == [2]
    npt.assert_array_equal(np.frombuffer(data, dtype=np.float32), program())
    assert terms == [
        ["m", [["1", "1", "2"]]],
        ["m", [["1", "1", "1"]]],
    ]


class _Constants:
    """A builder that keeps the one constant the module holds."""

    def begin_func(self, name):
        assert name == "main"

    def constant(self, dtype, dims, data):
        self.held = (dtype, dims, data)
        return 0

    def end_func(self, results):
        assert results == [0]
        return 0

    def finish(self, entry):
        assert entry == 0
        return self


@pytest.mark.parametrize(
    ("table", "written"),
    [
        # A splat, which MLIR writes once.
        (np.full(64, 2.5), "dense<2"),
        # A short literal, which it writes out in full.
        (np.arange(3) * 0.5 - 1.0, "dense<["),
        # A literal holding what no decimal says, which it writes in hex.
        (np.asarray([1.5, np.inf, -np.inf, np.nan]), "0x7F80"),
        # A coefficient table, which past a size it chooses it writes as the
        # raw little-endian elements in hex — the form every real table takes.
        (np.arange(105) * 0.5 - 1.0, 'dense<"0x'),
    ],
    ids=["splat", "literal", "non-finite", "hex"],
)
def test_a_bf16_constant_reaches_the_builder_exactly(table, written):
    table = np.asarray(table, dtype=jnp.bfloat16)
    lowered = jax.jit(lambda: jnp.asarray(table)).lower().compiler_ir(dialect="stablehlo")
    # The form is part of what this covers: each is read a different way, and a
    # case that stopped reaching its form would be testing another one twice.
    assert written in str(lowered)

    dtype, dims, data = lower_module(lowered, _Constants()).held

    assert (dtype, dims) == ("bf16", [len(table)])
    # Bit for bit: the elements are carried across, not re-rounded through a
    # decimal on the way.
    npt.assert_array_equal(
        np.frombuffer(data, dtype=jnp.bfloat16).view(np.uint16),
        table.view(np.uint16),
    )


def _reverses_an_axis(x):
    """An operation Loom has no method for. Reversal is a permutation a circuit
    expresses, but nothing here builds it, so the frontend says which operation it
    could not take."""
    return x[::-1]


def test_unsupported_stablehlo_diagnostic_includes_shapes():
    class Builder:
        """Enough of a builder to reach the first operation it has no method for:
        the diagnostic is about that operation, so everything before it has to
        succeed."""

        def __init__(self):
            self.values = 0

        def _value(self):
            self.values += 1
            return self.values

        def begin_func(self, name):
            assert name == "main"

        def param(self, dtype, dims):
            assert (dtype, dims) == ("f32", [3])
            return self._value()

        def constant(self, dtype, dims, data):
            return self._value()

        def end_func(self, results):
            raise AssertionError(f"unexpected supported results {results}")

    lowered = (
        jax.jit(_reverses_an_axis).lower(_sds((3,), jnp.float32)).compiler_ir(dialect="stablehlo")
    )
    with pytest.raises(NotImplementedError, match=r"stablehlo\.reverse") as exc_info:
        lower_module(lowered, Builder())
    message = str(exc_info.value)
    assert "operand shapes=[[3]], result shapes=[[3]]" in message


@gpu_only
def test_unsupported_stablehlo_fails_loudly(device):
    x = jnp.ones((2, 3, 4), jnp.float32)
    with pytest.raises(NotImplementedError, match=r"stablehlo\.reverse") as exc_info:
        _call(_reverses_an_axis, x, _sds(x.shape, x.dtype))
    message = str(exc_info.value)
    assert "operand shapes=[[3]], result shapes=[[3]]" in message
