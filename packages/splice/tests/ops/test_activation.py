# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scalar / gated activation, partial residual add, and identity passthrough."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3 import O3, Irreps
from splice.config import build_chain, load_config
from splice.ops.activation import (
    IrrepsAdd,
    IrrepsIdentity,
    ScalarGatedActivation,
    normalize_function,
)

IRREPS = Irreps(O3, "8x0e+8x1o+8x2e")


def _features():
    keys = jax.random.split(jax.random.key(1), len(IRREPS))
    return {
        ir: jax.random.normal(key, (4, ir.dim, mul), jnp.float32)
        for key, (mul, ir) in zip(keys, IRREPS, strict=True)
    }


def test_every_irrep_is_gated_by_a_sigmoid_of_scalars():
    """Non-scalars are gated, not passed through, and scalars are self-gated."""
    op = ScalarGatedActivation(IRREPS)
    x = _features()
    y = op(x)
    scalar = O3.parse("0e")
    gates = jax.nn.sigmoid(x[scalar])
    # The scalar branch is the same gate applied to itself: silu(x) == x*sigmoid(x).
    np.testing.assert_allclose(y[scalar], x[scalar] * gates, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y[scalar], jax.nn.silu(x[scalar]))
    for _, ir in IRREPS:
        if not ir.is_scalar():
            np.testing.assert_allclose(y[ir], x[ir] * gates[..., : x[ir].shape[-1]])
            # and it is genuinely not a passthrough
            assert not np.allclose(np.asarray(y[ir]), np.asarray(x[ir]))


def test_scalar_gated_activation_can_normalize_second_moments():
    op = ScalarGatedActivation(IRREPS, normalize=True)
    x = _features()
    y = op(x)
    scalar = O3.parse("0e")
    np.testing.assert_allclose(y[scalar], normalize_function(jax.nn.silu)(x[scalar]))
    gates = normalize_function(jax.nn.sigmoid)(x[scalar])
    for _, ir in IRREPS:
        if not ir.is_scalar():
            np.testing.assert_allclose(y[ir], x[ir] * gates[..., : x[ir].shape[-1]])


def test_yaml_defaults_activation_does_not_reach_the_gated_op(tmp_path):
    path = tmp_path / "gated.yaml"
    path.write_text(
        """
defaults: {param_dtype: f32, precision: f32, activation: relu}
steps:
  - op: scalar_gated_activation
    params: {irreps_in: 8x0e+8x1o}
"""
    )
    cfg = load_config(path)
    assert "activation" not in cfg[0]["params"]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.bfloat16])
def test_gate_parameter_dtype_and_gradients(dtype):
    with jax.enable_x64():
        op = ScalarGatedActivation(IRREPS, param_dtype=dtype)
        x = {ir: value.astype(dtype) for ir, value in _features().items()}
        y = eqx.filter_jit(op)(x)
        assert all(value.dtype == dtype for value in y.values())
        grads = eqx.filter_grad(lambda model: sum(jnp.sum(v) for v in model(x).values()))(op)
        for field in ("gate_scale", "gate_bias"):
            assert all(value.dtype == dtype for value in getattr(op, field).values())
            for value in getattr(grads, field).values():
                assert value.dtype == dtype
                assert bool(jnp.all(jnp.isfinite(value)))
                assert bool(jnp.any(value != 0))


def test_fp64_gate_preserves_scalar_precision():
    with jax.enable_x64():
        op = ScalarGatedActivation(IRREPS, param_dtype=jnp.float64)
        x = {ir: value.astype(jnp.float64) for ir, value in _features().items()}
        scalar = O3.parse("0e")
        x[scalar] = x[scalar] + 1e-9
        y = eqx.filter_jit(op)(x)
        for _, ir in IRREPS:
            if not ir.is_scalar():
                expected = x[ir] * jax.nn.sigmoid(x[scalar])
                np.testing.assert_allclose(y[ir], expected, rtol=1e-14, atol=1e-14)


def test_yaml_gate_receives_default_parameter_dtype(tmp_path):
    path = tmp_path / "gated_bf16.yaml"
    path.write_text(
        """
defaults: {param_dtype: bf16, precision: f32}
steps:
  - op: scalar_gated_activation
    params: {irreps_in: 8x0e+8x1o}
"""
    )
    chain = build_chain(load_config(path), key=jax.random.key(0))
    params = eqx.filter(chain, eqx.is_array)
    assert jax.tree.leaves(params)
    assert all(value.dtype == jnp.bfloat16 for value in jax.tree.leaves(params))


def test_partial_irrep_residual_add_and_identity():
    x = _features()
    scalar = O3.parse("0e")
    assert IrrepsIdentity()(x) is x
    residual = {scalar: jnp.ones_like(x[scalar])}
    with pytest.raises(AssertionError):
        IrrepsAdd()(x, residual)
    y = IrrepsAdd(allow_partial=True)(x, residual)
    np.testing.assert_array_equal(y[scalar], x[scalar] + 1.0)
    for ir, value in x.items():
        if ir != scalar:
            np.testing.assert_array_equal(y[ir], value)
