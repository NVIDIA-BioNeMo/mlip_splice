# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialization and scaling contracts for IrrepsLinear."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3 import O3, Irreps
from splice.ops.embedding import SpeciesEmbedding
from splice.ops.linear import IrrepsLinear
from splice.optim import ADAM, MUON, _op_label_params

IRREPS = Irreps(O3, "8x0e+8x1o")


def _features(irreps=IRREPS, n=5, seed=1):
    keys = jax.random.split(jax.random.key(seed), len(irreps))
    return {
        ir: jax.random.normal(key, (n, ir.dim, mul), jnp.float32)
        for key, (mul, ir) in zip(keys, irreps, strict=True)
    }


def _linear(**kwargs) -> IrrepsLinear:
    params = dict(
        irreps_in=IRREPS,
        irreps_out=IRREPS,
        precision="f32",
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )
    params.update(kwargs)
    return IrrepsLinear(**params)


def test_default_scaled_normal_matches_historical_runtime_fan_in():
    op = _linear()
    assert op.weight_scaling is True
    assert op.init == "normal"
    assert op.readout_scaling == "mup"
    x = _features()
    y = op(x)
    for ir, w in op.w.items():
        expected = jnp.einsum(
            "uv,...iu->...iv",
            w,
            x[op._ir_map[ir]],
            precision=op.precision.xla_dot,
        ) / math.sqrt(w.shape[0])
        np.testing.assert_allclose(y[op._ir_map[ir]], expected, rtol=1e-6, atol=1e-6)


def test_identity_init_is_effective_skip_with_scaled_weights():
    op = _linear(init="identity")
    x = _features()
    y = op(x)
    for ir, value in x.items():
        np.testing.assert_allclose(y[ir], value, rtol=1e-6, atol=1e-6)
    for w in op.w.values():
        np.testing.assert_allclose(w, np.eye(w.shape[0], dtype=np.float32) * math.sqrt(w.shape[0]))


def test_direct_identity_stores_unscaled_identity():
    op = _linear(weight_scaling=False, init="identity")
    x = _features()
    y = op(x)
    for ir, value in x.items():
        np.testing.assert_array_equal(y[ir], value)
    for w in op.w.values():
        np.testing.assert_array_equal(w, np.eye(w.shape[0], dtype=np.float32))


def test_direct_fan_in_normal_matches_scaled_normal_at_init():
    scaled = _linear(init="normal", weight_scaling=True, key=jax.random.key(7))
    direct = _linear(init="fan_in_normal", weight_scaling=False, key=jax.random.key(7))
    x = _features()
    for ir in x:
        np.testing.assert_allclose(direct(x)[ir], scaled(x)[ir], rtol=1e-6, atol=1e-6)


def test_identity_init_rejects_non_square_channels():
    with pytest.raises(ValueError, match="fan_in == fan_out"):
        _linear(irreps_out=Irreps(O3, "16x0e+8x1o"), init="identity")


def test_direct_readout_rejects_runtime_mup_factor():
    with pytest.raises(ValueError, match="readout_scaling"):
        _linear(is_readout=True, weight_scaling=False, init="xavier_uniform")


def test_mup_readout_applies_extra_fan_in_factor():
    op = _linear(is_readout=True, irreps_out=Irreps(O3, "1x0e+1x1o"))
    assert all(math.isclose(scale, 1.0 / 8.0) for scale in op._runtime_scale.values())
    x = _features()
    y = op(x)
    for key, w in op.w.items():
        ir = op._ir_map[key]
        expected = (
            jnp.einsum(
                "uv,...iu->...iv",
                w,
                x[ir],
                precision=op.precision.xla_dot,
            )
            / 8.0
        )
        np.testing.assert_allclose(y[ir], expected, rtol=1e-6, atol=1e-6)


def test_optimizer_defaults_follow_readout_flag():
    hidden = _linear()
    readout = _linear(is_readout=True, irreps_out=Irreps(O3, "1x0e+1x1o"))

    assert hidden.optimizer == "muon"
    assert readout.optimizer == "adam"
    assert set(_op_label_params(hidden).w.values()) == {MUON}
    assert set(_op_label_params(readout).w.values()) == {ADAM}


def test_hidden_linear_can_opt_into_adam():
    op = _linear(optimizer="adam")

    assert op.optimizer == "adam"
    assert set(_op_label_params(op).w.values()) == {ADAM}


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        _linear(optimizer="unknown")


def test_standard_readout_skips_extra_factor():
    op = _linear(
        is_readout=True,
        readout_scaling="standard",
        irreps_out=Irreps(O3, "1x0e+1x1o"),
    )
    assert all(math.isclose(scale, 1.0 / math.sqrt(8.0)) for scale in op._runtime_scale.values())


def test_array_checkpoint_leaves_match_pre_static_fields(tmp_path):
    """``.eqx`` stores array leaves only; new static fields are not in the file."""
    op = _linear()
    leaves = jax.tree.leaves(op)
    assert all(hasattr(leaf, "dtype") for leaf in leaves)
    assert len(leaves) == len(op.w) + len(op.b)
    path = tmp_path / "linear.eqx"
    eqx.tree_serialise_leaves(path, op)
    loaded = eqx.tree_deserialise_leaves(path, _linear(key=jax.random.key(99)))
    for key, weight in op.w.items():
        np.testing.assert_array_equal(loaded.w[key], weight)
    assert loaded.optimizer == "muon"


def test_direct_embedding_matches_scaled_embedding_at_init():
    species = jnp.asarray([0, 0, 1], jnp.int32)
    scaled = SpeciesEmbedding(
        2, 4, param_dtype=jnp.float32, key=jax.random.key(3), weight_scaling=True, init="normal"
    )
    direct = SpeciesEmbedding(
        2,
        4,
        param_dtype=jnp.float32,
        key=jax.random.key(3),
        weight_scaling=False,
        init="fan_in_normal",
    )
    np.testing.assert_allclose(
        scaled(species)[O3.trivial()],
        direct(species)[O3.trivial()],
        rtol=1e-6,
        atol=1e-6,
    )
