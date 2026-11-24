# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric contraction over ``e3``'s path basis."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3 import O3, Irreps
from loom import IndexMap
from splice.graph import build_graph
from splice.ops.sym_contr import SymmetricContraction

IRREPS_IN = Irreps(O3, "0e+1o+2e")
IRREPS_OUT = Irreps(O3, "0e+1o")


def _prec_pdt():
    """Precision + param_dtype following the effective fp dtype: fp64 when x64 is
    active (requested at import), fp32 in the multi-file fallback where jax already
    inited — jnp.float64 arrays are really float32 there, and the op's strict
    activation-dtype contract must match (the tol branches already handle fp32)."""
    x64 = jax.config.read("jax_enable_x64")
    return ("f64", jnp.float64) if x64 else ("f32", jnp.float32)


def _make(*, num_features=1, num_species=1, key=0, **kwargs):
    prec, pdt = _prec_pdt()
    return SymmetricContraction(
        IRREPS_IN,
        IRREPS_OUT,
        correlation=3,
        num_species=num_species,
        num_features=num_features,
        precision=prec,
        param_dtype=pdt,
        key=jax.random.key(key),
        **kwargs,
    )


def test_weights_use_the_minimal_path_basis():
    """One weight per path: 17 for 0e+1o+2e -> 0e+1o at correlation 3, not the
    41-dimensional over-complete coupling-solution basis."""
    assert _make().w.shape[1] == 17


def test_mace_init_scales_each_degree_output_path_block_by_its_count():
    normal = _make(num_features=4, num_species=3, key=31, init="normal")
    mace = _make(num_features=4, num_species=3, key=31, init="mace")
    path_scales = np.asarray(
        [1.0 / count for counts in normal.path_counts for _, count in counts for _ in range(count)]
    )
    np.testing.assert_allclose(
        np.asarray(mace.w),
        np.asarray(normal.w) * path_scales[None, :, None],
        rtol=1e-7,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="init must be"):
        _make(init="xavier_uniform")


def test_scale_dict_is_the_float_knob_per_irrep():
    """SC is linear in w, so `scale` as a dict is the same knob as the float,
    just per output irrep: uniform dict == float, and a per-irrep entry scales
    only its own irrep (missing entries default to 1)."""
    N, NS, MUL = 6, 3, 8

    def make(scale):
        prec, pdt = _prec_pdt()
        return SymmetricContraction(
            IRREPS_IN,
            IRREPS_OUT,
            correlation=3,
            num_species=NS,
            num_features=MUL,
            scale=scale,
            precision=prec,
            param_dtype=pdt,
            key=jax.random.key(11),
        )

    nf = {
        ir: jax.random.normal(jax.random.key(12), (N, ir.dim, MUL), jnp.float64)
        for _, ir in IRREPS_IN
    }
    species = jax.random.randint(jax.random.key(13), (N,), 0, NS)
    s, r = jax.random.randint(jax.random.key(14), (2, 16), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )

    y_float = make(0.7)(nf, g)
    y_uniform = make({"0e": 0.7, "1o": 0.7})(nf, g)
    y_mixed = make({"0e": 1.4})(nf, g)  # 1o missing -> 1.0
    y_unit = make(1.0)(nf, g)
    by_str = {str(ir): ir for ir in y_float}
    for ir in y_float:
        np.testing.assert_allclose(np.asarray(y_uniform[ir]), np.asarray(y_float[ir]), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(y_mixed[by_str["0e"]]), 1.4 * np.asarray(y_unit[by_str["0e"]]), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(y_mixed[by_str["1o"]]), np.asarray(y_unit[by_str["1o"]]), rtol=1e-5
    )


def test_native_output_is_finite_and_stable():
    N, NS, MUL = 6, 3, 8
    sc = _make(num_features=MUL, num_species=NS)
    nf = {
        ir: jax.random.normal(jax.random.key(7), (N, ir.dim, MUL), jnp.float64)
        for _, ir in IRREPS_IN
    }
    species = jax.random.randint(jax.random.key(8), (N,), 0, NS)
    s, r = jax.random.randint(jax.random.key(9), (2, 16), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )
    y1, y2 = sc(nf, g), sc(nf, g)
    for ir in y1:
        first, second = np.asarray(y1[ir]), np.asarray(y2[ir])
        assert np.all(np.isfinite(first))
        # Not bit-equal: the first compile in a process may tune a different
        # reduction order than the second, and Loom reassociates exactly, so
        # the two agree to the rounding of the block they belong to.
        np.testing.assert_array_less(np.abs(second - first), 8 * np.spacing(np.abs(first).max()))
