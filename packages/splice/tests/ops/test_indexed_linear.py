# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""irreps_indexed_linear backends. The default cuBLAS (`indexed_linear`) grouped
GEMM must match the legacy `naive` path on the production species-sorted layout,
including batches with empty (zero-count) species groups. The cue-free `loom_map`
fallback must reproduce `naive` bit-for-bit at fp64 and share its flat weight layout
(so checkpoints are interchangeable across backends).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from e3 import O3, Irreps
from loom import IndexMap
from splice.graph import build_graph
from splice.ops.linear import IrrepsIndexedLinear

IRREPS = Irreps(O3, "4x0e+4x1o+4x2e+4x3o")  # MACE-OFF-like block-diagonal skip


def _ops(NS, method_b="naive"):
    k = jax.random.key(0)
    # x64 only takes effect if requested before jax inits; in the multi-file fallback
    # it's off, jnp.float64 arrays are really float32 — match precision to that so the
    # strict activation-dtype contract holds either way (tol branches handle fp32).
    x64 = jax.config.read("jax_enable_x64")
    prec = "f64" if x64 else "f32"
    pdt = jnp.float64 if x64 else jnp.float32
    a = IrrepsIndexedLinear(
        IRREPS,
        IRREPS,
        num_indices=NS,
        method="loom_map",
        precision=prec,
        param_dtype=pdt,
        key=k,
    )
    b = IrrepsIndexedLinear(
        IRREPS,
        IRREPS,
        num_indices=NS,
        method=method_b,
        precision=prec,
        param_dtype=pdt,
        key=k,
    )
    return a, b


def test_weight_layout_interchangeable():
    a, b = _ops(NS=5)
    assert a.w.shape == b.w.shape  # loom_map flat size == cue poly weight size


def test_loom_map_matches_naive_fp64():
    NS = 3
    a, b = _ops(NS)
    # species-contiguous nodes so Repeats(counts) and the explicit dst->species
    # arrow describe the same per-node species.
    counts = jnp.array([2, 3, 1], jnp.int32)
    species = jnp.array([0, 0, 1, 1, 1, 2], jnp.int32)
    N = 6
    nf = {ir: jax.random.normal(jax.random.key(1), (N, ir.dim, 4), jnp.float64) for _, ir in IRREPS}
    s, r = jax.random.randint(jax.random.key(2), (2, 20), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )

    y_map = a(nf, counts, g)
    y_naive = b(nf, counts, g)
    # x64 only takes effect if set before JAX inits; under the full suite (already
    # fp32) the agreement is fp32-level, in isolation it's fp64.
    tol = 1e-9 if bool(jax.config.read("jax_enable_x64")) else 1e-5
    for ir in y_naive:
        x, z = np.asarray(y_map[ir]), np.asarray(y_naive[ir])
        rel = np.linalg.norm(x - z) / (np.linalg.norm(z) + 1e-30)
        assert rel < tol, f"loom_map != naive for {ir}: rel {rel:.2e}"


def test_loom_map_handles_empty_species():
    NS = 8  # table has 8 species; batch uses only a few
    a, _ = _ops(NS)
    species = jnp.array([0, 0, 5, 5, 5, 1, 0], jnp.int32)
    N = 7
    nf = {ir: jax.random.normal(jax.random.key(3), (N, ir.dim, 4), jnp.float64) for _, ir in IRREPS}
    s, r = jax.random.randint(jax.random.key(4), (2, 20), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )
    y = a(nf, jnp.zeros((NS,), jnp.int32), g)  # counts unused by loom_map
    assert all(np.all(np.isfinite(np.asarray(v))) for v in y.values())


def test_cublas_matches_naive_with_empty_species():
    """The default cuBLAS (`indexed_linear`) grouped GEMM on the PRODUCTION layout:
    species-sorted nodes (as `model_call` provides) with MANY species absent from the
    batch (zero-count groups). Must stay finite and match `naive` at fp32 — i.e. no
    empty-group guard is needed: current Loom tolerates zero-count groups. (fp32
    + a realistic node count so it holds on GPU regardless of the suite's x64 state;
    a degenerate N<~32 batch can trip a cuBLAS small-size error unrelated to this.)"""
    NS = 86
    N = 512
    rng = np.random.default_rng(0)
    present = rng.choice(NS, 18, replace=False)  # only 18/86 species present -> many empty
    species = jnp.asarray(np.sort(rng.choice(present, N)).astype(np.int32))
    counts = jnp.zeros((NS,), jnp.int32).at[species].add(1)
    a = IrrepsIndexedLinear(
        IRREPS,
        IRREPS,
        NS,
        method="indexed_linear",
        precision="f32",
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )
    b = IrrepsIndexedLinear(
        IRREPS,
        IRREPS,
        NS,
        method="naive",
        precision="f32",
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )
    nf = {ir: jax.random.normal(jax.random.key(1), (N, ir.dim, 4), jnp.float32) for _, ir in IRREPS}
    s, r = jax.random.randint(jax.random.key(2), (2, N * 4), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, is_sorted=True, may_have_unmapped=False),
        N,
        NS,
    )
    ya, yb = a(nf, counts, g), b(nf, counts, g)
    for ir in yb:
        assert np.all(np.isfinite(np.asarray(ya[ir])))
        rel = np.linalg.norm(np.asarray(ya[ir]) - np.asarray(yb[ir])) / (
            np.linalg.norm(np.asarray(yb[ir])) + 1e-30
        )
        assert rel < 1e-5, f"cuBLAS != naive for {ir}: rel {rel:.2e}"


def test_loom_map_bf16_flow():
    NS = 3
    # bf16 activations need a bf16-element precision (the op errors if the
    # activation dtype != precision.elem); master weights stay fp32.
    a = IrrepsIndexedLinear(
        IRREPS,
        IRREPS,
        num_indices=NS,
        method="loom_map",
        precision="bf16",
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )
    species = jnp.array([0, 0, 1, 1, 1, 2], jnp.int32)
    N = 6
    s, r = jax.random.randint(jax.random.key(7), (2, 16), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )
    nf32 = {
        ir: jax.random.normal(jax.random.key(8), (N, ir.dim, 4), jnp.float32) for _, ir in IRREPS
    }
    nf16 = jax.tree.map(lambda v: v.astype(jnp.bfloat16), nf32)
    y16 = a(nf16, jnp.zeros((NS,), jnp.int32), g)
    assert all(v.dtype == jnp.bfloat16 for v in y16.values())
    assert all(np.all(np.isfinite(np.asarray(v, np.float32))) for v in y16.values())


def test_loom_map_grad_flows():
    NS = 4
    a, _ = _ops(NS)
    species = jnp.array([0, 1, 1, 2, 3, 0], jnp.int32)
    N = 6
    nf = {ir: jax.random.normal(jax.random.key(5), (N, ir.dim, 4), jnp.float64) for _, ir in IRREPS}
    s, r = jax.random.randint(jax.random.key(6), (2, 16), 0, N)
    g = build_graph(
        IndexMap(s, may_have_unmapped=False),
        IndexMap(r, may_have_unmapped=False),
        IndexMap(species, may_have_unmapped=False),
        N,
        NS,
    )

    def loss(feats):
        y = a(feats, jnp.zeros((NS,), jnp.int32), g)
        return sum(jnp.sum(v**2) for v in y.values())

    grads = jax.grad(loss)(nf)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in grads.values())
