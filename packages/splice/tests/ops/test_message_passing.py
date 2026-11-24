# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke + correctness for the ``message_passing`` op.

Fused order-2/3 message passing ``R(r)·[x[s] ⊗ Y, x[s] ⊗ Y ⊗ x[r]] → x[r]`` through
``loom`` with the O(3) spherical harmonics ``Y(n̂)`` and the free-knot
cubic radial ``R(r)`` both computed in-polynomial from ``edge_vectors``. Checks
output structure + finiteness, gradient flow to the knot params, O(3) rotation and
parity equivariance, the *mixed* degree (order 2 linear, order 3 quadratic in the
node features), that both orders are present, the exact orthonormality of the
coupling-path coefficients (``AlgebraicReal`` identities, zero tolerance), that the sum
variant is exactly the 1/√n_paths block sum of the cat variant (the fused radial is
the whole path mixture — no gain), that ``scale`` is an exact output factor, and
that ``max_nnz`` grades the path set (and, with it, the harmonic degree).
"""

from collections import defaultdict

import e3.jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3 import O3, Irreps, clebsch_gordan
from loom import IndexMap
from splice.graph import build_graph
from splice.ops.message_passing import (
    MessagePassingOp,
    _coordinates_from_displacements,
    _paths,
)
from splice.ops.sph import SphericalHarmonics


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for the bit-exact equivariance checks, scoped to this
    module's tests. Restored afterwards so it does not leak into other test
    modules (a module-level ``jax.config.update`` flips x64 process-wide and
    breaks later fp32 dtype assertions)."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


MUL, N, E, R_MAX, MAX_NNZ = 32, 5, 10, 5.0, 20
FEATURES = Irreps(O3, "0e + 1o + 2e")
RADIALS = ["free_knot", "bump"]  # the two pluggable radial families


def test_channelwise_wrapper_exposes_sum_paths_without_changing_defaults():
    kwargs = dict(
        irreps_in=MUL * FEATURES,
        irreps_out=MUL * FEATURES,
        epsilon=0.125,
        r_max=R_MAX,
        l_max=2,
        precision="f32",
        out_dtype=jnp.float32,
        param_dtype=jnp.float32,
        key=jax.random.key(50),
    )
    legacy = MessagePassingOp(**kwargs)
    summed = MessagePassingOp(**kwargs, sum_paths=True)
    fixed = MessagePassingOp(
        **kwargs, extent_init="full_support", full_support_margin=0.02, n_fixed_reach=2
    )
    assert not legacy.mp.sum_paths
    assert summed.mp.sum_paths
    assert legacy.mp.radial.extent_init == "spread"
    assert legacy.mp.radial.n_fixed == 0
    np.testing.assert_array_equal(fixed.mp.radial.support_radius()[:2], [R_MAX, R_MAX])
    np.testing.assert_allclose(
        fixed.mp.radial.support_radius()[2:], R_MAX * (0.1 + 0.9 * 0.98), rtol=1e-6
    )


def _generators(ir):
    """The so(3) generators acting on ``ir``, from its own Clebsch-Gordan.

    The pseudovector ``1e`` is the adjoint representation, so ``1e (x) ir -> ir``
    is the generator triple. Our Clebsch-Gordan is unit-normalized, so the
    coupling has Frobenius norm 1 while the so(3) generators of degree ``l``
    have norm ``sqrt(l(l+1))``; rescaling by that makes ``exp(-i theta . X)``
    the rotation by ``theta``.
    """
    if ir.dim == 1:
        return np.zeros((3, 1, 1))
    [basis] = clebsch_gordan(O3.parse("1e"), ir, ir)
    dense = np.zeros(basis.shape)
    for coordinates, coefficient in basis:
        dense[coordinates] = float(coefficient)
    degree = (ir.dim - 1) // 2
    return np.sqrt(degree * (degree + 1)) * dense


def _wigner_d(ir, axis_angle):
    """Rotation matrix of irrep ``ir`` for an axis-angle vector (3,)."""
    x = jnp.asarray(_generators(ir), jnp.complex128)
    ix = 1j * jnp.einsum("a,aij->ij", axis_angle.astype(jnp.complex128), x)
    vals, vecs = jnp.linalg.eigh(ix)
    return jnp.real(vecs @ jnp.diag(jnp.exp(-1j * vals)) @ vecs.conj().T)


def test_both_orders_enumerated():
    """The default path set contains both order-2 and order-3 coupling trees."""
    irs = FEATURES
    orders = {p.order for p in _paths(irs, irs, irs, None)}
    assert orders == {2, 3}


def test_path_coefficients_exactly_orthonormal():
    """Per (legs, ir_out) group, the coupling-tree coefficient vectors satisfy
    ⟨C_a, C_b⟩ = δ_ab · dim(ir_out) — exact AlgebraicReal identities, zero tolerance.
    The normalisation contract: unit-variance inputs give unit variance per
    output block with the raw coefficients, no rescale."""
    from algebraic_real import AlgebraicReal

    irs = FEATURES
    paths = _paths(irs, irs, irs, None)
    groups = defaultdict(list)
    for p in paths:
        groups[(p.legs, str(p.ir_out))].append(p)
    assert any(len(v) > 1 for v in groups.values())  # multiple λ trees do occur
    for (_, _), ps in groups.items():
        d = ps[0].ir_out.dim
        for a, pa in enumerate(ps):
            for b, pb in enumerate(ps):
                g = AlgebraicReal()
                for idx, ca in pa.coefs.items():
                    cb = pb.coefs.get(idx)
                    if cb is not None:
                        g = g + ca * cb
                assert g == (AlgebraicReal(d) if a == b else AlgebraicReal()), (a, b, str(g))


def test_sqrt_free_geometry_avoids_square_root():
    displacement = jnp.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]], jnp.float32)
    squared_radius, harmonic_vector = _coordinates_from_displacements(
        displacement, "sqrt_free_solid_harmonics"
    )
    np.testing.assert_allclose(squared_radius, jnp.array([[25.0], [9.0]]))
    np.testing.assert_array_equal(harmonic_vector, displacement)

    jaxpr = jax.make_jaxpr(
        lambda x: _coordinates_from_displacements(x, "sqrt_free_solid_harmonics")
    )(displacement)
    assert not {"sqrt", "rsqrt"} & {equation.primitive.name for equation in jaxpr.jaxpr.eqns}


def test_sqrt_free_solid_harmonics_math_and_equivariance():
    op = MessagePassingOp(
        MUL * FEATURES,
        MUL * FEATURES,
        epsilon=0.125,
        r_max=R_MAX,
        l_max=2,
        geometry="sqrt_free_solid_harmonics",
        precision="f64",
        out_dtype=jnp.float64,
        param_dtype=jnp.float64,
        key=jax.random.key(51),
    )
    windows, a0, a1 = op.mp.radial.operands()
    (inverse_cutoff,) = windows
    np.testing.assert_allclose(inverse_cutoff, 1.0 / op.mp.radial.support_radius())
    squared_radius = jnp.array([0.0, 4.0, 25.0, 36.0], jnp.float64)[:, None]
    actual = op.mp.radial.evaluate(squared_radius, *windows, a0, a1)
    scaled_radius = squared_radius / op.mp.radial.support_radius() ** 2
    expected = jnp.where(
        scaled_radius <= 1.0,
        (a0 + a1 * scaled_radius) * (1.0 - scaled_radius) ** 2,
        0.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)

    x = {
        ir: jax.random.normal(jax.random.key(52), (N, ir.dim, MUL), jnp.float64)
        for _, ir in FEATURES
    }
    displacement = jax.random.normal(jax.random.key(53), (E, 3), jnp.float64)
    senders, receivers = jax.random.randint(jax.random.key(54), (2, E), 0, N)
    graph = build_graph(
        IndexMap(senders, may_have_unmapped=False),
        IndexMap(receivers, may_have_unmapped=False),
        IndexMap(jnp.zeros((N,), jnp.int32), may_have_unmapped=False),
        N,
        1,
    )
    axis_angle = jax.random.normal(jax.random.key(55), (3,), jnp.float64) * 0.4
    vector_rotation = _wigner_d(list(FEATURES)[1][1], axis_angle)

    def rotate(features):
        return {
            ir: jnp.einsum("ij,njm->nim", _wigner_d(ir, axis_angle), value)
            for ir, value in features.items()
        }

    rotated_input = op(
        rotate(x),
        graph,
        edge_vectors=jnp.einsum("ij,ej->ei", vector_rotation, displacement),
    )
    rotated_output = rotate(op(x, graph, edge_vectors=displacement))
    error = max(
        float(jnp.max(jnp.abs(rotated_input[ir] - rotated_output[ir]))) for ir in rotated_output
    )
    assert error < 1e-10, error


def test_sqrt_free_solid_harmonics_rejects_two_sided_bump():
    with pytest.raises(ValueError, match="requires radial='free_knot'"):
        MessagePassingOp(
            MUL * FEATURES,
            MUL * FEATURES,
            epsilon=0.125,
            r_max=R_MAX,
            l_max=2,
            radial="bump",
            geometry="sqrt_free_solid_harmonics",
            precision="f32",
            out_dtype=jnp.float32,
            param_dtype=jnp.float32,
        )


def test_order2_paths_match_loom_tensor_product():
    """Order-2 column layout equals ``e3.jax.tensor_product``'s.

    The order-2 branch of ``MessagePassing`` builds its path columns with
    ``tensor_product`` instead of emitting one term per nonzero CG coefficient.
    Column ORDER is the layout stored per-path weights (``gain``, the per-path
    radial) are indexed against, so a reordering upstream would silently corrupt
    every trained checkpoint while leaving fresh-weight tests green. Pin it here,
    for uniform AND repeated-irrep (non-uniform per-l width) inputs — the latter
    is the case ``tensor_product`` would collapse if it were not driven one
    (left, right) pair at a time.
    """
    import e3
    from e3 import Irreps
    from splice.ops.message_passing import MessagePassing

    for in_spec, out_spec, l_max in (
        ("0e+1o+2e", "0e+1o+2e+3o", 3),  # uniform
        ("0e+0e+1o", "0e+1o+2e", 2),  # repeated 0e leg = non-uniform CW
    ):
        op = MessagePassing(
            Irreps(O3, in_spec),
            Irreps(O3, out_spec),
            128,
            sum_paths=False,
            l_max=l_max,
            max_nnz=None,
            shared_t=True,
            gain=True,
            r_max=6.0,
            precision="f32",
            out_dtype=jnp.float32,
            name="probe",
            param_dtype=jnp.float32,
            key=jax.random.key(0),
        )
        assert all(len(legs) == 2 for legs, _, _ in op.paths), "expected order-2 only"

        expected = defaultdict(list)
        for legs, ir_out, _ in op.paths:
            expected[str(ir_out)].append(tuple(tuple(x) for x in legs))

        got = defaultdict(list)
        keep = [ir for ir, _ in op.blocks]
        for i_src, ir_src in enumerate(op.in_irs):
            for i_sh, ir_sh in enumerate(op.sh_irs):
                pair = e3.jax.tensor_product(
                    {ir_src: jnp.zeros((ir_src.dim,), jnp.float32)},
                    {ir_sh: jnp.zeros((ir_sh.dim,), jnp.float32)},
                    keep,
                )
                sig = (("src", i_src), ("sh", i_sh))
                for ir, value in pair.items():
                    got[str(ir)].extend([sig] * value.shape[1])

        assert dict(got) == dict(expected), f"path order diverged for {in_spec}"


def test_order3_emission_matches_stored_path_coefficients():
    """The chained-``tensor_product`` order-3 emission reproduces each stored path's
    exact CG coefficients, path for path, in the stored ORDER.

    This is the check equivariance cannot make: a permuted path set is still
    equivariant, it just pairs every tree with the wrong per-path radial/``gain``
    weight. So compare what the emission computes for path ``p`` -- the two chained
    couplings ``(x_s ⊗ Y)_λ ⊗ x_r`` -- against ``p.coefs``, which `_paths` derived
    independently with exact ``AlgebraicReal`` arithmetic.

    Runs on one lane with random streams, which is where the emission lives; the
    whole-op A/B against the previous scalar emission (values and force gradient,
    agreeing to 9.6e-7 = fp32 round-off) is ``triple_mp_order3_ab.py``.
    """
    import e3
    from e3 import Irreps
    from splice.ops.message_passing import MessagePassing, _only

    op = MessagePassing(
        Irreps(O3, "0e+1o+2e"),
        Irreps(O3, "0e+1o+2e"),
        128,
        sum_paths=False,
        l_max=2,
        max_nnz=8,
        shared_t=True,
        gain=True,
        r_max=6.0,
        precision="f32",
        out_dtype=jnp.float32,
        name="probe",
        param_dtype=jnp.float32,
        key=jax.random.key(0),
    )
    assert any(lam is not None for lam in op.lams), "expected order-3 paths"
    assert len(op.lams) == len(op.paths)

    rng = np.random.default_rng(0)
    src = [jnp.asarray(rng.normal(size=ir.dim)) for ir in op.in_irs]
    dst = [jnp.asarray(rng.normal(size=ir.dim)) for ir in op.in_irs]
    sh = [jnp.asarray(rng.normal(size=ir.dim)) for ir in op.sh_irs]
    streams = {"src": src, "dst": dst, "sh": sh}

    for (legs, ir_out, terms), ir_lam in zip(op.paths, op.lams, strict=True):
        assert (ir_lam is None) == (len(legs) == 2)
        # what the emission builds
        (_, i_src), (_, i_sh) = legs[0], legs[1]
        first = e3.jax.tensor_product(
            {op.in_irs[i_src]: src[i_src]},
            {op.sh_irs[i_sh]: sh[i_sh]},
            [ir_out if ir_lam is None else ir_lam],
        )
        if ir_lam is None:
            got = _only(first, ir_out)
        else:
            (_, i_dst) = legs[2]
            got = _only(
                e3.jax.tensor_product(
                    {ir_lam: _only(first, ir_lam)},
                    {op.in_irs[i_dst]: dst[i_dst]},
                    [ir_out],
                ),
                ir_out,
            )
        # what the stored coefficients say it should be
        want = np.zeros(ir_out.dim)
        for indices, coefficient in terms:
            *components_in, component_out = indices
            value = float(coefficient)
            for (stream, operand), component in zip(legs, components_in, strict=True):
                value *= float(streams[stream][operand][component])
            want[component_out] += value
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12)


def test_external_radial_matches_plain_channelwise_reference():
    """MACE source (``radial="external"`` + external ``edge_sph``) computes the plain
    channelwise tensor product ``epsilon · R · (x[src] ⊗ Y)`` scattered to ``dst``,
    with ``R`` the whole per-(edge, path, channel) ``mp_weights``.

    This is the exactness guarantee for routing standard MACE through the unified op:
    checked against ``splice.ops._common.native_channelwise`` — the same channelwise
    contraction the former ``channelwise_mp`` used — evaluated per edge and scattered
    to receivers by hand. Any drift in path order, weight layout, or the epsilon fold
    breaks it while fresh-weight smoke tests stay green.
    """
    from splice.ops._common import native_channelwise

    irreps_in = Irreps(O3, "3x0e+3x1o")
    irreps_sh = Irreps(O3, "1x0e+1x1o+1x2e")
    irreps_out = Irreps(O3, "3x0e+3x1o+3x2e")
    mul, eps = 3, 0.37
    op = MessagePassingOp(
        irreps_in,
        irreps_out,
        epsilon=eps,
        radial="external",
        irreps_sh=irreps_sh,
        precision="f64",
        out_dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    in_types = tuple(ir for _, ir in irreps_in)
    sh_types = op.mp.sh_irs
    native_in, native_sh = op.mp.in_irs, op.mp.sh_irs
    native_out, path_counts = tuple(ir for ir, _ in op.mp.blocks), op.mp.blocks

    def randn(i, ir):
        key = jax.random.fold_in(jax.random.key(1), i)
        return jax.random.normal(key, (N, ir.dim, mul), jnp.float64)

    x = {ir: randn(i, ir) for i, ir in enumerate(in_types)}
    ev = jax.random.normal(jax.random.key(2), (E, 3), jnp.float64)
    senders, receivers = jax.random.randint(jax.random.key(3), (2, E), 0, N)
    graph = build_graph(
        IndexMap(senders, may_have_unmapped=False),
        IndexMap(receivers, may_have_unmapped=False),
        IndexMap(jnp.zeros((N,), jnp.int32), may_have_unmapped=False),
        N,
        1,
    )
    sph = SphericalHarmonics(max((ir.dim - 1) // 2 for ir in sh_types))(ev)
    w = jax.random.normal(jax.random.key(4), (E, op.weight_numel), jnp.float64)
    got = op(x, graph, edge_sph=sph, mp_weights=w)

    # Reference: gather node features on senders, run the shared channelwise
    # contraction per edge AND per channel (native_channelwise is the per-lane kernel
    # body, so its operands are scalar-channel vectors), scatter to receivers, scale by
    # epsilon. The per-lane weight is mp_weights[e] reshaped [n_paths, mul], column c.
    s = np.asarray(senders)
    r = np.asarray(receivers)
    wp = np.asarray(w).reshape(E, op.n_paths, mul)
    ref = {ir: np.zeros((N, ir.dim, count * mul)) for ir, count in path_counts}
    for e in range(E):
        for c in range(mul):
            left = tuple(jnp.asarray(np.asarray(x[ir])[s[e], :, c]) for ir in native_in)
            right = tuple(jnp.asarray(np.asarray(sph[ir])[e, :, 0]) for ir in native_sh)
            paths = native_channelwise(
                jnp.asarray(wp[e, :, c]),
                left,
                right,
                native_in,
                native_sh,
                native_out,
                path_counts,
            )
            for ir, _count in path_counts:
                block = np.asarray(paths[ir])  # [ir.dim, count]
                ref[ir][r[e], :, c::mul] += eps * block

    for ir in got:
        np.testing.assert_allclose(np.asarray(got[ir]), ref[ir], rtol=0, atol=1e-11)
