# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RadialFreeKnot: cubic radial basis, orthonormal atoms mixed on the half circle.

Checks the structural invariants (chart on the c1 > 0 half circle for any t, extent
confined to (floor, 1), value/slope zero at each channel's support end for any
params, RMS exactly 1 for ANY params), the exact landmarks (t = 0 is the bump,
t = sqrt(5/7) the monotone decay), L^2-orthogonality of init pairs, channel
independence, and gradient flow to every param.
"""

import e3.jax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from splice.ops._fused_radial import make_radial
from splice.ops.radial_free_knot import (
    RadialFreeKnot,
    _chart,
    _combine_free_knot_radials,
)


def _op(K=8, Rc=5.0, **kwargs):
    return RadialFreeKnot(
        num_basis=K,
        r_max=Rc,
        param_dtype=jnp.float32,
        key=jax.random.key(0),
        **kwargs,
    )


def _randomized(op, seed=4, scale=3.0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    op = eqx.tree_at(lambda m: m.t, op, jax.random.normal(k1, op.t.shape) * scale)
    return eqx.tree_at(
        lambda m: m.raw_extent, op, jax.random.normal(k2, op.raw_extent.shape) * scale
    )


def _uniform_r_features(op, n=100_000):
    rr = jnp.linspace(0.0, op.r_max, n)
    return np.array(op._basis(rr), dtype=np.float64)


def test_param_shapes():
    op = _op(K=8)
    assert op.t.shape == (8,)
    assert op.raw_extent.shape == (8,)


def test_default_spread_initialization_is_unchanged():
    op = _op(K=8)
    squashed = np.repeat(np.linspace(0.3, 0.95, 4), 2)
    expected = op.extent_floor + (1.0 - op.extent_floor) * squashed
    assert np.allclose(op._extent(), expected, atol=1e-7, rtol=0)


def test_full_support_initialization_is_equal_and_near_rmax():
    op = _op(K=8, Rc=5.0, extent_init="full_support", full_support_margin=1e-3)
    expected_fraction = 0.1 + 0.9 * 0.999
    assert np.allclose(op._extent(), expected_fraction, atol=1e-7, rtol=0)
    fused = make_radial(
        "free_knot",
        8,
        2,
        True,
        r_max=5.0,
        extent_floor=0.1,
        geometry="normalized_harmonics",
        dtype=jnp.float32,
        extent_init="full_support",
        full_support_margin=1e-3,
    )
    assert np.allclose(fused._extent(), expected_fraction, atol=1e-7, rtol=0)


def test_full_support_extent_gradients_are_finite_and_nonzero():
    op = _op(K=8, extent_init="full_support")
    radii = jnp.linspace(0.2, 4.5, 64)

    def loss(raw_extent):
        model = eqx.tree_at(lambda m: m.raw_extent, op, raw_extent)
        return jnp.sum(model._basis(radii) ** 2)

    gradient = jax.grad(loss)(op.raw_extent)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.abs(gradient) > 0))


@pytest.mark.parametrize("extent_init", ["spread", "full_support"])
def test_fused_fixed_reach_preserves_initialization_of_learnable_channels(extent_init):
    kwargs = dict(
        r_max=5.0,
        extent_floor=0.1,
        geometry="normalized_harmonics",
        dtype=jnp.float32,
        extent_init=extent_init,
        full_support_margin=0.02,
    )
    free = make_radial("free_knot", 8, 2, True, **kwargs)
    fixed = make_radial("free_knot", 8, 2, True, n_fixed=2, **kwargs)
    np.testing.assert_array_equal(fixed.raw_extent, free.raw_extent)
    np.testing.assert_array_equal(fixed.support_radius()[:2], [5.0, 5.0])
    np.testing.assert_array_equal(fixed.support_radius()[2:], free.support_radius()[2:])

    def reach(raw_extent):
        model = eqx.tree_at(lambda m: m.raw_extent, fixed, raw_extent)
        return jnp.sum(model.support_radius())

    gradient = jax.grad(reach)(fixed.raw_extent)
    np.testing.assert_array_equal(gradient[:2], [0.0, 0.0])
    assert bool(jnp.all(jnp.isfinite(gradient[2:]) & (gradient[2:] > 0)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extent_init": "unknown"}, "extent_init"),
        ({"full_support_margin": 0.0}, "full_support_margin"),
        ({"full_support_margin": 1.0}, "full_support_margin"),
    ],
)
def test_invalid_extent_initialization_fails_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _op(**kwargs)


def test_extent_confined():
    """e_k in (extent_floor, 1) by construction, however wild raw_extent gets."""
    op = _randomized(_op(K=8), scale=100.0)
    e = op._extent()
    assert bool(jnp.all(e > op.extent_floor))
    assert bool(jnp.all(e < 1.0))


def test_chart_on_half_circle():
    """(c0, c1) lands on the c1 > 0 half of the unit circle for ANY t."""
    t = jax.random.normal(jax.random.key(0), (64,)) * 50.0
    c0, c1 = _chart(t)
    assert bool(jnp.allclose(c0 * c0 + c1 * c1, 1.0, atol=1e-6))
    assert bool(jnp.all(c1 > 0))


def test_pure_bump_at_t_zero():
    """t = 0 is exactly the bump atom sqrt(105/e) v (1-v)^2 — the chart's centre —
    with RMS exactly 1 under r ~ U(0, r_max). f(0) == 0 exactly."""
    op = _op(K=2)
    op = eqx.tree_at(lambda m: m.t, op, jnp.zeros(2))
    rr = jnp.linspace(0.0, op.r_max, 100_000)
    e = op._extent()
    v = jnp.clip(rr[:, None] / op.r_max / e, max=1.0)
    bump = jnp.sqrt(105.0 / e) * v * (1.0 - v) ** 2
    f = op._basis(rr)
    assert bool(jnp.allclose(f, bump, atol=1e-4))
    assert bool(jnp.all(f[0] == 0.0))
    rms = np.sqrt(np.mean(np.array(f, dtype=np.float64) ** 2, axis=0))
    assert np.allclose(rms, 1.0, atol=1e-3)


def test_pure_decay_at_t_sqrt_5_7():
    """t = sqrt(5/7) is exactly the monotone decay sqrt(5/e) (1-v)^2 — the
    RMS-standardized first atom, at an interior chart point."""
    op = _op(K=2)
    op = eqx.tree_at(lambda m: m.t, op, jnp.full(2, float(np.sqrt(5.0 / 7.0))))
    rr = jnp.linspace(0.0, op.r_max, 100_000)
    e = op._extent()
    v = jnp.clip(rr[:, None] / op.r_max / e, max=1.0)
    decay = jnp.sqrt(5.0 / e) * (1.0 - v) ** 2
    assert bool(jnp.allclose(op._basis(rr), decay, atol=1e-4))


def test_forward_shape_and_finite():
    op = _op(K=8)
    ev = jax.random.normal(jax.random.key(1), (200, 3)) * 2.0
    out = op(ev)
    assert out.shape == (200, 8)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_smooth_at_support_end_any_params():
    """Each channel vanishes with zero slope at its own support end
    r_k = e_k * r_max, for ARBITRARY params — the (1-v)^2 factor is
    structural. Beyond the cutoff everything is zero."""
    op = _randomized(_op(K=8, Rc=5.0))
    r_end = np.array(op._extent()) * op.r_max

    def channel(rscalar, k):
        ev = jnp.array([[rscalar, 0.0, 0.0]])
        return op(ev)[0, k]

    g = jax.grad(channel)
    for k in range(8):
        assert abs(float(channel(r_end[k], k))) < 1e-3
        assert abs(float(channel(r_end[k] + 0.3, k))) < 1e-6
        # slope tapers linearly to zero at the support end: ~100x between deltas
        s_far = abs(float(g(r_end[k] - 1e-2, k)))
        s_near = abs(float(g(r_end[k] - 1e-4, k)))
        assert s_near < 0.05 * s_far + 1e-4
        assert abs(float(g(r_end[k] + 0.3, k))) < 1e-6
    ev = jnp.array([[op.r_max + 0.2, 0.0, 0.0]])
    assert bool(jnp.all(op(ev) == 0.0))


def test_rms_unit_for_any_params():
    """RMS under r ~ U(0, r_max) is EXACTLY 1 for ANY (t, e): orthonormal atoms
    mixed with unit-norm chart coefficients."""
    rms = np.sqrt(np.mean(_uniform_r_features(_randomized(_op(K=8))) ** 2, axis=0))
    assert np.allclose(rms, 1.0, atol=1e-3)
    rms0 = np.sqrt(np.mean(_uniform_r_features(_op(K=8)) ** 2, axis=0))
    assert np.allclose(rms0, 1.0, atol=1e-3)


def test_grad_flows_to_every_param():
    """Every chart coord and every extent must receive gradient at init."""
    op = _op(K=8)
    ev = jax.random.normal(jax.random.key(3), (300, 3)) * 1.5

    def loss(m):
        return jnp.sum(m(ev) ** 2)

    grads = jax.grad(loss)(op)
    assert bool(jnp.all(jnp.isfinite(grads.t)))
    assert bool(jnp.all(jnp.abs(grads.t) > 0))
    assert bool(jnp.all(jnp.isfinite(grads.raw_extent)))
    assert bool(jnp.all(jnp.abs(grads.raw_extent) > 0))


def test_init_pairs_orthogonal():
    """The two channels sharing a knot are exactly L^2-orthogonal under
    r ~ U(0, r_max): with orthonormal atoms, pairs are orthogonal iff t t' = -1,
    so the init t = +-1 pair is orthogonal at every knot."""
    op = _op(K=8)
    F = _uniform_r_features(op)
    M = F.T @ F / F.shape[0]  # second-moment Gram
    for j in range(0, 8, 2):
        corr = M[j, j + 1] / np.sqrt(M[j, j] * M[j + 1, j + 1])
        assert abs(corr) < 1e-3
    assert np.allclose(np.repeat(op._extent()[::2], 2), op._extent(), atol=1e-6)


def test_channels_independent_at_init():
    """Two shapes per knot span the full C^1 cubic spline space on the knot set:
    the init feature matrix has numerical rank well above 2."""
    op = _op(K=16)
    rr = jnp.linspace(0.0, op.r_max, 512)
    F = np.array(op._basis(rr), dtype=np.float64)
    s = np.linalg.svd(F, compute_uv=False)
    assert s[5] / s[0] > 1e-4


def test_free_knot_single_envelope_matches_two_call():
    """The single-envelope form equals the two-``free_knot_radial`` combination.

    ``_combine_free_knot_radials`` used to evaluate the envelope and its cutoff
    select twice and combine them; it now evaluates the identical polynomial once.
    That is a ~1.6-1.8x speedup inside the fused kernel (it runs per edge AND per
    channel), so the arithmetic must stay pinned to the form checkpoints were
    trained against. Covers inside the cutoff, past it, and exactly on the knot,
    for the value and its r-derivative (the force path).
    """
    import e3

    def two_call(radius, inverse_cutoff, a0, a1):
        constant = e3.jax.free_knot_radial(radius, inverse_cutoff, 0.0)
        unit_slope = e3.jax.free_knot_radial(radius, inverse_cutoff, 1.0)
        return (a0 - a1) * constant + a1 * unit_slope

    rng = np.random.default_rng(0)
    r = jnp.asarray(
        np.concatenate([rng.uniform(0.0, 1.5, 20_000), [0.0, 1.0, 1 - 1e-7, 1 + 1e-7]]),
        jnp.float32,
    )
    inv = jnp.ones_like(r)
    a0 = jnp.asarray(rng.normal(0, 1, r.shape), jnp.float32)
    a1 = jnp.asarray(rng.normal(0, 1, r.shape), jnp.float32)

    got, want = _combine_free_knot_radials(r, inv, a0, a1), two_call(r, inv, a0, a1)
    assert np.allclose(got, want, atol=2e-6, rtol=0)

    # d/dr: the fused op's force path is where the duplicated envelope cost the most.
    d_got = jax.grad(lambda x: jnp.sum(_combine_free_knot_radials(x, inv, a0, a1)))(r)
    d_want = jax.grad(lambda x: jnp.sum(two_call(x, inv, a0, a1)))(r)
    assert np.allclose(d_got, d_want, atol=5e-6, rtol=0)
