# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RadialFreeKnot: cubic radial basis with a learnable free knot per channel.

    f_k = c_k0 d + c_k1 b,   b = v (1 - v)^2 sqrt(105/e_k),
    d = (2 - 7v)(1 - v)^2 sqrt(3/e_k),   v = r / (e_k r_max), zero past the knot.

Each channel vanishes C^1 at its knot e_k * r_max (no envelope); e_k is squashed into
(extent_floor, 1). The atoms (b, d) are ORTHONORMAL under r ~ U(0, r_max) in second
moment: <b^2> = <d^2> = 1, <b d> = 0 — no mean subtraction (that would inject bias
downstream, cf. `radial_mlp`). They are the Gram-Schmidt pair of the C^1-vanishing
cubics anchored at the bump b; all raw moments are linear in e, so the coefficients
are constants — exact at every extent, nothing calibrated, nothing frozen.

The shape param t charts the half circle (c0, c1) = (t, 1)/sqrt(1 + t^2), a bijection
onto the shapes up to sign. Output RMS is EXACTLY 1 for every (t, e), and t is the
angle-uniform arctan chart of that RMS circle: |df/dt|_{L^2} = 1/(1 + t^2), maximal
at t = 0 — which is exactly the bump b. t = sqrt(5/7) is exactly the monotone decay
(1 - v)^2 sqrt(5/e); the chart's one point at infinity is +-d, the anti-bump.
Compact bumps are sums of t = 0 channels at 1/e-uniform extents with binomial weights
(-1)^i C(K-1, i) e_i on the raw bumps v (1 - v)^2 (K = 2: O(u^2) inner tail; K = 4:
exactly zero outside [e_1, e_4] r_max).

Init: two channels per knot at t = +-1 — an orthonormal pair (pairs are orthogonal
iff t t' = -1) — with knots at fractions 0.3..0.95 of (extent_floor, 1) (edges beyond
a knot give zero gradient, so smaller supports would start dead on real data).
Emits ``bessel``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from splice.ops._common import algebraic_squash, inv_algebraic_squash
from splice.registry import register_op


def _initial_extent_fractions(
    count: int,
    extent_init: str = "spread",
    full_support_margin: float = 1e-3,
) -> np.ndarray:
    """Return squashed extent coordinates for deterministic knot initialization."""
    if extent_init not in {"spread", "full_support"}:
        raise ValueError(f"extent_init must be 'spread' or 'full_support', got {extent_init!r}")
    if not np.isfinite(full_support_margin) or not 0.0 < full_support_margin < 1.0:
        raise ValueError(
            "full_support_margin must be finite and strictly between zero and one, "
            f"got {full_support_margin!r}"
        )
    if extent_init == "spread":
        return np.linspace(0.3, 0.95, count)
    return np.full(count, 1.0 - full_support_margin)


def _combine_free_knot_radials(radius, inverse_cutoff, a0, a1):
    """``(1-x)^2 (a0 + a1 x)`` for ``x = r/c <= 1``, else 0 — ONE envelope.

    Algebraically identical to combining two ``free_knot_radial`` evaluations::

        (a0-a1)·(1-x)^2·(1+0·x) + a1·(1-x)^2·(1+1·x) = (1-x)^2·(a0 + a1·x)

    but that spelling evaluates the envelope *and* its cutoff select TWICE, and this
    runs per edge AND per channel inside the fused kernel — on a MACE-L layer it cost
    1.6x forward / 1.8x on the force backward, and ~2x the op's lowering time. The
    two-call form was adopted when Loom dropped ``free_knot_radial``'s ``value``
    argument; this restores the single-envelope arithmetic without changing the
    function (see ``test_free_knot_single_envelope_matches_two_call``).
    """
    x = radius * inverse_cutoff
    envelope = 1 - x
    return jnp.where(envelope >= 0, (a0 + a1 * x) * envelope * envelope, 0.0)


def _chart(t):
    """Chart of the half unit circle c1 > 0: (c0, c1) = (t, 1) / sqrt(1 + t^2).
    t = 0 is the pure bump (0, 1); +-(1, 0) — the anti-bump — is the asymptotic
    class. Pure arithmetic — runs on numpy and jax arrays alike."""
    d = (1.0 + t * t) ** -0.5
    return t * d, d


@register_op("radial_free_knot")
class RadialFreeKnot(eqx.Module):
    """Cubic radial basis with learnable shape (orthonormal atoms mixed on the unit
    half circle — output RMS exactly 1) and compact support per channel. Drop-in
    replacement for Bessel. Reads: edge_vectors. Writes: bessel."""

    t: Array  # chart coords, [num_basis]
    raw_extent: Array  # [num_basis]
    r_max: float = eqx.field(static=True)
    extent_floor: float = eqx.field(static=True)
    num_basis: int = eqx.field(static=True)
    extent_init: str = eqx.field(static=True)
    full_support_margin: float = eqx.field(static=True)

    OUTPUTS = "bessel"  # generic radial-feature slot -> drop-in for radial_mlp
    CUTOFF_PARAM = "r_max"  # resolve_arch injects the card cutoff here

    def __init__(
        self,
        num_basis: int,
        r_max: float,
        extent_floor: float = 0.1,
        extent_init: str = "spread",
        full_support_margin: float = 1e-3,
        *,
        param_dtype: Any,
        key: Array,
    ):
        del key  # init is deterministic (uniform knots + universal axes)
        assert num_basis % 2 == 0, "channels come in orthogonal pairs per knot"
        self.num_basis = int(num_basis)
        self.r_max = float(r_max)
        self.extent_floor = float(extent_floor)
        self.extent_init = extent_init
        self.full_support_margin = float(full_support_margin)
        # Two channels per knot: the atoms are orthonormal, so pairs are orthogonal
        # iff t t' = -1; t = +-1 is the symmetric choice — at any knot. Knots start
        # at 0.3 (~2 Å support at typical r_max): edges beyond a knot give exactly
        # zero gradient, so smaller supports would start dead on solid-state data.
        # ``full_support`` instead starts every knot near the outer cutoff.
        s_knot = _initial_extent_fractions(
            self.num_basis // 2, self.extent_init, self.full_support_margin
        )
        self.t = jnp.asarray(np.tile([1.0, -1.0], self.num_basis // 2), param_dtype)
        self.raw_extent = jnp.asarray(inv_algebraic_squash(np.repeat(s_knot, 2)), param_dtype)

    def _extent(self) -> Array:
        return self.extent_floor + (1.0 - self.extent_floor) * algebraic_squash(self.raw_extent)

    def _basis(self, r: Array) -> Array:
        e = self._extent()
        radius = r[..., None]
        c0, c1 = _chart(self.t)
        # f = c0 d + c1 b as (a0 + a1 v)(1 - v)^2: d = sqrt(3/e)(2 - 7v)(1 - v)^2
        a0 = c0 * (12.0 / e) ** 0.5
        a1 = c1 * (105.0 / e) ** 0.5 - c0 * (147.0 / e) ** 0.5
        inverse_cutoff = 1.0 / (self.r_max * e)
        return _combine_free_knot_radials(radius, inverse_cutoff, a0, a1)

    def __call__(self, edge_vectors: Array) -> Array:
        r = jnp.linalg.norm(edge_vectors, axis=-1)  # [num_edges]
        return self._basis(r)
