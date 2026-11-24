# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Radial families for the in-kernel ``message_passing`` source.

The per-(path, channel) shape ``R(r)`` is evaluated as JAX inside the Loom lane
program. Two families share everything but the window and the C^k envelope:

- ``free_knot`` — C¹ cubic ``(a0 + a1·v)(1 - v)²`` gated at one knot ``v ≤ 1``
  (``v = r/c`` or ``r²/c²``, ``c = e·r_max``). One-sided cutoff at ``e·r_max``;
  the orthonormal atoms ``b, d`` are mixed on the unit half-circle by ``t``. This
  is ``radial_free_knot`` computed in-poly.

- ``bump`` — C² two-sided compact bump ``A0·g³ + A1·z·g³`` with ``g = 1 - z²``,
  ``z = (r - r_u)·w_inv``, gated at ``|z| ≤ 1`` (one predicate ``g ≥ 0`` shared by
  every path → warp-collective zero-skip). The even/odd generators ``g³``, ``z·g³``
  are orthonormal in ``L²(dr)``; ``t`` charts their mixture ``(a0, a1) = (1, t)/√(1+t²)``.
  Right edge ``q = r_u + w`` bounded to ``(extent_floor, 1)·r_max`` (what the cutoff/
  zero-skip see — the same fractional floor as the free-knot knot); left edge
  ``p = r_u - w`` free (unbounded width). See ``bump_family.pdf``.

Both families share one ``extent_floor`` (a fraction of ``r_max``): the outer support
edge — free-knot knot ``e·r_max`` or bump right edge ``q`` — lives in
``(extent_floor, 1)·r_max``, so it can never collapse to zero (a dead channel).

Shared too: the shape chart ``_chart(t)`` (unit circle) and the exp-free reparam maps
``algebraic_squash`` (σ) / ``algebraic_softplus`` (φ). Each family is an ``eqx.Module``
strategy owning its window/shape leaves + init and exposing:

    n_window            static: #per-channel window operands the kernel eats
    evaluate(r, win, a0, a1)    numeric lane program, returning one value per path
    operands()          numeric: leaves → (window arrays, a0, a1) kernel operands
    support_radius()    numeric: per-channel outer cutoff (knot e·r_max / right edge q)
    permuted(pi)        reorder the per-channel (mul-axis) leaves by π (channel sort)

The ``MessagePassing`` core wires these into the fused polynomial; ``gain`` (a per-path
amplitude) and the sum/cat path handling stay at the ``MessagePassing`` level — orthogonal
to the radial kind.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from splice.ops._common import (
    algebraic_softplus,
    algebraic_squash,
    inv_algebraic_softplus,
    inv_algebraic_squash,
)
from splice.ops.radial_free_knot import (
    _chart,
    _combine_free_knot_radials,
    _initial_extent_fractions,
)


def _init_t(mul: int, n_paths: int, shared_t: bool, dtype: Any) -> Array:
    """Two channels per knot at ``t = ±1`` — an orthonormal shape pair (chart atoms
    are orthonormal, so ``t·t' = -1`` pairs are orthogonal). ``[mul]`` when the shape
    is shared across paths (``shared_t``), else broadcast to ``[n_paths, mul]``."""
    t0 = jnp.asarray(np.tile([1.0, -1.0], mul // 2), dtype)
    return t0 if shared_t else jnp.broadcast_to(t0, (n_paths, mul))


class _FreeKnotRadial(eqx.Module):
    """C¹ free-knot cubic radial (in-poly form of ``radial_free_knot``)."""

    t: Array  # [mul] (shared) or [n_paths, mul] chart coords
    raw_extent: Array  # [mul] knot per channel
    r_max: float = eqx.field(static=True)
    extent_floor: float = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    extent_init: str = eqx.field(static=True)
    full_support_margin: float = eqx.field(static=True)
    n_fixed: int = eqx.field(static=True)

    n_window = 1  # kernel eats one per-channel window operand: inv

    def __init__(
        self,
        mul,
        n_paths,
        shared_t,
        *,
        r_max,
        extent_floor,
        geometry,
        dtype,
        extent_init="spread",
        full_support_margin=1e-3,
        n_fixed=0,
        **_ignored,
    ):
        self.r_max = float(r_max)
        self.extent_floor = float(extent_floor)
        self.geometry = geometry
        self.extent_init = extent_init
        self.full_support_margin = float(full_support_margin)
        s_knot = np.repeat(
            _initial_extent_fractions(mul // 2, self.extent_init, self.full_support_margin),
            2,
        )
        # First ``n_fixed`` channels: knot PINNED at r_max (e == 1), a fixed
        # full-cutoff long-range basis. Only the KNOT is fixed -- t (shape) and the
        # op's gain (amplitude) stay learnable, so the model still shapes and weights
        # the long-range signal, it just cannot collapse the reach. Guarantees
        # long-range carriers from step 0 while leaving the other mul-n_fixed
        # channels fully free (short channels for local structure are untouched,
        # unlike a global grow penalty).
        self.n_fixed = max(0, min(int(n_fixed), int(mul)))
        self.raw_extent = jnp.asarray(inv_algebraic_squash(s_knot), dtype)
        self.t = _init_t(mul, n_paths, shared_t, dtype)

    def _extent(self) -> Array:
        e = self.extent_floor + (1.0 - self.extent_floor) * algebraic_squash(self.raw_extent)
        if self.n_fixed > 0:
            # Pin the first n_fixed channels at e=1 (support_radius = r_max). e is then
            # constant for those, so raw_extent[:n_fixed] receives zero gradient
            # (effectively frozen -- the knot is fixed, not learned).
            e = e.at[: self.n_fixed].set(1.0)
        return e

    def operands(self) -> tuple[tuple[Array, ...], Array, Array]:
        """radial_free_knot's map: e = floor + (1-floor)·σ(raw); (c0,c1) = (t,1)/√(1+t²);
        a0 = c0√(12/e), a1 = c1√(105/e) - c0√(147/e);
        inverse cutoff = 1/c."""
        e = self._extent()
        c0, c1 = _chart(self.t)
        a0 = c0 * (12.0 / e) ** 0.5
        a1 = c1 * (105.0 / e) ** 0.5 - c0 * (147.0 / e) ** 0.5
        inverse_cutoff = 1.0 / (e * self.r_max)
        return (inverse_cutoff,), a0, a1

    def support_radius(self) -> Array:
        """Per-channel outer cutoff e·r_max (past it every path is zero)."""
        return self._extent() * self.r_max

    def permuted(self, pi: Array) -> _FreeKnotRadial:
        t = self.t[pi] if self.t.ndim == 1 else self.t[:, pi]
        return eqx.tree_at(lambda m: (m.raw_extent, m.t), self, (self.raw_extent[pi], t))

    def evaluate(self, radius, inverse_cutoff, a0, a1):
        if self.geometry == "sqrt_free_solid_harmonics":
            inverse_cutoff = inverse_cutoff**2
        return _combine_free_knot_radials(radius, inverse_cutoff, a0, a1)


# L²(dr) norms of the compact-bump generators, on z ∈ [-1, 1] (support width absorbed
# as √w_inv in `operands`): ‖g³‖² = w·I₆, ‖z·g³‖² = w·(I₆ − I₇), with
# Iₙ = ∫₋₁¹(1−z²)ⁿ dz = 2^{2n+1}(n!)²/(2n+1)!. Then I₆ = 2048/3003 and (I₇/I₆ = 14/15
# so) I₆ − I₇ = I₆/15 = 2048/45045. The kernel coefficient on each raw generator is
# a·√w_inv·K, K = 1/‖·‖ at unit width.
_K0 = float((3003.0 / 2048.0) ** 0.5)  # 1/√I₆        ≈ 1.21091
_K1 = float((45045.0 / 2048.0) ** 0.5)  # 1/√(I₆−I₇)  ≈ 4.68985


class _BumpRadial(eqx.Module):
    """C² two-sided compact-support bump radial (``bump_family.pdf``)."""

    t: Array  # [mul] (shared) or [n_paths, mul] — odd/even shape mixture
    s_q: Array  # [mul] right-edge param (shared across paths, per channel)
    s_p: Array  # [mul] width param
    r_max: float = eqx.field(static=True)
    extent_floor: float = eqx.field(static=True)  # right edge q/r_max ∈ (floor, 1)
    width_scale: float = eqx.field(static=True)  # ℓ: width = ℓ·φ(s_p)

    n_window = 2  # kernel eats two per-channel window operands: w_inv, ru_winv

    def __init__(
        self, mul, n_paths, shared_t, *, r_max, extent_floor, width_scale, dtype, **_ignored
    ):
        self.r_max = float(r_max)
        self.extent_floor = float(extent_floor)
        self.width_scale = float(r_max if width_scale is None else width_scale)
        # σ(s_q) spread 0.3..0.95 (a pair per edge) — the SAME init as the free-knot
        # knot, so the right edge q = (floor + (1-floor)·σ)·r_max matches the free-knot's
        # [0, e·r_max] outer radius. Left edge p at the origin → support [0, q], the C²
        # mirror; width = q is hit by s_p = φ⁻¹(q/ℓ) (so p = 0 at init).
        sig = np.repeat(np.linspace(0.3, 0.95, mul // 2), 2)
        q = (self.extent_floor + (1.0 - self.extent_floor) * sig) * self.r_max
        self.s_q = jnp.asarray(inv_algebraic_squash(sig), dtype)
        self.s_p = jnp.asarray(inv_algebraic_softplus(q / self.width_scale), dtype)
        self.t = _init_t(mul, n_paths, shared_t, dtype)

    def _edges(self) -> tuple[Array, Array]:
        """(q, p): right edge q/r_max ∈ (extent_floor, 1) via σ — the same fractional
        floor as the free-knot knot; left edge free, p < q."""
        frac = self.extent_floor + (1.0 - self.extent_floor) * algebraic_squash(self.s_q)
        q = self.r_max * frac
        p = q - self.width_scale * algebraic_softplus(self.s_p)
        return q, p

    def operands(self) -> tuple[tuple[Array, ...], Array, Array]:
        """w_inv = 2/(q-p), ru_winv = r_u·w_inv = (p+q)/(q-p); (a0,a1) = (1,t)/√(1+t²)
        L²-normalized: A0 = a0·√w_inv·K0 (even g³), A1 = a1·√w_inv·K1 (odd z·g³)."""
        q, p = self._edges()
        w_inv = 2.0 / (q - p)
        ru_winv = (p + q) / (q - p)
        c0, c1 = _chart(self.t)  # (a0, a1) = (c1, c0): swap → t=0 is the even bump
        sw = w_inv**0.5
        a0 = c1 * sw * _K0
        a1 = c0 * sw * _K1
        return (w_inv, ru_winv), a0, a1

    def support_radius(self) -> Array:
        """Per-channel right edge q = r_u + w (the cutoff the zero-skip sees)."""
        q, _ = self._edges()
        return q

    def permuted(self, pi: Array) -> _BumpRadial:
        t = self.t[pi] if self.t.ndim == 1 else self.t[:, pi]
        return eqx.tree_at(lambda m: (m.s_q, m.s_p, m.t), self, (self.s_q[pi], self.s_p[pi], t))

    def evaluate(self, radius, width_inverse, center_times_width_inverse, a0, a1):
        z = radius * width_inverse - center_times_width_inverse
        g = 1 - z**2
        g3 = g**3
        return jnp.where(g >= 0, a0 * g3 + a1 * z * g3, 0)


_RADIALS = {"free_knot": _FreeKnotRadial, "bump": _BumpRadial}


def make_radial(kind: str, mul, n_paths, shared_t, **kwargs) -> Any:
    """Build a radial strategy by name; extra kwargs (r_max, extent_floor, width_scale,
    dtype) are consumed per family (the others ignored)."""
    try:
        cls = _RADIALS[kind]
    except KeyError:
        raise ValueError(f"unknown radial {kind!r}, expected one of {sorted(_RADIALS)}") from None
    return cls(mul, n_paths, shared_t, **kwargs)
