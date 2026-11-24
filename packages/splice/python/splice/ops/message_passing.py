# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General order-2/3 message passing ``R(r)·[x[s]⊗Y, x[s]⊗Y⊗x[r]] → x[r]`` in one
``loom`` lane program. The coupling core — path enumeration, the per-``(src, sh)``
tensor products, the per-path weight, the optional order-3 ``⊗ x[r]`` leg, the
``scale``, ``cat``/``sum`` and the scatter to ``dst`` — is one implementation. What
varies is only the two per-edge fields it consumes, and each has two sources:

- the harmonics ``Y``: computed in-kernel from ``edge_vectors`` (the fused families),
  or supplied as an external ``edge_sph`` operand (standard MACE);
- the per-path radial weight ``R(r)``: an in-kernel parametric family (``free_knot`` /
  ``bump``), or supplied whole as an external ``mp_weights`` operand (standard MACE).

Standard MACE is exactly the order-2 core with both fields external: ``mp_weights``
plays the role of the per-path weight and ``edge_sph`` the harmonics, so the computed
function equals the former ``channelwise_mp`` up to floating-point rounding.

The in-kernel radial ``R(r)`` is a pluggable family (``radial=``), a strategy in
``_fused_radial`` sharing the shape chart ``t → (a0, a1)`` and the exp-free reparam
maps: ``free_knot`` (C¹ cubic, one-sided knot) or ``bump`` (C² two-sided compact
support). ``MessagePassing`` holds one as a sub-module; ``gain`` (per-path amplitude)
and the sum/cat handling are radial-agnostic.

The in-kernel order-2 form can instead use squared radius: the free-knot coordinate
is ``r²/c²`` while solid harmonics are evaluated at ``r/c``. This mode is polynomial
in the displacement and never forms ``√r²``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import algebraic_real.jax
import e3
import e3.jax
import equinox as eqx
import jax
import jax.numpy as jnp
import loom
from algebraic_real import AlgebraicReal, pow
from e3 import O3, Irrep, Irreps, clebsch_gordan
from jax import Array

from splice.graph import Graph
from splice.ops._common import (
    replace_fields,
    safe_norm,
)
from splice.ops._fused_radial import make_radial
from splice.precision import Precision
from splice.precision import parse as parse_precision
from splice.registry import register_op

_VECTOR = O3.parse("1o")  # spherical harmonics are the O3 (parity-odd) vector
_GEOMETRIES = ("normalized_harmonics", "sqrt_free_solid_harmonics")


def _coordinates_from_displacements(
    displacements: Array,
    geometry: Literal["normalized_harmonics", "sqrt_free_solid_harmonics"],
) -> tuple[Array, Array]:
    if geometry == "sqrt_free_solid_harmonics":
        squared_radius = jnp.sum(displacements * displacements, axis=-1, keepdims=True)
        return squared_radius, displacements
    radius = safe_norm(displacements, axis=-1, keepdims=True)
    return radius, displacements / radius


def _flatten(irreps: Irreps) -> Irreps:
    """Expand local mults into unit-mult legs (``2x0e+1o`` → ``0e+0e+1o``) — the
    channel extent ``mul`` is separate, so repeated irreps are the non-uniform CW."""
    return Irreps(irreps.group, [(1, ir) for m, ir in irreps for _ in range(m)])


def _max_l(irreps: Irreps) -> int:
    return max((ir.dim - 1) // 2 for _, ir in irreps)


def _only(paths: dict[Irrep, Array], ir: Irrep) -> Array:
    """The single coupling column of ``ir`` in a ONE-irrep ⊗ ONE-irrep product.

    `_paths` assumes each ``(ir_a, ir_b, ir_out)`` triple has exactly one coupling —
    true for O3, where `clebsch_gordan` returns a single basis element — and indexes
    its weights on that assumption. Assert it rather than silently taking column 0,
    because a multiplicity >1 group would shift the whole per-path weight layout.
    """
    value = paths[ir]
    if value.shape[1] != 1:
        raise ValueError(f"{ir} has {value.shape[1]} couplings, expected 1")
    return value[:, 0]


def _contract(cg1, cg2) -> dict[tuple[int, ...], AlgebraicReal]:
    """``Σ_λ cg1[i,j,λ]·cg2[λ,k,m]`` — order-3 coefficient of ``x_s[i]·Y[j]·x_r[k] → out[m]``."""
    by_lam: dict[int, list] = {}
    for (mu, k, m), c in cg2:
        by_lam.setdefault(mu, []).append((k, m, c))
    coefs: dict[tuple[int, ...], AlgebraicReal] = {}
    for (i, j, mu), c1 in cg1:
        for k, m, c2 in by_lam.get(mu, ()):
            coefs[i, j, k, m] = coefs.get((i, j, k, m), AlgebraicReal()) + c1 * c2
    return coefs


@dataclass
class _Path:
    """A coupling tree: ``legs`` are its ``(stream, index)`` factors — two for order 2
    (``x_s ⊗ Y → out``), three for order 3 (``(x_s ⊗ Y)_λ ⊗ x_r → out``); ``coefs``
    maps ``(factor comps…, out comp)`` to its exact coefficient (non-zero only, so
    ``nnz`` counts its per-channel multiply–adds)."""

    legs: tuple[tuple[str, int], ...]
    ir_lam: Irrep | None
    ir_out: Irrep
    coefs: dict[tuple[int, ...], AlgebraicReal]

    @property
    def nnz(self) -> int:
        return len(self.coefs)

    @property
    def order(self) -> int:
        return len(self.legs)

    @property
    def l_sh(self) -> int:
        return next(i for s, i in self.legs if s == "sh")


def _paths(
    irreps_in: Irreps, irreps_sh: Irreps, filter_out: Irreps, max_nnz: int | None
) -> list[_Path]:
    """Order-2/3 coupling trees into ``filter_out`` with ``nnz ≤ max_nnz``. Order 2 is
    order 3 truncated at its intermediate, so both share the first coupling ``cg1``;
    sorted grouped by output irrep, order 2 first."""
    keep = {ir for _, ir in filter_out}
    nodes, sh = list(irreps_in), list(irreps_sh)
    paths: list[_Path] = []
    for i_s, (_, ir_s) in enumerate(nodes):
        for i_sh, (_, ir_y) in enumerate(sh):
            edge = (("src", i_s), ("sh", i_sh))
            for ir_lam in ir_s.selection_rule(ir_y):
                for cg1 in clebsch_gordan(ir_s, ir_y, ir_lam):
                    if ir_lam in keep and (c := {k: v for k, v in cg1 if v}):
                        paths.append(_Path(edge, None, ir_lam, c))
                    for i_r, (_, ir_r) in enumerate(nodes):
                        for ir_out in ir_lam.selection_rule(ir_r):
                            if ir_out not in keep:
                                continue
                            for cg2 in clebsch_gordan(ir_lam, ir_r, ir_out):
                                if c := {k: v for k, v in _contract(cg1, cg2).items() if v}:
                                    paths.append(_Path((*edge, ("dst", i_r)), ir_lam, ir_out, c))
    if max_nnz is not None:
        paths = [p for p in paths if p.nnz <= max_nnz]
    if not paths:
        raise ValueError(
            f"no {filter_out} in order-2/3 of {irreps_in} ⊗ {irreps_sh} (nnz ≤ {max_nnz})"
        )
    out_irs = sorted({p.ir_out for p in paths})
    paths.sort(key=lambda p: (out_irs.index(p.ir_out), p.order, p.legs))
    return paths


def _derive(
    in_flat: Irreps, filter_out: Irreps, l_max: int, max_nnz: int | None
) -> tuple[Irreps, list[_Path]]:
    """Order-2 trees with harmonics ``Y_0…Y_lmax`` plus, when ``max_nnz`` is given,
    order-3 trees pruned to ``nnz ≤ max_nnz`` (order-3 harmonics bounded by the
    coupling triangle inequality ``l ≤ 2·max_l(in) + max_l(out)``). Returns
    ``(irreps_sh = Y_0…Y_L, paths)``, ``L`` the highest degree any kept path uses."""
    l_cap = l_max if max_nnz is None else max(l_max, 2 * _max_l(in_flat) + _max_l(filter_out))
    sh = Irreps(
        O3,
        [
            (1, O3.parse(f"{degree}{'e' if degree % 2 == 0 else 'o'}"))
            for degree in range(l_cap + 1)
        ],
    )
    keep = [
        p
        for p in _paths(in_flat, sh, filter_out, None)
        if (p.order == 2 and p.l_sh <= l_max)
        or (p.order == 3 and max_nnz is not None and p.nnz <= max_nnz)
    ]
    if not keep:
        raise ValueError(f"no order-2/3 paths for l_max={l_max}, max_nnz={max_nnz}")
    big_l = max(p.l_sh for p in keep)
    if big_l < 1:
        raise ValueError(f"l_max={l_max} admits no l≥1 harmonic path — raise it")
    out_irs = sorted({p.ir_out for p in keep})
    keep.sort(key=lambda p: (out_irs.index(p.ir_out), p.order, p.legs))
    return Irreps(O3, list(sh)[: big_l + 1]), keep


class MessagePassing(eqx.Module):
    """Pure-Loom coupling core for every message-passing arm; the wrapper fixes
    ``sum_paths`` and converts irreps/dict keys.

    Two per-edge fields are source-polymorphic, each set once at construction and
    never branched on in the coupling core:

    - the harmonics ``Y``: ``external_harmonics`` reads an ``edge_sph`` operand
      (``{ir: [E, ir.dim, 1]}``, lane-broadcast); otherwise ``Y`` is evaluated
      in-kernel from ``edge_vectors``;
    - the per-path radial weight ``R``: ``external_radial`` reads an ``mp_weights``
      operand (flat ``[E, n_paths·mul]``, path-major) as the per-(edge, path, channel)
      weight; otherwise a parametric family (``free_knot`` / ``bump``) evaluates it
      in-kernel, optionally times a per-path ``gain``.

    Standard MACE is both external: the ``mp_weights`` operand is the per-path weight
    and ``edge_sph`` the harmonics, reproducing the former ``channelwise_mp`` up to
    floating-point rounding (same path order, same tensor products, same ``epsilon``).
    """

    name: str = eqx.field(static=True)
    irreps_out: Irreps = eqx.field(static=True)
    mul: int = eqx.field(static=True)
    in_irs: tuple[Irrep, ...] = eqx.field(static=True)
    sh_irs: tuple[Irrep, ...] = eqx.field(static=True)
    external_harmonics: bool = eqx.field(static=True)  # read edge_sph vs build from edge_vectors
    external_radial: bool = eqx.field(static=True)  # read mp_weights vs a parametric family
    paths: tuple = eqx.field(static=True)
    # Per-path intermediate irrep ``λ`` of the order-3 tree ``(x_s ⊗ Y)_λ ⊗ x_r``;
    # ``None`` for an order-2 path. Kept beside `paths` (rather than inside it) so the
    # 3-tuple layout `paths` consumers unpack stays as it was. The emission below
    # needs λ to address the first coupling's output column.
    lams: tuple = eqx.field(static=True)
    blocks: tuple[tuple[Irrep, int], ...] = eqx.field(static=True)
    n_paths: int = eqx.field(static=True)
    sum_paths: bool = eqx.field(static=True)
    l_max: int = eqx.field(static=True)  # order-2 harmonic degree
    max_nnz: int | None = eqx.field(static=True)  # order-3 nnz budget (None → no order-3)
    shared_t: bool = eqx.field(static=True)  # radial shape shared per channel vs per path
    r_max: float = eqx.field(static=True)
    radial_kind: str = eqx.field(static=True)  # "free_knot" | "bump"
    geometry: Literal["normalized_harmonics", "sqrt_free_solid_harmonics"] = eqx.field(static=True)
    scale: tuple[tuple[str, float | AlgebraicReal], ...] = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)
    # output activation storage dtype, independent of `precision` (the compute mode).
    # None = transparent: output dtype equals the input/element dtype (`precision.elem`),
    # the least-surprising default the existing configs rely on. An explicit dtype
    # overrides it, decoupling the emitted dtype from the input dtype.
    out_dtype: Any = eqx.field(static=True)
    radial: Any  # _fused_radial strategy: window + shape leaves (t, raw_extent|s_q,s_p)
    gain: Array | None  # [n_paths, mul] per-path amplitude, or None

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        mul: int,
        *,
        scale: float | AlgebraicReal | dict[str, float | AlgebraicReal] = 1.0,
        sum_paths: bool,
        l_max: int,
        max_nnz: int | None = None,
        shared_t: bool = False,
        gain: bool = False,
        external_harmonics: bool = False,
        external_radial: bool = False,
        radial: str = "free_knot",
        r_max: float,
        geometry: Literal[
            "normalized_harmonics", "sqrt_free_solid_harmonics"
        ] = "normalized_harmonics",
        extent_floor: float = 0.1,
        width_scale: float | None = None,
        extent_init: str = "spread",
        full_support_margin: float = 1e-3,
        n_fixed_reach: int = 0,
        precision: Precision,
        out_dtype: Any,
        name: str,
        param_dtype: Any,
        key: Array | None = None,
    ):
        self.name, self.precision, self.sum_paths = (
            name,
            parse_precision(precision),
            bool(sum_paths),
        )
        self.out_dtype = out_dtype
        self.external_harmonics = bool(external_harmonics)
        self.external_radial = bool(external_radial)
        if self.external_radial and gain:
            raise ValueError(
                f"{name}: gain is a parametric-radial amplitude; not with external_radial"
            )
        self.l_max = int(l_max)
        self.max_nnz = None if max_nnz is None else int(max_nnz)
        self.shared_t = bool(shared_t)
        self.r_max = float(r_max)
        self.radial_kind = str(radial)
        if geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
        if geometry == "sqrt_free_solid_harmonics" and self.radial_kind != "free_knot":
            raise ValueError("sqrt_free_solid_harmonics geometry requires radial='free_knot'")
        self.geometry = geometry
        self.mul = int(mul)
        if not self.external_radial and self.mul % 2 != 0:
            raise ValueError(f"mul must be even (±1 shape pairs per knot), got {self.mul}")
        # irreps_in = per-channel type multiset (local mults = non-uniform repetition);
        # `mul` is the channel extent. Flatten to legs; order-2 harmonics ≤ l_max,
        # order-3 graded by max_nnz.
        in_flat = _flatten(irreps_in)
        self.in_irs = tuple(ir for _, ir in in_flat)
        irreps_sh, paths = _derive(in_flat, irreps_out, self.l_max, self.max_nnz)
        self.sh_irs = tuple(ir for _, ir in irreps_sh)
        self.paths = tuple((path.legs, path.ir_out, tuple(path.coefs.items())) for path in paths)
        self.lams = tuple(path.ir_lam for path in paths)
        self.n_paths = len(paths)
        # the radial family (free_knot / bump): owns its window+shape leaves and init,
        # emits R(r) in-poly, maps leaves → kernel operands. shared_t → t is [mul] (one
        # shape per channel), else [n_paths, mul] (per path, same init). When
        # `external_radial`, the per-path weight arrives whole as `mp_weights` and no
        # family is built.
        self.radial = (
            None
            if self.external_radial
            else make_radial(
                self.radial_kind,
                self.mul,
                self.n_paths,
                self.shared_t,
                r_max=self.r_max,
                extent_floor=extent_floor,
                width_scale=width_scale,
                extent_init=extent_init,
                full_support_margin=full_support_margin,
                n_fixed=n_fixed_reach,
                geometry=self.geometry,
                dtype=param_dtype,
            )
        )
        block_counts: dict[Irrep, int] = {}
        for path in paths:
            block_counts[path.ir_out] = block_counts.get(path.ir_out, 0) + 1
        self.blocks = tuple(block_counts.items())
        self.irreps_out = Irreps(
            irreps_in.group,
            [(self.mul if self.sum_paths else count * self.mul, ir) for ir, count in self.blocks],
        )
        # a non-1-mult irreps_out is only allowed if it states the derived output exactly
        if any(m != 1 for m, _ in irreps_out) and irreps_out != self.irreps_out:
            filter_form = Irreps(irreps_in.group, [(1, ir) for _, ir in self.irreps_out])
            raise ValueError(
                f"{name}: irreps_out {irreps_out} matches neither the filter form "
                f"{filter_form} nor the derived output {self.irreps_out}"
            )
        if not isinstance(scale, dict):
            scale = {str(ir): scale for _, ir in self.irreps_out}
        self.scale = tuple(
            (str(ir), s if isinstance(s := scale.get(str(ir), 1.0), AlgebraicReal) else float(s))
            for _, ir in self.irreps_out
        )
        # gain → a per-path amplitude (N(0,1) given a key, else ones — trained or loaded).
        self.gain = (
            None
            if not gain
            else jax.random.normal(key, (self.n_paths, self.mul), param_dtype)
            if key is not None
            else jnp.ones((self.n_paths, self.mul), param_dtype)
        )

    def _node_operand(self, node_feats: dict[Irrep, Array]):
        """Node dict → flattened legs: passthrough when uniform, else split each repeated
        ``node_feats[ir]`` into width-``mul`` segments in leg order."""
        seen, segs = Counter(), []
        for ir in self.in_irs:
            j = seen[ir]
            seen[ir] += 1
            segs.append(node_feats[ir][..., j * self.mul : (j + 1) * self.mul])
        return tuple(segs)

    def __call__(
        self,
        node_feats: dict[Irrep, Array],
        graph: Graph,
        *,
        edge_vectors: Array | None = None,
        edge_sph: dict[Irrep, Array] | None = None,
        mp_weights: Array | None = None,
    ) -> dict[Irrep, Array]:
        # Each operand keeps its OWN dtype — the Loom engine binds every buffer at
        # its storage dtype and accumulates in math_dtype. So the
        # geometry (edge_vectors) and the radial params stay at their (fp32) precision
        # even when the node features are bf16 — no silent down-cast of the
        # precision-sensitive radial front-end. The node features must already be in
        # the precision's element dtype (require_kernel_storage errors otherwise); the
        # output carries that same dtype, the accumulation runs in `precision.accum`.
        feat_dt = jax.tree.leaves(node_feats)[0].dtype
        self.precision.require_kernel_storage(feat_dt, self.name)
        # output activation dtype is its OWN explicit axis, independent of `precision`
        # (which fixes the input element + accumulator). Stated in the config; no
        # implicit default. Set it equal to precision.elem for the transparent case.
        out_dt = jnp.dtype(self.out_dtype)
        md = self.precision.accum
        node = self._node_operand(node_feats)  # node features keep their dtype
        # Operands are a dict keyed by role. The coupling core never inspects the keys;
        # the two per-edge fields are assembled from whichever source is configured, so
        # the emission below is free of any fused/unfused branch.
        operands: dict[str, Any] = {"src": node, "dst": node}
        input_domains: dict[str, Any] = {
            "src": loom.domains["src"],
            "dst": loom.domains["dst"],
        }
        if self.external_harmonics:
            # `sph` arrives in ir_mul layout [E, ir.dim, 1]; the lane axis is 1 and
            # broadcasts over channels, matching the in-kernel harmonics [ir.dim].
            operands["sph"] = tuple(edge_sph[ir] for ir in self.sh_irs)
            input_domains["sph"] = tuple(loom.domains["edge"] for _ in self.sh_irs)
        else:
            radial_argument, harmonic_vector = _coordinates_from_displacements(
                edge_vectors, self.geometry
            )
            operands["harmonic_vector"] = harmonic_vector[..., None]
            input_domains["harmonic_vector"] = loom.domains["edge"]
            operands["radial_argument"] = radial_argument
            input_domains["radial_argument"] = loom.domains["edge"]
        if self.external_radial:
            # The full per-(edge, path, channel) weight, path-major on the lane axis —
            # the same layout channelwise_mp fed as `radial_values` per path.
            operands["mp_weights"] = jnp.reshape(
                mp_weights, (*mp_weights.shape[:-1], self.n_paths, self.mul)
            )
            input_domains["mp_weights"] = loom.domains["edge"]
        else:
            # radial family → (per-channel window arrays, a0, a1); leaves keep their dtype
            window, a0, a1 = self.radial.operands()
            operands["window"] = tuple(win[None] for win in window)
            input_domains["window"] = tuple(loom.domains[:] for _ in window)
            operands["a0"] = a0[None]
            input_domains["a0"] = loom.domains[:]
            operands["a1"] = a1[None]
            input_domains["a1"] = loom.domains[:]
            if self.gain is not None:
                operands["gain"] = self.gain[None]
                input_domains["gain"] = loom.domains[:]
        scales = dict(self.scale)

        def evaluate(values):
            src, dst = values["src"], values["dst"]
            if self.external_harmonics:
                harmonics = {
                    ir: value.astype(md)
                    for ir, value in zip(self.sh_irs, values["sph"], strict=True)
                }
            else:
                harmonic_vector = values["harmonic_vector"]
                if self.geometry == "sqrt_free_solid_harmonics":
                    harmonic_vector = harmonic_vector * values["window"][0]
                harmonics = e3.jax.spherical_harmonics(
                    {_VECTOR: harmonic_vector.astype(md)}, (self.sh_irs[-1].dim - 1) // 2
                )
            if self.external_radial:
                radial_values = values["mp_weights"].astype(md)
                gain_values = None
            else:
                radial_values = self.radial.evaluate(
                    values["radial_argument"].astype(md),
                    *(window.astype(md) for window in values["window"]),
                    values["a0"].astype(md),
                    values["a1"].astype(md),
                )
                gain_values = None if "gain" not in values else values["gain"].astype(md)
            streams = {
                "src": tuple(value.astype(md) for value in src),
                "dst": tuple(value.astype(md) for value in dst),
                "sh": tuple(harmonics[ir] for ir in self.sh_irs),
            }
            # `blocks_by_ir[ir]` is the (ir.dim, n_paths_for_ir) column stack, however
            # it was built; the weighting and output assembly below are shared.
            blocks_by_ir: dict[Irrep, Array] = {}
            if all(len(legs) == 2 for legs, _, _ in self.paths):
                # ORDER 2 — exactly what e3.jax.tensor_product enumerates.
                # Preferred over the scalar loop below: one array op per output irrep
                # instead of one jaxpr equation per nonzero CG term, which is ~2x
                # cheaper to LOWER (and lowering, not XLA, dominates our cold start).
                # Verified against the loop: path order identical for uniform AND
                # repeated-irrep (non-uniform) inputs, forward values BIT-EXACT, force
                # gradient equal to fp32 round-off (7.8e-6 rel). Driven one
                # (left, right) pair at a time because tensor_product keys operands by
                # irrep, so a repeated input irrep would otherwise collapse — the same
                # reason _common._tensor_product_ordered exists.
                keep = [ir for ir, _ in self.blocks]
                columns: dict[Irrep, list[Array]] = {}
                for i_src, ir_src in enumerate(self.in_irs):
                    for i_sh, ir_sh in enumerate(self.sh_irs):
                        pair = e3.jax.tensor_product(
                            {ir_src: streams["src"][i_src]},
                            {ir_sh: streams["sh"][i_sh]},
                            keep,
                        )
                        for ir, value in pair.items():
                            columns.setdefault(ir, []).append(value)
                index = 0
                for ir in sorted(columns):
                    block = jnp.concatenate(columns[ir], axis=1)
                    width = block.shape[1]
                    block = block * (
                        radial_values
                        if self.shared_t
                        else radial_values[index : index + width][None]
                    )
                    if gain_values is not None:
                        block = block * gain_values[index : index + width][None]
                    blocks_by_ir[ir] = block
                    index += width
            else:
                # ORDER 3 PRESENT — build every tree as two chained tensor_products,
                # `(x_s ⊗ Y)_λ` then `⊗ x_r`, instead of emitting one jaxpr equation
                # per nonzero CG coefficient.
                #
                # NOT `e3.triple_product`: that enumerates its own
                # trees and admits them by `max_effective_rank`, while `_paths` filters
                # by nonzero count (`max_nnz`). Swapping the dial would silently change
                # WHICH paths exist and in what order — and path order is the layout the
                # per-path radial and `gain` weights index against, which no
                # fresh-weight test can see. So `_paths` stays the single source of
                # truth for the path SET and its ORDER, and only the emission changes.
                #
                # The first coupling is shared: order-2 paths ARE the intermediate
                # (`_paths` emits `_Path(edge, None, ir_lam, cg1)` from the same `cg1`),
                # and every order-3 tree over the same `(src, sh)` pair reuses it across
                # receiver legs, where the scalar form recomputed the full contraction
                # per path.
                #
                # ★ NOT bit-exact against the scalar form: `_contract` sums over λ in
                # EXACT AlgebraicReal arithmetic and then multiplies once per nonzero, whereas
                # chaining sums over λ numerically at `md` precision. Same value to
                # round-off, different rounding.
                keep = [ir for ir, _ in self.blocks]
                # Ask the first coupling only for the intermediates some path actually
                # uses — `keep` is a filter, so a wider one is wasted work per lane.
                needed: dict[tuple[int, int], set[Irrep]] = {}
                for (legs, ir_out, _), ir_lam in zip(self.paths, self.lams, strict=True):
                    (_, i_src), (_, i_sh) = legs[0], legs[1]
                    needed.setdefault((i_src, i_sh), set()).add(
                        ir_out if ir_lam is None else ir_lam
                    )
                first: dict[tuple[int, int], dict[Irrep, Array]] = {
                    (i_src, i_sh): e3.jax.tensor_product(
                        {self.in_irs[i_src]: streams["src"][i_src]},
                        {self.sh_irs[i_sh]: streams["sh"][i_sh]},
                        sorted(irs),
                    )
                    for (i_src, i_sh), irs in needed.items()
                }
                # One second coupling per (src, sh, λ, dst) actually used by a path.
                second: dict[tuple[int, int, Irrep, int], dict[Irrep, Array]] = {}
                for (legs, _, _), ir_lam in zip(self.paths, self.lams, strict=True):
                    if ir_lam is None:
                        continue
                    (_, i_src), (_, i_sh), (_, i_dst) = legs
                    key = (i_src, i_sh, ir_lam, i_dst)
                    if key in second:
                        continue
                    second[key] = e3.jax.tensor_product(
                        {ir_lam: _only(first[i_src, i_sh], ir_lam)},
                        {self.in_irs[i_dst]: streams["dst"][i_dst]},
                        keep,
                    )
                by_ir: dict[Irrep, list[Array]] = {}
                for path_index, ((legs, ir_out, _), ir_lam) in enumerate(
                    zip(self.paths, self.lams, strict=True)
                ):
                    if ir_lam is None:
                        (_, i_src), (_, i_sh) = legs
                        column = _only(first[i_src, i_sh], ir_out)
                    else:
                        (_, i_src), (_, i_sh), (_, i_dst) = legs
                        column = _only(second[i_src, i_sh, ir_lam, i_dst], ir_out)
                    column = column * (
                        radial_values if self.shared_t else radial_values[path_index]
                    )
                    if gain_values is not None:
                        column = column * gain_values[path_index]
                    by_ir.setdefault(ir_out, []).append(column)
                blocks_by_ir = {ir: jnp.stack(cols, axis=1) for ir, cols in by_ir.items()}

            result = {}
            for ir, count in self.blocks:
                paths = blocks_by_ir[ir]
                scale = scales[str(ir)]
                scale = scale if isinstance(scale, AlgebraicReal) else Fraction(scale)
                factor = algebraic_real.jax.exact(scale, dtype=md)
                if self.sum_paths:
                    factor = factor * algebraic_real.jax.exact(
                        pow(Fraction(1, count), Fraction(1, 2)), dtype=md
                    )
                    paths = jnp.sum(paths, axis=1)
                result[ir] = (paths * factor).astype(out_dt)
            return result

        y = loom.call(
            evaluate,
            operands,
            u=self.mul,
            domain_graph=graph.domains,
            input_domains=input_domains,
            output_domains=loom.domains["dst"],
            compile_options=loom.Compile(),
        )
        if self.sum_paths:
            return y
        return {ir: jnp.reshape(value, (*value.shape[:2], -1)) for ir, value in y.items()}

    def with_cal_scale(self, cal_scale: tuple[tuple[str, float], ...]) -> MessagePassing:
        """Copy rescaled by exact ``1/Fraction(measured)`` per irrep (calibrate_chain hook)."""
        measured = dict(cal_scale)
        scale = tuple(
            (
                k,
                v / Fraction(measured[k])
                if isinstance(v, AlgebraicReal)
                else v / float(measured[k]),
            )
            for k, v in self.scale
        )
        return replace_fields(self, scale=scale)


@register_op("message_passing")
class MessagePassingOp(eqx.Module):
    """Registered message-passing op with fused or supplied edge fields."""

    mp: MessagePassing
    INPUTS: tuple = eqx.field(static=True)

    OUTPUTS = "node_feats"
    CUTOFF_PARAM = "r_max"
    NEIGHBOR_NORM_PARAM = "epsilon"  # 1/avg_num_neighbors; auto-filled from the card

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        epsilon: float = 1.0,
        radial: str = "free_knot",
        irreps_sh: Irreps | None = None,
        l_max: int | None = None,
        max_nnz: int | None = None,
        r_max: float = 0.0,
        gain: bool | None = None,
        sum_paths: bool = False,
        shared_t: bool = True,
        geometry: Literal[
            "normalized_harmonics", "sqrt_free_solid_harmonics"
        ] = "normalized_harmonics",
        precision: Precision,
        out_dtype: Any,
        name: str = "message_passing",
        extent_floor: float = 0.1,
        width_scale: float | None = None,
        extent_init: str = "spread",
        full_support_margin: float = 1e-3,
        n_fixed_reach: int = 0,
        param_dtype: Any,
        key: Array | None = None,
    ):
        external_radial = radial == "external"
        external_harmonics = irreps_sh is not None
        if external_harmonics:
            # The harmonics source only changes where the sh streams come from, not the
            # path enumeration: the core still derives Y_0…Y_L from the top degree, and
            # the supplied irreps_sh must be exactly those contiguous degrees.
            degree = max((ir.dim - 1) // 2 for _, ir in irreps_sh)
            l_max = degree if l_max is None else l_max
        if l_max is None:
            raise ValueError(f"{name}: give irreps_sh (external harmonics) or l_max (in-kernel)")
        if gain is None:
            gain = not external_radial  # a parametric-radial amplitude; not for external mp_weights
        # An external mp_weights operand is per-path (indexed by the path slice), so the
        # coupling core must take its per-path weighting branch, never the shared shape.
        shared_t = False if external_radial else bool(shared_t)
        mul = int(next(m for m, _ in irreps_in))
        in_types = irreps_in.set_mul(1)
        # The requested output is a type filter; path multiplicities are derived.
        out_types = irreps_out.set_mul(1)
        self.mp = MessagePassing(
            in_types,
            out_types,
            mul,
            scale=float(epsilon),
            sum_paths=sum_paths,
            l_max=int(l_max),
            max_nnz=max_nnz,
            shared_t=shared_t,
            gain=bool(gain),
            external_harmonics=external_harmonics,
            external_radial=external_radial,
            radial="free_knot" if external_radial else radial,
            r_max=r_max,
            geometry=geometry,
            extent_floor=extent_floor,
            width_scale=width_scale,
            extent_init=extent_init,
            full_support_margin=full_support_margin,
            n_fixed_reach=n_fixed_reach,
            precision=precision,
            out_dtype=out_dtype,
            name=name,
            param_dtype=param_dtype,
            key=key,
        )
        if external_harmonics:
            supplied = tuple(ir for _, ir in irreps_sh)
            if supplied != self.mp.sh_irs:
                raise ValueError(
                    f"{name}: irreps_sh {supplied} are not the derived harmonics "
                    f"{self.mp.sh_irs} (must be the contiguous degrees Y_0…Y_L)"
                )
        # Read slots follow the sources: node_feats + graph always; edge_sph or
        # edge_vectors for the harmonics; mp_weights or edge_vectors for the radial.
        reads = ["node_feats"]
        if external_harmonics:
            reads.append("edge_sph")
        if external_radial:
            reads.append("mp_weights")
        if not (external_harmonics and external_radial):
            reads.append("edge_vectors")
        reads.append("graph")
        self.INPUTS = tuple(reads)

    @property
    def irreps_out(self) -> Irreps:
        return self.mp.irreps_out

    @property
    def name(self) -> str:
        return self.mp.name

    @property
    def scale(self) -> tuple[tuple[str, float], ...]:
        return self.mp.scale

    @property
    def n_paths(self) -> int:
        return self.mp.n_paths

    @property
    def weight_numel(self) -> int:
        """Elements of the external ``mp_weights`` operand per edge = #paths × mul —
        the width a radial MLP must emit to feed this op."""
        return self.mp.n_paths * self.mp.mul

    def __call__(
        self,
        node_feats: dict[Irrep, Array],
        graph: Graph,
        *,
        edge_vectors: Array | None = None,
        edge_sph: dict[Irrep, Array] | None = None,
        mp_weights: Array | None = None,
    ) -> dict[Irrep, Array]:
        return self.mp(
            node_feats,
            graph,
            edge_vectors=edge_vectors,
            edge_sph=edge_sph,
            mp_weights=mp_weights,
        )

    def with_cal_scale(self, cal_scale: tuple[tuple[str, float], ...]) -> MessagePassingOp:
        return replace_fields(self, mp=self.mp.with_cal_scale(cal_scale))
