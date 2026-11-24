# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Math helpers shared by ops.

`safe_norm` and `polynomial_envelope` are the building blocks of the radial
basis. `flat_to_ir_mul` unpacks a flat kernel output into a dict that stays in
`ir_mul`, our storage convention).
"""

import dataclasses
from typing import Any

import e3
import e3.jax
import jax
import jax.numpy as jnp
import loom
import numpy as np
from e3 import Irrep
from jax import Array


def native_dict_to_flat(irreps, values):
    return jnp.concatenate(
        [
            jnp.reshape(values[ir], (*values[ir].shape[:-2], ir.dim * multiplicity))
            for multiplicity, ir in irreps
        ],
        axis=-1,
    )


def flat_to_native(irreps, data):
    result = {}
    offset = 0
    for multiplicity, ir in irreps:
        size = ir.dim * multiplicity
        result[ir] = jnp.reshape(
            data[..., offset : offset + size],
            (*data.shape[:-1], ir.dim, multiplicity),
        )
        offset += size
    return result


def _tensor_product_ordered(left, right, keep):
    """Tensor product over ORDERED operands that may repeat an irrep type.

    ``e3.jax.tensor_product`` keys its operands by irrep, so a
    repeated irrep would collapse. Channelwise message passing feeds repeated blocks
    (the non-uniform per-l widths), so drive it one ``(left, right)`` pair at a time
    and concatenate the path columns in operand order. ``native_channelwise`` uses this
    as an independent reference for ``message_passing``'s external-radial source: same
    path order, exact column layout.
    """
    # Materialise both operands: `right` is re-scanned for every left entry, so a
    # one-shot iterator (zip) would be drained by the first pass and silently
    # yield only the first left irrep's paths.
    left = tuple(left)
    right = tuple(right)
    paths: dict[Any, list] = {}
    for left_ir, left_value in left:
        for right_ir, right_value in right:
            pair = e3.jax.tensor_product({left_ir: left_value}, {right_ir: right_value}, keep)
            for ir, value in pair.items():
                paths.setdefault(ir, []).append(value)
    return {ir: jnp.concatenate(paths[ir], axis=1) for ir in sorted(paths)}


def native_channelwise(
    weights,
    left,
    right,
    native_in,
    native_sh,
    native_out,
    path_counts,
    *,
    normalize: bool = False,
):
    paths = _tensor_product_ordered(
        tuple(zip(native_in, left, strict=True)),
        tuple(zip(native_sh, right, strict=True)),
        native_out,
    )
    weights = e3.jax.unflatten_by_irrep(weights, dict(path_counts))
    if normalize:
        return e3.jax.linear_combination(paths, weights, normalize=True)
    return {ir: paths[ir] * weights[ir][None] for ir in paths}


def algebraic_squash(x: Array) -> Array:
    """Exp-free smooth map R -> (0, 1). Unlike a sigmoid its tails decay only
    polynomially (~|x|^-3), so reparametrized params keep useful gradients over
    a much wider range — no hard saturation that would freeze a param at its
    bound. One sqrt + one divide."""
    return 0.5 * (1.0 + x / jnp.sqrt(1.0 + x * x))


def inv_algebraic_squash(y: np.ndarray) -> np.ndarray:
    """Inverse of `algebraic_squash` (init only, numpy)."""
    t = 2.0 * y - 1.0
    return t / np.sqrt(1.0 - t * t)


def algebraic_softplus(x: Array) -> Array:
    """Exp-free smooth map R -> (0, ∞), the softplus sibling of `algebraic_squash`:
    same numerator ``x + √(x² + c)`` (here c = 4), denominator 2 instead of the
    squash's ``2√(x²+1)``. φ(0) = 1, φ(x) → x as x → +∞, φ(x) → 0⁺ (~ -1/x → 0) as
    x → -∞ — polynomial tails like the squash, so a reparametrized positive param
    keeps gradient over a wide range with no hard saturation. One sqrt + one add."""
    return 0.5 * (x + jnp.sqrt(x * x + 4.0))


def inv_algebraic_softplus(y: np.ndarray) -> np.ndarray:
    """Inverse of `algebraic_softplus` (init only, numpy): ``y - 1/y`` for ``y > 0``
    (solves ``2y - x = √(x²+4)``). y = 1 ↦ x = 0 (the φ(0)=1 default)."""
    return y - 1.0 / y


def safe_norm(x: Array, eps: float = 0.0, axis: int = -1, keepdims: bool = False) -> Array:
    sq = jnp.sum(jnp.conj(x) * x, axis=axis, keepdims=keepdims) + eps**2
    if eps == 0.0:
        safe_sq = jnp.where(sq == 0.0, 1.0, sq)
        n = jnp.sqrt(safe_sq)
        return jnp.where(sq == 0.0, 0.0, n)
    return jnp.sqrt(sq)


def polynomial_envelope(r: Array, r_max: float, p: int = 5) -> Array:
    """Smoothly-decaying envelope: 1 at r=0, 0 at r>=r_max, p-1 vanishing derivatives at r_max."""
    u = r / r_max
    up = jnp.power(u, p)
    env = 1.0 - 0.5 * (p + 1) * (p + 2) * up + p * (p + 2) * up * u - 0.5 * p * (p + 1) * up * u * u
    return jnp.where(r < r_max, env, 0.0)


def indexed_linear_map(
    x: dict[Irrep, Array],
    weights: dict[Irrep, Array],
    graph: Any,
    *,
    precision: Any = None,
    dst: str = "dst",
    index: str = "species",
) -> dict[Irrep, Array]:
    """Per-index (e.g. per-species) linear via ``loom``:

        out[ir][n] = x[ir][n] @ weights[ir][index[n]]

    ``x``: ``{ir: [N, ir.dim, mul_in]}`` on the ``dst`` domain.
    ``weights``: ``{ir: [num_index, mul_in, mul_out]}`` on the ``index`` domain.

    The engine gathers ``weights[index[n]]`` per node through the graph's
    ``(dst, index)`` arrow and fuses the matmul. Empty index groups are simply
    not gathered — no grouped-GEMM empty-group failure — and there is NO
    cuEquivariance dependency (the op runs entirely on Loom). One ``loom``
    per irrep (block-diagonal linear). ``weights`` need not be materialised
    per-atom; only the small ``[num_index, ...]`` table is held.
    """

    del precision
    out: dict[Irrep, Array] = {}
    for ir, xv in x.items():
        w = weights[ir]
        compute_dtype = jnp.float32 if xv.dtype in (jnp.float16, jnp.bfloat16) else xv.dtype
        dot_algorithm = (
            jax.lax.DotAlgorithmPreset.F64_F64_F64
            if compute_dtype == jnp.float64
            else jax.lax.DotAlgorithmPreset.F32_F32_F32
        )

        def matmul(
            values,
            compute_dtype=compute_dtype,
            dot_algorithm=dot_algorithm,
            output_dtype=xv.dtype,
        ):
            return jnp.matmul(
                values[0].astype(compute_dtype),
                values[1].astype(compute_dtype),
                precision=dot_algorithm,
            ).astype(output_dtype)

        out[ir] = loom.call(
            matmul,
            (xv[..., None], w),
            u=w.shape[-1],
            domain_graph=graph.domains,
            input_domains=(loom.domains[dst], loom.domains[index]),
            output_domains=loom.domains[dst],
            output_dtypes=xv.dtype,
            compile_options=loom.Compile(),
        )
    return out


def replace_fields(module: Any, **changes: Any) -> Any:
    """Copy an eqx.Module and overwrite specified fields.

    Bypasses `__init__` (ops have non-dataclass constructors) and the eqx
    frozen-attribute check via `object.__setattr__` — the standard eqx idiom
    for "replace a static field on a Module with a non-dataclass-shaped
    constructor".
    """
    new_module = object.__new__(type(module))
    for f in dataclasses.fields(module):
        value = changes.get(f.name, getattr(module, f.name))
        object.__setattr__(new_module, f.name, value)
    return new_module


def flat_to_ir_mul(irreps: e3.Irreps, data: Array) -> dict[Irrep, Array]:
    """Unpack a flat kernel output into a dict of `[..., ir.dim, mul]` arrays.

    Keeps the `ir_mul` layout so the next op consumes it without a rearrange.
    """
    result: dict[Irrep, Array] = {}
    offset = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        segment = data[..., offset : offset + size]
        result[ir] = jnp.reshape(segment, (*data.shape[:-1], ir.dim, mul))
        offset += size
    return result
