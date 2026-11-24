# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone edge spherical harmonics: ``edge_vectors`` -> per-edge ``edge_sph``
dict in ir_mul layout (``{ir: [num_edges, ir.dim, 1]}``).

The unfused ``message_passing`` source (``radial="external"``) needs the harmonics as
an explicit context slot (the in-kernel ``message_passing`` computes them from
``edge_vectors`` instead). ``e3``'s
value-level ``spherical_harmonics`` evaluates one vector at a time (it enforces
``shape == (ir.dim,)``), so this op ``jax.vmap``s it over the edge axis. It is
built on ``e3`` rather than ``cuex.segmented_polynomial``.
"""

from typing import Any

import e3
import e3.jax
import equinox as eqx
import jax
from e3 import O3, Irrep
from jax import Array

from splice.ops._common import safe_norm
from splice.registry import register_op

_VECTOR = O3.parse("1o")  # spherical harmonics are the O3 (parity-odd) vector


@register_op("sph")
class SphericalHarmonics(eqx.Module):
    """Per-edge solid harmonics of the unit displacement, degrees ``0..max_degree``.
    Reads ``edge_vectors``; writes ``edge_sph``."""

    max_degree: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    OUTPUTS = "edge_sph"

    def __init__(self, max_degree: int, eps: float = 0.0):
        self.max_degree = int(max_degree)
        self.eps = float(eps)

    def __call__(self, edge_vectors: Array) -> dict[Irrep, Array]:
        assert edge_vectors.shape[-1] == 3
        unit = edge_vectors / safe_norm(edge_vectors, self.eps, axis=-1, keepdims=True)

        def _one(vector: Array) -> dict[Irrep, Array]:
            return e3.jax.spherical_harmonics({_VECTOR: vector}, self.max_degree)

        out: dict[Irrep, Any] = jax.vmap(_one)(unit)  # {ir: [num_edges, ir.dim]}
        # ir_mul layout: [num_edges, ir.dim, mul=1]
        return {ir: value[..., None] for ir, value in out.items()}
