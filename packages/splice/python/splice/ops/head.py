# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heads: produce final per-atom or per-graph outputs from chain features.

* ``energy_head``: extensive-scalar head — accumulates per-atom contributions
  into ``atom_energies`` (later summed per graph to ``E`` in ``forward.py``).
  Used for E/F/virial via autograd.
* ``species_offsets``: per-species learnable energy offset.
* ``atomic_irrep_head``: per-atom non-extensive multi-irrep head — projects
  features into ``target_irreps`` and stores the result in a named ctx slot.
  Used for magmoms, atomic charges, atomic dipoles, POLAR-1 multipoles, and
  multi-fidelity head outputs (``target_irreps=Kx0e``).

Energy head details:
  Expects ``node_feats`` to contain exactly one scalar irrep entry of shape
  ``[num_nodes, 1, 1]`` (ir_mul with ir.dim=1, mul=1), as produced by an
  ``irreps_linear`` readout projecting to ``"1x0e"``.

Multi-head energy:
  The readout projects to ``Hx0e`` (H per-atom energy channels, one per
  dataset/functional head). This op selects the channel of each atom's head
  (``head_id`` is per-graph, gathered to atoms via ``graph_index``) and
  accumulates into ``atom_energies``. Gradients flow only into the selected
  channel, enabling per-head training on head-homogeneous batches.

Multi-head species offsets:
  Per-head extension of ``species_offsets``: ``offsets[head, species]``.
  Each atom's offset uses its graph's head (``head_id[graph_index]``) and
  species. Stays zero during training (per-head E0 is subtracted in the
  loader); seed at inference to recover absolute per-head energies.

Atomic irrep head:
  Parameterized by ``target_irreps`` to handle magmoms/charges (``1x0e``),
  atomic dipoles (``1x1o``), quadrupoles (``1x2e``), POLAR-1 multipoles
  (``Kx0e + Kx1o + Kx2e``), and multi-fidelity channels (``Kx0e``).
  Output shape is ``dict[Irrep, [N, ir.dim, mul]]`` (ir_mul). Optional
  ``sum_to_zero=True`` mean-subtracts per-graph (meaningful for pure-0e
  targets); reserved for charge conservation.

Graph irrep head:
  Mean-pools per-atom features over real atoms per graph (zeros padding atoms
  before pooling; dummy graphs receive only padding, output masked in loss),
  then projects to ``target_irreps``. Used for global non-extensive
  observables: bandgap, polarisability, dielectric tensor. Other pool
  variants (sum-then-mlp, attention) would warrant separate ops.
"""

from __future__ import annotations

# Paper-recipe option (env-gated, default 1.0 == baseline): MACE-style data-derived
# OUTPUT SCALE on the interaction-energy readout (ScaleShiftBlock scale = RMS of
# training forces; shift stays 0 since E0 is subtracted in the loader). µP has no
# such buffer. SPLICE_OUTPUT_SCALE=<rms_forces> emulates it (OMAT RMS ≈ 2.92).
import os as _os
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from e3 import Irrep
from jax import Array

from splice.registry import register_op

_OUTPUT_SCALE = float(_os.environ.get("SPLICE_OUTPUT_SCALE", "1.0"))


@register_op("energy_head")
class EnergyHead(eqx.Module):
    """Squeeze L=0/mul=1 readout into per-node scalar and add to atom_energies."""

    OUTPUTS = "atom_energies"

    def __call__(self, node_feats: dict[Irrep, Array], atom_energies: Array) -> Array:
        scalar = next(iter(node_feats.values()))  # [num_nodes, 1, 1]
        contrib = jnp.squeeze(scalar, axis=(-1, -2)) * _OUTPUT_SCALE
        return atom_energies + contrib


@register_op("species_offsets")
class SpeciesOffsets(eqx.Module):
    """Learnable per-species energy offsets (zero-init); added per-atom.

    Same gather as ``species_embedding``: index the offset table with the
    per-node ``species`` ids from context instead of reconstructing a
    run-length layout from ``species_counts``. Sorted batches match the old
    repeat bit-identically; unsorted ids still hit the right row.
    """

    offsets: Array
    num_species: int = eqx.field(static=True)
    SPECIES_COUNT_PARAM = "num_species"  # data species count; auto-filled from the card

    OUTPUTS = "atom_energies"

    def __init__(self, num_species: int, *, param_dtype: Any):
        self.num_species = num_species
        self.offsets = jnp.zeros((num_species,), dtype=param_dtype)

    def __call__(self, atom_energies: Array, species: Array) -> Array:
        per_node = self.offsets[species].astype(atom_energies.dtype)  # gather, not repeat
        return atom_energies + per_node


@register_op("multihead_energy")
class MultiHeadEnergy(eqx.Module):
    """Select per-head energy channel per atom and accumulate into atom_energies."""

    OUTPUTS = "atom_energies"

    def __call__(
        self,
        node_feats: dict[Irrep, Array],
        atom_energies: Array,
        graph_index: Array,
        head_id: Array,
    ) -> Array:
        per_head = jnp.squeeze(next(iter(node_feats.values())), axis=-2)  # [N, H]
        atom_head = head_id[graph_index]  # [N]
        contrib = jnp.take_along_axis(per_head, atom_head[:, None], axis=1)[:, 0]
        return atom_energies + contrib


@register_op("multihead_species_offsets")
class MultiHeadSpeciesOffsets(eqx.Module):
    """Per-head, per-species learnable energy offsets added per-atom."""

    offsets: Array  # [num_heads, num_species]
    num_species: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    OUTPUTS = "atom_energies"

    def __init__(self, num_species: int, num_heads: int, *, param_dtype: Any):
        self.num_species = num_species
        self.num_heads = num_heads
        self.offsets = jnp.zeros((num_heads, num_species), dtype=param_dtype)

    def __call__(
        self,
        atom_energies: Array,
        species: Array,
        graph_index: Array,
        head_id: Array,
    ) -> Array:
        atom_head = head_id[graph_index]  # [N]
        per_node = self.offsets[atom_head, species].astype(atom_energies.dtype)
        return atom_energies + per_node
