# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical execution context: the values dict the loader passes through the chain.

Ops pull auxiliary inputs from this dict by name. Outputs of ops are added to
the same dict under explicit or auto-generated names.

For the MACE reproduction the context carries:
  species          [num_nodes] int — per-atom species id (sorted ascending in
                   model_call). Table lookups (embedding, species offsets) gather
                   this array rather than expanding from counts: the ids are
                   already the row index, so a traced ``jnp.repeat`` would only
                   rebuild it.
  species_counts   [num_species] int — count per species id, still required by
                   grouped GEMM / cue ``Repeats`` (indexed linear), not by
                   embedding or offsets
  species_perm     [num_nodes] int — permutation that sorted species
  species_perm_inv [num_nodes] int — inverse of species_perm
  edge_vectors     [num_edges, 3] — kept in original orientation (autograd target)
  senders          [num_edges] int — remapped via species_perm_inv
  receivers        [num_edges] int — remapped via species_perm_inv
  graph_index      [num_nodes] int — remapped to sorted-node order
  graph            splice.graph.Graph — `edge → src/dst` index-map DAG,
                   consumed by every op that iterates the graph through
                   `loom`
  atom_energies    [num_nodes] — accumulator, init zero
  num_species      int (static)
  num_nodes        int (static)
  num_edges        int (static)
"""

from typing import Any

import jax.numpy as jnp
from jax import Array
from loom import IndexMap

from splice.graph import build_graph


def build_context(
    species: Array,
    edge_vectors: Array,
    senders: Array,
    receivers: Array,
    graph_index: Array,
    num_species: int,
) -> dict[str, Any]:
    perm = jnp.argsort(species)
    species_sorted = species[perm]
    perm_inv = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0]))
    senders = perm_inv[senders]
    receivers = perm_inv[receivers]
    graph_index = graph_index[perm]
    counts = jnp.zeros((num_species,), jnp.int32).at[species_sorted].add(1)
    num_nodes = int(species.shape[0])
    return {
        "species": species_sorted,
        "species_counts": counts,
        "species_perm": perm,
        "species_perm_inv": perm_inv,
        "edge_vectors": edge_vectors,
        "senders": senders,
        "receivers": receivers,
        "graph_index": graph_index,
        "graph": build_graph(
            IndexMap(senders, may_have_unmapped=False),
            IndexMap(receivers, may_have_unmapped=False),
            IndexMap(species_sorted, is_sorted=True, may_have_unmapped=False),
            num_nodes,
            num_species,
        ),
        "atom_energies": jnp.zeros(num_nodes, edge_vectors.dtype),
        "num_species": num_species,
        "num_nodes": num_nodes,
        "num_edges": int(edge_vectors.shape[0]),
    }
