# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Padding utilities: extend a variable-shape batch to fixed bucket sizes.

Strategy: every batch has a single **dummy graph** at the last position.
Padding atoms are appended at the end of the atom arrays with
``graph_index = dummy_graph_id``. Padding edges connect padding atoms to
padding atoms so they never perturb real-atom features. Every op in the
chain runs on the padded data normally — there is no padding-aware code
inside the chain. The loss is the only place that filters out the dummy
graph (``E[:-1]``) and padding atoms (``inde != dummy_graph_id``).

Bucket sizes are integer multiples of a base, so the JIT cache holds at
most one specialisation per (atom_bucket, edge_bucket, graph_bucket) tuple.

Edge cases the bucketing handles:
- isolated atom (a real graph with 1 atom, 0 edges): works as-is.
- exact-fit (n_real == k * base): we round up to ``ceil((n+1)/base)*base`` so
  there is always at least one padding atom slot for the dummy graph to
  hold padding edges. Tiny waste in the rare exact-fit case.

The raw batch is expected as a dict[str, Array] with keys:
  nn_vecs [N_edges, 3], species [N_atoms], inda [N_edges] (senders),
  indb [N_edges] (receivers), inde [N_atoms] (graph index per atom),
  nats [N_graphs] (atoms per graph).

The padded batch has the same keys at the padded shapes, plus:
  mask [N_edges_padded] (1 for real edges, 0 for padding edges),
  dummy_graph_id (int).
"""

from typing import Any

import numpy as np
from jax import Array


def _round_up(n: int, base: int) -> int:
    return ((n + base - 1) // base) * base


# Minimum number of padding atoms — guards the edge case where
# ``n_real_atoms + 1`` happens to be an exact multiple of ``base_atoms``,
# which would leave only one padding atom. Without enough padding atoms,
# the round-robin scatter in ``pad_batch`` collapses (all padding edges
# pile onto a single receiver) and the Loom message_passing scatter
# kernel hits atomic-add contention → ~2.2× slowdown on real data
# (2026-05-29 nsys profile). 16 is enough to keep contention well below
# real graphs' natural maximum (~50-100 edges per receiver).
_MIN_PAD_ATOMS = 16


def choose_bucket_sizes(
    n_real_atoms: int,
    n_real_edges: int,
    *,
    base_atoms: int = 256,
    base_edges: int = 4096,
) -> tuple[int, int]:
    """Pick padded sizes for (atoms, edges), each a multiple of its base.

    Atoms round up so the bucket has at least ``_MIN_PAD_ATOMS`` padding
    slots (the dummy graph holds padding edges there). Edges round up
    from ``n`` directly because padding edges live entirely within the
    dummy graph.

    The graph count is not bucketed: the caller picks ``n_graphs_padded =
    batch_size + 1`` (one slot for the dummy graph) and keeps it fixed
    across batches.
    """
    n_atoms_padded = _round_up(n_real_atoms + _MIN_PAD_ATOMS, base_atoms)
    n_edges_padded = _round_up(n_real_edges, base_edges)
    return n_atoms_padded, n_edges_padded


def pad_batch(
    raw: dict[str, Any],
    *,
    n_atoms_padded: int,
    n_edges_padded: int,
    n_graphs_padded: int,
) -> dict[str, Any]:
    """Pad raw to fixed sizes; place all padding into a dummy graph at the end.

    Padding atoms reuse species_id=0 (any real species would work; their
    energy contributions land in the dummy graph regardless via
    ``inde = dummy_graph_id`` and get dropped by the loss).

    Padding edges connect padding atoms to themselves with ``edge_vec=[1,0,0]``
    (any non-zero direction avoids ``0/0`` in spherical harmonics
    normalisation). They only contribute to dummy-graph features.
    """
    # Stay in NumPy through the whole assembly. Each `jnp.concatenate` would
    # trigger a JAX trace + CPU↔GPU shuttle and adds ~25 ms per call; for tiny
    # arrays like ours that turns into hundreds of ms per batch. The chain's
    # JIT will pull the final dict from CPU on first call.
    species = np.asarray(raw["species"])
    positions = np.asarray(raw["positions"])
    cell = np.asarray(raw["cell"])  # [G_real, 3, 3]
    nn_vecs = np.asarray(raw["nn_vecs"])
    inda = np.asarray(raw["inda"])
    indb = np.asarray(raw["indb"])
    inde = np.asarray(raw["inde"])
    nats = np.asarray(raw["nats"])

    n_real_atoms = species.shape[0]
    n_real_edges = nn_vecs.shape[0]
    n_real_graphs = nats.shape[0]

    if n_atoms_padded <= n_real_atoms:
        raise ValueError(
            f"n_atoms_padded ({n_atoms_padded}) must leave room for at least "
            f"one padding atom (n_real_atoms={n_real_atoms})"
        )
    if n_edges_padded < n_real_edges:
        raise ValueError(f"n_edges_padded ({n_edges_padded}) < n_real_edges ({n_real_edges})")
    if n_graphs_padded <= n_real_graphs:
        raise ValueError(
            f"n_graphs_padded ({n_graphs_padded}) must leave at least one slot "
            f"for the dummy graph (n_real_graphs={n_real_graphs})"
        )

    dummy_graph_id = n_graphs_padded - 1
    n_pad_atoms = n_atoms_padded - n_real_atoms
    n_pad_edges = n_edges_padded - n_real_edges
    first_pad_atom = n_real_atoms

    species_padded = np.concatenate([species, np.zeros((n_pad_atoms,), dtype=species.dtype)])
    positions_padded = np.concatenate(
        [positions, np.zeros((n_pad_atoms, 3), dtype=positions.dtype)]
    )
    inde_padded = np.concatenate([inde, np.full((n_pad_atoms,), dummy_graph_id, dtype=inde.dtype)])
    pad_vec = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=nn_vecs.dtype), (n_pad_edges, 1))
    nn_vecs_padded = np.concatenate([nn_vecs, pad_vec], axis=0)
    # Padding-edge endpoints are placeholders: ``model_call`` overwrites them
    # with ``-1`` (after the species-sort remap, gated by ``mask``) so Loom
    # skips them entirely — no TP work, no scatter contention. Point them at
    # ``first_pad_atom`` (a valid in-bounds index inside the dummy graph) so any
    # ``-1``-unaware consumer still sees a benign self-loop; the value is inert
    # once ``may_have_unmapped`` is declared. (Previously round-robined across
    # padding atoms to dodge atomic-add contention; the ``-1`` skip makes that
    # moot since the edges no longer touch the scatter at all.)
    pad_atom_targets = np.full((n_pad_edges,), first_pad_atom, dtype=inda.dtype)
    inda_padded = np.concatenate([inda, pad_atom_targets])
    indb_padded = np.concatenate([indb, pad_atom_targets])
    n_empty_graphs = n_graphs_padded - n_real_graphs - 1
    nats_padded = np.concatenate(
        [
            nats,
            np.zeros((n_empty_graphs,), dtype=nats.dtype),
            np.asarray([n_pad_atoms], dtype=nats.dtype),
        ]
    )
    # Pad ``cell`` to ``[n_graphs_padded, 3, 3]``. Padding atoms route to the
    # dummy graph and carry zero LR charge after the in-op mask, so the
    # dummy/empty cells just need to be non-singular — identity matches that.
    n_pad_cells = n_graphs_padded - n_real_graphs
    cell_padded = np.concatenate(
        [cell, np.broadcast_to(np.eye(3, dtype=cell.dtype), (n_pad_cells, 3, 3))],
        axis=0,
    )
    mask = np.concatenate(
        [
            np.ones((n_real_edges,), dtype=np.int32),
            np.zeros((n_pad_edges,), dtype=np.int32),
        ]
    )

    padded = {
        "nn_vecs": nn_vecs_padded,
        "species": species_padded,
        "positions": positions_padded,
        "cell": cell_padded,
        "inda": inda_padded,
        "indb": indb_padded,
        "inde": inde_padded,
        "nats": nats_padded,
        "mask": mask,
        "dummy_graph_id": dummy_graph_id,
    }

    # Per-graph charge / spin, when the dataset carries them. Padded to
    # n_graphs_padded with the NEUTRAL SINGLET baseline (charge 0, spin 1) --
    # the same default the embedding op falls back to, and the same one the
    # packers use. Zero-filling spin instead would feed multiplicity 0, which
    # is not a physical value and would be a live input to the dummy graph.
    for _k, _fill in (("charge", 0), ("spin", 1)):
        if _k in raw:
            _v = np.asarray(raw[_k])
            padded[_k] = np.concatenate([_v, np.full((n_pad_cells,), _fill, dtype=_v.dtype)])

    return padded


def real_atoms_mask(inde: Array, dummy_graph_id: int) -> Array:
    """Boolean mask over the atom axis: True for real atoms, False for padding."""
    return inde != dummy_graph_id
