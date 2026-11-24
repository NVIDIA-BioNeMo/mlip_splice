# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward wrapper around a Chain.

`model_call(chain, batch, num_species, ...)` runs the chain with the
species-sort dance and returns:
  - ``E`` (per-graph) and ``F`` (per-node) via autograd of summed atom_energies
  - ``virial`` (per-graph 3x3) via the same autograd path if requested
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import loom
from jax import Array
from loom import IndexMap

from splice.graph import build_graph

# Trace-time logging of compiled batch shapes — fires once per unique shape
# specialisation that JAX caches. Useful for spotting padding-bucket thrash
# and confirming the JIT cache holds the expected (atoms, edges, graphs)
# tuples. Set ``SPLICE_LOG_COMPILE_SHAPES=0`` to silence.
_LOG_COMPILE_SHAPES = os.environ.get("SPLICE_LOG_COMPILE_SHAPES", "1") != "0"
_COMPILE_COUNT = 0

# Sort edges by destination node so the channelwise-TP scatter is grouped and
# the engine can use each preimage-walk position directly (declared by
# ``is_sorted=True`` on the ``edge → dst`` map). Costs one argsort + a few
# edge-axis gathers per call; net effect is GPU-dependent, so it is an opt-in
# toggle we A/B-measure. Edge order does not change any per-atom/graph output.
_SORT_EDGES_BY_DST = os.environ.get("SPLICE_SORT_EDGES_BY_DST", "0") == "1"

# When the caller guarantees atoms arrive already grouped by species (ascending),
# skip model_call's per-step ``argsort(species)`` and every perm gather/scatter:
# the chain already runs on the species-sorted layout, so all ops (CW, SC,
# indexed-linear, edge gather) are bit-identical. It is a genuine saving only when
# the sorted order is reused across steps — MD / inference with constant species
# on one GPU — so it is opt-in (env default here, or the ``species_presorted``
# (LAMMPS domain-decomposition) path, which has no stable species order.
_SPECIES_PRESORTED = os.environ.get("SPLICE_SPECIES_PRESORTED", "0") == "1"

_REDUCTION_WORKSPACE_BYTES = 64 << 20


def _graph_reduce(graph, values: Array, src_domain: str, dst_domain: str, dst_extent: int) -> Array:
    """Sum ``values`` (leading axis on ``src_domain``) into the target
    ``dst_domain`` through the graph's index-map path — one ``loom``,
    a segment reduction on the engine with NO atomic adds and no XLA scatter.

    The DAG must have one path from ``src_domain`` to ``dst_domain``. A direct arrow
    (``edge → src``, ``edge → dst``, ``dst → graph``) or a unique multi-hop path
    (``edge → graph`` via ``edge → dst → graph``) both work — the DI layer
    composes the maps into one target index, so it stays one kernel. Unmapped
    source elements are absorbing and drop out, so padding edges and nodes
    contribute nothing. The lane function is the identity; the index-map path
    defines the accumulation.
    """
    return loom.call(
        lambda x: x,
        values,
        u=values.shape[-1],
        domain_graph=graph.domains,
        input_domains=loom.domains[src_domain],
        output_domains=loom.domains[dst_domain],
        output_dtypes=values.dtype,
        compile_options=loom.Compile(max_workspace_bytes=_REDUCTION_WORKSPACE_BYTES),
    )


def model_call(
    chain: eqx.Module,
    batch: dict[str, Array],
    num_species: int,
    *,
    compute_virial: bool = False,
    compute_edge_forces: bool = False,
    species_presorted: bool | None = None,
) -> dict:
    """Run chain on a batch.

    Batch keys: nn_vecs[E,3], species[N], inda[E], indb[E], inde[N], nats[G],
    mask[E] (bool|int).

    Returns a dict with ``E``/``F`` (and ``virial`` if requested).
    """
    edge_vectors = batch["nn_vecs"]
    positions = batch["positions"]
    cell = batch["cell"]
    edge_mask = batch["mask"].astype(jnp.bool_)
    num_graphs = batch["nats"].shape[0]
    # Per-graph dataset/fidelity head id (multi-head finetune). Per-graph, so
    # invariant to the per-atom species sort; the multi-head ops gather it to
    # atoms via the (sorted) graph_index. Absent for single-head training.
    head_id = batch.get("head_id")
    # Per-graph total charge / spin multiplicity (OMol-style global conditioning).
    # Per-graph like head_id, so invariant to the per-atom species sort; the
    # embedding op gathers them to atoms via the (sorted) graph_index. Optional:
    # datasets without them train exactly as before.
    charge = batch.get("charge")
    spin = batch.get("spin")
    if _LOG_COMPILE_SHAPES:
        global _COMPILE_COUNT
        _COMPILE_COUNT += 1
        print(
            f"[splice trace #{_COMPILE_COUNT}] "
            f"atoms={positions.shape[0]} edges={edge_vectors.shape[0]} "
            f"graphs={num_graphs} virial={compute_virial}",
            flush=True,
        )

    # Species-sort prep (independent of edge_vectors → outside the grad scope).
    # ``perm``/``perm_inv`` stay ``None`` as an identity sentinel in the presorted
    # fast path — the chain runs on the caller's already-sorted layout, so every
    # gather/scatter below is skipped (not multiplied by an identity perm, which
    # would still emit gather kernels). Bit-identical E/F/virial either way.
    presorted = _SPECIES_PRESORTED if species_presorted is None else species_presorted
    if presorted:
        perm = perm_inv = None
        species_s = batch["species"]
        # Edge indices already reference the sorted atom order; only mask padding.
        senders_s = jnp.where(edge_mask, batch["inda"], -1)
        receivers_s = jnp.where(edge_mask, batch["indb"], -1)
        graph_idx_s = batch["inde"]
    else:
        perm = jnp.argsort(batch["species"])
        perm_inv = jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0]))
        species_s = batch["species"][perm]
        # Mark padding edges (mask == 0) as unmapped (-1) AFTER the species-sort
        # remap — a naive -1 in batch["inda"] would be turned back into a valid
        # index by perm_inv[-1]. Loom then skips these edges entirely (no TP
        # work, no scatter), declared via may_have_unmapped on the edge arrows.
        senders_s = jnp.where(edge_mask, perm_inv[batch["inda"]], -1)
        receivers_s = jnp.where(edge_mask, perm_inv[batch["indb"]], -1)
        graph_idx_s = batch["inde"][perm]
    species_counts = jnp.zeros((num_species,), jnp.int32).at[species_s].add(1)
    num_nodes = int(batch["species"].shape[0])

    # Optionally group edges by destination node: valid edges sorted by dst
    # ascending, the -1 padding edges pushed to the tail — the exact layout
    # ``is_sorted`` + ``may_have_unmapped`` expect for the edge→dst
    # arrow. Reorder every edge-axis array consistently (outputs are per-atom /
    # per-graph, so edge order is invisible to E/F/virial).
    edge_perm = None
    if _SORT_EDGES_BY_DST:
        edge_perm = jnp.argsort(jnp.where(receivers_s >= 0, receivers_s, num_nodes))
        edge_vectors = edge_vectors[edge_perm]
        edge_mask = edge_mask[edge_perm]
        senders_s = senders_s[edge_perm]
        receivers_s = receivers_s[edge_perm]

    # Index-map DAG (edge→src/dst, dst→species, dst→graph). Depends only on the
    # sorted connectivity, not on edge_vectors/positions, so build it ONCE here and
    # reuse it both inside the chain and for the post-grad E/F/virial reductions
    # (which segment-reduce through it via loom — no atomic scatter).
    graph = build_graph(
        IndexMap(senders_s, may_have_unmapped=True),
        IndexMap(
            receivers_s,
            is_sorted=_SORT_EDGES_BY_DST,
            may_have_unmapped=True,
        ),
        IndexMap(species_s, is_sorted=True, may_have_unmapped=False),
        num_nodes,
        num_species,
        graph_index=IndexMap(graph_idx_s, may_have_unmapped=False),
        num_graphs=num_graphs,
    )

    def total_energy(edge_vectors_t: Array, positions_t: Array):
        # Long-range ops use absolute Cartesian positions; short-range ops keep
        # using ``edge_vectors``. Treating the two as independent of each other
        # under autograd lets each contribute to forces through its own path
        # (SR via edge-scatter, LR via direct per-atom grad), with no
        # double-counting because edge vectors are pre-computed by matscipy
        # and not recomputed from positions inside the chain.
        ctx = {
            "species": species_s,
            "species_counts": species_counts,
            "species_perm": perm,
            "species_perm_inv": perm_inv,
            "edge_vectors": edge_vectors_t,
            "positions": positions_t if perm is None else positions_t[perm],
            "cell": cell,
            "senders": senders_s,
            "receivers": receivers_s,
            "graph_index": graph_idx_s,
            "head_id": head_id,
            "charge": charge,
            "spin": spin,
            "num_graphs": num_graphs,
            "dummy_graph_id": num_graphs - 1,
            "graph": graph,
            "atom_energies": jnp.zeros(num_nodes, edge_vectors_t.dtype),
            "num_species": num_species,
            "num_nodes": num_nodes,
            "num_edges": int(edge_vectors_t.shape[0]),
        }
        values = chain(ctx)
        atom_energies = values["atom_energies"]
        return jnp.sum(atom_energies), atom_energies

    (grads, pos_grads), atom_energies = jax.grad(total_energy, argnums=(0, 1), has_aux=True)(
        edge_vectors, positions
    )
    grads = jnp.where(edge_mask[:, None], grads, 0.0)

    # Per-graph energy E_g = sum over the graph's atoms — a dst→graph segment
    # reduction via loom (no atomics). The old XLA scatter serialised all
    # num_nodes atomicAdds onto one bin at num_graphs==1 (~585 us / 2197 atoms — the
    # single largest kernel in the fwd+bwd pass).
    E = _graph_reduce(graph, atom_energies[:, None], "dst", "graph", num_graphs)[:, 0]
    # Force F[k] = sum_{src(e)=k} grad_e - sum_{dst(e)=k} grad_e: two edge→node
    # segment reductions through the existing arrows, atomic-free. Padding edges
    # (src/dst == -1, grad already 0) drop out via may_have_unmapped. Nodes are in
    # species-sorted order; unsort to the caller's order.
    src_sum = _graph_reduce(graph, grads, "edge", "src", num_nodes)
    dst_sum = _graph_reduce(graph, grads, "edge", "dst", num_nodes)
    node_forces = src_sum - dst_sum
    F = (node_forces if perm_inv is None else node_forces[perm_inv]) - pos_grads

    out: dict = {"E": E, "F": F}
    if compute_edge_forces:
        # Per-edge dE/d(edge_vector) (edge_vec = r_receiver - r_sender) + per-atom
        # energy, both in the caller's INPUT order. For the LAMMPS ML-IAP path,
        # which tallies forces+virial internally from per-pair gradients (so it
        # wants the raw edge gradient, not the scattered per-atom F). `grads` is in
        # the (optionally dst-sorted) internal edge order — unpermute if sorted.
        edge_forces = grads if edge_perm is None else jnp.zeros_like(grads).at[edge_perm].set(grads)
        out["edge_forces"] = edge_forces
        out["node_energy"] = (
            atom_energies if perm_inv is None else _unsort_atom_axis(atom_energies, perm_inv)
        )
    if compute_virial:
        # virial = +sum_e (dE/dr_e) (x) r_e = +dE/d(strain) = stress * volume
        # (ASE convention: stress = (1/V) dE/d(strain)). The sign MUST match
        # data.py's target_virial = DFT_stress * volume. A leading minus here
        # trains the model against -target: unlearnable, and NaN at high weight.
        edge_virials = jnp.einsum("ei,ej->eij", grads, edge_vectors)
        edge_virials = jnp.where(edge_mask[:, None, None], edge_virials, 0.0)
        # V_g = sum over the graph's edges — a single edge→graph segment reduction.
        # The DAG has no direct edge→graph arrow, but the DI layer composes the unique
        # path edge→dst→graph into one index (one kernel), and -1 is absorbing so
        # padding edges drop. An edge's endpoints share a graph, so edge→dst→graph
        # gives the same per-graph virial as the old scatter over graph_idx[senders].
        virials = _graph_reduce(graph, edge_virials, "edge", "graph", num_graphs)
        out["virial"] = 0.5 * (virials + jnp.swapaxes(virials, -1, -2))
    return out


def _unsort_atom_axis(value: Array, perm_inv: Array) -> Array:
    """Apply the inverse species-sort permutation to the per-atom (leading) axis,
    returning ``value`` untouched if its leading axis is not the atom axis."""
    n_atoms = perm_inv.shape[0]
    if hasattr(value, "shape") and value.shape and value.shape[0] == n_atoms:
        return value[perm_inv]
    return value
