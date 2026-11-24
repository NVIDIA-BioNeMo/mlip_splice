# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-graph bundle for lane programs.

The graph contains direct maps for ``edge → src``, ``edge → dst``, and
``dst → species``. It can also contain ``dst → graph``. ``DomainGraph``
derives direct preimages when built and derives only the unique multi-hop maps
required by each Loom call.
"""

from typing import NamedTuple

from loom import DomainGraph, IndexMap


class Graph(NamedTuple):
    domain_sizes: dict[str, int]
    domains: DomainGraph


def build_graph(
    senders: IndexMap,
    receivers: IndexMap,
    species: IndexMap,
    num_nodes: int,
    num_species: int,
    *,
    graph_index: IndexMap | None = None,
    num_graphs: int | None = None,
) -> Graph:
    """Build the model's direct index-map DAG from explicit map contracts."""
    for name, index_map in (
        ("senders", senders),
        ("receivers", receivers),
        ("species", species),
    ):
        if not isinstance(index_map, IndexMap):
            raise TypeError(f"build_graph {name} must be an IndexMap")

    domain_sizes = {
        "edge": int(senders.target_indices.shape[0]),
        "src": num_nodes,
        "dst": num_nodes,
        "species": num_species,
    }
    index_maps = {
        ("edge", "src"): senders,
        ("edge", "dst"): receivers,
        ("dst", "species"): species,
    }
    if graph_index is not None:
        if not isinstance(graph_index, IndexMap):
            raise TypeError("build_graph graph_index must be an IndexMap")
        if num_graphs is None:
            raise ValueError("graph_index needs num_graphs")
        domain_sizes["graph"] = num_graphs
        index_maps[("dst", "graph")] = graph_index
    return Graph(
        domain_sizes=domain_sizes,
        domains=DomainGraph(domain_sizes, index_maps),
    )
