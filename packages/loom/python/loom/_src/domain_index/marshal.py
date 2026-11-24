# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from loom._src.domain_index.dag import make_preimages, resolve_index_maps
from loom._src.domain_index.graph import DomainGraph
from loom._src.domain_index.types import (
    IndexMap,
    OperandDomains,
    Preimage,
    flatten_domain_graph_arrays,
    resolve_domains,
    unflatten_domain_graph_arrays,
)


class DomainLayout(NamedTuple):
    """Static layout for one domain graph's dynamic JAX arrays."""

    num_axes: int
    domain_sizes: tuple[tuple[str, int], ...]
    operand_domains: tuple[OperandDomains, ...]
    index_map_keys: tuple[tuple[str, str], ...]
    index_map_injective: tuple[bool, ...]
    index_map_sorted: tuple[bool, ...]
    index_map_unmapped: tuple[bool, ...]
    alignment_names: tuple[tuple[str, ...], ...]


class DomainWire(NamedTuple):
    """Dynamic domain arrays and their static layout for one Loom call."""

    index_map_arrays: list[jax.Array]
    aligned_target_index_arrays: list[jax.Array]
    layout: DomainLayout

    def flat(self) -> list[jax.Array]:
        return [*self.index_map_arrays, *self.aligned_target_index_arrays]


def unmarshal_domains(
    wire: DomainWire,
) -> tuple[dict[tuple[str, str], IndexMap], dict[tuple[str, str], Preimage]]:
    """Reconstruct the maps and preimages carried by one call's wire."""
    layout = wire.layout
    return unflatten_domain_graph_arrays(
        layout.index_map_keys,
        layout.index_map_injective,
        layout.index_map_sorted,
        layout.index_map_unmapped,
        layout.alignment_names,
        wire.index_map_arrays,
        wire.aligned_target_index_arrays,
    )


def _is_none(value):
    return value is None


def marshal_domains(
    inputs: Any,
    outputs: Any,
    *,
    domain_graph: DomainGraph,
    input_domains: Any,
    output_domains: Any,
    num_axes: int,
    num_inputs: int,
    num_outputs: int,
) -> DomainWire:
    domain_sizes = domain_graph.domain_sizes
    flat_inputs = jax.tree.leaves(inputs, _is_none)
    flat_outputs = jax.tree.leaves(outputs, _is_none)
    if len(flat_inputs) != num_inputs:
        raise ValueError(f"inputs has {len(flat_inputs)} leaves, expected {num_inputs}")
    if len(flat_outputs) != num_outputs:
        raise ValueError(f"outputs has {len(flat_outputs)} leaves, expected {num_outputs}")

    operand_domains = tuple(
        jax.tree.leaves(jax.tree.broadcast(input_domains, inputs, _is_none), _is_none)
    ) + tuple(jax.tree.leaves(jax.tree.broadcast(output_domains, outputs, _is_none), _is_none))
    if len(operand_domains) != num_inputs + num_outputs:
        raise ValueError(
            f"input_domains/output_domains has {len(operand_domains)} leaves, "
            f"expected {num_inputs + num_outputs}"
        )
    expanded_domains = []
    for leaf, spec in enumerate(operand_domains):
        try:
            resolved = resolve_domains(spec, num_axes)
        except (TypeError, ValueError) as error:
            raise ValueError(f"input_domains/output_domains leaf {leaf}: {error}") from error
        expanded_domains.append(resolved)
        for axis, domain in enumerate(resolved.domains):
            if domain is not None and domain not in domain_sizes:
                raise ValueError(
                    f"input_domains/output_domains leaf {leaf} axis {axis}: "
                    f"domain {domain!r} not in domain_sizes"
                )

    expanded_domain_rows = tuple(domains.domains for domains in expanded_domains)
    resolved_maps = resolve_index_maps(domain_graph.index_maps, expanded_domain_rows)
    preimages = (
        domain_graph.preimages
        if resolved_maps.keys() == domain_graph.index_maps.keys()
        else make_preimages(resolved_maps, domain_sizes)
    )
    (
        map_keys,
        injective,
        sorted_,
        unmapped,
        alignment_names,
        map_arrays,
        aligned_arrays,
    ) = flatten_domain_graph_arrays(resolved_maps, preimages)
    return DomainWire(
        index_map_arrays=map_arrays,
        aligned_target_index_arrays=aligned_arrays,
        layout=DomainLayout(
            num_axes=num_axes,
            domain_sizes=tuple(sorted(domain_sizes.items())),
            operand_domains=tuple(expanded_domains),
            index_map_keys=map_keys,
            index_map_injective=injective,
            index_map_sorted=sorted_,
            index_map_unmapped=unmapped,
            alignment_names=alignment_names,
        ),
    )
