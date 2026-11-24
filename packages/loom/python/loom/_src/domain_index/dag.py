# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from loom._src.domain_index.types import IndexMap, Preimage


def make_index_map_dag(
    domain_sizes: Mapping[str, int],
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> dict[tuple[str, str], IndexMap]:
    """Normalize one declared DAG without inferring any map contract."""
    normalized = {}
    for key, index_map in index_maps.items():
        if not isinstance(index_map, IndexMap):
            raise TypeError(
                f"DomainGraph index_maps[{key!r}] must be an IndexMap, "
                f"got {type(index_map).__name__}"
            )
        normalized[key] = IndexMap(
            jnp.asarray(index_map.target_indices, dtype=jnp.int32),
            bool(index_map.is_injective),
            bool(index_map.is_sorted),
            bool(index_map.may_have_unmapped),
        )
    _validate_dag(dict(domain_sizes), normalized)
    return normalized


def resolve_index_maps(
    index_maps: Mapping[tuple[str, str], IndexMap],
    operand_domains: Sequence[Sequence[str | None]],
) -> dict[tuple[str, str], IndexMap]:
    """Add only unique compositions required by the operand axes."""
    resolved = dict(index_maps)
    if not operand_domains:
        return resolved
    num_axes = len(operand_domains[0])
    for axis in range(num_axes):
        domains = {row[axis] for row in operand_domains if row[axis] is not None}
        if len(domains) < 2:
            continue
        source = _find_common_source(domains, index_maps)
        for target in domains:
            key = (source, target)
            if source == target or key in resolved:
                continue
            path = _find_index_map_path(source, target, index_maps)
            if path is None:
                raise RuntimeError(f"reachable pair {source!r}->{target!r} has no path")
            resolved[key] = IndexMap(
                _compose_path(path),
                is_injective=all(index_map.is_injective for index_map in path),
                is_sorted=all(index_map.is_sorted for index_map in path),
                may_have_unmapped=any(index_map.may_have_unmapped for index_map in path),
            )
    return resolved


def make_preimages(
    index_maps: Mapping[tuple[str, str], IndexMap],
    domain_sizes: Mapping[str, int],
) -> dict[tuple[str, str], Preimage]:
    """Derive each resolved map's preimage and target-index alignments."""
    targets_by_source: dict[str, list[str]] = {}
    for source, target in index_maps:
        targets_by_source.setdefault(source, []).append(target)

    preimages = {}
    for (source, target), index_map in index_maps.items():
        target_indices = index_map.target_indices
        source_indices = _preimage_source_indices(
            target_indices, domain_sizes[target], index_map.is_sorted
        )
        offsets = _preimage_offsets(target_indices, domain_sizes[target])
        aligned = {
            other_target: index_maps[(source, other_target)].target_indices[source_indices]
            for other_target in targets_by_source[source]
            if other_target != target
        }
        preimages[(source, target)] = Preimage(offsets, source_indices, aligned)
    return preimages


def _preimage_source_indices(
    target_indices: jax.Array, target_size: int, is_sorted: bool
) -> jax.Array:
    if is_sorted:
        return jnp.arange(target_indices.shape[0], dtype=jnp.int32)
    keys = jnp.where(target_indices < 0, target_size, target_indices)
    return jnp.argsort(keys, stable=True).astype(jnp.int32)


def _preimage_offsets(target_indices: jax.Array, target_size: int) -> jax.Array:
    keys = jnp.where(target_indices < 0, target_size, target_indices)
    counts = jnp.bincount(keys, length=target_size + 1)[:target_size]
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(counts).astype(jnp.int32)])


def _validate_dag(
    domain_sizes: dict[str, int],
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> None:
    adjacency: dict[str, list[str]] = {}
    for key, index_map in index_maps.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"DomainGraph index-map key must be (source, target), got {key!r}")
        source, target = key
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError(f"DomainGraph index-map names must be strings, got {key!r}")
        if source == target:
            raise ValueError(f"index map {source!r}->{target!r}: self-loop")
        for name in key:
            if name not in domain_sizes:
                raise ValueError(f"index map {source!r}->{target!r}: domain {name!r} has no size")
        expected = (domain_sizes[source],)
        shape = jnp.shape(index_map.target_indices)
        if shape != expected:
            raise ValueError(
                f"index map {source!r}->{target!r}: target_indices shape {shape}, "
                f"expected {expected}"
            )
        adjacency.setdefault(source, []).append(target)

    color = dict.fromkeys(domain_sizes, 0)
    path: list[str] = []

    def visit(domain: str) -> None:
        if color[domain] == 1:
            start = path.index(domain)
            raise ValueError("index-map DAG has a cycle: " + " -> ".join([*path[start:], domain]))
        if color[domain] == 2:
            return
        color[domain] = 1
        path.append(domain)
        for target in adjacency.get(domain, ()):
            visit(target)
        path.pop()
        color[domain] = 2

    for domain in domain_sizes:
        if color[domain] == 0:
            visit(domain)

    def validate_paths(source: str) -> None:
        counts: dict[str, int] = {}

        def count_paths(domain: str) -> None:
            for target in adjacency.get(domain, ()):
                counts[target] = counts.get(target, 0) + 1
                if counts[target] > 1:
                    raise ValueError(
                        f"index-map DAG has multiple directed paths from {source!r} to {target!r}"
                    )
                count_paths(target)

        count_paths(source)

    for source in domain_sizes:
        validate_paths(source)


def _find_index_map_path(
    source: str,
    target: str,
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> list[IndexMap] | None:
    queue: list[tuple[str, list[IndexMap]]] = [(source, [])]
    seen = {source}
    while queue:
        domain, path = queue.pop(0)
        for (map_source, map_target), index_map in index_maps.items():
            if map_source != domain or map_target in seen:
                continue
            next_path = [*path, index_map]
            if map_target == target:
                return next_path
            seen.add(map_target)
            queue.append((map_target, next_path))
    return None


def _compose_path(path: Sequence[IndexMap]) -> jax.Array:
    target_indices = path[0].target_indices
    for index_map in path[1:]:
        target_indices = jnp.where(
            target_indices < 0,
            -1,
            index_map.target_indices[jnp.maximum(target_indices, 0)],
        )
    return target_indices


def _reaches(
    source: str,
    target: str,
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> bool:
    if source == target:
        return True
    seen = {source}
    stack = [source]
    while stack:
        domain = stack.pop()
        for map_source, map_target in index_maps:
            if map_source == domain and map_target not in seen:
                if map_target == target:
                    return True
                seen.add(map_target)
                stack.append(map_target)
    return False


def _find_common_source(
    domains: set[str],
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> str:
    candidates = set(domains)
    candidates.update(source for source, _ in index_maps)
    matches = [
        candidate
        for candidate in candidates
        if all(_reaches(candidate, domain, index_maps) for domain in domains)
    ]
    if not matches:
        raise ValueError(f"no common source domain for {sorted(domains)}")
    in_domains = [match for match in matches if match in domains]
    pool = in_domains if in_domains else matches
    most_specific = [
        candidate
        for candidate in pool
        if not any(other != candidate and _reaches(other, candidate, index_maps) for other in pool)
    ]
    if len(most_specific) != 1:
        raise ValueError(
            f"common source is not unique for {sorted(domains)}: candidates {sorted(most_specific)}"
        )
    return most_specific[0]


def _compose_index_maps(
    source: str,
    target: str,
    index_maps: Mapping[tuple[str, str], IndexMap],
) -> jax.Array:
    if source == target:
        raise ValueError(f"trivial index map {source!r}->{target!r}")
    try:
        return index_maps[(source, target)].target_indices
    except KeyError:
        pass
    path = _find_index_map_path(source, target, index_maps)
    if path is None:
        raise ValueError(f"no index-map path from {source!r} to {target!r}")
    return _compose_path(path)
