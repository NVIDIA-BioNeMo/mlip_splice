# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax

from loom._src.domain_index.dag import make_index_map_dag, make_preimages
from loom._src.domain_index.types import IndexMap, Preimage

IndexMapKey = tuple[str, str]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class DomainGraph:
    """Finite named domains, direct index maps, and all derived preimages."""

    domain_sizes: Mapping[str, int]
    index_maps: Mapping[IndexMapKey, IndexMap]
    preimages: Mapping[IndexMapKey, Preimage]

    def __init__(
        self,
        domain_sizes: Mapping[str, int],
        index_maps: Mapping[IndexMapKey, IndexMap] | None = None,
    ):
        normalized_sizes = _normalize_domain_sizes(domain_sizes)
        normalized_maps = make_index_map_dag(normalized_sizes, dict(index_maps or {}))
        preimages = make_preimages(normalized_maps, normalized_sizes)
        self._set_parts(normalized_sizes, normalized_maps, preimages)

    def tree_flatten(self):
        map_keys = tuple(self.index_maps)
        children = (tuple(self.index_maps.values()), tuple(self.preimages.values()))
        return children, (tuple(self.domain_sizes.items()), map_keys)

    @classmethod
    def tree_unflatten(cls, aux, children):
        size_items, map_keys = aux
        index_maps, preimages = children
        return cls._from_parts(
            dict(size_items),
            dict(zip(map_keys, index_maps, strict=True)),
            dict(zip(map_keys, preimages, strict=True)),
        )

    @classmethod
    def _from_parts(cls, domain_sizes, index_maps, preimages):
        graph = object.__new__(cls)
        graph._set_parts(domain_sizes, index_maps, preimages)
        return graph

    def _set_parts(self, domain_sizes, index_maps, preimages) -> None:
        object.__setattr__(self, "domain_sizes", MappingProxyType(dict(domain_sizes)))
        object.__setattr__(self, "index_maps", MappingProxyType(dict(index_maps)))
        object.__setattr__(self, "preimages", MappingProxyType(dict(preimages)))


def _normalize_domain_sizes(domain_sizes: Mapping[str, int]) -> dict[str, int]:
    normalized = {}
    for name, size in domain_sizes.items():
        if not isinstance(name, str):
            raise TypeError(f"DomainGraph domain names must be strings, got {type(name).__name__}")
        try:
            size = operator.index(size)
        except TypeError as error:
            raise TypeError(
                f"DomainGraph size for {name!r} must be an integer, got {size!r}"
            ) from error
        if size <= 0:
            raise ValueError(f"DomainGraph size for {name!r} must be positive, got {size}")
        normalized[name] = size
    return normalized
