# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import EllipsisType, MappingProxyType

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IndexMap:
    """A source-domain element's target-domain index and static contracts.

    ``is_injective`` applies only to mapped entries; any number of entries may
    equal ``-1``. ``is_sorted`` means valid target indices are nondecreasing and
    are followed only by a ``-1`` suffix. These declarations are trusted and are
    never inferred or validated. ``may_have_unmapped`` declares whether ``-1``
    may occur.
    """

    target_indices: jax.Array
    is_injective: bool = False
    is_sorted: bool = False
    may_have_unmapped: bool = True

    def tree_flatten(self):
        return (self.target_indices,), (
            bool(self.is_injective),
            bool(self.is_sorted),
            bool(self.may_have_unmapped),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        is_injective, is_sorted, may_have_unmapped = aux
        (target_indices,) = children
        return cls(target_indices, is_injective, is_sorted, may_have_unmapped)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Preimage:
    """The public source-index preimage derived for one resolved index map."""

    offsets: jax.Array
    source_indices: jax.Array
    aligned_target_indices: Mapping[str, jax.Array]

    def __init__(
        self,
        offsets: jax.Array,
        source_indices: jax.Array,
        aligned_target_indices: Mapping[str, jax.Array],
    ):
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(
            self,
            "aligned_target_indices",
            MappingProxyType(dict(aligned_target_indices)),
        )

    def tree_flatten(self):
        names = tuple(self.aligned_target_indices)
        children = (
            self.offsets,
            self.source_indices,
            *[self.aligned_target_indices[name] for name in names],
        )
        return children, names

    @classmethod
    def tree_unflatten(cls, names, children):
        offsets, source_indices, *aligned = children
        return cls(offsets, source_indices, dict(zip(names, aligned, strict=True)))


def flatten_domain_graph_arrays(
    index_maps: Mapping[tuple[str, str], IndexMap],
    preimages: Mapping[tuple[str, str], Preimage],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[tuple[str, ...], ...],
    list[jax.Array],
    list[jax.Array],
]:
    map_keys = tuple(index_maps)
    injective = tuple(bool(index_maps[key].is_injective) for key in map_keys)
    sorted_ = tuple(bool(index_maps[key].is_sorted) for key in map_keys)
    unmapped = tuple(bool(index_maps[key].may_have_unmapped) for key in map_keys)
    map_arrays: list[jax.Array] = []
    alignment_names: list[tuple[str, ...]] = []
    aligned_arrays: list[jax.Array] = []
    for key in map_keys:
        index_map = index_maps[key]
        preimage = preimages[key]
        map_arrays += [index_map.target_indices, preimage.offsets, preimage.source_indices]
        names = tuple(preimage.aligned_target_indices)
        alignment_names.append(names)
        aligned_arrays.extend(preimage.aligned_target_indices[name] for name in names)
    return (
        map_keys,
        injective,
        sorted_,
        unmapped,
        tuple(alignment_names),
        map_arrays,
        aligned_arrays,
    )


def unflatten_domain_graph_arrays(
    map_keys: tuple[tuple[str, str], ...],
    injective: tuple[bool, ...],
    sorted_: tuple[bool, ...],
    unmapped: tuple[bool, ...],
    alignment_names: tuple[tuple[str, ...], ...],
    map_arrays: Sequence[jax.Array],
    aligned_arrays: Sequence[jax.Array],
) -> tuple[dict[tuple[str, str], IndexMap], dict[tuple[str, str], Preimage]]:
    index_maps = {}
    preimages = {}
    aligned_at = 0
    for index, key in enumerate(map_keys):
        target_indices, offsets, source_indices = map_arrays[3 * index : 3 * index + 3]
        index_maps[key] = IndexMap(
            target_indices,
            injective[index],
            sorted_[index],
            unmapped[index],
        )
        names = alignment_names[index]
        aligned = aligned_arrays[aligned_at : aligned_at + len(names)]
        aligned_at += len(names)
        preimages[key] = Preimage(
            offsets,
            source_indices,
            dict(zip(names, aligned, strict=True)),
        )
    if aligned_at != len(aligned_arrays):
        raise ValueError("unused aligned target-index arrays")
    return index_maps, preimages


@dataclass(frozen=True)
class OperandDomains:
    domains: tuple[str | None, ...]


@dataclass(frozen=True)
class DomainSpec:
    domains: tuple[str | EllipsisType | None, ...]


class _Domains:
    """Build operand domain specifications with ``domains[...]`` indexing."""

    def __getitem__(self, key) -> DomainSpec:
        """Return one immutable domain specification.

        Strings select named domains, ``:`` leaves an axis unindexed, and
        ``...`` fills all remaining axes with unindexed axes.
        """
        keys = key if isinstance(key, tuple) else (key,)
        entries = []
        for value in keys:
            if isinstance(value, slice):
                if value != slice(None):
                    raise TypeError("domains only supports the full slice `:`")
                entries.append(None)
                continue
            if not (isinstance(value, str) or value is Ellipsis):
                raise TypeError(
                    f"domains entries must be domain names, `:`, or `...`; got {value!r}"
                )
            entries.append(value)
        if entries.count(Ellipsis) > 1:
            raise ValueError("domains supports at most one `...`")
        return DomainSpec(tuple(entries))


def exact_domain_axis_count(spec) -> int | None:
    if not isinstance(spec, DomainSpec):
        raise TypeError(f"expected a `domains[...]` expression, got {type(spec).__name__}")
    return None if Ellipsis in spec.domains else len(spec.domains)


def resolve_domains(spec, num_axes: int) -> OperandDomains:
    if not isinstance(spec, DomainSpec):
        raise TypeError(f"expected a `domains[...]` expression, got {type(spec).__name__}")
    entries = spec.domains
    if Ellipsis not in entries:
        if len(entries) != num_axes:
            raise ValueError(f"has {len(entries)} axes, expected num_axes={num_axes}")
        return OperandDomains(entries)
    explicit = len(entries) - 1
    if explicit > num_axes:
        raise ValueError(
            f"has {explicit} explicit axes around `...`, expected at most num_axes={num_axes}"
        )
    at = entries.index(Ellipsis)
    return OperandDomains(entries[:at] + (None,) * (num_axes - explicit) + entries[at + 1 :])


domains = _Domains()
