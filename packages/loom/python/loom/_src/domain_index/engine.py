# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax

from loom._src.domain_index.dag import _compose_index_maps
from loom._src.domain_index.types import IndexMap, Preimage


def _index_map(
    index_maps: Mapping[tuple[str, str], IndexMap], source: str, target: str
) -> IndexMap:
    try:
        return index_maps[(source, target)]
    except KeyError:
        raise RuntimeError(
            f"planner slot wants index map {source!r}->{target!r}, "
            "absent from the supplied index_maps"
        ) from None


def _preimage(preimages: Mapping[tuple[str, str], Preimage], source: str, target: str) -> Preimage:
    try:
        return preimages[(source, target)]
    except KeyError:
        raise RuntimeError(
            f"planner slot wants preimage {source!r}->{target!r}, "
            "absent from the supplied preimages"
        ) from None


def marshal_index_bindings(
    slot_recipe: Sequence[tuple[str, str, str, str]],
    index_maps: Mapping[tuple[str, str], IndexMap],
    preimages: Mapping[tuple[str, str], Preimage],
) -> list[tuple[str, jax.Array]]:
    """Resolve planner slots to arrays with the common public port names."""
    arrays: list[tuple[str, jax.Array]] = []
    for kind, source, target, other_target in slot_recipe:
        if kind == "target_indices":
            arrays.append(
                (
                    f"target_indices:{source}->{target}",
                    _index_map(index_maps, source, target).target_indices,
                )
            )
        elif kind == "composed_target_indices":
            arrays.append(
                (
                    f"target_indices:{source}->{target}",
                    _compose_index_maps(source, target, index_maps).astype(jax.numpy.int32),
                )
            )
        elif kind == "preimage":
            preimage = _preimage(preimages, source, target)
            arrays += [
                (f"preimage_offsets:{source}->{target}", preimage.offsets),
                (f"preimage_source_indices:{source}->{target}", preimage.source_indices),
            ]
        elif kind == "preimage_sorted":
            arrays.append(
                (
                    f"preimage_offsets:{source}->{target}",
                    _preimage(preimages, source, target).offsets,
                )
            )
        elif kind == "aligned_target_indices":
            preimage = _preimage(preimages, source, target)
            try:
                aligned = preimage.aligned_target_indices[other_target]
            except KeyError:
                raise RuntimeError(
                    f"planner slot wants aligned target indices "
                    f"{source!r}->{target!r}->{other_target!r}, absent from the preimage"
                ) from None
            arrays.append(
                (
                    f"aligned_target_indices:{source}->{target}->{other_target}",
                    aligned,
                )
            )
        else:
            raise RuntimeError(f"unknown slot kind {kind!r} from the planner")
    return arrays


def marshal_index_arrays(
    slot_recipe: Sequence[tuple[str, str, str, str]],
    index_maps: Mapping[tuple[str, str], IndexMap],
    preimages: Mapping[tuple[str, str], Preimage],
) -> list[jax.Array]:
    return [array for _, array in marshal_index_bindings(slot_recipe, index_maps, preimages)]


def resolve_index_bindings(
    ports,
    index_maps: Mapping[tuple[str, str], IndexMap],
    preimages: Mapping[tuple[str, str], Preimage],
) -> list[jax.Array]:
    """Resolve named auxiliary ports carried by a compiled Recipe."""
    available = {}
    for (source, target), index_map in index_maps.items():
        preimage = preimages[(source, target)]
        available[f"target_indices:{source}->{target}"] = index_map.target_indices
        available[f"preimage_offsets:{source}->{target}"] = preimage.offsets
        available[f"preimage_source_indices:{source}->{target}"] = preimage.source_indices
        for other_target, aligned in preimage.aligned_target_indices.items():
            available[f"aligned_target_indices:{source}->{target}->{other_target}"] = aligned
    resolved = []
    for port in ports:
        name = port.name.split("/", 1)[-1]
        if name.startswith("target_indices:") and name not in available:
            source, target = name.removeprefix("target_indices:").split("->", 1)
            available[name] = _compose_index_maps(source, target, index_maps).astype(
                jax.numpy.int32
            )
        try:
            resolved.append(available[name])
        except KeyError:
            raise RuntimeError(f"recipe wants unknown domain binding {name!r}") from None
    return resolved
