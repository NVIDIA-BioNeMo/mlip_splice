# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from loom._src.domain_index.dag import _compose_index_maps, _find_common_source
from loom._src.domain_index.types import IndexMap


class DomainIndex:
    def __init__(
        self,
        num_axes: int,
        domain_sizes: dict[str, int],
        index_maps: Mapping[tuple[str, str], IndexMap],
        operand_domains: Sequence[Sequence[str | None]],
    ):
        self.num_axes = num_axes
        self.domain_sizes = domain_sizes
        self.index_maps = index_maps

        self.iteration_domain_per_axis: list[str | None] = []
        for axis in range(num_axes):
            used = {domains[axis] for domains in operand_domains if domains[axis] is not None}
            if not used:
                self.iteration_domain_per_axis.append(None)
            elif len(used) == 1:
                self.iteration_domain_per_axis.append(next(iter(used)))
            else:
                self.iteration_domain_per_axis.append(_find_common_source(used, index_maps))

        self._target_indices_cache: dict[tuple[str, str], jax.Array] = {}

    @property
    def iteration_sizes(self) -> tuple[int, ...]:
        return tuple(
            1 if domain is None else self.domain_sizes[domain]
            for domain in self.iteration_domain_per_axis
        )

    def _target_indices(self, source: str, target: str) -> jax.Array:
        key = (source, target)
        if key not in self._target_indices_cache:
            self._target_indices_cache[key] = _compose_index_maps(source, target, self.index_maps)
        return self._target_indices_cache[key]

    def gather(self, array: jax.Array, operand_domains: Sequence[str | None]) -> jax.Array:
        for axis, target in enumerate(operand_domains):
            source = self.iteration_domain_per_axis[axis]
            if target is None or source is None or target == source:
                continue
            target_indices = self._target_indices(source, target)
            taken = jnp.take(array, jnp.maximum(target_indices, 0), axis=axis)
            keep_shape = [1] * taken.ndim
            keep_shape[axis] = target_indices.shape[0]
            keep = jnp.reshape(target_indices >= 0, keep_shape)
            array = jnp.where(keep, taken, jnp.zeros((), taken.dtype))
        return array

    def scatter(self, output: jax.Array, operand_domains: Sequence[str | None]) -> jax.Array:
        for axis, target in enumerate(operand_domains):
            source = self.iteration_domain_per_axis[axis]
            if target is None or source is None or target == source:
                continue
            target_indices = self._target_indices(source, target)
            shape = list(output.shape)
            shape[axis] = self.domain_sizes[target]
            zeros = jnp.zeros(shape, output.dtype)
            output = jnp.moveaxis(output, axis, 0)
            zeros = jnp.moveaxis(zeros, axis, 0)
            safe_indices = jnp.where(target_indices < 0, self.domain_sizes[target], target_indices)
            output = jnp.moveaxis(zeros.at[safe_indices].add(output, mode="drop"), 0, axis)
        return output
