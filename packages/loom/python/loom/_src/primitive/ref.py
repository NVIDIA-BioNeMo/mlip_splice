# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from loom._src.domain_index.ref import DomainIndex
from loom._src.domain_index.types import IndexMap


def loom_ref(
    inputs: Sequence[jax.Array],
    outputs: Sequence[jax.ShapeDtypeStruct],
    *,
    jaxpr,
    u: int,
    num_axes: int,
    domain_sizes: dict[str, int],
    index_maps: Mapping[tuple[str, str], IndexMap],
    operand_domains: Sequence[Sequence[str | None]],
) -> list[jax.Array]:
    num_inputs = len(jaxpr.jaxpr.invars)
    domain_index = DomainIndex(num_axes, domain_sizes, index_maps, operand_domains)
    iteration_sizes = domain_index.iteration_sizes
    total = math.prod(iteration_sizes) if iteration_sizes else 1

    flat_inputs = []
    for operand, array in enumerate(inputs):
        gathered = domain_index.gather(array, operand_domains[operand])
        visible = gathered.shape[num_axes:-1]
        field = jnp.broadcast_to(gathered, tuple(iteration_sizes) + visible + (u,))
        field = jnp.moveaxis(field, -1, num_axes)
        flat_inputs.append(field.reshape(total * u, *visible))

    def evaluate(*args):
        return jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

    flat_outputs = jax.vmap(evaluate)(*flat_inputs)
    results = []
    for operand, (value, output) in enumerate(zip(flat_outputs, outputs, strict=True)):
        visible = output.shape[num_axes:-1]
        value = value.reshape(*iteration_sizes, u, *visible)
        value = jnp.moveaxis(value, num_axes, -1)
        if output.shape[-1] == 1:
            value = value.sum(axis=-1, keepdims=True)

        output_domains = operand_domains[num_inputs + operand]
        reduce_axes = tuple(
            axis
            for axis in range(num_axes)
            if output_domains[axis] is None or domain_index.iteration_domain_per_axis[axis] is None
        )
        if reduce_axes:
            value = value.sum(axis=reduce_axes, keepdims=True)
        results.append(domain_index.scatter(value, output_domains).astype(output.dtype))
    return results
