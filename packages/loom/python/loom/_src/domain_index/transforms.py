# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.extend import core as jex_core

from loom._src.domain_index.types import OperandDomains


def domain_batching_rule(
    prim: jex_core.Primitive,
    n_inputs: Callable[[dict[str, Any]], int],
    args: Sequence[jax.Array],
    batch_dims: Sequence[int | None],
    params: dict[str, Any],
):
    n_in = n_inputs(params)
    in_args, aux_args = list(args[:n_in]), args[n_in:]
    in_dims, aux_dims = list(batch_dims[:n_in]), batch_dims[n_in:]

    if any(d is not None for d in aux_dims):
        raise NotImplementedError("vmap over index maps / preimages is not supported")

    mapped_sizes = [a.shape[d] for a, d in zip(in_args, in_dims, strict=True) if d is not None]
    if not mapped_sizes:
        outs = prim.bind(*args, **params)
        return outs, [None] * len(outs)

    batch_size = mapped_sizes[0]
    operand_domains = params["layout"].operand_domains
    existing = {k for k, _ in params["layout"].domain_sizes}
    k = 0
    while f"__vmap_{k}" in existing:
        k += 1
    vmap_dom = f"__vmap_{k}"

    new_inputs: list[jax.Array] = []
    new_in_doms: list[OperandDomains] = []
    for x, d, ax in zip(in_args, in_dims, operand_domains[:n_in], strict=True):
        if d is None:
            new_inputs.append(x[None, ...])
            new_in_doms.append(OperandDomains((None, *ax.domains)))
        else:
            new_inputs.append(jnp.moveaxis(x, d, 0))
            new_in_doms.append(OperandDomains((vmap_dom, *ax.domains)))

    new_out_sd = tuple(
        jax.ShapeDtypeStruct((batch_size, *sd.shape), sd.dtype)
        for sd in params["output_shape_dtypes"]
    )
    new_out_doms = [OperandDomains((vmap_dom, *ax.domains)) for ax in operand_domains[n_in:]]
    new_domain_sizes = tuple(
        sorted({**dict(params["layout"].domain_sizes), vmap_dom: batch_size}.items())
    )

    outs = prim.bind(
        *new_inputs,
        *aux_args,
        **{
            **params,
            "layout": params["layout"]._replace(
                num_axes=params["layout"].num_axes + 1,
                domain_sizes=new_domain_sizes,
                operand_domains=tuple(new_in_doms) + tuple(new_out_doms),
            ),
            "output_shape_dtypes": new_out_sd,
            "input_names": tuple(f"vmap/{name}" for name in params["input_names"]),
            "output_names": tuple(f"vmap/{name}" for name in params["output_names"]),
        },
    )
    return outs, [0] * len(outs)


def domain_dce_rule(
    prune_kernel: Callable[[dict[str, Any], list[bool]], tuple[dict[str, Any], list[bool]]],
    n_inputs: Callable[[dict[str, Any]], int],
    used_outputs: list[bool],
    eqn: jex_core.JaxprEqn,
) -> tuple[list[bool], jex_core.JaxprEqn | None]:
    params = eqn.params
    n_in = n_inputs(params)

    if not any(used_outputs) and not eqn.effects:
        return [False] * len(eqn.invars), None

    kernel_overrides, used_inputs = prune_kernel(params, used_outputs)
    used_inputs = list(used_inputs)
    used_aux = [True] * (len(eqn.invars) - n_in)
    keep_operands = used_inputs + list(used_outputs)

    new_eqn = jex_core.JaxprEqn(
        [v for v, k in zip(eqn.invars, used_inputs + used_aux, strict=True) if k],
        [v for v, k in zip(eqn.outvars, used_outputs, strict=True) if k],
        eqn.primitive,
        dict(
            params,
            **kernel_overrides,
            layout=params["layout"]._replace(
                operand_domains=tuple(
                    ax
                    for ax, k in zip(params["layout"].operand_domains, keep_operands, strict=True)
                    if k
                )
            ),
            output_shape_dtypes=tuple(
                sd for sd, u in zip(params["output_shape_dtypes"], used_outputs, strict=True) if u
            ),
            input_names=tuple(
                name for name, used in zip(params["input_names"], used_inputs, strict=True) if used
            ),
            output_names=tuple(
                name
                for name, used in zip(params["output_names"], used_outputs, strict=True)
                if used
            ),
        ),
        eqn.effects,
        eqn.source_info,
        eqn.ctx,
    )
    return used_inputs + used_aux, new_eqn
