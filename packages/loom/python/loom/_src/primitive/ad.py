# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.interpreters import ad

from loom._src.jax.autodiff import derive_jvp_jaxpr, derive_transpose_jaxpr


def _bind(
    primitive,
    operands,
    jaxpr,
    params,
    operand_domains,
    outputs,
    *,
    planned_by_transpose=False,
):
    """Bind a derived program with the budget for its transpose order.

    ``ad_depth`` counts both JVP and transpose derivations, so it remains useful
    diagnostic metadata but cannot determine the transpose-rule budget. A JVP
    stays at the current order; a program planned by the transpose rule receives
    options for the next order.
    """
    ad_depth = params.get("ad_depth", 0) + 1
    options = params["options"]
    if planned_by_transpose:
        options = options.advance_transpose()
    return primitive.bind(
        *operands,
        **{
            **params,
            "jaxpr": jaxpr,
            "layout": params["layout"]._replace(operand_domains=tuple(operand_domains)),
            "output_shape_dtypes": tuple(outputs),
            "input_names": tuple(f"input[{index}]" for index in range(len(jaxpr.jaxpr.invars))),
            "output_names": tuple(f"output[{index}]" for index in range(len(outputs))),
            "options": options,
            "ad_depth": ad_depth,
        },
    )


def _is_differentiable(dtype) -> bool:
    return jax.dtypes.issubdtype(dtype, jnp.inexact)


def _zero_from_primal(primal):
    return ad.Zero(jax.typeof(primal).to_tangent_aval())


def _jvp(primals, tangents, **params):
    from loom._src.primitive.primitives import loom_p

    jaxpr = params["jaxpr"]
    domains = params["layout"].operand_domains
    outputs = params["output_shape_dtypes"]
    n_inputs = len(jaxpr.jaxpr.invars)
    inputs, aux = primals[:n_inputs], primals[n_inputs:]
    d_inputs = tangents[:n_inputs]
    primal_out = loom_p.bind(*primals, **params)

    differentiable_outputs = [
        i for i, output in enumerate(outputs) if _is_differentiable(output.dtype)
    ]
    nonzero = [
        i
        for i, dx in enumerate(d_inputs)
        if _is_differentiable(jaxpr.jaxpr.invars[i].aval.dtype) and not isinstance(dx, ad.Zero)
    ]
    if not nonzero or not differentiable_outputs:
        return primal_out, [_zero_from_primal(x) for x in primal_out]

    tangent_jaxpr = derive_jvp_jaxpr(
        jaxpr,
        nonzero_input_idx=nonzero,
        needed_output_idx=differentiable_outputs,
    )
    tangent_inputs = [d_inputs[i] for i in nonzero]
    tangent_domains = [domains[i] for i in nonzero]
    tangent_output_domains = [domains[n_inputs + i] for i in differentiable_outputs]
    new_domains = [
        *domains[:n_inputs],
        *tangent_domains,
        *tangent_output_domains,
    ]
    differentiable_tangents = _bind(
        loom_p,
        [*inputs, *tangent_inputs, *aux],
        tangent_jaxpr,
        params,
        new_domains,
        [outputs[i] for i in differentiable_outputs],
    )
    tangent_out = [_zero_from_primal(x) for x in primal_out]
    for output, tangent in zip(differentiable_outputs, differentiable_tangents, strict=True):
        tangent_out[output] = tangent
    return primal_out, tangent_out


def _transpose(cts, *args, **params):
    from loom._src.primitive.primitives import loom_p

    jaxpr = params["jaxpr"]
    domains = params["layout"].operand_domains
    outputs = params["output_shape_dtypes"]
    n_inputs = len(jaxpr.jaxpr.invars)
    inputs = args[:n_inputs]
    aux = args[n_inputs:]
    undefined = [
        i
        for i, x in enumerate(inputs)
        if ad.is_undefined_primal(x) and _is_differentiable(x.aval.dtype)
    ]
    defined = [i for i in range(n_inputs) if i not in undefined]
    nonzero_ct = [
        i
        for i, ct in enumerate(cts)
        if _is_differentiable(outputs[i].dtype) and not isinstance(ct, ad.Zero)
    ]

    if not undefined or not nonzero_ct:
        result = [ad.Zero(inputs[i].aval) if i in undefined else None for i in range(n_inputs)]
        return tuple(result) + (None,) * len(aux)

    transpose_jaxpr = derive_transpose_jaxpr(
        jaxpr,
        needed_input_idx=undefined,
        nonzero_ct_idx=nonzero_ct,
        cotangent_dtypes=[outputs[i].dtype for i in nonzero_ct],
    )
    out_specs = tuple(
        jax.ShapeDtypeStruct(inputs[i].aval.shape, inputs[i].aval.dtype) for i in undefined
    )
    new_domains = (
        [domains[n_inputs + i] for i in nonzero_ct]
        + [domains[i] for i in defined]
        + [domains[i] for i in undefined]
    )
    transposed = _bind(
        loom_p,
        [*[cts[i] for i in nonzero_ct], *[inputs[i] for i in defined], *aux],
        transpose_jaxpr,
        params,
        new_domains,
        out_specs,
        # The transpose rule is planning a new Loom program here, so attach the
        # next transpose-order budget. JVP-planned programs retain their order.
        planned_by_transpose=True,
    )
    result = [None] * len(args)
    for i, value in zip(undefined, transposed, strict=True):
        result[i] = value
    return tuple(result)


from loom._src.primitive.primitives import loom_p  # noqa: E402

ad.primitive_jvps[loom_p] = _jvp
ad.primitive_transposes[loom_p] = _transpose
