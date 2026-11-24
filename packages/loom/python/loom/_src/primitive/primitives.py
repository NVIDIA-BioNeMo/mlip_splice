# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

from jax import core as jax_core
from jax._src import dispatch
from jax._src.core import ClosedJaxpr as _ClosedJaxpr
from jax.extend import core as jex_core
from jax.interpreters import batching, mlir
from jax.interpreters import partial_eval as pe

from loom._src.domain_index.transforms import domain_batching_rule, domain_dce_rule
from loom._src.domain_index.types import unflatten_domain_graph_arrays
from loom._src.primitive.cuda import loom_engine_call
from loom._src.primitive.ref import loom_ref

loom_p = jex_core.Primitive("loom")
loom_p.multiple_results = True


@loom_p.def_abstract_eval
def _abstract_eval(*args, output_shape_dtypes, **kwargs):
    return [jax_core.ShapedArray(o.shape, o.dtype) for o in output_shape_dtypes]


def _unpack_args(args, jaxpr, layout):
    num_inputs = len(jaxpr.jaxpr.invars)
    inputs = list(args[:num_inputs])
    at = num_inputs
    map_array_count = 3 * len(layout.index_map_keys)
    map_arrays = args[at : at + map_array_count]
    at += map_array_count
    aligned_arrays = args[at:]
    index_maps, preimages = unflatten_domain_graph_arrays(
        layout.index_map_keys,
        layout.index_map_injective,
        layout.index_map_sorted,
        layout.index_map_unmapped,
        layout.alignment_names,
        map_arrays,
        aligned_arrays,
    )
    return inputs, index_maps, preimages


def _impl_cuda(*args, **params):
    inputs, index_maps, preimages = _unpack_args(
        args,
        params["jaxpr"],
        params["layout"],
    )
    return loom_engine_call(
        inputs=inputs,
        outputs=params["output_shape_dtypes"],
        jaxpr=params["jaxpr"],
        u=params["u"],
        num_axes=params["layout"].num_axes,
        domain_sizes=dict(params["layout"].domain_sizes),
        operand_domains=[d.domains for d in params["layout"].operand_domains],
        index_maps=index_maps,
        preimages=preimages,
        options=params["options"],
        input_names=params["input_names"],
        output_names=params["output_names"],
    )


def _impl_ref(*args, **params):
    inputs, index_maps, _ = _unpack_args(
        args,
        params["jaxpr"],
        params["layout"],
    )
    return loom_ref(
        inputs,
        params["output_shape_dtypes"],
        jaxpr=params["jaxpr"],
        u=params["u"],
        num_axes=params["layout"].num_axes,
        domain_sizes=dict(params["layout"].domain_sizes),
        index_maps=index_maps,
        operand_domains=[d.domains for d in params["layout"].operand_domains],
    )


mlir.register_lowering(loom_p, mlir.lower_fun(_impl_cuda, multiple_results=True), "cuda")
mlir.register_lowering(loom_p, mlir.lower_fun(_impl_ref, multiple_results=True), None)
loom_p.def_impl(partial(dispatch.apply_primitive, loom_p))


def _n_inputs(params) -> int:
    return len(params["jaxpr"].jaxpr.invars)


def _prune_kernel(params, used_outputs):
    jaxpr = params["jaxpr"]
    inner, used = pe.dce_jaxpr(jaxpr.jaxpr, list(used_outputs))
    return {"jaxpr": _ClosedJaxpr(inner, jaxpr.consts)}, list(used)


batching.primitive_batchers[loom_p] = lambda args, dims, **params: domain_batching_rule(
    loom_p, _n_inputs, args, dims, params
)
pe.dce_rules[loom_p] = lambda used_outputs, eqn: domain_dce_rule(
    _prune_kernel, _n_inputs, used_outputs, eqn
)

from loom._src.primitive import ad as _ad  # noqa: E402, F401
