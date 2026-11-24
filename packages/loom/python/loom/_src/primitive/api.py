# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from loom._src.domain_index.graph import DomainGraph
from loom._src.domain_index.marshal import DomainLayout, marshal_domains
from loom._src.domain_index.ref import DomainIndex
from loom._src.domain_index.types import OperandDomains, domains, exact_domain_axis_count
from loom._src.jax.capture import extract_captures
from loom._src.primitive.options import Compile
from loom._src.primitive.primitives import loom_p
from loom._src.primitive.profile import phase


def _resolve_u(leaves, num_axes: int, u: int | None) -> int:
    lanes = []
    for i, leaf in enumerate(leaves):
        if len(leaf.shape) < num_axes + 1:
            raise ValueError(
                f"loom operand {i} has rank {len(leaf.shape)}, expected at least "
                f"num_axes + 1 = {num_axes + 1}"
            )
        lanes.append(int(leaf.shape[-1]))

    wide = {lane for lane in lanes if lane != 1}
    if u is None:
        if not wide:
            raise ValueError(
                "loom cannot infer U because every operand has last extent 1; pass u explicitly"
            )
        if len(wide) != 1:
            raise ValueError(f"loom operands have conflicting last extents {sorted(wide)}")
        u = wide.pop()
    u = int(u)
    if u <= 0:
        raise ValueError(f"loom u must be positive, got {u}")
    for i, lane in enumerate(lanes):
        if lane not in (1, u):
            raise ValueError(f"loom operand {i} has last extent {lane}, expected 1 or u={u}")
    return u


def _validate_boolean_outputs(math_outputs, storage_outputs, wire, index_maps, u: int) -> None:
    domains = [axes.domains for axes in wire.layout.operand_domains]
    iteration = DomainIndex(
        wire.layout.num_axes, dict(wire.layout.domain_sizes), index_maps, domains
    ).iteration_domain_per_axis
    output_domains = domains[-len(math_outputs) :] if math_outputs else []
    for output, (math_output, storage_output, output_axis_domains) in enumerate(
        zip(math_outputs, storage_outputs, output_domains, strict=True)
    ):
        if jnp.dtype(math_output.dtype) != jnp.dtype(jnp.bool_):
            continue
        if storage_output.shape[-1] == 1 and u != 1:
            raise ValueError(
                f"loom output {output}: boolean hidden-lane reduction has no additive semantics"
            )
        if any(
            domain != iterator
            for domain, iterator in zip(output_axis_domains, iteration, strict=True)
        ):
            raise ValueError(
                f"loom output {output}: boolean DomainIndex reduction has no additive semantics"
            )


def _is_none(x):
    return x is None


def _fill_omitted_domain_specs(specs):
    return jax.tree.map(
        lambda spec: domains[...] if spec is None else spec,
        specs,
        is_leaf=_is_none,
    )


def _resolve_num_axes(input_domains, output_domains, num_axes: int | None) -> int:
    if num_axes is not None:
        num_axes = int(num_axes)
        if num_axes < 0:
            raise ValueError(f"loom num_axes must be nonnegative, got {num_axes}")
        return num_axes
    specs = jax.tree.leaves(input_domains, _is_none) + jax.tree.leaves(output_domains, _is_none)
    exact_counts = set()
    has_domain_spec = False
    for leaf, spec in enumerate(specs):
        if spec is None:
            continue
        has_domain_spec = True
        try:
            count = exact_domain_axis_count(spec)
        except TypeError as error:
            raise ValueError(f"domain spec leaf {leaf}: {error}") from error
        if count is not None:
            exact_counts.add(count)
    if not has_domain_spec:
        return 0
    if not exact_counts:
        raise ValueError(
            "loom cannot infer num_axes because every domain spec contains `...`; "
            "pass num_axes explicitly"
        )
    if len(exact_counts) != 1:
        raise ValueError(f"loom domain specs have conflicting axis counts {sorted(exact_counts)}")
    return exact_counts.pop()


def _trace(fn, inputs, num_axes: int):
    input_examples = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape[num_axes:-1], x.dtype), inputs
    )
    with phase("jax_trace_ms"):
        jaxpr, output_shapes = jax.make_jaxpr(fn, return_shape=True)(input_examples)
        jaxpr, captures = extract_captures(jaxpr)
    return jaxpr, captures, output_shapes


@partial(
    jax.jit,
    static_argnames=(
        "jaxpr",
        "u",
        "layout",
        "output_shape_dtypes",
        "output_treedef",
        "options",
        "input_names",
        "output_names",
    ),
)
def _bind(
    inputs,
    captures,
    domain_arrays,
    *,
    jaxpr,
    u: int,
    layout: DomainLayout,
    output_shape_dtypes,
    output_treedef,
    options: Compile,
    input_names,
    output_names,
):
    input_leaves = jax.tree.leaves(inputs)
    num_axes = layout.num_axes
    capture_inputs = [c.reshape((1,) * num_axes + c.shape + (1,)) for c in captures]
    capture_domains = (OperandDomains((None,) * num_axes),) * len(captures)
    layout = layout._replace(operand_domains=capture_domains + tuple(layout.operand_domains))

    results = loom_p.bind(
        *capture_inputs,
        *input_leaves,
        *domain_arrays,
        jaxpr=jaxpr,
        u=u,
        layout=layout,
        output_shape_dtypes=output_shape_dtypes,
        options=options,
        ad_depth=0,
        input_names=input_names,
        output_names=output_names,
    )
    return jax.tree.unflatten(output_treedef, results)


def _pytree_names(tree: Any) -> tuple[str, ...]:
    paths, _ = jax.tree.flatten_with_path(tree)
    return tuple(jax.tree_util.keystr(path, simple=True) or "value" for path, _ in paths)


def call(
    fn,
    inputs: Any,
    *,
    domain_graph: DomainGraph | None = None,
    input_domains: Any = None,
    output_domains: Any = None,
    reduce_lanes: Any = False,
    output_dtypes: Any = None,
    u: int | None = None,
    num_axes: int | None = None,
    compile_options: Compile | None = None,
) -> Any:
    """Run a StableHLO-expressed map over a hidden lane extent ``U``.

    Every physical operand has ``num_axes`` DomainIndex axes, arbitrary visible
    axes seen by ``fn``, and a final hidden axis of extent 1 or ``U``. Inputs of
    extent 1 broadcast over lanes. ``reduce_lanes`` selects outputs that sum
    over ``U``; the other outputs retain all lanes. Output DomainIndex axes and
    visible axes are inferred from ``output_domains`` and ``fn`` respectively.
    ``output_dtypes`` optionally narrows the completed outputs after all Loom
    reductions; by default each output keeps the dtype returned by ``fn``.
    ``domain_graph`` supplies the finite domain sizes, index maps, and derived preimages.
    Domain specs use ``loom.domains[...]``; ``:`` inserts one axis without a
    domain and ``...`` fills the remaining axes without domains. An omitted
    ``domain_graph`` is empty, and an omitted domain spec leaf is
    ``domains[...]``. If every domain spec is omitted, Loom uses zero
    DomainIndex axes, so ``call(fn, inputs)`` is a lane map.

    ``compile_options`` is a `loom.Compile` describing how to compile this one
    program: source budgets, resource bounds, and runtime tuning policy. Each
    omitted field inherits from `loom.aot`'s `Compile` object, then defaults.
    Explicit ``None`` disables nullable limits. The tuning cache and diagnostic
    paths belong to `loom.aot`, not to `Compile`.
    """
    domain_graph = DomainGraph({}) if domain_graph is None else domain_graph
    options = Compile() if compile_options is None else compile_options
    if not isinstance(options, Compile):
        raise TypeError(
            f"loom compile_options must be a loom.Compile, got {type(options).__name__}"
        )
    input_leaves = jax.tree.leaves(inputs)
    input_names = _pytree_names(inputs)
    resolved_num_axes = _resolve_num_axes(input_domains, output_domains, num_axes)
    input_domains = _fill_omitted_domain_specs(input_domains)
    output_domains = _fill_omitted_domain_specs(output_domains)
    resolved_u = _resolve_u(input_leaves, resolved_num_axes, u)
    jaxpr, captures, math_outputs = _trace(fn, inputs, resolved_num_axes)
    math_output_leaves, output_treedef = jax.tree.flatten(math_outputs)
    output_names = _pytree_names(math_outputs)
    wire = marshal_domains(
        inputs,
        math_outputs,
        domain_graph=domain_graph,
        input_domains=input_domains,
        output_domains=output_domains,
        num_axes=resolved_num_axes,
        num_inputs=len(input_leaves),
        num_outputs=len(math_output_leaves),
    )
    reduce_leaves = jax.tree.leaves(
        jax.tree.broadcast(reduce_lanes, math_outputs, _is_none), _is_none
    )
    for output, reduce in enumerate(reduce_leaves):
        if not isinstance(reduce, bool):
            raise TypeError(
                f"loom reduce_lanes leaf {output}: expected bool, got {type(reduce).__name__}"
            )
    dtype_leaves = jax.tree.leaves(
        jax.tree.broadcast(output_dtypes, math_outputs, _is_none), _is_none
    )
    output_domains_flat = wire.layout.operand_domains[len(input_leaves) :]
    domain_size_map = dict(wire.layout.domain_sizes)
    output_shape_dtypes = tuple(
        jax.ShapeDtypeStruct(
            tuple(1 if domain is None else domain_size_map[domain] for domain in axes.domains)
            + math.shape
            + (1 if reduce else resolved_u,),
            math.dtype if dtype is None else jnp.dtype(dtype),
        )
        for math, axes, reduce, dtype in zip(
            math_output_leaves,
            output_domains_flat,
            reduce_leaves,
            dtype_leaves,
            strict=True,
        )
    )
    _validate_boolean_outputs(
        math_output_leaves,
        output_shape_dtypes,
        wire,
        domain_graph.index_maps,
        resolved_u,
    )
    return _bind(
        inputs,
        captures,
        wire.flat(),
        jaxpr=jaxpr,
        u=resolved_u,
        layout=wire.layout,
        output_shape_dtypes=output_shape_dtypes,
        output_treedef=output_treedef,
        options=options,
        input_names=input_names,
        output_names=output_names,
    )
