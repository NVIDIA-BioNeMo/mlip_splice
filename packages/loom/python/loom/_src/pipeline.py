# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax._src.core import ClosedJaxpr

import loom._native as _engine
from loom._native import Dtype, Port, Recipe
from loom._src.domain_index.engine import marshal_index_bindings, resolve_index_bindings
from loom._src.domain_index.graph import DomainGraph
from loom._src.domain_index.marshal import (
    DomainWire,
    marshal_domains,
    unmarshal_domains,
)
from loom._src.domain_index.types import OperandDomains, exact_domain_axis_count
from loom._src.dtypes import dtype_id
from loom._src.primitive.options import Compile
from loom._src.primitive.stablehlo import lower_module
from loom._src.runtime.instrumentation import compile_sources
from loom._src.runtime.recipes import resolve_compile_options, run_recipe


def _is_none(value):
    return value is None


def _shape_dtype(value) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _pytree_names(tree: Any) -> tuple[str, ...]:
    paths, _ = jax.tree.flatten_with_path(tree)
    return tuple(jax.tree_util.keystr(path, simple=True) or "value" for path, _ in paths)


def _resolve_num_axes(input_domains, output_domains, num_axes: int | None) -> int:
    if num_axes is not None:
        num_axes = int(num_axes)
        if num_axes < 0:
            raise ValueError(f"loom num_axes must be nonnegative, got {num_axes}")
        return num_axes
    specs = jax.tree.leaves(input_domains, _is_none) + jax.tree.leaves(output_domains, _is_none)
    exact_counts = set()
    for leaf, spec in enumerate(specs):
        try:
            count = exact_domain_axis_count(spec)
        except TypeError as error:
            raise ValueError(f"domain spec leaf {leaf}: {error}") from error
        if count is not None:
            exact_counts.add(count)
    if not exact_counts:
        raise ValueError("loom cannot infer num_axes; pass num_axes explicitly")
    if len(exact_counts) != 1:
        raise ValueError(f"loom domain specs have conflicting axis counts {sorted(exact_counts)}")
    return exact_counts.pop()


def _resolve_u(inputs, num_axes: int, u: int | None) -> int:
    lanes = [int(value.shape[-1]) for value in jax.tree.leaves(inputs)]
    wide = {lane for lane in lanes if lane != 1}
    if u is None:
        if len(wide) != 1:
            raise ValueError("loom cannot infer one hidden width; pass u explicitly")
        u = wide.pop()
    u = int(u)
    if u <= 0:
        raise ValueError(f"loom u must be positive, got {u}")
    for leaf, (value, lane) in enumerate(zip(jax.tree.leaves(inputs), lanes, strict=True)):
        if len(value.shape) < num_axes + 1:
            raise ValueError(
                f"loom operand {leaf} has rank {len(value.shape)}, expected at least "
                f"num_axes + 1 = {num_axes + 1}"
            )
        if lane not in (1, u):
            raise ValueError(f"loom operand {leaf} has last extent {lane}, expected 1 or u={u}")
    return u


def trace(fn, inputs: Any, *, num_axes: int) -> tuple[ClosedJaxpr, Any]:
    """Trace one lane program and return its raw ClosedJaxpr and output specs."""
    num_axes = int(num_axes)
    if num_axes < 0:
        raise ValueError(f"loom num_axes must be nonnegative, got {num_axes}")
    examples = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape[num_axes:-1], value.dtype),
        inputs,
    )
    return jax.make_jaxpr(fn, return_shape=True)(examples)


def lower_to_stablehlo(closed_jaxpr: ClosedJaxpr):
    """Lower a raw ClosedJaxpr to a native StableHLO MLIR module."""
    if not isinstance(closed_jaxpr, ClosedJaxpr):
        raise TypeError("loom.lower_to_stablehlo expects a ClosedJaxpr")

    def evaluate(*args):
        return tuple(jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args))

    examples = [
        jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in closed_jaxpr.jaxpr.invars
    ]
    return jax.jit(evaluate).lower(*examples).compiler_ir(dialect="stablehlo")


def lower_to_xir(stablehlo):
    """Lower a native StableHLO MLIR module to a native xir.Module."""
    return lower_module(stablehlo, _engine.XirBuilder())


@dataclass(frozen=True)
class LoweredProgram:
    """A Loom XIR program with its complete physical domain contract."""

    xir: Any
    hidden_width: int
    num_domain_axes: int
    input_specs: Any
    math_output_specs: Any
    output_specs: Any
    domain_graph: DomainGraph
    operand_domains: tuple[OperandDomains, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    _wire: DomainWire
    _input_treedef: Any
    _output_treedef: Any

    @property
    def inputs(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return tuple(jax.tree.leaves(self.input_specs))

    @property
    def outputs(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return tuple(jax.tree.leaves(self.output_specs))

    @property
    def domain_sizes(self):
        return self.domain_graph.domain_sizes

    @property
    def index_maps(self):
        return self.domain_graph.index_maps

    @property
    def preimages(self):
        return self.domain_graph.preimages

    def __repr__(self) -> str:
        domains = ", ".join(f"{name}={size}" for name, size in self.domain_sizes.items())
        return (
            "LoweredProgram("
            f"hidden_width={self.hidden_width}, "
            f"domain_axes={self.num_domain_axes}, "
            f"inputs={len(self.inputs)}, outputs={len(self.outputs)}, "
            f"domains={{{domains}}})"
        )


def lower_domains(
    xir,
    inputs: Any,
    math_outputs: Any,
    *,
    domain_graph: DomainGraph,
    input_domains: Any,
    output_domains: Any,
    reduce_lanes: Any = False,
    output_dtypes: Any = None,
    u: int | None = None,
    num_axes: int | None = None,
) -> LoweredProgram:
    """Attach physical domains, lane reductions, and storage dtypes to XIR."""
    num_axes = _resolve_num_axes(input_domains, output_domains, num_axes)
    u = _resolve_u(inputs, num_axes, u)
    input_specs = jax.tree.map(_shape_dtype, inputs)
    math_output_specs = jax.tree.map(_shape_dtype, math_outputs)
    input_leaves = jax.tree.leaves(input_specs)
    math_leaves, output_treedef = jax.tree.flatten(math_output_specs)
    wire = marshal_domains(
        input_specs,
        math_output_specs,
        domain_graph=domain_graph,
        input_domains=input_domains,
        output_domains=output_domains,
        num_axes=num_axes,
        num_inputs=len(input_leaves),
        num_outputs=len(math_leaves),
    )
    reductions = jax.tree.leaves(
        jax.tree.broadcast(reduce_lanes, math_output_specs, _is_none), _is_none
    )
    dtypes = jax.tree.leaves(
        jax.tree.broadcast(output_dtypes, math_output_specs, _is_none), _is_none
    )
    domain_size_map = dict(wire.layout.domain_sizes)
    output_leaves = []
    for output, domains, reduce, dtype in zip(
        math_leaves,
        wire.layout.operand_domains[len(input_leaves) :],
        reductions,
        dtypes,
        strict=True,
    ):
        if not isinstance(reduce, bool):
            raise TypeError("loom reduce_lanes leaves must be bool")
        output_leaves.append(
            jax.ShapeDtypeStruct(
                tuple(
                    1 if domain is None else domain_size_map[domain] for domain in domains.domains
                )
                + output.shape
                + (1 if reduce else u,),
                output.dtype if dtype is None else jax.numpy.dtype(dtype),
            )
        )
    output_specs = jax.tree.unflatten(output_treedef, output_leaves)
    return LoweredProgram(
        xir=xir,
        hidden_width=u,
        num_domain_axes=num_axes,
        input_specs=input_specs,
        math_output_specs=math_output_specs,
        output_specs=output_specs,
        domain_graph=domain_graph,
        operand_domains=wire.layout.operand_domains,
        input_names=_pytree_names(input_specs),
        output_names=_pytree_names(math_output_specs),
        _wire=wire,
        _input_treedef=jax.tree.structure(input_specs),
        _output_treedef=output_treedef,
    )


def _port(name: str, shaped) -> Port:
    dtype = jax.numpy.dtype(shaped.dtype)
    return Port(name, Dtype(dtype.name, dtype.itemsize), shaped.shape)


def _compile(lowered: LoweredProgram, *, options: Compile):
    wire = lowered._wire
    index_maps, preimages = unmarshal_domains(wire)
    rust_index_maps = [
        (
            source,
            target,
            bool(index_map.is_injective),
            bool(index_map.is_sorted),
            bool(index_map.may_have_unmapped),
        )
        for (source, target), index_map in index_maps.items()
    ]
    operand_specs = [
        (
            list(value.shape[lowered.num_domain_axes : -1]),
            int(value.shape[-1]),
            int(dtype_id(value.dtype)),
        )
        for value in (*lowered.inputs, *lowered.outputs)
    ]
    alignments = [
        (source, target, other_target)
        for (source, target), preimage in preimages.items()
        for other_target in preimage.aligned_target_indices
    ]
    recipe_bytes, slots, profile = compile_sources(
        _engine.compile_loom_to_bytes,
        "loom",
        lowered.xir,
        lowered.xir.text(),
        lowered.hidden_width,
        dict(wire.layout.domain_sizes),
        rust_index_maps,
        [list(domains.domains) for domains in wire.layout.operand_domains],
        len(lowered.inputs),
        operand_specs,
        alignments,
        element_order=None,
        frontier_graph=None,
        dump_ptx=False,
        diagnostics=None,
        **resolve_compile_options(options).source_kwargs(),
    )
    auxiliary = marshal_index_bindings(slots, index_maps, preimages)
    recipe = Recipe(
        bytes(recipe_bytes),
        [
            *(
                _port(name, shaped)
                for name, shaped in zip(lowered.input_names, lowered.inputs, strict=True)
            ),
            *(
                _port(f"aux[{index}]/{name}", shaped)
                for index, (name, shaped) in enumerate(auxiliary)
            ),
        ],
        [
            _port(name, shaped)
            for name, shaped in zip(lowered.output_names, lowered.outputs, strict=True)
        ],
    )
    return recipe, profile


def compile(lowered: LoweredProgram, *, compile_options: Compile | None = None) -> Recipe:
    """Compile and shortlist source candidates, before assembly or measurement.

    A Recipe holds the executable, not runtime policy. Pass runtime settings
    through `execute`'s ``compile_options`` or the enclosing `aot` context.
    """
    recipe, _ = _compile(lowered, options=Compile() if compile_options is None else compile_options)
    return recipe


def _check_inputs(lowered: LoweredProgram, inputs: Any) -> list[Any]:
    if jax.tree.structure(inputs) != lowered._input_treedef:
        raise TypeError("inputs do not have the pytree structure used by lower_domains")
    leaves = jax.tree.leaves(inputs)
    for index, (value, expected) in enumerate(zip(leaves, lowered.inputs, strict=True)):
        if tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype:
            raise ValueError(
                f"input {index} has shape/dtype {value.shape}/{value.dtype}, "
                f"expected {expected.shape}/{expected.dtype}"
            )
    return leaves


def execute(
    lowered: LoweredProgram,
    recipe: Recipe,
    inputs: Any,
    *,
    compile_options: Compile | None = None,
) -> Any:
    """Execute a frozen Recipe with per-call or ambient runtime settings.

    Only ``scheduling_time_limit_ms`` applies here. Source search, resource
    bounds, and shortlist selection have already been applied by `compile`.
    """
    leaves = _check_inputs(lowered, inputs)
    index_maps, preimages = unmarshal_domains(lowered._wire)
    auxiliary = resolve_index_bindings(
        recipe.inputs[len(leaves) :],
        index_maps,
        preimages,
    )
    flat = run_recipe(
        recipe, [*leaves, *auxiliary], list(lowered.outputs), compile_options=compile_options
    )
    return jax.tree.unflatten(lowered._output_treedef, flat)
