# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import jax

import loom._native as _engine
from loom._native import Dtype, Port, Recipe
from loom._src.domain_index.engine import marshal_index_arrays
from loom._src.domain_index.types import IndexMap, Preimage
from loom._src.dtypes import dtype_id
from loom._src.primitive.options import Compile
from loom._src.primitive.profile import metadata, phase, record
from loom._src.primitive.stablehlo import lower_module
from loom._src.runtime.recipes import compile_recipe, debug_dump_path, run_recipe


def _to_xir(jaxpr):
    def fn(*args):
        return tuple(jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args))

    examples = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in jaxpr.jaxpr.invars]
    with phase("stablehlo_ms"):
        stablehlo = jax.jit(fn).lower(*examples).compiler_ir(dialect="stablehlo")
    with phase("xir_lowering_ms"):
        return lower_module(stablehlo, _engine.XirBuilder())


def _port(name: str, shaped) -> Port:
    dtype = jax.numpy.dtype(shaped.dtype)
    return Port(name, Dtype(dtype.name, dtype.itemsize), shaped.shape)


def loom_engine_call(
    inputs: Sequence[jax.Array],
    outputs: Sequence[jax.ShapeDtypeStruct],
    *,
    jaxpr,
    u: int,
    num_axes: int,
    domain_sizes: dict[str, int],
    operand_domains: Sequence[Sequence[str | None]],
    index_maps: Mapping[tuple[str, str], IndexMap],
    preimages: Mapping[tuple[str, str], Preimage],
    options: Compile,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> list[jax.Array]:
    module = _to_xir(jaxpr)
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
        (list(x.shape[num_axes:-1]), int(x.shape[-1]), int(dtype_id(x.dtype))) for x in inputs
    ] + [(list(o.shape[num_axes:-1]), int(o.shape[-1]), int(dtype_id(o.dtype))) for o in outputs]
    alignments = [
        (source, target, other_target)
        for (source, target), preimage in preimages.items()
        for other_target in preimage.aligned_target_indices
    ]
    cache_status = {}
    diagnostics_path = debug_dump_path("loom_compile")
    diagnostics = None if diagnostics_path is None else {}
    recipe_bytes, slots, rust_profile, capture = compile_recipe(
        _engine.compile_loom_to_bytes,
        "loom",
        module,
        module.text(),
        int(u),
        {name: int(size) for name, size in domain_sizes.items()},
        rust_index_maps,
        [list(domains) for domains in operand_domains],
        len(inputs),
        operand_specs,
        alignments,
        element_order=None,
        frontier_graph=None,
        dump_ptx=False,
        diagnostics=diagnostics,
        cache_status=cache_status,
        compile_options=options,
    )
    metadata(**cache_status)
    if not cache_status["cache_hit"]:
        record(rust_profile)
        if diagnostics_path is not None:
            with open(diagnostics_path, "w") as dump:
                json.dump({"profile": rust_profile, **diagnostics}, dump)
    auxiliary = marshal_index_arrays(slots, index_maps, preimages)
    output_specs = [jax.ShapeDtypeStruct(output.shape, output.dtype) for output in outputs]
    if isinstance(recipe_bytes, Recipe):
        recipe = recipe_bytes
    else:
        recipe = Recipe(
            bytes(recipe_bytes),
            [_port(name, shaped) for name, shaped in zip(input_names, inputs, strict=True)]
            + [_port(f"aux[{index}]", shaped) for index, shaped in enumerate(auxiliary)],
            [_port(name, shaped) for name, shaped in zip(output_names, outputs, strict=True)],
        )
    return run_recipe(
        recipe,
        [*inputs, *auxiliary],
        output_specs,
        capture=capture,
        compile_options=options,
    )
