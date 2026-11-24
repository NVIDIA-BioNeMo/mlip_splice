# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import loom
import numpy.testing as npt
from jax._src.core import ClosedJaxpr
from loom import Compile

from marks import gpu_only

pytestmark = gpu_only


def _pipeline(fn, inputs, *, domain_graph, input_domains, output_domains, **kwargs):
    num_axes = kwargs.get("num_axes", 0)
    jaxpr, math_outputs = loom.trace(fn, inputs, num_axes=num_axes)
    stablehlo = loom.lower_to_stablehlo(jaxpr)
    xir = loom.lower_to_xir(stablehlo)
    lowered = loom.lower_domains(
        xir,
        inputs,
        math_outputs,
        domain_graph=domain_graph,
        input_domains=input_domains,
        output_domains=output_domains,
        reduce_lanes=kwargs.get("reduce_lanes", False),
        output_dtypes=kwargs.get("output_dtypes"),
        u=kwargs.get("u"),
        num_axes=num_axes,
    )
    recipe = loom.compile(
        lowered,
        compile_options=Compile(
            search_budget_ms=1,
            max_workspace_bytes=kwargs.get("max_workspace_bytes"),
        ),
    )
    return jaxpr, stablehlo, xir, lowered, recipe


def test_explicit_pipeline_exposes_every_stage(device):
    values = jnp.arange(8, dtype=jnp.float32)
    stages = _pipeline(
        lambda value: value * value,
        values,
        domain_graph=loom.DomainGraph({}),
        input_domains=loom.domains[()],
        output_domains=loom.domains[()],
    )
    jaxpr, stablehlo, xir, lowered, recipe = stages

    assert isinstance(jaxpr, ClosedJaxpr)
    assert "stablehlo.multiply" in str(stablehlo)
    assert xir.entry_name() == "main"
    assert xir.function_names() == ["main"]
    assert "multiply" in xir.operation_names(0)
    assert "hidden_width=8" in repr(lowered)
    assert lowered.inputs[0].shape == values.shape
    assert lowered.outputs[0].shape == values.shape
    assert recipe.outputs[0].shape == values.shape

    result = loom.execute(lowered, recipe, values)
    npt.assert_array_equal(result, values * values)


def test_explicit_pipeline_executes_a_domain_scatter(device):
    target_indices = jnp.asarray([0, 1, 0, 2, 1, 2], jnp.int32)
    values = jnp.arange(12, dtype=jnp.float32).reshape(6, 2, 1)
    graph = loom.DomainGraph(
        {"edge": 6, "node": 3}, {("edge", "node"): loom.IndexMap(target_indices)}
    )
    _, _, _, lowered, recipe = _pipeline(
        lambda value: value,
        values,
        domain_graph=graph,
        input_domains=loom.domains["edge"],
        output_domains=loom.domains["node"],
        num_axes=1,
        u=1,
        max_workspace_bytes=1 << 20,
    )

    result = loom.execute(lowered, recipe, values)
    expected = jnp.zeros((3, 2, 1), jnp.float32).at[target_indices].add(values)
    npt.assert_array_equal(result, expected)


def test_explicit_pipeline_preserves_output_pytrees(device):
    values = jnp.arange(8, dtype=jnp.float32)
    _, _, _, lowered, recipe = _pipeline(
        lambda value: {"linear": value + 1, "square": value * value},
        values,
        domain_graph=loom.DomainGraph({}),
        input_domains=loom.domains[()],
        output_domains={"linear": loom.domains[()], "square": loom.domains[()]},
    )

    result = loom.execute(lowered, recipe, values)
    npt.assert_array_equal(result["linear"], values + 1)
    npt.assert_array_equal(result["square"], values * values)
