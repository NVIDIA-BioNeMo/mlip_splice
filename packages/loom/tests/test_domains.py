# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import loom
import numpy as np
import numpy.testing as npt
import pytest
from loom import Compile, DomainGraph, IndexMap, domains
from loom._src.domain_index.engine import marshal_index_bindings
from loom._src.domain_index.marshal import marshal_domains, unmarshal_domains

from _support import _INTEGER_DTYPES
from marks import gpu_only

pytestmark = pytest.mark.capped_autotune


def test_domains_expand_slice_and_ellipsis(device, dtype):
    nodes, batches, u = 3, 2, 4
    x = jnp.arange(nodes * u, dtype=dtype).reshape(nodes, 1, u)
    y = loom.call(
        lambda value: value,
        x,
        domain_graph=DomainGraph({"node": nodes, "batch": batches}),
        input_domains=domains["node", :],
        output_domains=domains[..., "batch"],
    )
    expected = jnp.broadcast_to(x.sum(axis=0, keepdims=True), (1, batches, u))
    npt.assert_array_equal(y, expected)


def test_omitted_domain_specs_are_unindexed(device, dtype):
    nodes, u = 3, 4
    shared = jnp.arange(u, dtype=dtype).reshape(1, u)
    per_node = jnp.arange(nodes * u, dtype=dtype).reshape(nodes, u)

    nodewise, total = loom.call(
        lambda values: (values[0] + values[1], values[1]),
        (shared, per_node),
        domain_graph=DomainGraph({"node": nodes}),
        input_domains=(None, domains["node"]),
        output_domains=(domains["node"], None),
    )

    npt.assert_array_equal(nodewise, shared + per_node)
    npt.assert_array_equal(total, per_node.sum(axis=0, keepdims=True))


def test_domains_support_arbitrary_domain_names(device, dtype):
    x = jnp.arange(12, dtype=dtype).reshape(3, 1, 4)
    y = loom.call(
        lambda value: value,
        x,
        domain_graph=DomainGraph({"node-type": x.shape[0], "batch": 1}),
        input_domains=domains["node-type", "batch"],
        output_domains=domains["node-type", "batch"],
    )
    npt.assert_array_equal(y, x)


def test_ellipsis_requires_an_exact_axis_count(device, dtype):
    x = jnp.arange(12, dtype=dtype).reshape(3, 1, 4)
    with pytest.raises(ValueError, match="every domain spec contains"):
        loom.call(
            lambda value: value,
            x,
            domain_graph=DomainGraph({"node": x.shape[0]}),
            input_domains=domains["node", ...],
            output_domains=domains["node", ...],
        )

    y = loom.call(
        lambda value: value,
        x,
        num_axes=2,
        domain_graph=DomainGraph({"node": x.shape[0]}),
        input_domains=domains["node", ...],
        output_domains=domains["node", ...],
    )
    npt.assert_array_equal(y, x)


def test_domains_reject_alternate_syntaxes():
    with pytest.raises(TypeError, match="full slice"):
        domains[1:]
    with pytest.raises(ValueError, match="at most one"):
        domains[..., ...]
    with pytest.raises(TypeError, match="domain names"):
        domains[None]
    assert not hasattr(domains, "node")
    with pytest.raises(TypeError):
        domains["node"]["batch"]


def test_domain_graph_derives_public_preimages_and_is_a_jax_pytree():
    source = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    destination = jnp.asarray([1, 0, 1, 0], dtype=jnp.int32)
    domain_graph = DomainGraph(
        {"edge": 4, "source": 2, "destination": 2},
        {
            ("edge", "source"): IndexMap(source),
            ("edge", "destination"): IndexMap(destination),
        },
    )

    preimage = domain_graph.preimages[("edge", "source")]
    npt.assert_array_equal(preimage.offsets, jnp.asarray([0, 2, 4], jnp.int32))
    npt.assert_array_equal(preimage.source_indices, jnp.asarray([0, 2, 1, 3], jnp.int32))
    npt.assert_array_equal(
        preimage.aligned_target_indices["destination"],
        destination[preimage.source_indices],
    )
    target_indices = jax.jit(lambda graph: graph.index_maps[("edge", "source")].target_indices)(
        domain_graph
    )
    npt.assert_array_equal(target_indices, source)
    with pytest.raises(TypeError):
        domain_graph.domain_sizes["edge"] = 5


def test_index_map_contracts_are_not_inferred():
    target_indices = jnp.asarray([0, 1, -1, -1], jnp.int32)
    default = DomainGraph(
        {"source": 4, "target": 2},
        {("source", "target"): IndexMap(target_indices)},
    ).index_maps[("source", "target")]
    assert not default.is_injective
    assert not default.is_sorted
    assert default.may_have_unmapped

    declared_indices = jnp.asarray([0, 1, 2, 3], jnp.int32)
    declared = IndexMap(
        declared_indices,
        is_injective=True,
        is_sorted=True,
        may_have_unmapped=False,
    )
    preserved = DomainGraph(
        {"source": 4, "target": 4},
        {("source", "target"): declared},
    ).index_maps[("source", "target")]
    assert preserved.is_injective
    assert preserved.is_sorted
    assert not preserved.may_have_unmapped


def test_domain_wire_resolves_unique_multihop_maps_and_preimages():
    source_to_middle = jnp.asarray([0, 1, -1], jnp.int32)
    middle_to_target = jnp.asarray([1, 0], jnp.int32)
    target_to_unused = jnp.asarray([0, 0], jnp.int32)
    domain_graph = DomainGraph(
        {"source": 3, "middle": 2, "target": 2, "unused": 1},
        {
            ("source", "middle"): IndexMap(source_to_middle),
            ("middle", "target"): IndexMap(middle_to_target),
            ("target", "unused"): IndexMap(
                target_to_unused,
                may_have_unmapped=False,
            ),
        },
    )

    assert ("source", "target") not in domain_graph.index_maps
    wire = marshal_domains(
        jnp.zeros((3, 1, 1), jnp.float32),
        jnp.zeros((2, 1, 1), jnp.float32),
        domain_graph=domain_graph,
        input_domains=domains["source"],
        output_domains=domains["target"],
        num_axes=1,
        num_inputs=1,
        num_outputs=1,
    )
    resolved_maps, preimages = unmarshal_domains(wire)
    assert ("source", "unused") not in resolved_maps
    resolved = resolved_maps[("source", "target")]
    npt.assert_array_equal(resolved.target_indices, jnp.asarray([1, 0, -1], jnp.int32))
    preimage = preimages[("source", "target")]
    npt.assert_array_equal(preimage.offsets, jnp.asarray([0, 1, 2], jnp.int32))
    npt.assert_array_equal(preimage.source_indices[:2], jnp.asarray([1, 0], jnp.int32))


def test_domain_graph_rejects_empty_domains():
    with pytest.raises(ValueError, match="must be positive"):
        DomainGraph({"empty": 0})


def test_engine_binding_names_follow_the_common_scheme():
    domain_graph = DomainGraph(
        {"edge": 3, "source": 2, "target": 2},
        {
            ("edge", "source"): IndexMap(jnp.asarray([0, 1, 0], jnp.int32)),
            ("edge", "target"): IndexMap(jnp.asarray([1, 0, 1], jnp.int32)),
        },
    )
    names = [
        name
        for name, _ in marshal_index_bindings(
            [
                ("target_indices", "edge", "source", "source"),
                ("preimage", "edge", "source", "source"),
                ("aligned_target_indices", "edge", "source", "target"),
            ],
            domain_graph.index_maps,
            domain_graph.preimages,
        )
    ]
    assert names == [
        "target_indices:edge->source",
        "preimage_offsets:edge->source",
        "preimage_source_indices:edge->source",
        "aligned_target_indices:edge->source->target",
    ]


def test_engine_composes_multihop_target_indices_for_compiler_slots():
    domain_graph = DomainGraph(
        {"edge": 3, "node": 2, "graph": 1},
        {
            ("edge", "node"): IndexMap(jnp.asarray([0, 1, -1], jnp.int32)),
            ("node", "graph"): IndexMap(jnp.asarray([0, 0], jnp.int32)),
        },
    )
    bindings = marshal_index_bindings(
        [("composed_target_indices", "edge", "graph", "graph")],
        domain_graph.index_maps,
        domain_graph.preimages,
    )
    assert bindings[0][0] == "target_indices:edge->graph"
    npt.assert_array_equal(bindings[0][1], jnp.asarray([0, 0, -1], jnp.int32))


def test_sorted_index_map_uses_source_order_with_trailing_padding():
    target_indices = jnp.asarray([0, 0, 1, 2, -1, -1], jnp.int32)
    graph = DomainGraph(
        {"source": 6, "target": 3},
        {("source", "target"): IndexMap(target_indices, is_sorted=True)},
    )
    preimage = graph.preimages[("source", "target")]
    npt.assert_array_equal(preimage.offsets, jnp.asarray([0, 2, 3, 4], jnp.int32))
    npt.assert_array_equal(preimage.source_indices, jnp.arange(6, dtype=jnp.int32))


def test_preimage_orders_unmapped_entries_after_larger_target_indices():
    target_indices = jnp.asarray([0, 2, 4, 6, 8, 10, 12, 1, -1, -1], jnp.int32)
    graph = DomainGraph(
        {"source": 10, "target": 13},
        {("source", "target"): IndexMap(target_indices)},
    )
    npt.assert_array_equal(
        graph.preimages[("source", "target")].source_indices,
        jnp.asarray([0, 7, 1, 2, 3, 4, 5, 6, 8, 9], jnp.int32),
    )


def test_domain_graph_rejects_raw_arrays_and_multiple_paths():
    raw = jnp.asarray([0, 1], jnp.int32)
    with pytest.raises(TypeError, match="must be an IndexMap"):
        DomainGraph({"source": 2, "target": 2}, {("source", "target"): raw})

    maps = {
        ("source", "middle"): IndexMap(raw),
        ("middle", "target"): IndexMap(raw),
        ("source", "target"): IndexMap(raw),
    }
    with pytest.raises(ValueError, match="multiple directed paths"):
        DomainGraph({"source": 2, "middle": 2, "target": 2}, maps)


def test_domain_scatter(device, dtype):
    edges, nodes, channels, u = 7, 3, 2, 4
    x = jnp.arange(edges * channels * u, dtype=dtype).reshape(edges, channels, u)
    target_indices = np.array([0, 2, 1, 0, 2, 1, 0], np.int32)
    domain_graph = DomainGraph(
        {"edge": edges, "node": nodes},
        {("edge", "node"): IndexMap(target_indices)},
    )
    y = loom.call(
        lambda v: v * 2,
        x,
        domain_graph=domain_graph,
        input_domains=domains["edge"],
        output_domains=domains["node"],
    )
    expected = jnp.zeros((nodes, channels, u), dtype).at[target_indices].add(x * 2)
    npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


def test_multihop_domain_scatter(device, dtype):
    sources, middle, targets, u = 5, 3, 2, 4
    values = jnp.arange(sources * u, dtype=dtype).reshape(sources, 1, u)
    source_to_middle = jnp.asarray([0, 1, 2, -1, 0], jnp.int32)
    middle_to_target = jnp.asarray([1, 0, 1], jnp.int32)
    graph = DomainGraph(
        {"source": sources, "middle": middle, "target": targets},
        {
            ("source", "middle"): IndexMap(source_to_middle),
            ("middle", "target"): IndexMap(
                middle_to_target,
                may_have_unmapped=False,
            ),
        },
    )
    actual = loom.call(
        lambda value: value,
        values,
        domain_graph=graph,
        input_domains=domains["source"],
        output_domains=domains["target"],
    )
    composed = jnp.asarray([1, 0, 1, -1, 1], jnp.int32)
    safe = jnp.where(composed < 0, targets, composed)
    expected = jnp.zeros((targets, 1, u), dtype).at[safe].add(values, mode="drop")
    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_domain_scatter(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        edges, nodes, channels, u = 7, 1, 2, 4
        count = edges * channels * u
        info = np.iinfo(integer_dtype)
        values = np.asarray(
            [int(info.max) - count * 3 + i * 3 for i in range(count)],
            dtype=integer_dtype,
        ).reshape(edges, channels, u)
        x = jnp.asarray(values)
        target_indices = np.zeros(edges, np.int32)
        domain_graph = DomainGraph(
            {"edge": edges, "node": nodes},
            {("edge", "node"): IndexMap(target_indices)},
        )
        y = loom.call(
            lambda value: value * 2,
            x,
            domain_graph=domain_graph,
            input_domains=domains["edge"],
            output_domains=domains["node"],
        )
        expected = np.zeros((nodes, channels, u), dtype=integer_dtype)
        np.add.at(expected, target_indices, values * np.asarray(2, dtype=integer_dtype))
        npt.assert_array_equal(y, expected)


@pytest.mark.parametrize("u", [1, 4])
def test_transposed_view(device, dtype, u):
    edges, sources, destinations, channels = 12, 4, 3, 2
    x = jnp.arange(sources * channels * u, dtype=dtype).reshape(sources, channels, u)
    source = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3], np.int32)
    destination = np.array([0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0], np.int32)
    domain_graph = DomainGraph(
        {"edge": edges, "source": sources, "destination": destinations},
        {("edge", "source"): IndexMap(source), ("edge", "destination"): IndexMap(destination)},
    )
    y = loom.call(
        lambda value: value * 2,
        x,
        u=u,
        domain_graph=domain_graph,
        input_domains=domains["source"],
        output_domains=domains["destination"],
    )
    expected = jnp.zeros((destinations, channels, u), dtype).at[destination].add(x[source] * 2)
    npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


@gpu_only
def test_mixed_output_domains_match_scatter_and_fiber_results(device):
    edges, sources, destinations, channels, u = 12, 4, 3, 2, 4
    edge = jnp.arange(edges * channels * u, dtype=jnp.float32).reshape(edges, channels, u)
    source_values = jnp.linspace(0.25, 2.0, sources * channels * u, dtype=jnp.float32).reshape(
        sources, channels, u
    )
    source = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3], np.int32)
    destination = np.array([0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0], np.int32)
    extents = {
        "edge": edges,
        "source": sources,
        "destination": destinations,
    }
    domain_graph = DomainGraph(
        extents,
        {("edge", "source"): IndexMap(source), ("edge", "destination"): IndexMap(destination)},
    )

    edge_output, destination_output = loom.call(
        lambda values: (
            values[0] * 2 + values[1],
            values[0] + values[1] * 3,
        ),
        (edge, source_values),
        u=u,
        domain_graph=domain_graph,
        input_domains=(domains["edge"], domains["source"]),
        output_domains=(domains["edge"], domains["destination"]),
        compile_options=Compile(),
    )
    gathered = source_values[source]
    npt.assert_allclose(edge_output, edge * 2 + gathered, rtol=1e-5, atol=1e-5)
    expected_destination = (
        jnp.zeros_like(destination_output).at[destination].add(edge + gathered * 3)
    )
    npt.assert_allclose(destination_output, expected_destination, rtol=1e-5, atol=1e-5)


@pytest.mark.capped_autotune(search_budget_ms=250.0)
@pytest.mark.parametrize("mask_mode", ["all_false", "mixed", "all_true"])
@gpu_only
def test_multi_gate_fiber_backward_matches_scatter_reference(device, mask_mode):
    edges, sources, destinations, u = 12, 4, 3, 4
    source = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3], np.int32)
    destination = np.array([0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0], np.int32)
    extents = {
        "edge": edges,
        "source": sources,
        "destination": destinations,
    }
    domain_graph = DomainGraph(
        extents,
        {("edge", "source"): IndexMap(source), ("edge", "destination"): IndexMap(destination)},
    )
    edge_values = jnp.linspace(0.25, 1.75, edges * u, dtype=jnp.float32).reshape(edges, 1, u)
    source_values = jnp.linspace(0.5, 1.5, sources * u, dtype=jnp.float32).reshape(sources, 1, u)
    weights = jnp.array(
        [[[0.75, 1.0, 1.25, 1.5], [1.5, 1.25, 1.0, 0.75]]],
        dtype=jnp.float32,
    )
    if mask_mode == "all_false":
        left_mask = jnp.zeros((edges, 1, u), dtype=jnp.bool_)
        right_mask = jnp.zeros_like(left_mask)
    elif mask_mode == "all_true":
        left_mask = jnp.ones((edges, 1, u), dtype=jnp.bool_)
        right_mask = jnp.ones_like(left_mask)
    else:
        indices = jnp.arange(edges * u).reshape(edges, 1, u)
        left_mask = indices % 2 == 0
        right_mask = indices % 3 == 0
    cotangent = jnp.linspace(0.5, 1.5, destinations * u, dtype=jnp.float32).reshape(
        destinations, 1, u
    )

    def pointwise(values):
        left = jnp.where(values[0], values[4][0] * values[2] * values[3], 0)
        right = jnp.where(values[1], values[4][1] * values[2] * values[3] * values[3], 0)
        return left + right

    def loom_loss(edge_arg, source_arg, weights_arg):
        output = loom.call(
            pointwise,
            (left_mask, right_mask, edge_arg, source_arg, weights_arg),
            u=u,
            domain_graph=domain_graph,
            input_domains=(
                domains["edge"],
                domains["edge"],
                domains["edge"],
                domains["source"],
                domains[:],
            ),
            output_domains=domains["destination"],
            compile_options=Compile(),
        )
        return jnp.sum(output * cotangent)

    def reference_loss(edge_arg, source_arg, weights_arg):
        gathered = source_arg[source]
        left = jnp.where(left_mask, weights_arg[:, 0:1] * edge_arg * gathered, 0)
        right = jnp.where(
            right_mask,
            weights_arg[:, 1:2] * edge_arg * gathered * gathered,
            0,
        )
        output = jnp.zeros((destinations, 1, u), jnp.float32).at[destination].add(left + right)
        return jnp.sum(output * cotangent)

    got = jax.grad(loom_loss, argnums=(0, 1, 2))(edge_values, source_values, weights)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(edge_values, source_values, weights)
    for actual, reference in zip(got, expected, strict=True):
        npt.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("integer_dtype", _INTEGER_DTYPES)
def test_integer_transposed_view_scatter(device, integer_dtype):
    with jax.enable_x64(jnp.dtype(integer_dtype).itemsize == 8):
        edges, sources, destinations, channels, u = 12, 4, 3, 2, 4
        count = sources * channels * u
        info = np.iinfo(integer_dtype)
        values = np.asarray(
            [int(info.max) - count * 3 + i * 3 for i in range(count)],
            dtype=integer_dtype,
        ).reshape(sources, channels, u)
        x = jnp.asarray(values)
        source = np.array([0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3], np.int32)
        destination = np.array([0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0], np.int32)
        domain_graph = DomainGraph(
            {"edge": edges, "source": sources, "destination": destinations},
            {("edge", "source"): IndexMap(source), ("edge", "destination"): IndexMap(destination)},
        )
        y = loom.call(
            lambda value: value * 2,
            x,
            domain_graph=domain_graph,
            input_domains=domains["source"],
            output_domains=domains["destination"],
        )
        expected = np.zeros((destinations, channels, u), dtype=integer_dtype)
        np.add.at(
            expected,
            destination,
            values[source] * np.asarray(2, dtype=integer_dtype),
        )
        npt.assert_array_equal(y, expected)
