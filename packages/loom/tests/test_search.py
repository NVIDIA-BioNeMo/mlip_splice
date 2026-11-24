# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import loom
import numpy as np
import pytest
from loom import Compile, DomainGraph, IndexMap, domains

from marks import gpu_only

pytestmark = gpu_only


def _assert_agreement(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    assert float(np.max(np.abs(actual - expected))) / scale <= 5e-6


@pytest.mark.parametrize("search_budget_ms", [1e-3, 2000.0])
@pytest.mark.parametrize("reduce_lanes", [False, True])
@pytest.mark.parametrize("lanes", [1, 4, 32, 35])
@pytest.mark.parametrize("missing_indices", [False, True])
def test_search_executes_mixed_folds_with_changed_inputs(
    device, search_budget_ms, reduce_lanes, lanes, missing_indices
):
    # p indexes seven coarse cells; n indexes 37 fine cells; e and u index
    # nine visible elements and hidden lanes. y[n,e,u] = w[p(n),e,0] x[n,e,u].
    indices = np.arange(37, dtype=np.int32) * 3 % 7
    if missing_indices:
        indices[(np.arange(37) % 4 == 0) | (indices == 6)] = -1
    target = jnp.asarray(indices)
    graph = DomainGraph({"fine": 37, "coarse": 7}, {("fine", "coarse"): IndexMap(target)})

    @jax.jit
    def compiled(weights, values):
        def pointwise(pair):
            product = pair[0] * pair[1]
            return product, -product, product + product

        return loom.call(
            pointwise,
            (weights, values),
            domain_graph=graph,
            input_domains=(domains["coarse"], domains["fine"]),
            output_domains=(domains["fine"], domains["coarse"], domains["coarse"]),
            reduce_lanes=(False, False, reduce_lanes),
            u=lanes,
            compile_options=Compile(search_budget_ms=search_budget_ms),
        )

    for seed in [3, 7]:
        rng = np.random.default_rng(seed)
        weights = jnp.asarray(rng.uniform(0.25, 1.0, (7, 9, 1)), jnp.float32)
        values = jnp.asarray(rng.uniform(0.25, 1.0, (37, 9, lanes)), jnp.float32)
        safe_target = jnp.maximum(target, 0)
        fine = jnp.where((target >= 0)[:, None, None], weights[safe_target] * values, 0)
        coarse = jnp.zeros((7, 9, lanes), jnp.float32).at[safe_target].add(fine)
        doubled = 2 * coarse
        folded = doubled.sum(axis=-1, keepdims=True) if reduce_lanes else doubled
        expected = fine, -coarse, folded
        result = compiled(weights, values)
        for actual, reference in zip(result, expected, strict=True):
            _assert_agreement(actual, reference)


@pytest.mark.parametrize("search_budget_ms", [1e-3, 2000.0])
def test_search_materializes_a_required_cut(device, search_budget_ms):
    @jax.jit
    def compiled(values):
        def polynomial(value):
            total = value
            for denominator in range(2, 30):
                total = total * value + value / denominator
            return total

        return loom.call(
            polynomial,
            values,
            domain_graph=DomainGraph({"sample": 3}),
            input_domains=domains["sample"],
            output_domains=domains["sample"],
            compile_options=Compile(search_budget_ms=search_budget_ms, max_kernel_statements=32),
        )

    for endpoint in [0.25, 0.5]:
        values = jnp.linspace(0.01, endpoint, 3 * 5 * 35).reshape(3, 5, 35)
        expected = values
        for denominator in range(2, 30):
            expected = expected * values + values / denominator
        _assert_agreement(compiled(values), expected)
