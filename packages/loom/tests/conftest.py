# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from loom import Compile
from loom._src.runtime.recipes import aot

_AOT_CACHE: dict = {}


@pytest.fixture(autouse=True)
def capped_autotune(request):
    marker = request.node.get_closest_marker("capped_autotune")
    if marker is None:
        yield
        return
    with aot(
        Compile(search_budget_ms=marker.kwargs.get("search_budget_ms", 25.0)),
        cache=_AOT_CACHE,
        workload_identity=request.node.nodeid,
    ):
        yield


@pytest.fixture(
    params=[
        pytest.param(jnp.float32, id="f32"),
        pytest.param(jnp.float64, id="f64", marks=pytest.mark.extensive),
    ]
)
def dtype(request):
    if request.param == jnp.float64:
        with jax.enable_x64(True):
            yield request.param
    else:
        yield request.param


@pytest.fixture(
    params=[
        pytest.param("gpu", id="gpu"),
        pytest.param("cpu", id="cpu", marks=pytest.mark.extensive),
    ]
)
def device(request):
    if request.param == "gpu":
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            pytest.skip("no GPU available")
        selected = devices[0]
    else:
        selected = jax.devices("cpu")[0]
    with jax.default_device(selected):
        yield selected
