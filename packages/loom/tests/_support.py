# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import loom
import pytest
from loom import DomainGraph, domains

_INTEGER_DTYPES = [
    pytest.param(jnp.int32, id="int32"),
    pytest.param(jnp.int8, id="int8", marks=pytest.mark.extensive),
    pytest.param(jnp.uint8, id="uint8", marks=pytest.mark.extensive),
    pytest.param(jnp.int16, id="int16", marks=pytest.mark.extensive),
    pytest.param(jnp.uint16, id="uint16", marks=pytest.mark.extensive),
    pytest.param(jnp.uint32, id="uint32", marks=pytest.mark.extensive),
    pytest.param(jnp.int64, id="int64", marks=pytest.mark.extensive),
    pytest.param(jnp.uint64, id="uint64", marks=pytest.mark.extensive),
]


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _call(
    fn,
    x,
    out,
    *,
    u=None,
    input_domain="n",
    output_domain="n",
    compile_options=None,
):
    return loom.call(
        fn,
        x,
        u=u,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains[input_domain],
        output_domains=domains[output_domain],
        reduce_lanes=out.shape[-1] == 1,
        output_dtypes=out.dtype,
        compile_options=compile_options,
    )
