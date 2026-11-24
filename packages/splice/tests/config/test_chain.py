# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seam unit tests: dummy ops through the draupnir engine (no GPU kernels).

Exercises nominal wiring — steps chain wherever an output name and the next
input name match — plus `in`/`out` rebinding and `chain.inputs` derivation, via
`splice.config.build_chain`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from splice.config import build_chain
from splice.registry import register_op


@register_op("_test_add_const")
class _AddConst(eqx.Module):
    c: float = eqx.field(static=True)

    OUTPUTS = "h"  # reads+writes "h": the replacement h = op(h); INPUTS inferred ("h",)

    def __init__(self, c: float):
        self.c = c

    def __call__(self, h):
        return h + self.c


@register_op("_test_sum")
class _Sum(eqx.Module):
    OUTPUTS = "h"  # INPUTS inferred: ("h", "b")

    def __call__(self, h, b):
        return h + b


def test_nominal_linear_chain():
    # Three ops all reading+writing "h" chain nominally, no in/out needed.
    cfg = [
        {"op": "_test_add_const", "params": {"c": 1.0}},
        {"op": "_test_add_const", "params": {"c": 10.0}},
        {"op": "_test_add_const", "params": {"c": 100.0}},
    ]
    chain = build_chain(cfg, key=jax.random.key(0))
    assert chain.inputs == frozenset({"h"})
    out = chain({"h": jnp.float32(0.0)})
    assert float(out["h"]) == pytest.approx(111.0)


def test_named_side_branch():
    # An explicit `out:` publishes a value without disturbing the main "h".
    cfg = [
        {"op": "_test_add_const", "params": {"c": 1.0}},  # h = 1
        {"op": "_test_add_const", "params": {"c": 100.0}, "out": "named"},  # named=101; h stays 1
        {"op": "_test_add_const", "params": {"c": 10.0}},  # h = 11
    ]
    chain = build_chain(cfg, key=jax.random.key(0))
    out = chain({"h": jnp.float32(0.0)})
    assert float(out["h"]) == pytest.approx(11.0)
    assert float(out["named"]) == pytest.approx(101.0)


def test_in_dict_binding():
    cfg = [
        {"op": "_test_add_const", "params": {"c": 1.0}},  # h = 1
        {"op": "_test_add_const", "params": {"c": 100.0}, "out": "b_branch"},  # b_branch = 101
        {"op": "_test_sum", "in": {"b": "b_branch"}},  # h = 1 + 101 = 102 (h nominal)
    ]
    chain = build_chain(cfg, key=jax.random.key(0))
    out = chain({"h": jnp.float32(0.0)})
    assert float(out["h"]) == pytest.approx(102.0)


def test_chain_inputs_derived():
    # Free names read by a step and written by none before it are the chain inputs.
    cfg = [{"op": "_test_sum", "in": {"h": "x", "b": "y"}}]
    chain = build_chain(cfg, key=jax.random.key(0))
    assert chain.inputs == frozenset({"x", "y"})
    out = chain({"x": jnp.float32(3.0), "y": jnp.float32(4.0)})
    assert float(out["h"]) == pytest.approx(7.0)
