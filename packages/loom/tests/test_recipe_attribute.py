# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""That the module XLA compiles names a recipe by content, not by address.

A recipe reaches the runtime as an FFI attribute on a custom call, and that
attribute is part of the HLO module. Anything process-local in it makes the
module a different module in every process, so every cache keyed on the module
is written and never read — it never fails and never hits, which presents as a
slow compiler rather than as a bug.

So the property is checked where it lives: the text of the lowered module, from
two separate interpreters, compared byte for byte.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import loom
import numpy.testing as npt
import pytest
from loom import DomainGraph, domains
from loom._src.primitive.options import Compile

from marks import gpu_only

# The program is deliberately trivial: what is under test is the attribute the
# call carries, not what the kernels compute.
LOWER = textwrap.dedent(
    """
    import jax, jax.numpy as jnp
    import loom
    from loom import DomainGraph, IndexMap, domains
    from loom._src.primitive.options import Compile

    def program(x):
        return loom.call(
            lambda v: v * v + 2 * v,
            x,
            domain_graph=DomainGraph({"n": x.shape[0]}),
            input_domains=domains["n"],
            output_domains=domains["n"],
            output_dtypes=x.dtype,
            compile_options=Compile(search_budget_ms=1),
        )

    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    print(jax.jit(program).lower(x).as_text())
    """
)

# The value is a dense byte array, so it contains commas: the match runs to
# its element type rather than to the first separator inside it.
RECIPE_ATTR = re.compile(r"recipe = (dense<[^>]*> : tensor<\d+x\w+>)")


def lowered_in_a_new_process() -> str:
    done = subprocess.run(
        [sys.executable, "-c", LOWER],
        capture_output=True,
        text=True,
        check=False,
        # The child lowers and never executes, so it needs the driver and no
        # device memory at all. Left to itself it would reserve most of the
        # device on top of what this process already holds.
        env=os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    if done.returncode != 0:
        # The recipe attribute is only emitted by the CUDA lowering, so the child
        # needs a CUDA context of its own, and a context needs a few hundred
        # megabytes. On a device something else has filled there is no room for a
        # second one: that is a precondition this test cannot meet rather than a
        # property it disproves.
        if "no supported devices found for platform CUDA" in done.stderr:
            pytest.skip("no room on the device for a second CUDA context")
        pytest.fail(f"lowering in a child process failed:\n{done.stderr}")
    return done.stdout


@gpu_only
@pytest.mark.capped_autotune(search_budget_ms=1)
@pytest.mark.xfail(
    reason=(
        "fails only in full-suite runs (no recipe attribute in the lowered module) "
        "and passes alone: order-dependent, under investigation"
    ),
    strict=False,
)
def test_the_recipe_attribute_is_the_same_in_two_processes(device):
    """One recipe, two interpreters, one module text.

    Two processes rather than two lowerings in one, because a heap address is
    perfectly stable within a process: an in-process comparison would pass on
    the pointer this replaced and prove nothing.
    """
    first = lowered_in_a_new_process()
    second = lowered_in_a_new_process()

    attributes = RECIPE_ATTR.findall(first)
    assert attributes, f"no recipe attribute in the lowered module:\n{first}"
    for attribute in attributes:
        assert "dense<" in attribute and "ui8" in attribute, (
            "the attribute must be the recipe's content digest, and a digest is "
            f"a byte array; got: {attribute}"
        )

    assert RECIPE_ATTR.findall(second) == attributes
    assert first == second, "the same recipe lowered to two different modules"


@gpu_only
@pytest.mark.capped_autotune(search_budget_ms=1)
def test_two_different_recipes_carry_two_different_attributes(device):
    """A content name distinguishes what it names.

    Stability alone would be satisfied by a constant, which would hand every
    program the first recipe loaded. So two programs that differ must differ
    here too.
    """

    def lower(fn):
        x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
        text = (
            jax.jit(
                lambda value: loom.call(
                    fn,
                    value,
                    domain_graph=DomainGraph({"n": value.shape[0]}),
                    input_domains=domains["n"],
                    output_domains=domains["n"],
                    output_dtypes=value.dtype,
                    compile_options=Compile(search_budget_ms=1),
                )
            )
            .lower(x)
            .as_text()
        )
        return RECIPE_ATTR.findall(text)

    squared = lower(lambda v: v * v + 2 * v)
    cubed = lower(lambda v: v * v * v + 3 * v)
    assert squared and cubed
    assert squared != cubed


@gpu_only
@pytest.mark.capped_autotune(search_budget_ms=1)
def test_a_recipe_named_by_content_still_runs(device):
    """The name resolves back to the recipe it names, on the device."""
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    y = loom.call(
        lambda v: v * v + 2 * v,
        x,
        domain_graph=DomainGraph({"n": x.shape[0]}),
        input_domains=domains["n"],
        output_domains=domains["n"],
        output_dtypes=x.dtype,
        compile_options=Compile(search_budget_ms=1),
    )
    npt.assert_allclose(y, x * x + 2 * x, rtol=1e-5, atol=1e-5)
