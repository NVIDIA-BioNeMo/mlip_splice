# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every output is a fold by key, and every bucket count reaches the same sum.

A key no two positions share stores; a key they share reduces. The two forms
live side by side in one kernel, and the bucket copies a colliding fold may
accumulate into are a temporary key a projection sums away — so the answer
cannot depend on how many copies the executor picked.
"""

import json

import jax
import jax.numpy as jnp
import loom
import numpy as np
import numpy.testing as npt
import pytest
from loom import Compile, DomainGraph, IndexMap, domains
from loom._src.runtime.recipes import aot

from marks import f32_only, gpu_only

pytestmark = pytest.mark.capped_autotune

# Enough receivers per species that the atomic chain the buckets break is
# longer than the launch that breaks it, so the ladder offers more than one
# count. Small enough to compile in a test.
RECEIVERS = 2_048
SPECIES = 4
CHANNELS = 8
WEIGHTS = 6


def _inputs(dtype, seed=0):
    rng = np.random.default_rng(seed)
    weights = jnp.asarray(rng.standard_normal((SPECIES, WEIGHTS, CHANNELS)), dtype=dtype)
    features = jnp.asarray(rng.standard_normal((RECEIVERS, WEIGHTS, CHANNELS)), dtype=dtype)
    target_indices = rng.integers(0, SPECIES, RECEIVERS, dtype=np.int32)
    return weights, features, target_indices


def _program(values):
    weights, features = values
    return weights * features


def _call_loom(values, target_indices):
    domain_graph = DomainGraph(
        {"receiver": RECEIVERS, "species": SPECIES},
        {("receiver", "species"): IndexMap(target_indices)},
    )
    return loom.call(
        _program,
        values,
        domain_graph=domain_graph,
        input_domains=(domains["species"], domains["receiver"]),
        output_domains=domains["receiver"],
        compile_options=Compile(),
    )


def _call_reference(values, target_indices):
    weights, features = values
    return jnp.take(weights, target_indices, axis=0) * features


def _loss(call, values, target_indices):
    return jnp.sum(call(values, target_indices) ** 2)


@gpu_only
def test_a_unique_key_and_a_colliding_key_agree_in_one_kernel(device, dtype):
    """The backward pass writes both forms at once: `dfeatures` has a unique
    receiver key and stores, `dweights` collides on species and reduces."""
    weights, features, target_indices = _inputs(dtype)
    values = (weights, features)

    npt.assert_allclose(
        _call_loom(values, target_indices),
        _call_reference(values, target_indices),
        rtol=3e-5,
        atol=3e-5,
    )

    got = jax.grad(lambda v: _loss(_call_loom, v, target_indices))(values)
    want = jax.grad(lambda v: _loss(_call_reference, v, target_indices))(values)
    for got_leaf, want_leaf in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        scale = float(np.abs(np.asarray(want_leaf)).max())
        npt.assert_allclose(got_leaf, want_leaf, rtol=1e-4, atol=1e-4 * max(scale, 1.0))


@gpu_only
@f32_only
def test_the_colliding_fold_offers_more_than_one_bucket_count(device, dtype, tmp_path):
    """The ladder is observable: the compile reports every count each output
    admits, why it stops there, and what the copies would cost."""
    weights, features, target_indices = _inputs(dtype)
    with aot(debug_dump_dir=str(tmp_path)):
        gradient = jax.grad(lambda values: _loss(_call_loom, values, target_indices))
        jax.block_until_ready(gradient((weights, features)))

    rows = []
    for path in sorted(tmp_path.glob("loom_compile_*.json")):
        with open(path) as dump:
            profile = json.load(dump)
        candidates = profile.get("fold_buckets", [])
        by_arm_output = {}
        for row in candidates:
            by_arm_output.setdefault((row["arm"], row["output"]), set()).add(row["buckets"])
        assert all(len(counts) == 1 for counts in by_arm_output.values()), (
            "each compiled arm takes one bucket count per output",
            by_arm_output,
        )
        rows.extend(candidates)
    assert rows, "the compile reports its folds"

    by_key = {row["key"] for row in rows}
    assert by_key == {"unique", "colliding"}, by_key
    for row in rows:
        if row["key"] == "unique":
            assert row["buckets"] == 1
            assert row["admitted"] == [1]
            assert row["cap"] == "unique_key"
        else:
            assert row["buckets"] in row["admitted"]
            # A bucketed arm names the projection that sums the copies away.
            assert (row["projection"] == "none") == (row["buckets"] == 1)
            if row["buckets"] > 1:
                assert row["entry"]
                assert 0 < len(row["tile_widths"]) <= 3
                assert all(
                    tile > 0 and tile <= 256 and tile.bit_count() == 1
                    for tile in row["tile_widths"]
                )
                assert row["workspace_bytes"] == (
                    row["elements"] * row["buckets"] * np.dtype(dtype).itemsize
                )
    colliding = [row for row in rows if row["key"] == "colliding"]
    assert max(row["buckets"] for row in colliding) > 1, (
        "a chain this deep admits more than one bucket count: "
        f"{sorted({row['cap']: row['cap_bound'] for row in colliding}.items())}"
    )
