# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An ahead-of-time cache is only reusable if it can be written to disk.

What tuning buys is kept by pickling the cache. A cache that cannot be pickled
serves only the process that filled it, so every later run pays the tuning
again — which is exactly what a persistent cache exists to avoid.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import loom
import numpy as np
from loom import Compile, DomainGraph, IndexMap, domains
from loom._src.runtime.recipes import aot

from marks import gpu_only

RECEIVERS = 256
SPECIES = 4
CHANNELS = 8


def _call_loom(weights, features, target_indices):
    return loom.call(
        lambda values: values[0] * values[1],
        (weights, features),
        domain_graph=DomainGraph(
            {"receiver": RECEIVERS, "species": SPECIES},
            {("receiver", "species"): IndexMap(target_indices)},
        ),
        input_domains=(domains["species"], domains["receiver"]),
        output_domains=domains["receiver"],
        compile_options=Compile(),
    )


@gpu_only
def test_a_tuned_aot_cache_survives_a_pickle_round_trip(device):
    rng = np.random.default_rng(0)
    weights = jnp.asarray(rng.standard_normal((SPECIES, CHANNELS)), dtype=jnp.float32)
    features = jnp.asarray(rng.standard_normal((RECEIVERS, CHANNELS)), dtype=jnp.float32)
    target_indices = rng.integers(0, SPECIES, RECEIVERS, dtype=np.int32)

    cache: dict = {}
    with aot(Compile(search_budget_ms=25.0), cache=cache, workload_identity=b"aot-pickle"):
        jax.block_until_ready(_call_loom(weights, features, target_indices))

    assert cache, "a realized Loom call fills the cache it was given"
    restored = pickle.loads(pickle.dumps(cache))
    assert set(restored) == set(cache)
    for key, (recipe, _) in restored.items():
        original, _ = cache[key]
        assert bytes(recipe) == bytes(original)
        assert recipe.inputs == original.inputs
        assert recipe.outputs == original.outputs
