# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding / offset lookups gather ``species`` instead of repeating counts.

On a species-sorted layout the two are bit-identical; gather is also correct
when ids are not contiguous. Keep both checks so a future repeat regression
is visible.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from e3 import O3
from splice.ops.embedding import ChargeSpinEmbeddingMLP, SpeciesEmbedding
from splice.ops.head import SpeciesOffsets


def test_embedding_gather_matches_sorted_repeat():
    species = jnp.asarray([0, 0, 2, 2, 2], jnp.int32)
    counts = jnp.asarray([2, 0, 3], jnp.int32)
    op = SpeciesEmbedding(3, 4, param_dtype=jnp.float32, key=jax.random.key(0))
    gathered = op(species)[O3.trivial()]
    repeated = jnp.repeat(op.embedding, counts, axis=0, total_repeat_length=5)
    if op.weight_scaling:
        repeated = repeated / jnp.sqrt(op.num_species)
    np.testing.assert_allclose(gathered, repeated[:, None, :], rtol=0, atol=0)


def test_embedding_gather_follows_unsorted_ids():
    species = jnp.asarray([2, 0, 2], jnp.int32)
    op = SpeciesEmbedding(
        3,
        2,
        weight_scaling=False,
        init="fan_in_normal",
        param_dtype=jnp.float32,
        key=jax.random.key(1),
    )
    out = op(species)[O3.trivial()][:, 0, :]
    np.testing.assert_allclose(out, op.embedding[species], rtol=0, atol=0)


def test_offset_gather_matches_sorted_repeat():
    species = jnp.asarray([1, 1, 1, 2], jnp.int32)
    counts = jnp.asarray([0, 3, 1], jnp.int32)
    op = SpeciesOffsets(3, param_dtype=jnp.float32)
    op = eqx.tree_at(lambda m: m.offsets, op, jnp.asarray([0.5, 1.5, 2.5], jnp.float32))
    energies = jnp.zeros(4, jnp.float32)
    gathered = op(energies, species)
    repeated = jnp.repeat(op.offsets, counts, total_repeat_length=4)
    np.testing.assert_allclose(gathered, repeated, rtol=0, atol=0)


def test_charge_spin_embedding_composes_with_species_gather():
    species = jnp.asarray([2, 0, 2], jnp.int32)
    embedding = SpeciesEmbedding(3, 4, param_dtype=jnp.float32, key=jax.random.key(2))
    charge_spin = ChargeSpinEmbeddingMLP(4, param_dtype=jnp.float32, key=jax.random.key(3))
    features = embedding(species)
    result = eqx.filter_jit(charge_spin)(
        features,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([1, 0, 1], jnp.int32),
    )[O3.trivial()]
    assert result.shape == (3, 1, 4)
    assert bool(jnp.all(jnp.isfinite(result)))
    np.testing.assert_array_equal(result[0], result[2])
    assert not np.allclose(result, features[O3.trivial()])
