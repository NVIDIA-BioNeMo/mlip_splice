# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local batch placement (no accidental default-device hop)."""

import numpy as np
from splice.parallel import (
    build_mesh,
    commit_tree,
    localize_replicated_tree,
    place_local_batch,
    shard_batch,
)


def test_place_local_batch_lands_on_local_device():
    import jax

    host = {
        "species": np.arange(8, dtype=np.int32),
        "nn_vecs": np.ones((4, 3), np.float32),
        "dummy_graph_id": 3,
    }
    placed = place_local_batch(host)
    local = jax.local_devices()[0]
    assert "dummy_graph_id" not in placed
    dev = placed["species"].devices()
    assert local in dev
    np.testing.assert_array_equal(np.asarray(placed["species"]), host["species"])


def test_shard_batch_wraps_local_buffers_without_scalar_transfer():
    import jax
    import pytest

    # Production topology is one process per GPU. A 1-process / N-GPU mesh
    # cannot be built from a single local shard.
    if jax.local_device_count() != 1:
        pytest.skip(
            "shard_batch matches init_distributed (1 local device); "
            f"this process sees {jax.local_device_count()}"
        )

    host = {
        "species": np.arange(8, dtype=np.int32),
        "nn_vecs": np.ones((4, 3), np.float32),
        "dummy_graph_id": 3,
    }
    placed = shard_batch(host, build_mesh())

    # The global data axis has one addressable shard in this test process.
    assert placed["species"].shape == (jax.process_count(), 8)
    assert placed["species"].is_fully_addressable
    np.testing.assert_array_equal(np.asarray(placed["species"][0]), host["species"])
    # This value is nats.shape[0]-1 by padding contract and is derived in loss.
    assert "dummy_graph_id" not in placed


def test_commit_tree_commits_arrays_and_preserves_statics():
    import jax

    tree = {"x": np.arange(4, dtype=np.float32), "static": 7, "scale": np.float32(1.5)}
    placed = commit_tree(tree)
    assert placed["static"] == 7
    assert isinstance(placed["scale"], np.floating)
    assert placed["x"].committed
    assert jax.local_devices()[0] in placed["x"].devices()


def test_localize_fully_addressable_tree_is_zero_copy():
    import jax

    x = jax.device_put(np.arange(4, dtype=np.float32), jax.local_devices()[0])
    got = localize_replicated_tree({"x": x, "static": "kept"})
    assert got["x"] is x
    assert got["static"] == "kept"
