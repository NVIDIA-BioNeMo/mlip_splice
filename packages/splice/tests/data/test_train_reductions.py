# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hot-path loss/stat reduction invariants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from splice.training.objective import _assemble_loss


def test_loss_components_share_one_exact_normalization_vector():
    # (optimized loss sum, squared-error sum, absolute-error sum, count)
    components = {
        "E": (np.float32(6), np.float32(8), np.float32(4), np.float32(2)),
        "F": (np.float32(15), np.float32(21), np.float32(9), np.float32(3)),
    }
    loss, mse, mae = _assemble_loss(components, {"E": 2.0, "F": 0.5})

    np.testing.assert_allclose(np.asarray(loss), 2.0 * (6 / 2) + 0.5 * (15 / 3))
    np.testing.assert_allclose(np.asarray(mse["E"]), 8 / 2)
    np.testing.assert_allclose(np.asarray(mse["F"]), 21 / 3)
    np.testing.assert_allclose(np.asarray(mae["E"]), 4 / 2)
    np.testing.assert_allclose(np.asarray(mae["F"]), 9 / 3)


def test_zero_count_is_clamped_without_changing_nonzero_terms():
    components = {
        "E": (np.float32(0), np.float32(0), np.float32(0), np.float32(0)),
        "F": (np.float32(4), np.float32(9), np.float32(3), np.float32(2)),
    }
    loss, mse, mae = _assemble_loss(components, {"E": 1.0, "F": 1.0})

    np.testing.assert_allclose(np.asarray(loss), 2.0)
    np.testing.assert_allclose(np.asarray(mse["E"]), 0.0)
    np.testing.assert_allclose(np.asarray(mse["F"]), 4.5)
    np.testing.assert_allclose(np.asarray(mae["F"]), 1.5)


def test_local_mean_matches_unmapped_assembly():
    components = {
        "E": (np.float32(6), np.float32(8), np.float32(4), np.float32(2)),
        "F": (np.float32(15), np.float32(21), np.float32(9), np.float32(3)),
    }
    a = _assemble_loss(components, {"E": 1.0, "F": 1.0})
    b = _assemble_loss(components, {"E": 1.0, "F": 1.0}, local_mean=True)
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_allclose(np.asarray(a[1]["E"]), np.asarray(b[1]["E"]))
    np.testing.assert_allclose(np.asarray(a[1]["F"]), np.asarray(b[1]["F"]))


def _jaxpr_text(*, local_mean: bool) -> str:
    def body(c):
        comps = {
            "E": (c, c, c, c),
            "F": (c, c, c, c),
        }
        loss, _, _ = _assemble_loss(
            comps, {"E": 1.0, "F": 1.0}, axis_name="i", local_mean=local_mean
        )
        return loss

    n = max(jax.device_count(), 1)
    jp = jax.make_jaxpr(jax.pmap(body, axis_name="i"))(jnp.ones((n,), dtype=jnp.float32))
    return str(jp)


def test_exact_mean_emits_count_psum():
    text = _jaxpr_text(local_mean=False)
    assert "psum" in text


def test_local_mean_omits_count_psum():
    text = _jaxpr_text(local_mean=True)
    assert "psum" not in text


@pytest.mark.skipif(jax.device_count() < 2, reason="needs ≥2 JAX devices")
def test_pmean_averages_mapped_grads():
    """DDP contract: allreduce-mean, not sum."""
    xs = jnp.arange(jax.device_count(), dtype=jnp.float32)

    def mean_fn(x):
        return jax.lax.pmean(x, "i")

    def sum_fn(x):
        return jax.lax.psum(x, "i")

    averaged = jax.pmap(mean_fn, axis_name="i")(xs)
    summed = jax.pmap(sum_fn, axis_name="i")(xs)
    np.testing.assert_allclose(np.asarray(averaged), np.full(xs.shape, xs.mean()))
    np.testing.assert_allclose(np.asarray(summed), np.full(xs.shape, xs.sum()))
    assert float(averaged[0]) != float(summed[0])
