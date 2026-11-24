# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke: load mace_smoke.draupnir.yaml, run a forward + one training step."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from splice.config import build_chain, load_config
from splice.forward import _REDUCTION_WORKSPACE_BYTES, _graph_reduce, model_call

from tests.conftest import CONFIGS

TINY_YAML = CONFIGS / "smoke.yaml"


def test_graph_reduce_bounds_loom_workspace(monkeypatch):
    recorded = {}

    def record(*args, **kwargs):
        recorded.update(kwargs)
        return None

    monkeypatch.setattr("splice.forward.loom.call", record)
    values = jax.ShapeDtypeStruct((7, 3), jnp.float32)
    graph = type("Graph", (), {"domain_sizes": {}, "domains": None})()

    _graph_reduce(graph, values, "edge", "dst", 5)

    assert recorded["compile_options"].max_workspace_bytes == _REDUCTION_WORKSPACE_BYTES


def _tiny_batch(N=32, E=200, G=4, num_species=4, seed=42):
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(seed), 5)
    edge_vecs = jax.random.normal(k1, (E, 3), jnp.float32)
    species = jax.random.randint(k2, (N,), 0, num_species, jnp.int32)
    senders = jax.random.randint(k3, (E,), 0, N, jnp.int32)
    receivers = jax.random.randint(k4, (E,), 0, N, jnp.int32)
    positions = jax.random.normal(k5, (N, 3), jnp.float32)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 10.0, (G, 3, 3))
    graph_index = jnp.sort(jax.random.randint(jax.random.key(0), (N,), 0, G, jnp.int32))
    nats = jnp.zeros((G,), jnp.int32).at[graph_index].add(1)
    mask = jnp.ones((E,), jnp.int32)
    return dict(
        nn_vecs=edge_vecs,
        species=species,
        positions=positions,
        cell=cell,
        inda=senders,
        indb=receivers,
        inde=graph_index,
        nats=nats,
        mask=mask,
    )


def _tiny_chain(num_species=4):
    cfg = load_config(TINY_YAML)
    return build_chain(cfg, key=jax.random.key(0))


def test_forward_finite():
    chain = _tiny_chain()
    batch = _tiny_batch()
    out = model_call(chain, batch, num_species=4)
    assert bool(jnp.all(jnp.isfinite(out["E"])))
    assert bool(jnp.all(jnp.isfinite(out["F"])))
    assert out["E"].shape == (batch["nats"].shape[0],)
    assert out["F"].shape == (batch["species"].shape[0], 3)


def test_one_train_step_moves_params():
    chain = _tiny_chain()
    batch = _tiny_batch()
    params, static = eqx.partition(chain, eqx.is_inexact_array)
    tx = optax.adam(1e-2)
    opt_state = tx.init(params)

    target_E = jnp.ones((batch["nats"].shape[0],), jnp.float32)
    target_F = jnp.ones((batch["species"].shape[0], 3), jnp.float32)

    def loss_fn(params):
        model = eqx.combine(params, static)
        out = model_call(model, batch, num_species=4)
        return jnp.mean((out["E"] - target_E) ** 2) + jnp.mean((out["F"] - target_F) ** 2)

    loss0, grads = eqx.filter_value_and_grad(loss_fn)(params)
    assert bool(jnp.isfinite(loss0))
    updates, _ = tx.update(grads, opt_state, params)
    new_params = eqx.apply_updates(params, updates)
    diffs = jax.tree.map(lambda a, b: float(jnp.max(jnp.abs(a - b))), params, new_params)
    max_move = max(jax.tree.leaves(diffs))
    assert max_move > 1e-6, f"no params moved after one step (max move {max_move})"


def test_species_presorted_matches_reference():
    """The opt-in presort fast path must reproduce the default path.

    The default path's ``argsort`` is stable, so on an already-sorted batch the
    permutation it builds is the identity and the two paths differ only by the
    gathers the fast path skips — a discrepancy beyond rounding means a skipped
    gather was not actually an identity.

    Both are compared against the magnitude of their own field rather than entry
    by entry. Loom rewrites arithmetic exactly, so it may reassociate
    a reduction differently between the two programs, and the error of such a sum
    scales with its largest terms: a nearly-cancelled small entry can be tens of
    ULP off in its own magnitude while being one ULP of the field. A skipped
    gather that was not the identity misplaces a whole contribution and lands far
    above this bound.
    """
    chain = _tiny_chain()
    batch = _tiny_batch()
    batch = {**batch, "species": jnp.sort(batch["species"])}

    ref = model_call(chain, batch, num_species=4)
    fast = model_call(chain, batch, num_species=4, species_presorted=True)

    for name in ("E", "F"):
        reference = np.asarray(ref[name], dtype=np.float32)
        np.testing.assert_array_less(
            np.abs(np.asarray(fast[name], dtype=np.float32) - reference),
            8 * np.spacing(np.abs(reference).max()),
        )
