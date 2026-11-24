# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Padding tests.

Critical invariant: running the chain on a padded batch must give the SAME
energies for real graphs as running on the unpadded batch (within fp32
tolerance). If padding atoms leak into real-atom features this test will
fail. Numerical tolerance is generous because the chain has many ops.
"""

import jax
import jax.numpy as jnp
from splice.config import build_chain, load_config
from splice.forward import model_call
from splice.padding import choose_bucket_sizes, pad_batch, real_atoms_mask

from tests.conftest import CONFIGS

TINY_YAML = CONFIGS / "smoke.yaml"


def _raw_batch(N=32, E=200, G=4, num_species=4, seed=42):
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(seed), 5)
    edge_vecs = jax.random.normal(k1, (E, 3), jnp.float32)
    species = jax.random.randint(k2, (N,), 0, num_species, jnp.int32)
    senders = jax.random.randint(k3, (E,), 0, N, jnp.int32)
    receivers = jax.random.randint(k4, (E,), 0, N, jnp.int32)
    positions = jax.random.normal(k5, (N, 3), jnp.float32)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * 10.0, (G, 3, 3))
    inde = jnp.sort(jax.random.randint(jax.random.key(0), (N,), 0, G, jnp.int32))
    nats = jnp.zeros((G,), jnp.int32).at[inde].add(1)
    return dict(
        nn_vecs=edge_vecs,
        species=species,
        positions=positions,
        cell=cell,
        inda=senders,
        indb=receivers,
        inde=inde,
        nats=nats,
    )


def test_choose_bucket_sizes_reserves_dummy_atom_slot():
    # Atoms round up from n+1 → always at least one padding slot.
    # Edges round up from n directly.
    na, ne = choose_bucket_sizes(100, 1000, base_atoms=256, base_edges=4096)
    assert na == 256 and ne == 4096
    assert na > 100

    # Exact-fit atoms: rounds up to next bucket so a padding slot exists.
    na, _ = choose_bucket_sizes(256, 4096)
    assert na == 512


def test_pad_batch_shapes_and_invariants():
    raw = _raw_batch(N=32, E=200, G=4)
    pad = pad_batch(raw, n_atoms_padded=64, n_edges_padded=256, n_graphs_padded=8)
    assert pad["species"].shape == (64,)
    assert pad["nn_vecs"].shape == (256, 3)
    assert pad["inde"].shape == (64,)
    assert pad["nats"].shape == (8,)
    assert pad["mask"].shape == (256,)
    # Real region preserved exactly.
    assert bool(jnp.all(pad["species"][:32] == raw["species"]))
    assert bool(jnp.all(pad["inde"][:32] == raw["inde"]))
    # Padding atoms all go to the dummy graph. The dummy graph is the last
    # nats slot: train.py recovers dummy_graph_id as nats.shape[-1] - 1 when
    # the key is stripped from the device batch.
    assert pad["dummy_graph_id"] == pad["nats"].shape[-1] - 1
    assert bool(jnp.all(pad["inde"][32:] == pad["dummy_graph_id"]))
    # nats: real graphs unchanged, last entry holds padding-atom count.
    assert bool(jnp.all(pad["nats"][:4] == raw["nats"]))
    assert int(pad["nats"][-1]) == 32
    # Real edges mask is 1, padding mask is 0.
    assert bool(jnp.all(pad["mask"][:200] == 1))
    assert bool(jnp.all(pad["mask"][200:] == 0))
    # positions + cell pad correctly.
    assert pad["positions"].shape == (64, 3)
    assert pad["cell"].shape == (8, 3, 3)
    assert bool(jnp.all(pad["positions"][:32] == raw["positions"]))
    assert bool(jnp.all(pad["positions"][32:] == 0))
    assert bool(jnp.all(pad["cell"][:4] == raw["cell"]))


def test_padding_preserves_real_graph_energies():
    """The critical correctness test."""
    cfg = load_config(TINY_YAML)
    chain = build_chain(cfg, key=jax.random.key(0))

    raw = _raw_batch(N=32, E=200, G=4)
    # Unpadded run: 4 graphs, 32 atoms, 200 edges, no mask.
    raw_with_mask = {**raw, "mask": jnp.ones((200,), jnp.int32)}
    out_unpadded = model_call(chain, raw_with_mask, num_species=4)

    # Padded run: pad to (64, 256, 8). Last graph is dummy.
    pad = pad_batch(raw, n_atoms_padded=64, n_edges_padded=256, n_graphs_padded=8)
    # Wrapper doesn't need dummy_graph_id; strip it from the batch dict.
    pad_for_wrapper = {k: v for k, v in pad.items() if k != "dummy_graph_id"}
    out_padded = model_call(chain, pad_for_wrapper, num_species=4)

    # Real graph energies must match. Tolerance is relative because the chain
    # at init can produce large-magnitude outputs (we tune the per-op scales
    # incrementally); padding correctness is unaffected by absolute scale.
    E_real_padded = out_padded["E"][:4]
    assert jnp.allclose(E_real_padded, out_unpadded["E"], rtol=1e-4, atol=1e-4), (
        f"unpadded E: {out_unpadded['E']}\npadded E[:4]: {E_real_padded}"
    )
    F_real_padded = out_padded["F"][:32]
    assert jnp.allclose(F_real_padded, out_unpadded["F"], rtol=1e-4, atol=1e-4), (
        f"max F diff: {float(jnp.max(jnp.abs(F_real_padded - out_unpadded['F'])))}"
    )


def test_loss_pattern_with_mask():
    """Show how a loss handles padding: E[:-1] + per-atom mask via inde."""
    cfg = load_config(TINY_YAML)
    chain = build_chain(cfg, key=jax.random.key(0))

    raw = _raw_batch(N=32, E=200, G=4)
    pad = pad_batch(raw, n_atoms_padded=64, n_edges_padded=256, n_graphs_padded=8)
    pad_for_wrapper = {k: v for k, v in pad.items() if k != "dummy_graph_id"}
    out = model_call(chain, pad_for_wrapper, num_species=4)

    # Real-graph energy slice (drop dummy at the end).
    E_real = out["E"][:-1]
    assert E_real.shape == (7,)  # 4 real graphs + 3 empty slots + dropped dummy

    # Per-atom mask (True on real atoms).
    atom_mask = real_atoms_mask(pad["inde"], pad["dummy_graph_id"])
    assert int(jnp.sum(atom_mask)) == 32
    # Mask is usable for force-loss reduction:
    target_F = jnp.zeros_like(out["F"])
    masked_sq = ((out["F"] - target_F) ** 2) * atom_mask[:, None]
    n_real = jnp.sum(atom_mask)
    f_loss = jnp.sum(masked_sq) / (n_real * 3)
    assert bool(jnp.isfinite(f_loss))
