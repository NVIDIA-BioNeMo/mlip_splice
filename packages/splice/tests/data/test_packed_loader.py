# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-loader planning and persistent prefetch."""

from __future__ import annotations

import json

import numpy as np
from splice.packed_data import PackedPrefetch, plan_balanced_batches


def _write_pack(path, n_struct=12, natoms=4):
    rng = np.random.default_rng(0)
    n_atoms = n_struct * natoms
    path.mkdir(parents=True)
    np.save(path / "Z.npy", np.ones(n_atoms, np.int8))
    np.save(path / "pos.npy", rng.normal(size=(n_atoms, 3)).astype(np.float32) * 0.4)
    np.save(path / "forces.npy", rng.normal(size=(n_atoms, 3)).astype(np.float32))
    np.save(path / "energy.npy", rng.normal(size=(n_struct,)).astype(np.float64))
    np.save(path / "natoms.npy", np.full(n_struct, natoms, np.int32))
    np.save(path / "offsets.npy", np.arange(0, n_atoms + 1, natoms, dtype=np.int64))
    np.save(path / "charge.npy", np.zeros(n_struct, np.int8))
    np.save(path / "spin.npy", np.ones(n_struct, np.int8))
    np.save(path / "source_id.npy", np.zeros(n_struct, np.int8))
    (path / "meta.json").write_text(
        json.dumps({"n_struct": n_struct, "n_atoms": n_atoms, "periodic": False})
    )


def test_balanced_plans_lockstep_across_ranks():
    natoms = np.array([3, 8, 5, 12, 4, 9, 6, 7, 11, 2, 10, 5], np.int32)
    a = plan_balanced_batches(natoms, atom_budget=20, max_structs=4, seed=0, rank=0, world_size=2)
    b = plan_balanced_batches(natoms, atom_budget=20, max_structs=4, seed=0, rank=1, world_size=2)
    assert len(a) == len(b)
    assert len(a) > 0
    for batch in a:
        assert int(natoms[batch].sum()) <= 20
        assert len(batch) <= 4


def test_prefetch_reuses_plan_and_two_epochs(tmp_path):
    pack = tmp_path / "pack"
    _write_pack(pack)
    spec = dict(path=str(pack), species_map={"H": 0}, cutoff=2.0, neighbour="free")
    with PackedPrefetch(
        spec,
        batch_size=4,
        base_atoms=64,
        base_edges=1024,
        num_workers=0,
        prefetch=2,
        atom_budget=16,
        edge_budget=512,
        graph_budget=8,
    ) as loader:
        first = loader.plan(seed=1)
        second = loader.plan(seed=1)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
        batches_a = list(loader.iter_epoch(1))
        batches_b = list(loader.iter_epoch(1))
        assert len(batches_a) == len(batches_b)
        assert batches_a[0]["species"].shape == batches_b[0]["species"].shape
        other = list(loader.iter_epoch(2))
        assert len(other) == len(batches_a)
        loader.prefetch_plan(3)
        prefetched = loader.take_plan(3)
        again = loader.plan(3)
        assert len(prefetched) == len(again)
        assert all(np.array_equal(a, b) for a, b in zip(prefetched, again, strict=True))


def test_prefetch_with_one_worker(tmp_path):
    pack = tmp_path / "pack"
    _write_pack(pack)
    spec = dict(path=str(pack), species_map={"H": 0}, cutoff=2.0, neighbour="free")
    with PackedPrefetch(
        spec,
        batch_size=4,
        base_atoms=64,
        base_edges=1024,
        num_workers=1,
        prefetch=2,
        atom_budget=16,
        edge_budget=512,
        graph_budget=8,
    ) as loader:
        a = list(loader.iter_epoch(1))
        b = list(loader.iter_epoch(1))
        assert len(a) == len(b) > 0
        assert a[0]["species"].shape == b[0]["species"].shape
