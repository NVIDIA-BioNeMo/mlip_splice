# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous multi-head data path for Stage-2 multi-head replay finetuning.

Unlike the single-dataset packed loader, here EACH BATCH mixes configurations from
all heads (datasets), proportional to their (sub-sampled) sizes — the standard
multi-task / MACE multi-head replay setup. A mixed batch averages every domain's
gradient per step (low-variance, stable) instead of whipsawing the shared backbone
one domain at a time, and it blends the dense-periodic / sparse-molecular edge
distributions so a single padded shape suffices.

Design (memmap philosophy, like packed_data):
  build_multihead_index(...) does the one-time work — subsample each pack (e.g. OMat
  100M -> 10M), concatenate into a global config list, and write the index as memmap
  arrays. Workers then memmap the index (no per-worker resampling, no big pickles).

Per global config g the index stores: pack_of[g], local_of[g] (index within that
pack), head_of[g], vir_of[g] (1.0 if the head trains stress else 0.0), natoms[g].

PackedMultiHeadDataset.make_frame(g) routes to the right pack, subtracts THAT head's
per-species E0 from the raw energy (per-config — heads have very different references,
molecular ~ -1000 eV/atom vs PBE ~ -1 eV/atom), and tags the frame with head_id +
virial_weight. The model ops (multihead_energy / multihead_species_offsets) route per
atom via head_id[graph_index]; the loss gates virial per graph — both already per-graph,
so nothing downstream changes between homogeneous and heterogeneous batches.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from splice.data import make_padded_batch
from splice.packed_data import PackedDataset, plan_balanced_batches


# --- E0 helpers -----------------------------------------------------------
def e0_row_for_head(e0_table: dict, head: str, species_map: dict[str, int]) -> np.ndarray:
    """Per-species E0 vector (species-id order) for ``head`` from a checkpoint's
    ``E0_eV[head]`` (keyed by Z-string). Missing species -> 0.0."""
    from ase.data import atomic_numbers

    tbl = e0_table[head]
    row = np.zeros(len(species_map), dtype=np.float64)
    for sym, sid in species_map.items():
        row[sid] = float(tbl.get(str(atomic_numbers[sym]), 0.0))
    return row


def fit_e0_lstsq(
    pack_path: str, species_map: dict[str, int], *, max_struct: int = 500_000, seed: int = 0
) -> np.ndarray:
    """Least-squares per-species E0 (the paper's overdetermined N c = E system),
    fit on up to ``max_struct`` configs of a packed dataset. Returns [num_species]
    in species-id order. Species absent from the sample get 0.0."""
    from ase.data import atomic_numbers

    d = Path(pack_path)
    Z = np.load(d / "Z.npy", mmap_mode="r")
    E = np.load(d / "energy.npy", mmap_mode="r")
    off = np.load(d / "offsets.npy", mmap_mode="r")
    nstruct = len(E)
    z2id = np.full(128, -1, np.int64)
    for sym, sid in species_map.items():
        z2id[atomic_numbers[sym]] = sid
    S = len(species_map)
    rng = np.random.default_rng(seed)
    idx = (
        np.arange(nstruct)
        if nstruct <= max_struct
        else np.sort(rng.choice(nstruct, max_struct, replace=False))
    )
    counts = np.zeros((len(idx), S), np.float64)
    energies = np.zeros(len(idx), np.float64)
    for r, i in enumerate(idx):
        zi = np.asarray(Z[int(off[i]) : int(off[i + 1])]).astype(int)
        sid = z2id[zi]
        np.add.at(counts[r], sid[sid >= 0], 1.0)
        energies[r] = float(E[i])
    coef, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return coef


# --- one-time index build -------------------------------------------------
def build_multihead_index(
    packs: list[dict],
    e0_table: np.ndarray,
    out_dir: str,
    *,
    species_map: dict[str, int],
    cutoff: float,
    neighbour: str = "free",
    subset_seed: int = 0,
) -> None:
    """Select per-pack subsets and write the global config index (memmap arrays).

    ``packs[i]`` = {path, head_id, periodic, virial (opt, default=periodic),
    subset (opt int)}. ``e0_table`` is [num_heads, num_species] (species-id order)
    — head ``head_id`` uses row ``head_id``. Run ONCE (subsampling is the only
    expensive bit); the result is memmapped read-only by all workers."""
    rng = np.random.default_rng(subset_seed)
    pack_of, local_of, head_of, vir_of, nat_of = [], [], [], [], []
    meta_packs = []
    for pi, p in enumerate(packs):
        ds = PackedDataset(
            path=p["path"], species_map=species_map, cutoff=cutoff, neighbour=neighbour
        )
        n = ds.n_struct
        sub = p.get("subset")
        if not sub or sub >= n:
            idx = np.arange(n, dtype=np.int64)
        else:
            idx = np.sort(rng.choice(n, int(sub), replace=False)).astype(np.int64)
        nat = np.asarray(ds.natoms)[idx].astype(np.int32)
        vir = 1.0 if bool(p.get("virial", p["periodic"])) else 0.0
        pack_of.append(np.full(len(idx), pi, np.int16))
        local_of.append(idx)
        head_of.append(np.full(len(idx), int(p["head_id"]), np.int8))
        vir_of.append(np.full(len(idx), vir, np.float32))
        nat_of.append(nat)
        meta_packs.append(
            {
                "path": p["path"],
                "head_id": int(p["head_id"]),
                "periodic": bool(p["periodic"]),
                "name": p.get("name", f"pack{pi}"),
                "n_total": int(n),
                "n_used": len(idx),
            }
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pack_of.npy", np.concatenate(pack_of))
    np.save(out / "local_of.npy", np.concatenate(local_of))
    np.save(out / "head_of.npy", np.concatenate(head_of))
    np.save(out / "vir_of.npy", np.concatenate(vir_of))
    np.save(out / "natoms.npy", np.concatenate(nat_of))
    np.save(out / "e0_table.npy", np.asarray(e0_table, np.float64))
    meta = {
        "packs": meta_packs,
        "species_map": species_map,
        "cutoff": float(cutoff),
        "neighbour": neighbour,
        "num_species": len(species_map),
        "n_struct": int(sum(len(x) for x in nat_of)),
        "subset_seed": subset_seed,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[mh-index] {meta['n_struct']} configs across {len(packs)} heads -> {out}")
    for mp_ in meta_packs:
        print(
            f"  head {mp_['head_id']:>2} {mp_['name']:14s} {mp_['n_used']:>10,}/{mp_['n_total']:,}"
        )


# --- unified memmapped dataset --------------------------------------------
class PackedMultiHeadDataset:
    """Read-only unified view over the multi-head index. Cheap to construct/pickle
    (memmaps opened lazily so it crosses process boundaries)."""

    def __init__(self, index_dir: str):
        self.index_dir = str(index_dir)
        with open(Path(self.index_dir) / "meta.json") as f:
            self.meta = json.load(f)
        self.species_map = self.meta["species_map"]
        self.cutoff = float(self.meta["cutoff"])
        self.neighbour = self.meta["neighbour"]
        self.num_species = int(self.meta["num_species"])
        self.n_struct = int(self.meta["n_struct"])
        self._idx = None  # lazy memmaps
        self._packs = None  # lazy PackedDataset list
        self._e0 = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_idx"] = s["_packs"] = s["_e0"] = None
        return s

    def _open(self):
        if self._idx is None:
            d = Path(self.index_dir)
            self._idx = {
                n: np.load(d / f"{n}.npy", mmap_mode="r")
                for n in ("pack_of", "local_of", "head_of", "vir_of", "natoms")
            }
            self._e0 = np.load(d / "e0_table.npy")
            self._packs = [
                PackedDataset(
                    path=p["path"],
                    species_map=self.species_map,
                    cutoff=self.cutoff,
                    neighbour=self.neighbour,
                )
                for p in self.meta["packs"]
            ]
        return self._idx

    @property
    def natoms(self) -> np.ndarray:
        return np.asarray(self._open()["natoms"])

    def make_frame(self, g: int) -> dict:
        idx = self._open()
        pi = int(idx["pack_of"][g])
        frame = self._packs[pi].make_frame(int(idx["local_of"][g]))
        head = int(idx["head_of"][g])
        species = frame["species"]
        # per-config E0: subtract THIS head's per-species reference from raw energy
        frame["energy"] = float(frame["energy"]) - float(self._e0[head, species].sum())
        frame["head_id"] = head
        frame["virial_weight"] = float(idx["vir_of"][g])
        return frame


# --- heterogeneous balanced-binning iterator (with prefetch pool) ---------
def _mh_build_batch(
    ds, indices, *, base_atoms, base_edges, fixed_atoms, fixed_edges, fixed_graphs, drop_guard
):
    frames = [ds.make_frame(int(i)) for i in indices]
    if drop_guard and fixed_atoms is not None and len(frames) > 1:
        ra = sum(len(f["species"]) for f in frames)
        re = sum(len(f["senders"]) for f in frames)
        cap_g = (fixed_graphs - 1) if fixed_graphs is not None else None
        while len(frames) > 1 and (
            ra + 1 > fixed_atoms or re > fixed_edges or (cap_g is not None and len(frames) > cap_g)
        ):
            f = frames.pop()
            ra -= len(f["species"])
            re -= len(f["senders"])
    batch = make_padded_batch(
        frames,
        base_atoms=base_atoms,
        base_edges=base_edges,
        energy_offsets=None,
        fixed_atoms=fixed_atoms,
        fixed_edges=fixed_edges,
        fixed_graphs=fixed_graphs,
    )
    ng = batch["nats"].shape[0]
    head = np.zeros(ng, np.int32)
    vir = np.zeros(ng, np.float32)
    head[: len(frames)] = [f["head_id"] for f in frames]
    vir[: len(frames)] = [f["virial_weight"] for f in frames]
    batch["head_id"] = head
    batch["virial_weight"] = vir
    return batch


_W = {}


def _mh_init(index_dir, base_atoms, base_edges, fa, fe, fg, drop_guard):
    _W.update(
        ds=PackedMultiHeadDataset(index_dir),
        ba=base_atoms,
        be=base_edges,
        fa=fa,
        fe=fe,
        fg=fg,
        dg=drop_guard,
    )


def _mh_work(indices):
    return _mh_build_batch(
        _W["ds"],
        indices,
        base_atoms=_W["ba"],
        base_edges=_W["be"],
        fixed_atoms=_W["fa"],
        fixed_edges=_W["fe"],
        fixed_graphs=_W["fg"],
        drop_guard=_W["dg"],
    )


def multihead_batched_iter(
    index_dir: str,
    *,
    atom_budget: int,
    edge_budget: int,
    graph_budget: int = 256,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    prefetch: int = 4,
    mega_batches: int = 32,
    base_atoms: int = 2048,
    base_edges: int = 49152,
):
    """Yield heterogeneous balanced-binning batches (one shared padded shape across
    all heads). Each batch mixes heads ~proportional to their index sizes; per-graph
    ``head_id`` / ``virial_weight`` ride along for the model + loss."""
    ds_main = PackedMultiHeadDataset(index_dir)
    fixed_atoms, fixed_edges, fixed_graphs = atom_budget + 1, edge_budget, graph_budget
    batches = plan_balanced_batches(
        ds_main.natoms,
        atom_budget=atom_budget,
        max_structs=graph_budget - 1,
        mega_batches=mega_batches,
        shuffle=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    if num_workers <= 0:
        for b in batches:
            yield _mh_build_batch(
                ds_main,
                b,
                base_atoms=base_atoms,
                base_edges=base_edges,
                fixed_atoms=fixed_atoms,
                fixed_edges=fixed_edges,
                fixed_graphs=fixed_graphs,
                drop_guard=True,
            )
        return
    ctx = mp.get_context("forkserver")
    with contextlib.suppress(RuntimeError, ValueError):
        ctx.set_forkserver_preload(["splice.multihead_data"])
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_mh_init,
        initargs=(index_dir, base_atoms, base_edges, fixed_atoms, fixed_edges, fixed_graphs, True),
    ) as ex:
        it = iter(batches)
        pending: deque = deque()
        for _ in range(max(1, prefetch)):
            b = next(it, None)
            if b is None:
                break
            pending.append(ex.submit(_mh_work, b))
        while pending:
            res = pending.popleft().result()
            b = next(it, None)
            if b is not None:
                pending.append(ex.submit(_mh_work, b))
            yield res
