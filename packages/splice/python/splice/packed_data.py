# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming loader for large non-periodic molecular datasets stored in a
packed, memmap-friendly layout (a directory of contiguous ``.npy`` arrays).

Why a separate path from ``data.py``: that loader reads an extxyz file fully
into RAM with per-frame neighbour lists precomputed — fine for ~10^5 frames,
infeasible at ~10^6+. Here the arrays are **memmapped** (shared read-only
across workers via the OS page cache) and frames are built **lazily per
batch**, with batch construction farmed to a process pool so batch N+1 is
assembled while the GPU consumes batch N.

Packed layout (one dir):
  Z.npy (Σatoms,) int8 | pos.npy (Σatoms,3) f32 | forces.npy (Σatoms,3) f32
  energy.npy (N,) f64 (RAW total energy) | natoms.npy (N,) i32
  offsets.npy (N+1,) i64 (CSR index: struct i = arr[off[i]:off[i+1]])
  charge.npy / spin.npy / source_id.npy (N,) i8 | meta.json
Energy is raw; the per-species offset subtraction happens in
``make_padded_batch`` (pass ``energy_offsets``), as on the extxyz path.

Batching is split into *planning* (which indices form a batch) and *building*
(slice → frame → collate → pad). The planner here groups a fixed number of
configs per batch; an atom/edge-budget planner is a drop-in replacement (it
only needs ``self.natoms``, which is memmapped without touching geometry).

Neighbour list: these frames are non-periodic, so a vectorised brute-force
list (``i`` sender, ``j`` receiver, ``D = r_j - r_i``) matches the extxyz
``neighbour_list("ijD", ...)`` convention without ASE/matscipy.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers

from splice.data import collate_for_pad, make_padded_batch

_ARRAY_NAMES = (
    "Z",
    "pos",
    "forces",
    "energy",
    "natoms",
    "offsets",
    "charge",
    "spin",
    "source_id",
)


# --- neighbour-list backends ---------------------------------------------
# Uniform signature: fn(positions, numbers, cell, pbc, cutoff) ->
#   (senders[int32], receivers[int32], edge_vectors[float32]),
# with the matscipy "ijD" convention: edge_vector = r_receiver - r_sender.
# Select per-dataset via `PackedDataset(neighbour=...)`. A custom callable
# (e.g. a future GPU list) can be passed directly.


def _free_neighbour_list(positions, numbers, cell, pbc, cutoff):
    """Brute-force list for a single **non-periodic** structure (no min-image).

    O(N^2) in atoms, but molecules here top out near ~180 atoms so this is
    trivial and avoids building an ASE/matscipy cell decomposition. ``numbers``
    /``cell``/``pbc`` are unused (no periodicity)."""
    diff = positions[None, :, :] - positions[:, None, :]  # diff[a,b]=pos[b]-pos[a]
    dist = np.sqrt((diff * diff).sum(-1))
    mask = dist < cutoff
    np.fill_diagonal(mask, False)
    senders, receivers = np.nonzero(mask)
    return (
        senders.astype(np.int32),
        receivers.astype(np.int32),
        diff[senders, receivers].astype(np.float32),
    )


def _matscipy_neighbour_list(positions, numbers, cell, pbc, cutoff):
    """ASE/matscipy list — handles periodicity (min-image). Use for periodic
    data, or to cross-check the free list on non-periodic frames."""
    from ase import Atoms
    from matscipy.neighbours import neighbour_list

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    i, j, d = neighbour_list("ijD", atoms, cutoff)
    return i.astype(np.int32), j.astype(np.int32), d.astype(np.float32)


_NEIGHBOUR_FNS = {"free": _free_neighbour_list, "matscipy": _matscipy_neighbour_list}


def _resolve_neighbour(neighbour):
    if callable(neighbour):
        return neighbour
    if neighbour in _NEIGHBOUR_FNS:
        return _NEIGHBOUR_FNS[neighbour]
    raise ValueError(
        f"unknown neighbour fn {neighbour!r}; options {list(_NEIGHBOUR_FNS)} or a callable"
    )


class PackedDataset:
    """Read-only view over a packed dataset dir. Cheap to construct/pickle;
    memmaps are opened lazily so the object can cross a process boundary."""

    def __init__(
        self,
        path: str | Path,
        species_map: dict[str, int],
        cutoff: float,
        neighbour: str | callable = "free",
    ):
        self.path = str(path)
        self.cutoff = float(cutoff)
        self.species_map = dict(species_map)
        self.neighbour = neighbour  # "free" | "matscipy" | callable (kept for pickling)
        self._neighbour_fn = _resolve_neighbour(neighbour)
        with open(Path(self.path) / "meta.json") as f:
            self.meta = json.load(f)
        self.n_struct = int(self.meta["n_struct"])
        # Periodic packs (e.g. OMat24) carry real cell + stress arrays and need
        # a pbc-aware neighbour list. The free list ignores pbc, so auto-upgrade
        # to matscipy when the caller left the molecular default in place.
        self.periodic = bool(self.meta.get("periodic", False))
        if self.periodic and neighbour == "free":
            print(f"[packed] {self.path}: periodic pack -> neighbour 'free'->'matscipy'")
            self.neighbour = "matscipy"
            self._neighbour_fn = _resolve_neighbour("matscipy")
        # Z (atomic number) -> species id, via the card's symbol->id map.
        self._z_to_id = np.full(128, -1, dtype=np.int32)
        for sym, sid in self.species_map.items():
            self._z_to_id[atomic_numbers[sym]] = sid
        self._arr: dict[str, np.ndarray] | None = None  # lazy memmaps

    # --- pickling: drop memmaps; reopen lazily in the child process ---
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_arr"] = None
        return s

    def _open(self) -> dict[str, np.ndarray]:
        if self._arr is None:
            d = Path(self.path)
            names = list(_ARRAY_NAMES) + (["cell", "stress"] if self.periodic else [])
            self._arr = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in names}
        return self._arr

    @property
    def natoms(self) -> np.ndarray:
        """Per-structure atom counts (for batch planning; no geometry read)."""
        return np.asarray(self._open()["natoms"])

    def make_frame(self, i: int) -> dict:
        arr = self._open()
        a, b = int(arr["offsets"][i]), int(arr["offsets"][i + 1])
        z = np.asarray(arr["Z"][a:b]).astype(np.int64)
        positions = np.asarray(arr["pos"][a:b], dtype=np.float32)
        forces = np.asarray(arr["forces"][a:b], dtype=np.float32)
        species = self._z_to_id[z]
        if np.any(species < 0):
            bad = sorted(set(int(zz) for zz in z[species < 0]))
            raise KeyError(f"structure {i} has Z {bad} not in the species map")
        species = species.astype(np.int32)
        n = len(z)
        if self.periodic:
            # Real lattice + stress; pbc list neighbours across cell faces.
            cell = np.asarray(arr["cell"][i], dtype=np.float32)
            stress = np.asarray(arr["stress"][i], dtype=np.float32)
            pbc = True
        else:
            # bounding-box cell (non-periodic); volume is unused by the loss here.
            pmin = positions.min(axis=0) - self.cutoff
            pmax = positions.max(axis=0) + self.cutoff
            cell = np.diag(pmax - pmin).astype(np.float32)
            stress = np.zeros((3, 3), dtype=np.float32)
            pbc = False
        senders, receivers, edge_vectors = self._neighbour_fn(
            positions, z.astype(np.int32), cell, pbc, self.cutoff
        )
        frame = {
            "species": species,
            "positions": positions,
            "cell": cell,
            "energy": float(arr["energy"][i]),
            "forces": forces,
            "stress": stress,
            "senders": senders,
            "receivers": receivers,
            "edge_vectors": edge_vectors,
        }
        # Per-STRUCTURE conditioning scalars (OMol: total charge + spin
        # multiplicity). Present in _ARRAY_NAMES and loaded from disk, but they
        # were never placed on the frame, so everything downstream saw None and
        # the charge embedding ran on its neutral default. Guarded: datasets
        # without these fields are unaffected.
        for _k in ("charge", "spin"):
            if _k in arr:
                frame[_k] = int(arr[_k][i])
        return frame


def plan_batches(
    n_struct: int,
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> list[np.ndarray]:
    """Plan fixed-config batches over a (sharded, shuffled) index sequence.

    Sharding (``rank``/``world_size``) sets up data-parallel: each rank gets a
    disjoint stride of the shuffled indices. A future atom/edge-budget planner
    would replace this body, consuming ``PackedDataset.natoms`` to pack each
    batch to a near-constant total-atom count (fewer pad shapes / recompiles).
    """
    idx = np.arange(n_struct)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    idx = idx[rank::world_size]
    batches = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    return batches


def plan_balanced_batches(
    natoms: np.ndarray,
    *,
    atom_budget: int,
    max_structs: int | None = None,
    mega_size: int | None = None,
    mega_batches: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> list[np.ndarray]:
    """Atom-budget balanced batching via the mega-batch trick (the OMat path).

    Replaces fixed-config batching when structures vary widely in size
    (OMat: ~1–236 atoms): packs each batch to ``sum(natoms) <= atom_budget`` so
    the SINGLE fixed pad shape ``(atom_budget, edge_budget)`` is ~95% full
    instead of count-based-padded to ~2× the real work. One compiled shape (the
    bottleneck), tight fill, and the global shuffle preserved:

    Per rank, per epoch::

        shuffle(all idx)                          # fresh order; same seed across ranks
        shard = idx[rank::world_size]             # disjoint per-rank slice
        for mega in chunks(shard, mega_size):     # window of ~mega_batches batches
            sort mega by natoms                   # consecutive ≈ equal size -> tight fill
            greedy-pack mega -> batches <= atom_budget
        shuffle(batch order)                      # SGD sees batches un-sorted-by-size
        truncate to min batch-count across ranks  # lockstep grad all-reduce

    The pack is **per-rank independent** (every struct ``<= atom_budget`` so it
    always fits a fresh batch; no global split). All ranks share ``seed`` +
    ``natoms`` so each can replay every rank's pack to find the lockstep
    ``min`` count without any collective. ``mega_size`` defaults to
    ``mega_batches × (atom_budget / mean_atoms)`` (≈ that many batches' worth,
    dataset-adaptive). Returns this rank's list of index arrays (absolute
    dataset indices); ``len`` is identical across ranks.

    ``atom_budget`` here is the cap on **real** atoms per batch; the caller pads
    to ``atom_budget + 1`` (room for the dummy atom). Edges are not capped at
    plan time (no NL over 100M structs): ``edge_budget`` is data-sized to never
    overflow, with a pop-last drop-guard in :func:`build_batch` as backstop.

    ``max_structs`` caps structs/batch so the padded GRAPH axis stays a single
    compiled shape (``fixed_graphs = max_structs + 1``). It is enforced by
    **flushing the batch early** (the leftover structs simply start the next
    batch) — never by dropping — so no structure is lost. Set it generously
    (graphs are cheap); on OMat real data structs/batch tops out ~128, so a cap
    of a few hundred essentially never binds. ``None`` = no cap.
    """
    natoms = np.asarray(natoms)
    n = len(natoms)
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    if mega_size is None:
        mean_atoms = max(float(natoms.mean()), 1.0)
        per_batch = max(1, int(atom_budget / mean_atoms))
        mega_size = max(per_batch, mega_batches * per_batch)

    def _pack(shard: np.ndarray) -> list[np.ndarray]:
        batches: list[np.ndarray] = []
        for m0 in range(0, len(shard), mega_size):
            mega = shard[m0 : m0 + mega_size]
            mega = mega[np.argsort(natoms[mega], kind="stable")]
            cur: list[int] = []
            cur_atoms = 0
            for i in mega:
                a = int(natoms[i])
                if cur and (
                    cur_atoms + a > atom_budget
                    or (max_structs is not None and len(cur) >= max_structs)
                ):
                    batches.append(np.asarray(cur, dtype=np.int64))
                    cur, cur_atoms = [], 0
                cur.append(int(i))
                cur_atoms += a
            if cur:
                batches.append(np.asarray(cur, dtype=np.int64))
        return batches

    # Replay every rank's pack (cheap — natoms only) to get the lockstep count.
    counts: list[int] = []
    my_batches: list[np.ndarray] | None = None
    for r in range(world_size):
        br = _pack(order[r::world_size])
        counts.append(len(br))
        if r == rank:
            my_batches = br
    keep = min(counts)
    # Shuffle batch ORDER (per-rank derived seed; reproducible) then truncate to
    # the lockstep min. Order/content differ per rank (different shards) — only
    # the COUNT must match for the gradient all-reduce to stay in step.
    perm = np.random.default_rng(seed + 1 + rank).permutation(len(my_batches))
    return [my_batches[k] for k in perm[:keep]]


def build_batch(
    dataset: PackedDataset,
    indices: np.ndarray,
    *,
    base_atoms: int,
    base_edges: int,
    energy_offsets: np.ndarray | None,
    fixed_atoms: int | None = None,
    fixed_edges: int | None = None,
    fixed_graphs: int | None = None,
    grid: bool = False,
    drop_guard: bool = False,
) -> dict:
    """Slice → frames → collated, padded batch. Pure + picklable (the unit of
    work farmed to the prefetch pool). ``fixed_atoms``/``fixed_edges`` force an
    identical padded shape across calls (multi-process lockstep).

    ``grid=True`` returns the UN-padded collate payload instead (a dict with
    ``raw``/``targets``/``n_frames`` + real ``real_atoms``/``real_edges``); the
    caller pads it in the main process after a cross-rank shape allgather, so
    every rank lands on the same grid bucket without over-padding to a global
    fixed max. ``fixed_*``/``base_*`` are ignored in grid mode.

    ``drop_guard=True`` (balanced-binning path): if the collated batch would
    overflow the fixed pad shape (real atoms+1 > ``fixed_atoms`` or real edges >
    ``fixed_edges``), pop trailing frames until it fits. The planner caps atoms
    so atoms never overflow; this only ever fires on a rare edge spike, and
    drops at most a struct or two (deferred to the next epoch's reshuffle). Keeps
    the fixed shape with no crash/recompile and no cross-rank coordination."""
    frames = [dataset.make_frame(int(i)) for i in indices]
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
    if grid:
        raw, targets, n_frames = collate_for_pad(frames, energy_offsets=energy_offsets)
        return {
            "_grid_payload": True,
            "raw": raw,
            "targets": targets,
            "n_frames": n_frames,
            "real_atoms": len(raw["species"]),
            "real_edges": len(raw["nn_vecs"]),
        }
    return make_padded_batch(
        frames,
        base_atoms=base_atoms,
        base_edges=base_edges,
        energy_offsets=energy_offsets,
        fixed_atoms=fixed_atoms,
        fixed_edges=fixed_edges,
        fixed_graphs=fixed_graphs,
    )


# --- process-pool worker globals (set once per worker via initializer) ---
_W_DS: PackedDataset | None = None
_W_BA = _W_BE = None
_W_OFF: np.ndarray | None = None
_W_FA = _W_FE = _W_FG = None
_W_GRID = False
_W_DROP = False


def _init_worker(
    spec,
    base_atoms,
    base_edges,
    energy_offsets,
    fixed_atoms,
    fixed_edges,
    grid=False,
    drop_guard=False,
    fixed_graphs=None,
):
    global _W_DS, _W_BA, _W_BE, _W_OFF, _W_FA, _W_FE, _W_FG, _W_GRID, _W_DROP
    _W_DS = PackedDataset(**spec)
    _W_BA, _W_BE = base_atoms, base_edges
    _W_OFF = energy_offsets
    _W_FA, _W_FE, _W_FG = fixed_atoms, fixed_edges, fixed_graphs
    _W_GRID = grid
    _W_DROP = drop_guard


def _work(indices):
    return build_batch(
        _W_DS,
        indices,
        base_atoms=_W_BA,
        base_edges=_W_BE,
        energy_offsets=_W_OFF,
        fixed_atoms=_W_FA,
        fixed_edges=_W_FE,
        fixed_graphs=_W_FG,
        grid=_W_GRID,
        drop_guard=_W_DROP,
    )


class PackedPrefetch:
    """Prefetch padded batches; keep the worker pool alive across epochs.

    Construct once per run. Call :meth:`iter_epoch` for each shuffle seed.
    ``num_workers<=0`` builds batches inline (eval/debug).
    """

    def __init__(
        self,
        spec: dict,
        *,
        batch_size: int,
        base_atoms: int = 2048,
        base_edges: int = 49152,
        energy_offsets: np.ndarray | list[float] | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 4,
        prefetch: int = 4,
        rank: int = 0,
        world_size: int = 1,
        max_struct: int | None = None,
        fixed_atoms: int | None = None,
        fixed_edges: int | None = None,
        grid: bool = False,
        atom_budget: int | None = None,
        edge_budget: int | None = None,
        graph_budget: int = 256,
        mega_size: int | None = None,
        mega_batches: int = 32,
    ):
        self.spec = spec
        self.ds_main = PackedDataset(**spec)
        self.n = (
            self.ds_main.n_struct
            if max_struct is None
            else min(self.ds_main.n_struct, int(max_struct))
        )
        self.batch_size = batch_size
        self.base_atoms = base_atoms
        self.base_edges = base_edges
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch = max(1, prefetch)
        self.rank = rank
        self.world_size = world_size
        self.grid = grid
        self.atom_budget = atom_budget
        self.edge_budget = edge_budget
        self.graph_budget = graph_budget
        self.mega_size = mega_size
        self.mega_batches = mega_batches
        self.balanced = atom_budget is not None
        self.fixed_atoms = fixed_atoms
        self.fixed_edges = fixed_edges
        self.fixed_graphs = None
        if self.balanced:
            if edge_budget is None:
                raise ValueError("balanced binning needs edge_budget alongside atom_budget")
            self.fixed_atoms, self.fixed_edges, self.fixed_graphs = (
                atom_budget + 1,
                edge_budget,
                graph_budget,
            )
        self.off = None if energy_offsets is None else np.asarray(energy_offsets)
        self._ex: ProcessPoolExecutor | None = None
        self._plan_ex: ThreadPoolExecutor | None = None
        self._next_plan: tuple[int, object] | None = None
        self.wait_s: deque[float] = deque(maxlen=4096)
        if num_workers > 0:
            ctx = mp.get_context("forkserver")
            with contextlib.suppress(RuntimeError, ValueError):
                ctx.set_forkserver_preload(["splice.packed_data"])
            self._ex = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(
                    spec,
                    base_atoms,
                    base_edges,
                    self.off,
                    self.fixed_atoms,
                    self.fixed_edges,
                    grid,
                    self.balanced,
                    self.fixed_graphs,
                ),
            )
        self._plan_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="splice-plan")

    def plan(self, seed: int) -> list[np.ndarray]:
        if self.balanced:
            return plan_balanced_batches(
                self.ds_main.natoms[: self.n],
                atom_budget=self.atom_budget,
                max_structs=self.graph_budget - 1,
                mega_size=self.mega_size,
                mega_batches=self.mega_batches,
                shuffle=self.shuffle,
                seed=seed,
                rank=self.rank,
                world_size=self.world_size,
            )
        return plan_batches(
            self.n,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=seed,
            drop_last=self.drop_last,
            rank=self.rank,
            world_size=self.world_size,
        )

    def prefetch_plan(self, seed: int) -> None:
        """Compute ``plan(seed)`` on a host thread (safe to call during GPU work)."""
        if self._plan_ex is None:
            return
        pending = self._next_plan
        if pending is not None and pending[0] == seed:
            return
        self._next_plan = (seed, self._plan_ex.submit(self.plan, seed))

    def take_plan(self, seed: int) -> list[np.ndarray]:
        """Return ``plan(seed)``, consuming a matching prefetch if it is ready."""
        pending = self._next_plan
        if pending is not None and pending[0] == seed:
            self._next_plan = None
            return pending[1].result()
        return self.plan(seed)

    def _build(self, indices: np.ndarray) -> dict:
        return build_batch(
            self.ds_main,
            indices,
            base_atoms=self.base_atoms,
            base_edges=self.base_edges,
            energy_offsets=self.off,
            fixed_atoms=self.fixed_atoms,
            fixed_edges=self.fixed_edges,
            fixed_graphs=self.fixed_graphs,
            grid=self.grid,
            drop_guard=self.balanced,
        )

    def iter_epoch(self, seed: int, batches: list[np.ndarray] | None = None):
        """Yield one shuffled epoch. The process pool is reused across calls."""
        batches = self.take_plan(seed) if batches is None else batches
        if self._ex is None:
            for b in batches:
                yield self._build(b)
            return
        it = iter(batches)
        pending: deque = deque()
        for _ in range(self.prefetch):
            b = next(it, None)
            if b is None:
                break
            pending.append(self._ex.submit(_work, b))
        while pending:
            t0 = time.perf_counter()
            res = pending.popleft().result()
            self.wait_s.append(time.perf_counter() - t0)
            b = next(it, None)
            if b is not None:
                pending.append(self._ex.submit(_work, b))
            yield res

    def close(self) -> None:
        if self._next_plan is not None:
            with contextlib.suppress(Exception):
                self._next_plan[1].cancel()
            self._next_plan = None
        if self._plan_ex is not None:
            self._plan_ex.shutdown(wait=True)
            self._plan_ex = None
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def packed_batched_iter(
    spec: dict,
    *,
    batch_size: int,
    base_atoms: int = 2048,
    base_edges: int = 49152,
    energy_offsets: np.ndarray | list[float] | None = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    num_workers: int = 4,
    prefetch: int = 4,
    rank: int = 0,
    world_size: int = 1,
    max_struct: int | None = None,
    fixed_atoms: int | None = None,
    fixed_edges: int | None = None,
    grid: bool = False,
    atom_budget: int | None = None,
    edge_budget: int | None = None,
    graph_budget: int = 256,
    mega_size: int | None = None,
    mega_batches: int = 32,
):
    """Yield padded batches from a packed dataset, prefetched across a pool.

    ``spec`` is the kwargs to build a ``PackedDataset`` (``path``,
    ``species_map``, ``cutoff``) — passed (not the opened object) so each
    worker memmaps independently. ``num_workers<=0`` runs inline (no pool),
    useful for eval/debug. Batches are yielded in plan order; up to
    ``prefetch`` are kept in flight.

    ``fixed_atoms``/``fixed_edges`` pin every yielded batch to the same padded
    shape — required for multi-process data parallelism (each rank's sub-batch
    must be identically shaped). ``rank``/``world_size`` give each rank a
    disjoint slice of the dataset (set to process_index/process_count).

    **Balanced binning** (``atom_budget`` set): pack each batch to
    ``sum(natoms) <= atom_budget`` via :func:`plan_balanced_batches` instead of
    a fixed struct count — tight ~95% fill of one compiled shape for
    wide-size-distribution data (OMat). The single pad shape is derived as
    ``fixed_atoms = atom_budget + 1`` (dummy atom), ``fixed_edges = edge_budget``
    (must never be exceeded; backed by the drop-guard), ``fixed_graphs =
    graph_budget`` (struct count capped by early-flush so the graph axis is
    fixed; graphs are cheap so over-pad freely). ``batch_size`` is ignored here.
    """
    with PackedPrefetch(
        spec,
        batch_size=batch_size,
        base_atoms=base_atoms,
        base_edges=base_edges,
        energy_offsets=energy_offsets,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        prefetch=prefetch,
        rank=rank,
        world_size=world_size,
        max_struct=max_struct,
        fixed_atoms=fixed_atoms,
        fixed_edges=fixed_edges,
        grid=grid,
        atom_budget=atom_budget,
        edge_budget=edge_budget,
        graph_budget=graph_budget,
        mega_size=mega_size,
        mega_batches=mega_batches,
    ) as loader:
        yield from loader.iter_epoch(seed)


def avg_num_neighbours(spec: dict, *, n_sample: int = 5000, seed: int = 0) -> float:
    """Mean edges-per-atom over a random sample (for the card's channelwise
    epsilon = 1/avg_num_neighbours at this cutoff)."""
    ds = PackedDataset(**spec)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, ds.n_struct, size=min(n_sample, ds.n_struct))
    edges = atoms = 0
    for i in picks:
        f = ds.make_frame(int(i))
        edges += len(f["senders"])
        atoms += len(f["species"])
    return edges / max(atoms, 1)
