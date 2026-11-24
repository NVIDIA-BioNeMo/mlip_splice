# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data loading: extxyz → frames → padded batches.

Pipeline:
  1. `load_extxyz(path, species_to_id, cutoff)` reads an extxyz file via ASE,
     computes a per-frame neighbour list with matscipy, and returns a list of
     dicts (one per frame) carrying species, positions, forces, energy, and
     the (senders, receivers, edge_vectors) edge tuple.
  2. `batched_iter(frames, batch_size, ...)` shuffles and groups frames into
     batches. Each batch is collated (atom/edge indices remapped to
     concatenated arrays), padded via `splice.padding.pad_batch`, and
     yielded as a dict ready for `model_call`. Targets (`target_E`,
     `target_F`) are also padded to the bucketed shapes (zeros on padding
     entries — masked out by the loss).

Neighbour-list convention (matscipy `neighbour_list("ijD", ...)`):
  i = sender index, j = receiver index, D = r_j - r_i = edge vector.
  Matches `model_call`'s force scatter.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from matscipy.neighbours import neighbour_list

from splice.padding import choose_bucket_sizes, pad_batch


def _get_energy(atoms) -> float:
    """Pull total energy from `info["energy"]` or `calc.results["energy"]`.

    ASE's extxyz reader attaches a `SinglePointCalculator` to loaded frames
    when the comment line has `energy=...`/`forces=...` — the values land
    under `atoms.calc.results`, not in `atoms.info`/`atoms.arrays`.
    """
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if atoms.calc is not None and "energy" in atoms.calc.results:
        return float(atoms.calc.results["energy"])
    return 0.0


def _get_forces(atoms, n_atoms: int) -> np.ndarray:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=np.float32)
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return np.asarray(atoms.calc.results["forces"], dtype=np.float32)
    return np.zeros((n_atoms, 3), dtype=np.float32)


def _get_stress(atoms) -> np.ndarray:
    """Return a ``(3, 3)`` symmetric stress tensor in eV/Å³, or zeros if absent.

    ASE stores stress in calc.results either as a 6-component Voigt vector
    ``[σxx, σyy, σzz, σyz, σxz, σxy]`` or as a full 3x3 — handle both.
    """
    s = None
    if atoms.calc is not None and "stress" in atoms.calc.results:
        s = np.asarray(atoms.calc.results["stress"], dtype=np.float32)
    elif "stress" in atoms.info:
        s = np.asarray(atoms.info["stress"], dtype=np.float32)
    if s is None:
        return np.zeros((3, 3), dtype=np.float32)
    if s.shape == (6,):
        return np.array(
            [
                [s[0], s[5], s[4]],
                [s[5], s[1], s[3]],
                [s[4], s[3], s[2]],
            ],
            dtype=np.float32,
        )
    return s.reshape(3, 3).astype(np.float32)


def _atoms_to_frame(atoms, species_to_id: dict[str, int], cutoff: float) -> dict:
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    species = np.fromiter(
        (species_to_id[s] for s in atoms.get_chemical_symbols()),
        dtype=np.int32,
        count=len(atoms),
    )
    energy = _get_energy(atoms)
    forces = _get_forces(atoms, len(atoms))
    # matscipy.neighbour_list needs a non-singular cell even when pbc=False
    # (it uses the cell for spatial decomposition). Non-periodic frames may
    # ship with an all-zeros cell — replace with a bounding box around the
    # atoms expanded by the cutoff so it doesn't affect the neighbour list.
    if np.linalg.det(np.asarray(atoms.get_cell())) == 0.0:
        pmin = positions.min(axis=0) - cutoff
        pmax = positions.max(axis=0) + cutoff
        atoms = atoms.copy()
        atoms.set_cell(np.diag(pmax - pmin))
        atoms.set_pbc(False)
    senders, receivers, displacements = neighbour_list("ijD", atoms, cutoff)
    frame = {
        "species": species,
        "positions": positions,
        # ASE's `Cell` object doesn't satisfy numpy-2.0 zero-copy semantics;
        # asarray it first, then cast.
        "cell": np.asarray(atoms.get_cell()).astype(np.float32),
        "energy": energy,
        "forces": forces,
        "stress": _get_stress(atoms),
        "senders": senders.astype(np.int32),
        "receivers": receivers.astype(np.int32),
        "edge_vectors": displacements.astype(np.float32),
    }
    return frame


def load_extxyz(
    path: str | Path,
    species_to_id: dict[str, int],
    *,
    cutoff: float,
    max_frames: int | None = None,
) -> list[dict]:
    """Read an extxyz file and return one frame-dict per configuration."""
    index = ":" if max_frames is None else f":{max_frames}"
    atoms_list = ase_read(str(path), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [_atoms_to_frame(a, species_to_id, cutoff) for a in atoms_list]


def _collate(frames: list[dict]) -> tuple[dict, dict]:
    n_atoms_per = [len(f["species"]) for f in frames]
    offsets = np.cumsum([0, *n_atoms_per[:-1]])
    raw = {
        "species": np.concatenate([f["species"] for f in frames]),
        "positions": np.concatenate([f["positions"] for f in frames]).astype(np.float32),
        "cell": np.stack([f["cell"] for f in frames]).astype(np.float32),
        "nn_vecs": np.concatenate([f["edge_vectors"] for f in frames]),
        "inda": np.concatenate([f["senders"] + o for f, o in zip(frames, offsets, strict=True)]),
        "indb": np.concatenate([f["receivers"] + o for f, o in zip(frames, offsets, strict=True)]),
        "inde": np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(n_atoms_per)]),
        "nats": np.asarray(n_atoms_per, dtype=np.int32),
    }
    # Per-GRAPH scalars, one value per frame (same axis as `nats`). int8 on disk;
    # cast to int32 so the downstream gather and the /scale division behave.
    for _k in ("charge", "spin"):
        if _k in frames[0]:
            raw[_k] = np.asarray([f[_k] for f in frames], dtype=np.int32)
    # ``target_virial = stress * volume`` (ASE convention: stress in eV/Å³).
    # The model produces an extensive virial in eV via autograd; converting
    # the stress label to virial here means the loss compares same-unit
    # quantities and the per-graph volumes don't need to be carried through
    # the loss function.
    volumes = np.asarray(
        [abs(float(np.linalg.det(np.asarray(f["cell"])))) for f in frames],
        dtype=np.float32,
    )
    # Raw total energy stays float64 here: molecular |E| reaches 1e5-1e6 eV,
    # where an f32 ulp is 8-63 meV. `collate_for_pad` subtracts the species
    # offsets in float64 and casts only the small residual to float32.
    targets = {
        "target_E": np.asarray([f["energy"] for f in frames], dtype=np.float64),
        "target_F": np.concatenate([f["forces"] for f in frames]).astype(np.float32),
        "target_virial": np.stack([f["stress"] for f in frames]).astype(np.float32)
        * volumes[:, None, None],
    }
    return raw, targets


def make_padded_batch(
    frames: list[dict],
    *,
    base_atoms: int = 256,
    base_edges: int = 4096,
    energy_offsets: np.ndarray | list[float] | None = None,
    fixed_atoms: int | None = None,
    fixed_edges: int | None = None,
    fixed_graphs: int | None = None,
) -> dict:
    """Collate + pad a list of frames into one batch ready for the model.

    Returns a dict containing all chain inputs (``nn_vecs``, ``species``,
    ``inda``, ``indb``, ``inde``, ``nats``, ``mask``, ``dummy_graph_id``)
    plus the padded targets ``target_E`` and ``target_F``.

    If `energy_offsets` (a per-species array, e.g. from a model card) is
    given, ``target_E`` is shifted by subtracting the per-graph sum of
    species offsets. This brings the loss into the small interaction-
    energy range and avoids catastrophic cancellation at fp32: instead of
    fitting `model_E + sum_offsets ≈ DFT_E` (both ~10^4 eV) the loss
    becomes `model_E ≈ DFT_E - sum_offsets` (both ~few eV). Forces are not
    shifted — per-species constants have zero gradient w.r.t. positions.

    ``fixed_atoms``/``fixed_edges`` force the padded sizes to exactly those
    values (instead of the data-dependent ceil bucketing). This is required
    for multi-process data parallelism: every rank must emit identically-shaped
    sub-batches so the global sharded array is rectangular. The batch must fit
    (``real_atoms + 1 <= fixed_atoms`` and ``real_edges <= fixed_edges``);
    a ``ValueError`` is raised otherwise so an undersized bucket fails loudly
    rather than silently truncating.
    """
    raw, targets, n_frames = collate_for_pad(frames, energy_offsets=energy_offsets)
    if fixed_atoms is not None or fixed_edges is not None:
        n_real_atoms, n_real_edges = len(raw["species"]), len(raw["nn_vecs"])
        if fixed_atoms is None or fixed_edges is None:
            raise ValueError("fixed_atoms and fixed_edges must be set together")
        if n_real_atoms + 1 > fixed_atoms or n_real_edges > fixed_edges:
            raise ValueError(
                f"batch ({n_real_atoms} atoms, {n_real_edges} edges) exceeds fixed bucket "
                f"({fixed_atoms} atoms, {fixed_edges} edges); raise the bucket size"
            )
        n_atoms_padded, n_edges_padded = fixed_atoms, fixed_edges
    else:
        n_atoms_padded, n_edges_padded = choose_bucket_sizes(
            len(raw["species"]),
            len(raw["nn_vecs"]),
            base_atoms=base_atoms,
            base_edges=base_edges,
        )
    return pad_to(
        raw,
        targets,
        n_frames,
        n_atoms_padded=n_atoms_padded,
        n_edges_padded=n_edges_padded,
        fixed_graphs=fixed_graphs,
    )


def collate_for_pad(frames: list[dict], *, energy_offsets=None):
    """Collate frames + apply the energy-offset shift, WITHOUT padding.

    Returns ``(raw, targets, n_frames)`` — picklable numpy dicts. Real sizes are
    ``len(raw["species"])`` atoms / ``len(raw["nn_vecs"])`` edges. Split out of
    :func:`make_padded_batch` so the grid-bucketing data-parallel path can build
    in the worker but pad in the main process AFTER a cross-rank shape allgather.
    """
    raw, targets = _collate(frames)
    if energy_offsets is not None:
        offsets = np.asarray(energy_offsets, dtype=np.float64)
        per_frame_offset = np.asarray(
            [offsets[f["species"]].sum() for f in frames], dtype=np.float64
        )
        targets["target_E"] = targets["target_E"] - per_frame_offset
    # Cast after the subtraction so the large-magnitude cancellation runs in
    # float64 and the label carries f32 rounding of the residual, not of |E|.
    targets["target_E"] = targets["target_E"].astype(np.float32)
    return raw, targets, len(frames)


def pad_to(
    raw: dict,
    targets: dict,
    n_frames: int,
    *,
    n_atoms_padded: int,
    n_edges_padded: int,
    fixed_graphs: int | None = None,
) -> dict:
    """Pad a collated ``(raw, targets)`` to fixed sizes (the second half of
    :func:`make_padded_batch`, factored out for the grid-bucketing path).

    ``fixed_graphs`` pins the padded graph count (the balanced-binning path,
    where structs-per-batch varies): the extra slots become empty graphs
    (``nats=0``), keeping the graph axis a single compiled shape. Graphs are
    cheap (per-graph ops are O(graphs); the heavy work is per-atom/per-edge), so
    a generous fixed value over-pads negligibly. When ``None`` the graph axis is
    ``n_frames + 1`` (the fixed-count path, where it is already constant)."""
    n_graphs_padded = n_frames + 1 if fixed_graphs is None else fixed_graphs  # +1 dummy graph
    padded = pad_batch(
        raw,
        n_atoms_padded=n_atoms_padded,
        n_edges_padded=n_edges_padded,
        n_graphs_padded=n_graphs_padded,
    )
    # Pad targets in NumPy too (same reason as inside pad_batch).
    n_pad_atoms = n_atoms_padded - len(raw["species"])
    n_pad_graphs = n_graphs_padded - n_frames
    padded["target_E"] = np.concatenate(
        [targets["target_E"], np.zeros((n_pad_graphs,), np.float32)]
    )
    padded["target_F"] = np.concatenate(
        [targets["target_F"], np.zeros((n_pad_atoms, 3), np.float32)]
    )
    padded["target_virial"] = np.concatenate(
        [targets["target_virial"], np.zeros((n_pad_graphs, 3, 3), np.float32)],
        axis=0,
    )
    return padded


def batched_iter(
    frames: list[dict],
    *,
    batch_size: int,
    base_atoms: int = 256,
    base_edges: int = 4096,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
    energy_offsets: np.ndarray | list[float] | None = None,
    fixed_graphs: int | None = None,
) -> Iterator[dict]:
    """Shuffle frames and yield padded batches of ``batch_size`` graphs each.

    Pass `energy_offsets` (per-species, from a model card) to pre-shift
    ``target_E`` into the small interaction-energy range. See
    `make_padded_batch` for the rationale.

    ``fixed_graphs`` pins the padded graph axis to a constant (extra slots
    become empty ``nats=0`` graphs, masked out of every metric). Without it the
    last (remainder) batch has a different graph count than the full batches, so
    each distinct remainder triggers a fresh XLA recompile mid-run — under
    multi-process JAX that recompile stalls a rank past the coordination
    heartbeat and aborts the whole job. Pass ``batch_size + 1`` for a single
    compiled eval shape. (Atoms/edges are already constant via the base-grid
    bucketing.)
    """
    indices = np.arange(len(frames))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        if drop_last and len(batch_idx) < batch_size:
            continue
        batch_frames = [frames[j] for j in batch_idx]
        yield make_padded_batch(
            batch_frames,
            base_atoms=base_atoms,
            base_edges=base_edges,
            energy_offsets=energy_offsets,
            fixed_graphs=fixed_graphs,
        )


# 3BPA convenience: standard species map for the four-element molecule.
SPECIES_3BPA = {"H": 0, "C": 1, "N": 2, "O": 3}
