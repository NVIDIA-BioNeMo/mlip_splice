# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert extended-XYZ files into the packed, memmap-friendly training format.

Reads one or more ``.xyz`` / ``.extxyz`` files (a single file, or a directory of
them) and writes the array layout documented in ``examples/data/dataset-format.md``:

  Z.npy (sum_atoms,) int8 | pos.npy (sum_atoms,3) f32 | forces.npy f32
  energy.npy (N,) f64 (raw total E) | natoms.npy (N,) i32 | offsets.npy (N+1,) i64
  charge.npy / spin.npy / source_id.npy (N,) i8 | meta.json
  cell.npy (N,3,3) f32 | stress.npy (N,3,3) f32     [periodic frames only]

Energies are written raw; the per-species offset subtraction happens at load
time from the model card, so one packed set can be reused with several cards.
Stress is read exactly as ASE parses the extxyz ``stress=`` field (3x3,
eV/Angstrom^3) so the runtime sees an identical tensor.

The train/val split is by FILE, not by frame, so frames from one trajectory
cannot straddle the split.

Usage:
  python packages/splice/scripts/pack_extxyz.py --src <dir-or-file> --out <packed_dir>
  python packages/splice/scripts/pack_extxyz.py --src <dir> --out <train_dir> \
      --val-out <val_dir> --val-frac 0.02
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
from ase.io import read as ase_read


def _list_files(src):
    """One file, or every .xyz/.extxyz in a directory.

    Uses scandir rather than glob: a large dataset directory can hold >10^5
    files, which blows the shell argument list.
    """
    if os.path.isfile(src):
        return [src]
    return sorted(e.path for e in os.scandir(src) if e.name.endswith((".extxyz", ".xyz")))


def _energy(atoms):
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if atoms.calc is not None and "energy" in atoms.calc.results:
        return float(atoms.calc.results["energy"])
    return float("nan")


def _forces(atoms, n):
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return np.asarray(atoms.calc.results["forces"], dtype=np.float32).reshape(n, 3)
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=np.float32).reshape(n, 3)
    return np.zeros((n, 3), dtype=np.float32)


def _stress_3x3(atoms):
    """ASE gives stress as 6-Voigt [xx,yy,zz,yz,xz,xy] or 3x3; return symmetric 3x3."""
    s = None
    if atoms.calc is not None and "stress" in atoms.calc.results:
        s = np.asarray(atoms.calc.results["stress"], dtype=np.float32)
    elif "stress" in atoms.info:
        s = np.asarray(atoms.info["stress"], dtype=np.float32)
    if s is None:
        return np.zeros((3, 3), dtype=np.float32)
    s = s.reshape(-1)
    if s.size == 6:
        return np.array(
            [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]], dtype=np.float32
        )
    return s.reshape(3, 3).astype(np.float32)


def _process_files(arg):
    paths, _idx = arg
    Zs, POS, FRC, NAT, ENE, CEL, STR = [], [], [], [], [], [], []
    nbad = 0
    for p in paths:
        try:
            frames = ase_read(p, index=":")
        except Exception:
            nbad += 1
            continue
        if not isinstance(frames, list):
            frames = [frames]
        for a in frames:
            n = len(a)
            e = _energy(a)
            if not np.isfinite(e) or n == 0:
                nbad += 1
                continue
            Zs.append(a.get_atomic_numbers().astype(np.int8))
            POS.append(np.asarray(a.get_positions(), dtype=np.float32).reshape(n, 3))
            FRC.append(_forces(a, n))
            CEL.append(np.asarray(a.cell.array, dtype=np.float32).reshape(3, 3))
            STR.append(_stress_3x3(a))
            NAT.append(n)
            ENE.append(e)
    return (
        np.concatenate(Zs) if Zs else np.zeros(0, np.int8),
        np.concatenate(POS) if POS else np.zeros((0, 3), np.float32),
        np.concatenate(FRC) if FRC else np.zeros((0, 3), np.float32),
        np.asarray(NAT, np.int32),
        np.asarray(ENE, np.float64),
        np.asarray(CEL, np.float32).reshape(-1, 3, 3),
        np.asarray(STR, np.float32).reshape(-1, 3, 3),
        nbad,
    )


def _write_pack(out, Z, POS, FRC, NAT, ENE, CEL, STR, meta_extra):
    SRC = np.zeros(len(NAT), np.int8)
    CHG = np.zeros(len(NAT), np.int8)
    SPN = np.zeros(len(NAT), np.int8)
    offsets = np.zeros(len(NAT) + 1, dtype=np.int64)
    np.cumsum(NAT, out=offsets[1:])
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Z.npy"), Z)
    np.save(os.path.join(out, "pos.npy"), POS)
    np.save(os.path.join(out, "forces.npy"), FRC)
    np.save(os.path.join(out, "energy.npy"), ENE)
    np.save(os.path.join(out, "natoms.npy"), NAT)
    np.save(os.path.join(out, "offsets.npy"), offsets)
    np.save(os.path.join(out, "cell.npy"), CEL)
    np.save(os.path.join(out, "stress.npy"), STR)
    np.save(os.path.join(out, "charge.npy"), CHG)
    np.save(os.path.join(out, "spin.npy"), SPN)
    np.save(os.path.join(out, "source_id.npy"), SRC)
    meta = {
        "n_struct": len(NAT),
        "n_atoms": len(Z),
        "mean_atoms": float(NAT.mean()) if len(NAT) else 0.0,
        "max_atoms": int(NAT.max()) if len(NAT) else 0,
        "elements": sorted(int(z) for z in np.unique(Z)) if len(Z) else [],
        "periodic": True,
        "source_map": {meta_extra.get("source_name", "default"): 0},
        "dtypes": {
            "Z": "int8",
            "pos": "f32",
            "forces": "f32",
            "energy": "f64",
            "cell": "f32",
            "stress": "f32",
        },
        **meta_extra,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    gb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e9
    print(
        f"[pack] wrote {meta['n_struct']} structs / {meta['n_atoms']} atoms -> {out} ({gb:.2f} GB)"
    )
    print(f"[pack] elements({len(meta['elements'])})={meta['elements']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a .xyz/.extxyz file, or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None, help="optional separate val pack")
    ap.add_argument("--val-frac", type=float, default=0.0, help="fraction of FILES -> val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit-files", type=int, default=0, help=">0: first N files (smoke)")
    args = ap.parse_args()

    files = _list_files(args.src)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    print(f"[pack] {len(files)} extxyz files under {args.src}", flush=True)

    # split into train/val by file (no same-trajectory leakage)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files))
    n_val = int(args.val_frac * len(files)) if args.val_out else 0
    val_files = [files[i] for i in perm[:n_val]]
    train_files = [files[i] for i in perm[n_val:]]
    print(f"[pack] train files={len(train_files)} val files={len(val_files)}", flush=True)

    def _pack(file_list, out, label):
        if not file_list:
            return
        # chunk files across workers (each chunk ~equal file count)
        nchunks = max(args.workers * 4, 1)
        chunks = [file_list[i::nchunks] for i in range(nchunks)]
        jobs = [(c, i) for i, c in enumerate(chunks) if c]
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            results = pool.map(_process_files, jobs)
        nbad = sum(r[7] for r in results)
        Z = np.concatenate([r[0] for r in results])
        POS = np.concatenate([r[1] for r in results])
        FRC = np.concatenate([r[2] for r in results])
        NAT = np.concatenate([r[3] for r in results])
        ENE = np.concatenate([r[4] for r in results])
        CEL = np.concatenate([r[5] for r in results])
        STR = np.concatenate([r[6] for r in results])
        print(
            f"[pack:{label}] read {len(NAT)} frames in {time.time() - t0:.1f}s (skipped {nbad})",
            flush=True,
        )
        _write_pack(
            out,
            Z,
            POS,
            FRC,
            NAT,
            ENE,
            CEL,
            STR,
            {"seed": args.seed, "split": label, "n_files": len(file_list)},
        )

    _pack(train_files, args.out, "train")
    if args.val_out and val_files:
        _pack(val_files, args.val_out, "val")
    print("[pack] DONE", flush=True)


if __name__ == "__main__":
    main()
