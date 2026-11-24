# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the bundled example dataset.

Geometries are constructed here from standard bond lengths and angles (not taken
from any published structure database), then rattled with a fixed seed. Energies
and forces are computed fresh with PySCF (Apache-2.0) at wB97X/def2-SVP. Nothing
in the output is derived from a third-party dataset.

Usage:  python packages/splice/scripts/make_example_data.py --out examples/data/molecules.xyz
"""

import argparse
import math
import os

import numpy as np

BOHR = 0.529177210903
HARTREE = 27.211386245988


def _tetra(bond):
    """Four unit-ish directions of a regular tetrahedron, scaled to `bond`."""
    d = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
    return d / np.linalg.norm(d[0]) * bond


def _pyramidal(bond, angle_deg):
    """Three bonds from an apex, X-M-X angle `angle_deg` (NH3/PH3 shape)."""
    a = math.radians(angle_deg)
    # angle from the C3 axis such that the X-M-X angle is `a`
    cz = math.sqrt(max(0.0, 1.0 - (2.0 / 3.0) * (1.0 - math.cos(a))))
    r = math.sqrt(max(0.0, 1.0 - cz * cz))
    return (
        np.array(
            [
                [r * math.cos(t), r * math.sin(t), cz]
                for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
            ]
        )
        * bond
    )


def _bent(bond, angle_deg):
    a = math.radians(angle_deg)
    return np.array([[bond, 0.0, 0.0], [bond * math.cos(a), bond * math.sin(a), 0.0]])


def molecules():
    """(name, symbols, positions[A]) built from standard bond lengths/angles."""
    out = []
    out.append(("H2O", ["O", "H", "H"], np.vstack([[0, 0, 0], _bent(0.9572, 104.52)])))
    out.append(("H2S", ["S", "H", "H"], np.vstack([[0, 0, 0], _bent(1.336, 92.1)])))
    out.append(("NH3", ["N", "H", "H", "H"], np.vstack([[0, 0, 0], _pyramidal(1.012, 106.7)])))
    out.append(("PH3", ["P", "H", "H", "H"], np.vstack([[0, 0, 0], _pyramidal(1.420, 93.5)])))
    out.append(("CH4", ["C"] + ["H"] * 4, np.vstack([[0, 0, 0], _tetra(1.087)])))

    t = _tetra(1.095)
    out.append(
        (
            "CH3F",
            ["C", "F", "H", "H", "H"],
            np.vstack([[0, 0, 0], t[0] / np.linalg.norm(t[0]) * 1.383, t[1], t[2], t[3]]),
        )
    )
    out.append(
        (
            "CH3Cl",
            ["C", "Cl", "H", "H", "H"],
            np.vstack([[0, 0, 0], t[0] / np.linalg.norm(t[0]) * 1.785, t[1], t[2], t[3]]),
        )
    )

    # methanol: C at origin, O along the first tetrahedral direction, O-H staggered
    o = t[0] / np.linalg.norm(t[0]) * 1.427
    oh = o + (t[1] / np.linalg.norm(t[1])) * 0.956
    out.append(
        ("CH3OH", ["C", "O", "H", "H", "H", "H"], np.vstack([[0, 0, 0], o, t[1], t[2], t[3], oh]))
    )

    # formaldehyde: planar, H-C-H 121.9 deg
    a = math.radians(121.9 / 2)
    out.append(
        (
            "CH2O",
            ["C", "O", "H", "H"],
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1.208],
                    [1.116 * math.sin(a), 0, -1.116 * math.cos(a)],
                    [-1.116 * math.sin(a), 0, -1.116 * math.cos(a)],
                ]
            ),
        )
    )

    # ethane: staggered
    c2 = np.array([0, 0, 1.535])
    up, dn = _tetra(1.094), _tetra(1.094)
    out.append(
        (
            "C2H6",
            ["C", "C"] + ["H"] * 6,
            np.vstack([[0, 0, 0], c2, up[1], up[2], up[3], c2 - dn[1], c2 - dn[2], c2 - dn[3]]),
        )
    )
    return out


def single_point(symbols, pos, basis, xc):
    from pyscf import dft, gto

    mol = gto.M(
        atom=[(s, tuple(p)) for s, p in zip(symbols, pos, strict=True)],
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    e = mf.kernel()
    if not mf.converged:
        return None, None
    grad = mf.nuc_grad_method().kernel()  # dE/dR, Hartree/Bohr
    forces = -grad * (HARTREE / BOHR)  # eV/Angstrom
    return e * HARTREE, forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/data/molecules.xyz")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--rattle", type=float, default=0.05, help="stdev of the displacement, A")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--xc", default="wb97x")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ase import Atoms
    from ase.io import write

    rng = np.random.default_rng(a.seed)
    frames, failed = [], 0
    for name, syms, pos0 in molecules():
        for k in range(a.conformers):
            pos = pos0 + rng.normal(0.0, a.rattle, pos0.shape)
            e, f = single_point(syms, pos, a.basis, a.xc)
            if e is None:
                failed += 1
                continue
            at = Atoms(symbols=syms, positions=pos)
            at.info.update(
                energy=float(e),
                molecule=name,
                level=f"{a.xc}/{a.basis}",
                source="generated with PySCF",
            )
            at.arrays["forces"] = f
            frames.append(at)
        print(
            f"  {name}: {sum(1 for x in frames if x.info['molecule'] == name)} frames", flush=True
        )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write(a.out, frames, format="extxyz")
    print(f"wrote {len(frames)} frames to {a.out} ({failed} unconverged, skipped)")


if __name__ == "__main__":
    main()
