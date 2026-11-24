# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a model card from a dataset.

A card binds a model to a dataset: the species map, the neighbour-list cutoff,
the average neighbour count (which sets the message normalisation), and the
per-species energy offsets subtracted from the training targets. The card is
dataset-side and does not name an architecture -- `resolve_arch(arch, card)`
applies its values to whichever architecture YAML you train.

Run once per dataset and commit the result.

Usage:
  python packages/splice/scripts/build_model_card.py --data examples/data/molecules.xyz \
      --cutoff 5.0 --out examples/card.yaml
"""

import argparse
from pathlib import Path

from ase.io import read
from splice.data import load_extxyz
from splice.model_card import compute_avg_num_neighbors, fit_species_offsets, save_card


def species_map(paths):
    """Species -> contiguous id, ordered by atomic number for stability."""
    from ase.data import atomic_numbers

    seen = set()
    for p in paths:
        for atoms in read(p, ":"):
            seen.update(atoms.get_chemical_symbols())
    return {s: i for i, s in enumerate(sorted(seen, key=lambda s: atomic_numbers[s]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="extxyz file(s) to fit on")
    ap.add_argument("--out", required=True, help="output card YAML")
    ap.add_argument("--cutoff", type=float, required=True, help="neighbour-list cutoff, Angstrom")
    ap.add_argument("--outputs", nargs="+", default=["E", "F"])
    ap.add_argument("--note", default=None, help="free-text note recorded in the card")
    a = ap.parse_args()

    species = species_map(a.data)
    print(f"species ({len(species)}): {species}")

    frames = []
    for p in a.data:
        frames += load_extxyz(p, species, cutoff=a.cutoff)
    print(f"loaded {len(frames)} frames")

    avg_nn = compute_avg_num_neighbors(frames)
    fitted = fit_species_offsets(frames, num_species=len(species))
    print(f"avg_num_neighbors @ cutoff={a.cutoff}: {avg_nn:.4f}")
    print(f"lstsq offsets: {[round(float(x), 4) for x in fitted]}")

    notes = {"dataset": a.note} if a.note else None
    save_card(
        Path(a.out),
        species=species,
        cutoff=a.cutoff,
        avg_num_neighbors=avg_nn,
        offsets_lstsq=fitted.tolist(),
        offsets_default="lstsq",
        outputs=a.outputs,
        notes=notes,
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
