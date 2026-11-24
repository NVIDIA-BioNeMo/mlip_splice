# Example dataset

`data/molecules.xyz` — 200 frames: 10 small molecules (H2O, H2S, NH3, PH3, CH4,
CH3F, CH3Cl, CH3OH, CH2O, C2H6), 20 rattled conformers each, covering
H C N O F P S Cl.

Energies (eV) and forces (eV/Angstrom) computed with PySCF at wB97X/def2-SVP.
Geometries are built from standard bond lengths and angles, then rattled with a
fixed seed. Regenerate with `packages/splice/scripts/make_example_data.py`.

`card.yaml` binds this dataset to a model: species map, 5.0 Angstrom cutoff,
average neighbour count, and per-species energy offsets fitted by least squares
on these frames.

Train the tiny model on it:

```bash
python packages/splice/scripts/train.py --arch examples/configs/smoke.yaml --card examples/card.yaml \
  --train examples/data/molecules.xyz --output-dir runs/example --n-epochs 5 \
  --search-budget-ms 250 --shortlist-budget-ms 2000
```

Note that `examples/configs/smoke.yaml` is deliberately tiny; it is for checking that the
pipeline runs, not for accuracy.
