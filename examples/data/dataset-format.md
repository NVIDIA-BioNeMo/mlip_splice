# Dataset formats

Two loaders. `extxyz` is the on-ramp; `packed` is the throughput path.

## extxyz

Any ASE-readable extended XYZ file. Per frame: per-atom `species`, `pos`,
`forces`, plus `energy` in the comment line (and `Lattice=` / `stress=` for
periodic data). Read fully into memory with a neighbour list per frame, so it is
suited to roughly 10^5 frames or fewer.

```bash
python packages/splice/scripts/train.py --arch ... --card ... --train data.xyz
```

## packed

A directory of contiguous `.npy` arrays, memory-mapped and shared read-only
across worker processes. Frames are assembled lazily per batch, so batch N+1 is
built while the GPU consumes batch N. Use this above ~10^6 frames.

| file | shape | dtype | meaning |
|---|---|---|---|
| `Z.npy` | `(sum_atoms,)` | int8 | atomic numbers, all frames concatenated |
| `pos.npy` | `(sum_atoms, 3)` | f32 | positions, Angstrom |
| `forces.npy` | `(sum_atoms, 3)` | f32 | forces, eV/Angstrom |
| `energy.npy` | `(N,)` | f64 | total energy per frame, eV, **raw** |
| `natoms.npy` | `(N,)` | i32 | atoms per frame |
| `offsets.npy` | `(N+1,)` | i64 | CSR index: frame `i` is `arr[off[i]:off[i+1]]` |
| `charge.npy` | `(N,)` | i8 | optional, metadata only |
| `spin.npy` | `(N,)` | i8 | optional, metadata only |
| `source_id.npy` | `(N,)` | i8 | optional, dataset/source index |
| `cell.npy` | `(N, 3, 3)` | f32 | **periodic data only** |
| `stress.npy` | `(N, 3, 3)` | f32 | **periodic data only**, eV/Angstrom^3 |
| `meta.json` | | | element list, source map, counts |

Energies are stored raw. The per-species offset subtraction happens at load
time from the model card, so one packed set can be reused with different cards.

Presence of `cell.npy` selects the periodic neighbour-list path; without it the
loader uses a brute-force non-periodic list.

## Converting

`packages/splice/scripts/pack_extxyz.py` converts a directory of extxyz files into the packed
layout, including the periodic arrays when the frames carry a cell.

```bash
python packages/splice/scripts/pack_extxyz.py --src <dir-of-xyz> --out <packed-dir> --val-frac 0.02
```

To pack from another source, write the arrays above directly; nothing else in
the loader is source-specific.
