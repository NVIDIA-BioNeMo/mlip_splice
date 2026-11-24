# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Energy targets: species offsets cancel in float64 before the f32 cast."""

import numpy as np
from splice.data import collate_for_pad, make_padded_batch

# Molecular-scale per-species references (eV), H/C/N/O.
_OFFSETS = np.array([-13.6, -1029.1, -1485.3, -2043.7], np.float64)


def _frame(species, energy):
    n = len(species)
    return {
        "species": np.asarray(species, np.int32),
        "positions": np.arange(3 * n, dtype=np.float32).reshape(n, 3),
        "cell": np.eye(3, dtype=np.float32) * 50.0,
        "energy": energy,
        "forces": np.zeros((n, 3), np.float32),
        "stress": np.zeros((3, 3), np.float32),
        "senders": np.array([0, 1], np.int32),
        "receivers": np.array([1, 0], np.int32),
        "edge_vectors": np.array([[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]], np.float32),
    }


def test_offset_subtraction_keeps_large_energies_in_float64():
    # 40 O + 20 C + 30 H + 10 N: Σ offsets ≈ -1.2e5 eV, where an f32 ulp is 7.8 meV.
    species = [3] * 40 + [1] * 20 + [0] * 30 + [2] * 10
    residual = -3.14159265
    total = float(_OFFSETS[species].sum()) + residual
    _, targets, _ = collate_for_pad([_frame(species, total)], energy_offsets=_OFFSETS)
    expected = total - _OFFSETS[species].sum()
    target = targets["target_E"]
    assert target.dtype == np.float32
    tol = abs(np.spacing(np.float32(expected))) / 2
    assert abs(float(target[0]) - expected) <= tol
    # Cancelling on f32-rounded operands misses by far more than the residual's rounding.
    f32_path = np.float32(total) - np.float32(_OFFSETS[species].sum())
    assert abs(float(f32_path) - expected) > 100 * tol


def test_padded_target_is_float32_with_and_without_offsets():
    frames = [_frame([0, 1, 3], -3100.25), _frame([3, 3, 2], -5572.5)]
    for offsets in (None, _OFFSETS):
        batch = make_padded_batch(frames, base_atoms=8, base_edges=8, energy_offsets=offsets)
        assert batch["target_E"].dtype == np.float32
