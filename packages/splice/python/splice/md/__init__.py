# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molecular dynamics with a SPLICE potential.

A GPU-resident harness: the state, the cell neighbour list, the integrator and
the thermostat all stay in device memory, and the model's forces are written
straight into it over DLPack. Nothing crosses to the host inside the step loop
except the scalar energy when it is logged.

    from ase.io import read
    from splice.ase_calculator import load_model
    from splice.md import run_md

    chain, species, cutoff, n = load_model("arch.yaml", "model.eqx", card_path="card.yaml")
    trace, state = run_md(chain, n, species, read("water.xyz"), cutoff, steps=1000)

The CUDA extension is a separate build step from the Rust extension:

    make -C packages/splice/python/splice/md/_ops_src
"""

import os

# Leave device memory for the neighbour list and the state; JAX would otherwise
# preallocate most of the card at first use.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    from splice.md import _ops as _ops
except ImportError as exc:  # pragma: no cover - build guidance, not a code path
    raise ImportError(
        "splice.md requires the MD CUDA extension, which is built separately from "
        "the Rust extension:\n\n    make -C packages/splice/python/splice/md/_ops_src\n\n"
        "It needs nvcc and pybind11 (pip install pybind11)."
    ) from exc

from splice.md.driver import build_state, minimise, run_md
from splice.md.forces import make_force_fn, step_forces
from splice.md.integrator import Langevin, Verlet
from splice.md.neighbors import CellNeighborList, NeighborBackend
from splice.md.state import MDState

__all__ = [
    "CellNeighborList",
    "Langevin",
    "MDState",
    "NeighborBackend",
    "Verlet",
    "build_state",
    "make_force_fn",
    "minimise",
    "run_md",
    "step_forces",
]
