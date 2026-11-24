# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The whole loop: neighbour list -> model -> forces -> integrator.

The model here is untrained, which is the point: a random potential is still a
smooth one, so NVE must conserve energy just as well as a trained model would.
What is under test is the harness, not the physics.
"""

import jax
import numpy as np
import pytest
from ase import Atoms
from splice.config import build_chain
from splice.md import run_md
from splice.model_card import resolve_arch

from tests.conftest import CONFIGS

CARD = {
    "species": {"H": 0, "O": 1},
    "cutoff": 5.0,
    "avg_num_neighbors": 40.0,
    "offsets": {"default": "lstsq", "lstsq": [-16.0, -2045.0]},
    "outputs": ["E", "F"],
}


def _water_box(ncell=6, spacing=3.2, seed=0):
    """A periodic box of rattled water molecules on a simple cubic lattice."""
    rng = np.random.default_rng(seed)
    symbols, positions = [], []
    for i in range(ncell):
        for j in range(ncell):
            for k in range(ncell):
                oxygen = np.array([i, j, k]) * spacing + rng.normal(0, 0.15, 3)
                symbols.append("O")
                positions.append(oxygen)
                for offset in ((0.76, 0.59, 0.0), (-0.76, 0.59, 0.0)):
                    symbols.append("H")
                    positions.append(oxygen + np.array(offset))
    edge = ncell * spacing
    return Atoms(symbols=symbols, positions=np.array(positions), cell=np.diag([edge] * 3), pbc=True)


@pytest.fixture(scope="module")
def chain():
    return build_chain(resolve_arch(CONFIGS / "smoke.yaml", CARD), key=jax.random.key(0))


def test_nve_conserves_energy(chain):
    trace, _ = run_md(
        chain,
        2,
        CARD["species"],
        _water_box(),
        CARD["cutoff"],
        steps=300,
        dt_fs=0.25,
        temperature=100.0,
        log_every=50,
    )
    drift = np.array([row[-1] for row in trace])  # meV/atom
    assert np.abs(drift).max() < 0.5, f"NVE drift up to {np.abs(drift).max():.3f} meV/atom"


def test_thermostat_holds_the_setpoint(chain):
    target = 150.0
    trace, _ = run_md(
        chain,
        2,
        CARD["species"],
        _water_box(ncell=5),
        CARD["cutoff"],
        steps=400,
        dt_fs=0.25,
        temperature=target,
        thermostat=True,
        friction=5.0,
        log_every=100,
    )
    # Second half only: the first is the approach to the setpoint.
    reached = np.mean([row[1] for row in trace[len(trace) // 2 :]])
    assert reached == pytest.approx(target, rel=0.2), f"held {reached:.1f} K, wanted {target}"


def test_rebuild_interval_does_not_change_the_trajectory(chain):
    """A skin wide enough for the motion between rebuilds must give the same
    edges, and so the same trajectory, as rebuilding every step."""
    atoms = _water_box(ncell=4)
    velocities = np.zeros((len(atoms), 3), np.float32)
    common = dict(steps=60, dt_fs=0.25, temperature=100.0, log_every=20, velocities=velocities)

    _, every_step = run_md(chain, 2, CARD["species"], atoms, CARD["cutoff"], **common)
    _, with_skin = run_md(chain, 2, CARD["species"], atoms, CARD["cutoff"], skin_A=1.0, **common)
    np.testing.assert_allclose(every_step.positions(), with_skin.positions(), atol=1e-5)


def test_minimisation_lowers_the_energy(chain):
    """Steepest descent must not increase the potential energy."""
    import cupy as cp
    import jax.numpy as jnp
    from splice.md.driver import build_state, minimise
    from splice.md.forces import make_force_fn, step_forces
    from splice.md.neighbors import CellNeighborList

    atoms = _water_box(ncell=4)
    state, species, _ = build_state(atoms, CARD["species"])
    neighbors = CellNeighborList(state, CARD["cutoff"] / 10.0)
    force_fn = make_force_fn(chain, 2)
    box_jax = jnp.asarray(cp.asnumpy(state.box))

    def evaluate():
        return step_forces(force_fn, state, neighbors, species, box_jax)

    before = evaluate()
    after = minimise(state, evaluate, 20, 0.002)
    assert after < before, f"minimisation raised the energy: {before:.2f} -> {after:.2f}"
