# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The integrator and thermostat, against cases with a closed-form answer."""

import cupy as cp
import numpy as np
import pytest
from splice.md.integrator import Langevin, Verlet
from splice.md.state import KB, MASS_UNIT, MDState


def _state(natom=64, box_nm=4.0, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.5, box_nm - 0.5, (natom, 3)).astype(np.float32)
    masses = np.full(natom, 12.0, np.float32)
    return MDState(positions, np.full(3, box_nm, np.float32), masses)


def test_free_particles_move_ballistically():
    """With no force, x(t) = x0 + v t exactly, to float32."""
    state = _state()
    velocities = np.full((state.natom, 3), 0.01, np.float32)  # nm/ps
    state.set_velocities(velocities)
    start = state.positions()

    dt, steps = 0.001, 50
    integrator = Verlet(state, dt)
    for _ in range(steps):
        integrator.step()

    # Compare the displacement, not the absolute position: at x ~ 2 nm a
    # float32 ulp is 2.4e-7 nm, which is a large share of a 5e-4 nm move.
    np.testing.assert_allclose(state.positions() - start, velocities * dt * steps, rtol=2e-3)


def test_constant_force_gives_uniform_acceleration():
    """Under a constant force the velocity is exactly f/m * t."""
    state = _state()
    natom = state.natom
    force = 1.0  # kcal/mol/nm
    state.f[:] = cp.asarray(np.full((natom, 3), force, np.float32))

    dt, steps = 0.0005, 40
    integrator = Verlet(state, dt)
    for _ in range(steps):
        integrator.step()

    mass = 12.0 / MASS_UNIT
    expected = force / mass * dt * steps
    np.testing.assert_allclose(cp.asnumpy(state.v), expected, rtol=1e-4)


def test_nve_conserves_a_harmonic_oscillator():
    """A potential the harness does not know about, integrated by the kernel:
    total energy must stay flat over many periods."""
    natom = 256
    rng = np.random.default_rng(0)
    centre = np.full(3, 2.0, np.float32)
    positions = (centre + rng.normal(0, 0.1, (natom, 3))).astype(np.float32)
    state = MDState(positions, np.full(3, 4.0, np.float32), np.full(natom, 12.0, np.float32))
    state.set_temperature(300.0)

    k = 200.0  # kcal/mol/nm^2
    origin = cp.asarray(np.tile(centre, (natom, 1)))

    def energy_and_force():
        displacement = state.x - origin
        state.f[:] = -k * displacement
        return 0.5 * k * float((displacement * displacement).sum())

    dt = 0.0002
    integrator = Verlet(state, dt)
    potential = energy_and_force()
    totals = []
    for step in range(2000):
        integrator.step()
        potential = energy_and_force()
        if step % 50 == 0:
            totals.append(potential + state.kinetic_energy())

    totals = np.array(totals)
    drift = abs(totals[-1] - totals[0]) / abs(totals[0])
    assert drift < 2e-3, f"NVE drift {drift:.2e} over 2000 steps"


def test_langevin_reaches_the_target_temperature():
    """Started cold, the thermostat must warm the system to its setpoint."""
    natom = 4096
    state = _state(natom=natom, box_nm=8.0, seed=2)
    state.set_temperature(10.0)
    assert state.temperature()[0] == pytest.approx(10.0, rel=1e-3)

    target, dt = 300.0, 0.001
    integrator = Verlet(state, dt, Langevin(state, dt, target, friction=20.0))
    for _ in range(500):
        integrator.step()

    reached = state.temperature()[0]
    # 4096 free particles: the relative spread of the instantaneous temperature
    # is sqrt(2/ndof) ~ 1.3%, so a few percent is the honest tolerance.
    assert reached == pytest.approx(target, rel=0.05), f"reached {reached:.1f} K"


def test_thermostat_is_seeded_reproducibly():
    results = []
    for _ in range(2):
        state = _state(natom=512, seed=4)
        state.set_temperature(50.0)
        dt = 0.001
        integrator = Verlet(state, dt, Langevin(state, dt, 300.0, friction=10.0))
        for _ in range(20):
            integrator.step()
        results.append(state.positions())
    np.testing.assert_array_equal(results[0], results[1])


def test_state_temperature_matches_its_definition():
    state = _state(natom=1024, seed=6)
    state.set_temperature(275.0)
    velocities = cp.asnumpy(state.v)
    masses = cp.asnumpy(state.arrays["masses"])
    ke = 0.5 * float((masses[:, None] * velocities * velocities).sum())
    ndof = 3 * state.natom - state.ddof
    assert state.temperature()[0] == pytest.approx(2.0 * ke / (ndof * KB), rel=1e-6)
