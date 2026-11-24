# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Velocity-Verlet integration and the Langevin thermostat, on the GPU.

One step is: kick the velocities by the current forces, drift half a timestep,
optionally apply the thermostat, drift the other half. The caller refreshes
``state.f`` between steps; the integrator never computes a force itself.

Timesteps are in ps and temperatures in K, matching :class:`~splice.md.state.MDState`.
"""

from splice.md import _ops


class Langevin:
    """Ornstein-Uhlenbeck velocity update, used as the Verlet velocity-scaling stage.

    Parameters
    ----------
    state : MDState
    dt : float
        Timestep, ps. Must match the integrator's.
    temperature : float
        Target temperature, K.
    friction : float
        Collision frequency, 1/ps. Larger couples harder to the bath and damps
        the dynamics more.
    """

    def __init__(self, state, dt, temperature, friction):
        self.state = state
        self.temperature = float(temperature)
        self.friction = float(friction)
        self._ws = _ops.langevin.workspace()
        _ops.langevin.setup(
            self._ws, float(dt), self.temperature, self.friction, state.natom, state.addr("masses")
        )
        _ops.langevin.setinterface(self._ws, state.addr("x"), state.addr("v"), state.addr("f"))

    @property
    def workspace(self):
        return self._ws

    def step(self):
        _ops.langevin.process(self._ws)

    def __del__(self):
        ws = getattr(self, "_ws", None)
        if ws is not None:
            _ops.langevin.release(ws)
            self._ws = None


class Verlet:
    """Velocity-Verlet integrator.

    Parameters
    ----------
    state : MDState
    dt : float
        Timestep, ps.
    thermostat : Langevin | None
        Attached between the two half drifts. ``None`` integrates NVE.
    """

    def __init__(self, state, dt, thermostat=None):
        self.state = state
        self.dt = float(dt)
        self._ws = _ops.verlet.workspace()
        _ops.verlet.setinterface(
            self._ws,
            state.natom,
            self.dt,
            state.addr("box"),
            state.addr("masses"),
            state.addr("x"),
            state.addr("v"),
            state.addr("f"),
        )
        # Held so the thermostat outlives the raw pointer the integrator keeps.
        self._thermostat = thermostat
        if thermostat is not None:
            _ops.verlet.vs(self._ws, thermostat.workspace)

    def step(self):
        """Advance one timestep using the forces currently in ``state.f``."""
        _ops.verlet.process(self._ws)

    def __del__(self):
        ws = getattr(self, "_ws", None)
        if ws is not None:
            _ops.verlet.release(ws)
            self._ws = None
