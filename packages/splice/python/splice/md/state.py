# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MD state: the per-atom arrays the CUDA kernels read and write.

Everything lives on the device as CuPy arrays and is handed to the kernels once,
as raw addresses. Nothing is copied per step.

Units are the kernels' own: **nm**, **ps**, **kcal/mol**, and mass in amu/4.184
(the factor that makes ``force / mass`` come out in nm/ps² for a kcal/mol
energy). :mod:`splice.md.forces` converts the model's Å / eV at the boundary.
"""

import cupy as cp
import numpy as np

KB = 1.987204259e-3  # Boltzmann constant, kcal/mol/K
MASS_UNIT = 4.184  # amu -> the kernels' mass unit (kcal/mol basis)


class MDState:
    """Positions, velocities, forces and the box for one orthorhombic system.

    Parameters
    ----------
    positions : (N, 3) array, nm
    box : (3,) array, nm — orthorhombic edge lengths
    masses : (N,) array, amu
    ddof : int
        Degrees of freedom removed when reporting temperature. 3 for a system
        whose centre-of-mass motion is not thermalised, which is our case.
    """

    def __init__(self, positions, box, masses, ddof=3):
        positions = np.ascontiguousarray(positions, np.float32)
        box = np.ascontiguousarray(box, np.float32)
        masses = np.ascontiguousarray(masses, np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {positions.shape}")
        if box.shape != (3,):
            raise ValueError(f"box must be (3,), got {box.shape}")
        if masses.shape != (positions.shape[0],):
            raise ValueError(f"masses must be (N,), got {masses.shape}")

        self.natom = positions.shape[0]
        self.ddof = ddof
        self.arrays = {
            "box": cp.asarray(box),
            "x": cp.asarray(positions),
            "v": cp.zeros((self.natom, 3), cp.float32),
            "f": cp.zeros((self.natom, 3), cp.float32),
            "masses": cp.asarray(masses / MASS_UNIT),
        }

    def addr(self, name):
        """Device address of one array, for handing to a kernel."""
        return self.arrays[name].data.ptr

    @property
    def box(self):
        return self.arrays["box"]

    @property
    def x(self):
        return self.arrays["x"]

    @property
    def v(self):
        return self.arrays["v"]

    @property
    def f(self):
        return self.arrays["f"]

    def wrap(self):
        """Fold positions back into [0, box). The neighbour list assumes it."""
        x, box = self.arrays["x"], self.arrays["box"]
        cp.subtract(x, box * cp.floor(x / box), out=x)

    def kinetic_energy(self, velocities=None):
        """Kinetic energy in kcal/mol."""
        v = self.arrays["v"] if velocities is None else velocities
        m = self.arrays["masses"]
        return 0.5 * float((m[:, None] * v * v).sum())

    def temperature(self, velocities=None):
        """Instantaneous temperature in K, and the kinetic energy it came from."""
        ke = self.kinetic_energy(velocities)
        return 2.0 * ke / ((3 * self.natom - self.ddof) * KB), ke

    def set_temperature(self, temperature, seed=0):
        """Draw velocities, remove the centre-of-mass drift, rescale to `temperature`."""
        rng = np.random.default_rng(seed)
        v = rng.uniform(-0.1, 0.1, (self.natom, 3)).astype(np.float32)
        v -= v.mean(axis=0)
        m = cp.asnumpy(self.arrays["masses"])
        ke = 0.5 * float((m[:, None] * v * v).sum())
        t = 2.0 * ke / ((3 * self.natom - self.ddof) * KB)
        v *= np.sqrt(temperature / t)
        self.arrays["v"].set(v)

    def set_velocities(self, velocities):
        """Set velocities directly (nm/ps), e.g. to reproduce a recorded run."""
        self.arrays["v"].set(np.ascontiguousarray(velocities, np.float32))

    def positions(self):
        """Positions as a host array, nm."""
        return cp.asnumpy(self.arrays["x"])
