# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seam between the potential and the integrator.

The model works in Å and eV; the MD kernels work in nm, ps and kcal/mol. This
module owns that conversion and nothing else. Positions and the edge list cross
from CuPy to JAX and the forces cross back, both over DLPack — same device
memory, no host round trip.

Edge vectors are formed here rather than by the neighbour list because they must
be differentiable: forces come from the gradient with respect to them.
"""

import cupy as cp
import jax
import jax.numpy as jnp

from splice.forward import model_call

K_E = 23.060548  # eV -> kcal/mol
LENGTH_SCALE = 10.0  # nm -> Å
FORCE_SCALE = K_E * LENGTH_SCALE  # eV/Å -> kcal/mol/nm


def _minimum_image(d, box):
    return d - box * jnp.round(d / box)


def make_force_fn(chain, num_species, head_id=None, charge=None, spin=None):
    """A jitted ``(positions, box, senders, receivers, mask, species) -> (F, E)``.

    ``charge``/``spin`` are the per-graph total charge and spin multiplicity, for
    models that condition on them. They are OPTIONAL in the chain and read with
    ``batch.get``, so omitting them on a charge-aware model does not raise -- it
    silently evaluates every ion as a neutral singlet. Pass them for any system
    with a net charge; a solvated protein is essentially never neutral.

    Inputs are nm and the model's own species ids; forces come back already in
    the kernels' kcal/mol/nm. The unit conversion rides inside the jit rather
    than as a separate multiply afterwards: as its own kernel it cost 0.088 ms
    on an H100, more than the copy it feeds. The energy stays in eV, since it
    crosses to the host as a scalar anyway.
    """

    @jax.jit
    def forces(positions_nm, box_nm, senders, receivers, mask, species):
        natom = positions_nm.shape[0]
        positions = positions_nm * LENGTH_SCALE
        box = box_nm * LENGTH_SCALE
        edge_vectors = _minimum_image(positions[receivers] - positions[senders], box)
        # Padded slots hold (0, 0): sender == receiver, so the vector above is
        # exactly zero and d|r|/dr = r/|r| is 0/0 -- NaN, on 13% of the edge
        # tensor at a 1.15 headroom. `splice.padding` never hands the chain such
        # an edge: training pads nn_vecs with the UNIT vector (1, 0, 0) and
        # points both endpoints at a ghost atom, so every op sees a well-defined
        # direction. Match that convention here. Masking the gradient afterwards
        # is not enough -- `pos_grads` is added to F unmasked, and 0/0 has
        # already happened by then.
        edge_vectors = jnp.where(
            mask.astype(jnp.bool_)[:, None],
            edge_vectors,
            jnp.asarray([1.0, 0.0, 0.0], edge_vectors.dtype),
        )
        batch = {
            "nn_vecs": edge_vectors,
            "positions": positions,
            "cell": jnp.diag(box),
            "species": species.astype(jnp.int32),
            "inda": senders,
            "indb": receivers,
            "inde": jnp.zeros(natom, jnp.int32),
            "nats": jnp.asarray([natom], jnp.int32),
            "mask": mask.astype(jnp.bool_),
        }
        if head_id is not None:
            batch["head_id"] = jnp.asarray([head_id], jnp.int32)
        if charge is not None:
            batch["charge"] = jnp.asarray([charge], jnp.float32)
        if spin is not None:
            batch["spin"] = jnp.asarray([spin], jnp.float32)
        out = model_call(chain, batch, num_species=num_species)
        return out["F"] * FORCE_SCALE, out["E"][0]

    return forces


def step_forces(force_fn, state, neighbors, species, box_jax):
    """Rebuild the neighbour list, evaluate the model, write ``state.f``.

    Returns the potential energy in kcal/mol.
    """
    neighbors.update()
    senders, receivers, mask = neighbors.edges()

    forces, energy_model = force_fn(
        jax.dlpack.from_dlpack(state.x),
        box_jax,
        jax.dlpack.from_dlpack(senders),
        jax.dlpack.from_dlpack(receivers),
        jax.dlpack.from_dlpack(mask),
        species,
    )
    forces.block_until_ready()
    # The integrator was handed a raw pointer to state.f once; JAX allocates a
    # fresh buffer per call, so the values move into that fixed buffer. Both
    # live on the device — this is a D2D copy at full bandwidth (12 us for
    # 61 KiB on an H100), not a host round trip.
    cp.copyto(state.f, cp.from_dlpack(forces))
    return float(energy_model) * K_E
