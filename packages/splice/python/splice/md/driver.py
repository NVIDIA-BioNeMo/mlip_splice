# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MD loop: minimise, equilibrate, then run.

This is a harness, not a production engine — it exists to show that a trained
potential holds up under real dynamics, at a real system size, at the speed the
fused kernels make possible. It is deliberately narrow: one orthorhombic
periodic box, one potential, no bonded terms, no constraints.
"""

import contextlib
import time

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from ase.data import atomic_masses, atomic_numbers

from splice.aot import aot_session
from splice.md.forces import K_E, LENGTH_SCALE, make_force_fn, step_forces
from splice.md.integrator import Langevin, Verlet
from splice.md.neighbors import CellNeighborList
from splice.md.state import KB, MDState


def build_state(atoms, species_map):
    """An :class:`MDState` and the model's species ids from an ASE ``Atoms``.

    The box must be orthorhombic — the neighbour list and the minimum-image
    convention here both assume it.
    """
    cell = np.asarray(atoms.cell.array, float)
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError(f"orthorhombic boxes only, got cell\n{cell}")
    box_A = np.diag(cell).astype(np.float32)
    if not np.all(atoms.pbc):
        raise ValueError("periodic boundaries in all three directions are required")

    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(species_map))
    if missing:
        raise ValueError(f"species not in the model card: {missing}")

    species = np.array([species_map[s] for s in symbols], np.int32)
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols], np.float32)
    state = MDState(
        atoms.get_positions().astype(np.float32) / LENGTH_SCALE,
        box_A / LENGTH_SCALE,
        masses,
    )
    return state, jnp.asarray(species), box_A


def minimise(state, evaluate, steps, max_step_nm, log_every=10):
    """Capped steepest descent, to relieve packing clashes before dynamics.

    The atom feeling the largest force moves exactly ``max_step_nm``; everyone
    else moves in proportion. Capping by the largest force rather than by a
    fixed learning rate keeps the first step bounded no matter how bad the
    starting clash is, and makes the step size independent of the force units.
    """
    print(f"[min] steepest descent, {steps} steps, <= {max_step_nm * LENGTH_SCALE:.3f} A/step")
    energy = None
    for it in range(1, steps + 1):
        energy = evaluate()
        gradient = state.f
        gmax = float(cp.sqrt((gradient * gradient).sum(axis=1)).max())
        if gmax == 0.0:
            break
        cp.add(state.x, gradient * (max_step_nm / gmax), out=state.arrays["x"])
        if it == 1 or it % log_every == 0 or it == steps:
            print(f"  min {it:4d}  PE={energy:12.2f} kcal/mol  |f|max={gmax:.3e}", flush=True)
    return energy


def run_md(
    chain,
    num_species,
    species_map,
    atoms,
    cutoff_A,
    *,
    steps=2000,
    equil=0,
    dt_fs=0.5,
    temperature=300.0,
    thermostat=False,
    friction=1.0,
    minimize=0,
    min_step_nm=0.01,
    skin_A=0.0,
    edge_budget=0,
    cell_width_A=0.0,
    head_id=None,
    charge=None,
    spin=None,
    log_every=200,
    seed=0,
    velocities=None,
    frame_every=0,
    on_frame=None,
    probe_every=0,
    aot=True,
    aot_cache_path=None,
    aot_identity_files=(),
):
    """Run NVE (or NVT) dynamics and return the energy trace.

    Parameters
    ----------
    chain, num_species, species_map : the loaded model and its card
    atoms : ase.Atoms — orthorhombic, fully periodic
    cutoff_A : float — neighbour cutoff, Å
    steps, equil : production and Langevin-equilibration step counts
    dt_fs : timestep in fs
    thermostat : run production under Langevin (NVT) rather than NVE
    skin_A : neighbour-list skin, Å. The cell grid and the candidate search
        widen by it; the emitted list is filtered to the cutoff either way, so
        results do not depend on it. Reuse across steps is not implemented, so
        for now a skin costs a little and saves nothing — see
        `CellNeighborList.update`.
    edge_budget : fixed padded edge count; 0 sizes it from the first build
    cell_width_A : cell width, Å. 0 uses cutoff + skin. Narrower cells scan more
        and emptier cells, and measured slower.
    velocities : optional (N, 3) nm/ps starting velocities, in place of a draw
    charge, spin : float | None
        Per-graph net charge and spin multiplicity. A chain with
        `charge_spin_embedding_mlp` reads these with `batch.get`, so leaving
        them None does NOT raise on a charge-aware model -- it silently
        evaluates the system as a neutral singlet. That is the right default
        for a neutral box and a wrong answer for anything else, with no log
        line to say which happened, so pass them whenever the system carries a
        net charge.

    aot : bool
        Compile inside a bounded Loom context. Uncapped autotuning on a cold
        cache was measured at 98 minutes against 6, and prints nothing while it
        runs, so it reads as a hang. Pass False to restore the unbounded search.
    aot_cache_path : str | None
        Persist realized recipes here, so a second run of the same system — or
        another box of the same size — skips lowering and tuning entirely.
    aot_identity_files : sequence of str
        Files whose contents key that cache; pass the arch and card so a changed
        model can never silently reuse stale recipes.
    frame_every : int
        Call ``on_frame`` every this many production steps. 0 disables it.
    probe_every : int
        Print live edge count, edges per atom and instantaneous temperature
        every this many steps. 0 disables it. Cheap (one device-to-host int and
        one reduction) and worth having on any long run: a trajectory that is
        about to blow up shows a rising edges-per-atom well before it overflows
        the budget, and the line distinguishes a diverged run from one that
        merely hit a wall-clock limit.
    on_frame : callable(step, state) | None
        Invoked with the live :class:`MDState`; read ``state.positions()`` (nm)
        from it. Provided so a trajectory can be written from ONE call to this
        function.

        Without it the only way to sample a trajectory is to call ``run_md``
        repeatedly for short segments, which is expensive in a way that is not
        obvious: the jitted force function is built HERE, per call, and
        ``jax.jit`` caches per function object -- so every segment recompiles
        even though the shapes are identical (``natom`` is fixed and
        ``edge_budget`` pins the edge axis). Each recompile re-runs autotune,
        whose workspace on a 35-42k atom system is ~64-74 GB against a ~26 GB
        steady footprint; repeating it per segment exhausts or fragments an
        80 GB card at unpredictable points.

    Returns
    -------
    list of (step, T_leapfrog, T_onstep, KE, PE, E_total, drift_meV_per_atom)
    """
    # Every op states its matmul precision explicitly, so the global default only
    # backstops stray matmuls — pin it so nothing silently drops to TF32.
    jax.config.update("jax_default_matmul_precision", "highest")

    with contextlib.ExitStack() as stack:
        if aot:
            stack.enter_context(
                aot_session(
                    identity_files=aot_identity_files,
                    extra=f"{len(atoms)}:{edge_budget}:{cutoff_A}",
                    cache_path=aot_cache_path,
                )
            )
        return _run_md(
            chain,
            num_species,
            species_map,
            atoms,
            cutoff_A,
            steps=steps,
            equil=equil,
            dt_fs=dt_fs,
            temperature=temperature,
            thermostat=thermostat,
            friction=friction,
            minimize=minimize,
            min_step_nm=min_step_nm,
            skin_A=skin_A,
            edge_budget=edge_budget,
            cell_width_A=cell_width_A,
            head_id=head_id,
            charge=charge,
            spin=spin,
            log_every=log_every,
            seed=seed,
            velocities=velocities,
            frame_every=frame_every,
            on_frame=on_frame,
            probe_every=probe_every,
        )


def _run_md(
    chain,
    num_species,
    species_map,
    atoms,
    cutoff_A,
    *,
    steps,
    equil,
    dt_fs,
    temperature,
    thermostat,
    friction,
    minimize,
    min_step_nm,
    skin_A,
    edge_budget,
    cell_width_A,
    head_id,
    charge,
    spin,
    log_every,
    seed,
    velocities,
    frame_every,
    on_frame,
    probe_every,
):
    """The loop itself. Split out so `run_md` owns the compile context and this
    stays a plain function -- the body is long enough without another level of
    indentation."""

    dt = dt_fs / 1000.0  # fs -> ps
    cutoff_nm = cutoff_A / LENGTH_SCALE
    skin_nm = skin_A / LENGTH_SCALE

    state, species, box_A = build_state(atoms, species_map)
    natom = state.natom
    if box_A.min() <= 2 * cutoff_A:
        raise ValueError(
            f"box {box_A} too small for the minimum-image convention at cutoff {cutoff_A} A"
        )

    neighbors = CellNeighborList(
        state,
        cutoff_nm,
        skin=skin_nm,
        cell_width=(cell_width_A / LENGTH_SCALE) if cell_width_A else None,
        edge_budget=int(edge_budget) or None,
    )

    force_fn = make_force_fn(chain, num_species, head_id=head_id, charge=charge, spin=spin)
    box_jax = jnp.asarray(cp.asnumpy(state.box))

    def evaluate():
        return step_forces(force_fn, state, neighbors, species, box_jax)

    print(
        f"[md] N={natom} box={box_A[0]:.3f} A cutoff={cutoff_A:.2f} A "
        f"skin={skin_A:.2f} A dt={dt_fs:g} fs",
        flush=True,
    )
    energy = evaluate()
    print(
        f"[md] edge budget={neighbors.edge_budget} ({neighbors.edge_budget / natom:.0f}/atom)",
        flush=True,
    )

    if minimize > 0:
        minimise(state, evaluate, minimize, min_step_nm)

    if velocities is not None:
        state.set_velocities(velocities)
    else:
        state.set_temperature(temperature, seed=seed)
    energy = evaluate()

    if equil > 0:
        equilibration = Verlet(state, dt, Langevin(state, dt, temperature, friction))
        print(f"[equil] Langevin, {equil} steps at {temperature} K", flush=True)
        for step in range(1, equil + 1):
            equilibration.step()
            energy = evaluate()
            if step % log_every == 0:
                print(
                    f"  equil {step:6d}  T={state.temperature()[0]:7.2f}  PE={energy:12.2f}",
                    flush=True,
                )
        del equilibration

    bath = Langevin(state, dt, temperature, friction) if thermostat else None
    integrator = Verlet(state, dt, bath)
    ndof = 3 * natom - state.ddof
    masses, velocity = state.arrays["masses"], state.arrays["v"]
    velocity_before = cp.empty_like(velocity)

    energy = evaluate()
    label = "nvt" if thermostat else "nve"
    print(f"[{label}] {steps} steps", flush=True)
    print(
        "  step    T_leap   T_onstep      KE          PE          Etot    drift(meV/atom)",
        flush=True,
    )

    trace, reference, t0 = [], None, time.time()
    for step in range(1, steps + 1):
        cp.copyto(velocity_before, velocity)
        integrator.step()
        energy = evaluate()
        if probe_every and step % probe_every == 0:
            live = neighbors.live_edges
            ke_p = 0.5 * float((masses[:, None] * velocity * velocity).sum())
            print(
                f"[probe] step {step} edges {live} ({live / natom:.1f}/atom) "
                f"budget {neighbors.edge_budget} T {2 * ke_p / (ndof * KB):.1f}K",
                flush=True,
            )
        if frame_every and step % frame_every == 0 and on_frame is not None:
            on_frame(step, state)
        if step % log_every == 0:
            # Leapfrog KE: the integrator carries velocities at half steps, so
            # the on-step kinetic energy is the cross term of the two halves.
            ke = 0.5 * float((masses[:, None] * velocity_before * velocity).sum())
            temp = 2.0 * ke / (ndof * KB)
            # Velocity-Verlet KE: v_n = v_{n+1/2} + (f_n/m)(dt/2). Same
            # trajectory, less finite-timestep bias on the stiff modes.
            v_on = velocity + state.f * (dt * 0.5) / masses[:, None]
            ke_on = 0.5 * float((masses[:, None] * v_on * v_on).sum())
            temp_on = 2.0 * ke_on / (ndof * KB)
            total = ke + energy
            if reference is None:
                reference = total
            drift = (total - reference) / natom / K_E * 1000.0
            print(
                f"{step:6d}{temp:9.2f}{temp_on:10.2f}{ke:12.1f}{energy:12.1f}"
                f"{total:13.2f}{drift:11.3f}",
                flush=True,
            )
            trace.append((step, temp, temp_on, ke, energy, total, drift))

    wall = time.time() - t0
    print(
        f"[done] {steps} steps in {wall:.2f}s = {wall / steps * 1000:.2f} ms/step "
        f"({wall / steps / natom * 1e6:.3f} us/atom/step)",
        flush=True,
    )
    return trace, state
