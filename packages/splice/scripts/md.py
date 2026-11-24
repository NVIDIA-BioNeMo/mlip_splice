# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molecular dynamics driven by a trained SPLICE potential.

The forces come from a checkpoint (architecture YAML + card + ``.eqx`` weights);
the neighbour list, integrator and thermostat are the fused CUDA kernels in
``splice.md``. Everything stays on the GPU between steps.

Build the MD kernels once before using this::

    make -C packages/splice/python/splice/md/_ops_src

Example::

    python packages/splice/scripts/md.py --arch runs/example/arch.yaml --card runs/example/card.yaml \\
        --weights runs/example/ckpt_epoch_0005.eqx --positions water.xyz \\
        --steps 3000 --equil 2000 --dt 0.25 --temperature 300 --trace-out nve.dat
"""

import argparse
import os

import numpy as np
from ase.io import read, write
from splice.ase_calculator import load_model
from splice.md import run_md
from splice.md.forces import LENGTH_SCALE


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True, help="architecture YAML")
    p.add_argument("--weights", required=True, help=".eqx checkpoint")
    p.add_argument("--card", default=None, help="model card (species map, cutoff)")
    p.add_argument("--positions", required=True, help="starting structure, any ASE format")
    p.add_argument("--replicate", type=int, default=1, help="tile into a KxKxK supercell")

    p.add_argument("--steps", type=int, default=2000, help="production steps")
    p.add_argument("--equil", type=int, default=0, help="Langevin equilibration steps first")
    p.add_argument("--dt", type=float, default=0.5, help="timestep, fs")
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--thermostat", action="store_true", help="NVT production instead of NVE")
    p.add_argument("--friction", type=float, default=1.0, help="Langevin friction, 1/ps")
    p.add_argument("--minimize", type=int, default=0, help="steepest-descent steps before MD")
    p.add_argument("--min-step", type=float, default=0.01, help="max displacement per min step, nm")

    p.add_argument("--cutoff", type=float, default=None, help="override the card's cutoff, A")
    p.add_argument(
        "--skin",
        type=float,
        default=0.0,
        help="neighbour-list skin, A. Widens the search; the emitted list is "
        "filtered to the cutoff regardless, and is rebuilt every step",
    )
    p.add_argument(
        "--edge-budget",
        type=int,
        default=0,
        help="fixed padded edge count (0 sizes it from the first build)",
    )
    p.add_argument(
        "--cell-width",
        type=float,
        default=0.0,
        help="neighbour-list cell width, A (0 uses cutoff + skin)",
    )
    p.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "hybrid", "tf32"],
        help="fp32 (lossless); hybrid (TF32 dense linears, fp32 radial); "
        "tf32 (all TF32, a speed bracket only)",
    )
    p.add_argument("--head-id", type=int, default=None, help="head index for a multi-head model")
    p.add_argument(
        "--charge",
        type=float,
        default=None,
        help="per-graph net charge. A charge-aware model reads this with "
        "batch.get() and SILENTLY assumes a neutral singlet when it is absent, "
        "so omitting it on an ion gives plausible-looking wrong forces",
    )
    p.add_argument(
        "--spin",
        type=float,
        default=None,
        help="spin multiplicity. Defaults to the closed-shell choice for the "
        "given charge: 1 if (charge + sum_Z) is even, else 2",
    )

    p.add_argument("--seed", type=int, default=0, help="initial-velocity seed")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="print live edges/atom and temperature every N steps (0 = off). "
        "Rising edges/atom is the earliest sign of a trajectory going bad",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="write positions, velocities and step count here every "
        "--checkpoint-every steps, so a run outliving its wall clock can "
        "continue with --resume",
    )
    p.add_argument("--checkpoint-every", type=int, default=2000, help="steps between checkpoints")
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue from --checkpoint if it exists. Runs the REMAINING steps "
        "of --steps and skips equilibration and minimisation, which already "
        "happened; a no-op when the file is absent, so the same command line "
        "works for the first submission and every resubmission after it",
    )
    p.add_argument("--trace-out", default=None, help="write the energy trace here")
    p.add_argument("--out", default=None, help="write the final frame here")
    args = p.parse_args()

    chain, species_map, cutoff_A, num_species = load_model(
        args.arch, args.weights, args.precision, card_path=args.card
    )
    atoms = read(args.positions)
    if args.replicate > 1:
        atoms = atoms * (args.replicate,) * 3

    # Say which conditioning the model actually got. A chain containing
    # `charge_spin_embedding_mlp` reads charge/spin with `batch.get`, so the
    # absent case is not an error -- it is the neutral singlet, chosen silently.
    # Printing it means a wrong answer on an ion is visible in the log instead
    # of only in the numbers.
    charge, spin = args.charge, args.spin
    if charge is not None and spin is None:
        spin = 1 if (int(round(charge)) + int(atoms.get_atomic_numbers().sum())) % 2 == 0 else 2
    if charge is None:
        print(
            "[md] no --charge: a charge-aware model treats this as a NEUTRAL SINGLET",
            flush=True,
        )
    else:
        print(f"[md] charge={charge:+g} spin multiplicity={spin}", flush=True)

    # Resume, if asked and if there is something to resume from. Velocities must
    # be carried as well as positions: restarting from positions alone silently
    # redraws them from the Maxwell-Boltzmann distribution, which is a different
    # trajectory wearing the same name.
    steps_done, velocities = 0, None
    if args.resume and args.checkpoint and os.path.exists(args.checkpoint):
        ck = np.load(args.checkpoint)
        steps_done = int(ck["steps_done"])
        atoms.set_positions(ck["x"])
        velocities = ck["v"]
        print(f"[md] resumed at {steps_done:,}/{args.steps:,} steps", flush=True)
        if steps_done >= args.steps:
            print("[md] already complete; nothing to do", flush=True)
            return

    def _checkpoint(step, state):
        if not args.checkpoint or step % args.checkpoint_every:
            return
        tmp = args.checkpoint + ".tmp"
        # Hand savez an open FILE, not a path: given a path without a .npz
        # suffix it silently appends one, so the write lands somewhere other
        # than where the replace below looks and every checkpoint is lost.
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                x=state.positions() * LENGTH_SCALE,
                v=state.arrays["v"].get(),
                steps_done=steps_done + step,
            )
        os.replace(tmp, args.checkpoint)  # atomic: never leave a half-written state

    frame_every = args.checkpoint_every if args.checkpoint else 0

    trace, state = run_md(
        chain,
        num_species,
        species_map,
        atoms,
        args.cutoff or cutoff_A,
        dt_fs=args.dt,
        temperature=args.temperature,
        thermostat=args.thermostat,
        friction=args.friction,
        min_step_nm=args.min_step,
        skin_A=args.skin,
        edge_budget=args.edge_budget,
        cell_width_A=args.cell_width,
        head_id=args.head_id,
        charge=charge,
        spin=spin,
        log_every=args.log_every,
        seed=args.seed,
        steps=args.steps - steps_done,
        equil=0 if steps_done else args.equil,
        minimize=0 if steps_done else args.minimize,
        velocities=velocities,
        frame_every=frame_every,
        on_frame=_checkpoint if args.checkpoint else None,
        probe_every=args.probe_every,
    )

    if args.trace_out:
        np.savetxt(
            args.trace_out,
            np.array(trace),
            header="step T_leapfrog_K T_onstep_K KE_kcal PE_kcal Etot_kcal drift_meV_per_atom",
        )
        print(f"[trace] -> {args.trace_out}", flush=True)

    if args.out:
        atoms.set_positions(state.positions() * LENGTH_SCALE)
        write(args.out, atoms)
        print(f"[out] final frame -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
