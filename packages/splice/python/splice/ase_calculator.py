# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASE Calculator for a SPLICE potential (E / forces / stress).

This is the enabler for the MACE-MH-1 benchmark-suite reproduction: it wraps a
trained SPLICE checkpoint (resolved-config YAML + ``.eqx`` weights) behind the
standard ``ase.calculators.Calculator`` interface so that every external
benchmark harness (ML-PEG / janus-core / our own scripts) can drive the model.

Design goals:
  * torch-free and cupy-free — depends only on splice, jax, ase, matscipy.
  * periodic *and* molecular (matscipy neighbour list; a bounding box is
    synthesised for open systems so the same code path serves both).
  * E (eV), forces (eV/Å) and stress (eV/Å³, Voigt) straight from
    ``splice.forward.model_call`` (the ``virial`` it returns is already the
    ASE stress·V convention, so stress = virial / V).
  * optional additive **D3(BJ)** dispersion via a pluggable second calculator
    (``d3=...``) — kept separate because the D3 head depends on which model
    head we benchmark (OMAT/PBE → +D3; OMOL/ωB97M-VV10 → no D3).

JAX recompiles per unique (n_atoms, n_edges) shape. For the many-structure
single-point benchmarks pass ``pad_to`` buckets (atoms, edges) to amortise the
compile cache; left unset, every distinct shape compiles once (fine for the
small validation/diatomic sets).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import inspect
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from ase.calculators.calculator import Calculator, all_changes
from ase.data import chemical_symbols
from ase.stress import full_3x3_to_voigt_6_stress
from matscipy.neighbours import neighbour_list

from splice.config import build_chain, load_config
from splice.forward import model_call
from splice.model_card import load_card, resolve_arch
from splice.registry import get_op

_RADIAL_FP32_OPS = {"radial_mlp", "radial_mlp_ln"}


def _apply_precision_mode(steps, mode):
    """Override per-op matmul precision in the resolved config (in place), by
    setting an EXPLICIT entry from the common precision vocabulary — never a
    silent global default.

    ``fp32`` leaves the config's stated precision as-is; ``hybrid`` drops dense
    linears to ``tf32`` but keeps the radial MLP front-end at its stated (fp32)
    precision; ``tf32`` drops every precision-bearing op to ``tf32`` (speed
    bracket, inaccurate).
    """
    if mode == "fp32":
        return
    for step in steps:
        op = step["op"]
        if mode == "hybrid" and op in _RADIAL_FP32_OPS:
            continue
        sig = inspect.signature(get_op(op).__init__).parameters
        if "precision" in sig:
            step["params"]["precision"] = "tf32"


def load_model(config_path, weights_path, precision="fp32", card_path=None):
    """Torch-free load. Returns (chain, species_map{sym:id}, cutoff_A, num_species).

    ``config_path`` is an architecture YAML. The species map and cutoff come
    from ``card_path`` (the usual case: a training run writes ``arch.yaml`` and
    ``card.yaml`` side by side). A self-contained config that already carries
    its own ``species:``/``cutoff:`` keys is also accepted, and then
    ``card_path`` may be omitted.
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    if card_path is not None:
        card = load_card(card_path)
        species_map = card["species"]
        cutoff_A = float(card["cutoff"])
        cfg = resolve_arch(config_path, card)
    elif isinstance(raw, dict) and "species" in raw:
        species_map = raw["species"]
        cutoff_A = float(raw["cutoff"])
        cfg = load_config(config_path)
    else:
        raise ValueError(
            f"{config_path} carries no `species:`/`cutoff:`; pass card_path=<card.yaml>"
        )
    num_species = len(species_map)
    _apply_precision_mode(cfg, precision)
    chain = build_chain(cfg, key=jax.random.key(0))
    chain = eqx.tree_deserialise_leaves(weights_path, chain)
    return chain, species_map, cutoff_A, num_species


def _safe_cell(atoms, cutoff):
    """Cell + pbc usable by matscipy for any system.

    matscipy needs a non-singular cell even for open boundaries (it only bins
    against it). For a fully non-periodic system with no cell we synthesise a
    bounding box padded by 2·cutoff; for partially periodic / already-celled
    systems we pass the real cell through.
    """
    pbc = np.asarray(atoms.pbc, bool)
    cell = np.asarray(atoms.cell.array, float)
    if not pbc.any() and np.abs(cell).sum() < 1e-8:
        pos = atoms.get_positions()
        extent = pos.max(0) - pos.min(0) + 2.0 * cutoff + 1.0
        cell = np.diag(np.maximum(extent, 2.0 * cutoff + 1.0))
    return cell, pbc


class SpliceCalculator(Calculator):
    """ASE calculator backed by a SPLICE checkpoint.

    Parameters
    ----------
    config_path, weights_path : str
        Architecture YAML and ``.eqx`` weights.
    card_path : str | None
        Model card supplying the species map and cutoff. Required unless
        ``config_path`` is a self-contained config that carries them itself.
    precision : {"fp32","hybrid","tf32"}
    head_id : int | None
        Per-graph dataset/fidelity head index for a multi-head finetuned model.
        Left ``None`` for single-head checkpoints.
    d3 : ase Calculator | None
        Optional additive dispersion calculator (e.g. a D3(BJ) calculator). Its
        energy/forces/stress are added to the model's. Use ONLY with a
        dispersion-free head (OMAT/PBE), never with the OMOL/ωB97M-VV10 head.
    pad_to : (n_atoms, n_edges) | None
        Pad every batch to these shapes to keep the JAX compile cache to one
        specialisation across differently-sized structures.

    Per-structure charge and spin
    -----------------------------
    A chain containing ``charge_spin_embedding_mlp`` conditions on the total
    charge and spin multiplicity of the structure. This calculator reads them
    from ``atoms.info`` on every call:

    ``charge``        net charge, e. Falls back to ``total_charge``, then 0.
    ``multiplicity``  spin multiplicity 2S+1 (1 = singlet). Falls back to
                      ``spin``, then 1.

    ⚠️ ``charge_spin_embedding_mlp`` reads both with ``batch.get``, so a missing
    value does NOT raise on a charge-aware model -- it silently evaluates the
    structure as a NEUTRAL SINGLET and returns confident numbers. That default
    is right for a neutral closed-shell molecule and wrong for anything else,
    which is why the keys are read here rather than left to the caller, and why
    a charge-aware model logs once what convention it is using.

    ``multiplicity`` is 2S+1, NOT ``2S`` and not ``n_up - n_down``. A caller
    holding ``2S`` must add one. Both values are passed as traced arrays, so
    changing either never triggers a recompile.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        config_path,
        weights_path,
        card_path=None,
        precision="fp32",
        head_id=None,
        d3=None,
        pad_to=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Every op states its matmul precision explicitly, so the global default
        # only backstops stray unhandled matmuls — pin it to `highest` (true fp32)
        # so nothing silently drops to TF32. Low precision is opt-in per op/config.
        jax.config.update("jax_default_matmul_precision", "highest")
        self.chain, self.species_map, self.cutoff, self.num_species = load_model(
            config_path, weights_path, precision, card_path=card_path
        )
        self.head_id = head_id
        self.d3 = d3
        self.pad_to = pad_to
        # Does this chain actually consume charge? `Chain.inputs` is the set of
        # names the program reads but never writes, so "charge" appears there
        # exactly when some op consumes it. Used only to decide whether to
        # announce the convention once -- a charge-aware model fed nothing looks
        # identical in its output to a neutral one, so that line is the only way
        # a caller learns which it got before trusting the numbers.
        self._charge_aware = "charge" in getattr(self.chain, "inputs", frozenset())
        self._announced = False
        self._fwd = self._make_forward()

    def _read_electronic_state(self, atoms):
        """Net charge and spin multiplicity (2S+1) for this structure.

        Falls back to the neutral singlet, which is correct for a neutral
        closed-shell molecule and wrong for everything else -- so on a
        charge-aware model say once, out loud, which keys were found.
        """
        info = atoms.info
        charge = info.get("charge", info.get("total_charge", 0))
        mult = info.get("multiplicity", info.get("spin", 1))
        charge = 0.0 if charge is None else float(charge)
        mult = 1.0 if mult is None else float(mult)
        if mult < 1:
            raise ValueError(
                f"spin multiplicity must be >= 1 (2S+1 convention), got {mult}. "
                "A caller holding 2S or n_up-n_down must convert."
            )
        if self._charge_aware and not self._announced:
            self._announced = True
            found = [k for k in ("charge", "total_charge", "multiplicity", "spin") if k in info]
            print(
                f"[splice] charge-aware model: charge={charge:+g} multiplicity={mult:g} "
                f"(2S+1) from atoms.info keys {found or 'NONE -> neutral singlet default'}",
                flush=True,
            )
        return charge, mult

    def _make_forward(self):
        num_species = self.num_species
        chain = self.chain
        head_id = self.head_id

        @jax.jit
        def fwd(positions, cell, species, inda, indb, nn_vecs, mask, charge, spin):
            N = positions.shape[0]
            batch = {
                "nn_vecs": nn_vecs,
                "positions": positions,
                "cell": cell,
                "species": species.astype(jnp.int32),
                "inda": inda,
                "indb": indb,
                "inde": jnp.zeros(N, jnp.int32),
                "nats": jnp.asarray([N], jnp.int32),
                "mask": mask,
                # Per-graph, shape [1]. TRACED, not static, so evaluating the
                # same geometry at a different charge costs no recompile -- a
                # benchmark sweeping charge states would otherwise pay one
                # compile per state. A chain with no charge_spin op ignores them.
                "charge": charge,
                "spin": spin,
            }
            if head_id is not None:
                batch["head_id"] = jnp.asarray([head_id], jnp.int32)
            # node-energy path: padded atoms (no edges) carry a spurious atomic
            # self-energy, so the summed out["E"] is wrong under padding. Sum the
            # per-node energies over the REAL atoms only (caller slices [:N_real]).
            out = model_call(
                chain,
                batch,
                num_species=num_species,
                compute_virial=True,
                compute_edge_forces=True,
            )
            return out["node_energy"], out["F"], out["virial"][0]

        return fwd

    def _build_batch(self, atoms):
        cutoff = self.cutoff
        cell_np, pbc = _safe_cell(atoms, cutoff)
        i, j, D = neighbour_list(
            "ijD", pbc=pbc, cell=cell_np, positions=atoms.get_positions(), cutoff=cutoff
        )
        species = np.array([self.species_map[chemical_symbols[z]] for z in atoms.numbers], np.int32)
        N, E = len(atoms), len(i)
        pos = atoms.get_positions().astype(np.float32)
        cell_j = np.asarray(cell_np, np.float32)

        if self.pad_to is not None:
            Np, Ep = self.pad_to
            assert Np >= N and Ep >= E, f"pad_to {self.pad_to} too small for N={N} E={E}"
            pos = np.pad(pos, ((0, Np - N), (0, 0)))
            species = np.pad(species, (0, Np - N))
            inda = np.pad(i.astype(np.int32), (0, Ep - E))
            indb = np.pad(j.astype(np.int32), (0, Ep - E))
            nn = np.pad(D.astype(np.float32), ((0, Ep - E), (0, 0)))
            mask = np.arange(Ep) < E
        else:
            inda, indb = i.astype(np.int32), j.astype(np.int32)
            nn = D.astype(np.float32)
            mask = np.ones(E, bool)

        return (
            jnp.asarray(pos),
            jnp.asarray(cell_j),
            jnp.asarray(species),
            jnp.asarray(inda),
            jnp.asarray(indb),
            jnp.asarray(nn),
            jnp.asarray(mask),
        ), (N, cell_np, pbc)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        args, (N, cell_np, pbc) = self._build_batch(self.atoms)
        charge, mult = self._read_electronic_state(self.atoms)
        node_e, F, virial = self._fwd(
            *args, jnp.asarray([charge], jnp.float32), jnp.asarray([mult], jnp.float32)
        )
        E = float(np.asarray(node_e)[:N].sum())
        F = np.asarray(F)[:N]
        results = {"energy": E, "free_energy": E, "forces": F}

        if pbc.any():
            vol = float(abs(np.linalg.det(cell_np)))
            stress3 = np.asarray(virial) / vol  # virial = +dE/dstrain = stress·V
            results["stress"] = full_3x3_to_voigt_6_stress(stress3)

        if self.d3 is not None:
            # Additive dispersion. Use the getters (not .calculate(properties=...))
            # so the d3 calculator computes the full E/F[/stress] set regardless of
            # which single property ASE happened to request from us.
            e_d3 = self.d3.get_potential_energy(self.atoms)
            f_d3 = self.d3.get_forces(self.atoms)
            results["energy"] += e_d3
            results["free_energy"] += e_d3
            results["forces"] = results["forces"] + f_d3
            if "stress" in results:
                results["stress"] = results["stress"] + self.d3.get_stress(self.atoms)

        self.results = results
