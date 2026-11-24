# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model card: dataset-derived metadata that travels with a model.

A card binds a SPLICE chain config to a dataset. It records:
- the species-symbol → species-id map the loader should use
- the neighbour-list cutoff the dataset was prepared with
- the average number of neighbours at that cutoff (the canonical value
  the MP `epsilon = 1 / avg_num_neighbours` is meant to track)
- per-species energy offsets, with both "tabulated" (isolated-atom DFT
  references) and "lstsq" (least-squares fit on the dataset) variants
  stored side by side and a chosen default
- which outputs the model produces (E, F, stress, multipole, ...)

The card is generated **once** per dataset by `packages/splice/scripts/build_model_card.py`
and read at the start of every training / inference run. The chain's
learnable `SpeciesOffsets` array is seeded from the card via
`seed_species_offsets(chain, card)`.

The card is dataset-side: it does NOT bind to a specific architecture YAML.
Callers pick the arch yaml at training time and apply the card via
`resolve_arch(arch_path, card)` (which uses `overrides`) and
`seed_species_offsets(chain, card)`.

YAML schema (all top-level keys, kept flat enough to read quickly):
    species: {<symbol>: <id>, ...}
    cutoff: <float>
    avg_num_neighbors: <float>
    offsets:
        default: tabulated | lstsq
        tabulated: [<float>, ...]
        lstsq:     [<float>, ...]
    outputs: [<name>, ...]
    overrides:                        # optional; applied via `resolve_arch`
        <op_name>.<field>: <value>    # replaces this field in every step where
                                      # op==op_name. Useful for dataset-coupled
                                      # values that shouldn't be baked in the
                                      # architecture YAML (e.g. epsilon = 1/<N>).
    notes: { <free-form keys> }
"""

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import yaml

from splice.config import Chain, rebuild


def fit_species_offsets(frames: list[dict], num_species: int) -> np.ndarray:
    """Least-squares fit of per-species offsets: ``E_total ≈ Σ_s N_s · offset_s``.

    Returns a length-`num_species` array (in species-id order). For
    single-stoichiometry datasets (every frame has the same composition,
    e.g. 3BPA), the count matrix is rank-deficient and `np.linalg.lstsq`
    returns a min-norm solution — usable but not physically interpretable.
    """
    counts = np.zeros((len(frames), num_species), dtype=np.float64)
    energies = np.zeros((len(frames),), dtype=np.float64)
    for i, f in enumerate(frames):
        for s in range(num_species):
            counts[i, s] = int(np.sum(f["species"] == s))
        energies[i] = f["energy"]
    offsets, _, _, _ = np.linalg.lstsq(counts, energies, rcond=None)
    return offsets.astype(np.float32)


def compute_avg_num_neighbors(frames: list[dict]) -> float:
    """Mean directed-edge count per atom across the dataset."""
    total_edges = sum(int(f["senders"].shape[0]) for f in frames)
    total_atoms = sum(int(f["species"].shape[0]) for f in frames)
    return total_edges / total_atoms


def avg_neighbors_for_cutoff(card: dict, cutoff: float) -> float | None:
    """Pick the measured ``avg_num_neighbors`` for ``cutoff`` from the card's
    ``neighbors_by_cutoff`` table (keys = cutoff radii in Å, values = measured
    mean directed-edge count per atom on this dataset).

    Returns the matching value, or ``None`` if the table is absent or has no
    entry within 1e-6 of ``cutoff``. This is what lets one dataset card carry
    several measured neighbour counts so an Rc sweep (``--cutoff``) stays
    correctly normalised without recomputing.
    """
    table = card.get("neighbors_by_cutoff") or {}
    for k, v in table.items():
        if abs(float(k) - float(cutoff)) < 1e-6:
            return float(v)
    return None


def save_card(
    path: str | Path,
    *,
    species: dict[str, int],
    cutoff: float,
    avg_num_neighbors: float,
    offsets_tabulated: list[float] | np.ndarray | None = None,
    offsets_lstsq: list[float] | np.ndarray | None = None,
    offsets_default: str = "tabulated",
    outputs: list[str],
    overrides: dict[str, Any] | None = None,
    notes: dict[str, str] | None = None,
) -> None:
    """Write a model card to YAML.

    The card is dataset-side and intentionally does NOT bind to a specific
    architecture YAML. Use `resolve_arch(arch_path, card)` to apply card
    overrides to any compatible arch at training time.
    """
    if offsets_tabulated is None and offsets_lstsq is None:
        raise ValueError("at least one of offsets_tabulated/offsets_lstsq must be provided")
    if offsets_default not in {"tabulated", "lstsq"}:
        raise ValueError(f"offsets_default must be 'tabulated' or 'lstsq', got {offsets_default!r}")

    offsets: dict[str, Any] = {"default": offsets_default}
    if offsets_tabulated is not None:
        offsets["tabulated"] = [float(x) for x in offsets_tabulated]
    if offsets_lstsq is not None:
        offsets["lstsq"] = [float(x) for x in offsets_lstsq]

    card = {
        "species": dict(species),
        "cutoff": float(cutoff),
        "avg_num_neighbors": float(avg_num_neighbors),
        "offsets": offsets,
        "outputs": list(outputs),
    }
    if overrides:
        card["overrides"] = dict(overrides)
    if notes:
        card["notes"] = dict(notes)
    Path(path).write_text(yaml.safe_dump(card, sort_keys=False))


def load_card(path: str | Path) -> dict:
    """Read a model card YAML and return its dict."""
    return yaml.safe_load(Path(path).read_text())


def card_offsets(card: dict, *, source: str | None = None) -> np.ndarray:
    """Pick which offsets array to use from a card.

    `source` overrides `card["offsets"]["default"]`. Raises if the requested
    variant isn't present in the card.
    """
    offsets_block = card["offsets"]
    src = source if source is not None else offsets_block["default"]
    if src not in offsets_block:
        raise KeyError(
            f"card has no {src!r} offsets; available: "
            f"{[k for k in offsets_block if k != 'default']}"
        )
    return np.asarray(offsets_block[src], dtype=np.float32)


def seed_species_offsets(
    chain: Chain,
    offsets_or_card: np.ndarray | list[float] | dict,
    *,
    source: str | None = None,
) -> Chain:
    """Replace the `SpeciesOffsets` module's learnable offsets array.

    Accepts a 1-D offsets array directly or a card dict (in which case
    `card_offsets(card, source=source)` selects the variant). Pads with
    zeros at the end if shorter than the chain's `num_species` slot;
    raises if longer.
    """
    from splice.ops.head import SpeciesOffsets

    if isinstance(offsets_or_card, dict):
        offsets = card_offsets(offsets_or_card, source=source)
    else:
        offsets = np.asarray(offsets_or_card, dtype=np.float32)

    indices = [i for i, m in enumerate(chain.ops) if isinstance(m, SpeciesOffsets)]
    if not indices:
        raise ValueError("chain has no SpeciesOffsets op")
    if len(indices) > 1:
        raise ValueError(f"chain has multiple SpeciesOffsets ops at positions {indices}; ambiguous")
    idx = indices[0]

    n_total = int(chain.ops[idx].offsets.shape[0])
    if offsets.shape[0] > n_total:
        raise ValueError(f"offsets length {offsets.shape[0]} > model num_species {n_total}")
    if offsets.shape[0] < n_total:
        offsets = np.concatenate(
            [offsets, np.zeros((n_total - offsets.shape[0],), dtype=np.float32)]
        )

    new_module = eqx.tree_at(lambda m: m.offsets, chain.ops[idx], jnp.asarray(offsets))
    new_modules = tuple(new_module if i == idx else m for i, m in enumerate(chain.ops))
    return rebuild(chain, new_modules)


def resolve_arch(arch_path: str | Path, card: dict) -> list[dict]:
    """Load an architecture YAML with the card's `overrides` applied.

    The card's `overrides` block (if present) is a flat dict whose keys are
    ``"<op_name>.<field>"`` and whose values replace that field in every step
    of the arch yaml where ``op == op_name``. This is the canonical way for a
    model card to inject dataset-coupled per-op values without those values
    being baked into the architecture YAML itself.

    Auto-derived overrides (applied unless the card explicitly overrides them):

      ``<op>.<NEIGHBOR_NORM_PARAM> = 1 / card["avg_num_neighbors"]``
          The standard MACE epsilon. Auto-injected from the card so that
          an architecture YAML need not hardcode ``epsilon`` —
          the value follows the dataset. An explicit override in the card's
          ``overrides`` block wins (e.g. for ablations).

          Self-declared per op, for the same reason as the two below: this used
          to be a hardcoded ``{channelwise_mp, escn_mp}`` pair, so every other
          message-passing op silently kept whatever epsilon its arch YAML was
          written with. That made a std-vs-fused A/B on a new dataset compare
          two DIFFERENT message normalisations — the fused arch kept OMat's
          0.125 while the standard one correctly picked up the card's value.

      ``<op>.<SPECIES_COUNT_PARAM> = len(card["species"])``
          Each species-coupled op (species_embedding, symmetric_contraction,
          species_offsets, irreps_indexed_linear, alchemical_*) self-declares
          which init param is the data species count; it is auto-filled here so
          arch YAMLs need not hardcode ``num_species`` / ``num_indices`` (and
          can't drift). Arch-specific dims like alchemical ``K`` are untouched.
    """
    from splice.config import load_config

    overrides = dict(card.get("overrides") or {})
    auto = {}
    if "avg_num_neighbors" in card and float(card["avg_num_neighbors"]) > 0:
        import splice.ops  # ensure ops self-register before lookup
        from splice.registry import get_op, list_ops

        eps_auto = 1.0 / float(card["avg_num_neighbors"])
        for op_name in list_ops():
            param = getattr(get_op(op_name), "NEIGHBOR_NORM_PARAM", None)
            if param:
                auto[f"{op_name}.{param}"] = eps_auto
    if "cutoff" in card and float(card["cutoff"]) > 0:
        # The cutoff (r_max) follows the card so the radial basis cannot drift
        # from the neighbour-list cutoff, and a ``--cutoff`` override (same
        # dataset, different Rc) auto-propagates without editing the arch. Each
        # radial-basis op self-declares the param via ``CUTOFF_PARAM`` (same
        # drift-proof pattern as ``SPECIES_COUNT_PARAM`` below), so any basis
        # variant (Bessel, learnable-local, ...) picks it up.
        import splice.ops
        from splice.registry import get_op, list_ops

        cutoff = float(card["cutoff"])
        for op_name in list_ops():
            param = getattr(get_op(op_name), "CUTOFF_PARAM", None)
            if param:
                auto[f"{op_name}.{param}"] = cutoff
    # Auto-fill each op's data-species-count param from the card's species map.
    # Ops self-declare the param name via ``SPECIES_COUNT_PARAM`` (``num_species``,
    # or ``num_indices`` for the per-species indexed linear); arch-specific dims
    # (e.g. alchemical ``K``) are not declared and stay arch-controlled. Arch
    # YAMLs thus need not hardcode the species count, and it cannot drift.
    if card.get("species"):
        import splice.ops  # noqa: F401 — ensure ops self-register before lookup
        from splice.registry import get_op, list_ops

        n_species = len(card["species"])
        for op_name in list_ops():
            param = getattr(get_op(op_name), "SPECIES_COUNT_PARAM", None)
            if param:
                auto[f"{op_name}.{param}"] = n_species
    # explicit overrides take precedence over auto-derived ones
    merged = {**auto, **overrides}
    return load_config(arch_path, overrides=merged)
