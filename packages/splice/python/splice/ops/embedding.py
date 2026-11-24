# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Species embedding: lookup table -> initial scalar node_feats dict."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from e3 import O3, Irrep
from jax import Array

from splice.ops._initializers import initialize_linear
from splice.registry import register_op


@register_op("species_embedding")
class SpeciesEmbedding(eqx.Module):
    """Per-species learned scalar features.

    Output is ``{0e: [num_nodes, 1, mul]}`` in ir_mul layout.

    Lookup is ``embedding[species]``, not ``repeat(embedding, species_counts)``.
    ``model_call`` already materializes per-node species ids; repeating from
    counts would rebuild that index with a traced run-length expand (extra
    scatter / fusions) and is only correct when atoms form contiguous species
    blocks. The gather is the same on the sorted layout and also correct when
    they do not. ``species_counts`` stays in context for grouped GEMMs.
    """

    embedding: Array
    num_species: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    weight_scaling: bool = eqx.field(static=True)
    init: str = eqx.field(static=True)
    SPECIES_COUNT_PARAM = "num_species"  # data species count; auto-filled from the card

    OUTPUTS = "node_feats"

    def __init__(
        self,
        num_species: int,
        num_features: int,
        weight_scaling: bool = True,
        init: str = "normal",
        *,
        param_dtype: Any,
        key: Array,
    ):
        self.num_species = num_species
        self.num_features = num_features
        self.weight_scaling = bool(weight_scaling)
        self.init = init
        self.embedding = initialize_linear(
            key,
            (num_species, num_features),
            param_dtype,
            init=self.init,
            weight_scaling=self.weight_scaling,
        )

    def __call__(self, species: Array) -> dict[Irrep, Array]:
        # Per-node gather: species is already the row index into the table.
        feats = self.embedding[species]  # [num_nodes, num_features]
        if self.weight_scaling:
            feats = feats / jnp.sqrt(self.num_species)
        # ir_mul: [num_nodes, ir.dim=1, mul=num_features]
        return {O3.trivial(): feats[:, None, :]}


@register_op("charge_spin_embedding_mlp")
class ChargeSpinEmbeddingMLP(eqx.Module):
    """Inject per-molecule total charge Q + spin multiplicity S into the initial
    l0 node features: Linear(2->hidden) -> SiLU -> Linear(hidden->num_features),
    ADDED to the 0e channels.

    ``charge``/``spin`` are per-GRAPH and are gathered to atoms through
    ``graph_index`` -- the same (species-sorted) atom order as ``node_feats``.
    Because it ADDS rather than concatenates, every downstream irrep width is
    unchanged, so no other op and no card edit is needed.

    Insert directly after ``species_embedding``, whose output is the sole ``0e``
    entry ``{0e: [N, 1, mul]}``; this op preserves that shape and layout.

    Inputs are normalised by ``charge_scale``/``spin_scale`` (OMol ranges: charge
    -10..+10, spin multiplicity 1..11) so the projection sees O(1) magnitudes.
    A batch may omit charge/spin entirely (e.g. an eval batcher that carries no
    per-graph inputs) -> those graphs fall back to the neutral singlet baseline
    (charge 0, spin 1), the same default the packers use.

    NORMAL-init, not zero-init: training is from scratch, so the charge/spin
    channel is a live contribution from step 0 even on neutral data.
    """

    w1: Array  # [2, hidden]
    b1: Array  # [hidden]
    w2: Array  # [hidden, num_features]
    num_features: int = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    charge_scale: float = eqx.field(static=True)
    spin_scale: float = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(
        self,
        num_features: int,
        *,
        hidden: int = 64,
        charge_scale: float = 10.0,
        spin_scale: float = 10.0,
        param_dtype: Any,
        key: Array,
    ):
        self.num_features = num_features
        self.hidden = hidden
        self.charge_scale = charge_scale
        self.spin_scale = spin_scale
        k1, k2 = jax.random.split(key, 2)
        self.w1 = jax.random.normal(k1, (2, hidden), param_dtype) / jnp.sqrt(2.0)
        self.b1 = jnp.zeros((hidden,), param_dtype)
        self.w2 = jax.random.normal(k2, (hidden, num_features), param_dtype) / jnp.sqrt(hidden)

    def __call__(
        self,
        node_feats: dict[Irrep, Array],
        charge: Array,
        spin: Array,
        graph_index: Array,
    ) -> dict[Irrep, Array]:
        scalar = node_feats[O3.trivial()]  # [N, 1, mul]
        dtype = scalar.dtype
        n = scalar.shape[0]
        q = (
            jnp.zeros((n,), dtype)
            if charge is None
            else (charge[graph_index] / self.charge_scale).astype(dtype)
        )
        s = (
            jnp.full((n,), 1.0 / self.spin_scale, dtype)
            if spin is None
            else (spin[graph_index] / self.spin_scale).astype(dtype)
        )
        qs = jnp.stack([q, s], axis=-1)  # [N, 2]
        h = jax.nn.silu(qs @ self.w1.astype(dtype) + self.b1.astype(dtype))  # [N, hidden]
        add = h @ self.w2.astype(dtype)  # [N, mul]
        out = dict(node_feats)
        out[O3.trivial()] = scalar + add[:, None, :]
        return out
