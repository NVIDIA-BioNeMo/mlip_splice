# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Muon optimizer wiring for Chain.

Muon (MomentUm Orthogonalized by Newton-Schulz) applies to 2D *hidden* weight
matrices: it orthogonalizes the momentum-averaged gradient of each matrix via a
few Newton-Schulz steps before the LR step. In SPLICE the equivariant
``IrrepsLinear`` weight is ``w: dict[str, Array]`` — one ``[mul_in, mul_out]``
block per irrep. Each block mixes *channels* within an irrep, applied uniformly
across the ``ir.dim`` components, so orthogonalizing a block in channel space
**commutes with the irrep action** — i.e. per-block Muon is equivariance-
preserving. Each ``IrrepsLinear`` records whether its matrices use Muon
or Adam; omitted ``optimizer`` keeps the historical split (hidden Muon,
readout Adam). ``optax.multi_transform`` applies those choices through
an explicit chain-shaped label tree.

Phase-1 scope (deliberate):
  * Muon   : ``IrrepsLinear.w`` matrices marked ``optimizer="muon"``.
  * Adam(W): everything else. symmetric_contraction / sparse_uvu weights are
             higher-rank tensors with no clean 2D orthogonalization — they stay
             on Adam and can be nudged via the annotation LR handle (SCx4, ...).
  * Readouts default to Adam (standard Muon usage; 1-column readout blocks are
    degenerate under NS anyway), and species embeddings remain on Adam.

``optax.scale_by_muon``'s default dimension numbers (reduction_axis=0,
output_axis=1) already match the ``[mul_in, mul_out]`` blocks, so no per-leaf
``MuonDimensionNumbers`` are needed.
"""

from typing import Any

import equinox as eqx
import jax
import optax
from optax import contrib as _oc

from splice.config import Chain, rebuild
from splice.ops.linear import IrrepsLinear

MUON = "muon"
ADAM = "adam"


# Every Muon-governed leaf is a 2D [mul_in, mul_out] block: reduction axis 0
# (mul_in), output axis 1 (mul_out). Passed as a callable so it resolves against
# the (None-masked) update tree multi_transform hands to scale_by_muon — its
# default ``weight_dimension_numbers=None`` mismatches that masked structure.
def _weight_dim_nums(updates):
    dn = _oc.MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    return jax.tree.map(lambda _x: dn, updates)


def _op_label_params(op: Any) -> Any:
    """Op-shaped labels using each ``IrrepsLinear.optimizer`` declaration."""
    params, _ = eqx.partition(op, eqx.is_inexact_array)
    labels = jax.tree.map(lambda _x: ADAM, params)
    if isinstance(op, IrrepsLinear) and op.optimizer == MUON and getattr(op, "w", None):
        labels = eqx.tree_at(lambda t: t.w, labels, {k: MUON for k in op.w})
    return labels


def muon_label_tree(chain: Chain) -> Chain:
    """Chain-shaped PyTree of ``"muon"``/``"adam"`` labels aligned with the
    ``eqx.partition(chain, is_inexact_array)`` params partition."""
    return rebuild(chain, tuple(_op_label_params(op) for op in chain.ops))


def muon_leaf_counts(chain: Chain) -> tuple[int, int]:
    """(#muon leaves, #adam leaves) — for a launch-time sanity print."""
    leaves = jax.tree.leaves(muon_label_tree(chain))
    return sum(x == MUON for x in leaves), sum(x == ADAM for x in leaves)


def _scaled_lr(base_lr: Any, scale: float) -> Any:
    if scale == 1.0:
        return base_lr
    if callable(base_lr):
        return lambda step: base_lr(step) * scale
    return base_lr * scale


def make_muon_optimizer(
    chain: Chain,
    base_lr: Any,
    *,
    muon_lr_scale: float = 1.0,
    weight_decay: float = 0.0,
    adam_lr: Any = None,
    b1: float = 0.9,
    b2: float = 0.999,
    muon_beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    amsgrad: bool = False,
) -> optax.GradientTransformation:
    """Muon on the equivariant hidden linear blocks + Adam(W) on the rest.

    ``base_lr`` may be a float or an optax schedule (callable). The Muon group
    uses ``base_lr * muon_lr_scale`` (Muon usually wants a somewhat higher LR
    than Adam); the Adam group uses ``adam_lr`` if given, else ``base_lr``.
    """
    labels = muon_label_tree(chain)
    muon_tx = optax.chain(
        _oc.scale_by_muon(
            beta=muon_beta,
            ns_steps=ns_steps,
            nesterov=nesterov,
            weight_dimension_numbers=_weight_dim_nums,
        ),
        optax.scale_by_learning_rate(_scaled_lr(base_lr, muon_lr_scale)),
    )
    # Adam(W) on the non-Muon group (embeddings/SC/uvu/radial/readouts/biases).
    # Use AMSGrad when the baseline does, so those layers — notably the critical
    # symmetric_contraction — are identical to the AdamW+handle baseline and only
    # the hidden linears differ (Muon vs Adam).
    alr = adam_lr if adam_lr is not None else base_lr
    if weight_decay > 0.0:
        adam_tx = optax.adamw(alr, b1=b1, b2=b2, weight_decay=weight_decay)
    elif amsgrad:
        adam_tx = optax.amsgrad(alr, b1=b1, b2=b2)
    else:
        adam_tx = optax.adam(alr, b1=b1, b2=b2)
    # NB: pass labels via a callable — a draupnir ``Chain`` is itself callable, so
    # optax.multi_transform would otherwise mistake the label tree for a
    # ``params -> labels`` function and try to invoke it.
    return optax.multi_transform({MUON: muon_tx, ADAM: adam_tx}, lambda _params: labels)
