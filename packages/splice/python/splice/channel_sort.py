# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sort channelwise-MP channels by radial support — a free, exact model transform.

Each in-kernel ``message_passing`` layer carries a per-channel outer radial cutoff
(the free-knot knot ``e·r_max``, or the bump right edge ``q``). The channel axis
(``mul``) is a pure relabelling: the tensor product is channelwise, so permuting the
channels is absorbed exactly by permuting the two neighbouring linears —

    preceding linear:  permute weight **columns** (mul_out) by π
    the CW op:         permute the radial's per-channel leaves + ``gain`` by π
    following linear:  permute weight **rows** (mul_in) by π, within each
                       ``mul``-sized path block (a repeated-irrep CW output is
                       ``#paths × mul`` channels, path-major)

Sorting by cutoff makes each warp's radial supports coherent, so anvil's
``zero_skip`` warp-gate fires more (edges past a warp's max cutoff skip the whole
warp). Measured on MACE-L 128: ≈14% faster inference (E+F), and the whole win is
the TP kernel (−18% GPU), with bit-identical outputs.

Use ``sort_channels_by_extent(chain)`` on any chain; check the result with a full-model
energy/force equivalence test.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from splice.ops.linear import IrrepsLinear
from splice.ops.message_passing import MessagePassingOp


def _is_channelwise(op) -> bool:
    """A fused-sph channelwise MP: a MessagePassingOp with an in-kernel per-channel
    radial (``radial is None`` for the external-``mp_weights`` MACE source)."""
    return isinstance(op, MessagePassingOp) and getattr(op.mp, "radial", None) is not None


def extent_argsort(op: MessagePassingOp) -> np.ndarray:
    """Channel order that sorts this layer's outer radial cutoff ascending (free-knot
    knot ``e·r_max`` or bump right edge ``q`` — both per channel)."""
    return np.argsort(np.asarray(op.mp.radial.support_radius())).astype(np.int32)


def _perm_linear_out(lin: IrrepsLinear, pi: Array, mul: int) -> IrrepsLinear:
    """Permute a linear's output channels (mul_out axis) — feeds the sorted CW input.

    The CW carries ``mul`` per-channel radials; ``pi`` (length ``mul``) sorts them.
    A repeated-irrep CW consumes its input as ``#legs`` back-to-back ``mul``-blocks
    per irrep (e.g. ``128x0e`` = 4 legs × ``mul=32``), so this linear's ``mul_out``
    axis is permuted **per block** by the same ``pi`` — mirroring ``_perm_linear_in``
    on the column axis. Uniform CW (``width == mul``) → one block → plain ``[:, pi]``."""
    wk = list(lin.w)

    def cols(width: int) -> Array:
        n_block, rem = divmod(width, mul)
        if rem:
            raise ValueError(f"preceding-linear out-width {width} is not a multiple of mul={mul}")
        return jnp.concatenate([b * mul + pi for b in range(n_block)])

    lin = eqx.tree_at(
        lambda m: [m.w[k] for k in wk], lin, [lin.w[k][:, cols(lin.w[k].shape[1])] for k in wk]
    )
    if lin.b:
        bk = list(lin.b)
        lin = eqx.tree_at(
            lambda m: [m.b[k] for k in bk], lin, [lin.b[k][cols(lin.b[k].shape[0])] for k in bk]
        )
    return lin


def _perm_linear_in(lin: IrrepsLinear, pi: Array, mul: int) -> IrrepsLinear:
    """Permute a linear's input channels (mul_in axis), within each ``mul`` path block."""
    wk = list(lin.w)
    new = []
    for k in wk:
        w = lin.w[k]
        n_path, rem = divmod(w.shape[0], mul)
        if rem:
            raise ValueError(
                f"following-linear irrep {k!r} has {w.shape[0]} in-channels, "
                f"not a multiple of mul={mul}"
            )
        block = jnp.concatenate([b * mul + pi for b in range(n_path)])
        new.append(w[block])
    return eqx.tree_at(lambda m: [m.w[k] for k in wk], lin, new)


def _perm_cw(op: MessagePassingOp, pi: Array) -> MessagePassingOp:
    """Permute the CW op's per-channel radial leaves by π (radial owns t + window; gain
    may be shared/None)."""
    mp = op.mp
    new_radial = mp.radial.permuted(pi)
    if mp.gain is None:
        return eqx.tree_at(lambda o: o.mp.radial, op, new_radial)
    return eqx.tree_at(lambda o: (o.mp.radial, o.mp.gain), op, (new_radial, mp.gain[:, pi]))


def sort_channels_by_extent(chain):
    """Return a copy of ``chain`` with every channelwise-MP layer's channels sorted
    by radial extent, the permutation absorbed into the neighbouring linears.

    Exact: the model output is unchanged (to fp32 roundoff). Assumes each CW op is
    flanked by an ``irreps_linear`` (the MACE MP block) — raises if not, rather than
    silently corrupting. Verify end-to-end with a full-model energy/force equivalence check.
    """
    ops = list(chain.ops)
    for k, op in enumerate(ops):
        if not _is_channelwise(op):
            continue
        if not (0 < k < len(ops) - 1):
            raise ValueError(f"CW op {k} has no flanking linear to absorb the permutation")
        pre, post = ops[k - 1], ops[k + 1]
        if not isinstance(pre, IrrepsLinear):
            raise TypeError(f"expected irreps_linear before CW op {k}, got {type(pre).__name__}")
        if not isinstance(post, IrrepsLinear):
            raise TypeError(f"expected irreps_linear after CW op {k}, got {type(post).__name__}")
        pi = jnp.asarray(extent_argsort(op))
        ops[k - 1] = _perm_linear_out(pre, pi, op.mp.mul)
        ops[k] = _perm_cw(op, pi)
        ops[k + 1] = _perm_linear_in(post, pi, op.mp.mul)
    return eqx.tree_at(lambda c: c.ops, chain, tuple(ops))


def radial_support_penalty(model, *, exponent: float = 2.0) -> Array:
    """Mean over every fused channelwise-MP radial channel of ``(support_radius /
    r_max) ** exponent`` — a differentiable, batch-independent scalar in
    ``(extent_floor**exponent, 1)``.

    A training loss adds this to *push per-channel radial cutoffs shorter*: shorter
    supports make anvil's warp-collective ``zero_skip`` fire more (edges past a
    warp's max cutoff skip the whole warp — the very lever ``sort_channels_by_extent``
    exploits at deploy time; see the module docstring). The support runs through the
    squash floor, so the outer edge is bounded to ``extent_floor·r_max`` — the penalty
    can only *shorten* a channel, never collapse it to a dead channel.

    ``exponent=2`` (default) concentrates the pressure on the long-cutoff tail — the
    channels that set each warp's max cutoff, i.e. the ones that actually cost. Returns
    a traced ``0.0`` when the model carries no fused CW radial (nothing to shorten).
    """
    ops = [
        leaf
        for leaf in jax.tree_util.tree_leaves(model, is_leaf=_is_channelwise)
        if _is_channelwise(leaf)
    ]
    if not ops:
        return jnp.asarray(0.0)
    fracs = [jnp.ravel(op.mp.radial.support_radius()) / op.mp.radial.r_max for op in ops]
    return jnp.mean(jnp.concatenate(fracs) ** exponent)
