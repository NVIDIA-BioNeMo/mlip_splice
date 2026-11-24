# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-irrep activation (scalars only) and per-irrep dict surgery (add/slice/cat)."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from e3 import Irrep, Irreps
from jax import Array

from splice.registry import register_op

# A deterministic quantile grid, so the normalization constant is reproducible.
_SAMPLES = 1_000_001


def normalize_function(activation: Callable) -> Callable:
    """Rescale ``activation`` to unit second moment under a standard normal.

    Returns ``phi(x) / c`` with ``c**2 = E[phi(N(0, 1))**2]``, so composing
    activations neither grows nor shrinks the scale of the features. The
    constant is computed once, at trace time, over a deterministic quantile
    grid rather than a random sample: ``x_i`` splits the normal distribution
    into ``_SAMPLES + 1`` equal-probability intervals.
    """
    with jax.ensure_compile_time_eval():
        quantiles = jnp.linspace(-1.0, 1.0, _SAMPLES + 2)[1:-1]
        x = jnp.sqrt(2.0) * jax.scipy.special.erfinv(quantiles)
        scale = float(jnp.mean(jax.vmap(activation)(x) ** 2) ** 0.5)
    if scale < 1e-5:
        raise ValueError("cannot normalize an activation whose second moment is zero")
    if abs(scale - 1.0) < 1e-6:
        return activation
    return lambda value: activation(value) / scale


@register_op("scalar_activation")
class ScalarActivation(eqx.Module):
    """Apply activation to scalar irreps; non-scalars pass through."""

    activation: Callable = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(self, activation: Callable):
        self.activation = normalize_function(activation)

    def __call__(self, node_feats: dict[Irrep, Array]) -> dict[Irrep, Array]:
        act = self.activation
        return {ir: act(v) if ir.is_scalar() else v for ir, v in node_feats.items()}


@register_op("scalar_gated_activation")
class ScalarGatedActivation(eqx.Module):
    """Sigmoid gates from the scalar channels, applied to every irrep.

    Unlike ``scalar_activation``, non-scalars are gated rather than passed
    through. The scalar branch is the same gate applied to itself
    (``silu(x) == x * sigmoid(x)``); higher irreps use a per-channel
    affine of the leading scalar channels before sigmoid. No ``activation``
    argument: the gate *is* the nonlinearity, so YAML ``defaults.activation``
    does not reach this op.

    ``normalize`` rescales SiLU and sigmoid separately to unit second moment.
    Gate parameters use ``param_dtype`` (FP32 by default). The affine runs in
    at least FP32, retaining FP64 when either the scalars or parameters use it.
    """

    gate_scale: dict[str, Array]
    gate_bias: dict[str, Array]
    irreps_in: Irreps = eqx.field(static=True)
    scalar_activation: Callable = eqx.field(static=True)
    gate_activation: Callable = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    _scalar_ir: Irrep = eqx.field(static=True)
    _higher_muls: tuple[tuple[Irrep, int], ...] = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(
        self, irreps_in: Irreps, normalize: bool = False, *, param_dtype: Any = jnp.float32
    ):
        scalar_entries = [(mul, ir) for mul, ir in irreps_in if ir.is_scalar()]
        if len(scalar_entries) != 1:
            raise ValueError("scalar_gated_activation requires exactly one trivial scalar irrep")
        scalar_mul, self._scalar_ir = scalar_entries[0]
        higher = [(mul, ir) for mul, ir in irreps_in if not ir.is_scalar()]
        if too_wide := [(str(ir), mul) for mul, ir in higher if mul > scalar_mul]:
            raise ValueError(
                "scalar_gated_activation needs at least as many scalar channels as every "
                f"higher irrep; scalar mul={scalar_mul}, higher={too_wide}"
            )

        self.irreps_in = irreps_in
        self.normalize = bool(normalize)
        self.scalar_activation = normalize_function(jax.nn.silu) if self.normalize else jax.nn.silu
        self.gate_activation = (
            normalize_function(jax.nn.sigmoid) if self.normalize else jax.nn.sigmoid
        )
        self._higher_muls = tuple((ir, mul) for mul, ir in higher)
        self.gate_scale = {str(ir): jnp.ones((mul,), dtype=param_dtype) for mul, ir in higher}
        self.gate_bias = {str(ir): jnp.zeros((mul,), dtype=param_dtype) for mul, ir in higher}

    def __call__(self, node_feats: dict[Irrep, Array]) -> dict[Irrep, Array]:
        scalar = node_feats[self._scalar_ir]
        out = dict(node_feats)
        out[self._scalar_ir] = self.scalar_activation(scalar)
        for ir, mul in self._higher_muls:
            scalar_channels = scalar[..., :mul]
            gate_scale = self.gate_scale[str(ir)]
            compute_dtype = jnp.result_type(scalar_channels.dtype, gate_scale.dtype, jnp.float32)
            logits = scalar_channels.astype(compute_dtype) * gate_scale
            logits = logits + self.gate_bias[str(ir)]
            gates = self.gate_activation(logits).astype(node_feats[ir].dtype)
            out[ir] = node_feats[ir] * gates
        return out


@register_op("irreps_add")
class IrrepsAdd(eqx.Module):
    """Per-irrep residual add. Strict key match is the default.

    With ``allow_partial``, missing fields pass through without allocating zeros.
    """

    allow_partial: bool = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(self, allow_partial: bool = False):
        self.allow_partial = bool(allow_partial)

    def __call__(self, node_feats: dict[Irrep, Array], b: dict[Irrep, Array]) -> dict[Irrep, Array]:
        if not self.allow_partial:
            assert node_feats.keys() == b.keys()
            return {ir: node_feats[ir] + b[ir] for ir in node_feats}

        out = dict(node_feats)
        for ir, value in b.items():
            out[ir] = out[ir] + value if ir in out else value
        return out


@register_op("irreps_identity")
class IrrepsIdentity(eqx.Module):
    """Return every currently present irrep unchanged, without allocating buffers."""

    OUTPUTS = "node_feats"

    def __call__(self, node_feats: dict[Irrep, Array]) -> dict[Irrep, Array]:
        return node_feats


@register_op("irreps_slice")
class IrrepsSlice(eqx.Module):
    """Leading channel-slice of every irrep: ``v[..., :mul]``. A free XLA view —
    used to feed a narrower op (e.g. a per-l symmetric-contraction chunk) a
    channel subset of a shared wide buffer. Channel selection beyond a prefix
    belongs in the feeding linear (any gather is a 0/1 matrix), keeping this op
    a pure view."""

    mul: int = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(self, mul: int):
        self.mul = int(mul)

    def __call__(self, node_feats: dict[Irrep, Array]) -> dict[Irrep, Array]:
        return {ir: v[..., : self.mul] for ir, v in node_feats.items()}


@register_op("irreps_cat")
class IrrepsCat(eqx.Module):
    """Per-irrep concatenation of two dicts along the channel axis; irreps
    present in only one operand pass through. With ``b`` disjoint from
    ``node_feats`` this is a pure dict union (e.g. assembling non-uniform node
    features from per-l symmetric-contraction chunks). Chain for >2 operands."""

    OUTPUTS = "node_feats"

    def __call__(self, node_feats: dict[Irrep, Array], b: dict[Irrep, Array]) -> dict[Irrep, Array]:
        out = dict(node_feats)
        for ir, v in b.items():
            out[ir] = jnp.concatenate([out[ir], v], axis=-1) if ir in out else v
        return out
