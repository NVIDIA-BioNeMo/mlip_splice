# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The common precision language.

One vocabulary for *every* numeric compute mode in the chain — dense matmul,
per-species grouped GEMM, Loom tensor product, cue symmetric contraction.
It is JAX's own ``DotAlgorithmPreset`` (form ``{elem}_{elem}_{accum}[_xN]``),
with short lowercase aliases and the per-backend translations bolted on.

Design rule (see the ops): **nothing is implicit.** Every weight- or
matmul-bearing op takes an explicit ``precision:`` of this vocabulary and a
``param_dtype:`` for its weight storage, both *required* — omit one and
construction fails loudly. There is no silent TF32, no "follow the input"
accumulation guess, and no automatic activation down-cast. If you want lower
precision you name it; activations change dtype only through an explicit
``cast`` op.

The three orthogonal axes this pins down:

- **param_dtype** — storage dtype of the weights (``f32`` master, etc.). Lives
  in the checkpoint; a constructor arg only because init needs it.
- **activation dtype** — what flows through the chain's named buffers. Carried
  by the arrays themselves; set by ``cast`` ops, never guessed by an op.
- **precision** — how a bilinear op multiplies-accumulates: the element dtype it
  truncates inputs to, the accumulator dtype, and the split-pass count. This is
  the ``Precision`` below; it subsumes the old ``math_dtype`` (its accumulator)
  and the old ``jax.lax.Precision`` (its element mode).

Vocabulary (alias → JAX preset):

    f64      F64_F64_F64        true fp64
    f32      F32_F32_F32        true fp32 (no tf32 truncation)
    tf32     TF32_TF32_F32      1-pass tf32, fp32 accumulate
    tf32x3   TF32_TF32_F32_X3   3-pass tf32 ≈ fp32
    bf16     BF16_BF16_F32      bf16 inputs, fp32 accumulate
    bf16x3   BF16_BF16_F32_X3
    bf16x6   BF16_BF16_F32_X6   ≈ fp32
    bf16x9   BF16_BF16_F32_X9
    bf16_acc BF16_BF16_BF16     bf16 accumulate (low dynamic range)
    f16      F16_F16_F32        fp16 inputs, fp32 accumulate
    f16_acc  F16_F16_F16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset as _Preset

# short alias -> JAX preset. The alias is the canonical name we print and store
# in YAML; the preset is the machine-readable dot algorithm.
_ALIAS_TO_PRESET: dict[str, _Preset] = {
    "f64": _Preset.F64_F64_F64,
    "f32": _Preset.F32_F32_F32,
    "tf32": _Preset.TF32_TF32_F32,
    "tf32x3": _Preset.TF32_TF32_F32_X3,
    "bf16": _Preset.BF16_BF16_F32,
    "bf16x3": _Preset.BF16_BF16_F32_X3,
    "bf16x6": _Preset.BF16_BF16_F32_X6,
    "bf16x9": _Preset.BF16_BF16_F32_X9,
    "bf16_acc": _Preset.BF16_BF16_BF16,
    "f16": _Preset.F16_F16_F32,
    "f16_acc": _Preset.F16_F16_F16,
}
_PRESET_TO_ALIAS: dict[_Preset, str] = {v: k for k, v in _ALIAS_TO_PRESET.items()}

# Operand dtype passed to XLA. TF32 and split-BF16 are internal dot algorithms,
# not storage formats, so they receive FP32 operands.
_ALIAS_TO_ELEM: dict[str, Any] = {
    "f64": jnp.float64,
    "f32": jnp.float32,
    "tf32": jnp.float32,
    "tf32x3": jnp.float32,
    "bf16": jnp.bfloat16,
    # Split BF16 algorithms decompose FP32 inputs into BF16 fragments. Casting
    # to BF16 first would discard the residual fragments and defeat xN.
    "bf16x3": jnp.float32,
    "bf16x6": jnp.float32,
    "bf16x9": jnp.float32,
    "bf16_acc": jnp.bfloat16,
    "f16": jnp.float16,
    "f16_acc": jnp.float16,
}

# Modes whose element multiply happens at a truncated precision below the
# storage dtype (tf32 truncates fp32) — legal on the XLA/cuBLAS tensor-core
# matmul, meaningless for the Loom / cue kernels (which consume inputs at
# storage precision and never truncate).
_TRUNCATING = {"tf32", "tf32x3", "bf16x3", "bf16x6", "bf16x9"}

# alias -> cuBLAS compute-type string for the cuex `indexed_linear` grouped GEMM.
# There is no native <32-bit cuBLAS accumulate type we want, so bf16/f16 inputs
# accumulate in fp32 (COMPUTE_32F). The xN split algorithms have no cuBLAS
# compute type — they are XLA-only — and error in `cublas_compute`.
_ALIAS_TO_CUBLAS: dict[str, str] = {
    "f64": "CUBLAS_COMPUTE_64F",
    "f32": "CUBLAS_COMPUTE_32F",
    "tf32": "CUBLAS_COMPUTE_32F_FAST_TF32",
    "bf16": "CUBLAS_COMPUTE_32F",  # bf16 in, fp32 accumulate (no <32b accumulate we want)
    "f16": "CUBLAS_COMPUTE_32F",  # fp16 in, fp32 accumulate
    "f16_acc": "CUBLAS_COMPUTE_16F",  # fp16 in, fp16 accumulate
    # bf16_acc (bf16 accumulate) has no cuBLAS compute type -> cublas_compute errors.
}


@dataclass(frozen=True)
class Precision:
    """A single compute mode from the common vocabulary. Immutable and hashable,
    so it lives in an ``eqx.field(static=True)``. Build via :func:`parse`."""

    alias: str
    preset: _Preset

    def __repr__(self) -> str:
        return f"Precision({self.alias})"

    # ---- per-backend translations -------------------------------------------
    @property
    def accum(self) -> jnp.dtype:
        """Accumulator dtype — the ``math_dtype`` for Loom / cue kernels."""
        return jnp.dtype(self.preset.accumulation_type)

    @property
    def elem(self) -> jnp.dtype:
        """Operand dtype supplied to the backend.

        TF32 and split-BF16 presets receive FP32 so the backend performs their
        truncation/decomposition. Plain BF16 and FP16 receive narrowed operands.
        """
        return jnp.dtype(_ALIAS_TO_ELEM[self.alias])

    @property
    def truncates(self) -> bool:
        """True if the backend truncates or decomposes FP32 operands internally."""
        return self.alias in _TRUNCATING

    @property
    def xla_dot(self):
        """What to hand ``jnp.einsum(precision=...)`` / ``jnp.dot``.

        Almost always the ``DotAlgorithm`` preset — it is the explicit knob. The
        exception is ``f64``: XLA's GPU backend has no ``F64_F64_F64`` algorithm
        token (it lowers fp64 dots through the default path), so passing the
        preset raises "Unsupported dot precision algorithm". ``Precision.HIGHEST``
        on fp64 operands is exact fp64 and portable, so ``f64`` maps to it."""
        if self.alias == "f64":
            return jax.lax.Precision.HIGHEST
        return self.preset

    def cublas_compute(self) -> str:
        """cuBLAS compute-type string for the grouped-GEMM (indexed_linear) path."""
        try:
            return _ALIAS_TO_CUBLAS[self.alias]
        except KeyError:
            raise ValueError(
                f"precision {self.alias!r} has no cuBLAS compute type "
                f"(the xN split algorithms are XLA-only); pick one of "
                f"{sorted(_ALIAS_TO_CUBLAS)} for a grouped-GEMM op"
            ) from None

    def require_storage(self, activation_dtype: Any, where: str) -> jnp.dtype:
        """The activation dtype the op's inputs must already be in — always
        ``self.elem`` (``tf32`` operates on ``f32`` storage). Raises on mismatch
        rather than down-casting: activations change dtype only via a `cast` op.
        This is the single explicit dtype contract, uniform across matmul and
        kernel ops — you cannot accidentally run bf16 math on f32 data."""
        want = self.elem
        got = jnp.dtype(activation_dtype)
        if got != want:
            raise ValueError(
                f"{where}: precision {self.alias!r} expects {want.name} activations "
                f"but got {got.name}. Insert an explicit `cast` op — an op never "
                f"down-casts its activations silently."
            )
        return want

    def require_kernel_storage(self, activation_dtype: Any, where: str) -> jnp.dtype:
        """Like :meth:`require_storage`, but for the Loom / cue kernels, which
        consume inputs at storage precision and do not implement XLA's internal
        truncation/decomposition algorithms — so those modes are rejected."""
        if self.truncates:
            raise ValueError(
                f"{where}: precision {self.alias!r} transforms FP32 operands internally; that "
                f"mode is only valid on the XLA matmul, not this kernel. Use "
                f"f32/f64/bf16/f16 (name the storage precision explicitly)."
            )
        return self.require_storage(activation_dtype, where)


def parse(spec: Any) -> Precision:
    """Resolve a precision spec to the canonical :class:`Precision`.

    Accepts a short alias (``"tf32x3"``), a full JAX preset name
    (``"TF32_TF32_F32_X3"``), a :class:`jax.lax.DotAlgorithmPreset`, or an
    already-parsed :class:`Precision`. Anything else — including ``None`` — is a
    hard error naming the valid vocabulary, because a missing precision is
    exactly the bug this module exists to prevent."""
    if isinstance(spec, Precision):
        return spec
    if isinstance(spec, _Preset):
        alias = _PRESET_TO_ALIAS.get(spec)
        if alias is None:
            raise ValueError(f"preset {spec.name} is outside the SPLICE precision vocabulary")
        return Precision(alias, spec)
    if isinstance(spec, str):
        if spec in _ALIAS_TO_PRESET:
            return Precision(spec, _ALIAS_TO_PRESET[spec])
        # accept the verbose JAX preset name too
        for alias, preset in _ALIAS_TO_PRESET.items():
            if spec.upper() == preset.name:
                return Precision(alias, preset)
    raise ValueError(
        f"precision {spec!r} is not a recognised mode; use one of "
        f"{sorted(_ALIAS_TO_PRESET)} (short) or the JAX preset name"
    )


def encode(prec: Precision) -> str:
    """Canonical short alias, for YAML round-tripping."""
    return parse(prec).alias
