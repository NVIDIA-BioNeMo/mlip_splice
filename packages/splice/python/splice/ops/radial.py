# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bessel radial basis + variance-preserving radial MLP: the standard-MACE radial
front-end for the *unfused* channelwise interaction.

``radial_basis`` emits the Bessel features with a polynomial-cutoff envelope, each
feature calibrated to ~unit-std / zero-mean under a uniform-r assumption (a
dynamic-range-friendly contract for the downstream MLP). ``radial_mlp`` maps that
embedding to the per-edge tensor-product path weights that ``message_passing``
(with ``radial="external"``)
consumes (``OUTPUTS = "mp_weights"``); it is variance-preserving by construction —
each linear carries a ``1/sqrt(fan_in)`` scale and each activation is normalized to
unit second moment. Neither uses a Loom kernel.

``radial_mlp`` has an optional inference-only backend that runs the same math on
fused Pallas kernels; see :mod:`splice.ops._pallas_mlp` and the ``backend``
argument.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from splice.ops._common import polynomial_envelope
from splice.ops.activation import normalize_function
from splice.precision import Precision
from splice.precision import parse as parse_precision
from splice.registry import register_op


def _calibrate_bessel(r_max: float, num_basis: int, p: int, n_samples: int = 10_000):
    """Per-feature mean and std of the (bessel x envelope) basis, assuming r is
    uniform on ``[0.4 * r_max, r_max]``. Used to normalise the basis so the
    downstream MLP can rely on a "unit-std, zero-mean" edge-feature contract."""
    r_min = 0.4 * r_max
    r = np.linspace(r_min, r_max, n_samples, dtype=np.float64)
    u = r / r_max
    up = np.power(u, p)
    env = 1.0 - 0.5 * (p + 1) * (p + 2) * up + p * (p + 2) * up * u - 0.5 * p * (p + 1) * up * u * u
    env = np.where(r < r_max, env, 0.0)
    ns = np.arange(1, num_basis + 1, dtype=np.float64)
    # `np.sinc(x) = sin(pi*x) / (pi*x)` -- matches `jnp.sinc`.
    b = np.sqrt(2.0 / r_max) * np.pi * ns / r_max * np.sinc(ns * r[:, None] / r_max)
    b = b * env[:, None]
    return b.mean(axis=0).astype(np.float32), b.std(axis=0).astype(np.float32)


@register_op("radial_basis")
class RadialBasis(eqx.Module):
    """Bessel basis with polynomial envelope, each feature normalized to unit std
    and zero mean. Reads ``edge_vectors``; writes ``bessel``."""

    r_max: float = eqx.field(static=True)
    num_basis: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    apply_envelope: bool = eqx.field(static=True)
    _norm_mean: tuple[float, ...] = eqx.field(static=True)
    _norm_std: tuple[float, ...] = eqx.field(static=True)

    OUTPUTS = "bessel"
    CUTOFF_PARAM = "r_max"  # resolve_arch injects the card cutoff here

    def __init__(
        self,
        r_max: float,
        num_basis: int,
        p: int = 5,
        normalize: bool = True,
        apply_envelope: bool = True,
    ):
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.p = int(p)
        self.normalize = bool(normalize)
        self.apply_envelope = bool(apply_envelope)
        mean, std = _calibrate_bessel(self.r_max, self.num_basis, self.p)
        self._norm_mean = tuple(float(x) for x in mean)
        self._norm_std = tuple(float(x) for x in std)

    def __call__(self, edge_vectors: Array) -> Array:
        r = jnp.linalg.norm(edge_vectors, axis=-1)  # [num_edges]
        ns = jnp.arange(1, self.num_basis + 1, dtype=r.dtype)
        bessel = (
            jnp.sqrt(2.0 / self.r_max)
            * jnp.pi
            * ns
            / self.r_max
            * jnp.sinc(ns * r[..., None] / self.r_max)
        )  # [num_edges, num_basis]
        if self.apply_envelope:
            envelope = polynomial_envelope(r, self.r_max, self.p)
            bessel = bessel * envelope[..., None]
        if not self.normalize:
            return bessel
        mean = jnp.asarray(self._norm_mean, r.dtype)
        std = jnp.asarray(self._norm_std, r.dtype)
        out = (bessel - mean) / std
        if self.apply_envelope:
            # Re-impose the hard cut AFTER the mean shift so the basis vanishes past
            # the cutoff instead of saturating at the constant `-mean/std`. Bit-exact
            # no-op for r < r_max (mean/std are only ever used there).
            out = jnp.where((r < self.r_max)[..., None], out, 0.0)
        return out


#: The ``backend`` value that routes to the fused Pallas kernel. It is inference
#: only: it does not produce weight gradients (see :mod:`splice.ops._pallas_mlp`).
#: The name says so, because enabling it for training fails silently, by freezing
#: the MLP, rather than raising.
PALLAS_INFERENCE = "pallas_inference"

#: Precisions the fused backend implements: the two modes the Triton dot has. The
#: multi-pass presets are deliberately not offered. ``tf32x3`` was implemented and
#: measured, and it did not earn its keep: the fused chain cannot run the 3 passes
#: itself at a profit (89/99 us against fp32's 85/95 us per chain on an H100 at
#: 102k edges, with a *larger* error, 3.2e-6 against 3.5e-7, because these
#: 64x64x64 dots are latency-bound rather than FLOP-bound), so all it could do was
#: pass the X3 algorithm to the XLA output GEMM, which moved the whole force path
#: from 0.685 ms to 0.657 ms. That is not worth a third precision in this backend.
_PALLAS_PRECISIONS = frozenset({"f32", "tf32"})


@register_op("radial_mlp")
class RadialMLP(eqx.Module):
    """MLP ``bessel`` -> per-edge tensor-product weights, variance-preserving with
    optional RMS-only output normalization. Reads ``bessel``; writes ``mp_weights``.

    ``backend`` selects the implementation:

    ``xla``
        The default. Plain XLA matmuls; trains and deploys.
    ``pallas_inference``
        Hidden forward layers fuse GEMM + scale + activation, and backward folds
        activation derivatives into the reverse GEMMs. ``forward_fusion`` and
        ``backward_fusion`` select XLA, per-layer kernels, chained hidden kernels,
        complete fusion (``full``), or measured automatic heuristics. Large batches
        leave the final wide output GEMM in XLA. Complete fusion replicates hidden
        work across output tiles and is intended for small batches. Kernels tile
        in blocks of rows, so the edge count must be a multiple
        of :data:`~splice.ops._pallas_mlp.BLOCK_ROWS`; padding to reach a block was
        measured to cost more than the kernel saves.

        ``auto`` is tuned for energy+force inference. On Hopper, three equal hidden
        layers of width 64 or 128 use chained kernels at large edge counts; small
        FP32 batches can use complete fusion. Unmeasured GPU families retain their
        previous width-64 automatic policy. Other shapes use XLA forward and
        per-layer backward fusion. Energy-only callers
        can select ``forward_fusion="xla"`` or benchmark an explicit policy.

        ``precision`` is the largest lever on this op and the kernels retile for
        it (see :data:`~splice.ops._pallas_mlp._TF32_NUM_WARPS`), so it is worth
        choosing deliberately: at 102k edges the chained force path runs in 0.685
        ms at ``f32`` and 0.484 ms at ``tf32``, against 0.950 and 0.564 ms for the
        XLA backend at the same precision. ``tf32`` costs about 5e-4 relative
        error against ``f32``'s ~3e-7. Those two are the whole vocabulary here;
        see :data:`_PALLAS_PRECISIONS`.

    **The Pallas backend is inference only: it returns zero weight gradients.**
    Choosing it is a deployment decision; train with ``xla``.
    """

    linears: list[Array]
    activation: Callable = eqx.field(static=True)
    output_activation: bool = eqx.field(static=True)
    layer_sizes: tuple[int, ...] = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)
    normalize_output: bool = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    forward_fusion: str = eqx.field(static=True)
    backward_fusion: str = eqx.field(static=True)
    silu_k: float = eqx.field(static=True)

    OUTPUTS = "mp_weights"

    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        activation: Callable = jax.nn.silu,
        output_activation: bool = False,
        normalize_output: bool = False,
        backend: str = "xla",
        forward_fusion: str = "auto",
        backward_fusion: str = "auto",
        *,
        precision: Precision,
        param_dtype: Any,
        key: Array,
    ):
        self.activation = normalize_function(activation)
        self.output_activation = bool(output_activation)
        self.layer_sizes = tuple(layer_sizes)
        self.precision = parse_precision(precision)
        self.normalize_output = bool(normalize_output)
        if backend not in ("xla", PALLAS_INFERENCE):
            raise ValueError(
                f"radial_mlp backend must be one of {('xla', PALLAS_INFERENCE)}, got {backend!r}"
            )
        self.backend = backend
        self.forward_fusion = str(forward_fusion)
        self.backward_fusion = str(backward_fusion)
        # The kernels inline the activation, so its normalization constant has to be a
        # Python float: it cannot be concretized inside a trace. Resolving it here also
        # validates the activation and the layer widths at build time rather than at the
        # first call, and refuses rather than falling back to XLA, so a config that asks
        # for the kernels cannot quietly run without them.
        self.silu_k = self._check_pallas() if backend == PALLAS_INFERENCE else 0.0
        keys = jax.random.split(key, len(self.layer_sizes) - 1)
        self.linears = [
            jax.random.normal(k, (in_dim, out_dim), param_dtype)
            for k, in_dim, out_dim in zip(
                keys, self.layer_sizes[:-1], self.layer_sizes[1:], strict=True
            )
        ]

    def _check_pallas(self) -> float:
        """Validate that this configuration can run on the fused kernels, and return
        the activation's normalization constant. Raises instead of falling back.

        Configuration checks come first, then environment checks: a config that the
        kernels cannot express is wrong on every machine, and should say so even on a
        machine that could not have run it anyway."""
        from splice.ops._pallas_mlp import (
            BLOCK_WIDTH,
            HAVE_PALLAS,
            silu_norm_const,
            validate_fusion_options,
        )

        validate_fusion_options(self.forward_fusion, self.backward_fusion)
        if (
            self.forward_fusion in ("chain", "full") or self.backward_fusion in ("chain", "full")
        ) and not (
            len(self.layer_sizes) == 5 and self.layer_sizes[1:-1] in ((64, 64, 64), (128, 128, 128))
        ):
            raise ValueError(
                "chain fusion requires exactly three 64-wide or 128-wide hidden layers followed by "
                f"an unfused output GEMM; got layer sizes {self.layer_sizes}"
            )

        if self.precision.alias not in _PALLAS_PRECISIONS:
            raise ValueError(
                f"radial_mlp backend {self.backend!r} is an fp32-storage kernel and only "
                f"implements precisions {sorted(_PALLAS_PRECISIONS)}; got "
                f"{self.precision.alias!r}. The XLA backend covers the rest of the vocabulary."
            )
        bad = [n for n in self.layer_sizes[1:] if n % BLOCK_WIDTH]
        if bad:
            raise ValueError(
                f"radial_mlp backend {self.backend!r} needs every hidden and output width to be "
                f"a multiple of {BLOCK_WIDTH}; layer sizes {self.layer_sizes} include {bad}"
            )
        if not HAVE_PALLAS:
            raise RuntimeError(
                f"radial_mlp backend {self.backend!r} needs jax.experimental.pallas, which is "
                f"unavailable in this build; use backend='xla'"
            )
        # The Triton lowering resolves its target from a fixed table of device kinds
        # and raises mid-trace, far from the config that asked for the kernels, when
        # the card is not in it. Ask up front where we can, so the message can say
        # what to do. Older JAX exposes no such probe, in which case this is simply
        # skipped and JAX reports the problem itself at the first call.
        from jax.experimental.pallas import triton as plgpu

        probe = getattr(plgpu, "get_gpu_info", None)
        if probe is not None:
            try:
                probe()
            except ValueError as exc:
                raise RuntimeError(
                    f"radial_mlp backend {self.backend!r} cannot target this device: {exc}. "
                    f"The fused kernels lower through Triton, which supports a fixed set of "
                    f"GPUs; use backend='xla' here."
                ) from None
        return silu_norm_const(self.activation)

    def _require_aligned(self, num_edges: int) -> None:
        """The fused kernel tiles in whole blocks of rows and cannot mask a partial
        one, so the edge count has to be a multiple of the block.

        This refuses rather than working around it. Padding up and slicing back was
        implemented and measured: at 500,830 edges on an H100 it ran at 0.93 and 0.83
        against XLA, so the workaround costs more than the kernel saves. Falling back
        to XLA instead would be correct but silent, and a backend that quietly does
        not engage is worse than one that says the batch is wrong. The batching is
        ours to choose, so the fix belongs at the caller.
        """
        from splice.ops._pallas_mlp import BLOCK_ROWS

        if num_edges % BLOCK_ROWS:
            raise ValueError(
                f"radial_mlp backend {self.backend!r} tiles in blocks of {BLOCK_ROWS} edges "
                f"and this batch has {num_edges}. Size the edge bucket (`base_edges` in "
                f"`splice.padding`, default 4096) as a multiple of {BLOCK_ROWS}, or use "
                f"backend='xla'."
            )

    def _call_pallas(self, bessel: Array) -> Array:
        from splice.ops._pallas_mlp import fused_dense_mlp

        return fused_dense_mlp(
            bessel,
            list(self.linears),
            [(1.0 / n) ** 0.5 for n in self.layer_sizes[:-1]],
            self.silu_k,
            act_last=self.output_activation,
            allow_tf32=self.precision.alias == "tf32",
            xla_dot=self.precision.xla_dot,
            forward_fusion=self.forward_fusion,
            backward_fusion=self.backward_fusion,
        )

    def __call__(self, bessel: Array) -> Array:
        # Dtype-transparent: emit the incoming activation dtype `io`, matmul
        # accumulation set by the precision preset, RMS reduction in the accumulator
        # dtype `md`. The radial MLP is precision-sensitive; state `precision: f32`.
        io = bessel.dtype
        self.precision.require_storage(io, "radial_mlp")
        md = self.precision.accum
        if self.backend == PALLAS_INFERENCE:
            self._require_aligned(bessel.shape[0])
            x = self._call_pallas(bessel)
        else:
            x = bessel
            for i, w in enumerate(self.linears):
                in_dim = self.layer_sizes[i]
                y = jnp.matmul(
                    x, w.astype(io), precision=self.precision.xla_dot, preferred_element_type=io
                )
                x = (jnp.sqrt(1.0 / in_dim) * y).astype(io)
                if i < len(self.linears) - 1 or self.output_activation:
                    x = self.activation(x)
        if self.normalize_output:
            # Per-edge RMS over the feature axis (LayerNorm-style, RMS-only). Reduction
            # in `md` so a bf16 stream still gets an fp32-accurate RMS.
            x32 = x.astype(md)
            rms = jnp.sqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)
            x = (x32 / rms).astype(io)
        return x
