# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused inference kernels (Pallas/Triton) for the radial MLP.

**Inference only. This does not produce weight gradients.** The custom VJP returns
zeros for the weight cotangents, because computing them during a force evaluation
would cost about as much as the forward pass and would undo the speedup. Training
with it silently freezes the MLP. Nothing here can tell that it is being trained,
so the guard is at the op level: :class:`~splice.ops.radial.RadialMLP` reaches this
module only when its ``backend`` is set to an explicitly named inference value.
Train with the default XLA backend, then switch for deployment.

For large batches the final wide GEMM stays in XLA. Small batches can instead
fuse the complete MLP, replicating the hidden work across output tiles to expose
more parallelism; see :mod:`splice.ops._pallas_mlp_full`. Compiled HLO shows that
XLA emits each hidden forward layer as
a GEMM+scale fusion followed by a separate SiLU loop. On an H100, fusing
GEMM+scale+SiLU per layer is 1.04--1.24x faster at the standard 32k--262k edge
counts, and chaining the three 64-wide hidden layers is 1.09--1.32x faster. The
chained forward writes its pre-activations as residuals for force inference.

Backward first runs the wide output transpose GEMM in XLA, then either fuses the
activation derivative into each hidden GEMM separately or chains all three
64-wide reverse transforms in one kernel. The chained forward+backward path was
1.31--1.42x faster than XLA and 1.04--1.17x faster than the per-layer backward
at 32k--262k edges. H200 profiling extends the measured hidden width to 128 using blocked
contractions and separate forward/backward tiles. Explicit strategy overrides
remain available; automatic policies on unmeasured GPU families are preserved.

Shape contract, and one deliberate design decision:

* Widths. Every hidden and output width must be a multiple of ``_BN``; the input
  width is padded to a power of two, at least 16, inside this module.
* Rows. Pallas does not mask a partial trailing block, so the row count must be a
  multiple of ``_BM``. This module requires that of its caller rather than padding
  to it, and the reason is measured. Padding the input and slicing the output back
  is the obvious fix, but ``message_passing`` sizes its Loom operands against the
  compiled recipe and rejects an array with more rows than the edge domain, so the
  padding cannot simply be passed on, and the round trip costs more than the kernel
  saves: at 500,830 edges on an H100 the pad-and-slice version ran at 0.93 and 0.83
  against XLA on the two standard radial widths, where the same code on a
  block-aligned count runs at 1.09 to 1.17. So neither this module nor
  :class:`~splice.ops.radial.RadialMLP` works around an unaligned count; they refuse
  it, and the caller sizes its batches to suit.

  The edge bucket in :mod:`splice.padding` is ``base_edges``, whose default of 4096
  is a multiple of ``_BM``, but it is a caller-chosen number and nothing guarantees
  it, which is exactly why this is checked rather than assumed.

Precision. The Triton dot takes a backend default rather than plain fp32, and this
MLP is the precision-sensitive part of the model, so the caller passes
``allow_tf32`` explicitly instead of inheriting whatever the backend prefers.

Precision is also the largest single lever on this op, and it is not just a
throughput/accuracy trade: it moves the kernels between roofline regimes, so the
tiling has to follow it. The fp32 dot issues on the FMA pipe and the TF32 dot on
tensor cores, and the two want different warp counts by a factor of nearly two in
runtime, not a few percent -- see :data:`_TF32_NUM_WARPS`. For the whole force
path at 102k edges and output width 512 on an H100:

===========  ==============  ==================  ==============
precision    XLA backend     chained kernels     relative error
===========  ==============  ==================  ==============
``f32``      0.950 ms        0.685 ms            3e-7
``tf32``     0.564 ms        0.484 ms            5e-4
===========  ==============  ==================  ==============

For the large shapes above the wide output GEMM stays in XLA at both precisions,
and that was checked rather
than assumed: a Pallas version of it was written and swept over tiles, warps and
stages, and it lost every time (239 against 198 us at fp32, 185 against 125 at
tf32). These are large, regular GEMMs, which is what XLA's autotuner is for; the
narrow chained stack is where a custom kernel still pays.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    # `dot` lives in the Triton namespace on newer JAX and at the Pallas top level
    # on older ones, and the repo supports both. Bind whichever this version has
    # rather than pinning a spelling that breaks half the supported range.
    _dot = getattr(plgpu, "dot", None) or pl.dot
    HAVE_PALLAS = True
except ImportError:  # pragma: no cover - Pallas/Triton unavailable (e.g. CPU-only build)
    HAVE_PALLAS = False

_BM = 64  # rows (edges) per block
_BN = 64  # feature-axis granularity the widths must respect

#: Public aliases for the shape constraints callers have to satisfy.
BLOCK_ROWS = _BM
BLOCK_WIDTH = _BN
_NUM_WARPS = 8  # dominant knob for these tall, narrow fp32 GEMMs
# The TF32 dot runs on tensor cores instead of the FMA pipe, which changes the
# best warp count outright rather than by a few percent: at 102k edges the chain
# forward/backward take 61/71 us at 4 warps against 110/126 us at 8, and 146/160
# against 265/307 at 262k. Tuning one number for both precisions made `tf32`
# slower than `f32` (110 us against 85 us), which defeats the point of asking for
# it, so the two paths carry their own value. Measured on H100.
_TF32_NUM_WARPS = 4
_NUM_STAGES = 2
_SMALL_CHAIN_MAX_ROWS = 4096
_SMALL_CHAIN_BM = 16
_SMALL_CHAIN_WARPS = 4

FORWARD_FUSIONS = ("auto", "xla", "per_layer", "chain", "full")
BACKWARD_FUSIONS = ("auto", "xla", "per_layer", "chain", "full")

# Measured on H100 for the standard 8→64→64→64→{512,1280} radial MLP.
# Chain fusion wins the complete energy+force path from 4k through 262k rows.
# A tiny energy-only forward can still prefer XLA; select forward_fusion="xla"
# explicitly for that workload. ``auto`` is force-inference tuned because that
# is the purpose of this zero-weight-gradient backend.
CHAIN_WIDTH = 64
CHAIN_HIDDEN_LAYERS = 3


def _layer_warps(allow_tf32: bool) -> int:
    """Warps per block for one fused hidden GEMM, by dot precision."""
    return _TF32_NUM_WARPS if allow_tf32 else _NUM_WARPS


def _is_hopper(device_kind=None):
    if device_kind is None:
        device_kind = jax.devices()[0].device_kind
    return "H100" in device_kind or "H200" in device_kind


def _chain_config(
    num_rows: int,
    allow_tf32: bool = False,
    *,
    hidden_width: int = 64,
    direction: str = "forward",
) -> tuple[int, int]:
    """Row tile and warps for the hidden stack (H200 device-trace sweep).

    Forward and backward have different register lifetimes and load patterns.
    The 8k TF32 forward benefits from more CTAs; extending the 16-row tile to
    backward instead makes that workload slower. Width 128 additionally needs
    contraction blocking; a row-tile change alone leaves severe FP32 slowdowns.
    """
    if hidden_width == 128:
        if not allow_tf32:
            if num_rows <= 4096:
                return 16, 4
            return (64, 8) if direction == "forward" else (32, 4)
        if direction == "forward":
            if num_rows <= 1024:
                return 16, 8
            if num_rows <= 8192:
                return 32, 8
            return 32, 4
        if num_rows <= 2048:
            return 16, 8
        return (32, 4) if num_rows <= 16384 else (32, 8)
    if num_rows <= _SMALL_CHAIN_MAX_ROWS:
        return _SMALL_CHAIN_BM, _SMALL_CHAIN_WARPS
    if allow_tf32 and direction == "forward" and _is_hopper():
        if num_rows <= 8192:
            return 32, 8
        if num_rows <= 16384:
            return 32, 4
    return _BM, _layer_warps(allow_tf32)


def _chain_block_k(num_rows, hidden_width, allow_tf32, direction):
    """Zero selects the original dot; nonzero bounds its contraction live range."""
    if hidden_width != 128:
        return 0
    if not allow_tf32:
        return 32
    if direction == "backward":
        return 64
    if num_rows <= 4096:
        return 0
    return 64 if num_rows <= 16384 else 32


def _blocked_dot(a, w_ref, allow_tf32, block_k, transpose_weights=False):
    """Keep only one weight contraction block live inside an explicit loop.

    Pallas/Triton does not lower dynamic_slice on register arrays. A masked
    reduction selects the register-resident input group without a memory round
    trip. Nonselected groups contribute exact zeros. This trades a few selects
    and additions for bounded weight liveness, which matters for width-128 FP32.
    """
    if transpose_weights:
        # The measured width-64 TF32 path uses a native contraction. Presenting
        # B in this layout improves both directions in the large-edge device sweep.
        if block_k:
            raise ValueError("transposed weight layout requires an unblocked contraction")
        return _dot(a, w_ref[...], trans_b=True, allow_tf32=allow_tf32)
    m, k_in = a.shape
    if not block_k or k_in <= block_k:
        return _dot(a, w_ref[...], allow_tf32=allow_tf32)
    if k_in % block_k:
        raise ValueError("blocked MLP contraction must be divisible by block_k")
    groups = k_in // block_k

    def body(i, acc):
        lhs = jnp.sum(
            jnp.where(
                jnp.arange(groups)[None, :, None] == i,
                a.reshape(m, groups, block_k),
                0.0,
            ),
            axis=1,
        )
        rhs = w_ref[pl.dslice(i * block_k, block_k), :]
        return acc + _dot(lhs, rhs, allow_tf32=allow_tf32)

    return jax.lax.fori_loop(0, groups, body, jnp.zeros((m, w_ref.shape[1]), jnp.float32))


def _transpose_chain_weights(num_rows, hidden_width, input_width, allow_tf32):
    """H200 sweep: tensor-dot weight layout for large edge batches.

    Both directions improve at 250k and 2M edges. FP32 regresses with this
    layout, so it retains the original contraction. Wider inputs and other GPU
    families retain their existing path until measured.
    """
    return (
        allow_tf32
        and hidden_width == 64
        and input_width == 16
        and num_rows >= 65536
        and _is_hopper()
    )


def _pad_chain_input(x):
    # Pallas/Triton cannot lower lax.pad on register arrays. A broadcast/select
    # creates the upper zero half without a global padded input allocation.
    if x.shape[1] == 8:
        return jnp.where(
            jnp.arange(2)[None, :, None] == 0,
            x.reshape(x.shape[0], 1, 8),
            0.0,
        ).reshape(x.shape[0], 16)
    return x


def _store_chain_gradient(dx, output_width):
    if dx.shape[1] == output_width:
        return dx
    # Select the first eight columns without an unsupported register-array slice.
    return jnp.sum(
        jnp.where(
            jnp.arange(2)[None, :, None] == 0,
            dx.reshape(dx.shape[0], 2, output_width),
            0.0,
        ),
        axis=1,
    )


def silu_norm_const(activation) -> float:
    """The scalar ``k`` with ``activation(z) == k * silu(z)``.

    The kernel inlines the activation's derivative, so it needs the normalization
    factor that :func:`splice.ops.activation.normalize_function` folds in as a
    number. Sampled at a few points and checked for constancy, so a non-SiLU or
    non-scalar-multiple activation is rejected instead of being computed wrongly.
    """
    z = jnp.array([0.7, 1.9, -1.3], jnp.float32)
    ratio = jnp.asarray(activation(z) / jax.nn.silu(z))
    k = float(jnp.median(ratio))
    if not bool(jnp.allclose(ratio, k, rtol=1e-4)):
        raise ValueError(
            "the fused MLP backend inlines a normalized SiLU and needs an activation of "
            "the form k*silu(z); got one whose ratio to silu is not constant"
        )
    return k


def _sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


# ---- forward: GEMM + normalized SiLU, with pre-activations returned for the VJP ----
def _forward_kernel(x_ref, w_ref, h_ref, acc_ref, *, k: float, allow_tf32: bool):
    acc = _dot(x_ref[...], w_ref[...], allow_tf32=allow_tf32)
    # Store the residual immediately to shorten its live range. The activation
    # remains in registers and is written once for the next layer.
    acc_ref[...] = acc
    h_ref[...] = k * acc * _sigmoid(acc)


def _forward_gemm(x, w, k, allow_tf32):
    m, k_in = x.shape
    n_out = w.shape[1]
    shape = jax.ShapeDtypeStruct((m, n_out), jnp.float32)
    out_spec = pl.BlockSpec((_BM, n_out), lambda i: (i, 0))
    return pl.pallas_call(
        partial(_forward_kernel, k=k, allow_tf32=allow_tf32),
        grid=(m // _BM,),
        in_specs=[
            pl.BlockSpec((_BM, k_in), lambda i: (i, 0)),
            pl.BlockSpec((k_in, n_out), lambda i: (0, 0)),
        ],
        out_specs=(out_spec, out_spec),
        out_shape=(shape, shape),
        compiler_params=plgpu.CompilerParams(
            num_warps=_layer_warps(allow_tf32), num_stages=_NUM_STAGES
        ),
        name="radial_mlp_forward_gemm_silu",
    )(x, w)


def _forward_chain3_kernel(
    x_ref,
    w0_ref,
    w1_ref,
    w2_ref,
    h_ref,
    acc0_ref,
    acc1_ref,
    acc2_ref,
    *,
    k: float,
    allow_tf32: bool,
    block_k: int = 0,
    transpose_weights: bool = False,
):
    acc0 = _blocked_dot(
        _pad_chain_input(x_ref[...]), w0_ref, allow_tf32, block_k, transpose_weights
    )
    acc0_ref[...] = acc0
    h0 = k * acc0 * _sigmoid(acc0)

    acc1 = _blocked_dot(h0, w1_ref, allow_tf32, block_k, transpose_weights)
    acc1_ref[...] = acc1
    h1 = k * acc1 * _sigmoid(acc1)

    acc2 = _blocked_dot(h1, w2_ref, allow_tf32, block_k, transpose_weights)
    acc2_ref[...] = acc2
    h_ref[...] = k * acc2 * _sigmoid(acc2)


def _forward_chain3(x, weights, k, allow_tf32):
    """Fuse the standard three 64-wide hidden layers into one launch.

    The wide output GEMM is deliberately not part of this kernel. All three
    pre-activations are returned because force inference consumes them in the
    custom VJP.
    """
    m, k_in = x.shape
    n = weights[0].shape[1]
    block_rows, num_warps = _chain_config(m, allow_tf32, hidden_width=n)
    transpose_weights = _transpose_chain_weights(m, n, weights[0].shape[0], allow_tf32)
    operands = tuple(w.T for w in weights) if transpose_weights else weights
    shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    out_spec = pl.BlockSpec((block_rows, n), lambda i: (i, 0))
    return pl.pallas_call(
        partial(
            _forward_chain3_kernel,
            k=k,
            allow_tf32=allow_tf32,
            block_k=_chain_block_k(m, n, allow_tf32, "forward"),
            transpose_weights=transpose_weights,
        ),
        grid=(m // block_rows,),
        in_specs=[
            pl.BlockSpec((block_rows, k_in), lambda i: (i, 0)),
            pl.BlockSpec(operands[0].shape, lambda i: (0, 0)),
            pl.BlockSpec((n, n), lambda i: (0, 0)),
            pl.BlockSpec((n, n), lambda i: (0, 0)),
        ],
        out_specs=(out_spec, out_spec, out_spec, out_spec),
        out_shape=(shape, shape, shape, shape),
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=_NUM_STAGES),
        name=f"radial_mlp_forward_hidden_chain3_m{block_rows}_w{num_warps}"
        + ("_transposed" if transpose_weights else ""),
    )(x, operands[0], operands[1], operands[2])


# ---- one inner layer of the backward: (dout * silu'(acc)) @ W^T, activation fused in ----
def _backward_kernel(dout_ref, acc_ref, wt_ref, da_ref, *, k: float, allow_tf32: bool):
    acc = acc_ref[...]
    s = _sigmoid(acc)
    dact = k * (s + acc * s * (1.0 - s))
    da_ref[...] = _dot(dout_ref[...] * dact, wt_ref[...], allow_tf32=allow_tf32)


def _backward_gemm(dout, acc, wt, k, allow_tf32):
    m, n_out = dout.shape
    k_in = wt.shape[1]
    return pl.pallas_call(
        partial(_backward_kernel, k=k, allow_tf32=allow_tf32),
        # Row count is validated as a multiple of _BM at the entry point, and the
        # BlockSpec tiles are unmasked, so a rounded-up grid would read and write
        # out of bounds rather than handle a partial block. Divide exactly.
        grid=(m // _BM,),
        in_specs=[
            pl.BlockSpec((_BM, n_out), lambda i: (i, 0)),
            pl.BlockSpec((_BM, n_out), lambda i: (i, 0)),
            pl.BlockSpec((n_out, k_in), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((_BM, k_in), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, k_in), jnp.float32),
        compiler_params=plgpu.CompilerParams(
            num_warps=_layer_warps(allow_tf32), num_stages=_NUM_STAGES
        ),
        # Named so that a profiler can select it: `ncu --kernel-name
        # regex:radial_mlp_backward_gemm_silu`. Without a name it appears under a
        # generated symbol and cannot be filtered reliably across builds.
        name="radial_mlp_backward_gemm_silu",
    )(dout, acc, wt)


def _backward_chain3_kernel(
    dout_ref,
    acc2_ref,
    wt2_ref,
    acc1_ref,
    wt1_ref,
    acc0_ref,
    wt0_ref,
    dx_ref,
    *,
    k: float,
    allow_tf32: bool,
    block_k: int = 0,
    transpose_weights: bool = False,
):
    acc2 = acc2_ref[...]
    s2 = _sigmoid(acc2)
    d2 = k * (s2 + acc2 * s2 * (1.0 - s2))
    dh = _blocked_dot(dout_ref[...] * d2, wt2_ref, allow_tf32, block_k, transpose_weights)

    acc1 = acc1_ref[...]
    s1 = _sigmoid(acc1)
    d1 = k * (s1 + acc1 * s1 * (1.0 - s1))
    dh = _blocked_dot(dh * d1, wt1_ref, allow_tf32, block_k, transpose_weights)

    acc0 = acc0_ref[...]
    s0 = _sigmoid(acc0)
    d0 = k * (s0 + acc0 * s0 * (1.0 - s0))
    dx_ref[...] = _store_chain_gradient(
        _blocked_dot(dh * d0, wt0_ref, allow_tf32, block_k, transpose_weights), dx_ref.shape[1]
    )


def _backward_chain3(dout, accs, weights_t, k, allow_tf32, *, output_width=None):
    m, n = dout.shape
    k0 = weights_t[0].shape[1]
    output_width = k0 if output_width is None else output_width
    block_rows, num_warps = _chain_config(m, allow_tf32, hidden_width=n, direction="backward")
    transpose_weights = _transpose_chain_weights(m, n, k0, allow_tf32)
    operands = tuple(w.T for w in weights_t) if transpose_weights else weights_t
    return pl.pallas_call(
        partial(
            _backward_chain3_kernel,
            k=k,
            allow_tf32=allow_tf32,
            block_k=_chain_block_k(m, n, allow_tf32, "backward"),
            transpose_weights=transpose_weights,
        ),
        grid=(m // block_rows,),
        in_specs=[
            pl.BlockSpec((block_rows, n), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n), lambda i: (i, 0)),
            pl.BlockSpec((n, n), lambda i: (0, 0)),
            pl.BlockSpec((block_rows, n), lambda i: (i, 0)),
            pl.BlockSpec((n, n), lambda i: (0, 0)),
            pl.BlockSpec((block_rows, n), lambda i: (i, 0)),
            pl.BlockSpec(operands[0].shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, output_width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, output_width), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=_NUM_STAGES),
        name=f"radial_mlp_backward_hidden_chain3_m{block_rows}_w{num_warps}"
        + ("_transposed" if transpose_weights else ""),
    )(
        dout,
        accs[2],
        operands[2],
        accs[1],
        operands[1],
        accs[0],
        operands[0],
    )


def _layers_xla(x, weights, k, act_last, xla_dot):
    """The XLA MLP and the hidden pre-activations its backward reads."""
    h = x
    last = len(weights) - 1
    accs = []
    for i, w in enumerate(weights):
        z = jnp.matmul(h, w, precision=xla_dot, preferred_element_type=jnp.float32)
        if i < last or act_last:
            accs.append(z)
        h = (k * z * _sigmoid(z)) if (i < last or act_last) else z
    return h, accs


def _layers_per_layer(x, weights, k, act_last, allow_tf32, xla_dot):
    """Fuse GEMM + normalized SiLU one hidden layer at a time."""
    h = x
    accs = []
    for w in weights[:-1]:
        h, acc = _forward_gemm(h, w, k, allow_tf32)
        accs.append(acc)
    z = jnp.matmul(h, weights[-1], precision=xla_dot, preferred_element_type=jnp.float32)
    if act_last:
        accs.append(z)
        z = k * z * _sigmoid(z)
    return z, accs


def _layers_chain3(x, weights, k, act_last, allow_tf32, xla_dot):
    """Fuse three 64-wide hidden layers; leave the wide output GEMM in XLA."""
    h, acc0, acc1, acc2 = _forward_chain3(x, weights[:3], k, allow_tf32)
    z = jnp.matmul(h, weights[-1], precision=xla_dot, preferred_element_type=jnp.float32)
    accs = [acc0, acc1, acc2]
    if act_last:
        accs.append(z)
        z = k * z * _sigmoid(z)
    return z, accs


def _can_chain3(weights) -> bool:
    if len(weights) != CHAIN_HIDDEN_LAYERS + 1:
        return False
    hidden = weights[:-1]
    return (
        hidden[0].shape[1] in (64, 128)
        and all(w.shape == (hidden[0].shape[1], hidden[0].shape[1]) for w in hidden[1:])
        and weights[-1].shape[0] == hidden[0].shape[1]
    )


def validate_fusion_options(forward_fusion: str, backward_fusion: str) -> None:
    if forward_fusion not in FORWARD_FUSIONS:
        raise ValueError(f"forward_fusion must be one of {FORWARD_FUSIONS}, got {forward_fusion!r}")
    if backward_fusion not in BACKWARD_FUSIONS:
        raise ValueError(
            f"backward_fusion must be one of {BACKWARD_FUSIONS}, got {backward_fusion!r}"
        )


def resolve_fusions(
    weights,
    num_rows,
    forward_fusion,
    backward_fusion,
    *,
    device_kind=None,
    allow_tf32=False,
    original_input_width=None,
):
    """Resolve policies using the measured shape and device family.

    Hopper uses complete fusion for measured tiny FP32 standard-input batches and
    hidden-chain fusion otherwise, at widths 64 and 128. Full fusion can lose CUDA
    graph capture as reduction/output shapes change, so its automatic range is
    restricted to the validated 64-row, input-8, output-512 through 1280 shapes.
    RTX 6000 Ada uses the chain for standard-input large-edge batches with
    output widths 256 through 1280, validated at 100k through 2M rows. Other
    devices retain the pre-existing automatic width-64 policy. Explicit modes
    are available on every supported device and preserve the same numerical API.
    """
    validate_fusion_options(forward_fusion, backward_fusion)
    can_chain = _can_chain3(weights)
    if device_kind is None:
        device_kind = jax.devices()[0].device_kind
    hopper = _is_hopper(device_kind)
    ada_large_chain = (
        can_chain
        and device_kind == "NVIDIA RTX 6000 Ada Generation"
        and original_input_width == 8
        and num_rows >= 100_000
        and 256 <= weights[-1].shape[1] <= 1280
    )
    measured = can_chain and (hopper or weights[0].shape[1] == 64 or ada_large_chain)
    small_full = (
        measured
        and hopper
        and not allow_tf32
        and original_input_width == 8
        and 512 <= weights[-1].shape[1] <= 1280
        and num_rows == 64
    )
    if forward_fusion == "auto":
        forward_fusion = "full" if small_full else ("chain" if measured else "xla")
    if backward_fusion == "auto":
        backward_fusion = "full" if small_full else ("chain" if measured else "per_layer")
    if (
        any(mode in ("chain", "full") for mode in (forward_fusion, backward_fusion))
        and not can_chain
    ):
        shapes = tuple(w.shape for w in weights)
        raise ValueError(
            "chain fusion requires exactly three 64-wide or 128-wide hidden layers followed by "
            f"an output GEMM; got weight shapes {shapes}"
        )
    return forward_fusion, backward_fusion


def _layers(x, weights, k, act_last, allow_tf32, xla_dot, forward_fusion):
    if forward_fusion == "xla":
        return _layers_xla(x, weights, k, act_last, xla_dot)
    if forward_fusion == "per_layer":
        return _layers_per_layer(x, weights, k, act_last, allow_tf32, xla_dot)
    if forward_fusion == "full":
        from splice.ops._pallas_mlp_full import forward_full

        return forward_full(x, weights, k, act_last, allow_tf32)
    if forward_fusion == "chain":
        return _layers_chain3(x, weights, k, act_last, allow_tf32, xla_dot)
    raise AssertionError(f"unresolved forward fusion {forward_fusion!r}")


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _mlp(
    x,
    weights,
    k,
    act_last,
    allow_tf32,
    xla_dot,
    forward_fusion,
    backward_fusion,
    input_width,
):
    del backward_fusion, input_width
    return _layers(x, weights, k, act_last, allow_tf32, xla_dot, forward_fusion)[0]


def _mlp_fwd(
    x,
    weights,
    k,
    act_last,
    allow_tf32,
    xla_dot,
    forward_fusion,
    backward_fusion,
    input_width,
):
    del backward_fusion, input_width
    h, accs = _layers(x, weights, k, act_last, allow_tf32, xla_dot, forward_fusion)
    return h, (tuple(accs), weights)


def _mlp_bwd(
    k,
    act_last,
    allow_tf32,
    xla_dot,
    forward_fusion,
    backward_fusion,
    input_width,
    res,
    dy,
):
    del forward_fusion
    accs, weights = res
    last = len(weights) - 1
    wt = [w.T for w in weights]
    if backward_fusion == "full":
        from splice.ops._pallas_mlp_full import backward_full

        dx = backward_full(dy, accs, wt, k, act_last, allow_tf32)
        return dx, tuple(jnp.zeros_like(w) for w in weights)
    # Final layer: its contraction is the output width, which is wide, so XLA's GEMM
    # is the right tool, and there is no activation to fuse unless act_last.
    if act_last:
        acc = accs[last]
        s = _sigmoid(acc)
        dact = k * (s + acc * s * (1.0 - s))
        dh = jnp.matmul(dy * dact, wt[last], precision=xla_dot, preferred_element_type=jnp.float32)
    else:
        dh = jnp.matmul(dy, wt[last], precision=xla_dot, preferred_element_type=jnp.float32)
    if backward_fusion == "chain":
        dh = _backward_chain3(dh, accs[:3], wt[:3], k, allow_tf32, output_width=input_width)
    else:
        # Inner layers: narrow contraction, activation derivative fused into each
        # GEMM, or expressed as XLA for an explicit comparison/fallback.
        for i in range(last - 1, -1, -1):
            acc = accs[i]
            if backward_fusion == "per_layer":
                dh = _backward_gemm(dh, acc, wt[i], k, allow_tf32)
            elif backward_fusion == "xla":
                s = _sigmoid(acc)
                dact = k * (s + acc * s * (1.0 - s))
                dh = jnp.matmul(
                    dh * dact,
                    wt[i],
                    precision=xla_dot,
                    preferred_element_type=jnp.float32,
                )
            else:
                raise AssertionError(f"unresolved backward fusion {backward_fusion!r}")
    # Weight cotangents are zero by construction: see the module docstring. In the
    # force path they are dead and XLA removes them; they exist only to satisfy the
    # custom-VJP signature.
    return dh, tuple(jnp.zeros_like(w) for w in weights)


_mlp.defvjp(_mlp_fwd, _mlp_bwd)


def fused_dense_mlp(
    x: Array,
    weights: list[Array],
    scales: list[float],
    k: float,
    *,
    act_last: bool = False,
    allow_tf32: bool = False,
    xla_dot: Any = None,
    forward_fusion: str = "auto",
    backward_fusion: str = "auto",
) -> Array:
    """A per-layer ``linear -> scale -> normalized SiLU`` MLP whose backward runs on
    the fused kernel. Inference only.

    Args:
        x: ``[M, K0]`` fp32 activations.
        weights: per-layer ``[in, out]`` fp32 matrices.
        scales: per-layer ``1/sqrt(fan_in)``, folded into the weights here. With
            frozen weights this fold is constant-folded away at compile time.
        k: the normalized-SiLU constant from :func:`silu_norm_const`.
        act_last: also apply the activation after the final layer.
        allow_tf32: allow TF32 in the Triton dot.
        xla_dot: precision token for the XLA matmuls (the whole forward, and the
            final layer of the backward).
        forward_fusion: ``auto``, ``xla``, ``per_layer``, ``chain`` or ``full``.
            ``full`` also fuses the output GEMM and is intended for small batches.
        backward_fusion: ``auto``, ``xla``, ``per_layer``, ``chain`` or ``full``.

    Returns:
        ``[M, N]``.

    Raises:
        ValueError: if ``M`` is not a multiple of :data:`BLOCK_ROWS`. Callers pick
            the path; see the module docstring for why this does not pad.
    """
    if not HAVE_PALLAS:
        raise RuntimeError("the fused MLP backend requires jax.experimental.pallas (GPU build)")
    validate_fusion_options(forward_fusion, backward_fusion)
    m, k0 = x.shape
    if m % _BM:
        raise ValueError(
            f"the fused MLP backend tiles in blocks of {_BM} rows and does not pad; got {m}"
        )
    folded = [(w * s).astype(jnp.float32) for w, s in zip(weights, scales, strict=True)]
    for i, w in enumerate(folded[1:], 1):
        if w.shape[0] % _BN or w.shape[1] % _BN:
            raise ValueError(
                f"the fused MLP backend needs hidden and output widths that are multiples "
                f"of {_BN}; layer {i} is {w.shape}"
            )
    forward_fusion, backward_fusion = resolve_fusions(
        folded,
        m,
        forward_fusion,
        backward_fusion,
        allow_tf32=allow_tf32,
        original_input_width=k0,
    )
    # H200 sweep, 100k--2M edges: keep the standard eight input columns and
    # zero-extend them in registers. Both VJP directions must use the chain;
    # mixed policies retain the original shared padded representation.
    register_padding = (
        k0 == 8
        and m >= 100_000
        and allow_tf32
        and folded[0].shape[1] == 64
        and forward_fusion == backward_fusion == "chain"
        and _is_hopper()
    )
    kp = max(16, 1 << (k0 - 1).bit_length())
    if kp != k0:
        if not register_padding:
            x = jnp.pad(x, ((0, 0), (0, kp - k0)))
        folded[0] = jnp.pad(folded[0], ((0, kp - k0), (0, 0)))
    return _mlp(
        x,
        tuple(folded),
        float(k),
        bool(act_last),
        bool(allow_tf32),
        xla_dot,
        forward_fusion,
        backward_fusion,
        x.shape[1],
    )
