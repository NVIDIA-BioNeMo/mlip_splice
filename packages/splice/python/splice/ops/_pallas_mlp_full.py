# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete radial MLP fusion for small inference batches.

Each output-channel tile recomputes the narrow hidden stack. This exposes more
CTAs when there are too few rows to occupy the GPU. In backward, the Jacobian is
linear in the output cotangent once the activation residuals are fixed: each
output tile traverses the whole reverse stack, then its input-gradient partials
are summed. This deliberate recomputation is unsuitable for large batches.
"""

from functools import partial

import jax
import jax.numpy as jnp

from splice.ops import _pallas_mlp as p


def _full_config(num_rows, hidden_width, output_width, allow_tf32):
    """Row tile, output tile, warps, contraction block; H200 small-batch sweep."""
    # Keep the replicated grid near 128 CTAs as rows/output width change.
    # TF32 prefers at least 64 output channels in the small-batch sweep.
    columns = -(-(num_rows // 16) * output_width // 128)
    output_tile = 1 << (max(16, columns) - 1).bit_length()
    output_tile = min(256, max(64 if allow_tf32 else 16, output_tile))
    # Widths such as 768/1280 are valid. Never round down a trailing output tile.
    output_tile = min(output_tile, output_width & -output_width)
    warps = 8 if hidden_width == 128 or num_rows > 256 else 4
    return 16, output_tile, warps, 64 if hidden_width == 128 else 32


def _dot(a, weights, allow_tf32, block_k):
    if a.shape[1] >= 128 or weights.shape[1] >= 128:
        return p._blocked_dot(a, weights, allow_tf32, block_k)
    return p._dot(a, weights[...], allow_tf32=allow_tf32)


def _forward_kernel(x, w0, w1, w2, w3, *outputs, k, allow_tf32, block_k, act_last):
    y, acc0, acc1, acc2 = outputs[:4]
    z0 = _dot(x[...], w0, allow_tf32, block_k)

    @p.pl.when(p.pl.program_id(1) == 0)
    def store0():
        acc0[...] = z0

    z1 = _dot(k * z0 * p._sigmoid(z0), w1, allow_tf32, block_k)

    @p.pl.when(p.pl.program_id(1) == 0)
    def store1():
        acc1[...] = z1

    z2 = _dot(k * z1 * p._sigmoid(z1), w2, allow_tf32, block_k)

    @p.pl.when(p.pl.program_id(1) == 0)
    def store2():
        acc2[...] = z2

    z3 = _dot(k * z2 * p._sigmoid(z2), w3, allow_tf32, block_k)
    if act_last:
        outputs[4][...] = z3
        y[...] = k * z3 * p._sigmoid(z3)
    else:
        y[...] = z3


def _backward_kernel(dy, w3, acc2, w2, acc1, w1, acc0, w0, *tail, k, allow_tf32, block_k, act_last):
    if act_last:
        last, dx = tail
        z = last[...]
        s = p._sigmoid(z)
        dout = dy[...] * (k * (s + z * s * (1.0 - s)))
    else:
        (dx,) = tail
        dout = dy[...]
    dh = _dot(dout, w3, allow_tf32, block_k)
    z = acc2[...]
    s = p._sigmoid(z)
    dh = _dot(dh * (k * (s + z * s * (1.0 - s))), w2, allow_tf32, block_k)
    z = acc1[...]
    s = p._sigmoid(z)
    dh = _dot(dh * (k * (s + z * s * (1.0 - s))), w1, allow_tf32, block_k)
    z = acc0[...]
    s = p._sigmoid(z)
    dx[...] = _dot(dh * (k * (s + z * s * (1.0 - s))), w0, allow_tf32, block_k)[None, :, :]


def forward_full(x, weights, k, act_last, allow_tf32):
    m, k0 = x.shape
    hidden, output = weights[-1].shape
    bm, bn, warps, bk = _full_config(m, hidden, output, allow_tf32)
    pl = p.pl
    h_shape = jax.ShapeDtypeStruct((m, hidden), jnp.float32)
    y_shape = jax.ShapeDtypeStruct((m, output), jnp.float32)
    h_spec = pl.BlockSpec((bm, hidden), lambda i, j: (i, 0))
    y_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
    out_shapes = (y_shape, h_shape, h_shape, h_shape) + ((y_shape,) if act_last else ())
    out_specs = (y_spec, h_spec, h_spec, h_spec) + ((y_spec,) if act_last else ())
    y, *accs = pl.pallas_call(
        partial(_forward_kernel, k=k, allow_tf32=allow_tf32, block_k=bk, act_last=act_last),
        grid=(m // bm, output // bn),
        in_specs=[
            pl.BlockSpec((bm, k0), lambda i, j: (i, 0)),
            pl.BlockSpec((k0, hidden), lambda i, j: (0, 0)),
            pl.BlockSpec((hidden, hidden), lambda i, j: (0, 0)),
            pl.BlockSpec((hidden, hidden), lambda i, j: (0, 0)),
            pl.BlockSpec((hidden, bn), lambda i, j: (0, j)),
        ],
        out_specs=out_specs,
        out_shape=out_shapes,
        compiler_params=p.plgpu.CompilerParams(num_warps=warps, num_stages=2),
        name=f"radial_mlp_forward_full_chain3_m{bm}_n{bn}_w{warps}",
    )(x, *weights)
    return y, accs


def backward_full(dy, accs, weights_t, k, act_last, allow_tf32):
    m, output = dy.shape
    k0, hidden = weights_t[0].shape[1], weights_t[-1].shape[1]
    bm, bn, warps, bk = _full_config(m, hidden, output, allow_tf32)
    pl = p.pl
    h_spec = pl.BlockSpec((bm, hidden), lambda i, j: (i, 0))
    dy_spec = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
    inputs = (dy, weights_t[3], accs[2], weights_t[2], accs[1], weights_t[1], accs[0], weights_t[0])
    in_specs = [
        dy_spec,
        pl.BlockSpec((bn, hidden), lambda i, j: (j, 0)),
        h_spec,
        pl.BlockSpec((hidden, hidden), lambda i, j: (0, 0)),
        h_spec,
        pl.BlockSpec((hidden, hidden), lambda i, j: (0, 0)),
        h_spec,
        pl.BlockSpec((hidden, k0), lambda i, j: (0, 0)),
    ]
    if act_last:
        inputs += (accs[3],)
        in_specs.append(dy_spec)
    partials = pl.pallas_call(
        partial(_backward_kernel, k=k, allow_tf32=allow_tf32, block_k=bk, act_last=act_last),
        grid=(m // bm, output // bn),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, bm, k0), lambda i, j: (j, i, 0)),
        out_shape=jax.ShapeDtypeStruct((output // bn, m, k0), jnp.float32),
        compiler_params=p.plgpu.CompilerParams(num_warps=warps, num_stages=2),
        name=f"radial_mlp_backward_full_chain3_m{bm}_n{bn}_w{warps}",
    )(*inputs)
    return jnp.sum(partials, axis=0)
