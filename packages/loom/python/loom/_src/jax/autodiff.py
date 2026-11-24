# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import numpy as np


def jaxpr_to_fn(closed_jaxpr):
    def fn(*args):
        return tuple(jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args))

    return fn


def _sds(var):
    return jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)


def _make_partial(fn, inputs, active_idx):
    def partial_fn(*active):
        all_inputs = list(inputs)
        for i, k in enumerate(active_idx):
            all_inputs[k] = active[i]
        return fn(*all_inputs)

    return partial_fn, tuple(inputs[k] for k in active_idx)


def derive_jvp_jaxpr(jaxpr, nonzero_input_idx=None, needed_output_idx=None):
    invars = jaxpr.jaxpr.invars
    n_inputs = len(invars)

    if nonzero_input_idx is None:
        nonzero_input_idx = list(range(n_inputs))
    if needed_output_idx is None:
        needed_output_idx = list(range(len(jaxpr.jaxpr.outvars)))

    n_nz = len(nonzero_input_idx)
    fn = jaxpr_to_fn(jaxpr)

    def jvp_fn(*all_args):
        primals = all_args[:n_inputs]
        tangents_nz = all_args[n_inputs : n_inputs + n_nz]

        fn_partial, diff_primals = _make_partial(fn, primals, nonzero_input_idx)
        _, tangent_outputs = jax.jvp(fn_partial, diff_primals, tangents_nz)
        return tuple(tangent_outputs[k] for k in needed_output_idx)

    input_exs = [_sds(v) for v in invars]
    tangent_exs = [_sds(invars[k]) for k in nonzero_input_idx]
    return jax.make_jaxpr(jvp_fn)(*input_exs, *tangent_exs)


def derive_transpose_jaxpr(
    jaxpr, needed_input_idx=None, nonzero_ct_idx=None, cotangent_dtypes=None
):
    invars = jaxpr.jaxpr.invars
    outvars = jaxpr.jaxpr.outvars
    n_inputs = len(invars)
    n_outputs = len(outvars)

    if needed_input_idx is None:
        needed_input_idx = list(range(n_inputs))
    if nonzero_ct_idx is None:
        nonzero_ct_idx = list(range(n_outputs))
    if cotangent_dtypes is None:
        cotangent_dtypes = [outvars[k].aval.dtype for k in nonzero_ct_idx]

    defined_input_idx = [k for k in range(n_inputs) if k not in needed_input_idx]
    n_nz_ct = len(nonzero_ct_idx)
    fn = jaxpr_to_fn(jaxpr)

    zero_consts = tuple(
        np.zeros(invars[k].aval.shape, dtype=invars[k].aval.dtype) for k in needed_input_idx
    )

    def selected_fn(*inputs):
        outputs = fn(*inputs)
        return tuple(outputs[k] for k in nonzero_ct_idx)

    def transpose_fn(*args):
        ct_nz = tuple(
            ct.astype(outvars[k].aval.dtype)
            for ct, k in zip(args[:n_nz_ct], nonzero_ct_idx, strict=True)
        )
        defined_vals = args[n_nz_ct:]

        all_inputs = [None] * n_inputs
        for i, k in enumerate(defined_input_idx):
            all_inputs[k] = defined_vals[i]
        for i, k in enumerate(needed_input_idx):
            all_inputs[k] = zero_consts[i]

        fn_partial, diff_args = _make_partial(selected_fn, all_inputs, needed_input_idx)
        return jax.linear_transpose(fn_partial, *diff_args)(tuple(ct_nz))

    ct_exs = [
        jax.ShapeDtypeStruct(outvars[j].aval.shape, dtype)
        for j, dtype in zip(nonzero_ct_idx, cotangent_dtypes, strict=True)
    ]
    defined_exs = [_sds(invars[k]) for k in defined_input_idx]
    return jax.make_jaxpr(transpose_fn)(*ct_exs, *defined_exs)
