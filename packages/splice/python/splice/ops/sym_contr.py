# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MACE symmetric contraction via a pure-JAX ``loom`` program."""

from typing import Any

import e3
import e3.jax
import equinox as eqx
import jax
import jax.numpy as jnp
import loom
from e3 import Irrep, Irreps
from jax import Array

from splice.graph import Graph
from splice.precision import Precision
from splice.precision import parse as parse_precision
from splice.registry import register_op


def _path_counts(irreps_in, irreps_out, degrees):
    features = {ir: jax.ShapeDtypeStruct((ir.dim,), jnp.float32) for _, ir in irreps_in}
    return tuple(
        tuple(
            (ir, paths.shape[1])
            for ir, paths in jax.eval_shape(
                lambda values, degree=degree: e3.jax.symmetric_power(values, degree, irreps_out),
                features,
            ).items()
        )
        for degree in degrees
    )


@register_op("symmetric_contraction")
class SymmetricContraction(eqx.Module):
    """Per-species symmetric contraction, mapping node features to node features.

    Reads node features; outputs equivariant per-node tensors via a polynomial
    contraction on per-species weights. Weights are stored directly in the
    one-weight-per-path basis of ``e3.jax.symmetric_contraction``.
    """

    w: Array
    num_species: int = eqx.field(static=True)
    irreps_out: Irreps = eqx.field(static=True)
    SPECIES_COUNT_PARAM = "num_species"  # data species count; auto-filled from the card
    name: str = eqx.field(static=True)
    native_in: tuple = eqx.field(static=True)
    native_out: tuple = eqx.field(static=True)
    degrees: tuple[int, ...] = eqx.field(static=True)
    path_counts: tuple = eqx.field(static=True)
    # Multiplicative scale — a single float, or per output irrep
    # (`{"0e": 0.5, "1o": 1.3, ...}`, missing irreps default to 1; see
    # `IrrepsAttention.scale` — constants measured on real data, baked in the
    # arch YAML). SC is linear in `w`, so both forms are the same knob: the
    # float is applied to the weight tensor (keeps kernel intermediates bounded
    # in dynamic range for low precision), the dict at the output. Default:
    # auto-computed at __init__ to `1/sqrt(num_features)` so that SC's raw
    # output is bounded without changing model behavior at init (the post-SC
    # norm absorbs the rescaling). Pass `scale=1.0` explicitly to disable
    # (e.g. for µP coord-check tests in isolation).
    scale: float = eqx.field(static=True)
    out_scale: tuple[tuple[str, float], ...] = eqx.field(static=True)
    # Per-irrep output rescale, see `splice.calibration`. Default identity.
    cal_scale: tuple[tuple[str, float], ...] = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)
    init: str = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        correlation: int,
        num_species: int,
        num_features: int,
        scale: float | dict[str, float] | None = None,
        init: str = "normal",
        *,
        precision: Precision,
        name: str = "symmetric_contraction",
        param_dtype: Any,
        key: Array,
    ):
        self.num_species = num_species
        self.irreps_out = irreps_out
        self.name = name
        self.precision = parse_precision(precision)
        if init not in ("normal", "mace"):
            raise ValueError(f"symmetric_contraction init must be 'normal' or 'mace', got {init!r}")
        self.init = init
        # Default: 1/sqrt(num_features). The op owns this choice; configs
        # shouldn't have to know about it. A dict is hand-tuned absolute
        # amplitude — the auto default does not apply.
        if isinstance(scale, dict):
            self.scale = 1.0
            self.out_scale = tuple(
                (str(ir), float(scale.get(str(ir), 1.0))) for _, ir in irreps_out
            )
        else:
            self.scale = float(scale) if scale is not None else 1.0 / (num_features**0.5)
            self.out_scale = tuple((str(ir), 1.0) for _, ir in irreps_out)
        self.degrees = tuple(range(1, correlation + 1))
        native_inputs = irreps_in.set_mul(1)
        native_outputs = irreps_out.set_mul(1)
        self.native_in = tuple(ir for _, ir in native_inputs)
        self.native_out = tuple(ir for _, ir in native_outputs)
        self.path_counts = _path_counts(native_inputs, self.native_out, self.degrees)
        n_paths = sum(count for counts in self.path_counts for _, count in counts)
        weights = jax.random.normal(key, (num_species, n_paths, num_features), param_dtype)
        if self.init == "mace":
            path_scales = jnp.asarray(
                [
                    1.0 / count
                    for counts in self.path_counts
                    for _, count in counts
                    for _ in range(count)
                ],
                dtype=param_dtype,
            )
            weights = weights * path_scales[None, :, None]
        self.w = weights
        self.cal_scale = tuple((str(ir), 1.0) for _, ir in self.irreps_out)

    def __call__(self, node_feats: dict[Irrep, Array], graph: Graph) -> dict[Irrep, Array]:
        # Dtype-transparent: run the contraction in the carrier (`io`) dtype,
        # accumulate in `md` (default fp32). The weight premix einsum stays on the
        # fp32 masters (cheap, and the high-order polynomial is precision-sensitive
        # — see the SC `scale=1/sqrt(num_features)` dynamic-range guard); only the
        # premixed weight is cast to `io` at the kernel boundary.
        io = jax.tree.leaves(node_feats)[0].dtype
        self.precision.require_kernel_storage(io, "symmetric_contraction")
        md = self.precision.accum
        w = (self.w * self.scale).astype(io)
        x = {ir: value.astype(io) for ir, value in node_feats.items()}
        num_features = self.w.shape[2]
        node_operand = tuple(x[ir] for ir in self.native_in)

        def evaluate(values):
            flat_weights, features = values
            weights_by_degree = {}
            offset = 0
            for degree, counts in zip(self.degrees, self.path_counts, strict=True):
                size = sum(count for _, count in counts)
                weights_by_degree[degree] = e3.jax.unflatten_by_irrep(
                    flat_weights[offset : offset + size].astype(md), dict(counts)
                )
                offset += size
            result = e3.jax.symmetric_contraction(
                {ir: value.astype(md) for ir, value in zip(self.native_in, features, strict=True)},
                weights_by_degree,
            )
            return {
                ir: (result[ir] if ir in result else jnp.zeros((ir.dim,), dtype=md)).astype(io)
                for ir in self.native_out
            }

        y = loom.call(
            evaluate,
            (w, node_operand),
            u=num_features,
            domain_graph=graph.domains,
            input_domains=(
                loom.domains["species"],
                loom.domains["dst"],
            ),
            output_domains=loom.domains["dst"],
            compile_options=loom.Compile(),
        )
        cal = dict(self.cal_scale)
        out_scale = dict(self.out_scale)
        return {ir: y[ir] * out_scale[str(ir)] / cal[str(ir)] for _, ir in self.irreps_out}
