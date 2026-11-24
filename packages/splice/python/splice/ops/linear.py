# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-irrep linear maps: dense (IrrepsLinear) and per-species-indexed (IrrepsIndexedLinear).

IrrepsLinear: `is_readout=True` marks readout layers, applying 1/sqrt(mul_in)
multiplier on top of 1/sqrt(fan_in) init scaling (net 1/fan_in, matching µP rule
for output layers). µP LR multiplier is 1/fan_in for both hidden and readout
(Adam-µP rule); they differ only in init/forward output magnitude.

IrrepsIndexedLinear: Three backends (``method=``): ``indexed_linear`` (cuBLAS
grouped GEMM, fastest; requires species-contiguous nodes), ``loom_map``
(cue-free, slow backward ~20-70× cuBLAS), ``naive`` (pure XLA, for validation).
All share the same flat block-diagonal weight layout (irreps order, [mul_in, mul_out]
per matching irrep, scaled by scale/sqrt(num_indices)/sqrt(mul_in)).
"""

import math
from collections.abc import Mapping
from typing import Any, Literal

import cuequivariance as cue
import cuequivariance_jax as cuex
import e3
import equinox as eqx
import jax
import jax.numpy as jnp
from e3 import Irrep, Irreps
from jax import Array

from splice.ops._common import (
    flat_to_ir_mul,
    indexed_linear_map,
    native_dict_to_flat,
)
from splice.ops._initializers import initialize_linear
from splice.precision import Precision
from splice.precision import parse as parse_precision
from splice.registry import register_op

# cuEquivariance appears in this file alone, for the cuBLAS grouped GEMM behind
# `indexed_linear`. Its descriptor needs cue's own irreps, so they are built
# here from ours, which share a string form.
_CUE_GROUPS = {"_O3": cue.O3, "_SO3": cue.SO3, "_SU2": cue.SU2}


def _cue_irreps(irreps: e3.Irreps) -> cue.Irreps:
    return cue.Irreps(_CUE_GROUPS[type(irreps.group).__name__], str(irreps))


def _boundary_dtype(prec: Precision):
    return jnp.float64 if prec.alias == "f64" else jnp.float32


def _cast_if_needed(value: Array, dtype) -> Array:
    return value if value.dtype == dtype else value.astype(dtype)


def _resolve_irrep_precision(
    overrides: Mapping,
    group,
    matching: tuple[Irrep, ...],
    global_precision: Precision,
) -> tuple[dict[str, Precision], dict[str, Precision]]:
    """Per-irrep XLA-dot overrides on top of the op's global ``precision``.

    Omitted matching irreps keep ``global_precision``. This is only valid on the
    dense ``einsum`` path (``IrrepsLinear``): grouped-GEMM indexed linear has no
    xN / per-irrep compute type.
    """
    explicit: dict[str, Precision] = {}
    by_ir: dict[Irrep, Precision] = {}
    matching_set = set(matching)
    for raw_key, raw_value in overrides.items():
        if raw_value is None:
            raise ValueError(
                "irreps_linear irrep_precision cannot be null; omit the irrep to use "
                "the global precision"
            )
        ir = raw_key if isinstance(raw_key, Irrep) else group.parse(str(raw_key).strip())
        if ir.group is not group:
            raise ValueError(f"irrep_precision key {ir!r} is not an irrep of {group}")
        if ir not in matching_set:
            raise ValueError(
                f"irrep_precision key {ir} is not a matching irrep of this linear "
                f"(matching: {', '.join(str(m) for m in matching) or 'none'})"
            )
        key = str(ir)
        if key in explicit:
            raise ValueError(f"irrep_precision has duplicate key {key!r}")
        parsed = parse_precision(raw_value)
        explicit[key] = parsed
        by_ir[ir] = parsed
    resolved = {str(ir): by_ir.get(ir, global_precision) for ir in matching}
    return explicit, resolved


@register_op("irreps_linear")
class IrrepsLinear(eqx.Module):
    """Per-irrep dense linear; stores one weight matrix per (mul_in → mul_out) pair.

    ``weight_scaling=True`` (default) stores unit-variance weights and applies a
    compile-time ``1/sqrt(fan_in)`` factor. Direct parameterization
    (``weight_scaling=False``) bakes that variance into the stored weights.
    Readouts default to µP ``readout_scaling='mup'`` (extra ``1/sqrt(fan_in)``).

    ``precision`` is the global compute preset (historical storage contract).
    Optional ``irrep_precision`` overrides that preset on named matching irreps
    for the dense XLA ``einsum`` only; omitted irreps keep the global value.
    When overrides are set, mapping mode keeps FP32 activation/weight/output
    boundaries unless an irrep resolves to FP64. Indexed / grouped-GEMM linears
    do not take this map.

    In mapping mode the global preset is only the compute fallback for unlisted
    irreps; its storage requirement is replaced by these per-irrep boundaries.
    For example, global ``precision='bf16'`` alone requires BF16 activations,
    while ``irrep_precision={'0e': 'bf16'}`` requires FP32 activations for 0e
    and requests a BF16 dot. An empty map keeps the historical storage contract.
    """

    w: dict[str, Array]
    b: dict[str, Array]
    irreps_in: Irreps = eqx.field(static=True)
    irreps_out: Irreps = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    is_readout: bool = eqx.field(static=True)
    weight_scaling: bool = eqx.field(static=True)
    init: str = eqx.field(static=True)
    readout_scaling: str = eqx.field(static=True)
    optimizer: Literal["muon", "adam"] = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)
    irrep_precision: dict[str, Precision] = eqx.field(static=True)
    _ir_map: dict[str, Irrep] = eqx.field(static=True)
    _runtime_scale: dict[str, float] = eqx.field(static=True)
    _mapped_precision: bool = eqx.field(static=True)
    _dot_precision: dict[str, Precision] = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        scale: float = 1.0,
        is_readout: bool = False,
        use_bias: bool = False,
        weight_scaling: bool = True,
        init: str = "normal",
        readout_scaling: str = "mup",
        optimizer: Literal["muon", "adam"] | None = None,
        *,
        precision: Precision,
        irrep_precision: Mapping[str, Any] | None = None,
        param_dtype: Any,
        key: Array,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.scale = scale
        self.is_readout = is_readout
        self.weight_scaling = bool(weight_scaling)
        self.init = init
        if readout_scaling not in ("mup", "standard"):
            raise ValueError(
                "irreps_linear readout_scaling must be 'mup' or 'standard', "
                f"got {readout_scaling!r}"
            )
        if not self.weight_scaling and self.is_readout and readout_scaling != "standard":
            raise ValueError(
                "direct irreps_linear weights require readout_scaling='standard' "
                "to avoid runtime weight factors"
            )
        self.readout_scaling = readout_scaling
        optimizer = ("adam" if self.is_readout else "muon") if optimizer is None else optimizer
        if optimizer not in ("muon", "adam"):
            raise ValueError(f"irreps_linear optimizer must be 'muon' or 'adam', got {optimizer!r}")
        self.optimizer = optimizer
        pairs = [
            (mul_in, ir_in, mul_out)
            for mul_in, ir_in in irreps_in
            for mul_out, ir_out in irreps_out
            if ir_in == ir_out
        ]
        matching = tuple(ir_in for _, ir_in, _ in pairs)
        if isinstance(precision, Mapping):
            raise ValueError(
                "irreps_linear precision must be a single preset; per-irrep XLA-dot "
                "overrides go in irrep_precision"
            )
        parsed = parse_precision(precision)
        self.precision = parsed
        if irrep_precision:
            stored, dots = _resolve_irrep_precision(
                irrep_precision, irreps_in.group, matching, parsed
            )
            self.irrep_precision = stored
            self._mapped_precision = True
            self._dot_precision = dots
        else:
            self.irrep_precision = {}
            self._mapped_precision = False
            self._dot_precision = {str(ir): parsed for ir in matching}
        keys = jax.random.split(key, max(len(pairs), 1))
        weights: dict[str, Array] = {}
        ir_map: dict[str, Irrep] = {}
        runtime_scale: dict[str, float] = {}
        for k, (mul_in, ir_in, mul_out) in zip(keys, pairs, strict=True):
            key_str = str(ir_in)
            weights[key_str] = initialize_linear(
                k,
                (mul_in, mul_out),
                param_dtype,
                init=self.init,
                weight_scaling=self.weight_scaling,
            )
            ir_map[key_str] = ir_in
            factor = float(self.scale)
            if self.weight_scaling:
                factor /= math.sqrt(mul_in)
            if self.is_readout and self.readout_scaling == "mup":
                factor /= math.sqrt(mul_in)
            runtime_scale[key_str] = factor
        self.w = weights
        self._ir_map = ir_map
        self._runtime_scale = runtime_scale
        # Optional additive bias on *scalar* (0e) outputs only (e3nn allows biases
        # solely on even-scalar irreps). Added RAW (no 1/sqrt(fan_in), no scale) to
        # match e3nn o3.Linear(biases=True). Empty dict when disabled → tree-stable
        # and a no-op in __call__, so all existing bias-free chains are unchanged.
        bias: dict[str, Array] = {}
        if use_bias:
            for mul_out, ir_out in irreps_out:
                if ir_out.is_scalar():
                    bias[str(ir_out)] = jnp.zeros((mul_out,), param_dtype)
        self.b = bias

    def __call__(self, node_feats: dict[Irrep, Array]) -> dict[Irrep, Array]:
        x = {ir: value for ir, value in node_feats.items()}
        out = self.native(x)
        return {ir: out[ir] for _, ir in self.irreps_out}

    def native(self, x):
        x0 = jax.tree.leaves(x)[0]
        shape = x0.shape[:-2]
        if self._mapped_precision:
            y = {}
            for mul, ir in self.irreps_out:
                key = str(ir)
                prec = self._dot_precision.get(key)
                dtype = _boundary_dtype(prec) if prec is not None else jnp.float32
                y[ir] = jnp.zeros((*shape, ir.dim, mul), dtype)
        else:
            io = x0.dtype
            self.precision.require_storage(io, "irreps_linear")
            y = {ir: jnp.zeros((*shape, ir.dim, mul), io) for mul, ir in self.irreps_out}
        for key_str, w in self.w.items():
            ir = self._ir_map[key_str]
            prec = self._dot_precision[key_str]
            if self._mapped_precision:
                io = _boundary_dtype(prec)
                if x[ir].dtype != io:
                    raise ValueError(
                        "irreps_linear mapping mode keeps "
                        f"{'f64' if io == jnp.float64 else 'f32'} boundaries; "
                        f"got {x[ir].dtype} for irrep {ir}"
                    )
                lhs = _cast_if_needed(w, io)
                rhs = x[ir]
            else:
                io = x0.dtype
                lhs = w.astype(io)
                rhs = x[ir]
            # Output dtype = `io` (the accumulator is set by the precision preset,
            # not the output type) so the down-cast fuses into the matmul epilogue
            # — no separate fp32->bf16 convert kernel at each linear boundary.
            # Fan-in / readout factors are Python floats baked at init so XLA
            # constant-folds them instead of emitting a runtime sqrt.
            out = jnp.einsum(
                "uv,...iu->...iv",
                lhs,
                rhs,
                precision=prec.xla_dot,
                preferred_element_type=io,
            )
            y[ir] = (out * self._runtime_scale[key_str]).astype(io)
        # raw additive bias on 0e (shape [mul] broadcasts over the [..., 1, mul] scalar)
        for key_str, bvec in self.b.items():
            ir = self._ir_map.get(key_str)
            if ir is None:
                ir = next(out_ir for _, out_ir in self.irreps_out if str(out_ir) == key_str)
            dt = y[ir].dtype
            y[ir] = (y[ir] + bvec.astype(dt)).astype(dt)
        return y


@register_op("irreps_indexed_linear")
class IrrepsIndexedLinear(eqx.Module):
    """Per-species indexed linear; weight is `[num_species, irreps_in.dim * irreps_out.dim]`."""

    w: Array
    irreps_in: Irreps = eqx.field(static=True)
    irreps_out: Irreps = eqx.field(static=True)
    num_indices: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    SPECIES_COUNT_PARAM = "num_indices"  # data species count; auto-filled from the card
    name: str = eqx.field(static=True)
    poly: Any = eqx.field(static=True)
    precision: Precision = eqx.field(static=True)
    method: str = eqx.field(static=True)

    OUTPUTS = "node_feats"

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        num_indices: int,
        scale: float = 1.0,
        *,
        precision: Precision,
        name: str = "indexed_linear",
        method: str = "indexed_linear",
        param_dtype: Any,
        key: Array,
    ):
        assert irreps_in.regroup() == irreps_in
        assert irreps_out.regroup() == irreps_out
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_indices = num_indices
        self.scale = scale / math.sqrt(num_indices)
        self.name = name
        self.precision = parse_precision(precision)
        self.method = method
        if method == "loom_map":
            # cue-free: size the flat weight directly (same layout cue produces:
            # one [mul_in, mul_out] block per matching irrep, irreps order) and
            # reshape to per-irrep blocks at call time. No descriptor built.
            flat_size = sum(mi * mo for mi, ii in irreps_in for mo, io in irreps_out if ii == io)
            self.poly = None
            self.w = jax.random.normal(key, (num_indices, flat_size), param_dtype)
        else:
            self.poly = (
                cue.descriptors.linear(_cue_irreps(irreps_in), _cue_irreps(irreps_out)).polynomial
                * self.scale
            )
            self.w = jax.random.normal(key, (num_indices, self.poly.inputs[0].size), param_dtype)

    def __call__(
        self, node_feats: dict[Irrep, Array], species_counts: Array, graph: Any = None
    ) -> dict[Irrep, Array]:
        # graph is optional: only the loom_map backend uses it (dst->species
        # gather); cuBLAS/naive use species_counts and ignore it.
        io = jax.tree.leaves(node_feats)[0].dtype
        self.precision.require_storage(io, "irreps_indexed_linear")
        if self.method == "loom_map":
            assert graph is not None, "loom_map indexed_linear needs the chain graph"
            # Slice the flat weight into per-irrep [num_indices, mul_in, mul_out]
            # blocks, gather + matmul per node via Loom. The cue linear
            # bakes a 1/sqrt(mul_in) fan-in norm into its coefficients, so apply it
            # here (plus self.scale) to match the cuBLAS/naive output.
            mul_in_of = {ir: mul for mul, ir in self.irreps_in}
            blocks, off = {}, 0
            for mi, ii in self.irreps_in:
                for mo, oo in self.irreps_out:
                    if ii == oo:
                        blocks[ii] = (
                            self.w[:, off : off + mi * mo]
                            .reshape(self.num_indices, mi, mo)
                            .astype(io)
                        )
                        off += mi * mo
            xs = {ir: node_feats[ir].astype(io) for _, ir in self.irreps_in if ir in blocks}
            out = indexed_linear_map(xs, blocks, graph)
            y = {}
            for mo, oo in self.irreps_out:
                native = oo
                y[oo] = (
                    out[native] * self.scale / jnp.sqrt(mul_in_of[native])
                    if native in out
                    else jnp.zeros((graph.domain_sizes["dst"], oo.dim, mo), io)
                )
            return y
        # ir_mul end-to-end: input [..., ir.dim, mul]; kernel expects flat in ir_mul packing,
        # which is what `dict_to_flat` produces when called on ir_mul-shaped dicts.
        #
        # Unlike the XLA ops, `indexed_linear` is a cuBLAS GEMM whose compute mode
        # is a CUBLAS compute-type string, not a plain dtype — the precision maps
        # to it (f32 -> CUBLAS_COMPUTE_32F, bit-identical to cuex's fp32 default;
        # tf32 -> _32F_FAST_TF32; bf16 -> _32F i.e. bf16 GEMM with fp32 accumulate).
        x_flat = native_dict_to_flat(self.irreps_in, node_feats).astype(io)
        num_elements = x_flat.shape[0]
        # `indexed_linear` (cuBLAS) needs a CUBLAS compute-type string, but cuex
        # silently FALLS BACK to the `naive` XLA kernel when no CUDA device is
        # present (e.g. CPU model inspection / tests) — and naive only understands a
        # plain dtype, so a CUBLAS_COMPUTE_* string would crash that fallback. Pick
        # the option format by the actual backend.
        cublas = self.method == "indexed_linear" and jax.default_backend() == "gpu"
        if cublas:
            options = {"math_dtype": self.precision.cublas_compute()}
        else:
            # `naive` (and the CPU fallback of `indexed_linear`) take a plain math
            # dtype, not a cuBLAS compute-type string — use the accumulator dtype.
            options = {"math_dtype": self.precision.accum}
        [y_flat] = cuex.segmented_polynomial(
            self.poly,
            [self.w.astype(io), x_flat],
            [jax.ShapeDtypeStruct((num_elements, self.poly.outputs[0].size), io)],
            [cuex.Repeats(species_counts), None, None],
            method=self.method,
            options=options,
            name=self.name,
        )
        return flat_to_ir_mul(self.irreps_out, y_flat)
