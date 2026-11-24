// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::smallvec;

use crate::dim_numbers::DotDimensionNumbers;
use crate::op::def_ops;
use crate::shape::{Shape, ShapeError};

use super::infer_shapes::{axes_str, expect_arity, expect_axes, expect_disjoint_axes, same_dtype};

fn dot_infer(lhs: &Shape, rhs: &Shape, d: &DotDimensionNumbers) -> Result<Shape, ShapeError> {
    let op = "dot_general";
    same_dtype(op, lhs, rhs, "operand")?;
    for (name, axes) in [
        (
            "lhs batching dimensions",
            d.lhs_batching_dimensions.as_slice(),
        ),
        (
            "lhs contracting dimensions",
            d.lhs_contracting_dimensions.as_slice(),
        ),
    ] {
        expect_axes(op, name, axes, lhs.rank(), false)?;
    }
    expect_disjoint_axes(
        op,
        "lhs batching dimensions",
        &d.lhs_batching_dimensions,
        "lhs contracting dimensions",
        &d.lhs_contracting_dimensions,
    )?;
    for (name, axes) in [
        (
            "rhs batching dimensions",
            d.rhs_batching_dimensions.as_slice(),
        ),
        (
            "rhs contracting dimensions",
            d.rhs_contracting_dimensions.as_slice(),
        ),
    ] {
        expect_axes(op, name, axes, rhs.rank(), false)?;
    }
    expect_disjoint_axes(
        op,
        "rhs batching dimensions",
        &d.rhs_batching_dimensions,
        "rhs contracting dimensions",
        &d.rhs_contracting_dimensions,
    )?;
    if d.lhs_batching_dimensions.len() != d.rhs_batching_dimensions.len() {
        return Err(ShapeError::new(op, "batch dimension count mismatch"));
    }
    if d.lhs_contracting_dimensions.len() != d.rhs_contracting_dimensions.len() {
        return Err(ShapeError::new(op, "contracting dimension count mismatch"));
    }
    let dim = |s: &Shape, a: u32| -> Result<i64, ShapeError> {
        s.dims.get(a as usize).copied().ok_or_else(|| {
            ShapeError::new(op, format!("dim {a} out of range for rank {}", s.rank()))
        })
    };
    for (l, r) in d
        .lhs_batching_dimensions
        .iter()
        .zip(&d.rhs_batching_dimensions)
    {
        if dim(lhs, *l)? != dim(rhs, *r)? {
            return Err(ShapeError::new(op, "batch dimension sizes differ"));
        }
    }
    for (l, r) in d
        .lhs_contracting_dimensions
        .iter()
        .zip(&d.rhs_contracting_dimensions)
    {
        if dim(lhs, *l)? != dim(rhs, *r)? {
            return Err(ShapeError::new(op, "contracting dimension sizes differ"));
        }
    }
    let mut out: Vec<i64> = Vec::new();
    for &b in &d.lhs_batching_dimensions {
        out.push(dim(lhs, b)?);
    }
    for i in 0..lhs.rank() as u32 {
        if !d.lhs_batching_dimensions.contains(&i) && !d.lhs_contracting_dimensions.contains(&i) {
            out.push(dim(lhs, i)?);
        }
    }
    for i in 0..rhs.rank() as u32 {
        if !d.rhs_batching_dimensions.contains(&i) && !d.rhs_contracting_dimensions.contains(&i) {
            out.push(dim(rhs, i)?);
        }
    }
    Ok(lhs.with_dims(out))
}

/// A dot operand/accumulator compute precision. Distinct from storage
/// [`DType`](crate::dtype::DType):
/// `Tf32` is a compute-only precision an f32-stored operand is rounded to, so an
/// f32 array contracted at `Tf32` is exactly the TF32 tensor-core path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrecisionType {
    F16,
    Bf16,
    Tf32,
    F32,
    F64,
}

impl PrecisionType {
    pub fn name(self) -> &'static str {
        match self {
            PrecisionType::F16 => "f16",
            PrecisionType::Bf16 => "bf16",
            PrecisionType::Tf32 => "tf32",
            PrecisionType::F32 => "f32",
            PrecisionType::F64 => "f64",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "f16" => PrecisionType::F16,
            "bf16" => PrecisionType::Bf16,
            "tf32" => PrecisionType::Tf32,
            "f32" => PrecisionType::F32,
            "f64" => PrecisionType::F64,
            _ => return None,
        })
    }
}

/// The explicit contraction algorithm carried by every [`DotGeneral`]: operand and
/// accumulator precisions, operand-splitting for emulation, and the primitive-op
/// count of the `_X3`/`_X6`/`_X9` families. Mirrors StableHLO/`jax.lax`
/// `DotAlgorithm`; xir keeps no vague default, so this is always present.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DotAlgorithm {
    pub lhs: PrecisionType,
    pub rhs: PrecisionType,
    pub accum: PrecisionType,
    pub lhs_components: u8,
    pub rhs_components: u8,
    pub num_primitive_ops: u8,
    pub allow_imprecise_accum: bool,
}

impl DotAlgorithm {
    const fn plain(lhs: PrecisionType, rhs: PrecisionType, accum: PrecisionType) -> Self {
        DotAlgorithm {
            lhs,
            rhs,
            accum,
            lhs_components: 1,
            rhs_components: 1,
            num_primitive_ops: 1,
            allow_imprecise_accum: false,
        }
    }

    const fn ops(mut self, n: u8) -> Self {
        self.num_primitive_ops = n;
        self
    }

    pub const fn f16_f16_f16() -> Self {
        Self::plain(PrecisionType::F16, PrecisionType::F16, PrecisionType::F16)
    }
    pub const fn f16_f16_f32() -> Self {
        Self::plain(PrecisionType::F16, PrecisionType::F16, PrecisionType::F32)
    }
    pub const fn bf16_bf16_f32() -> Self {
        Self::plain(PrecisionType::Bf16, PrecisionType::Bf16, PrecisionType::F32)
    }
    pub const fn bf16_bf16_f32_x3() -> Self {
        Self::bf16_bf16_f32().ops(3)
    }
    pub const fn tf32_tf32_f32() -> Self {
        Self::plain(PrecisionType::Tf32, PrecisionType::Tf32, PrecisionType::F32)
    }
    pub const fn tf32_tf32_f32_x3() -> Self {
        Self::tf32_tf32_f32().ops(3)
    }
    pub const fn f32_f32_f32() -> Self {
        Self::plain(PrecisionType::F32, PrecisionType::F32, PrecisionType::F32)
    }
    pub const fn f64_f64_f64() -> Self {
        Self::plain(PrecisionType::F64, PrecisionType::F64, PrecisionType::F64)
    }

    /// The preset name, when this matches a known preset; else `None` and the
    /// printer falls back to the structured fields.
    fn preset_name(&self) -> Option<&'static str> {
        let table = [
            ("f16_f16_f16", Self::f16_f16_f16()),
            ("f16_f16_f32", Self::f16_f16_f32()),
            ("bf16_bf16_f32", Self::bf16_bf16_f32()),
            ("bf16_bf16_f32_x3", Self::bf16_bf16_f32_x3()),
            ("tf32_tf32_f32", Self::tf32_tf32_f32()),
            ("tf32_tf32_f32_x3", Self::tf32_tf32_f32_x3()),
            ("f32_f32_f32", Self::f32_f32_f32()),
            ("f64_f64_f64", Self::f64_f64_f64()),
        ];
        table.iter().find(|(_, a)| a == self).map(|(n, _)| *n)
    }

    fn attr_str(&self) -> String {
        if let Some(n) = self.preset_name() {
            return n.to_string();
        }
        format!(
            "{{lhs={}, rhs={}, accum={}, lhs_components={}, rhs_components={}, num_primitive_ops={}, allow_imprecise_accum={}}}",
            self.lhs.name(),
            self.rhs.name(),
            self.accum.name(),
            self.lhs_components,
            self.rhs_components,
            self.num_primitive_ops,
            self.allow_imprecise_accum,
        )
    }
}

def_ops! {
    /// General contraction (matmul / batched matmul / tensordot). `algorithm`
    /// carries the explicit contraction precision (see [`DotAlgorithm`]).
    DotGeneral { dnums: DotDimensionNumbers, algorithm: DotAlgorithm }
    mnemonic = "dot_general"
    attrs = |this| {
        vec![
            ("lhs_contracting_dimensions", axes_str(&this.dnums.lhs_contracting_dimensions)),
            ("rhs_contracting_dimensions", axes_str(&this.dnums.rhs_contracting_dimensions)),
            ("lhs_batching_dimensions", axes_str(&this.dnums.lhs_batching_dimensions)),
            ("rhs_batching_dimensions", axes_str(&this.dnums.rhs_batching_dimensions)),
            ("algorithm", this.algorithm.attr_str()),
        ]
    }
    infer = |this, inputs| {
        expect_arity("dot_general", inputs, 2)?;
        Ok(smallvec![dot_infer(&inputs[0], &inputs[1], &this.dnums)?])
    }
}
