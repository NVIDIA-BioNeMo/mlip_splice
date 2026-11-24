// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::{SmallVec, smallvec};

use crate::dtype::DType;
use crate::op::def_ops;
use crate::shape::{Dims, Shape, ShapeError};

use super::Literal;
use super::infer_shapes::{axes_str, dims_str, expect_arity, expect_scalar, same_dtype};

def_ops! {
    /// Element-type cast, dims unchanged.
    Convert { dtype: DType }
    mnemonic = "convert"
    attrs = |this| { vec![("dtype", this.dtype.name().to_string())] }
    infer = |this, inputs| {
        expect_arity("convert", inputs, 1)?;
        Ok(smallvec![inputs[0].with_dtype(this.dtype)])
    }

    /// Broadcast `broadcast_dimensions[i]` of the operand to result dim `i`; each
    /// mapped operand dim must equal the result size or be 1. `out_dims` is the
    /// target shape: StableHLO carries it in the result type, but xir is
    /// infer-based, so the op must carry it.
    BroadcastInDim { out_dims: Dims, broadcast_dimensions: SmallVec<[u32; 4]> }
    mnemonic = "broadcast_in_dim"
    attrs = |this| {
        vec![
            ("out_dims", dims_str(&this.out_dims)),
            ("broadcast_dimensions", axes_str(&this.broadcast_dimensions)),
        ]
    }
    infer = |this, inputs| {
        expect_arity("broadcast_in_dim", inputs, 1)?;
        let x = &inputs[0];
        if this.broadcast_dimensions.len() != x.rank() {
            return Err(ShapeError::new(
                "broadcast_in_dim",
                format!(
                    "broadcast_dimensions len {} != operand rank {}",
                    this.broadcast_dimensions.len(),
                    x.rank()
                ),
            ));
        }
        for (i, &od) in this.broadcast_dimensions.iter().enumerate() {
            let od = od as usize;
            let target = *this.out_dims.get(od).ok_or_else(|| {
                ShapeError::new("broadcast_in_dim", format!("mapped dim {od} out of result rank"))
            })?;
            let src = x.dims[i];
            if src != target && src != 1 {
                return Err(ShapeError::new(
                    "broadcast_in_dim",
                    format!("operand dim {i} (size {src}) cannot broadcast to {target}"),
                ));
            }
        }
        Ok(smallvec![x.with_dims(this.out_dims.iter().copied())])
    }

    /// Permute dimensions; `permutation` is a permutation of `0..rank`.
    Transpose { permutation: SmallVec<[u32; 4]> }
    mnemonic = "transpose"
    attrs = |this| { vec![("permutation", axes_str(&this.permutation))] }
    infer = |this, inputs| {
        expect_arity("transpose", inputs, 1)?;
        let x = &inputs[0];
        if this.permutation.len() != x.rank() {
            return Err(ShapeError::new("transpose", "permutation length != operand rank"));
        }
        let mut seen = vec![false; x.rank()];
        for &p in &this.permutation {
            let p = p as usize;
            if p >= x.rank() || seen[p] {
                return Err(ShapeError::new("transpose", "permutation is not a permutation"));
            }
            seen[p] = true;
        }
        let dims = this.permutation.iter().map(|&p| x.dims[p as usize]);
        Ok(smallvec![x.with_dims(dims)])
    }

    /// Reinterpret the elements as `out_dims` (same element count).
    Reshape { out_dims: Dims }
    mnemonic = "reshape"
    attrs = |this| { vec![("out_dims", dims_str(&this.out_dims))] }
    infer = |this, inputs| {
        expect_arity("reshape", inputs, 1)?;
        let x = &inputs[0];
        let out = x.with_dims(this.out_dims.iter().copied());
        let (out_count, in_count) = (
            out.element_count().map_err(|e| ShapeError::new("reshape", e))?,
            x.element_count().map_err(|e| ShapeError::new("reshape", e))?,
        );
        if out_count != in_count {
            return Err(ShapeError::new(
                "reshape",
                format!("element count differs: {out_count} vs {in_count}"),
            ));
        }
        Ok(smallvec![out])
    }

    /// Strided sub-array: `out_i = ceil((limit_i - start_i) / stride_i)`.
    Slice { start_indices: Dims, limit_indices: Dims, strides: Dims }
    mnemonic = "slice"
    attrs = |this| {
        vec![
            ("start_indices", dims_str(&this.start_indices)),
            ("limit_indices", dims_str(&this.limit_indices)),
            ("strides", dims_str(&this.strides)),
        ]
    }
    infer = |this, inputs| {
        expect_arity("slice", inputs, 1)?;
        let x = &inputs[0];
        let r = x.rank();
        if this.start_indices.len() != r || this.limit_indices.len() != r || this.strides.len() != r {
            return Err(ShapeError::new("slice", "start_indices/limit_indices/strides rank mismatch"));
        }
        let mut out = Vec::with_capacity(r);
        for i in 0..r {
            let (s, l, st) = (this.start_indices[i], this.limit_indices[i], this.strides[i]);
            if st < 1 {
                return Err(ShapeError::new("slice", "stride must be >= 1"));
            }
            if s < 0 || l > x.dims[i] || s > l {
                return Err(ShapeError::new("slice", format!("invalid range on dim {i}")));
            }
            // ceil((l - s) / st) for non-negative span and stride >= 1.
            out.push((l - s + st - 1) / st);
        }
        Ok(smallvec![x.with_dims(out)])
    }

    /// Join operands along `dimension`; all other dims (and dtype) must match.
    Concatenate { dimension: u32 }
    mnemonic = "concatenate"
    attrs = |this| { vec![("dimension", this.dimension.to_string())] }
    infer = |this, inputs| {
        if inputs.is_empty() {
            return Err(ShapeError::new("concatenate", "needs at least one operand"));
        }
        let first = &inputs[0];
        let d = this.dimension as usize;
        if d >= first.rank() {
            return Err(ShapeError::new("concatenate", "dim out of range"));
        }
        let mut size = 0i64;
        for x in inputs {
            same_dtype("concatenate", first, x, "operand")?;
            if x.rank() != first.rank() {
                return Err(ShapeError::new("concatenate", "operand ranks differ"));
            }
            for k in 0..first.rank() {
                if k != d && x.dims[k] != first.dims[k] {
                    return Err(ShapeError::new(
                        "concatenate",
                        format!("operand dim {k} differs off the concat axis"),
                    ));
                }
            }
            size += x.dims[d];
        }
        let mut out = first.dims.clone();
        out[d] = size;
        Ok(smallvec![first.with_dims(out)])
    }

    /// Pad with `edge_padding_low`/`edge_padding_high` edge padding and
    /// `interior_padding` between-element padding. Operand 1 is the (scalar)
    /// padding value. The edge paddings may be negative.
    Pad { edge_padding_low: Dims, edge_padding_high: Dims, interior_padding: Dims }
    mnemonic = "pad"
    attrs = |this| {
        vec![
            ("edge_padding_low", dims_str(&this.edge_padding_low)),
            ("edge_padding_high", dims_str(&this.edge_padding_high)),
            ("interior_padding", dims_str(&this.interior_padding)),
        ]
    }
    infer = |this, inputs| {
        expect_arity("pad", inputs, 2)?;
        let x = &inputs[0];
        let pad_value = &inputs[1];
        expect_scalar("pad", pad_value, "padding value")?;
        same_dtype("pad", x, pad_value, "operand and padding value")?;
        let r = x.rank();
        if this.edge_padding_low.len() != r
            || this.edge_padding_high.len() != r
            || this.interior_padding.len() != r
        {
            return Err(ShapeError::new(
                "pad",
                "edge_padding_low/edge_padding_high/interior_padding rank mismatch",
            ));
        }
        let mut out = Vec::with_capacity(r);
        for i in 0..r {
            if this.interior_padding[i] < 0 {
                return Err(ShapeError::new("pad", "interior padding must be >= 0"));
            }
            let d = x.dims[i];
            let interior_total = (d - 1).max(0) * this.interior_padding[i];
            let size = this.edge_padding_low[i] + this.edge_padding_high[i] + d + interior_total;
            if size < 0 {
                return Err(ShapeError::new("pad", format!("negative padded size on dim {i}")));
            }
            out.push(size);
        }
        Ok(smallvec![x.with_dims(out)])
    }

    /// Reverse the operand along each of `dimensions`; shape-preserving.
    Reverse { dimensions: SmallVec<[u32; 4]> }
    mnemonic = "reverse"
    attrs = |this| { vec![("dimensions", axes_str(&this.dimensions))] }
    infer = |this, inputs| {
        expect_arity("reverse", inputs, 1)?;
        let x = &inputs[0];
        for &dd in &this.dimensions {
            if dd as usize >= x.rank() {
                return Err(ShapeError::new("reverse", "dim out of range"));
            }
        }
        Ok(smallvec![x.clone()])
    }

    /// 0,1,2,... along `iota_dimension` over a tensor of the given `shape`. No
    /// operands. `shape` is xir-only (infer-based): StableHLO carries it in the
    /// result type, like `BroadcastInDim.out_dims`.
    Iota { shape: Shape, iota_dimension: u32 }
    mnemonic = "iota"
    attrs = |this| { vec![("iota_dimension", this.iota_dimension.to_string())] }
    infer = |this, inputs| {
        expect_arity("iota", inputs, 0)?;
        if this.iota_dimension as usize >= this.shape.rank() {
            return Err(ShapeError::new("iota", "dim out of range"));
        }
        Ok(smallvec![this.shape.clone()])
    }

    /// A constant tensor. No operands.
    Constant { literal: Literal }
    mnemonic = "constant"
    attrs = |this| {
        this.literal.exact.as_ref().map_or_else(Vec::new, |values| {
            let values = values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            vec![("exact", format!("[{values}]"))]
        })
    }
    infer = |this, inputs| {
        expect_arity("constant", inputs, 0)?;
        let expected = this
            .literal
            .shape
            .byte_size()
            .map_err(|e| ShapeError::new("constant", e))?;
        if this.literal.data.len() != expected {
            return Err(ShapeError::new(
                "constant",
                format!("data is {} bytes, shape needs {expected}", this.literal.data.len()),
            ));
        }
        if let Some(exact) = &this.literal.exact {
            if !this.literal.shape.dtype.is_float() {
                return Err(ShapeError::new(
                    "constant",
                    format!(
                        "exact values require a floating dtype, got {}",
                        this.literal.shape.dtype.name()
                    ),
                ));
            }
            let needed = this
                .literal
                .shape
                .element_count()
                .map_err(|e| ShapeError::new("constant", e))?;
            if exact.len() != needed {
                return Err(ShapeError::new(
                    "constant",
                    format!(
                        "exact value count is {}, shape needs {needed}",
                        exact.len(),
                    ),
                ));
            }
        }
        Ok(smallvec![this.literal.shape.clone()])
    }

}
