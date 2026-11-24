// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::{SmallVec, smallvec};

use crate::dim_numbers::ConvDimensionNumbers;
use crate::op::def_ops;
use crate::shape::{Dims, Shape, ShapeError};

use super::infer_shapes::{
    Padding, axes_str, bools_str, dims_str, expect_arity, padding_str, same_dtype,
};

def_ops! {
    /// N-D convolution / cross-correlation. Operands: input (lhs), kernel (rhs).
    /// The per-spatial-dim vectors have length `dnums.input_spatial.len()`;
    /// `padding[i]` is `[low, high]` and `window_reversal[i]` reverses the kernel
    /// along spatial dim `i` (empty = no reversal, StableHLO default).
    Convolution {
        dnums: ConvDimensionNumbers,
        window_strides: Dims,
        padding: Padding,
        lhs_dilation: Dims,
        rhs_dilation: Dims,
        window_reversal: SmallVec<[bool; 4]>,
        feature_group_count: i64,
        batch_group_count: i64,
    }
    mnemonic = "convolution"
    attrs = |this| {
        let mut a = vec![
            ("input_batch_dimension", this.dnums.input_batch_dimension.to_string()),
            ("input_feature_dimension", this.dnums.input_feature_dimension.to_string()),
            ("input_spatial_dimensions", axes_str(&this.dnums.input_spatial_dimensions)),
            ("kernel_input_feature_dimension", this.dnums.kernel_input_feature_dimension.to_string()),
            ("kernel_output_feature_dimension", this.dnums.kernel_output_feature_dimension.to_string()),
            ("kernel_spatial_dimensions", axes_str(&this.dnums.kernel_spatial_dimensions)),
            ("output_batch_dimension", this.dnums.output_batch_dimension.to_string()),
            ("output_feature_dimension", this.dnums.output_feature_dimension.to_string()),
            ("output_spatial_dimensions", axes_str(&this.dnums.output_spatial_dimensions)),
            ("window_strides", dims_str(&this.window_strides)),
            ("padding", padding_str(&this.padding)),
            ("lhs_dilation", dims_str(&this.lhs_dilation)),
            ("rhs_dilation", dims_str(&this.rhs_dilation)),
        ];
        if this.window_reversal.iter().any(|&b| b) {
            a.push(("window_reversal", bools_str(&this.window_reversal)));
        }
        a.push(("feature_group_count", this.feature_group_count.to_string()));
        a.push(("batch_group_count", this.batch_group_count.to_string()));
        a
    }
    infer = |this, inputs| {
        let opn = "convolution";
        expect_arity(opn, inputs, 2)?;
        let (lhs, rhs) = (&inputs[0], &inputs[1]);
        same_dtype(opn, lhs, rhs, "operand")?;
        let d = &this.dnums;
        let s = d.input_spatial_dimensions.len();
        if d.kernel_spatial_dimensions.len() != s || d.output_spatial_dimensions.len() != s {
            return Err(ShapeError::new(opn, "spatial dim counts disagree"));
        }
        for v in [&this.window_strides, &this.lhs_dilation, &this.rhs_dilation] {
            if v.len() != s {
                return Err(ShapeError::new(opn, "window vectors must have the spatial rank"));
            }
        }
        if this.padding.len() != s {
            return Err(ShapeError::new(opn, "padding must have the spatial rank"));
        }
        if !this.window_reversal.is_empty() && this.window_reversal.len() != s {
            return Err(ShapeError::new(opn, "window_reversal must have the spatial rank"));
        }
        if this.batch_group_count < 1 || this.feature_group_count < 1 {
            return Err(ShapeError::new(opn, "group counts must be >= 1"));
        }
        let rank = lhs.rank();
        let dim = |t: &Shape, a: u32| -> Result<i64, ShapeError> {
            t.dims
                .get(a as usize)
                .copied()
                .ok_or_else(|| ShapeError::new(opn, "dimension index out of range"))
        };
        let mut out = vec![0i64; rank];
        let ob = d.output_batch_dimension as usize;
        let of = d.output_feature_dimension as usize;
        if ob >= rank || of >= rank {
            return Err(ShapeError::new(opn, "output batch/feature dim out of range"));
        }
        out[ob] = dim(lhs, d.input_batch_dimension)? / this.batch_group_count;
        out[of] = dim(rhs, d.kernel_output_feature_dimension)?;
        for i in 0..s {
            let in_sp = dim(lhs, d.input_spatial_dimensions[i])?;
            let k_sp = dim(rhs, d.kernel_spatial_dimensions[i])?;
            let dilated_input = if in_sp == 0 {
                0
            } else {
                (in_sp - 1) * this.lhs_dilation[i] + 1
            };
            let [pad_low, pad_high] = this.padding[i];
            let padded = dilated_input + pad_low + pad_high;
            let dilated_kernel = (k_sp - 1) * this.rhs_dilation[i] + 1;
            let stride = this.window_strides[i];
            if stride < 1 {
                return Err(ShapeError::new(opn, "window stride must be >= 1"));
            }
            let pos = d.output_spatial_dimensions[i] as usize;
            if pos >= rank {
                return Err(ShapeError::new(opn, "output spatial dim out of range"));
            }
            out[pos] = if padded < dilated_kernel {
                0
            } else {
                (padded - dilated_kernel) / stride + 1
            };
        }
        Ok(smallvec![lhs.with_dims(out)])
    }
}
