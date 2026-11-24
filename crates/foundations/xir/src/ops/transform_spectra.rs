// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::smallvec;

use crate::dtype::DType;
use crate::op::def_ops;
use crate::shape::{Dims, ShapeError};

use super::infer_shapes::{dims_str, expect_arity};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FftType {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

impl FftType {
    fn name(self) -> &'static str {
        match self {
            FftType::Fft => "fft",
            FftType::Ifft => "ifft",
            FftType::Rfft => "rfft",
            FftType::Irfft => "irfft",
        }
    }
}

def_ops! {
    /// Discrete Fourier transform over the trailing `fft_length.len()` dims.
    Fft { fft_type: FftType, fft_length: Dims }
    mnemonic = "fft"
    attrs = |this| {
        vec![
            ("fft_type", this.fft_type.name().to_string()),
            ("fft_length", dims_str(&this.fft_length)),
        ]
    }
    infer = |this, inputs| {
        expect_arity("fft", inputs, 1)?;
        let x = &inputs[0];
        let n = this.fft_length.len();
        if n == 0 || n > x.rank() {
            return Err(ShapeError::new("fft", "fft_length rank invalid for operand"));
        }
        let last = x.rank() - 1;
        match this.fft_type {
            FftType::Fft | FftType::Ifft => {
                if !x.dtype.is_complex() {
                    return Err(ShapeError::new("fft", "fft/ifft operand must be complex"));
                }
                Ok(smallvec![x.clone()])
            }
            FftType::Rfft => {
                let dtype = match x.dtype {
                    DType::F32 => DType::Complex64,
                    DType::F64 => DType::Complex128,
                    _ => return Err(ShapeError::new("fft", "rfft operand must be real f32/f64")),
                };
                let mut dims = x.dims.clone();
                dims[last] = this.fft_length[n - 1] / 2 + 1;
                Ok(smallvec![x.with_dims(dims).with_dtype(dtype)])
            }
            FftType::Irfft => {
                let dtype = match x.dtype {
                    DType::Complex64 => DType::F32,
                    DType::Complex128 => DType::F64,
                    _ => return Err(ShapeError::new("fft", "irfft operand must be complex")),
                };
                let mut dims = x.dims.clone();
                dims[last] = this.fft_length[n - 1];
                Ok(smallvec![x.with_dims(dims).with_dtype(dtype)])
            }
        }
    }
}
