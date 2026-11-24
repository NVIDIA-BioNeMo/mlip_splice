// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use num_bigint::BigInt;
use num_rational::BigRational;

use crate::dtype::DType;
use crate::shape::Shape;

// ----------------------------------------------------------------------------

/// A constant tensor with its runtime element bytes and optional exact real values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Literal {
    pub shape: Shape,
    pub data: Vec<u8>,
    pub exact: Option<Vec<AlgebraicReal>>,
}

impl Literal {
    pub fn new(shape: Shape, data: Vec<u8>) -> Literal {
        Literal {
            shape,
            data,
            exact: None,
        }
    }

    pub fn exact(shape: Shape, values: Vec<AlgebraicReal>) -> Result<Literal, String> {
        if !shape.dtype.is_float() {
            return Err(format!(
                "exact literal requires a floating dtype, got {}",
                shape.dtype.name()
            ));
        }
        let needed = shape.element_count()?;
        if values.len() != needed {
            return Err(format!(
                "exact literal has {} values, shape needs {}",
                values.len(),
                needed
            ));
        }
        let data = encode_exact(&values, shape.dtype);
        Ok(Literal {
            shape,
            data,
            exact: Some(values),
        })
    }

    pub fn with_exact(mut self, values: Vec<AlgebraicReal>) -> Result<Literal, String> {
        if !self.shape.dtype.is_float() {
            return Err(format!(
                "exact literal requires a floating dtype, got {}",
                self.shape.dtype.name()
            ));
        }
        let needed = self.shape.element_count()?;
        if values.len() != needed {
            return Err(format!(
                "exact literal has {} values, shape needs {}",
                values.len(),
                needed
            ));
        }
        self.exact = Some(values);
        Ok(self)
    }

    /// Real values for algebraic rewrites, preferring declared exact values to
    /// the rationals encoded by finite floating-point elements. Both signed
    /// zeros denote zero; non-floating or non-finite elements have no value.
    pub fn algebraic_values(&self) -> Option<Vec<AlgebraicReal>> {
        let (exponent_bits, fraction_bits) = match self.shape.dtype {
            DType::F16 => (5, 10),
            DType::Bf16 => (8, 7),
            DType::F32 => (8, 23),
            DType::F64 => (11, 52),
            _ => return None,
        };
        if let Some(values) = &self.exact {
            return (values.len() == self.shape.num_elements()?).then(|| values.clone());
        }
        if self.data.len() != self.shape.byte_size().ok()? {
            return None;
        }
        self.data
            .chunks_exact(self.shape.dtype.size_bytes())
            .map(|element| decode_float_rational(element, exponent_bits, fraction_bits))
            .collect()
    }

    /// An all-zero constant of the given shape, refused for a shape with no
    /// byte extent to zero.
    pub fn zeros(shape: Shape) -> Result<Literal, String> {
        let n = shape.byte_size()?;
        Ok(Literal {
            data: vec![0u8; n],
            shape,
            exact: None,
        })
    }
}

fn decode_float_rational(
    element: &[u8],
    exponent_bits: u32,
    fraction_bits: u32,
) -> Option<AlgebraicReal> {
    let mut bytes = [0u8; 8];
    bytes[..element.len()].copy_from_slice(element);
    let bits = u64::from_le_bytes(bytes);
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    if exponent == exponent_mask {
        return None;
    }
    let fraction = bits & ((1u64 << fraction_bits) - 1);
    let significand = fraction | (u64::from(exponent != 0) << fraction_bits);
    let mut numerator = BigInt::from(significand);
    if bits >> (exponent_bits + fraction_bits) != 0 {
        numerator = -numerator;
    }
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let power = exponent.max(1) as i32 - bias - fraction_bits as i32;
    let value = if power >= 0 {
        BigRational::from_integer(numerator << power as usize)
    } else {
        BigRational::new(numerator, BigInt::from(1) << (-power) as usize)
    };
    Some(AlgebraicReal::rational(value))
}

fn encode_exact(values: &[AlgebraicReal], dtype: DType) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * dtype.size_bytes());
    for value in values {
        let value = value.to_f64();
        match dtype {
            DType::F16 => data.extend_from_slice(&f16_bits(value).to_le_bytes()),
            DType::Bf16 => data.extend_from_slice(&bf16_bits(value).to_le_bytes()),
            DType::F32 => data.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F64 => data.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!("exact literal dtype was validated as floating"),
        }
    }
    data
}

fn f16_bits(value: f64) -> u16 {
    let bits = (value as f32).to_bits();
    let (sign, exponent, mantissa) = (bits >> 31, ((bits >> 23) & 0xff) as i32, bits & 0x7f_ffff);
    let sign = (sign as u16) << 15;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (mantissa + half - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let half = 0x0fff + ((mantissa >> 13) & 1);
    let rounded = ((((exponent as u32) << 23) | mantissa) + half) >> 13;
    sign | rounded as u16
}

fn bf16_bits(value: f64) -> u16 {
    let bits = (value as f32).to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x7f_ffff != 0 {
        return ((bits >> 16) | 0x40) as u16;
    }
    (((bits as u64 + 0x7fff + ((bits >> 16) & 1) as u64) >> 16) & 0xffff) as u16
}
