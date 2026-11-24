// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, CmpOp};
use xir::dtype::DType;
use xir::element_mapping::num_elements;
use xir::ops::{ComparisonDirection, ScalarOp};

use super::Value;
use super::scalar::{math_scalar, one, pred, zero};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Const {
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

impl Const {
    pub(super) fn as_i64(self) -> Result<i64, String> {
        match self {
            Const::Signed(value) => Ok(value),
            Const::Unsigned(value) => i64::try_from(value)
                .map_err(|_| format!("integer constant {value} does not fit i64")),
            Const::Bool(_) | Const::Float(_) => Err("index constant must be an integer".into()),
        }
    }

    pub(super) fn as_f64(self) -> Result<f64, String> {
        Ok(match self {
            Const::Float(value) => value,
            Const::Signed(value) => value as f64,
            Const::Unsigned(value) => value as f64,
            Const::Bool(_) => return Err("boolean power exponent".into()),
        })
    }
}

pub(super) fn eval_const(op: ScalarOp, inputs: &[Const], dtype: DType) -> Result<Const, String> {
    let input = |index: usize| {
        inputs
            .get(index)
            .copied()
            .ok_or_else(|| format!("constant `{op:?}` has no operand {index}"))
    };
    let signed = |value: Const| match value {
        Const::Signed(value) => Ok(value),
        Const::Unsigned(value) => i64::try_from(value)
            .map_err(|_| "unsigned constant does not fit signed arithmetic".to_string()),
        _ => Err("integer constant operation has a non-integer operand".into()),
    };
    let pair = || Ok::<_, String>((input(0)?, input(1)?));
    let value = match op {
        ScalarOp::Negate => Const::Signed(-signed(input(0)?)?),
        ScalarOp::Abs => Const::Signed(signed(input(0)?)?.abs()),
        ScalarOp::Add => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)? + signed(right)?)
        }
        ScalarOp::Subtract => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)? - signed(right)?)
        }
        ScalarOp::Multiply => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)? * signed(right)?)
        }
        ScalarOp::Divide => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)? / signed(right)?)
        }
        ScalarOp::Remainder => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)? % signed(right)?)
        }
        ScalarOp::Maximum => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)?.max(signed(right)?))
        }
        ScalarOp::Minimum => {
            let (left, right) = pair()?;
            Const::Signed(signed(left)?.min(signed(right)?))
        }
        ScalarOp::And => Const::Bool(match pair()? {
            (Const::Bool(left), Const::Bool(right)) => left && right,
            _ => return Err("boolean and has a non-boolean operand".into()),
        }),
        ScalarOp::Or => Const::Bool(match pair()? {
            (Const::Bool(left), Const::Bool(right)) => left || right,
            _ => return Err("boolean or has a non-boolean operand".into()),
        }),
        ScalarOp::Xor => Const::Bool(match pair()? {
            (Const::Bool(left), Const::Bool(right)) => left ^ right,
            _ => return Err("boolean xor has a non-boolean operand".into()),
        }),
        ScalarOp::Compare(direction) => {
            let (left, right) = pair()?;
            let (left, right) = (signed(left)?, signed(right)?);
            Const::Bool(match direction {
                ComparisonDirection::Eq => left == right,
                ComparisonDirection::Ne => left != right,
                ComparisonDirection::Lt => left < right,
                ComparisonDirection::Le => left <= right,
                ComparisonDirection::Gt => left > right,
                ComparisonDirection::Ge => left >= right,
            })
        }
        ScalarOp::Select => match input(0)? {
            Const::Bool(true) => input(1)?,
            Const::Bool(false) => input(2)?,
            _ => return Err("constant select predicate is not boolean".into()),
        },
        _ => return Err(format!("unsupported constant scalar op {op:?}")),
    };
    Ok(cast_const(value, dtype))
}

pub(super) fn constant_value(
    b: &mut Builder,
    literal: &xir::ops::Literal,
) -> Result<Value, String> {
    let size = literal.shape.dtype.size_bytes();
    let count = num_elements(&literal.shape.dims)?;
    if literal.data.len() != count * size {
        return Err(format!(
            "constant has {} bytes, expected {}",
            literal.data.len(),
            count * size
        ));
    }
    let constants = literal
        .data
        .chunks_exact(size)
        .map(|bytes| decode_const(literal.shape.dtype, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let elems = constants
        .iter()
        .copied()
        .map(|value| const_value(b, value, literal.shape.dtype))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value {
        reductions: vec![None; elems.len()],
        guards: vec![None; elems.len()],
        elems,
        constants: Some(constants),
        indexed: None,
    })
}

fn decode_const(dtype: DType, bytes: &[u8]) -> Result<Const, String> {
    Ok(match dtype {
        DType::Bool => Const::Bool(bytes[0] != 0),
        DType::I8 => Const::Signed(i8::from_le_bytes(bytes.try_into().unwrap()) as i64),
        DType::I16 => Const::Signed(i16::from_le_bytes(bytes.try_into().unwrap()) as i64),
        DType::I32 => Const::Signed(i32::from_le_bytes(bytes.try_into().unwrap()) as i64),
        DType::I64 => Const::Signed(i64::from_le_bytes(bytes.try_into().unwrap())),
        DType::U8 => Const::Unsigned(u8::from_le_bytes(bytes.try_into().unwrap()) as u64),
        DType::U16 => Const::Unsigned(u16::from_le_bytes(bytes.try_into().unwrap()) as u64),
        DType::U32 => Const::Unsigned(u32::from_le_bytes(bytes.try_into().unwrap()) as u64),
        DType::U64 => Const::Unsigned(u64::from_le_bytes(bytes.try_into().unwrap())),
        DType::F16 => {
            Const::Float(f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())) as f64)
        }
        DType::Bf16 => Const::Float(f32::from_bits(
            (u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16,
        ) as f64),
        DType::F32 => Const::Float(f32::from_le_bytes(bytes.try_into().unwrap()) as f64),
        DType::F64 => Const::Float(f64::from_le_bytes(bytes.try_into().unwrap())),
        other => return Err(format!("unsupported constant dtype {other:?}")),
    })
}

fn const_value(b: &mut Builder, value: Const, dtype: DType) -> Result<gpu_ir::Value, String> {
    let scalar = math_scalar(dtype)?;
    Ok(match value {
        Const::Bool(value) => pred(b, value),
        Const::Signed(value) => b.int(scalar, value),
        Const::Unsigned(value) => b.int(scalar, value as i64),
        Const::Float(value) => b.float(scalar, value),
    })
}

pub(super) fn cast_const(value: Const, dtype: DType) -> Const {
    match dtype {
        DType::Bool => Const::Bool(match value {
            Const::Bool(value) => value,
            value => value.as_f64().is_ok_and(|value| value != 0.0),
        }),
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            Const::Signed(value.as_f64().unwrap_or(0.0) as i64)
        }
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
            Const::Unsigned(value.as_f64().unwrap_or(0.0) as u64)
        }
        _ => Const::Float(value.as_f64().unwrap_or(0.0)),
    }
}

pub(super) fn convert(
    b: &mut Builder,
    value: gpu_ir::Value,
    src: DType,
    dst: DType,
    lanes: u8,
) -> Result<gpu_ir::Value, String> {
    if src == dst {
        return Ok(value);
    }
    if dst == DType::Bool {
        let zero = zero(b, math_scalar(src)?);
        let zero = b.splat(lanes, zero);
        return Ok(b.cmp(CmpOp::Ne, value, zero));
    }
    let dst_scalar = math_scalar(dst)?;
    if src == DType::Bool {
        let one = one(b, dst_scalar);
        let one = b.splat(lanes, one);
        let zero = zero(b, dst_scalar);
        let zero = b.splat(lanes, zero);
        return Ok(b.select(value, one, zero));
    }
    Ok(b.cast(dst_scalar, value))
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction as u32;
            let mut exponent = 113u32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(value)
}
