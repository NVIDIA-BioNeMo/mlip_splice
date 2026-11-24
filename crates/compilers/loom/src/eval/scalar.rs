// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, CmpOp, Scalar, UnOp, Value};
use xir::dtype::DType;
use xir::ops::ComparisonDirection;

pub(crate) fn math_scalar(dtype: DType) -> Result<Scalar, String> {
    Ok(match dtype {
        DType::Bool => Scalar::Pred,
        DType::I8 => Scalar::S8,
        DType::I16 => Scalar::S16,
        DType::I32 => Scalar::S32,
        DType::I64 => Scalar::S64,
        DType::U8 => Scalar::U8,
        DType::U16 => Scalar::U16,
        DType::U32 => Scalar::U32,
        DType::U64 => Scalar::U64,
        DType::F16 => Scalar::F16,
        DType::Bf16 => Scalar::BF16,
        DType::F32 => Scalar::F32,
        DType::F64 => Scalar::F64,
        other => return Err(format!("unsupported Loom math dtype {other:?}")),
    })
}

pub(super) fn signed_integer(scalar: Scalar) -> Result<Scalar, String> {
    Ok(match scalar {
        Scalar::S8 | Scalar::U8 => Scalar::S8,
        Scalar::S16 | Scalar::U16 => Scalar::S16,
        Scalar::S32 | Scalar::U32 => Scalar::S32,
        Scalar::S64 | Scalar::U64 => Scalar::S64,
        _ => return Err(format!("arithmetic shift on {scalar:?}")),
    })
}

pub(super) fn unsigned_integer(scalar: Scalar) -> Result<Scalar, String> {
    Ok(match scalar {
        Scalar::S8 | Scalar::U8 => Scalar::U8,
        Scalar::S16 | Scalar::U16 => Scalar::U16,
        Scalar::S32 | Scalar::U32 => Scalar::U32,
        Scalar::S64 | Scalar::U64 => Scalar::U64,
        _ => return Err(format!("logical shift on {scalar:?}")),
    })
}

pub(crate) fn zero(b: &mut Builder, scalar: Scalar) -> Value {
    if scalar == Scalar::Pred {
        pred(b, false)
    } else {
        b.zero(scalar)
    }
}

pub(super) fn one(b: &mut Builder, scalar: Scalar) -> Value {
    if scalar == Scalar::Pred {
        pred(b, true)
    } else if scalar.is_float() {
        b.float(scalar, 1.0)
    } else {
        b.int(scalar, 1)
    }
}

/// The scalar types the hardware's approximate instructions accept. There is no
/// approximate double-precision transcendental in PTX, so a double one is
/// refused rather than silently computed at single precision.
pub(super) fn approximate_scalar(scalar: Scalar) -> Result<(), String> {
    match scalar {
        Scalar::F16 | Scalar::BF16 | Scalar::F32 => Ok(()),
        Scalar::F64 => Err(
            "loom has no double-precision transcendental: the hardware's approximate \
             instructions are single-precision"
                .into(),
        ),
        other => Err(format!("loom transcendental of {other:?}")),
    }
}

/// One float constant, spread across the lanes.
pub(super) fn spread_float(b: &mut Builder, value: f64, scalar: Scalar, lanes: u8) -> Value {
    let constant = b.float(scalar, value);
    b.splat(lanes, constant)
}

/// `e` raised to the operand, as the hardware's base-two exponential of the
/// operand scaled by the logarithm of `e`.
pub(super) fn exponential(b: &mut Builder, value: Value, scalar: Scalar, lanes: u8) -> Value {
    let scale = spread_float(b, std::f64::consts::LOG2_E, scalar, lanes);
    let scaled = b.mul(value, scale);
    b.un(UnOp::Exp2, scaled)
}

/// The natural logarithm of the operand, as the hardware's base-two logarithm
/// scaled by the logarithm of two.
pub(super) fn natural_log(b: &mut Builder, value: Value, scalar: Scalar, lanes: u8) -> Value {
    let logarithm = b.un(UnOp::Log2, value);
    let scale = spread_float(b, std::f64::consts::LN_2, scalar, lanes);
    b.mul(logarithm, scale)
}

pub(super) fn magnitude(b: &mut Builder, value: Value) -> Value {
    let negated = b.neg(value.clone());
    b.max(value, negated)
}

/// Minus one, zero or one, carrying the operand through unchanged where it is
/// neither below nor above zero, so a NaN stays a NaN.
pub(super) fn sign_of(b: &mut Builder, value: Value, scalar: Scalar, lanes: u8) -> Value {
    let zero = zero(b, scalar);
    let zero = b.splat(lanes, zero);
    let one = one(b, scalar);
    let one = b.splat(lanes, one);
    let below = b.cmp(CmpOp::Lt, value.clone(), zero.clone());
    let above = b.cmp(CmpOp::Gt, value.clone(), zero);
    let positive = b.select(above, one.clone(), value);
    let negative = b.neg(one);
    b.select(below, negative, positive)
}

/// The arc tangent of `y / x` in the correct quadrant, branch-free.
///
/// PTX has no arc tangent, so this is the classic three-interval reduction:
/// the polynomial is accurate on $[0, tan(pi/8)]$, and $x > tan(3pi/8)$ or
/// $x > tan(pi/8)$ each map into it with an angle added back. Selection rather
/// than branching, so every lane runs the same instructions.
pub(super) fn arc_tangent_of_ratio(
    b: &mut Builder,
    y: Value,
    x: Value,
    scalar: Scalar,
    lanes: u8,
) -> Value {
    let constant = |b: &mut Builder, value: f64| spread_float(b, value, scalar, lanes);

    let zero = constant(b, 0.0);
    let one = constant(b, 1.0);
    let abs_y = magnitude(b, y.clone());
    let abs_x = magnitude(b, x.clone());
    let (abs_y_kept, abs_x_kept) = (abs_y.clone(), abs_x.clone());
    let ratio = b.div(abs_y, abs_x);

    // Reduce: t stands for the tangent of the reduced angle, and `added` for the
    // angle the reduction took out.
    let high = constant(b, std::f64::consts::FRAC_PI_2);
    let middle = constant(b, std::f64::consts::FRAC_PI_4);
    let high_edge = constant(b, 2.414_213_562_373_095);
    let above_high = b.cmp(CmpOp::Gt, ratio.clone(), high_edge);
    let middle_edge = constant(b, 0.414_213_562_373_095);
    let above_middle = b.cmp(CmpOp::Gt, ratio.clone(), middle_edge);
    let inverted = {
        let negated = b.neg(one.clone());
        b.div(negated, ratio.clone())
    };
    let shifted = {
        let numerator = b.sub(ratio.clone(), one.clone());
        let denominator = b.add(ratio.clone(), one.clone());
        b.div(numerator, denominator)
    };
    let reduced = b.select(above_middle.clone(), shifted, ratio);
    let t = b.select(above_high.clone(), inverted, reduced);
    let added = {
        let middle_or_none = b.select(above_middle, middle, zero.clone());
        b.select(above_high, high.clone(), middle_or_none)
    };

    // Horner over the odd powers, minimax on the reduced interval.
    let square = b.mul(t.clone(), t.clone());
    let mut series = constant(b, 8.053_744_495_38e-2);
    for coefficient in [
        -1.387_768_560_32e-1,
        1.997_771_064_78e-1,
        -3.333_294_915_39e-1,
    ] {
        let scaled = b.mul(series, square.clone());
        series = {
            let next = constant(b, coefficient);
            b.add(scaled, next)
        };
    }
    let odd = {
        let scaled = b.mul(series, square);
        let corrected = b.mul(scaled, t.clone());
        b.add(corrected, t)
    };
    let first_quadrant = b.add(added, odd);

    // Place it: reflect for a negative x, then carry y's sign. The axes fall out
    // of the same two selects, since the ratio is zero or infinite there.
    let pi = constant(b, std::f64::consts::PI);
    let reflected = b.sub(pi, first_quadrant.clone());
    let x_negative = b.cmp(CmpOp::Lt, x, zero.clone());
    let placed = b.select(x_negative, reflected, first_quadrant);
    let negated = b.neg(placed.clone());
    let y_negative = b.cmp(CmpOp::Lt, y, zero.clone());
    let placed = b.select(y_negative, negated, placed);

    // At the origin the ratio is zero over zero, which no later select can
    // recover from, and the answer there is zero.
    let x_is_zero = b.cmp(CmpOp::Eq, abs_x_kept, zero.clone());
    let y_is_zero = b.cmp(CmpOp::Eq, abs_y_kept, zero.clone());
    let at_origin = b.select(y_is_zero, zero, placed.clone());
    b.select(x_is_zero, at_origin, placed)
}

pub(super) fn pred(b: &mut Builder, value: bool) -> Value {
    let zero = b.i32(0);
    b.cmp(
        if value { CmpOp::Eq } else { CmpOp::Ne },
        zero.clone(),
        zero,
    )
}

pub(super) fn half_power(
    b: &mut Builder,
    base: Value,
    exponent: f64,
    scalar: Scalar,
    lanes: u8,
) -> Result<Value, String> {
    let twice = exponent * 2.0;
    let rounded = twice.round();
    if !twice.is_finite()
        || (twice - rounded).abs() > 1e-12
        || rounded < i32::MIN as f64
        || rounded > i32::MAX as f64
    {
        return Err(format!("power exponent {exponent} is not a half-integer"));
    }
    let num = rounded as i32;
    let (mut factor, mut exponent) = if num % 2 == 0 {
        (base, (num / 2).unsigned_abs())
    } else {
        (b.sqrt(base), num.unsigned_abs())
    };
    let one = one(b, scalar);
    let one = b.splat(lanes, one);
    if exponent == 0 {
        return Ok(one);
    }
    let mut result = factor.clone();
    exponent -= 1;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = b.mul(result, factor.clone());
        }
        exponent >>= 1;
        if exponent != 0 {
            factor = b.mul(factor.clone(), factor);
        }
    }
    Ok(if num < 0 { b.div(one, result) } else { result })
}

pub(super) fn comparison(direction: ComparisonDirection) -> CmpOp {
    match direction {
        ComparisonDirection::Eq => CmpOp::Eq,
        ComparisonDirection::Ne => CmpOp::Ne,
        ComparisonDirection::Lt => CmpOp::Lt,
        ComparisonDirection::Ge => CmpOp::Ge,
        ComparisonDirection::Gt => CmpOp::Gt,
        ComparisonDirection::Le => CmpOp::Le,
    }
}
