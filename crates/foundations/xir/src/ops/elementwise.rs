// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;

use smallvec::{SmallVec, smallvec};

use crate::dtype::DType;
use crate::op::{Elementwise, InferCtx, Op};
use crate::shape::{Shape, ShapeError};

use super::infer_shapes::{expect_arity, same_axes, same_domain_axes, same_dtype};

// ----------------------------------------------------------------------------

/// The direction of a `compare`, StableHLO-named.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComparisonDirection {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ComparisonDirection {
    pub fn name(self) -> &'static str {
        use ComparisonDirection::*;
        match self {
            Eq => "eq",
            Ne => "ne",
            Lt => "lt",
            Le => "le",
            Gt => "gt",
            Ge => "ge",
        }
    }
}

/// How a `compare` interprets its operands, StableHLO-named (`ComparisonType`).
/// `None` on the op means infer-from-element-type (StableHLO's default).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComparisonType {
    Float,
    TotalOrder,
    Signed,
    Unsigned,
}

impl ComparisonType {
    pub fn name(self) -> &'static str {
        use ComparisonType::*;
        match self {
            Float => "float",
            TotalOrder => "totalorder",
            Signed => "signed",
            Unsigned => "unsigned",
        }
    }
}

/// The scalar math an [`Elementwise`] op applies pointwise. Lowering-agnostic:
/// these name the operation, not any lowering. Variant arity (1/2/3) matches the
/// elementwise op's operand count.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarOp {
    // unary
    Negate,
    Abs,
    Exponential,
    ExponentialMinusOne,
    Log,
    LogPlusOne,
    Sqrt,
    Rsqrt,
    Cbrt,
    Tanh,
    Sine,
    Cosine,
    Tan,
    Floor,
    Ceil,
    RoundNearestEven,
    RoundNearestAfz,
    Sign,
    Not,
    CountLeadingZeros,
    Popcnt,
    IsFinite,
    Logistic,
    // binary
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Power,
    Remainder,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightArithmetic,
    ShiftRightLogical,
    Atan2,
    Compare(ComparisonDirection),
    // ternary
    Select,
    Clamp,
}

/// The shared elementwise type rule: arity check, all operand dims equal, and
/// the result dtype. `result_bool` (Compare/IsFinite) yields `bool`; otherwise
/// the result preserves the (common) operand dtype.
fn elementwise_infer(
    op: &'static str,
    inputs: &[Shape],
    arity: usize,
    result_bool: bool,
) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
    expect_arity(op, inputs, arity)?;
    for x in &inputs[1..] {
        same_axes(op, &inputs[0], x, "operand")?;
        same_dtype(op, &inputs[0], x, "operand")?;
    }
    let out_dtype = if result_bool {
        DType::Bool
    } else {
        inputs[0].dtype
    };
    Ok(smallvec![inputs[0].with_dtype(out_dtype)])
}

/// Declare distinct elementwise ops, each named 1:1 with StableHLO. Per row it
/// generates the (fieldless) struct, its `impl Op` (mnemonic, the shared
/// [`elementwise_infer`], `as_any`/`clone_box`, and `as_elementwise → Some`),
/// and its [`Elementwise`] impl (`scalar()`). `Compare` is special-cased: it
/// carries a `comparison_direction` and prints it as an attr.
macro_rules! def_elementwise {
    (
        $(
            $(#[$meta:meta])*
            $name:ident, $mn:literal, $arity:literal, $bool:literal, $scalar:expr
        )*
    ) => {$(
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name;

        impl Op for $name {
            fn mnemonic(&self) -> &'static str { $mn }

            fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
                elementwise_infer($mn, ctx.inputs(), $arity, $bool)
            }

            fn as_any(&self) -> &dyn Any { self }

            fn as_elementwise(&self) -> Option<&dyn Elementwise> { Some(self) }

            fn clone_box(&self) -> Box<dyn Op> { Box::new(self.clone()) }
        }

        impl Elementwise for $name {
            fn scalar(&self) -> ScalarOp { $scalar }
        }
    )*};
}

def_elementwise! {
    /// `-x`, shape-preserving.
    Negate, "negate", 1, false, ScalarOp::Negate
    Abs, "abs", 1, false, ScalarOp::Abs
    Exponential, "exponential", 1, false, ScalarOp::Exponential
    ExponentialMinusOne, "exponential_minus_one", 1, false, ScalarOp::ExponentialMinusOne
    Log, "log", 1, false, ScalarOp::Log
    LogPlusOne, "log_plus_one", 1, false, ScalarOp::LogPlusOne
    Sqrt, "sqrt", 1, false, ScalarOp::Sqrt
    Rsqrt, "rsqrt", 1, false, ScalarOp::Rsqrt
    Cbrt, "cbrt", 1, false, ScalarOp::Cbrt
    Tanh, "tanh", 1, false, ScalarOp::Tanh
    Sine, "sine", 1, false, ScalarOp::Sine
    Cosine, "cosine", 1, false, ScalarOp::Cosine
    Tan, "tan", 1, false, ScalarOp::Tan
    Floor, "floor", 1, false, ScalarOp::Floor
    Ceil, "ceil", 1, false, ScalarOp::Ceil
    RoundNearestEven, "round_nearest_even", 1, false, ScalarOp::RoundNearestEven
    RoundNearestAfz, "round_nearest_afz", 1, false, ScalarOp::RoundNearestAfz
    Sign, "sign", 1, false, ScalarOp::Sign
    Not, "not", 1, false, ScalarOp::Not
    CountLeadingZeros, "count_leading_zeros", 1, false, ScalarOp::CountLeadingZeros
    Popcnt, "popcnt", 1, false, ScalarOp::Popcnt
    /// `true` where `x` is finite (not inf/nan); yields a `bool` tensor.
    IsFinite, "is_finite", 1, true, ScalarOp::IsFinite
    Logistic, "logistic", 1, false, ScalarOp::Logistic

    Add, "add", 2, false, ScalarOp::Add
    Subtract, "subtract", 2, false, ScalarOp::Subtract
    Multiply, "multiply", 2, false, ScalarOp::Multiply
    Divide, "divide", 2, false, ScalarOp::Divide
    Maximum, "maximum", 2, false, ScalarOp::Maximum
    Minimum, "minimum", 2, false, ScalarOp::Minimum
    Power, "power", 2, false, ScalarOp::Power
    Remainder, "remainder", 2, false, ScalarOp::Remainder
    And, "and", 2, false, ScalarOp::And
    Or, "or", 2, false, ScalarOp::Or
    Xor, "xor", 2, false, ScalarOp::Xor
    ShiftLeft, "shift_left", 2, false, ScalarOp::ShiftLeft
    /// Arithmetic (sign-extending) right shift — distinct from the logical one.
    ShiftRightArithmetic, "shift_right_arithmetic", 2, false, ScalarOp::ShiftRightArithmetic
    /// Logical (zero-filling) right shift — distinct from the arithmetic one.
    ShiftRightLogical, "shift_right_logical", 2, false, ScalarOp::ShiftRightLogical
    Atan2, "atan2", 2, false, ScalarOp::Atan2
}

/// Elementwise `pred ? on_true : on_false`. The predicate is `bool` (scalar or
/// matching the branch shape); the branches share dtype+shape; the result is the
/// branch shape+dtype. Hand-written rather than in `def_elementwise!` because
/// `pred`'s dtype differs from the branches' (the shared rule forces them equal).
#[derive(Clone, Debug, PartialEq)]
pub struct Select;

impl Op for Select {
    fn mnemonic(&self) -> &'static str {
        "select"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        expect_arity("select", inputs, 3)?;
        let (p, a, b) = (&inputs[0], &inputs[1], &inputs[2]);
        if p.dtype != DType::Bool {
            return Err(ShapeError::new("select", "predicate must be bool"));
        }
        // The predicate is either a scalar or matches the branch shape.
        if !p.is_scalar() {
            same_axes("select", p, a, "predicate and branch")?;
        } else {
            same_domain_axes("select", p, a, "predicate and branch")?;
        }
        same_axes("select", a, b, "branch")?;
        same_dtype("select", a, b, "branch")?;
        Ok(smallvec![a.clone()])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_elementwise(&self) -> Option<&dyn Elementwise> {
        Some(self)
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

impl Elementwise for Select {
    fn scalar(&self) -> ScalarOp {
        ScalarOp::Select
    }
}

/// Elementwise `min(max(operand, lo), hi)`. The bounds `lo`/`hi` are either
/// scalars (broadcast) or match the operand shape; all three share dtype; the
/// result is the operand shape+dtype. Hand-written rather than in
/// `def_elementwise!` because scalar bounds are allowed (the shared rule forces
/// equal dims).
#[derive(Clone, Debug, PartialEq)]
pub struct Clamp;

impl Op for Clamp {
    fn mnemonic(&self) -> &'static str {
        "clamp"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        expect_arity("clamp", inputs, 3)?;
        let (lo, x, hi) = (&inputs[0], &inputs[1], &inputs[2]);
        // The bounds are either scalars or match the operand shape.
        if !lo.is_scalar() {
            same_axes("clamp", lo, x, "lower bound and operand")?;
        } else {
            same_domain_axes("clamp", lo, x, "lower bound and operand")?;
        }
        if !hi.is_scalar() {
            same_axes("clamp", hi, x, "upper bound and operand")?;
        } else {
            same_domain_axes("clamp", hi, x, "upper bound and operand")?;
        }
        same_dtype("clamp", lo, x, "operand")?;
        same_dtype("clamp", x, hi, "operand")?;
        Ok(smallvec![x.clone()])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_elementwise(&self) -> Option<&dyn Elementwise> {
        Some(self)
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

impl Elementwise for Clamp {
    fn scalar(&self) -> ScalarOp {
        ScalarOp::Clamp
    }
}

/// Elementwise comparison; yields a `bool` tensor. Carries its direction and an
/// optional comparison type (`None` = infer from element type, StableHLO default).
#[derive(Clone, Debug, PartialEq)]
pub struct Compare {
    pub comparison_direction: ComparisonDirection,
    pub compare_type: Option<ComparisonType>,
}

impl Op for Compare {
    fn mnemonic(&self) -> &'static str {
        "compare"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        elementwise_infer("compare", ctx.inputs(), 2, true)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        let mut a = vec![(
            "comparison_direction",
            self.comparison_direction.name().to_string(),
        )];
        if let Some(t) = self.compare_type {
            a.push(("compare_type", t.name().to_string()));
        }
        a
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_elementwise(&self) -> Option<&dyn Elementwise> {
        Some(self)
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

impl Elementwise for Compare {
    fn scalar(&self) -> ScalarOp {
        ScalarOp::Compare(self.comparison_direction)
    }
}
