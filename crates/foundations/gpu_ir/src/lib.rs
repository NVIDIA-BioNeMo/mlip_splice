// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed device-level GPU programs, and the PTX they emit.
//!
//! A program is one arena — see [`arena::Function`] — of values, operations,
//! blocks and regions. Values are `Scalar × lanes` vectors or typed pointers;
//! every operation is elementwise over lanes and every conversion is an explicit
//! [`op::OpKind::Cast`]. There is no textual type, no union, no inline asm, and
//! no arch dispatch in the arena: instruction selection (packed `f32x2` /
//! `f16x2` forms versus lane-split scalars) happens entirely in the PTX backend
//! ([`ptx`]), which lowers every verified program's arithmetic for every target
//! — no candidate is lost to the device generation over how it computes. What a
//! target must actually have an instruction for is decided before rendering, by
//! [`verify::verify_module`]; today that is the float widths `red.global.add`
//! covers.

pub mod addition_chain;
pub mod arena;
mod bits;
mod build;
pub mod dialect;
pub mod effect;
pub mod func;
mod inline;
pub mod op;
pub mod ptx;
pub mod stride;
pub mod verify;

pub use addition_chain::{AdditionChain, longest_addition_chain};
pub use arena::{
    ArenaTag, BlockId, Declaration, Def, FrozenBlockId, FrozenFunction, FrozenOpId, FrozenRegionId,
    FrozenValueId, Function, OpDef, OpId, RegionId, ValueDef, ValueId,
};
pub use build::{Arm, Body, Builder, Carry, Counted, Fragment, Import, LoopBody, OpenLoop, Value};
// `Sync` stays behind [`effect`]: at the root it would shadow the trait of
// that name for anyone importing this crate wholesale.
pub use effect::{Effect, Effects};
pub use func::{Access, DeviceFunction, Implementation, Launch, Signature};
pub use inline::inline_calls;
pub use op::{OpKind, Trip, WriteMode};
pub use verify::{
    Diagnostic, Rule, Site, VerifiedFunction, VerifiedModule, diagnostics, verified_module, verify,
    verify_frozen_function, verify_launch, verify_module,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    Pred,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl Scalar {
    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::BF16 | Scalar::F32 | Scalar::F64)
    }

    pub fn size(self) -> i64 {
        match self {
            Scalar::Pred | Scalar::S8 | Scalar::U8 => 1,
            Scalar::S16 | Scalar::U16 | Scalar::F16 | Scalar::BF16 => 2,
            Scalar::S32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::S64 | Scalar::U64 | Scalar::F64 => 8,
        }
    }
}

/// A data value type: `lanes` elements of `scalar`, contiguous in memory.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ty {
    pub scalar: Scalar,
    pub lanes: u8,
}

impl Ty {
    pub fn scalar(scalar: Scalar) -> Ty {
        Ty { scalar, lanes: 1 }
    }

    pub fn vec(scalar: Scalar, lanes: u8) -> Ty {
        assert!(lanes > 0, "zero-lane value");
        Ty { scalar, lanes }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    Global,
    Shared,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PtrTy {
    pub elem: Scalar,
    pub space: Space,
}

/// The type of any value: a data vector or a typed pointer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValTy {
    Data(Ty),
    Ptr(PtrTy),
}

impl ValTy {
    pub fn data(self) -> Ty {
        match self {
            ValTy::Data(t) => t,
            ValTy::Ptr(p) => panic!("expected a data value, got pointer to {:?}", p.elem),
        }
    }

    pub fn ptr(self) -> PtrTy {
        match self {
            ValTy::Ptr(p) => p,
            ValTy::Data(t) => panic!("expected a pointer, got {t:?}"),
        }
    }
}

/// Cache-eviction priority of one global access; `Normal` renders the plain
/// instruction. `First` marks data written or read once (streaming), `Last`
/// marks data expected to be re-read (a gather with fanout).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Evict {
    #[default]
    Normal,
    First,
    Last,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Sreg {
    TidX,
    CtaidX,
    CtaidY,
    NtidX,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    /// Correctly-rounded square root, per lane; f16/bf16 compute via f32.
    Sqrt,
    /// Two raised to the operand, to the hardware's approximation. The
    /// approximate instructions exist for f32 alone, so f16 and bf16 compute
    /// via f32 and f64 is not typable.
    Exp2,
    /// The base-two logarithm of the operand, to the hardware's approximation,
    /// under [`UnOp::Exp2`]'s precision rule.
    Log2,
    /// The sine of the operand in radians, under [`UnOp::Exp2`]'s precision rule.
    Sin,
    /// The cosine of the operand in radians, under [`UnOp::Exp2`]'s precision rule.
    Cos,
    /// The hyperbolic tangent of the operand, under [`UnOp::Exp2`]'s precision
    /// rule. Below sm_75, where no instruction exists, it is composed from
    /// [`UnOp::Exp2`].
    Tanh,
    /// The largest integer no greater than the operand, in the operand's own
    /// float type.
    Floor,
    /// The smallest integer no less than the operand, in the operand's own
    /// float type.
    Ceil,
    /// The nearest integer to the operand, halves to even, in the operand's own
    /// float type.
    RoundEven,
    /// How many leading bits of the operand are zero, in the operand's own
    /// integer type. A type narrower than a register counts its own width, not
    /// the register's.
    LeadingZeros,
    /// How many bits of the operand are one, in the operand's own integer type.
    OneBits,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamTy {
    /// Global-memory pointer to `Scalar` elements.
    Ptr(Scalar),
    Scalar(Scalar),
}

impl ParamTy {
    /// The type the value standing for this declaration holds.
    pub fn val_ty(&self) -> ValTy {
        match *self {
            ParamTy::Ptr(elem) => ValTy::Ptr(PtrTy {
                elem,
                space: Space::Global,
            }),
            ParamTy::Scalar(scalar) => ValTy::Data(Ty::scalar(scalar)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: ParamTy,
}

/// A kernel-scope `.shared` array; its declaration is a `Ptr(elem, Shared)`
/// value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedArr {
    pub name: String,
    pub elem: Scalar,
    pub len: i64,
}

/// The widest block any target admits, and so the widest `.maxntid` a launch
/// may promise.
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// The per-thread register cap `.maxnreg` admits.
pub const REGISTER_CAP: std::ops::RangeInclusive<u32> = 16..=255;
