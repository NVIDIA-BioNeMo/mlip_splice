// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a scalar is called in each instruction position, and what an
//! expression has become once part of it is in registers.

use crate::{BinOp, Evict, PtrTy, Scalar, Ty, ValTy};

pub(super) fn reg_class(s: Scalar) -> &'static str {
    match s {
        Scalar::Pred => "%p",
        // Sub-word ints have no register of their own width: they live in
        // 16-bit `%rs` registers, sign/zero-extended on load.
        Scalar::S8 | Scalar::U8 | Scalar::S16 | Scalar::U16 => "%rs",
        Scalar::S32 | Scalar::U32 => "%r",
        Scalar::S64 | Scalar::U64 => "%rd",
        Scalar::F16 | Scalar::BF16 => "%h",
        Scalar::F32 => "%f",
        Scalar::F64 => "%fd",
    }
}

/// Instruction type suffix for arithmetic.
pub(super) fn arith(s: Scalar) -> &'static str {
    match s {
        Scalar::Pred => "pred",
        Scalar::S8 => "s8",
        Scalar::U8 => "u8",
        Scalar::S16 => "s16",
        Scalar::U16 => "u16",
        Scalar::S32 => "s32",
        Scalar::U32 => "u32",
        Scalar::S64 => "s64",
        Scalar::U64 => "u64",
        Scalar::F16 => "f16",
        Scalar::BF16 => "bf16",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
    }
}

/// Memory / mov type suffix (bit width for ints and halves). Sub-word ints
/// ride their 16-bit register container — `b8` is not a legal reg/mov type.
pub(super) fn mem(s: Scalar) -> &'static str {
    match s {
        Scalar::Pred => "pred",
        Scalar::S8 | Scalar::U8 | Scalar::S16 | Scalar::U16 => "b16",
        Scalar::S32 | Scalar::U32 => "b32",
        Scalar::S64 | Scalar::U64 => "b64",
        Scalar::F16 | Scalar::BF16 => "b16",
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
    }
}

/// Load / store memory type: signed-aware for sub-word ints so the byte is
/// sign/zero-extended into its 16-bit register. Wider types move bits, so
/// they reuse [`mem`].
pub(super) fn load_type(s: Scalar) -> &'static str {
    match s {
        Scalar::S8 | Scalar::U8 | Scalar::S16 | Scalar::U16 => arith(s),
        _ => mem(s),
    }
}

pub(super) fn store_type(s: Scalar) -> &'static str {
    match s {
        Scalar::S8 | Scalar::U8 => arith(s),
        _ => mem(s),
    }
}

/// Whether an access of this type renders as one vector instruction rather
/// than as a lane at a time.
///
/// This is the table the PTX memory renderer selects from, read by the verifier so
/// the alignment a vector form needs is decided before anything is emitted:
/// two-byte lanes ride `b32` words, four- and eight-byte lanes have `v2`/`v4`
/// forms up to sixteen bytes, and everything else splits into element
/// accesses that ask nothing beyond an element.
pub fn vector_form(ty: Ty) -> bool {
    let size = ty.scalar.size();
    let bytes = size * ty.lanes as i64;
    ty.lanes > 1 && matches!((size, bytes), (2, 4 | 8 | 16) | (4, 8 | 16) | (8, 16))
}

pub(super) fn subword_wide(s: Scalar) -> Option<Scalar> {
    match s {
        Scalar::S8 | Scalar::S16 => Some(Scalar::S32),
        Scalar::U8 | Scalar::U16 => Some(Scalar::U32),
        _ => None,
    }
}

/// Multiplier and shift for signed `bits`-wide division by `d` (Hacker's
/// Delight 10-4): `n / d = sra(mulhi(M, n) [+ n | - n], s)` plus the sign
/// bit. Requires `2 <= |d| < 2^(bits-1)`. The multiplier is returned

#[derive(Clone, Debug)]
pub(super) enum Operand {
    Reg(String),
    ImmI(i64),
    ImmF(f64),
}

#[derive(Clone, Debug)]
pub(super) enum Val {
    Data {
        ops: Vec<Operand>,
        ty: Ty,
    },
    Ptr {
        reg: String,
        byte_off: i64,
        ty: PtrTy,
    },
}

impl Val {
    pub(super) fn ty(&self) -> ValTy {
        match self {
            Val::Data { ty, .. } => ValTy::Data(*ty),
            Val::Ptr { ty, .. } => ValTy::Ptr(*ty),
        }
    }

    pub(super) fn data(self) -> (Vec<Operand>, Ty) {
        match self {
            Val::Data { ops, ty } => (ops, ty),
            Val::Ptr { ty, .. } => {
                unreachable!("a data value is an address into {:?}", ty.elem)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum Binding {
    Data {
        regs: Vec<String>,
        ty: Ty,
    },
    /// A loop's index or one of its carried values. It reads like `Data`, but
    /// the loop writes it where the tree shows no statement — the compare and
    /// increment at the bottom, the carry across the backedge — so nothing may
    /// take its registers over while the loop is running.
    LoopState {
        regs: Vec<String>,
        ty: Ty,
    },
    Ptr {
        reg: String,
        ty: PtrTy,
    },
}

impl Binding {
    /// The registers this value occupies. A pointer occupies the one it is.
    pub(super) fn registers(&self) -> &[String] {
        match self {
            Binding::Data { regs, .. } | Binding::LoopState { regs, .. } => regs,
            Binding::Ptr { reg, .. } => std::slice::from_ref(reg),
        }
    }
}

/// What a float operation lowers to at a given scalar type and target.
pub(super) enum FloatLower {
    /// Native instruction at this scalar type.
    Native,
    /// Compute in f32, convert operands up and the result back down.
    ViaF32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum MathK {
    Bin(BinOp),
    Fma,
}

/// The `.L1::` eviction-priority qualifier for a global access, empty for
/// `Normal` — the plain instruction, byte-identical to an unhinted render.
pub(super) fn evict_qualifier(evict: Evict) -> &'static str {
    match evict {
        Evict::Normal => "",
        Evict::First => ".L1::evict_first",
        Evict::Last => ".L1::evict_last",
    }
}
