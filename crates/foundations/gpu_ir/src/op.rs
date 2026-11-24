// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an operation is: its kind, and the operands, results and regions that
//! kind admits.
//!
//! An operation's meaning is decided by its kind, its operands, its result
//! types and the attributes the kind carries — nothing else. Every attribute is
//! part of this closed enum, so a program is `(kind, operands, results,
//! regions)` all the way down.

use std::sync::Arc;

use crate::effect::{Effect, Effects, Sync};
use crate::func::DeviceFunction;
use crate::{BinOp, CmpOp, Evict, Scalar, Sreg, UnOp};

/// A constant array the renderer emits once under `name`, and what it holds.
#[derive(Clone, Debug)]
pub struct ConstArray {
    pub name: String,
    pub scalar: Scalar,
    pub data: Arc<[u8]>,
}

/// How a write combines with what the memory already holds.
///
/// `Set` replaces it; `Add` adds to it as one indivisible step, which is what
/// makes several blocks able to fold into one cell. The mode is part of the
/// operation rather than of the caller, because the two differ in what they
/// read as well as what they write.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WriteMode {
    Set,
    Add,
}

/// Whether every thread of the block runs a loop the same number of times.
///
/// A loop's bounds are values, so how often a thread goes round is data the
/// arena cannot read. A producer that knows the count agrees says so here, and
/// that fact is what admits a barrier in the body: a barrier the threads of a
/// block reach a different number of times is one they never all reach.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Trip {
    PerThread,
    BlockUniform,
}

/// The operations a device program is made of.
///
/// They fall in two groups, and which group an operation is in decides how it
/// is evaluated. Every operation occupies one position in one block. **Stored**
/// operations — [`OpKind::Bind`], the writes, calls, regions and jumps — run
/// there once and give their results storage. Everything else is an expression:
/// its block position defines scope and order, while each result use explicitly
/// re-evaluates it. Reading one twice therefore computes it twice.
#[derive(Clone, Debug)]
pub enum OpKind {
    /// Integer (or predicate: 0/1) constant of any non-float scalar.
    IntConst {
        scalar: Scalar,
        value: i64,
    },
    /// Float constant of any float scalar; the backend materializes the bits.
    FloatConst {
        scalar: Scalar,
        value: f64,
    },
    Sreg(Sreg),
    /// `lanes` contiguous elements at the operand. `lanes > 1` requires the
    /// address to be `lanes * elem.size()`-aligned, which the verifier decides
    /// from the offset algebra and the binding owes for the base.
    Load {
        lanes: u8,
        nc: bool,
        evict: Evict,
    },
    /// `ptr + elems` in element units, like C pointer arithmetic.
    Offset,
    Un(UnOp),
    /// Elementwise; both sides must have the identical data type, except the
    /// `Shl`/`Shr` amount, which is a scalar `U32`.
    Bin(BinOp),
    /// `a * b + c`, fused: one rounding, never split. Contraction is the
    /// emitter's explicit choice.
    Fma,
    /// Elementwise comparison; produces `Pred` of the operands' lane count.
    Cmp(CmpOp),
    /// `pred ? a : b`, elementwise; a scalar predicate selects whole vectors.
    Select,
    /// Lane-preserving conversion between non-`Pred` scalars.
    Cast(Scalar),
    /// Scalar broadcast to `lanes`.
    Splat(u8),
    /// One lane of a vector, as a scalar.
    Extract(u8),
    /// A vector from scalar lanes (all of the same scalar type).
    Pack,
    /// `__shfl_down_sync(0xffffffff, a, delta)` over the full warp.
    ShflDown,
    /// The lanes of this warp that are participating here, as a `u32` bit per
    /// lane. It is what the warp holds at this point of the program, so where it
    /// stands is part of what it means.
    ActiveMask,
    /// One value read out of the lane named by the second operand, over exactly
    /// the lanes the third operand names. The source lane is absolute, so
    /// shifting, rotating and broadcasting are all this one operation over
    /// different lane arithmetic.
    ShflFrom,
    /// `or`-reduce a scalar predicate across the warp's active lanes: true in
    /// every active lane iff true in any. Collapses a per-lane guard into one
    /// warp-uniform predicate, so a branch on it does not diverge. Reduces over
    /// the active mask, so it is sound inside a partially-retired warp.
    WarpAny,
    ConstLoad(Arc<ConstArray>),
    /// Computes its operand here, once, into storage of its own. This is the
    /// only way a value becomes a place the rest of the block reads rather than
    /// an expression it recomputes.
    Bind,
    /// Contiguous write of the second operand (its lane count) at the first,
    /// combining as `mode` says.
    Write {
        mode: WriteMode,
        evict: Evict,
    },
    /// Adds the second operand to the scalar at the first as one indivisible
    /// step, and produces what was there before it. The result may be read or
    /// not; the addition happens either way, which is why this is a step of the
    /// block rather than an expression evaluated where it is read.
    FetchAdd,
    /// One predicate operand, a then and an else region, one result per value
    /// the regions yield.
    If,
    /// `start, stop, step` then one starting value per carried result. Its
    /// region's block takes the index and each carried value and yields what
    /// the next iteration receives.
    Loop {
        index: Scalar,
        trip: Trip,
    },
    /// A region that runs where it stands, producing nothing.
    Scope,
    /// Runs `callee` on the arguments, producing its results. A callee is a
    /// whole function, not an index into a table beside the program, so a call
    /// stays meaningful wherever the operation is carried.
    Call(Arc<DeviceFunction>),
    /// Leaves the innermost enclosing [`OpKind::Loop`].
    Break,
    /// Leaves the kernel. The thread exits, so a barrier after it is reached by
    /// the threads that remain.
    Ret,
    /// `__syncthreads`.
    Bar,
}

impl OpKind {
    /// Everything running an operation of this kind does beyond producing
    /// values.
    ///
    /// A region-owning kind reports only its own effects; what its body does is
    /// the union over that body, which the arena walks
    /// ([`crate::Function::synchronisation`]). A call reports its callee's
    /// ([`DeviceFunction::effects`]), which is the one place a set with more
    /// than one member comes from.
    pub fn effects(&self) -> Effects {
        match self {
            OpKind::Load { .. } => Effects::of(Effect::Read { ptr: 0 }),
            OpKind::ConstLoad(_) => Effects::of(Effect::ReadConst),
            OpKind::Write { mode, .. } => Effects::of(match mode {
                WriteMode::Set => Effect::Write { ptr: 0 },
                WriteMode::Add => Effect::ReadWrite { ptr: 0 },
            }),
            OpKind::FetchAdd => Effects::of(Effect::ReadWrite { ptr: 0 }),
            OpKind::Bar => Effects::of(Effect::Barrier(Sync::Block)),
            OpKind::ShflDown | OpKind::ShflFrom | OpKind::ActiveMask | OpKind::WarpAny => {
                Effects::of(Effect::Barrier(Sync::Warp))
            }
            OpKind::Call(callee) => callee.effects(),
            OpKind::If | OpKind::Loop { .. } | OpKind::Scope | OpKind::Break | OpKind::Ret => {
                Effects::of(Effect::Control)
            }
            _ => Effects::none(),
        }
    }

    /// Whether operations of this kind compute storage once at their block
    /// position rather than being re-evaluated at each result use.
    pub fn must_be_placed(&self) -> bool {
        matches!(
            self,
            OpKind::Bind
                | OpKind::Write { .. }
                | OpKind::FetchAdd
                | OpKind::Scope
                | OpKind::Call(_)
                | OpKind::Break
                | OpKind::Ret
                | OpKind::Bar
        )
    }

    /// Whether operations of this kind are either, as a loop is: one that folds
    /// an accumulator across a body is a step, and one standing for the value it
    /// reduces to is an operand of the expression it was written in.
    pub fn may_be_either(&self) -> bool {
        matches!(self, OpKind::If | OpKind::Loop { .. })
    }

    /// Whether an operation of this kind may be evaluated where it would not
    /// otherwise run, given its operands may be.
    ///
    /// Arithmetic on values already live may; a memory read may not, because
    /// the address may only be in range where the guard holds, and neither may
    /// a warp collective, which observes which lanes arrived.
    pub fn is_speculatable(&self) -> bool {
        self.effects().is_pure()
    }

    /// How many regions an operation of this kind owns.
    pub fn regions(&self) -> usize {
        match self {
            OpKind::If => 2,
            OpKind::Loop { .. } | OpKind::Scope => 1,
            _ => 0,
        }
    }
}

/// Two kinds are the same operation. Float constants compare by bits, so `+0`
/// and `-0` differ, and a constant load names the array it reads.
pub(crate) fn same_kind(a: &OpKind, b: &OpKind) -> bool {
    match (a, b) {
        (
            OpKind::IntConst {
                scalar: x,
                value: a,
            },
            OpKind::IntConst {
                scalar: y,
                value: b,
            },
        ) => x == y && a == b,
        (
            OpKind::FloatConst {
                scalar: x,
                value: a,
            },
            OpKind::FloatConst {
                scalar: y,
                value: b,
            },
        ) => x == y && a.to_bits() == b.to_bits(),
        (OpKind::Sreg(a), OpKind::Sreg(b)) => a == b,
        (
            OpKind::Load {
                lanes: a,
                nc: c,
                evict: e,
            },
            OpKind::Load {
                lanes: b,
                nc: d,
                evict: f,
            },
        ) => a == b && c == d && e == f,
        (OpKind::Offset, OpKind::Offset) => true,
        (OpKind::Un(a), OpKind::Un(b)) => a == b,
        (OpKind::Bin(a), OpKind::Bin(b)) => a == b,
        (OpKind::Fma, OpKind::Fma) => true,
        (OpKind::Cmp(a), OpKind::Cmp(b)) => a == b,
        (OpKind::Select, OpKind::Select) => true,
        (OpKind::Cast(a), OpKind::Cast(b)) => a == b,
        (OpKind::Splat(a), OpKind::Splat(b)) => a == b,
        (OpKind::Extract(a), OpKind::Extract(b)) => a == b,
        (OpKind::Pack, OpKind::Pack) => true,
        (OpKind::ShflDown, OpKind::ShflDown) => true,
        (OpKind::ActiveMask, OpKind::ActiveMask) => true,
        (OpKind::ShflFrom, OpKind::ShflFrom) => true,
        (OpKind::WarpAny, OpKind::WarpAny) => true,
        (OpKind::ConstLoad(a), OpKind::ConstLoad(b)) => {
            a.name == b.name && a.scalar == b.scalar && a.data == b.data
        }
        (OpKind::Bind, OpKind::Bind) => true,
        (
            OpKind::Write {
                mode: a,
                evict: from,
            },
            OpKind::Write { mode: b, evict: to },
        ) => a == b && from == to,
        (OpKind::FetchAdd, OpKind::FetchAdd) => true,
        (OpKind::If, OpKind::If) => true,
        (
            OpKind::Loop {
                index: a,
                trip: from,
            },
            OpKind::Loop { index: b, trip: to },
        ) => a == b && from == to,
        (OpKind::Scope, OpKind::Scope) => true,
        // Two calls are the same call when they run the same function: a callee
        // shared by both is one function, and two copies of one body are two.
        (OpKind::Call(a), OpKind::Call(b)) => Arc::ptr_eq(a, b),
        (OpKind::Break, OpKind::Break) => true,
        (OpKind::Ret, OpKind::Ret) => true,
        (OpKind::Bar, OpKind::Bar) => true,
        _ => false,
    }
}
