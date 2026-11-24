// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural algorithms over an SSA arena.
//!
//! An intermediate representation carries meaning its infrastructure does not
//! read: which instruction a operation stands for, what its attributes say,
//! which target it will be rendered to. The questions asked *about* the
//! program's shape — what reads what, what a set of operations reaches, what
//! nothing holds any more — do not read that meaning, and answering them once,
//! well, is what this crate is.
//!
//! An arena describes itself through [`Operations`]: how many operations and
//! values it holds, which operations are still live, and which values each
//! operation reads and defines. [`reach`] and [`uses`] are written against
//! that alone, so an arena carrying any payload gets those answers, and gets
//! them at the same cost.
//!
//! [`Ir`] is the arena this crate supplies for callers wanting one: a nested
//! operation, region and block tree with SSA results and block arguments,
//! extensible types and attributes, a dialect [`Registry`], [`verify()`], and
//! one deterministic generic syntax. It is deliberately dialect-neutral —
//! `xir` and `gpu_ir` each register their own without either being embedded
//! here.

pub mod arena;
pub mod diagnostic;
pub mod frozen;
pub mod ir;
pub mod name;
pub mod reach;
pub mod registry;
pub mod syntax;
pub mod types;
pub mod uses;
pub mod verify;

pub use diagnostic::{Diagnostic, Severity, SourceSpan};
pub use frozen::{
    FreezeMap, FrozenBlock, FrozenBlockId, FrozenIr, FrozenOperation, FrozenOperationId,
    FrozenRegion, FrozenRegionId, FrozenSuccessor, FrozenValue, FrozenValueDefinition,
    FrozenValueId,
};
pub use ir::{
    Block, BlockId, EditableIr, Ir, IrError, Operation, OperationId, Region, RegionId, Successor,
    Value, ValueDefinition, ValueId,
};
pub use name::{DialectName, InvalidName, OperationName};
pub use reach::{Reached, first_cycle, reachable};
pub use registry::{OperationDefinition, OperationTraits, Registry, RegistryError};
pub use syntax::{parse, print};
pub use types::{Attribute, AttributeData, Type, TypeData};
pub use uses::Uses;
pub use verify::verify;

/// An arena of operations over values, described by shape alone.
///
/// Operations and values are dense indices: an arena assigns them in order and
/// does not reuse them, so a question about one is a position rather than a
/// key. That is what lets the answers here be arrays.
///
/// An implementation reports operands and results through callbacks rather
/// than slices, because an arena is free to hold them in whatever form its own
/// work wants — this asks what they are, not how they are kept.
pub trait Operations {
    /// One past the greatest operation index, live or deleted.
    fn operation_count(&self) -> usize;

    /// One past the greatest value index.
    fn value_count(&self) -> usize;

    /// Whether this operation is still part of the program. A deleted
    /// operation keeps its index, so that a handle to it goes stale rather
    /// than coming to mean whatever was added next.
    fn is_live(&self, operation: usize) -> bool;

    /// Calls `read` with each value this operation reads.
    fn each_operand(&self, operation: usize, read: impl FnMut(usize));

    /// Calls `defined` with each value this operation produces.
    fn each_result(&self, operation: usize, defined: impl FnMut(usize));
}
