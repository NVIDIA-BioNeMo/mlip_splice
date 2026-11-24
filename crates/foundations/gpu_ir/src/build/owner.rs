// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Who owns a value, and what a finished body is.

use crate::arena::{ArenaTag, BlockId, Function, OpId, ValueId};
use crate::op::Trip;
use crate::{Scalar, Ty};

/// Which region a value was bound in.
///
/// A region opened for an arm or a loop body is a scope of its own, and what it
/// binds is gone once it closes. Ancestry decides what may be read: a value of
/// an enclosing scope is in scope here, a value of a scope this one is not
/// inside is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ScopeId(pub(super) u32);

impl ScopeId {
    pub(super) fn root() -> ScopeId {
        ScopeId(0)
    }
}

/// A value some builder bound.
///
/// What it holds is private and stays private: a value is read by handing it
/// back to its builder, never by opening it. Cloning is explicit and shares the
/// definition — an operand read twice is computed twice, which is what the
/// backend does with it.
#[derive(Clone, Debug)]
pub struct Value {
    tag: ArenaTag,
    scope: ScopeId,
    id: ValueId,
}

impl Value {
    pub(super) fn new(tag: ArenaTag, scope: ScopeId, id: ValueId) -> Value {
        Value { tag, scope, id }
    }

    /// The value this stands for, for a builder that owns it and is inside the
    /// scope it was bound in.
    pub(super) fn read(self, owner: ArenaTag, open: &[ScopeId]) -> ValueId {
        assert_eq!(
            self.tag, owner,
            "a value of arena {:?} was handed to arena {owner:?}",
            self.tag
        );
        assert!(
            open.contains(&self.scope),
            "a value bound in {:?} was read outside it; here {open:?} are open",
            self.scope
        );
        self.id
    }

    /// The value this stands for, without a scope check, for a caller that
    /// holds the whole body rather than a builder inside it.
    pub fn id_in(&self, function: &Function) -> ValueId {
        assert_eq!(
            self.tag,
            function.tag(),
            "a value of arena {:?} was read against arena {:?}",
            self.tag,
            function.tag()
        );
        self.id
    }
}

/// A finished body: the arena, and everything the kernel it belongs to
/// declares.
///
/// Values that outlive their builder are read against this: a root a producer
/// hands on stays paired with the arena it came from.
#[derive(Debug)]
pub struct Body {
    pub(super) function: Function,
}

impl From<Function> for Body {
    fn from(function: Function) -> Body {
        Body { function }
    }
}

impl Body {
    /// A finished arena as a body again, for a caller that holds one and means
    /// to go on writing it.
    pub fn of(function: Function) -> Body {
        Body { function }
    }

    pub fn function(&self) -> &Function {
        &self.function
    }

    pub fn into_function(self) -> Function {
        self.function
    }

    /// The operations the entry block runs, in order.
    pub fn ops(&self) -> &[OpId] {
        &self.function.block(self.function.entry()).ops
    }

    /// The operations that execute at their block position.
    pub fn steps(&self) -> Vec<OpId> {
        self.function.block_steps(self.function.entry()).collect()
    }
}

/// Operations a builder built without placing them, for its caller to place:
/// appended back, or taken as a list.
///
/// A fragment is built by one builder, so every value it holds belongs to that
/// builder.
pub struct Fragment {
    pub(super) tag: ArenaTag,
    pub(super) scope: ScopeId,
    pub(super) ops: Vec<OpId>,
}

impl Drop for Fragment {
    /// A fragment holds operations no block lists yet, so letting one go
    /// silently would strand them. It is placed, or discarded, by name.
    fn drop(&mut self) {
        assert!(
            self.ops.is_empty() || std::thread::panicking(),
            "a fragment of {} operations was neither placed nor discarded",
            self.ops.len()
        );
    }
}

impl Fragment {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// The operations this fragment holds, for a caller that places them
    /// itself.
    pub fn into_ops(mut self) -> Vec<OpId> {
        std::mem::take(&mut self.ops)
    }

    /// Those of them that execute where they stand, for a caller reading the
    /// fragment as a list of steps rather than placing it.
    ///
    /// The arena is read for this because a fragment holds operations, not the
    /// arena that answers for them.
    pub fn steps(&self, f: &Function) -> Vec<OpId> {
        self.ops
            .iter()
            .copied()
            .filter(|&op| f.is_step(op))
            .collect()
    }
}

/// What a counted loop counts: the type of its index, where it runs from and to,
/// how far it steps, and how often the threads of a block go round.
///
/// The trip fact belongs here because it is a fact *about these bounds*: how
/// often a loop goes round is how they come out, which the arena cannot read.
pub struct Counted {
    pub index: Scalar,
    pub start: Value,
    pub stop: Value,
    pub step: Value,
    pub trip: Trip,
}

/// What a carried value of a loop is: what it holds, where it starts, and the
/// place it updates, if it updates one.
#[derive(Clone)]
pub struct Carry {
    pub ty: Ty,
    pub init: Value,
    /// The value whose storage the carried result takes over. A loop that
    /// carries a fresh value has none; one that folds an accumulator names it,
    /// because that place is what it updates.
    pub into: Option<Value>,
}

/// One region of a construct that produces values, built and closed but not yet
/// attached to the operation that owns it.
#[must_use = "an arm is a region of an operation that does not exist yet; \
              dropping it drops the region"]
pub struct Arm {
    pub(super) ops: Vec<OpId>,
    pub(super) yields: Vec<ValueId>,
}

/// A loop body a builder has opened: the block holding it, and the index and
/// carried values it was entered with.
#[must_use = "an open loop body leaves a region open until it is closed"]
pub struct OpenLoop {
    pub(super) placement: ScopeId,
    pub(super) block: BlockId,
    pub(super) index_scalar: Scalar,
    pub(super) index: Value,
    pub(super) given: Vec<Value>,
    pub(super) carried: Vec<Carry>,
}

impl OpenLoop {
    /// What this iteration is on.
    pub fn index(&self) -> Value {
        self.index.clone()
    }

    /// What this iteration was given, one per carried value.
    pub fn given(&self) -> &[Value] {
        &self.given
    }
}

/// A loop body under construction: the block that holds it, the index and
/// carried values it was entered with, and what one iteration yields.
///
/// There is no way to take the operations out. A caller holding them would have
/// to write the loop itself, and the block would stop being what says which
/// loop they belong under; placing them is the only thing to do with them.
#[must_use = "a loop body is a body under a loop that does not exist yet; \
              dropping it drops the body"]
pub struct LoopBody {
    pub(super) tag: ArenaTag,
    /// Where the loop will be placed: the parent's scope, not the scope its
    /// operations were built in.
    pub(super) scope: ScopeId,
    pub(super) block: BlockId,
    pub(super) index: Scalar,
    pub(super) carried: Vec<Carry>,
}
