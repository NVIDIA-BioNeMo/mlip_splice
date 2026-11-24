// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an operation does besides producing a value, as a property of its kind.
//!
//! An effect is not a separate walk over the program and not a declaration
//! beside it: [`crate::OpKind::effects`] answers for every operation, and a
//! caller that wants what a region does folds that answer over the region's
//! steps. Where an effect names memory it names the *operand* that addresses
//! it, so the address's provenance is a use-def walk from that operand rather
//! than a name to resolve.
//!
//! One operation may do several of these at once — a call to a function this
//! language has no body for reads one argument, writes another and synchronises
//! besides — so what a kind answers is a *set*. A kind that does nothing beyond
//! producing its value answers the empty one.

use crate::arena::{BlockId, Function, OpId, ValueId};

/// Which threads must arrive together for an operation to mean what it says.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sync {
    /// The warp's active lanes: a shuffle or a vote reads what they hold.
    Warp,
    /// Every non-exited thread of the block.
    Block,
}

/// One thing running an operation does to memory or to the threads around it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    /// Reads memory the operation addresses itself — a constant array — so no
    /// operand names it. Nothing writes that memory, but the read still happens
    /// where it is written and may not be speculated past a guard.
    ReadConst,
    Read {
        ptr: usize,
    },
    Write {
        ptr: usize,
    },
    ReadWrite {
        ptr: usize,
    },
    /// Observes which threads arrived, so where it stands is part of what it
    /// means.
    Barrier(Sync),
    /// Decides what runs next.
    Control,
}

impl Effect {
    /// What an operand is touched by, for a caller stating it as two facts.
    /// Nothing is not an effect, so an untouched operand has none.
    pub fn touching(ptr: usize, read: bool, written: bool) -> Option<Effect> {
        match (read, written) {
            (true, true) => Some(Effect::ReadWrite { ptr }),
            (true, false) => Some(Effect::Read { ptr }),
            (false, true) => Some(Effect::Write { ptr }),
            (false, false) => None,
        }
    }

    /// The operand holding the address this touches, when one does.
    pub fn address(self) -> Option<usize> {
        match self {
            Effect::Read { ptr } | Effect::Write { ptr } | Effect::ReadWrite { ptr } => Some(ptr),
            _ => None,
        }
    }

    pub fn reads(self) -> bool {
        matches!(
            self,
            Effect::Read { .. } | Effect::ReadWrite { .. } | Effect::ReadConst
        )
    }

    pub fn writes(self) -> bool {
        matches!(self, Effect::Write { .. } | Effect::ReadWrite { .. })
    }
}

/// Everything running one operation does beyond producing its values.
///
/// The empty set is what a pure operation does: the same operands give the same
/// result, and running it where it would not otherwise run changes nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects(Vec<Effect>);

impl Effects {
    pub fn none() -> Effects {
        Effects(Vec::new())
    }

    pub fn of(effect: Effect) -> Effects {
        Effects(vec![effect])
    }

    pub fn with(mut self, effect: Effect) -> Effects {
        self.0.push(effect);
        self
    }

    pub fn each(&self) -> &[Effect] {
        &self.0
    }

    pub fn is_pure(&self) -> bool {
        self.0.is_empty()
    }

    /// The strongest synchronisation any of them performs.
    pub fn sync(&self) -> Option<Sync> {
        self.0
            .iter()
            .filter_map(|held| match held {
                Effect::Barrier(sync) => Some(*sync),
                _ => None,
            })
            .max()
    }

    pub fn reads(&self) -> bool {
        self.0.iter().any(|held| held.reads())
    }

    pub fn writes(&self) -> bool {
        self.0.iter().any(|held| held.writes())
    }
}

impl Function {
    /// The strongest synchronisation running `op` performs: its own, whatever
    /// its operands perform on the way to a value, and whatever the regions it
    /// owns perform.
    ///
    /// This is the union the effect model is for. An operand can synchronise —
    /// a loop standing for the value it reduces to is evaluated where it is
    /// read, and its body may hold a barrier — so the walk follows operands as
    /// well as regions.
    pub fn synchronisation(&self, op: OpId) -> Option<Sync> {
        let mut held = self.op(op).kind.effects().sync();
        let widen = |held: Option<Sync>, found: Option<Sync>| match (held, found) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (held, None) => held,
            (None, found) => found,
        };
        for index in 0..self.op(op).operands.len() {
            held = widen(
                held,
                self.value_synchronisation(self.op(op).operands[index]),
            );
        }
        for index in 0..self.op(op).regions.len() {
            let region = self.op(op).regions[index];
            for slot in 0..self.region(region).blocks.len() {
                held = widen(
                    held,
                    self.block_synchronisation(self.region(region).blocks[slot]),
                );
            }
        }
        held
    }

    /// As [`Function::synchronisation`], for evaluating one value. A place was
    /// computed where it was bound, so reading one synchronises nothing.
    pub fn value_synchronisation(&self, value: ValueId) -> Option<Sync> {
        self.operand_op(value)
            .and_then(|op| self.synchronisation(op))
    }

    /// As [`Function::synchronisation`], for everything a block runs.
    pub fn block_synchronisation(&self, block: BlockId) -> Option<Sync> {
        let steps = (0..self.block(block).ops.len()).filter_map(|index| {
            let op = self.block(block).ops[index];
            self.op(op)
                .placed
                .then(|| self.synchronisation(op))
                .flatten()
        });
        let yields = (0..self.block(block).terminator.yields.len()).filter_map(|index| {
            self.value_synchronisation(self.block(block).terminator.yields[index])
        });
        steps.chain(yields).max()
    }
}
