// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The threads a barrier needs, and what the arena can prove about them.
//!
//! `bar.sync` completes when every **non-exited** thread of the block has
//! arrived. Two things therefore break it, and only two:
//!
//! - a thread that is still running and takes a path around the barrier — an
//!   arm it did not enter, or an iteration it did not run;
//! - a thread that leaves the loop the barrier is in while others go round
//!   again.
//!
//! Leaving the kernel is not one of them: a thread that returned has exited,
//! and the barrier is satisfied by the rest. That is why the early-out guard
//! every grid-strided kernel starts with is legal above a barrier.
//!
//! What the arena can decide is which values every thread of a block holds
//! alike. A block index, a launch parameter and a constant are alike by
//! construction; a thread index is not, and neither is anything read from
//! memory, which may hold something different per thread. A loop's trip count
//! is not a value at all — it is how its bounds come out — so a loop carries
//! the producer's [`Trip`] fact, and a barrier inside one is admitted by that
//! fact rather than by a guess.

//! A barrier is reached wherever the operation holding it is *evaluated*, and a
//! placed step is not the only thing that is evaluated. An operand is computed
//! where it is read, so a loop standing for what it reduces to — a fold — runs
//! inside whichever block reads it, and the rules below apply to it there.
//! Walking only the placed steps would leave every such loop unchecked.

use super::structure::inner_blocks;
use super::{Checking, Result, Rule, Site};
use crate::Sreg;
use crate::arena::{BlockId, Def, OpId, ValueId};
use crate::effect::Sync;
use crate::op::{OpKind, Trip};

pub(super) fn check(c: &Checking<'_>) -> Result {
    let mut walk = Walk::default();
    block(c, c.f.entry(), &mut walk)
}

/// What this pass has decided and where it has been. The visited set is over
/// operations rather than blocks, because an operand cone is reached from every
/// read of it and is the same cone each time.
#[derive(Default)]
struct Walk {
    uniform: Uniformity,
    seen: fixedbitset::FixedBitSet,
}

/// What every thread of the block holds alike, remembered as it is decided.
#[derive(Default)]
struct Uniformity {
    known: Vec<Option<bool>>,
    walking: fixedbitset::FixedBitSet,
}

impl Uniformity {
    /// Whether every thread of the block holds the same `value`.
    ///
    /// A value defined in terms of itself — what a loop carries — proves
    /// nothing on this path, which is the conservative answer and the one that
    /// terminates.
    fn of(&mut self, c: &Checking<'_>, value: ValueId) -> bool {
        let index = value.index();
        if let Some(held) = self.known.get(index).copied().flatten() {
            return held;
        }
        self.walking.grow(c.f.num_values());
        if self.walking.contains(index) {
            return false;
        }
        self.walking.insert(index);
        let held = self.decide(c, value);
        self.walking.set(index, false);
        if self.known.len() < c.f.num_values() {
            self.known.resize(c.f.num_values(), None);
        }
        self.known[index] = Some(held);
        held
    }

    fn decide(&mut self, c: &Checking<'_>, value: ValueId) -> bool {
        match c.f.value(value).def {
            // What the kernel was launched with is the same in every thread;
            // what a region binds on the way in is the region's own question.
            Def::Param { block, index } => {
                if block == c.f.entry() {
                    return true;
                }
                let Some(op) = c.f.block_owner(block) else {
                    return false;
                };
                if !matches!(c.f.op(op).kind, OpKind::Loop { .. }) {
                    return false;
                }
                // The index counts from a start by a step; a carried value is
                // whatever the body made of it.
                index == 0
                    && [0, 2]
                        .iter()
                        .all(|&slot| self.of(c, c.f.op(op).operands[slot]))
            }
            Def::Result { op, index } => self.produced(c, op, index),
        }
    }

    fn produced(&mut self, c: &Checking<'_>, op: OpId, index: u32) -> bool {
        match c.f.op(op).kind {
            OpKind::Sreg(sreg) => matches!(sreg, Sreg::CtaidX | Sreg::CtaidY | Sreg::NtidX),
            // A read is uniform only if nothing wrote something else into the
            // cells it reaches, which the arena does not know.
            OpKind::Load { .. } | OpKind::ConstLoad(_) => false,
            // A collective reads what the warp holds, which is a warp's answer
            // and not the block's; an atomic reads a cell every thread updates.
            OpKind::ShflDown
            | OpKind::ShflFrom
            | OpKind::ActiveMask
            | OpKind::WarpAny
            | OpKind::FetchAdd => false,
            // A call runs a body this walk cannot see into.
            OpKind::Call(_) => false,
            OpKind::If => {
                self.of(c, c.f.op(op).operands[0]) && self.regions_uniform(c, op, index as usize)
            }
            OpKind::Loop { .. } => {
                (0..3).all(|slot| self.of(c, c.f.op(op).operands[slot]))
                    && self.of(c, c.f.op(op).operands[3 + index as usize])
                    && self.regions_uniform(c, op, index as usize)
            }
            _ => (0..c.f.op(op).operands.len()).all(|slot| self.of(c, c.f.op(op).operands[slot])),
        }
    }

    fn regions_uniform(&mut self, c: &Checking<'_>, op: OpId, index: usize) -> bool {
        inner_blocks(c, op)
            .collect::<Vec<_>>()
            .into_iter()
            .all(|block| {
                c.f.block(block)
                    .terminator
                    .yields
                    .get(index)
                    .is_some_and(|&yielded| self.of(c, yielded))
            })
    }
}

/// Whether a block, or anything under it, synchronises the whole block.
fn synchronises(c: &Checking<'_>, at: BlockId) -> bool {
    c.f.block_synchronisation(at) == Some(Sync::Block)
}

/// Whether a block, or anything under it that is not a loop of its own, leaves
/// the loop it is the body of.
fn breaks_out(c: &Checking<'_>, at: BlockId) -> bool {
    c.f.block_steps(at).any(|op| match c.f.op(op).kind {
        OpKind::Break => true,
        OpKind::Loop { .. } => false,
        _ => inner_blocks(c, op).any(|inner| breaks_out(c, inner)),
    })
}

/// The rules one operation answers to, wherever it is evaluated.
fn rules(c: &Checking<'_>, op: OpId, walk: &mut Walk) -> Result {
    match c.f.op(op).kind {
        OpKind::If => {
            let pred = c.f.op(op).operands[0];
            if inner_blocks(c, op).any(|inner| synchronises(c, inner)) && !walk.uniform.of(c, pred)
            {
                return c.fail(
                    Rule::Convergence,
                    Site::Op(op),
                    "it guards a barrier on a predicate the threads of a block need not \
                     agree on, so the threads that skip the arm never arrive",
                );
            }
        }
        OpKind::Loop { trip, .. } => {
            let inner = c.f.region_block(c.f.op(op).regions[0]);
            if synchronises(c, inner) {
                c.require(
                    trip == Trip::BlockUniform,
                    Rule::Convergence,
                    Site::Op(op),
                    || {
                        "its body holds a barrier and it is not stated to run the same \
                         number of times in every thread of the block"
                            .into()
                    },
                )?;
                c.require(
                    !breaks_out(c, inner),
                    Rule::Convergence,
                    Site::Op(op),
                    || {
                        "a thread that breaks out of it leaves the barriers in its body \
                         unreached by a thread still running"
                            .into()
                    },
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Everything reached by evaluating `op`: the rules it answers to, the blocks
/// of its regions, and the cones of its operands.
fn evaluated(c: &Checking<'_>, op: OpId, walk: &mut Walk) -> Result {
    walk.seen.grow(c.f.num_ops());
    if walk.seen.contains(op.index()) {
        return Ok(());
    }
    walk.seen.insert(op.index());
    rules(c, op, walk)?;
    for inner in inner_blocks(c, op).collect::<Vec<_>>() {
        block(c, inner, walk)?;
    }
    for slot in 0..c.f.op(op).operands.len() {
        cone(c, c.f.op(op).operands[slot], walk)?;
    }
    Ok(())
}

/// The operations evaluated to produce one value. A parameter is bound rather
/// than computed, so nothing runs to obtain it.
fn cone(c: &Checking<'_>, value: ValueId, walk: &mut Walk) -> Result {
    match c.f.value(value).def {
        Def::Param { .. } => Ok(()),
        Def::Result { op, .. } => evaluated(c, op, walk),
    }
}

fn block(c: &Checking<'_>, at: BlockId, walk: &mut Walk) -> Result {
    for index in 0..c.f.block(at).ops.len() {
        let op = c.f.block(at).ops[index];
        if !c.f.op(op).placed {
            continue;
        }
        rules(c, op, walk)?;
        for inner in inner_blocks(c, op).collect::<Vec<_>>() {
            block(c, inner, walk)?;
        }
        // What this step reads is computed here, so an unscheduled loop or
        // conditional in its operands runs in this block and answers here.
        for slot in 0..c.f.op(op).operands.len() {
            cone(c, c.f.op(op).operands[slot], walk)?;
        }
    }
    for slot in 0..c.f.block(at).terminator.yields.len() {
        cone(c, c.f.block(at).terminator.yields[slot], walk)?;
    }
    Ok(())
}
