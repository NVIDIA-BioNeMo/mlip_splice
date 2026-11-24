// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether a produced value may be given the storage its destination names.
//!
//! The question is about storage and not about names: a place and a binding
//! produced into it are one register under two names, and a read of either is a
//! read of it. What follows keeps the two apart — which name denotes a slot's
//! storage as an arm runs, and which registers that arm has already written.

use std::collections::BTreeSet;

use super::kernel::K;
use crate::arena::{BlockId, OpId, ValueId};
use crate::op::OpKind;

/// What the storage a branch's results name is called as one of its arms runs,
/// and which of that storage the arm has already written.
///
/// `current` holds, per result slot, the one name that still denotes what its
/// storage holds — or nothing, where the arm has put something there that no
/// name denotes. A name is superseded once something is produced into it, and
/// the registers it stood in are superseded with it, so another name for the
/// same storage is caught too.
///
/// Names and registers are kept apart on purpose. A copy into a slot writes
/// that slot's registers; it does not move the value copied from, which stays
/// where it was and stays readable there.
#[derive(Clone, Default)]
struct Lineage {
    current: Vec<Option<ValueId>>,
    superseded: BTreeSet<ValueId>,
    held: BTreeSet<String>,
}

impl Lineage {
    /// One live name per result, to start with the place each names.
    fn of(into: &[Option<ValueId>]) -> Lineage {
        Lineage {
            current: into.iter().flatten().copied().map(Some).collect(),
            ..Lineage::default()
        }
    }

    /// The slot `value` is the live name of, or the slot whose storage it
    /// stands in — a write through another name for that storage is a write to
    /// the slot, whatever the write calls it.
    fn named(&self, k: &K<'_, impl super::text::Text>, value: ValueId) -> Option<usize> {
        if let Some(slot) = self.current.iter().position(|&held| held == Some(value)) {
            return Some(slot);
        }
        let regs = k.occupies(value)?;
        self.current.iter().position(|held| {
            held.and_then(|held| k.occupies(held))
                .is_some_and(|held| held.iter().any(|reg| regs.contains(reg)))
        })
    }

    /// Whether `value` names storage this arm has already written, so a write
    /// through it is one the walk cannot follow.
    fn written(&self, k: &K<'_, impl super::text::Text>, value: ValueId) -> bool {
        self.superseded.contains(&value)
            || (self.named(k, value).is_none()
                && k.occupies(value)
                    .is_some_and(|regs| regs.iter().any(|reg| self.held.contains(reg))))
    }

    /// Marks a name, and the storage it stands in, as written.
    fn supersede(&mut self, k: &K<'_, impl super::text::Text>, value: ValueId) {
        self.superseded.insert(value);
        if let Some(regs) = k.occupies(value) {
            self.held.extend(regs.iter().cloned());
        }
    }

    /// Marks a slot's storage as written, leaving it with no live name where
    /// what it now holds is a copy no name denotes.
    fn overwrite(&mut self, k: &K<'_, impl super::text::Text>, slot: usize) {
        if let Some(held) = self.current[slot] {
            self.supersede(k, held);
        }
        self.current[slot] = None;
    }

    /// Renames the slot `destination` names, if it names one, to `produced` —
    /// the storage is the same and the name for it has moved on.
    fn rename(
        &mut self,
        k: &K<'_, impl super::text::Text>,
        destination: ValueId,
        produced: ValueId,
    ) {
        let Some(named) = self.named(k, destination) else {
            return;
        };
        self.overwrite(k, named);
        self.supersede(k, destination);
        self.current[named] = Some(produced);
    }

    /// Whether any of `reads` names storage the arm has already written.
    ///
    /// The name a slot goes by now stands in that storage too, and reading it is
    /// reading what the arm just put there — which is the fold, not a stale
    /// read. Any other name for it is a name from before the write.
    fn stale(&self, k: &K<'_, impl super::text::Text>, reads: BTreeSet<ValueId>) -> bool {
        if !reads.is_disjoint(&self.superseded) {
            return true;
        }
        let older = reads
            .into_iter()
            .filter(|value| !self.current.contains(&Some(*value)))
            .collect();
        k.touches(&older, &self.held)
    }
}

impl<T: super::text::Text> K<'_, T> {
    /// Every place a value's own evaluation writes: the destinations named
    /// anywhere in its operand cone, regions included. An operand that owns a
    /// region can write storage on its way to a value, and that write happens
    /// where the operand is read.
    pub(super) fn writes_of(&self, value: ValueId, into: &mut BTreeSet<ValueId>) {
        let Some(op) = self.f.operand_op(value) else {
            return;
        };
        self.written_by(op, into);
    }

    /// As [`K::writes_of`], for the cone under one operation.
    fn written_by(&self, op: OpId, into: &mut BTreeSet<ValueId>) {
        into.extend(self.f.op(op).into.iter().flatten().copied());
        for index in 0..self.f.op(op).operands.len() {
            self.writes_of(self.f.op(op).operands[index], into);
        }
        for index in 0..self.f.op(op).regions.len() {
            let region = self.f.op(op).regions[index];
            for slot in 0..self.f.region(region).blocks.len() {
                let block = self.f.region(region).blocks[slot];
                for step in self.f.block_steps(block) {
                    self.written_by(step, into);
                }
                for step in 0..self.f.block(block).terminator.yields.len() {
                    self.writes_of(self.f.block(block).terminator.yields[step], into);
                }
            }
        }
    }

    /// The registers the storage a set of values occupies, taken together.
    /// Reuse is a question about storage and not about names: a place and a
    /// binding produced into it are one register under two names, and a read of
    /// either is a read of it.
    pub(super) fn storage_of(&self, values: impl Iterator<Item = ValueId>) -> BTreeSet<String> {
        values
            .filter_map(|value| self.occupies(value))
            .flat_map(|regs| regs.iter().cloned())
            .collect()
    }

    /// Whether anything in `reads` stands in storage `held` covers.
    pub(super) fn touches(&self, reads: &BTreeSet<ValueId>, held: &BTreeSet<String>) -> bool {
        reads.iter().any(|&value| {
            self.occupies(value)
                .is_some_and(|regs| regs.iter().any(|reg| held.contains(reg)))
        })
    }

    /// Whether an `if` writes the storage its results name whichever way it
    /// goes, so giving them that storage changes nothing it does.
    ///
    /// The else arm must hand the places straight back, holding no step and
    /// yielding exactly them: then the default written before the branch and the
    /// arm's own join are moves of a register onto itself, the predicate reads
    /// what it always read, and only the then arm is left to account for.
    pub(super) fn hands_the_places_back(&self, op: OpId) -> bool {
        let f = self.f;
        let into = &f.op(op).into;
        if into.is_empty() || into.iter().any(Option::is_none) {
            return false;
        }
        let else_block = f.region_block(f.op(op).regions[1]);
        if !f.block(else_block).ops.is_empty() {
            return false;
        }
        let back = &f.block(else_block).terminator.yields;
        if back.len() != into.len()
            || !back
                .iter()
                .zip(into)
                .all(|(&yielded, place)| *place == Some(yielded))
        {
            return false;
        }
        let then_block = f.region_block(f.op(op).regions[0]);
        let mut held = Lineage::of(into);
        self.lineage_holds(then_block, &mut held) && self.lands(then_block, &mut held)
    }

    /// Whether the arms write the storage the results name in an order that
    /// reads nothing they have already written — the demand a carried value
    /// makes, where the storage is the loop's own and a result produced into it
    /// is the update the body performs.
    pub(super) fn lands_in_order(&self, op: OpId) -> bool {
        let f = self.f;
        let into = &f.op(op).into;
        if into.is_empty() || into.iter().any(Option::is_none) {
            return false;
        }
        (0..f.op(op).regions.len()).all(|slot| {
            let block = f.region_block(f.op(op).regions[slot]);
            let mut held = Lineage::of(into);
            self.lineage_holds(block, &mut held) && self.lands(block, &mut held)
        })
    }

    /// Whether the values a block hands back may be written into the storage the
    /// slots name, one after another: a later one may not read what an earlier
    /// one has already landed on.
    fn lands(&self, block: BlockId, held: &mut Lineage) -> bool {
        let f = self.f;
        let yields = f.block(block).terminator.yields.clone();
        if yields.len() != held.current.len() {
            return false;
        }
        for (slot, &yielded) in yields.iter().enumerate() {
            if self.disturbs(yielded, held) {
                return false;
            }
            // Where the value already stands in the slot's registers the copy is
            // a move of a register onto itself and writes nothing. Otherwise it
            // writes them, and what they hold afterwards is a copy no name
            // denotes — the value copied from is still its own, where it was.
            let settled = held.current[slot] == Some(yielded)
                || match (self.occupies(yielded), held.current[slot]) {
                    (Some(from), Some(held)) => self.occupies(held) == Some(from),
                    _ => false,
                };
            if !settled {
                held.overwrite(self, slot);
            }
        }
        true
    }

    /// Whether every read of the storage a result is produced into comes before
    /// the step that writes it, following what that storage is called as the arm
    /// runs.
    ///
    /// A slot starts out called by the place its result names. A step produced
    /// into what a slot is called now renames it to what that step produced —
    /// the storage is the same and the name for it has moved on — and the name
    /// it had is superseded. Following the names is what sees a chain of them: a
    /// binding opened into a place and a loop folding across that binding write
    /// one storage under three names, and only the last is live.
    ///
    /// A block's own end is a read like any other, and a destination naming a
    /// name already superseded writes storage a live name stands for, which is a
    /// write this walk cannot follow and so one it refuses.
    fn lineage_holds(&self, block: BlockId, held: &mut Lineage) -> bool {
        let f = self.f;
        let ops = f.block_steps(block).collect::<Vec<_>>();
        for (index, &op) in ops.iter().enumerate() {
            for slot in 0..f.op(op).operands.len() {
                let operand = f.op(op).operands[slot];
                if held.stale(self, f.reads(operand)) {
                    return false;
                }
                // A structured operand may write storage on its way to a value,
                // and this walk cannot say where in the arm that lands.
                let mut writes = BTreeSet::new();
                self.writes_of(operand, &mut writes);
                if writes
                    .iter()
                    .any(|&place| held.named(self, place).is_some() || held.written(self, place))
                {
                    return false;
                }
            }
            // A block ending in exactly what its last step produces produces it
            // where the slots stand: the registers are handed down, so that step
            // writes them whatever its own destinations say.
            let handed = index + 1 == ops.len()
                && f.block(block).terminator.yields == f.op(op).results
                && !f.op(op).results.is_empty();
            if !self.regions_hold(op, held, handed) {
                return false;
            }
            for slot in 0..f.op(op).into.len() {
                let Some(destination) = f.op(op).into[slot] else {
                    continue;
                };
                if held.written(self, destination) {
                    return false;
                }
                held.rename(self, destination, f.op(op).results[slot]);
            }
            if handed {
                for slot in 0..f.op(op).results.len().min(held.current.len()) {
                    if f.op(op).into.get(slot).copied().flatten().is_none() {
                        held.overwrite(self, slot);
                        held.current[slot] = Some(f.op(op).results[slot]);
                    }
                }
            }
        }
        let yields = f.block(block).terminator.yields.clone();
        yields.iter().all(|&yielded| !self.disturbs(yielded, held))
    }

    /// Whether reading a value would read storage the arm has written, or write
    /// storage the walk is following. A value is read where it stands, and what
    /// it takes to evaluate it happens there too.
    fn disturbs(&self, value: ValueId, held: &Lineage) -> bool {
        if held.stale(self, self.f.reads(value)) {
            return true;
        }
        let mut writes = BTreeSet::new();
        self.writes_of(value, &mut writes);
        writes
            .iter()
            .any(|&place| held.named(self, place).is_some() || held.written(self, place))
    }

    /// The regions of one step, walked where that step stands.
    ///
    /// Where more than one runs in the same position they must leave the slots
    /// named alike, since nothing after could name the storage otherwise.
    ///
    /// A loop is walked as what it is. What it carries into a slot is that slot
    /// under the name the body has for it, so the body walks with its own
    /// parameter in hand and may not read the name it came in under — the loop
    /// writes that at the backedge. What the loop leaves the slot called is
    /// settled by the step's own destination.
    ///
    /// One walk of the body is enough, and the backedge needs no second. The
    /// only name that may read that storage inside is the parameter, which is
    /// the carry itself and holds on every iteration what the body is meant to
    /// see; every other name for it was superseded on the way in, so a read
    /// through one is refused where it stands rather than one trip later.
    fn regions_hold(&self, op: OpId, held: &mut Lineage, handed: bool) -> bool {
        let f = self.f;
        if f.op(op).regions.is_empty() {
            return true;
        }
        let around = matches!(f.op(op).kind, OpKind::Loop { .. });
        let mut settled: Option<Vec<Option<ValueId>>> = None;
        for slot in 0..f.op(op).regions.len() {
            let region = f.op(op).regions[slot];
            for inner in 0..f.region(region).blocks.len() {
                let block = f.region(region).blocks[inner];
                let mut branch = held.clone();
                // What the loop carries into a slot is that slot under the name
                // the body has for it. Inside, the body's parameter is the live
                // name and the one it came in under is the loop's to write.
                let mut aliased = Vec::new();
                if around {
                    for carry in 0..f.op(op).results.len() {
                        // Which slot this carry writes: the one it names, or —
                        // where the region hands its registers down — the one
                        // its result lands in, whatever the carry names.
                        let named = match f.op(op).into.get(carry).copied().flatten() {
                            Some(destination) => branch.named(self, destination),
                            None if handed && carry < branch.current.len() => Some(carry),
                            None => None,
                        };
                        let Some(named) = named else {
                            continue;
                        };
                        let param = f.block(block).params[1 + carry];
                        if let Some(destination) = f.op(op).into.get(carry).copied().flatten() {
                            branch.supersede(self, destination);
                        } else {
                            branch.overwrite(self, named);
                        }
                        branch.current[named] = Some(param);
                        aliased.push((named, param));
                    }
                }
                if !self.lineage_holds(block, &mut branch) {
                    return false;
                }
                // A region handed the slots' registers writes them at its own
                // end, one after another — unless it is a loop, which writes
                // them at the backedge instead.
                if handed && !around && !self.lands(block, &mut branch) {
                    return false;
                }
                // The name a carried slot came in under is the loop's own to
                // write, and the step's destination is what settles it.
                for carry in 0..f.op(op).into.len() {
                    if let Some(destination) = f.op(op).into[carry] {
                        branch.superseded.remove(&destination);
                    }
                }
                for &(_, param) in &aliased {
                    branch.superseded.remove(&param);
                }
                held.superseded.extend(branch.superseded.iter().copied());
                held.held.extend(branch.held.iter().cloned());
                let leaves = if around {
                    held.current.clone()
                } else {
                    branch.current.clone()
                };
                match &settled {
                    None => settled = Some(leaves),
                    Some(agreed) if *agreed == leaves => {}
                    Some(_) => return false,
                }
            }
        }
        if let Some(settled) = settled {
            held.current = settled;
        }
        true
    }
}
