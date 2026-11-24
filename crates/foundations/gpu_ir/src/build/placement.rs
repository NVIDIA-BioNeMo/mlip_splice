// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Move and slice recorded operations while retaining their operands and ownership.

use super::Builder;
use crate::arena::{OpId, Readers};

impl Builder {
    /// Places operations this builder recorded earlier, in the order given.
    ///
    /// An operation occupies one position in one block, so this moves it:
    /// wherever it was listed before, it is listed here now. Placing a step
    /// into a region it was built outside of is exactly this move, and leaving
    /// the old entry behind would define what it produces twice.
    pub fn place_all(&mut self, ops: impl IntoIterator<Item = OpId>) {
        let named = ops.into_iter().collect::<Vec<_>>();
        let operation_count = self.function.num_ops();
        // An operand is evaluated where it is read, so it goes where its reader
        // goes: one left behind would be defined outside the region its reader
        // moved into, and read across a boundary that closes. Only operands
        // nothing else reads travel, since one read from outside is still owed
        // to whatever reads it there.
        let mut is_named = vec![false; operation_count];
        for op in &named {
            is_named[op.index()] = true;
        }
        let mut regions = std::mem::take(&mut self.walks.regions);
        self.function.reach_region_ops(&named, &mut regions);
        let mut ancestors = std::mem::take(&mut self.walks.ancestors);
        self.function.reach_operands(&named, &mut ancestors);
        let mut movable = ancestors
            .iter()
            .filter(|&at| !is_named[at] && !regions.contains(at))
            .map(|at| self.function.op_at(at))
            .filter(|&op| !self.function.is_step(op))
            .collect::<Vec<_>>();
        let mut is_movable = vec![false; operation_count];
        for op in &movable {
            is_movable[op.index()] = true;
        }
        self.walks.ancestors = ancestors;
        // Who reads each candidate is asked once per round, of every candidate,
        // so it is read off the arena once rather than once per question — and
        // not at all for the many callers whose operations carry nothing.
        let readers = if movable.is_empty() {
            Readers::default()
        } else {
            self.function.readers_with_roots(&self.returns)
        };
        while !movable.is_empty() {
            let retained = movable
                .iter()
                .copied()
                .filter(|&carried| {
                    !readers.has_use_outside(&self.function, carried, |user| {
                        is_named[user.index()]
                            || regions.contains(user.index())
                            || is_movable[user.index()]
                    })
                })
                .collect::<Vec<_>>();
            if retained.len() == movable.len() {
                break;
            }
            is_movable.fill(false);
            for op in &retained {
                is_movable[op.index()] = true;
            }
            movable = retained;
        }
        self.walks.regions = regions;
        let mut moved = named.clone();
        moved.extend(movable);
        let mut travelling = vec![false; operation_count];
        for op in &moved {
            travelling[op.index()] = true;
        }
        for frame in &mut self.frames {
            frame.ops.retain(|op| !travelling[op.index()]);
        }
        self.function.remove_all_from_blocks(&travelling);
        // The named operations keep the order the caller gave; an operand
        // travels ahead of the reader that carried it. Which operands travel
        // together is a property of the set, so their order is read off it
        // once: asking the whole set about each reader in turn asks a question
        // whose answer is a rank, and pays for it as a product.
        let mut rank = vec![usize::MAX; operation_count];
        for (index, &op) in moved.iter().enumerate() {
            if rank[op.index()] == usize::MAX {
                rank[op.index()] = index;
            }
        }
        let mut placed = Vec::with_capacity(moved.len());
        let mut written = vec![false; operation_count];
        for op in named {
            // What this reads, asked once: the reachable set of one operation
            // does not change while its carried operands are being written.
            // A reader reaches few operands and the travelling set may hold
            // thousands, so the small side is the one walked.
            let mut reaches = std::mem::take(&mut self.walks.reaches);
            self.function.reach_operands(&[op], &mut reaches);
            let mut carried = reaches
                .iter()
                .map(|at| self.function.op_at(at))
                .filter_map(|carried| {
                    let rank = rank[carried.index()];
                    (rank != usize::MAX && !written[carried.index()]).then_some((rank, carried))
                })
                .collect::<Vec<_>>();
            for &(_, carried) in &carried {
                written[carried.index()] = true;
            }
            carried.sort_unstable_by_key(|&(rank, _)| rank);
            placed.extend(carried.into_iter().map(|(_, carried)| carried));
            self.walks.reaches = reaches;
            written[op.index()] = true;
            placed.push(op);
        }
        self.frame_mut().ops.extend(placed);
    }

    /// How many operations the frame being built holds, for a caller about to
    /// append to it.
    pub(super) fn frame_len(&self) -> usize {
        self.frame().ops.len()
    }

    /// Holds operations created since `before`, together with `made`, in the
    /// order they were created.
    ///
    /// An operation is created after everything it reads: a copy builds an
    /// operand through the operation reading it, so creation order is already
    /// an order in which every operand stands before its reader. Freshly
    /// created operations belong to no block and are read by nothing outside
    /// themselves, so which of them travel and where they came from are not
    /// questions to ask — the general placement asks both, of the whole arena,
    /// and for a fresh import both answers are known.
    pub(super) fn reorder_fresh(&mut self, before: usize, made: &[OpId]) {
        let mut arrived = self.frame_mut().ops.split_off(before);
        arrived.extend_from_slice(made);
        arrived.sort_unstable_by_key(|op| op.index());
        arrived.dedup();
        self.frame_mut().ops.extend(arrived);
    }

    /// Replaces the steps being built, for a caller that reordered or sliced
    /// them.
    ///
    /// Choosing which steps a body runs, and in what order, is all this is: the
    /// operations are the ones already recorded, so nothing is copied and no
    /// value is renumbered.
    ///
    /// Operands are kept, whether they are named here or were already held:
    /// every operation belongs to a block, and one is evaluated where it is
    /// read rather than at a position, so dropping it would leave it owned by
    /// no block. Which operands are read is a consequence of the steps, so a
    /// caller states its steps and need not track the rest.
    pub fn replace_placed(&mut self, ops: Vec<OpId>) {
        let held = std::mem::take(&mut self.frame_mut().ops);
        let (kept, named): (Vec<OpId>, Vec<OpId>) =
            ops.into_iter().partition(|&op| self.function.is_step(op));
        // An operand is evaluated where it is read, so the ones this frame goes
        // on holding are exactly those the surviving steps reach. One reached
        // by no step here was read only by a step the slice dropped, and
        // keeping it would leave it reading what those steps defined.
        let reached_set = self.function.operands_reached_by(&kept);
        let mut reached = vec![false; self.function.num_ops()];
        for op in &reached_set {
            reached[op.index()] = true;
        }
        // A value is readable after the operation defining it, so an operand
        // stands before whatever reads it. Emitting each step's operands just
        // before it, deepest first, is that order: what an operand itself
        // reads is emitted ahead of it by the same rule.
        let mut ops = Vec::with_capacity(kept.len() + reached_set.len());
        let mut written = vec![false; self.function.num_ops()];
        for &step in &kept {
            self.emit_operands_before(step, &reached, &mut written, &mut ops);
            ops.push(step);
        }
        let mut dropped = Vec::new();
        let mut emitted = vec![false; self.function.num_ops()];
        for op in &ops {
            emitted[op.index()] = true;
        }
        for op in held.into_iter().chain(named) {
            if emitted[op.index()] {
                continue;
            }
            if !self.function.is_step(op) && reached[op.index()] {
                ops.push(op);
            } else {
                dropped.push(op);
            }
        }
        self.frame_mut().ops = ops;
        // Every operation belongs to a block, and this frame is the block it
        // belonged to. One the slice neither runs nor reaches is now held by
        // nothing, so it stops being part of the program rather than staying
        // live and homeless — unless something elsewhere holds or reads it.
        let held_elsewhere = self.held_anywhere();
        dropped.retain(|op| !held_elsewhere[op.index()]);
        let spared = self.function.forget_unheld(&dropped);
        // What the arena would not let go of is still the program's, so this
        // frame goes on holding it.
        self.frame_mut().ops.extend(spared);
    }

    /// Every operation any frame this builder has open, or any block it has
    /// closed, lists.
    ///
    /// What is held is a property of the frames and the blocks, so it is read
    /// off them once. Asking about one operation at a time reads every frame
    /// and every block per question, and the questions come by the thousand.
    fn held_anywhere(&self) -> Vec<bool> {
        let mut held = vec![false; self.function.num_ops()];
        for op in self.frames.iter().flat_map(|frame| frame.ops.iter()) {
            held[op.index()] = true;
        }
        self.function.mark_listed_ops(&mut held);
        held
    }

    /// Emits the operands `op` reads, and what those read in turn, ahead of
    /// where `op` will stand. Anything already emitted stays where it is, so
    /// each operand is written once, before its first reader.
    ///
    /// The walk is iterative and each operation is visited once: a body can
    /// hold thousands of operands, and revisiting them would cost more than
    /// the slice it serves.
    fn emit_operands_before(
        &self,
        op: OpId,
        reached: &[bool],
        written: &mut [bool],
        into: &mut Vec<OpId>,
    ) {
        // Post-order: an operand is written after everything it reads, which
        // is what puts it before everything that reads it.
        let mut walk = vec![(op, false)];
        while let Some((held, expanded)) = walk.pop() {
            if expanded {
                if !written[held.index()] {
                    written[held.index()] = true;
                    into.push(held);
                }
                continue;
            }
            if written[held.index()] {
                continue;
            }
            if held != op {
                walk.push((held, true));
            }
            for index in 0..self.function.op(held).operands.len() {
                let operand = self.function.op(held).operands[index];
                if let Some(defining) = self.function.operand_op(operand)
                    && reached[defining.index()]
                    && !written[defining.index()]
                {
                    walk.push((defining, false));
                }
            }
        }
    }
}
