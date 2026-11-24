// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The longest chain of floating-point additions a program builds.
//!
//! A floating-point sum of terms `t_i` is not the real sum: each addition
//! rounds, and a term carries its rounding into every addition it later passes
//! through. Writing `d` for the greatest number of additions any one term
//! passes through on its way to a result, the standard forward bound is
//!
//! ```text
//! |computed - exact| <= d * u * sum |t_i| / (1 - d * u)
//! ```
//!
//! with `u` the unit roundoff. Everything data-dependent lives in `sum |t_i|`;
//! `d` belongs to the program alone, so it can be read off the arena before any
//! kernel runs. A sequential accumulation of `n` terms has `d = n`, a balanced
//! tree has `d = ceil(log2 n)`, and a compensated sum has `d = 2` whatever `n`
//! is. Two programs computing the same polynomial can therefore be ordered by
//! how much of that polynomial's cancellation they are able to lose.
//!
//! The walk is an evaluation in the (max, +) semiring. A loop body is affine
//! there — its yield is `max(fresh, seed + slope)` in the depth its carried
//! value arrives with — so evaluating the body at two seeds recovers both
//! coefficients exactly, and the loop's own closed form follows without
//! unrolling it.

use std::collections::BTreeMap;

use crate::{
    BinOp, OpKind, Space, Trip, ValTy, WriteMode,
    arena::{BlockId, Function, OpId, ValueId},
    func::Implementation,
};

/// A probe seed large enough that no program's own chains reach it, so that a
/// body's response to it is its slope rather than its fresh part.
const PROBE: u64 = 1 << 40;

/// How many additions deep a launch's arithmetic goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdditionChain {
    /// Additions one term passes through inside one thread.
    pub within_thread: u64,
    /// Whether threads then fold their results into the same memory cell, so
    /// that the launch's fan-in extends the chain by as many additions as
    /// there are contributing threads or blocks — a number the arena cannot
    /// see, since it is the launch geometry's.
    pub folds_across_threads: bool,
}

/// The longest addition chain any value written to memory stands at the end of,
/// and `None` when some part of the program admits no bound: an unbounded loop,
/// a callee with no body, or a reduction through memory that never settles.
pub fn longest_addition_chain(function: &Function) -> Option<AdditionChain> {
    let mut shared = 0;
    for _ in 0..8 {
        let mut walk = Walk {
            shared,
            reached: 0,
            stored_to_shared: 0,
            folds_across_threads: false,
            unknown: false,
        };
        let mut env = BTreeMap::new();
        walk.block(function, function.entry(), &mut env);
        if walk.unknown {
            return None;
        }
        if walk.stored_to_shared <= shared {
            return Some(AdditionChain {
                within_thread: walk.reached,
                folds_across_threads: walk.folds_across_threads,
            });
        }
        shared = walk.stored_to_shared;
    }
    None
}

/// One evaluation of a program's depths, with the depth a shared-memory load
/// hands back held fixed.
///
/// Shared memory is one pool: a load from it stands at the deepest chain any
/// store into it ends. That is what makes a cross-thread reduction through
/// shared memory visible at all, and it costs only the precision of not
/// telling two shared arrays apart.
struct Walk {
    shared: u64,
    reached: u64,
    stored_to_shared: u64,
    folds_across_threads: bool,
    unknown: bool,
}

type Depths = BTreeMap<usize, u64>;

impl Walk {
    fn depth(&self, env: &Depths, value: ValueId) -> u64 {
        env.get(&value.index()).copied().unwrap_or(0)
    }

    fn widest(&self, env: &Depths, values: &[ValueId]) -> u64 {
        values
            .iter()
            .map(|&v| self.depth(env, v))
            .max()
            .unwrap_or(0)
    }

    fn block(&mut self, function: &Function, block: BlockId, env: &mut Depths) {
        for op in function.block(block).ops.clone() {
            self.op(function, op, env);
        }
    }

    fn set(&self, env: &mut Depths, values: &[ValueId], depth: u64) {
        for &value in values {
            env.insert(value.index(), depth);
        }
    }

    fn op(&mut self, function: &Function, op: OpId, env: &mut Depths) {
        let def = function.op(op);
        let operands = def.operands.clone();
        let results = def.results.clone();
        match &def.kind {
            OpKind::IntConst { .. }
            | OpKind::FloatConst { .. }
            | OpKind::Sreg(_)
            | OpKind::ActiveMask
            | OpKind::ConstLoad(_) => self.set(env, &results, 0),
            OpKind::Load { .. } => {
                let space = match function.ty(operands[0]) {
                    ValTy::Ptr(ptr) => ptr.space,
                    ValTy::Data(_) => return self.give_up(),
                };
                let depth = match space {
                    Space::Shared => self.shared,
                    Space::Global => 0,
                };
                self.set(env, &results, depth);
            }
            OpKind::Bin(BinOp::Add | BinOp::Sub) | OpKind::Fma => {
                let depth = self.widest(env, &operands) + 1;
                self.set(env, &results, depth);
            }
            OpKind::Offset
            | OpKind::Un(_)
            | OpKind::Bin(_)
            | OpKind::Cmp(_)
            | OpKind::Select
            | OpKind::Cast(_)
            | OpKind::Splat(_)
            | OpKind::Extract(_)
            | OpKind::Pack
            | OpKind::ShflDown
            | OpKind::ShflFrom
            | OpKind::WarpAny
            | OpKind::Bind => {
                let depth = self.widest(env, &operands);
                self.set(env, &results, depth);
            }
            OpKind::Write { mode, .. } => {
                let space = match function.ty(operands[0]) {
                    ValTy::Ptr(ptr) => ptr.space,
                    ValTy::Data(_) => return self.give_up(),
                };
                let written = self.depth(env, operands[1]);
                self.record(space, *mode == WriteMode::Add, written);
            }
            OpKind::FetchAdd => {
                let space = match function.ty(operands[0]) {
                    ValTy::Ptr(ptr) => ptr.space,
                    ValTy::Data(_) => return self.give_up(),
                };
                let added = self.depth(env, operands[1]);
                self.record(space, true, added);
                // It hands back what the cell held: whatever any thread put there.
                let depth = match space {
                    Space::Shared => self.shared.max(added),
                    Space::Global => added,
                };
                self.set(env, &results, depth);
            }
            OpKind::If => {
                let regions = def.regions.clone();
                let mut depths = vec![0u64; results.len()];
                for region in regions {
                    let block = function.region_block(region);
                    self.block(function, block, env);
                    let yields = function.block(block).terminator.yields.clone();
                    if yields.len() != results.len() {
                        return self.give_up();
                    }
                    for (slot, value) in depths.iter_mut().zip(&yields) {
                        *slot = (*slot).max(self.depth(env, *value));
                    }
                }
                for (&result, depth) in results.iter().zip(depths) {
                    env.insert(result.index(), depth);
                }
            }
            OpKind::Scope => {
                let regions = def.regions.clone();
                for region in regions {
                    let block = function.region_block(region);
                    self.block(function, block, env);
                }
            }
            OpKind::Loop { trip, .. } => {
                let trip = *trip;
                let regions = def.regions.clone();
                self.loop_op(function, &operands, &results, &regions, trip, env);
            }
            OpKind::Call(callee) => {
                let callee = callee.clone();
                let Implementation::Body(body) = &callee.implementation else {
                    return self.give_up();
                };
                let mut inner = BTreeMap::new();
                let parameters = body.block(body.entry()).params.clone();
                if parameters.len() != operands.len() {
                    return self.give_up();
                }
                for (parameter, &argument) in parameters.iter().zip(&operands) {
                    inner.insert(parameter.index(), self.depth(env, argument));
                }
                self.block(body, body.entry(), &mut inner);
                let returned = body.returned().to_vec();
                if returned.len() != results.len() {
                    return self.give_up();
                }
                for (&result, &value) in results.iter().zip(&returned) {
                    let depth = inner.get(&value.index()).copied().unwrap_or(0);
                    env.insert(result.index(), depth);
                }
            }
            OpKind::Break | OpKind::Ret | OpKind::Bar => {}
        }
    }

    /// One counted loop's closed form, from its body evaluated at two seeds.
    fn loop_op(
        &mut self,
        function: &Function,
        operands: &[ValueId],
        results: &[ValueId],
        regions: &[crate::arena::RegionId],
        trip: Trip,
        env: &mut Depths,
    ) {
        let [start, stop, step] = operands[..3] else {
            return self.give_up();
        };
        let initial = &operands[3..];
        let Some(trips) = self.trip_count(function, start, stop, step, trip) else {
            return self.give_up();
        };
        if regions.len() != 1 {
            return self.give_up();
        }
        let block = function.region_block(regions[0]);
        let parameters = function.block(block).params.clone();
        if parameters.len() != initial.len() + 1 || results.len() != initial.len() {
            return self.give_up();
        }
        let mut yielded = Vec::with_capacity(2);
        for seed in [0, PROBE] {
            let mut inner = env.clone();
            inner.insert(parameters[0].index(), 0);
            for parameter in &parameters[1..] {
                inner.insert(parameter.index(), seed);
            }
            self.block(function, block, &mut inner);
            let values = function.block(block).terminator.yields.clone();
            if values.len() != results.len() {
                return self.give_up();
            }
            yielded.push(
                values
                    .iter()
                    .map(|&value| self.depth(&inner, value))
                    .collect::<Vec<_>>(),
            );
        }
        for (index, &result) in results.iter().enumerate() {
            let fresh = yielded[0][index];
            let probed = yielded[1][index];
            // The body is affine in (max, +): the probe either survives, and its
            // excess is the chain the carried value gains per iteration, or it
            // does not, and the loop adds nothing to what arrives.
            let slope = probed.saturating_sub(PROBE);
            let arrives = self.depth(env, initial[index]);
            let depth = if slope == 0 {
                arrives.max(fresh)
            } else {
                // A term entering at the last iteration still passes the fresh
                // part; one entering first passes every iteration's slope.
                (arrives + trips.saturating_mul(slope))
                    .max(fresh + trips.saturating_sub(1).saturating_mul(slope))
            };
            env.insert(result.index(), depth);
        }
    }

    /// How many times one thread goes round, as a bound.
    ///
    /// A counted loop Loom emits starts at a thread or block index and stops at
    /// a constant extent, so a nonnegative start and constant stop and step
    /// bound the count. Anything else is refused rather than guessed.
    fn trip_count(
        &self,
        function: &Function,
        start: ValueId,
        stop: ValueId,
        step: ValueId,
        _trip: Trip,
    ) -> Option<u64> {
        let constant = |value: ValueId| match function.defining_op(value) {
            Some(op) => match function.op(op).kind {
                OpKind::IntConst { value, .. } => Some(value),
                _ => None,
            },
            None => None,
        };
        let stop = constant(stop)?;
        let step = constant(step)?;
        if step <= 0 || stop <= 0 {
            return None;
        }
        let first = constant(start).unwrap_or(0);
        if first < 0 {
            return None;
        }
        let span = stop.saturating_sub(first).max(0) as u64;
        Some(span.div_ceil(step as u64))
    }

    fn record(&mut self, space: Space, accumulating: bool, depth: u64) {
        match space {
            Space::Shared => {
                self.stored_to_shared = self.stored_to_shared.max(depth);
                if accumulating {
                    self.folds_across_threads = true;
                }
            }
            Space::Global => {
                self.reached = self.reached.max(depth);
                if accumulating {
                    self.folds_across_threads = true;
                }
            }
        }
    }

    fn give_up(&mut self) {
        self.unknown = true;
    }
}

#[cfg(test)]
mod tests;
