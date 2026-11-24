// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The steps of a body: bindings, writes, conditionals and counted loops, each
//! emitted where the block it sits in says it belongs.

use std::collections::BTreeSet;

use super::kernel::K;
use super::types::{Binding, arith, load_type, store_type};
use crate::arena::{BlockId, OpId, ValueId};
use crate::op::{OpKind, WriteMode};
use crate::{Scalar, Space, Ty, ValTy};

impl<T: super::text::Text> K<'_, T> {
    /// The places a list of steps reads once, plus the place each binding takes
    /// over — a binding written into storage that outlives it is a use of what
    /// that storage held.
    fn add_step_reads(&self, ops: &[OpId], into: &mut BTreeSet<ValueId>) {
        for &op in ops {
            self.f.add_op_reads(op, into);
            if let Some(destination) = self.f.op(op).destination(0)
                && matches!(self.f.op(op).kind, OpKind::Bind)
            {
                into.insert(destination);
            }
        }
    }

    /// The results a step produces, if it produces any.
    fn produced(&self, op: OpId) -> &[ValueId] {
        match self.f.op(op).kind {
            OpKind::If | OpKind::Loop { .. } => &self.f.op(op).results,
            _ => &[],
        }
    }

    fn placed_ops(&self, block: BlockId) -> Vec<OpId> {
        self.f.block_steps(block).collect()
    }

    /// Whether a block ends by yielding exactly what its last stored operation
    /// produces, each in place — so those values can be produced straight into
    /// the region's own registers.
    fn passes_through(&self, block: BlockId, results: &[ValueId]) -> bool {
        let Some(last) = self.placed_ops(block).last().copied() else {
            return false;
        };
        let produced = self.produced(last);
        let yields = &self.f.block(block).terminator.yields;
        produced.len() == results.len()
            && !produced.is_empty()
            && yields.len() == produced.len()
            && yields
                .iter()
                .zip(produced)
                .zip(results)
                .all(|((&value, &produced), &result)| {
                    value == produced && self.f.ty(produced) == self.f.ty(result)
                })
    }

    /// The registers a loop's carried values live in.
    ///
    /// A carried value that names a place takes it over — then the register it
    /// would have been copied from is the one it runs in. Only a place this
    /// region gave storage to: one from further out outlives the loop, and
    /// writing it across iterations would make the emitted code mutate a value
    /// scoped here, where a discarded loop would leave an ancestor's storage
    /// changed. Reuse also needs every starting value to read no place another
    /// carried value takes over, so no register is claimed while another still
    /// starts from it.
    fn carried_regs(&mut self, op: OpId, tys: &[Ty]) -> Vec<Vec<String>> {
        let f = self.f;
        let into = f.op(op).into.clone();
        let inits = f.op(op).operands[3..].to_vec();
        let places = into.iter().flatten().copied().collect::<BTreeSet<_>>();
        let coalesce = into
            .iter()
            .zip(&inits)
            .all(|(place, &init)| *place == Some(init) || f.reads(init).is_disjoint(&places));
        // What each named place holds, and whether it may be taken over at all.
        let held = into
            .iter()
            .zip(tys)
            .map(|(place, &ty)| {
                place
                    .and_then(|place| self.data_place_here(place))
                    .filter(|&(regs, kept)| kept == ty && !self.reserves(regs))
                    .map(|(regs, _)| regs.to_vec())
            })
            .collect::<Vec<_>>();
        // Two carried values in one register are one value: distinct names for
        // one storage collide just as a repeated name would. Unless every named
        // place is its own storage, none is taken over.
        let claimed = held.iter().flatten().flatten().collect::<BTreeSet<_>>();
        let apart = claimed.len() == held.iter().flatten().map(Vec::len).sum::<usize>()
            && into
                .iter()
                .zip(&held)
                .all(|(place, regs)| place.is_none() || regs.is_some());
        held.into_iter()
            .zip(tys)
            .map(|(held, &ty)| match held {
                Some(regs) if coalesce && apart => regs,
                _ => self.regs(ty),
            })
            .collect()
    }

    /// Writes the values an iteration yields into the carried registers. Where
    /// more than one is carried and a yield reads one of them, every yield goes
    /// through a fresh register first, so the update is simultaneous.
    fn carry_into(&mut self, params: &[ValueId], regs: &[Vec<String>], next: &[ValueId]) {
        let carried = params.iter().copied().collect::<BTreeSet<_>>();
        let tys = params
            .iter()
            .map(|&param| self.f.data_ty(param))
            .collect::<Vec<_>>();
        // A yield that reads or writes a carried register — under whatever name,
        // and anywhere in what it takes to evaluate it — must be read before any
        // of them is written. One carried value is enough for that: a yield may
        // write the register it lands in on its way to a value.
        let claimed = regs.iter().flatten().cloned().collect::<BTreeSet<_>>();
        let staged = next.iter().enumerate().any(|(slot, &value)| {
            let reads = self.f.reads(value);
            let mut writes = BTreeSet::new();
            self.writes_of(value, &mut writes);
            // A yield runs in the register it lands in — that is what carrying
            // is, not a hazard. Reading another carry's register, under whatever
            // name, is one; so is writing any of them on the way to a value,
            // which one carried value is enough for.
            let others = regs
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != slot)
                .flat_map(|(_, regs)| regs.iter().cloned())
                .collect::<BTreeSet<_>>();
            (params.len() > 1 && !reads.is_disjoint(&carried))
                || self.touches(&reads, &others)
                || self.touches(&writes, &claimed)
        });
        if staged {
            // The carried registers are spoken for while the yields are read, so
            // nothing evaluated on the way may be produced into one.
            let held = regs.iter().flatten().cloned().collect::<Vec<_>>();
            let mark = self.reserve(&held);
            let staged_regs = tys
                .iter()
                .zip(next)
                .map(|(&ty, &value)| {
                    let staging = self.regs(ty);
                    self.eval_into(value, &staging, ty);
                    staging
                })
                .collect::<Vec<_>>();
            self.release(mark);
            for ((&ty, regs), staging) in tys.iter().zip(regs).zip(staged_regs) {
                for (dst, src) in regs.iter().zip(&staging) {
                    self.ins(format_args!(
                        "mov.{} {dst}, {src};",
                        super::types::mem(ty.scalar)
                    ));
                }
            }
            return;
        }
        for ((&ty, regs), &value) in tys.iter().zip(regs).zip(next) {
            self.eval_into(value, regs, ty);
        }
    }

    /// Lowers a region's block in its own scope and writes what it yields into
    /// the result registers, which every region of the operation shares.
    /// `dangerous` is what runs once each enclosing region has ended.
    fn region(
        &mut self,
        results: &[ValueId],
        regs: &[Vec<String>],
        block: BlockId,
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
        resident: Option<&[ValueId]>,
    ) {
        self.open += 1;
        // Where a region ends by yielding exactly what its last step produces,
        // that step produces it here: the region's registers are handed down as
        // the destination, and the yield is the values arriving rather than a
        // copy of them.
        let passed = self.passes_through(block, results);
        let ops = self.placed_ops(block);
        let yields = self.f.block(block).terminator.yields.clone();
        let last = ops.len() - usize::from(passed);
        self.lowered(&ops, last, &yields, dangerous, pass);
        if passed {
            let into = self.inherited(dangerous, &[], &yields);
            let onward = self.inherited(pass, &[], &yields);
            // What the registers already hold is only known to still be there
            // if nothing ran in between.
            let resident = resident.filter(|_| last == 0);
            self.produce(ops[last], regs, &into, &onward, resident);
        } else {
            self.yield_into(results, regs, &yields);
        }
        self.open -= 1;
    }

    /// Lowers a result-producing step into `destination`.
    fn produce(
        &mut self,
        op: OpId,
        destination: &[Vec<String>],
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
        resident: Option<&[ValueId]>,
    ) {
        match self.f.op(op).kind {
            OpKind::If => {
                self.conditional(op, &[], &[], dangerous, pass, Some(destination));
            }
            OpKind::Loop { .. } => {
                self.counted_loop(op, &[], &[], dangerous, Some(destination), resident);
            }
            _ => unreachable!("only a region-owning step produces values"),
        }
    }

    /// Lowers a list of steps, telling each both what its own list has left to
    /// do and what runs after every region enclosing that list has ended.
    pub(super) fn statements(
        &mut self,
        ops: &[OpId],
        yields: &[ValueId],
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
    ) {
        self.lowered(ops, ops.len(), yields, dangerous, pass);
    }

    /// Lowers the first `count` of `ops`. What follows a step is read from the
    /// whole list, not the part being lowered: a step left for its caller to
    /// lower still runs after these, and still reads what it reads.
    fn lowered(
        &mut self,
        ops: &[OpId],
        count: usize,
        yields: &[ValueId],
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
    ) {
        for position in 0..count {
            if self.f.op(ops[position]).placed {
                self.step(ops[position], &ops[position + 1..], yields, dangerous, pass);
            }
        }
    }

    /// The places a value entering a region may not be allocated over: what an
    /// enclosing list still reads once that region has ended.
    fn inherited(
        &self,
        dangerous: &BTreeSet<ValueId>,
        tail: &[OpId],
        yields: &[ValueId],
    ) -> BTreeSet<ValueId> {
        let mut inherited = dangerous.clone();
        self.add_step_reads(tail, &mut inherited);
        for &value in yields {
            self.f.add_reads(value, &mut inherited);
        }
        inherited
    }

    /// As [`K::inherited`], for the body of a loop: the body runs again after
    /// this iteration ends, so all of it comes after any step in it — once
    /// around the backedge.
    fn inherited_by_body(
        &self,
        dangerous: &BTreeSet<ValueId>,
        tail: &[OpId],
        yields: &[ValueId],
        block: BlockId,
    ) -> BTreeSet<ValueId> {
        let mut inherited = self.inherited(dangerous, tail, yields);
        self.f.add_block_reads(block, &mut inherited);
        inherited
    }

    /// Writes a region's yields into the result registers, which every region of
    /// the operation shares.
    fn yield_into(&mut self, results: &[ValueId], regs: &[Vec<String>], values: &[ValueId]) {
        for ((&result, regs), &value) in results.iter().zip(regs).zip(values) {
            let ty = self.f.data_ty(result);
            self.eval_into(value, regs, ty);
        }
    }

    /// A predicate and two arms, producing one value per result.
    fn conditional(
        &mut self,
        op: OpId,
        tail: &[OpId],
        yields: &[ValueId],
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
        destination: Option<&[Vec<String>]>,
    ) {
        let f = self.f;
        let pred = f.op(op).operands[0];
        let results = f.op(op).results.clone();
        let into = f.op(op).into.clone();
        let then_block = f.region_block(f.op(op).regions[0]);
        let else_block = f.region_block(f.op(op).regions[1]);
        // One register per result, before the branch, so every arm writes the
        // same one and the join needs no copy. A result is a new value; whether
        // it may run in the register of the place it is produced into is what
        // the clauses below decide, and they decide it about storage rather than
        // about names — a place and a binding produced into it are one register
        // under two names, and a read of either is a read of it.
        let taken = into
            .iter()
            .flatten()
            .map(|&place| self.storage_of(std::iter::once(place)))
            .collect::<Vec<_>>();
        let places = taken.iter().flatten().cloned().collect::<BTreeSet<_>>();
        // Two results in one register are two values in one register, and the
        // second would land on the first. Every named place must be its own
        // storage, whatever it is called.
        let distinct = taken.iter().map(BTreeSet::len).sum::<usize>() == places.len();
        let mut arms = BTreeSet::new();
        f.add_block_reads(then_block, &mut arms);
        f.add_block_reads(else_block, &mut arms);
        let mut asked = BTreeSet::new();
        f.add_reads(pred, &mut asked);
        // Evaluating the predicate may itself write storage. Reading a place is
        // something the arms do too and says nothing on its own, but writing one
        // does: the predicate runs between the default and the arms, so a result
        // given that storage would be the predicate's leavings whichever way the
        // branch goes. No takeover survives it.
        let mut asks = BTreeSet::new();
        self.writes_of(pred, &mut asks);
        let untouched = !self.touches(&asks, &places);
        // What runs after this step: the rest of its own list, what that list
        // hands back, and what runs once every region enclosing it has ended.
        let after = self.inherited(dangerous, tail, yields);
        // What runs after it in this pass alone, which is what a loop's own
        // storage is read by before the backedge comes round.
        let ahead = self.inherited(pass, tail, yields);
        // One decision for the whole result set, as with the registers
        // themselves: a default written before the branch must not land on a
        // register another result still starts from, so unless every result
        // qualifies, none does.
        //
        // A result qualifies one of two ways. Nothing reads the place again —
        // not the predicate, not either region, and nothing that runs after this
        // step, its own list included — so the place is dead from here on and
        // its registers are the ones to allocate into.
        let dead_from_here = !self.touches(&arms, &places)
            && !self.touches(&asked, &places)
            && !self.touches(&after, &places);
        // Or the branch writes those registers anyway, and nothing reads the
        // place once it has. An accumulator folded under a guard is the shape:
        // reuse there is not an optimization over the arms — it is what the arms
        // already do, written once instead of copied twice. The arms may read
        // the place, because they are what writes it; what follows may not,
        // because naming the place the result replaced is naming storage that
        // has moved on. The predicate needs no clause of its own here: the else
        // arm hands the places back, so the default written before the predicate
        // is a move of a register onto itself.
        let eligible = distinct
            && untouched
            && (dead_from_here
                || (!self.touches(&after, &places) && self.hands_the_places_back(op)));
        // What a loop carries is its own storage: the loop writes it at the
        // backedge, so a result produced into it is the update the body performs
        // and reading it next time round is that update being carried. Reading
        // it in this pass after the branch is not, and neither is the predicate,
        // which is evaluated after a default that may already have landed.
        let carried = distinct
            && untouched
            && !self.touches(&ahead, &places)
            && !self.touches(&asked, &places)
            && self.lands_in_order(op);
        // The storage each result could take over, where its place names one it
        // may: a carried value where the loop's own demand is met, a data place
        // where the branch's is. A carried value's storage is what an
        // accumulator update targets, and only a builder that named it can mean
        // that — the loop writes it at the backedge, so a result produced into
        // it is the update the body performs.
        //
        // Nothing is taken over unless every result that names a place can be:
        // one falling back while its siblings reuse is not the decision the
        // clauses above made.
        let candidates = results
            .iter()
            .zip(&into)
            .map(|(&result, place)| {
                let ty = f.data_ty(result);
                let place = (*place)?;
                let (regs, kept) = self
                    .loop_carried(place)
                    .filter(|_| carried)
                    .or_else(|| self.data_place(place).filter(|_| eligible))?;
                (kept == ty && !self.reserves(regs)).then(|| regs.to_vec())
            })
            .collect::<Vec<_>>();
        let all = candidates
            .iter()
            .zip(&into)
            .all(|(regs, place)| place.is_none() || regs.is_some());
        let regs = match destination {
            Some(destination) => destination.to_vec(),
            None => results
                .iter()
                .zip(candidates)
                .map(|(&result, regs)| match regs {
                    Some(regs) if all => regs,
                    _ => self.regs(f.data_ty(result)),
                })
                .collect::<Vec<_>>(),
        };
        // A default that may be evaluated where the predicate fails is written
        // before the branch and overwritten by the then arm, which spends one
        // label instead of a join.
        let else_yields = f.block(else_block).terminator.yields.clone();
        let defaulted = !results.is_empty()
            && f.block(else_block).ops.is_empty()
            && else_yields.iter().all(|&value| f.is_speculatable(value));
        if defaulted {
            self.yield_into(&results, &regs, &else_yields);
        }
        let (pops, pty) = self.eval(pred).data();
        assert_eq!(pty, Ty::scalar(Scalar::Pred), "if on {pty:?}");
        let p = self.materialize(&pops[0], Scalar::Pred);
        let lelse = self.label("else");
        self.ins(format_args!("@!{p} bra {lelse};"));
        let inner = self.inherited(dangerous, tail, yields);
        let onward = self.inherited(pass, tail, yields);
        let hoisted = defaulted.then_some(else_yields.as_slice());
        self.region(&results, &regs, then_block, &inner, &onward, hoisted);
        let else_empty = f.block(else_block).ops.is_empty() && else_yields.is_empty();
        if defaulted || else_empty {
            self.write_label(&lelse);
        } else {
            let lend = self.label("end");
            self.ins(format_args!("bra {lend};"));
            self.write_label(&lelse);
            self.region(&results, &regs, else_block, &inner, &onward, None);
            self.write_label(&lend);
        }
        // The results become visible only after the branch: an arm reads
        // whatever the enclosing scope bound before it.
        for (&result, regs) in results.iter().zip(regs) {
            let ty = f.data_ty(result);
            self.bind(result, Binding::Data { regs, ty });
        }
    }

    /// A counted loop, carrying one value per entry of its carried results.
    ///
    /// Gives back the registers each carried value runs in, which is what a loop
    /// standing for the value it reduces to reads.
    pub(super) fn counted_loop(
        &mut self,
        op: OpId,
        tail: &[OpId],
        yields: &[ValueId],
        dangerous: &BTreeSet<ValueId>,
        destination: Option<&[Vec<String>]>,
        resident: Option<&[ValueId]>,
    ) -> Vec<Vec<String>> {
        let f = self.f;
        let OpKind::Loop { index: ty, .. } = f.op(op).kind else {
            unreachable!("a counted loop")
        };
        assert!(
            matches!(ty, Scalar::S32 | Scalar::U32 | Scalar::S64 | Scalar::U64),
            "loop variable of {ty:?}"
        );
        let (start, stop, step) = (
            f.op(op).operands[0],
            f.op(op).operands[1],
            f.op(op).operands[2],
        );
        let inits = f.op(op).operands[3..].to_vec();
        let into = f.op(op).into.clone();
        let results = f.op(op).results.clone();
        let block = f.region_block(f.op(op).regions[0]);
        let params = f.block(block).params.clone();
        let (index_param, carried_params) = params.split_first().expect("a loop binds its index");
        let tys = results
            .iter()
            .map(|&result| f.data_ty(result))
            .collect::<Vec<_>>();
        // The carried registers are allocated and initialized here, outside the
        // loop, where the places they take over stood.
        let cregs = match destination {
            Some(destination) => destination.to_vec(),
            None => self.carried_regs(op, &tys),
        };
        // The carried registers are written one after another, so a starting
        // value that reads a register an earlier one lands in must be read
        // before that happens. Where one does, every starting value goes through
        // a register of its own first and the carried ones are written together.
        let claimed = cregs.iter().flatten().cloned().collect::<BTreeSet<_>>();
        let crossed = inits.iter().enumerate().any(|(index, &init)| {
            let earlier = cregs[..index]
                .iter()
                .flatten()
                .cloned()
                .collect::<BTreeSet<_>>();
            let mut writes = BTreeSet::new();
            self.writes_of(init, &mut writes);
            (!earlier.is_empty() && self.touches(&f.reads(init), &earlier))
                || self.touches(&writes, &claimed)
        });
        if crossed {
            // The carried registers are spoken for while the starting values are
            // read, so nothing evaluated on the way may be produced into one.
            let held = cregs.iter().flatten().cloned().collect::<Vec<_>>();
            let mark = self.reserve(&held);
            let staged = inits
                .iter()
                .zip(&tys)
                .map(|(&init, &carried_ty)| {
                    let staging = self.regs(carried_ty);
                    self.eval_into(init, &staging, carried_ty);
                    staging
                })
                .collect::<Vec<_>>();
            self.release(mark);
            for ((regs, staging), &carried_ty) in cregs.iter().zip(&staged).zip(&tys) {
                for (dst, src) in regs.iter().zip(staging) {
                    self.ins(format_args!(
                        "mov.{} {dst}, {src};",
                        super::types::mem(carried_ty.scalar)
                    ));
                }
            }
        }
        for (index, ((&init, regs), &carried_ty)) in inits.iter().zip(&cregs).zip(&tys).enumerate()
        {
            if crossed {
                break;
            }
            // A starting value already written into these registers — the
            // default of the `if` handing them down — does not need writing
            // again, if it is the same value bit for bit.
            let already_there =
                resident
                    .and_then(|values| values.get(index))
                    .is_some_and(|&value| {
                        f.is_speculatable(value)
                            && f.is_speculatable(init)
                            && f.same_value(value, init)
                    });
            let held = into[index] == Some(init)
                && self
                    .data_place_here(init)
                    .is_some_and(|(regs_of, _)| regs_of == regs.as_slice());
            if !already_there && !held {
                self.eval_into(init, regs, carried_ty);
            }
        }
        for ((&param, regs), &carried_ty) in carried_params.iter().zip(&cregs).zip(&tys) {
            self.bind(
                param,
                Binding::LoopState {
                    regs: regs.clone(),
                    ty: carried_ty,
                },
            );
        }
        self.open += 1;
        let t = Ty::scalar(ty);
        let regs = self.regs(t);
        self.eval_into(start, &regs, t);
        let vreg = regs[0].clone();
        self.bind(*index_param, Binding::LoopState { regs, ty: t });
        // loop-invariant bound and step evaluate once, up front
        let (sops, sty) = self.eval(stop).data();
        assert_eq!(sty, t, "loop bound type");
        let stop_op = self.operand(&sops[0], ty);
        let (stepops, stepty) = self.eval(step).data();
        assert_eq!(stepty, t, "loop step type");
        let step_op = self.operand(&stepops[0], ty);
        let ltop = self.label("loop");
        let lend = self.label("done");
        self.write_label(&ltop);
        let p = self.reg(Scalar::Pred);
        self.ins(format_args!(
            "setp.lt.{} {p}, {vreg}, {stop_op};",
            arith(ty)
        ));
        self.ins(format_args!("@!{p} bra {lend};"));
        self.loop_ends.push(lend.clone());
        let carrying = self.carry(&cregs);
        let body_ops = self.placed_ops(block);
        let body_yields = f.block(block).terminator.yields.clone();
        let inherited = self.inherited_by_body(dangerous, tail, yields, block);
        // Nothing outside the loop can name what its body carries, so within an
        // iteration the body starts with nothing behind it.
        self.statements(&body_ops, &body_yields, &inherited, &BTreeSet::new());
        self.carry_into(carried_params, &cregs, &body_yields);
        self.released(carrying);
        self.loop_ends.pop();
        self.ins(format_args!("add.{} {vreg}, {vreg}, {step_op};", arith(ty)));
        self.ins(format_args!("bra {ltop};"));
        self.write_label(&lend);
        self.open -= 1;
        // A loop that is a step of the body produces places: what it carried is
        // read from the registers it left them in. One standing for the value it
        // reduces to produces nothing of the sort — it is an operand, evaluated
        // wherever it is read, and giving its result storage would make a second
        // read free where the program says it runs again.
        if self.f.op(op).placed {
            for ((&result, regs), &carried_ty) in results.iter().zip(&cregs).zip(&tys) {
                self.bind(
                    result,
                    Binding::Data {
                        regs: regs.clone(),
                        ty: carried_ty,
                    },
                );
            }
        }
        cregs
    }

    fn step(
        &mut self,
        op: OpId,
        tail: &[OpId],
        yields: &[ValueId],
        dangerous: &BTreeSet<ValueId>,
        pass: &BTreeSet<ValueId>,
    ) {
        let f = self.f;
        match &f.op(op).kind {
            OpKind::Bind => {
                let init = f.op(op).operands[0];
                let result = f.op(op).results[0];
                let place = f.op(op).destination(0);
                // A binding produced into a place writes that place. While a
                // write is landing, its destinations are reserved: a binding
                // that would take one over gets storage of its own instead, so
                // nothing mutates what the landing is still to read.
                match f.ty(result) {
                    ValTy::Data(t) => {
                        let held = place.map(|place| self.place_regs(place).0);
                        let regs = match held {
                            Some(regs) if !self.reserves(&regs) => regs,
                            _ => self.regs(t),
                        };
                        self.eval_into(init, &regs, t);
                        self.bind(result, Binding::Data { regs, ty: t });
                    }
                    ValTy::Ptr(p) => {
                        let held = place.map(|place| self.place_pointer(place).0);
                        let reg = match held {
                            Some(reg) if !self.reserves(std::slice::from_ref(&reg)) => reg,
                            _ => self.reg(match p.space {
                                Space::Global => Scalar::U64,
                                Space::Shared => Scalar::U32,
                            }),
                        };
                        self.assign_ptr(&reg, p, init);
                        self.bind(result, Binding::Ptr { reg, ty: p });
                    }
                }
            }
            OpKind::Write { mode, evict } => {
                let (mode, evict) = (*mode, *evict);
                let (ops, ty) = self.eval(f.op(op).operands[1]).data();
                let p = self.eval(f.op(op).operands[0]);
                match mode {
                    WriteMode::Set => self.store(&p, &ops, ty, evict),
                    WriteMode::Add => self.reduce_add(&p, &ops, ty),
                }
            }
            // The addition runs either way; what the result is read by decides
            // whether the previous value is asked for. Nothing reads it, so
            // nothing is bound and no register is spent.
            OpKind::FetchAdd => {
                let result = f.op(op).results[0];
                let ty = f.data_ty(result);
                let (ops, held) = self.eval(f.op(op).operands[1]).data();
                assert_eq!(held, ty, "atomic add value type");
                let p = self.eval(f.op(op).operands[0]);
                if f.is_used(result) {
                    let regs = self.regs(ty);
                    self.fetch_add(&p, &ops[0], ty, Some(&regs[0]));
                    self.bind(result, Binding::Data { regs, ty });
                } else {
                    self.fetch_add(&p, &ops[0], ty, None);
                }
            }
            OpKind::If => self.conditional(op, tail, yields, dangerous, pass, None),
            OpKind::Loop { .. } => {
                self.counted_loop(op, tail, yields, dangerous, None, None);
            }
            OpKind::Scope => {
                let block = f.region_block(f.op(op).regions[0]);
                let ops = self.placed_ops(block);
                self.open += 1;
                let inherited = self.inherited(dangerous, tail, yields);
                let onward = self.inherited(pass, tail, yields);
                self.statements(&ops, yields, &inherited, &onward);
                self.open -= 1;
            }
            OpKind::Break => {
                let lend = self
                    .loop_ends
                    .last()
                    .expect("break outside any loop")
                    .clone();
                self.ins(format_args!("bra {lend};"));
            }
            OpKind::Ret => self.ins(format_args!("bra $EXIT;")),
            OpKind::Bar => self.ins(format_args!("bar.sync 0;")),
            OpKind::Call(callee) => {
                let callee = callee.clone();
                self.call(op, &callee);
            }
            // Every other kind is an operand; the verifier settled that.
            kind => unreachable!("{kind:?} is not a step of a rendered body"),
        }
    }

    /// One call to a function declared elsewhere: the arguments into its
    /// parameter space, the call, and what it hands back read out again.
    ///
    /// A call to a function this language defines never reaches here — it was
    /// run where it stood — so the callee is a declaration, and the contract it
    /// states is what the parameter space is laid out from. Each parameter is a
    /// slot of its own width and is written and read through the same typed
    /// instructions a global access of that scalar uses, so a sub-word integer
    /// crosses as its own byte and comes back sign- or zero-extended.
    fn call(&mut self, op: OpId, callee: &crate::DeviceFunction) {
        let signature = callee
            .signature()
            .expect("a rendered call runs a function declared elsewhere");
        self.calls += 1;
        let at = format!("{}_{}", callee.name, self.calls);
        // The arguments are read into registers first: evaluating one may emit
        // instructions, and those belong outside the parameter block.
        let held = (0..self.f.op(op).operands.len())
            .map(|slot| {
                let value = self.f.op(op).operands[slot];
                match signature.parameters[slot].ty {
                    crate::ParamTy::Ptr(elem) => {
                        let ty = crate::PtrTy {
                            elem,
                            space: Space::Global,
                        };
                        let reg = self.reg(Scalar::U64);
                        self.assign_ptr(&reg, ty, value);
                        (reg, "b64", ValTy::Ptr(ty))
                    }
                    crate::ParamTy::Scalar(scalar) => {
                        let (ops, ty) = self.eval(value).data();
                        let reg = self.materialize(&ops[0], scalar);
                        (reg, store_type(scalar), ValTy::Data(ty))
                    }
                }
            })
            .collect::<Vec<_>>();
        self.ins(format_args!("{{"));
        for (slot, (reg, word, ty)) in held.iter().enumerate() {
            self.ins(format_args!(
                ".param .{} {at}_param_{slot};",
                super::param_space(*ty)
            ));
            self.ins(format_args!("st.param.{word} [{at}_param_{slot}], {reg};"));
        }
        let results = self.f.op(op).results.clone();
        let returned = results.first().map(|&result| self.f.ty(result));
        if let Some(ty) = returned {
            self.ins(format_args!(
                ".param .{} {at}_retval0;",
                super::param_space(ty)
            ));
        }
        let arguments = (0..held.len())
            .map(|slot| format!("{at}_param_{slot}"))
            .collect::<Vec<_>>()
            .join(", ");
        let takes = match arguments.is_empty() {
            true => String::new(),
            false => format!(", ({arguments})"),
        };
        match returned {
            None => self.ins(format_args!("call {}{takes};", callee.name)),
            Some(_) => self.ins(format_args!("call ({at}_retval0), {}{takes};", callee.name)),
        }
        // The one value a PTX function hands back becomes the call's result,
        // read out of the return parameter before the block that declared it
        // ends.
        let handed = returned.map(|ty| match ty {
            ValTy::Ptr(ptr) => {
                let reg = self.reg(Scalar::U64);
                self.ins(format_args!("ld.param.b64 {reg}, [{at}_retval0];"));
                Binding::Ptr { reg, ty: ptr }
            }
            ValTy::Data(ty) => {
                let regs = self.regs(ty);
                self.ins(format_args!(
                    "ld.param.{} {}, [{at}_retval0];",
                    load_type(ty.scalar),
                    regs[0]
                ));
                Binding::Data { regs, ty }
            }
        });
        self.ins(format_args!("}}"));
        if let Some(binding) = handed {
            self.bind(results[0], binding);
        }
    }
}
