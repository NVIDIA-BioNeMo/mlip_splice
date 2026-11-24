// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph analyses and deletion of unheld arena nodes.

use std::collections::{BTreeMap, BTreeSet};

use crate::op::same_kind;

use super::{BlockId, Def, Function, OpId, RegionId, ValueId};

/// The operations a cone walk has already accounted for.
///
/// An operand cone is a graph, not a tree: one value may be read by many, and
/// what a shared operation reads is the same however the walk arrives at it.
/// Remembering the operations already entered makes the walk linear in the
/// arena rather than exponential in its sharing.
#[derive(Default)]
pub struct Visited(BTreeSet<usize>);

impl Visited {
    fn enter(&mut self, op: OpId) -> bool {
        self.0.insert(op.index())
    }
}

impl Function {
    /// The places `value` reads: the materialized values its operand cone
    /// reaches, itself when it is one.
    pub fn add_reads(&self, value: ValueId, into: &mut BTreeSet<ValueId>) {
        self.add_reads_seen(value, into, &mut Visited::default());
    }

    /// [`Function::add_reads`], continuing a walk that has already entered the
    /// operations in `seen`.
    ///
    /// The cone is walked on an explicit stack. A scalarized polynomial is a
    /// chain of arithmetic thousands of operations long, and the depth of the
    /// walk is the length of that chain: recursing over it exhausts the thread's
    /// stack, which is a crash rather than a diagnostic. Only an operand owning
    /// a region recurses, because it needs a set of its own to subtract what the
    /// region binds, and regions nest as deep as the program's loops do.
    pub fn add_reads_seen(&self, value: ValueId, into: &mut BTreeSet<ValueId>, seen: &mut Visited) {
        let mut pending = vec![value];
        while let Some(value) = pending.pop() {
            let Some(op) = self.operand_op(value) else {
                into.insert(value);
                continue;
            };
            if !seen.enter(op) {
                continue;
            }
            if self.op(op).regions.is_empty() {
                pending.extend(self.op(op).operands.iter().copied());
                continue;
            }
            // An operand that owns a region reads what its body reads, less what
            // the region binds for itself: a loop standing for the value it
            // reduces to reads the places its update gathers from, not its own
            // index or the accumulator it carries.
            let mut inner = BTreeSet::new();
            self.add_op_reads_seen(op, &mut inner, seen);
            let mut bound = BTreeSet::new();
            self.add_region_definitions(op, &mut bound);
            into.extend(inner.difference(&bound).copied());
        }
    }

    /// Every place the regions of `op` define: what their blocks take on the way
    /// in, and what the steps inside them bind.
    fn add_region_definitions(&self, op: OpId, into: &mut BTreeSet<ValueId>) {
        for index in 0..self.op(op).regions.len() {
            let region = self.op(op).regions[index];
            for slot in 0..self.region(region).blocks.len() {
                let block = self.region(region).blocks[slot];
                into.extend(self.block(block).params.iter().copied());
                for step in 0..self.block(block).ops.len() {
                    let inner = self.block(block).ops[step];
                    into.extend(self.op(inner).results.iter().copied());
                    self.add_region_definitions(inner, into);
                }
            }
        }
    }

    /// Every operand operation reached through the operands of `ops`, without
    /// entering regions owned by those operations.
    pub fn operand_ancestors_of(&self, ops: &[OpId]) -> BTreeSet<OpId> {
        let mut reached = weave::Reached::default();
        self.reach_operand_ancestors(ops, &mut reached);
        reached.iter().map(|at| OpId::new(self.tag, at)).collect()
    }

    /// [`Function::operand_ancestors_of`], into a walk a caller reuses.
    ///
    /// The walk holds one bit per operation, so allocating it costs the size
    /// of the arena. A caller placing operations one at a time asks this a
    /// million times over one arena, and paying that size per question is what
    /// reusing the walk is for.
    pub(crate) fn reach_operand_ancestors(&self, ops: &[OpId], into: &mut weave::Reached) {
        into.walk(
            self.ops.len(),
            ops.iter().map(|op| op.at(self.tag)),
            |operation, leads_to| self.each_operand_definition(operation, leads_to),
        );
    }

    /// Calls `found` with each operand operation the operation at `at` reads.
    ///
    /// An operand is evaluated where it is read, so this is the relation a
    /// cone of operands is the closure of. A placed definition is not one: it
    /// is evaluated at its own position, so reading it reaches no further.
    fn each_operand_definition(&self, at: usize, found: &mut dyn FnMut(usize)) {
        for &operand in self.ops[at]
            .operands
            .iter()
            .chain(self.ops[at].into.iter().flatten())
        {
            if let Some(defining) = self.operand_op(operand) {
                found(defining.at(self.tag));
            }
        }
    }

    /// The operations owned by regions nested under `ops`, without following reads.
    pub(crate) fn reach_region_ops(&self, ops: &[OpId], into: &mut weave::Reached) {
        into.walk(
            self.ops.len(),
            ops.iter().map(|op| op.at(self.tag)),
            |operation, found| {
                for &region in &self.ops[operation].regions {
                    for &block in &self.region(region).blocks {
                        for &op in &self.block(block).ops {
                            found(op.at(self.tag));
                        }
                    }
                }
            },
        );
    }

    /// Every operand operation the given steps evaluate, directly or through
    /// one another, the bodies of their regions included.
    ///
    /// A step is evaluated at its position and an operand wherever it is read,
    /// so this is what a list of steps actually computes: the operands that
    /// must go on being held for those steps to mean anything. One outside it
    /// is read by no step here.
    pub fn operands_reached_by(&self, steps: &[OpId]) -> BTreeSet<OpId> {
        let mut reached = weave::Reached::default();
        self.reach_operands(steps, &mut reached);
        reached.iter().map(|at| OpId::new(self.tag, at)).collect()
    }

    /// [`Function::operands_reached_by`], into a walk a caller reuses.
    pub(crate) fn reach_operands(&self, steps: &[OpId], into: &mut weave::Reached) {
        into.walk(
            self.ops.len(),
            steps.iter().map(|op| op.at(self.tag)),
            |operation, leads_to| {
                self.each_operand_definition(operation, leads_to);
                // An operation inside a region reads what encloses it, so the
                // operands its body evaluates are reached through the step
                // owning the region just as an operand of the step itself is.
                for index in 0..self.ops[operation].regions.len() {
                    let region = self.ops[operation].regions[index].at(self.tag);
                    for slot in 0..self.regions[region].blocks.len() {
                        let block = self.regions[region].blocks[slot].at(self.tag);
                        for held in 0..self.blocks[block].ops.len() {
                            leads_to(self.blocks[block].ops[held].at(self.tag));
                        }
                        for slot in 0..self.blocks[block].terminator.yields.len() {
                            let yielded = self.blocks[block].terminator.yields[slot];
                            if let Some(defining) = self.operand_op(yielded) {
                                leads_to(defining.at(self.tag));
                            }
                        }
                    }
                }
            },
        );
    }

    /// Whether any live block lists `op` among its operations.
    /// The operations this arena holds that no block lists.
    ///
    /// Every operation belongs to exactly one block, so this is empty in a
    /// program. It is the cheapest question to ask of a body under
    /// construction, and the first one worth asking when a later rule fails:
    /// an operation belonging nowhere makes every ordering question moot.
    pub fn stranded(&self) -> Vec<OpId> {
        self.live_ops()
            .into_iter()
            .filter(|&op| !self.any_block_lists(op))
            .collect()
    }

    /// The operations that stand before something they read, block by block.
    ///
    /// A value is readable after the operation defining it, so within a block
    /// an operand stands ahead of every operation reading it. This asks that
    /// directly, without the rest of verification: a body can be well ordered
    /// and still break a typing rule, and knowing which is which saves
    /// guessing.
    pub fn misordered(&self) -> Vec<(BlockId, OpId)> {
        let mut found = Vec::new();
        for block in self.live_blocks() {
            let ops = &self.block(block).ops;
            let mut seen = BTreeSet::new();
            for &op in ops {
                for index in 0..self.op(op).operands.len() {
                    let operand = self.op(op).operands[index];
                    let Some(defining) = self.operand_op(operand) else {
                        continue;
                    };
                    if ops.contains(&defining) && !seen.contains(&defining) {
                        found.push((block, op));
                    }
                }
                seen.insert(op);
            }
        }
        found
    }

    pub(crate) fn remove_all_from_blocks(&mut self, removed: &[bool]) {
        for block in &mut self.blocks {
            block.ops.retain(|held| !removed[held.index()]);
        }
    }

    pub(crate) fn any_block_lists(&self, op: OpId) -> bool {
        (0..self.blocks.len())
            .filter(|index| !self.deleted.blocks.contains(&(*index as u32)))
            .any(|index| self.blocks[index].ops.contains(&op))
    }

    /// Adds every operation a live block lists.
    ///
    /// What the blocks list is a property of the blocks, so a caller asking
    /// about many operations reads them once instead of once per question.
    pub(crate) fn mark_listed_ops(&self, marked: &mut [bool]) {
        for index in 0..self.blocks.len() {
            if self.deleted.blocks.contains(&(index as u32)) {
                continue;
            }
            for op in &self.blocks[index].ops {
                marked[op.index()] = true;
            }
        }
    }

    /// Stops holding these operations, and the operands only they reached.
    ///
    /// An operand is evaluated where it is read, so one nothing reads computes
    /// nothing. Every operation belongs to a block, and these belong to no
    /// list any more, so they stop being part of the program rather than
    /// staying live and homeless.
    pub(crate) fn forget_unheld(&mut self, ops: &[OpId]) -> Vec<OpId> {
        // These go together, so they are judged together: one read only by
        // another of them is read by nothing that survives. Judging them one at
        // a time would keep the whole set, each held up by the next.
        let mut going = vec![false; self.ops.len()];
        for &op in ops {
            let at = op.at(self.tag);
            if !self.deleted.ops.contains(&(at as u32)) {
                going[at] = true;
            }
        }
        // An operation is evaluated wherever it is read, so one still read from
        // outside the set goes on meaning what it meant, and with it everything
        // it reads.
        //
        // Sparing one spares what it reads, and that can leave a third of the
        // set read by a survivor in turn, so this settles rather than passing
        // once. What is spared drives the settling: an operation is examined
        // when a reason to spare it appears, where asking which one stays,
        // sparing it, and starting over rescans the whole set per survivor.
        let readers = self.readers();
        let mut sparing = ops
            .iter()
            .copied()
            .filter(|&op| {
                going[op.index()] && readers.has_use_outside(self, op, |user| going[user.index()])
            })
            .collect::<Vec<_>>();
        while let Some(stays) = sparing.pop() {
            if !going[stays.index()] {
                continue;
            }
            going[stays.index()] = false;
            // What a spared operation reads is read by something outside the
            // set now, which is the same reason `stays` itself was spared.
            for index in 0..self.op(stays).operands.len() {
                let operand = self.op(stays).operands[index];
                if let Some(defining) = self.defining_op(operand)
                    && going[defining.index()]
                {
                    sparing.push(defining);
                }
            }
        }

        // Whatever is spared is still part of the program, so it is handed back
        // for the caller to go on holding: it was taken out of a list to be
        // judged, and one held by nobody would belong to no block.
        let spared = ops
            .iter()
            .copied()
            .filter(|op| !going[op.index()])
            .collect::<Vec<_>>();
        let mut walk = ops
            .iter()
            .copied()
            .filter(|op| going[op.index()])
            .collect::<Vec<_>>();
        for &op in &walk {
            self.forget_op(op);
        }
        // What they read may now be read by nothing at all.
        while let Some(op) = walk.pop() {
            for index in 0..self.op(op).operands.len() {
                let operand = self.op(op).operands[index];
                let Some(defining) = self.operand_op(operand) else {
                    continue;
                };
                let at = defining.at(self.tag) as u32;
                if !self.deleted.ops.contains(&at)
                    && !self.any_block_lists(defining)
                    && !readers.has_use_outside(self, defining, |_| false)
                {
                    self.forget_op(defining);
                    walk.push(defining);
                }
            }
        }
        spared
    }

    /// Stops holding one operation, and whatever it alone held.
    ///
    /// A region belongs to the operation listing it, so an operation that
    /// stops being part of the program takes its regions, their blocks, and
    /// the operations in them along: a region whose owner is gone is owned by
    /// nothing, which is not a program.
    fn forget_op(&mut self, op: OpId) {
        let at = op.at(self.tag);
        if !self.deleted.ops.insert(at as u32) {
            return;
        }
        for index in 0..self.ops[at].regions.len() {
            let region = self.ops[at].regions[index];
            let held = region.at(self.tag);
            if !self.deleted.regions.insert(held as u32) {
                continue;
            }
            for slot in 0..self.regions[held].blocks.len() {
                let block = self.regions[held].blocks[slot];
                let inner = block.at(self.tag);
                if !self.deleted.blocks.insert(inner as u32) {
                    continue;
                }
                for step in self.blocks[inner].ops.clone() {
                    // The body's own operations were never offered for
                    // deletion, so each is judged here: one a live operation
                    // still names goes on meaning what it meant.
                    if self.uses_left(step).is_empty() && !self.is_yielded_anywhere(step) {
                        self.forget_op(step);
                    }
                }
            }
        }
    }

    /// Whether a live block hands back what `op` produces. A value leaves a
    /// region through what the region yields, so a yield is a read.
    fn is_yielded_anywhere(&self, op: OpId) -> bool {
        let results = &self.op(op).results;
        (0..self.blocks.len())
            .filter(|index| !self.deleted.blocks.contains(&(*index as u32)))
            .any(|index| {
                self.blocks[index]
                    .terminator
                    .yields
                    .iter()
                    .any(|yielded| results.contains(yielded))
            })
    }

    /// The live operations that read what `op` produces.
    /// The live operations that read what `op` produces.
    ///
    /// This reads the whole arena to answer about one operation. A caller
    /// asking about many builds [`Function::readers`] once instead.
    fn uses_left(&self, op: OpId) -> Vec<OpId> {
        let results = self.op(op).results.clone();
        (0..self.ops.len())
            .filter(|index| !self.deleted.ops.contains(&(*index as u32)))
            .map(|index| OpId::new(self.tag, index))
            .filter(|&other| {
                other != op
                    && self
                        .op(other)
                        .operands
                        .iter()
                        .any(|operand| results.contains(operand))
            })
            .collect()
    }

    /// Which operations read each value.
    ///
    /// Who reads one operation is a question about every operation, so asking
    /// it one operation at a time reads the whole arena once per question.
    /// Reading the arena once answers every such question, and deleting an
    /// operation only ever removes a reader — so an index built before a round
    /// of deletion stays true through it, once the deleted are filtered out.
    pub(crate) fn readers(&self) -> Readers {
        self.readers_with_roots(&[])
    }

    pub(crate) fn readers_with_roots(&self, roots: &[ValueId]) -> Readers {
        let mut returned = self
            .block(self.entry())
            .terminator
            .yields
            .iter()
            .chain(roots)
            .copied()
            .collect::<Vec<_>>();
        returned.sort_unstable();
        returned.dedup();
        Readers {
            uses: weave::Uses::of_arena(self),
            returned,
        }
    }

    /// [`Function::add_reads`], as a set.
    pub fn reads(&self, value: ValueId) -> BTreeSet<ValueId> {
        let mut reads = BTreeSet::new();
        self.add_reads(value, &mut reads);
        reads
    }

    /// Every place the operations of `block`, and the regions under them, read.
    pub fn add_block_reads(&self, block: BlockId, into: &mut BTreeSet<ValueId>) {
        self.add_block_reads_seen(block, into, &mut Visited::default());
    }

    /// [`Function::add_block_reads`], continuing an existing walk.
    fn add_block_reads_seen(
        &self,
        block: BlockId,
        into: &mut BTreeSet<ValueId>,
        seen: &mut Visited,
    ) {
        for index in 0..self.block(block).ops.len() {
            let op = self.block(block).ops[index];
            if self.op(op).placed {
                self.add_op_reads_seen(op, into, seen);
            }
        }
        for index in 0..self.block(block).terminator.yields.len() {
            self.add_reads_seen(self.block(block).terminator.yields[index], into, seen);
        }
    }

    /// Every place `op` reads, its regions included.
    pub fn add_op_reads(&self, op: OpId, into: &mut BTreeSet<ValueId>) {
        self.add_op_reads_seen(op, into, &mut Visited::default());
    }

    /// [`Function::add_op_reads`], continuing an existing walk.
    pub fn add_op_reads_seen(&self, op: OpId, into: &mut BTreeSet<ValueId>, seen: &mut Visited) {
        for index in 0..self.op(op).operands.len() {
            self.add_reads_seen(self.op(op).operands[index], into, seen);
        }
        for index in 0..self.op(op).regions.len() {
            let region = self.op(op).regions[index];
            for block in 0..self.region(region).blocks.len() {
                self.add_block_reads_seen(self.region(region).blocks[block], into, seen);
            }
        }
    }

    /// Whether `value` may be evaluated where it would not otherwise run.
    pub fn is_speculatable(&self, value: ValueId) -> bool {
        match self.operand_op(value) {
            None => true,
            Some(op) => {
                self.op(op).kind.is_speculatable()
                    && (0..self.op(op).operands.len())
                        .all(|index| self.is_speculatable(self.op(op).operands[index]))
            }
        }
    }

    /// Whether two values are structurally the same expression: the same value,
    /// the same place, or the same operation over operands that are themselves
    /// the same — regions included.
    ///
    /// Structural sameness is not sameness of value: a place denotes whatever
    /// its storage holds when it is read, so the caller must know that both are
    /// observed with nothing writing in between.
    pub fn same_value(&self, a: ValueId, b: ValueId) -> bool {
        self.same_value_under(a, b, &BTreeMap::new())
    }

    /// [`Function::same_value`], with `bound` pairing the values two regions
    /// being compared bind for themselves: an index or an accumulator is the
    /// same as the one it stands opposite, and nothing else.
    fn same_value_under(&self, a: ValueId, b: ValueId, bound: &BTreeMap<ValueId, ValueId>) -> bool {
        if let Some(&paired) = bound.get(&a) {
            return paired == b;
        }
        if a == b {
            return true;
        }
        if self.ty(a) != self.ty(b) {
            return false;
        }
        match (self.operand_op(a), self.operand_op(b)) {
            (None, None) => false,
            // Two results of one operation are two values, so which of them a
            // caller reads is part of what it read.
            (Some(left), Some(right)) => match (self.value(a).def, self.value(b).def) {
                (Def::Result { index: x, .. }, Def::Result { index: y, .. }) if x == y => {
                    self.same_op_under(left, right, bound)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether two operations are the same, over operands, destinations and
    /// region bodies alike.
    fn same_op_under(&self, left: OpId, right: OpId, bound: &BTreeMap<ValueId, ValueId>) -> bool {
        if left == right {
            return true;
        }
        let (a, b) = (self.op(left), self.op(right));
        if !same_kind(&a.kind, &b.kind)
            || a.placed != b.placed
            || a.operands.len() != b.operands.len()
            || a.results.len() != b.results.len()
            || a.regions.len() != b.regions.len()
        {
            return false;
        }
        // Every result, not only the one a caller asked for: a sibling nothing
        // reads is still part of what the operation is, and two operations
        // producing differently typed values are two operations.
        if !a
            .results
            .iter()
            .zip(&b.results)
            .all(|(&x, &y)| self.ty(x) == self.ty(y))
        {
            return false;
        }
        let paired = |x: ValueId, y: ValueId| self.same_value_under(x, y, bound);
        if !a
            .operands
            .iter()
            .zip(&b.operands)
            .all(|(&x, &y)| paired(x, y))
        {
            return false;
        }
        for index in 0..a.results.len() {
            match (a.destination(index), b.destination(index)) {
                (None, None) => {}
                (Some(x), Some(y)) if paired(x, y) => {}
                _ => return false,
            }
        }
        (0..a.regions.len())
            .all(|index| self.same_region(a.regions[index], b.regions[index], bound))
    }

    /// Whether two regions hold the same body, under a pairing extended by what
    /// their blocks bind.
    fn same_region(
        &self,
        left: RegionId,
        right: RegionId,
        outer: &BTreeMap<ValueId, ValueId>,
    ) -> bool {
        let (a, b) = (&self.region(left).blocks, &self.region(right).blocks);
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).all(|(&x, &y)| self.same_block(x, y, outer))
    }

    fn same_block(
        &self,
        left: BlockId,
        right: BlockId,
        outer: &BTreeMap<ValueId, ValueId>,
    ) -> bool {
        let (a, b) = (self.block(left), self.block(right));
        if a.params.len() != b.params.len()
            || a.ops.len() != b.ops.len()
            || a.terminator.yields.len() != b.terminator.yields.len()
        {
            return false;
        }
        let mut bound = outer.clone();
        for (&x, &y) in a.params.iter().zip(&b.params) {
            if self.ty(x) != self.ty(y) {
                return false;
            }
            bound.insert(x, y);
        }
        for (&x, &y) in a.ops.iter().zip(&b.ops) {
            if !self.same_op_under(x, y, &bound) {
                return false;
            }
            for (&produced, &against) in self.op(x).results.iter().zip(&self.op(y).results) {
                bound.insert(produced, against);
            }
        }
        a.terminator
            .yields
            .iter()
            .zip(&b.terminator.yields)
            .all(|(&x, &y)| self.same_value_under(x, y, &bound))
    }
}

impl weave::Operations for Function {
    fn operation_count(&self) -> usize {
        self.ops.len()
    }

    fn value_count(&self) -> usize {
        self.values.len()
    }

    fn is_live(&self, operation: usize) -> bool {
        !self.deleted.ops.contains(&(operation as u32))
    }

    fn each_operand(&self, operation: usize, mut read: impl FnMut(usize)) {
        let op = &self.ops[operation];
        for operand in op.operands.iter().chain(op.into.iter().flatten()) {
            read(operand.index());
        }
        for &region in &op.regions {
            for &block in &self.region(region).blocks {
                for value in &self.block(block).terminator.yields {
                    read(value.index());
                }
            }
        }
    }

    fn each_result(&self, operation: usize, mut defined: impl FnMut(usize)) {
        for result in &self.ops[operation].results {
            defined(result.index());
        }
    }
}

/// Which operations read each value, read off the arena once.
///
/// [`Function::readers`] states why this exists and when it stays true.
#[derive(Default)]
pub(crate) struct Readers {
    uses: weave::Uses,
    returned: Vec<ValueId>,
}

impl Readers {
    /// Whether any result is read outside the selected operations. Function
    /// results are external roots; a region yield is a use by its owning op.
    pub(crate) fn has_use_outside(
        &self,
        f: &Function,
        op: OpId,
        mut inside: impl FnMut(OpId) -> bool,
    ) -> bool {
        f.op(op)
            .results
            .iter()
            .any(|value| self.returned.binary_search(value).is_ok())
            || self.uses.any_of(f, op.index(), |reader| {
                !inside(OpId::new(f.tag, reader as usize))
            })
    }
}
