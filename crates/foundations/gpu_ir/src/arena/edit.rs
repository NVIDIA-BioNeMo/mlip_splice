// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Changing an arena that already exists, without ever leaving it corrupt.
//!
//! Building writes an arena forwards and asserts on a mistake, because a
//! producer holding the builder cannot make one by accident. An editor can: the
//! ids it is handed come from a caller who may hold two arenas, may hold an id
//! from before a change, or may ask for something the structure cannot be. So
//! every entry point here is checked and answers with an [`EditError`], and none
//! of them can stop the process.
//!
//! **Every edit decides before it changes anything.** What a refused edit leaves
//! behind is exactly what it was handed — the same ids, the same text, the same
//! diagnostics — so there is no half-applied state and nothing to roll back. An
//! edit that cannot be decided in one pass is written as one that can.
//!
//! **Ids are never reused and never renumbered.** An operation taken out of the
//! block it was a step of stays in the arena as the value it always was, and one
//! *deleted* keeps its slot while its id stops naming anything. So a handle a
//! caller kept either goes on naming exactly what it named or goes stale — it
//! can never come to mean whatever was added next. [`Function::compacted`] is
//! where deleted slots close up, and it runs only where a whole program is
//! written down or handed on, which is where ids start again anyway.
//!
//! What is *not* decided here is meaning. An arena between two edits may be
//! typed wrongly, may read a value where it is not defined, may hold an
//! operation nothing runs; [`crate::verify()`] is what says so. This file refuses
//! only what would leave the arena unreadable — a foreign id, an index past the
//! end, an operation that is a step of two blocks at once.

mod error;

pub use error::EditError;

use super::{BlockId, Declaration, Function, OpId, RegionId, Terminator, ValueId};
use crate::op::{OpKind, Trip};
use crate::{Param, ParamTy, PtrTy, SharedArr, Space, Ty, ValTy};

type Result<T = ()> = std::result::Result<T, EditError>;

/// Where one value is read. A definition is not a use: a block parameter and an
/// operation result are where a value comes from, and these are where it goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Use {
    Operand {
        op: OpId,
        index: usize,
    },
    /// The storage a result is produced into.
    Destination {
        op: OpId,
        index: usize,
    },
    Yield {
        block: BlockId,
        index: usize,
    },
}

impl Use {
    fn describe(self) -> String {
        match self {
            Use::Operand { op, index } => format!("operand {index} of operation {}", op.index()),
            Use::Destination { op, index } => format!(
                "the storage result {index} of operation {} takes over",
                op.index()
            ),
            Use::Yield { block, index } => {
                format!("what block {} yields at {index}", block.index())
            }
        }
    }
}

impl Function {
    // ---- what this arena holds -------------------------------------------

    pub fn holds_value(&self, id: ValueId) -> bool {
        id.arena() == self.tag && id.index() < self.values.len() && self.value_is_live(id.index())
    }

    pub fn holds_op(&self, id: OpId) -> bool {
        id.arena() == self.tag
            && id.index() < self.ops.len()
            && !self.deleted.ops.contains(&(id.index() as u32))
    }

    pub fn holds_block(&self, id: BlockId) -> bool {
        id.arena() == self.tag
            && id.index() < self.blocks.len()
            && !self.deleted.blocks.contains(&(id.index() as u32))
    }

    pub fn holds_region(&self, id: RegionId) -> bool {
        id.arena() == self.tag
            && id.index() < self.regions.len()
            && !self.deleted.regions.contains(&(id.index() as u32))
    }

    /// Whether a value's definition is still part of the program. A value goes
    /// with whatever defines it: the operation that produces it, or the block
    /// that takes it.
    pub(super) fn value_is_live(&self, index: usize) -> bool {
        match self.values[index].def {
            super::Def::Param { block, .. } => {
                !self.deleted.blocks.contains(&(block.index() as u32))
            }
            super::Def::Result { op, .. } => !self.deleted.ops.contains(&(op.index() as u32)),
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    pub fn block_at(&self, index: usize) -> BlockId {
        assert!(index < self.blocks.len(), "no block {index} in this arena");
        BlockId::new(self.tag, index)
    }

    pub fn region_at(&self, index: usize) -> RegionId {
        assert!(
            index < self.regions.len(),
            "no region {index} in this arena"
        );
        RegionId::new(self.tag, index)
    }

    /// Every value this arena still holds, in id order.
    pub fn live_values(&self) -> Vec<ValueId> {
        (0..self.values.len())
            .filter(|&index| self.value_is_live(index))
            .map(|index| ValueId::new(self.tag, index))
            .collect()
    }

    pub fn live_ops(&self) -> Vec<OpId> {
        (0..self.ops.len())
            .filter(|&index| !self.deleted.ops.contains(&(index as u32)))
            .map(|index| OpId::new(self.tag, index))
            .collect()
    }

    pub fn live_blocks(&self) -> Vec<BlockId> {
        (0..self.blocks.len())
            .filter(|&index| !self.deleted.blocks.contains(&(index as u32)))
            .map(|index| BlockId::new(self.tag, index))
            .collect()
    }

    pub fn live_regions(&self) -> Vec<RegionId> {
        (0..self.regions.len())
            .filter(|&index| !self.deleted.regions.contains(&(index as u32)))
            .map(|index| RegionId::new(self.tag, index))
            .collect()
    }

    /// Whether anything at all was deleted from this arena.
    pub fn has_deletions(&self) -> bool {
        self.deleted.any()
    }

    /// Deletes blocks and regions that are not structurally reachable from the
    /// entry block, together with operations defined only in those blocks.
    ///
    /// The deletion is atomic. If reachable structure reads a value defined by
    /// what would be deleted, the arena is left unchanged and the edit is
    /// refused.
    pub fn delete_unattached(&mut self) -> Result {
        let mut blocks = std::collections::BTreeSet::new();
        let mut regions = std::collections::BTreeSet::new();
        let mut ops = std::collections::BTreeSet::new();
        let mut pending = vec![self.entry];
        while let Some(block) = pending.pop() {
            if !blocks.insert(block) {
                continue;
            }
            for &op in &self.blocks[block.index()].ops {
                if !self.holds_op(op) || !ops.insert(op) {
                    continue;
                }
                for &region in &self.ops[op.index()].regions {
                    if !self.holds_region(region) || !regions.insert(region) {
                        continue;
                    }
                    pending.extend(self.regions[region.index()].blocks.iter().copied());
                }
            }
        }

        let removed_blocks: std::collections::BTreeSet<_> = self
            .live_blocks()
            .into_iter()
            .filter(|block| !blocks.contains(block))
            .collect();
        let removed_regions: std::collections::BTreeSet<_> = self
            .live_regions()
            .into_iter()
            .filter(|region| !regions.contains(region))
            .collect();
        let removed_ops: std::collections::BTreeSet<_> = removed_blocks
            .iter()
            .flat_map(|block| self.blocks[block.index()].ops.iter().copied())
            .filter(|op| self.holds_op(*op) && !ops.contains(op))
            .collect();
        let removed_values: Vec<_> = self
            .live_values()
            .into_iter()
            .filter(|value| match self.values[value.index()].def {
                super::Def::Param { block, .. } => removed_blocks.contains(&block),
                super::Def::Result { op, .. } => removed_ops.contains(&op),
            })
            .collect();

        for value in removed_values {
            if let Some(held) = self.uses(value)?.into_iter().find(|held| match held {
                Use::Operand { op, .. } | Use::Destination { op, .. } => !removed_ops.contains(op),
                Use::Yield { block, .. } => !removed_blocks.contains(block),
            }) {
                return Err(EditError::StillRead {
                    value: value.index(),
                    by: held.describe(),
                });
            }
        }

        self.deleted
            .ops
            .extend(removed_ops.into_iter().map(|op| op.index() as u32));
        self.deleted
            .blocks
            .extend(removed_blocks.into_iter().map(|block| block.index() as u32));
        self.deleted.regions.extend(
            removed_regions
                .into_iter()
                .map(|region| region.index() as u32),
        );
        Ok(())
    }

    /// Every name a caller gave a value, by value, in value order.
    pub fn names(&self) -> impl Iterator<Item = (ValueId, &str)> {
        self.names
            .iter()
            .filter(|(index, _)| self.value_is_live(**index as usize))
            .map(|(&index, name)| (ValueId::new(self.tag, index as usize), name.as_str()))
    }

    fn value_of(&self, id: ValueId) -> Result<usize> {
        self.held(
            "value",
            id.arena() == self.tag,
            id.index() < self.values.len(),
            id.index() < self.values.len() && self.value_is_live(id.index()),
            id.index(),
        )
    }

    fn op_of(&self, id: OpId) -> Result<usize> {
        self.held(
            "operation",
            id.arena() == self.tag,
            id.index() < self.ops.len(),
            !self.deleted.ops.contains(&(id.index() as u32)),
            id.index(),
        )
    }

    fn block_of_id(&self, id: BlockId) -> Result<usize> {
        self.held(
            "block",
            id.arena() == self.tag,
            id.index() < self.blocks.len(),
            !self.deleted.blocks.contains(&(id.index() as u32)),
            id.index(),
        )
    }

    fn region_of_id(&self, id: RegionId) -> Result<usize> {
        self.held(
            "region",
            id.arena() == self.tag,
            id.index() < self.regions.len(),
            !self.deleted.regions.contains(&(id.index() as u32)),
            id.index(),
        )
    }

    /// The three ways an id names nothing here, told apart.
    fn held(
        &self,
        what: &'static str,
        ours: bool,
        exists: bool,
        live: bool,
        index: usize,
    ) -> Result<usize> {
        if !ours || !exists {
            return Err(EditError::Foreign { what, index });
        }
        if !live {
            return Err(EditError::Deleted { what, index });
        }
        Ok(index)
    }

    // ---- reading the use-def graph ---------------------------------------

    /// Every place `value` is read, in a fixed order: operands and destinations
    /// by operation, then what blocks yield.
    ///
    /// What a deleted operation or block still names is not a use: it is not
    /// part of the program, so nothing it mentions is read.
    pub fn uses(&self, value: ValueId) -> Result<Vec<Use>> {
        self.value_of(value)?;
        let mut found = Vec::new();
        self.each_use(value, &mut |held| {
            found.push(held);
            false
        });
        Ok(found)
    }

    /// Whether anything at all reads `value`.
    ///
    /// The same question [`Function::uses`] answers, stopped at the first
    /// answer — which is what a renderer deciding whether a result needs a
    /// register is asking. One walk states what a use *is*, so the two can
    /// never come to disagree; what a function hands back is among them,
    /// because those are the yields of its entry block.
    pub fn is_used(&self, value: ValueId) -> bool {
        let mut found = false;
        self.each_use(value, &mut |_| {
            found = true;
            true
        });
        found
    }

    /// Every place `value` is read, in a fixed order, until `each` says to stop.
    fn each_use(&self, value: ValueId, each: &mut dyn FnMut(Use) -> bool) {
        for id in self.live_ops() {
            let op = &self.ops[id.index()];
            for (slot, &operand) in op.operands.iter().enumerate() {
                if operand == value
                    && each(Use::Operand {
                        op: id,
                        index: slot,
                    })
                {
                    return;
                }
            }
            for (slot, destination) in op.into.iter().enumerate() {
                if *destination == Some(value)
                    && each(Use::Destination {
                        op: id,
                        index: slot,
                    })
                {
                    return;
                }
            }
        }
        for id in self.live_blocks() {
            let block = &self.blocks[id.index()];
            for (slot, &yielded) in block.terminator.yields.iter().enumerate() {
                if yielded == value
                    && each(Use::Yield {
                        block: id,
                        index: slot,
                    })
                {
                    return;
                }
            }
        }
    }

    /// The block `op` belongs to, and its position in that block.
    pub fn step_of(&self, op: OpId) -> Result<Option<(BlockId, usize)>> {
        let held = self.op_of(op)?;
        let op = OpId::new(self.tag, held);
        for id in self.live_blocks() {
            if let Some(position) = self.blocks[id.index()]
                .ops
                .iter()
                .position(|&held| held == op)
            {
                return Ok(Some((id, position)));
            }
        }
        Ok(None)
    }

    // ---- edits over what is already here ---------------------------------

    /// Reads another value at one operand of one operation.
    pub fn set_operand(&mut self, op: OpId, index: usize, value: ValueId) -> Result {
        let held = self.op_of(op)?;
        let value = ValueId::new(self.tag, self.value_of(value)?);
        let operands = self.ops[held].operands.len();
        if index >= operands {
            return Err(EditError::OutOfRange {
                what: "operand",
                index,
                held: operands,
            });
        }
        self.ops[held].operands[index] = value;
        Ok(())
    }

    /// Names, or stops naming, the storage one result is produced into.
    pub fn set_destination(&mut self, op: OpId, index: usize, place: Option<ValueId>) -> Result {
        let held = self.op_of(op)?;
        let place = place.map(|place| self.value_of(place)).transpose()?;
        let results = self.ops[held].results.len();
        if index >= results {
            return Err(EditError::OutOfRange {
                what: "result",
                index,
                held: results,
            });
        }
        if self.ops[held].into.is_empty() {
            self.ops[held].into = vec![None; results];
        }
        self.ops[held].into[index] = place.map(|place| ValueId::new(self.tag, place));
        Ok(())
    }

    /// Reads `new` wherever `old` was read, and says how many places that was.
    pub fn replace_uses(&mut self, old: ValueId, new: ValueId) -> Result<usize> {
        let uses = self.uses(old)?;
        let new = ValueId::new(self.tag, self.value_of(new)?);
        for held in &uses {
            match *held {
                Use::Operand { op, index } => self.ops[op.index()].operands[index] = new,
                Use::Destination { op, index } => self.ops[op.index()].into[index] = Some(new),
                Use::Yield { block, index } => {
                    self.blocks[block.index()].terminator.yields[index] = new;
                }
            }
        }
        Ok(uses.len())
    }

    /// What a block hands back to the operation owning the region it is in.
    pub fn set_yields(&mut self, block: BlockId, yields: Vec<ValueId>) -> Result {
        let held = self.block_of_id(block)?;
        let yields = yields
            .into_iter()
            .map(|value| Ok(ValueId::new(self.tag, self.value_of(value)?)))
            .collect::<Result<Vec<_>>>()?;
        self.blocks[held].terminator = Terminator { yields };
        Ok(())
    }

    /// How often the threads of a block go round one loop, which is a fact
    /// about its bounds and not about the arena.
    pub fn set_trip(&mut self, op: OpId, trip: Trip) -> Result {
        let held = self.op_of(op)?;
        match &mut self.ops[held].kind {
            OpKind::Loop { trip: stated, .. } => {
                *stated = trip;
                Ok(())
            }
            _ => Err(EditError::NotALoop { op: held }),
        }
    }

    /// Runs another function at one call.
    pub fn set_callee(
        &mut self,
        op: OpId,
        callee: std::sync::Arc<crate::func::DeviceFunction>,
    ) -> Result {
        let held = self.op_of(op)?;
        match &mut self.ops[held].kind {
            OpKind::Call(target) => {
                *target = callee;
                Ok(())
            }
            _ => Err(EditError::NotACall { op: held }),
        }
    }

    pub fn name_value(&mut self, value: ValueId, name: Option<String>) -> Result {
        let held = self.value_of(value)?;
        match name {
            Some(name) => self.names.insert(held as u32, name),
            None => self.names.remove(&(held as u32)),
        };
        Ok(())
    }

    // ---- adding to the arena ---------------------------------------------

    /// An operation of this kind, over these operands, producing these types.
    ///
    /// It is created before a caller inserts it into its block. A finished,
    /// verified function never retains that transient detached state.
    pub fn new_op(
        &mut self,
        kind: OpKind,
        placed: bool,
        operands: Vec<ValueId>,
        result_tys: &[ValTy],
        into: Vec<Option<ValueId>>,
        regions: Vec<RegionId>,
    ) -> Result<OpId> {
        if regions.len() != kind.regions() {
            return Err(EditError::RegionCount {
                kind: kind_name(&kind),
                held: regions.len(),
                wanted: kind.regions(),
            });
        }
        if placed != kind.must_be_placed() && !kind.may_be_either() {
            return Err(EditError::Placement {
                kind: kind_name(&kind),
                placed,
            });
        }
        if !into.is_empty() && into.len() != result_tys.len() {
            return Err(EditError::DestinationCount {
                results: result_tys.len(),
                into: into.len(),
            });
        }
        for &ty in result_tys {
            if matches!(ty, ValTy::Data(Ty { lanes: 0, .. })) {
                return Err(EditError::ZeroLanes);
            }
        }
        let operands = operands
            .into_iter()
            .map(|value| Ok(ValueId::new(self.tag, self.value_of(value)?)))
            .collect::<Result<Vec<_>>>()?;
        let into = into
            .into_iter()
            .map(|place| match place {
                None => Ok(None),
                Some(place) => Ok(Some(ValueId::new(self.tag, self.value_of(place)?))),
            })
            .collect::<Result<Vec<_>>>()?;
        // A region belongs to one operation, so one named twice here would be
        // owned twice — decided before anything takes ownership of anything.
        let mut held = Vec::with_capacity(regions.len());
        for region in regions {
            let index = self.region_of_id(region)?;
            if held.contains(&RegionId::new(self.tag, index)) {
                return Err(EditError::Twice {
                    what: "region",
                    index,
                });
            }
            if let Some(owner) = self.regions[index].owner {
                return Err(EditError::AlreadyAStep {
                    op: owner.index(),
                    block: index,
                });
            }
            held.push(RegionId::new(self.tag, index));
        }
        Ok(self.add_op(kind, placed, operands, result_tys, into, held))
    }

    /// A block belonging to no region yet, for a caller building one.
    pub fn new_block(&mut self) -> BlockId {
        self.add_block()
    }

    /// A region over blocks that belong to no other.
    pub fn new_region(&mut self, blocks: Vec<BlockId>) -> Result<RegionId> {
        let mut held = Vec::with_capacity(blocks.len());
        for block in blocks {
            let index = self.block_of_id(block)?;
            // A block belongs to one region, so one named twice here would
            // belong to this one twice — decided before this region exists.
            if held.contains(&BlockId::new(self.tag, index)) {
                return Err(EditError::Twice {
                    what: "block",
                    index,
                });
            }
            if let Some(region) = self.blocks[index].region {
                return Err(EditError::BlockTaken {
                    block: index,
                    region: region.index(),
                });
            }
            held.push(BlockId::new(self.tag, index));
        }
        Ok(self.add_region(held))
    }

    /// One more value a block takes on the way in. The entry block is refused:
    /// what it takes is exactly what the kernel declares, position by position.
    pub fn new_block_param(&mut self, block: BlockId, ty: ValTy) -> Result<ValueId> {
        let held = self.block_of_id(block)?;
        if BlockId::new(self.tag, held) == self.entry {
            return Err(EditError::EntryTakesDeclarations);
        }
        if matches!(ty, ValTy::Data(Ty { lanes: 0, .. })) {
            return Err(EditError::ZeroLanes);
        }
        Ok(self.add_block_param(BlockId::new(self.tag, held), ty))
    }

    /// One more parameter of the kernel this body belongs to, appended to what
    /// it declares and to what its entry block takes.
    pub fn declare_param(&mut self, param: Param) -> Result<ValueId> {
        self.refuse_second_declaration(&param.name)?;
        let ty = match param.ty {
            ParamTy::Ptr(elem) => ValTy::Ptr(PtrTy {
                elem,
                space: Space::Global,
            }),
            ParamTy::Scalar(scalar) => ValTy::Data(Ty::scalar(scalar)),
        };
        let name = param.name.clone();
        let id = self.declare(Declaration::Param(param), ty);
        self.set_name(id, name);
        Ok(id)
    }

    /// One more shared array of the kernel this body belongs to.
    pub fn declare_shared(&mut self, array: SharedArr) -> Result<ValueId> {
        self.refuse_second_declaration(&array.name)?;
        let ty = ValTy::Ptr(PtrTy {
            elem: array.elem,
            space: Space::Shared,
        });
        let name = array.name.clone();
        let id = self.declare(Declaration::Shared(array), ty);
        self.set_name(id, name);
        Ok(id)
    }

    fn refuse_second_declaration(&self, name: &str) -> Result {
        let taken = self.declarations.iter().any(|declared| match declared {
            Declaration::Param(param) => param.name == name,
            Declaration::Shared(array) => array.name == name,
        });
        if taken {
            return Err(EditError::Redeclared {
                name: name.to_string(),
            });
        }
        Ok(())
    }

    // ---- where a step runs ------------------------------------------------

    /// Places `op` as the `position`-th step of `block`.
    pub fn insert_step(&mut self, block: BlockId, position: usize, op: OpId) -> Result {
        let block = self.block_of_id(block)?;
        let held = self.op_of(op)?;
        let op = OpId::new(self.tag, held);
        if let Some((at, _)) = self.step_of(op)? {
            return Err(EditError::AlreadyAStep {
                op: held,
                block: at.index(),
            });
        }
        let steps = self.blocks[block].ops.len();
        if position > steps {
            return Err(EditError::OutOfRange {
                what: "step",
                index: position,
                held: steps,
            });
        }
        self.blocks[block].ops.insert(position, op);
        Ok(())
    }

    /// Compatibility operation: every operation remains in its block. It
    /// returns the stored-step position without changing the function.
    pub fn detach_step(&mut self, op: OpId) -> Result<(BlockId, usize)> {
        let held = self.op_of(op)?;
        let op = OpId::new(self.tag, held);
        if !self.ops[held].placed {
            return Err(EditError::NotAStep { op: held });
        }
        let Some((block, raw_position)) = self.step_of(op)? else {
            return Err(EditError::NotAStep { op: held });
        };
        for &result in &self.ops[held].results.clone() {
            if let Some(first) = self.uses(result)?.first() {
                return Err(EditError::StillRead {
                    value: result.index(),
                    by: first.describe(),
                });
            }
        }
        let position = self.blocks[block.index()].ops[..raw_position]
            .iter()
            .filter(|&&candidate| self.ops[candidate.index()].placed)
            .count();
        // Kept for compatibility: an operation now always remains in its block.
        Ok((block, position))
    }

    /// Takes `op` out of the arena, with the values it produced and whatever it
    /// held.
    ///
    /// It must already be out of the block it ran in, nothing may read what it
    /// produced, and anything under it must be empty — so what this decides is
    /// only that the operation has stopped being part of the program. Its id,
    /// and the ids of its results, stop naming anything from here: a handle
    /// somebody kept goes stale rather than coming to mean something else, and
    /// every other handle goes on meaning what it meant.
    pub fn delete_op(&mut self, op: OpId) -> Result {
        let held = self.op_of(op)?;
        let op = OpId::new(self.tag, held);
        for &result in &self.ops[held].results.clone() {
            if let Some(first) = self.uses(result)?.first() {
                return Err(EditError::StillRead {
                    value: result.index(),
                    by: first.describe(),
                });
            }
        }
        let home = self.step_of(op)?;
        // What it holds goes with it, and only if no stored operation in there
        // still runs and nothing is still handed back.
        let mut blocks = Vec::new();
        let mut regions = Vec::new();
        for &region in &self.ops[held].regions.clone() {
            let at = self.region_of_id(region)?;
            regions.push(at);
            for &block in &self.regions[at].blocks.clone() {
                let inner = self.block_of_id(block)?;
                if self.blocks[inner]
                    .ops
                    .iter()
                    .any(|&candidate| self.ops[candidate.index()].placed)
                    || !self.blocks[inner].terminator.yields.is_empty()
                {
                    return Err(EditError::NotEmpty { block: inner });
                }
                for &param in &self.blocks[inner].params.clone() {
                    if let Some(first) = self.uses(param)?.first() {
                        return Err(EditError::StillRead {
                            value: param.index(),
                            by: first.describe(),
                        });
                    }
                }
                blocks.push(inner);
            }
        }
        self.deleted.ops.insert(held as u32);
        for &block in &blocks {
            for operand in self.blocks[block].ops.clone() {
                self.deleted.ops.insert(operand.index() as u32);
            }
        }
        for at in regions {
            self.deleted.regions.insert(at as u32);
        }
        for at in blocks {
            self.deleted.blocks.insert(at as u32);
        }
        if let Some((block, _)) = home {
            self.blocks[block.index()]
                .ops
                .retain(|&candidate| candidate != op);
        }
        Ok(())
    }

    /// Runs `op` somewhere else: out of its current block and into `block` at
    /// `position`, counted among operations that execute at their position.
    ///
    /// Decided before anything moves, so a refusal leaves the two blocks exactly
    /// as they were.
    pub fn move_step(&mut self, op: OpId, block: BlockId, position: usize) -> Result {
        let held = self.op_of(op)?;
        let into = self.block_of_id(block)?;
        let op = OpId::new(self.tag, held);
        if !self.ops[held].placed {
            return Err(EditError::NotAStep { op: held });
        }
        let Some((from, at)) = self.step_of(op)? else {
            return Err(EditError::NotAStep { op: held });
        };
        let visible = self.blocks[into]
            .ops
            .iter()
            .filter(|&&candidate| candidate != op && self.ops[candidate.index()].placed)
            .count();
        if position > visible {
            return Err(EditError::OutOfRange {
                what: "step",
                index: position,
                held: visible,
            });
        }
        let mut insertion = self.blocks[into].ops.len();
        let mut seen = 0;
        for (index, &candidate) in self.blocks[into].ops.iter().enumerate() {
            if candidate == op {
                continue;
            }
            if self.ops[candidate.index()].placed {
                if seen == position {
                    insertion = index;
                    break;
                }
                seen += 1;
            }
        }
        if position == visible {
            insertion = self.blocks[into]
                .ops
                .iter()
                .rposition(|&candidate| candidate != op && self.ops[candidate.index()].placed)
                .map_or(0, |index| index + 1);
        }
        self.blocks[from.index()].ops.remove(at);
        let insertion =
            insertion.saturating_sub(usize::from(from.index() == into && at < insertion));
        self.blocks[into].ops.insert(insertion, op);
        Ok(())
    }

    /// The steps of one block, for a caller replacing the whole list.
    ///
    /// One operation is a step of one block, so an operation another block
    /// already runs is refused here rather than quietly run twice — moving one
    /// between blocks is [`Function::move_step`], which takes it out of the
    /// first. Decided over the whole arena before this block changes, so a
    /// refusal leaves both blocks exactly as they were.
    pub fn set_steps(&mut self, block: BlockId, ops: Vec<OpId>) -> Result {
        let block = self.block_of_id(block)?;
        let here = BlockId::new(self.tag, block);
        let mut held = Vec::with_capacity(ops.len());
        for op in ops {
            let index = self.op_of(op)?;
            let op = OpId::new(self.tag, index);
            if held.contains(&op) {
                return Err(EditError::Twice {
                    what: "operation",
                    index,
                });
            }
            if let Some((at, _)) = self.step_of(op)?
                && at != here
            {
                return Err(EditError::AlreadyAStep {
                    op: index,
                    block: at.index(),
                });
            }
            held.push(op);
        }
        self.blocks[block].ops = held;
        Ok(())
    }
}

/// One word per kind, for a caller naming what it met. The match is
/// exhaustive, so a kind added to the language has to be named here.
pub fn kind_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::IntConst { .. } => "an integer constant",
        OpKind::FloatConst { .. } => "a float constant",
        OpKind::Sreg(_) => "a special register",
        OpKind::Load { .. } => "a load",
        OpKind::Offset => "an offset",
        OpKind::Un(_) => "a unary operation",
        OpKind::Bin(_) => "a binary operation",
        OpKind::Fma => "a fused multiply-add",
        OpKind::Cmp(_) => "a comparison",
        OpKind::Select => "a select",
        OpKind::Cast(_) => "a cast",
        OpKind::Splat(_) => "a splat",
        OpKind::Extract(_) => "an extract",
        OpKind::Pack => "a pack",
        OpKind::ShflDown => "a shuffle",
        OpKind::ActiveMask => "an active mask",
        OpKind::ShflFrom => "a shuffle from a lane",
        OpKind::WarpAny => "a warp reduction",
        OpKind::ConstLoad(_) => "a constant load",
        OpKind::Bind => "a binding",
        OpKind::Write { .. } => "a write",
        OpKind::FetchAdd => "an atomic add",
        OpKind::If => "a conditional",
        OpKind::Loop { .. } => "a loop",
        OpKind::Scope => "a scope",
        OpKind::Call(_) => "a call",
        OpKind::Break => "a break",
        OpKind::Ret => "a return",
        OpKind::Bar => "a barrier",
    }
}
