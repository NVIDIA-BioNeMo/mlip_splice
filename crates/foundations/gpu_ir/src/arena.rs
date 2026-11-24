// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The arena a device function is: one owner, dense ids, no interior pointers.
//!
//! A [`Function`] owns every value, operation, block and region of one program.
//! Nothing outside it owns a node, so an id is an index and a use-def edge is a
//! [`ValueId`] rather than a name to look up. A finished function is plain data
//! — `Send + Sync`, no reference counting inside, no cells — which is what lets
//! candidates compile in parallel and, later, be held behind an `Arc`.
//!
//! Every id carries the tag of the arena it indexes and the tag is always
//! checked. Two functions both number from zero, so an unchecked index would
//! name a real value of the wrong arena; that is the mistake the tag exists to
//! stop, and it costs one `u32` compare against a `HashMap` probe it replaces.
//!
//! Cloning **re-tags**: an arena is cloned in order to diverge from it, so an id
//! of the original must not silently address the copy.
//! [`Function::clone_with_map`] is the way to carry ids across.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::op::OpKind;
use crate::{Param, SharedArr, Ty, ValTy};

pub mod compact;
pub use compact::{FrozenBlockId, FrozenFunction, FrozenOpId, FrozenRegionId, FrozenValueId};
pub mod edit;
mod graph;
pub(crate) use graph::Readers;
pub mod text;
pub mod transact;
pub mod wire;

#[cfg(test)]
mod tests;

/// Which arena an id indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaTag(u32);

impl ArenaTag {
    fn mint() -> ArenaTag {
        static NEXT: AtomicU32 = AtomicU32::new(1);
        // Reusing a tag would make two arenas indistinguishable, which is the
        // one thing the tag exists to prevent, so running out is a stop.
        let tag = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |tag| {
                tag.checked_add(1)
            })
            .expect("arena tags exhausted");
        ArenaTag(tag)
    }
}

macro_rules! tagged_id {
    ($name:ident, $what:literal) => {
        #[doc = concat!("A dense index of one ", $what, ", tagged with its arena.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            arena: ArenaTag,
            index: u32,
        }

        impl $name {
            fn new(arena: ArenaTag, index: usize) -> $name {
                $name {
                    arena,
                    index: u32::try_from(index).expect(concat!($what, " ids exhausted")),
                }
            }

            /// Where this sits in its arena. Only meaningful beside the arena
            /// that issued it.
            pub fn index(self) -> usize {
                self.index as usize
            }

            /// Which arena issued it, for a caller that answers a foreign id
            /// rather than stopping on one.
            pub(crate) fn arena(self) -> ArenaTag {
                self.arena
            }

            pub(crate) fn at(self, arena: ArenaTag) -> usize {
                assert!(
                    self.arena == arena,
                    concat!("a ", $what, " of {:?} was used in {:?}"),
                    self.arena,
                    arena
                );
                self.index as usize
            }
        }
    };
}

tagged_id!(ValueId, "value");
tagged_id!(OpId, "operation");
tagged_id!(BlockId, "block");
tagged_id!(RegionId, "region");

/// Where a value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Def {
    /// The `index`-th parameter a block takes: a kernel's own parameters and
    /// shared arrays on the entry block, a loop's index and carried values on
    /// its body.
    Param { block: BlockId, index: u32 },
    /// The `index`-th result of an operation.
    Result { op: OpId, index: u32 },
}

#[derive(Clone, Debug)]
pub struct ValueDef {
    pub ty: ValTy,
    pub def: Def,
}

#[derive(Clone, Debug)]
pub struct OpDef {
    pub kind: OpKind,
    /// Whether this operation produces storage computed once at its block
    /// position. Every operation belongs to a block; when this is false, its
    /// result is re-evaluated at each use.
    pub placed: bool,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    /// Per result, the value whose storage it is produced into.
    ///
    /// A caller updating an accumulator knows which place it updates, while no
    /// liveness analysis can recover that choice from one value per definition.
    /// Empty when no result takes anything over.
    pub into: Vec<Option<ValueId>>,
    pub regions: Vec<RegionId>,
}

impl OpDef {
    /// The value the `index`-th result is produced into, if it takes one over.
    pub fn destination(&self, index: usize) -> Option<ValueId> {
        self.into.get(index).copied().flatten()
    }
}

/// What a block ends with: one value per result of the operation owning the
/// region it belongs to. A kernel's entry block yields nothing.
#[derive(Clone, Debug, Default)]
pub struct Terminator {
    pub yields: Vec<ValueId>,
}

#[derive(Clone, Debug, Default)]
pub struct BlockDef {
    pub params: Vec<ValueId>,
    /// The operations this block runs, in order.
    pub ops: Vec<OpId>,
    pub terminator: Terminator,
    /// The region this block belongs to, `None` for the entry block, which
    /// belongs to the function itself. Set as the region is created, so asking
    /// what encloses a block is a lookup rather than a search.
    pub region: Option<RegionId>,
}

#[derive(Clone, Debug)]
pub struct RegionDef {
    /// The operation this region belongs to; every region belongs to exactly
    /// one. Set as the operation is created, the only moment both ids exist.
    pub owner: Option<OpId>,
    /// One block in phase C; the verifier is what will relax that, not the
    /// format.
    pub blocks: Vec<BlockId>,
}

/// What a kernel declares, in the order it was declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Declaration {
    Param(Param),
    Shared(SharedArr),
}

/// One device function: every value, operation, block and region it is made of.
#[derive(Debug)]
pub struct Function {
    tag: ArenaTag,
    values: Vec<ValueDef>,
    ops: Vec<OpDef>,
    blocks: Vec<BlockDef>,
    regions: Vec<RegionDef>,
    entry: BlockId,
    /// What the kernel takes, paired with the entry parameter standing for it:
    /// `declarations[k]` is `blocks[entry].params[k]`.
    declarations: Vec<Declaration>,
    /// Names a caller chose, keyed by value. They carry meaning the arena does
    /// not — which stage a value belongs to, which output it addresses — and
    /// nothing about the program's meaning reads them, so identity excludes
    /// them for free.
    names: BTreeMap<u32, String>,
    /// What an editor deleted. Ids are dense and are never reused, so a deleted
    /// entity keeps its slot and its id stops naming anything: a handle a caller
    /// kept goes stale rather than coming to mean whatever was added next.
    /// [`Function::compacted`] is where the slots actually close up, and it is
    /// reached only where a whole program is written down or handed on.
    deleted: Deleted,
}

/// The ids an editor deleted. A value is deleted with whatever defines it —
/// the operation that produces it, or the block that takes it — so there is
/// nothing to record about values separately.
#[derive(Clone, Debug, Default)]
pub(crate) struct Deleted {
    pub(crate) ops: DeletedIds,
    pub(crate) blocks: DeletedIds,
    pub(crate) regions: DeletedIds,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DeletedIds {
    deleted: fixedbitset::FixedBitSet,
}

impl DeletedIds {
    pub(crate) fn contains(&self, id: &u32) -> bool {
        self.deleted.contains(*id as usize)
    }

    pub(crate) fn insert(&mut self, id: u32) -> bool {
        let index = id as usize;
        if self.deleted.len() <= index {
            self.deleted.grow(index + 1);
        }
        let inserted = !self.deleted.contains(index);
        self.deleted.insert(index);
        inserted
    }

    pub(crate) fn extend(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            self.insert(id);
        }
    }

    fn is_empty(&self) -> bool {
        self.deleted.is_clear()
    }
}

impl Deleted {
    pub(crate) fn any(&self) -> bool {
        !self.ops.is_empty() || !self.blocks.is_empty() || !self.regions.is_empty()
    }
}

impl Default for Function {
    fn default() -> Function {
        Function::new()
    }
}

/// Cloning an arena re-tags it: a clone is made in order to diverge, so an id of
/// the original must not address the copy. [`Function::clone_with_map`] is the
/// way to carry ids across.
impl Clone for Function {
    fn clone(&self) -> Function {
        self.clone_with_map().0
    }
}

impl Function {
    pub fn new() -> Function {
        let tag = ArenaTag::mint();
        Function {
            tag,
            values: Vec::new(),
            ops: Vec::new(),
            blocks: vec![BlockDef::default()],
            regions: Vec::new(),
            entry: BlockId::new(tag, 0),
            declarations: Vec::new(),
            names: BTreeMap::new(),
            deleted: Deleted::default(),
        }
    }

    pub fn tag(&self) -> ArenaTag {
        self.tag
    }

    pub fn entry(&self) -> BlockId {
        self.entry
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// The value at `index`, for a caller walking every value of this arena.
    pub fn value_at(&self, index: usize) -> ValueId {
        assert!(index < self.values.len(), "no value {index} in this arena");
        ValueId::new(self.tag, index)
    }

    /// The operation at `index`, for a caller walking every operation of this
    /// arena.
    pub fn op_at(&self, index: usize) -> OpId {
        assert!(index < self.ops.len(), "no operation {index} in this arena");
        OpId::new(self.tag, index)
    }

    pub fn value(&self, value: ValueId) -> &ValueDef {
        &self.values[value.at(self.tag)]
    }

    pub fn ty(&self, value: ValueId) -> ValTy {
        self.value(value).ty
    }

    pub fn data_ty(&self, value: ValueId) -> Ty {
        self.ty(value).data()
    }

    pub fn op(&self, op: OpId) -> &OpDef {
        &self.ops[op.at(self.tag)]
    }

    pub fn block(&self, block: BlockId) -> &BlockDef {
        &self.blocks[block.at(self.tag)]
    }

    pub fn region(&self, region: RegionId) -> &RegionDef {
        &self.regions[region.at(self.tag)]
    }

    /// Whether `op` is an operation that executes at its block position.
    pub fn is_step(&self, op: OpId) -> bool {
        self.op(op).placed
    }

    /// The operations of `block` that execute there, in order.
    pub fn block_steps(&self, block: BlockId) -> impl Iterator<Item = OpId> + '_ {
        self.block(block)
            .ops
            .iter()
            .copied()
            .filter(|&op| self.is_step(op))
    }

    /// The operation whose region holds `block`, and `None` for the entry
    /// block, which the function itself holds.
    pub fn block_owner(&self, block: BlockId) -> Option<OpId> {
        self.region(self.block(block).region?).owner
    }

    /// The single block of a region, which is all a region has in phase C.
    pub fn region_block(&self, region: RegionId) -> BlockId {
        let blocks = &self.region(region).blocks;
        assert_eq!(blocks.len(), 1, "a region holds one block");
        blocks[0]
    }

    pub fn declarations(&self) -> &[Declaration] {
        &self.declarations
    }

    pub fn parameters(&self) -> Vec<Param> {
        self.declarations
            .iter()
            .filter_map(|declared| match declared {
                Declaration::Param(param) => Some(param.clone()),
                Declaration::Shared(_) => None,
            })
            .collect()
    }

    pub fn shared_arrays(&self) -> Vec<SharedArr> {
        self.declarations
            .iter()
            .filter_map(|declared| match declared {
                Declaration::Shared(array) => Some(array.clone()),
                Declaration::Param(_) => None,
            })
            .collect()
    }

    /// The value standing for the `index`-th declaration.
    pub fn declared(&self, index: usize) -> ValueId {
        self.block(self.entry).params[index]
    }

    /// What this function hands back: the values its entry block yields. A
    /// kernel yields nothing, which is what the launch ABI requires of it.
    pub fn returned(&self) -> &[ValueId] {
        &self.block(self.entry).terminator.yields
    }

    /// The value a caller named, for a name that carries meaning across two
    /// evaluations of the same program — which binding a materialisation cuts
    /// at, say. Values do not cross arenas; names of the same program do.
    pub fn value_named(&self, name: &str) -> Option<ValueId> {
        self.names
            .iter()
            .find(|(_, held)| held.as_str() == name)
            .map(|(&index, _)| ValueId::new(self.tag, index as usize))
    }

    pub fn name(&self, value: ValueId) -> Option<&str> {
        self.names
            .get(&(value.at(self.tag) as u32))
            .map(String::as_str)
    }

    /// Whether this value has storage of its own that the block computes once,
    /// as opposed to an expression recomputed wherever it is read.
    ///
    /// A block parameter and the result of a [`OpKind::Bind`], an
    /// [`OpKind::If`] or an [`OpKind::Loop`] are places; everything else is an
    /// operand.
    pub fn is_materialized(&self, value: ValueId) -> bool {
        match self.value(value).def {
            Def::Param { .. } => true,
            Def::Result { op, .. } => self.op(op).placed,
        }
    }

    /// The operation defining `value`, when it is an operand rather than a
    /// place.
    pub fn operand_op(&self, value: ValueId) -> Option<OpId> {
        match self.value(value).def {
            Def::Result { op, .. } if !self.op(op).placed => Some(op),
            _ => None,
        }
    }

    /// The operation defining `value`, whatever it is.
    pub fn defining_op(&self, value: ValueId) -> Option<OpId> {
        match self.value(value).def {
            Def::Result { op, .. } => Some(op),
            Def::Param { .. } => None,
        }
    }

    // ---- construction ----------------------------------------------------

    fn add_value(&mut self, ty: ValTy, def: Def) -> ValueId {
        let id = ValueId::new(self.tag, self.values.len());
        self.values.push(ValueDef { ty, def });
        id
    }

    pub(crate) fn add_block(&mut self) -> BlockId {
        let id = BlockId::new(self.tag, self.blocks.len());
        self.blocks.push(BlockDef::default());
        id
    }

    pub(crate) fn add_block_param(&mut self, block: BlockId, ty: ValTy) -> ValueId {
        let index = self.block(block).params.len() as u32;
        let value = self.add_value(ty, Def::Param { block, index });
        let block = block.at(self.tag);
        self.blocks[block].params.push(value);
        value
    }

    pub(crate) fn add_region(&mut self, blocks: Vec<BlockId>) -> RegionId {
        let id = RegionId::new(self.tag, self.regions.len());
        for &block in &blocks {
            let block = block.at(self.tag);
            assert!(
                self.blocks[block].region.is_none(),
                "a block already belongs to a region"
            );
            self.blocks[block].region = Some(id);
        }
        self.regions.push(RegionDef {
            owner: None,
            blocks,
        });
        id
    }

    /// Adds an operation and the values it results in.
    pub(crate) fn add_op(
        &mut self,
        kind: OpKind,
        placed: bool,
        operands: Vec<ValueId>,
        result_tys: &[ValTy],
        into: Vec<Option<ValueId>>,
        regions: Vec<RegionId>,
    ) -> OpId {
        assert_eq!(regions.len(), kind.regions(), "region count of {kind:?}");
        assert!(
            placed == kind.must_be_placed() || kind.may_be_either(),
            "{kind:?} cannot be {}",
            if placed { "placed" } else { "an operand" }
        );
        assert!(
            into.is_empty() || into.len() == result_tys.len(),
            "one destination per result"
        );
        let op = OpId::new(self.tag, self.ops.len());
        let results = result_tys
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                self.add_value(
                    ty,
                    Def::Result {
                        op,
                        index: index as u32,
                    },
                )
            })
            .collect();
        for &region in &regions {
            let region = region.at(self.tag);
            self.regions[region].owner = Some(op);
        }
        self.ops.push(OpDef {
            kind,
            placed,
            operands,
            results,
            into,
            regions,
        });
        self.name_after_destinations(op);
        op
    }

    /// Gives each result that is produced into a place the name that place
    /// carries.
    ///
    /// A name is provenance — which stage a value belongs to, which output it
    /// accumulates — and a value produced into a place continues that place, so
    /// it continues its provenance. Nothing about the program's meaning reads a
    /// name, so this shares one rather than inventing a second.
    fn name_after_destinations(&mut self, op: OpId) {
        for index in 0..self.op(op).into.len() {
            let Some(destination) = self.op(op).destination(index) else {
                continue;
            };
            let Some(name) = self.name(destination).map(str::to_owned) else {
                continue;
            };
            let result = self.op(op).results[index];
            self.set_name(result, name);
        }
    }

    pub(crate) fn set_terminator(&mut self, block: BlockId, yields: Vec<ValueId>) {
        let block = block.at(self.tag);
        self.blocks[block].terminator = Terminator { yields };
    }

    /// Replaces the operations of `block`, keeping its parameters and what it
    /// ends with. Every operation belongs to a block; `placed` only decides
    /// whether its results have storage or are re-evaluated at their uses.
    pub fn set_block_ops(&mut self, block: BlockId, ops: Vec<OpId>) {
        let block = block.at(self.tag);
        self.blocks[block].ops = ops;
    }

    pub(crate) fn declare(&mut self, declaration: Declaration, ty: ValTy) -> ValueId {
        let entry = self.entry;
        let value = self.add_block_param(entry, ty);
        self.declarations.push(declaration);
        value
    }

    pub(crate) fn set_name(&mut self, value: ValueId, name: String) {
        let index = value.at(self.tag) as u32;
        self.names.insert(index, name);
    }

    // ---- carrying ids across ---------------------------------------------

    /// A copy of this function under a fresh tag, with the copy's value for
    /// every value of the original, by index.
    ///
    /// The copy is a different arena, so an id of the original does not address
    /// it; the map is the only way across, which is what makes the crossing
    /// explicit rather than accidental.
    pub fn clone_with_map(&self) -> (Function, Vec<ValueId>) {
        let copy = self.copied(ArenaTag::mint());
        let mapped = (0..self.values.len())
            .map(|index| ValueId::new(copy.tag, index))
            .collect();
        (copy, mapped)
    }

    /// A copy under whatever tag is asked for.
    ///
    /// A copy under a *fresh* tag is what [`Function::clone`] and
    /// [`Function::clone_with_map`] hand out. A copy under this arena's **own**
    /// tag is a twin whose ids are indistinguishable from this one's, so it is
    /// sound only where the original cannot be observed while it lives: this is
    /// private, and [`transact`](super::arena::transact) — the arena's own
    /// descendant, holding `&mut self` throughout — is the only caller that asks
    /// for the same tag.
    fn copied(&self, tag: ArenaTag) -> Function {
        let value = |id: &ValueId| ValueId::new(tag, id.index as usize);
        let op = |id: &OpId| OpId::new(tag, id.index as usize);
        let block = |id: &BlockId| BlockId::new(tag, id.index as usize);
        let region = |id: &RegionId| RegionId::new(tag, id.index as usize);
        Function {
            tag,
            values: self
                .values
                .iter()
                .map(|held| ValueDef {
                    ty: held.ty,
                    def: match &held.def {
                        Def::Param { block: at, index } => Def::Param {
                            block: block(at),
                            index: *index,
                        },
                        Def::Result { op: at, index } => Def::Result {
                            op: op(at),
                            index: *index,
                        },
                    },
                })
                .collect(),
            ops: self
                .ops
                .iter()
                .map(|held| OpDef {
                    kind: held.kind.clone(),
                    placed: held.placed,
                    operands: held.operands.iter().map(value).collect(),
                    results: held.results.iter().map(value).collect(),
                    into: held
                        .into
                        .iter()
                        .map(|destination| destination.as_ref().map(value))
                        .collect(),
                    regions: held.regions.iter().map(region).collect(),
                })
                .collect(),
            blocks: self
                .blocks
                .iter()
                .map(|held| BlockDef {
                    params: held.params.iter().map(value).collect(),
                    ops: held.ops.iter().map(op).collect(),
                    terminator: Terminator {
                        yields: held.terminator.yields.iter().map(value).collect(),
                    },
                    region: held.region.as_ref().map(region),
                })
                .collect(),
            regions: self
                .regions
                .iter()
                .map(|held| RegionDef {
                    owner: held.owner.as_ref().map(op),
                    blocks: held.blocks.iter().map(block).collect(),
                })
                .collect(),
            entry: block(&self.entry),
            declarations: self.declarations.clone(),
            names: self.names.clone(),
            deleted: self.deleted.clone(),
        }
    }
}
