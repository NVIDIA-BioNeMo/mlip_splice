// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where the two numberings meet.
//!
//! Ids are stable *within* an arena, which is what an editor needs: one deleted
//! keeps its slot and stops naming anything, so a handle a caller kept goes
//! stale rather than coming to mean whatever was added next. Ids are dense *in*
//! one, which is what everything reading a whole program needs: bytes, text and
//! the backend all address by index.
//!
//! Those are two numberings, and this is the one place one becomes the other.

use index_vec::IndexVec;

index_vec::define_index_type! {
    pub struct FrozenValueId = u32;
    DEBUG_FORMAT = "frozen-value({})";
}
index_vec::define_index_type! {
    pub struct FrozenOpId = u32;
    DEBUG_FORMAT = "frozen-operation({})";
}
index_vec::define_index_type! {
    pub struct FrozenBlockId = u32;
    DEBUG_FORMAT = "frozen-block({})";
}
index_vec::define_index_type! {
    pub struct FrozenRegionId = u32;
    DEBUG_FORMAT = "frozen-region({})";
}

use super::{
    BlockDef, BlockId, Deleted, Function, OpDef, OpId, RegionDef, RegionId, Terminator, ValueDef,
    ValueId,
};

/// A compact function whose ids are dense and whose contents no editor can mutate.
pub struct FrozenFunction {
    tag: super::ArenaTag,
    function: Function,
    uses: weave::Uses,
    operation_blocks: IndexVec<FrozenOpId, Option<FrozenBlockId>>,
    operation_positions: IndexVec<FrozenOpId, Option<usize>>,
    operation_listings: IndexVec<FrozenOpId, u32>,
}

impl FrozenFunction {
    pub fn function(&self) -> &Function {
        &self.function
    }

    pub fn frozen_value(&self, value: ValueId) -> FrozenValueId {
        FrozenValueId::new(value.at(self.tag))
    }

    pub fn frozen_operation(&self, operation: OpId) -> FrozenOpId {
        FrozenOpId::new(operation.at(self.tag))
    }

    pub fn frozen_block(&self, block: BlockId) -> FrozenBlockId {
        FrozenBlockId::new(block.at(self.tag))
    }

    pub fn frozen_region(&self, region: RegionId) -> FrozenRegionId {
        FrozenRegionId::new(region.at(self.tag))
    }

    pub fn into_function(self) -> Function {
        self.function
    }

    pub fn uses(&self) -> &weave::Uses {
        &self.uses
    }

    pub fn operation_block(&self, operation: FrozenOpId) -> Option<FrozenBlockId> {
        self.operation_blocks[operation]
    }

    pub fn operation_position(&self, operation: FrozenOpId) -> Option<usize> {
        self.operation_positions[operation]
    }

    pub fn operation_listing_count(&self, operation: FrozenOpId) -> u32 {
        self.operation_listings[operation]
    }

    pub fn readers(&self, value: FrozenValueId) -> impl Iterator<Item = FrozenOpId> + '_ {
        self.uses
            .of_value(value.index())
            .iter()
            .map(|&reader| FrozenOpId::new(reader as usize))
    }

    pub fn value(&self, value: FrozenValueId) -> &ValueDef {
        &self.function.values[value.index()]
    }

    pub fn operation(&self, operation: FrozenOpId) -> &OpDef {
        &self.function.ops[operation.index()]
    }

    pub fn block(&self, block: FrozenBlockId) -> &BlockDef {
        &self.function.blocks[block.index()]
    }

    pub fn region(&self, region: FrozenRegionId) -> &RegionDef {
        &self.function.regions[region.index()]
    }

    pub fn value_count(&self) -> usize {
        self.function.values.len()
    }

    pub fn operation_count(&self) -> usize {
        self.function.ops.len()
    }

    pub fn block_count(&self) -> usize {
        self.function.blocks.len()
    }

    pub fn region_count(&self) -> usize {
        self.function.regions.len()
    }

    pub fn values(&self) -> impl Iterator<Item = FrozenValueId> {
        (0..self.value_count()).map(FrozenValueId::new)
    }

    pub fn operations(&self) -> impl Iterator<Item = FrozenOpId> {
        (0..self.operation_count()).map(FrozenOpId::new)
    }

    pub fn blocks(&self) -> impl Iterator<Item = FrozenBlockId> {
        (0..self.block_count()).map(FrozenBlockId::new)
    }

    pub fn regions(&self) -> impl Iterator<Item = FrozenRegionId> {
        (0..self.region_count()).map(FrozenRegionId::new)
    }
}

impl Function {
    /// Consumes the editable arena and closes every deleted slot.
    pub fn freeze(self) -> FrozenFunction {
        let function = if self.has_deletions() {
            self.compacted()
        } else {
            self
        };
        let tag = function.tag;
        let uses = weave::Uses::of_arena(&function);
        let mut operation_blocks = IndexVec::from_vec(vec![None; function.num_ops()]);
        let mut operation_positions = IndexVec::from_vec(vec![None; function.num_ops()]);
        let mut operation_listings = IndexVec::from_vec(vec![0; function.num_ops()]);
        for block in function.live_blocks() {
            let frozen_block = FrozenBlockId::new(block.index());
            for (position, &operation) in function.block(block).ops.iter().enumerate() {
                let frozen_operation = FrozenOpId::new(operation.index());
                operation_blocks[frozen_operation] = Some(frozen_block);
                operation_positions[frozen_operation] = Some(position);
                operation_listings[frozen_operation] += 1;
            }
        }
        FrozenFunction {
            tag,
            function,
            uses,
            operation_blocks,
            operation_positions,
            operation_listings,
        }
    }

    /// This arena with the deleted slots closed up: the same program, numbered
    /// from zero again.
    ///
    /// An arena with nothing deleted comes through unchanged, position for
    /// position, which is why writing an unedited program down is still the
    /// bytes it came from.
    pub fn compacted(&self) -> Function {
        let value_of = renumbering(self.values.len(), |index| self.value_is_live(index));
        let op_of = renumbering(self.ops.len(), |index| {
            !self.deleted.ops.contains(&(index as u32))
        });
        let block_of = renumbering(self.blocks.len(), |index| {
            !self.deleted.blocks.contains(&(index as u32))
        });
        let region_of = renumbering(self.regions.len(), |index| {
            !self.deleted.regions.contains(&(index as u32))
        });
        let tag = super::ArenaTag::mint();
        let value = |id: &ValueId| {
            ValueId::new(
                tag,
                value_of[FrozenValueId::new(id.index())].expect("live value"),
            )
        };
        let op = |id: &OpId| {
            OpId::new(
                tag,
                op_of[FrozenOpId::new(id.index())].expect("live operation"),
            )
        };
        let block = |id: &BlockId| {
            BlockId::new(
                tag,
                block_of[FrozenBlockId::new(id.index())].expect("live block"),
            )
        };
        let region = |id: &RegionId| {
            RegionId::new(
                tag,
                region_of[FrozenRegionId::new(id.index())].expect("live region"),
            )
        };
        Function {
            tag,
            values: (0..self.values.len())
                .filter(|&index| value_of[FrozenValueId::new(index)].is_some())
                .map(|index| ValueDef {
                    ty: self.values[index].ty,
                    def: match &self.values[index].def {
                        super::Def::Param { block: at, index } => super::Def::Param {
                            block: block(at),
                            index: *index,
                        },
                        super::Def::Result { op: at, index } => super::Def::Result {
                            op: op(at),
                            index: *index,
                        },
                    },
                })
                .collect(),
            ops: (0..self.ops.len())
                .filter(|&index| op_of[FrozenOpId::new(index)].is_some())
                .map(|index| OpDef {
                    kind: self.ops[index].kind.clone(),
                    placed: self.ops[index].placed,
                    operands: self.ops[index].operands.iter().map(value).collect(),
                    results: self.ops[index].results.iter().map(value).collect(),
                    into: self.ops[index]
                        .into
                        .iter()
                        .map(|held| held.as_ref().map(value))
                        .collect(),
                    regions: self.ops[index].regions.iter().map(region).collect(),
                })
                .collect(),
            blocks: (0..self.blocks.len())
                .filter(|&index| block_of[FrozenBlockId::new(index)].is_some())
                .map(|index| BlockDef {
                    params: self.blocks[index].params.iter().map(value).collect(),
                    ops: self.blocks[index].ops.iter().map(op).collect(),
                    terminator: Terminator {
                        yields: self.blocks[index]
                            .terminator
                            .yields
                            .iter()
                            .map(value)
                            .collect(),
                    },
                    region: self.blocks[index].region.as_ref().map(region),
                })
                .collect(),
            regions: (0..self.regions.len())
                .filter(|&index| region_of[FrozenRegionId::new(index)].is_some())
                .map(|index| RegionDef {
                    owner: self.regions[index].owner.as_ref().map(op),
                    blocks: self.regions[index].blocks.iter().map(block).collect(),
                })
                .collect(),
            entry: block(&self.entry),
            declarations: self.declarations.clone(),
            names: self
                .names
                .iter()
                .filter(|(index, _)| value_of[FrozenValueId::new(**index as usize)].is_some())
                .map(|(&index, name)| {
                    (
                        value_of[FrozenValueId::new(index as usize)].expect("live named value")
                            as u32,
                        name.clone(),
                    )
                })
                .collect(),
            deleted: Deleted::default(),
        }
    }
}

/// Where each surviving index lands once the holes close up.
fn renumbering<I: index_vec::Idx>(
    count: usize,
    live: impl Fn(usize) -> bool,
) -> IndexVec<I, Option<usize>> {
    let mut next = 0;
    (0..count)
        .map(|index| {
            live(index).then(|| {
                let renamed = next;
                next += 1;
                renamed
            })
        })
        .collect()
}
