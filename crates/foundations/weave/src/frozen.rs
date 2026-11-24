// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compact, immutable IR for analysis, verification, and parallel readers.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::arena::Id;
use crate::ir::{Block, Operation, Region, Value};
use crate::{Attribute, OperationName, Operations, Type, Uses};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FrozenArenaId(u32);

impl FrozenArenaId {
    pub(crate) fn mint() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(1);
        let id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("frozen IR identities exhausted");
        Self(id)
    }
}

macro_rules! frozen_id {
    ($name:ident, $debug:literal) => {
        #[derive(Clone, Copy, Eq, Hash, PartialEq)]
        pub struct $name {
            arena: FrozenArenaId,
            index: u32,
        }

        impl $name {
            pub fn index(self) -> usize {
                self.index as usize
            }

            pub(crate) fn new(arena: FrozenArenaId, index: usize) -> Self {
                Self {
                    arena,
                    index: u32::try_from(index).expect(concat!($debug, " IDs exhausted")),
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!($debug, "({})"), self.index)
            }
        }
    };
}

frozen_id!(FrozenOperationId, "operation");
frozen_id!(FrozenRegionId, "region");
frozen_id!(FrozenBlockId, "block");
frozen_id!(FrozenValueId, "value");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrozenValueDefinition {
    OperationResult {
        operation: FrozenOperationId,
        index: u32,
    },
    BlockArgument {
        block: FrozenBlockId,
        index: u32,
    },
}

#[derive(Clone, Debug)]
pub struct FrozenValue {
    pub ty: Type,
    pub definition: FrozenValueDefinition,
}

#[derive(Clone, Debug)]
pub struct FrozenSuccessor {
    pub block: FrozenBlockId,
    pub arguments: Vec<FrozenValueId>,
}

#[derive(Clone, Debug)]
pub struct FrozenOperation {
    pub name: OperationName,
    pub parent: Option<FrozenBlockId>,
    pub operands: Vec<FrozenValueId>,
    pub results: Vec<FrozenValueId>,
    pub successors: Vec<FrozenSuccessor>,
    pub regions: Vec<FrozenRegionId>,
    pub attributes: BTreeMap<String, Attribute>,
}

#[derive(Clone, Debug)]
pub struct FrozenRegion {
    pub parent: FrozenOperationId,
    pub blocks: Vec<FrozenBlockId>,
}

#[derive(Clone, Debug)]
pub struct FrozenBlock {
    pub parent: FrozenRegionId,
    pub arguments: Vec<FrozenValueId>,
    pub operations: Vec<FrozenOperationId>,
}

/// One compact IR. Every ID is a typed position in one contiguous table.
pub struct FrozenIr {
    identity: FrozenArenaId,
    operations: Vec<FrozenOperation>,
    regions: Vec<FrozenRegion>,
    blocks: Vec<FrozenBlock>,
    values: Vec<FrozenValue>,
    roots: Vec<FrozenOperationId>,
    uses: Uses,
}

impl FrozenIr {
    pub(crate) fn new(
        identity: FrozenArenaId,
        operations: Vec<FrozenOperation>,
        regions: Vec<FrozenRegion>,
        blocks: Vec<FrozenBlock>,
        values: Vec<FrozenValue>,
        roots: Vec<FrozenOperationId>,
    ) -> Self {
        let mut frozen = Self {
            identity,
            operations,
            regions,
            blocks,
            values,
            roots,
            uses: Uses::default(),
        };
        frozen.uses = Uses::of_arena(&frozen);
        frozen
    }

    pub fn operation(&self, id: FrozenOperationId) -> &FrozenOperation {
        &self.operations[self.operation_index(id)]
    }

    pub fn region(&self, id: FrozenRegionId) -> &FrozenRegion {
        &self.regions[self.region_index(id)]
    }

    pub fn block(&self, id: FrozenBlockId) -> &FrozenBlock {
        &self.blocks[self.block_index(id)]
    }

    pub fn value(&self, id: FrozenValueId) -> &FrozenValue {
        &self.values[self.value_index(id)]
    }

    pub fn roots(&self) -> &[FrozenOperationId] {
        &self.roots
    }

    pub fn uses(&self) -> &Uses {
        &self.uses
    }

    pub fn operations(&self) -> impl Iterator<Item = FrozenOperationId> + '_ {
        (0..self.operations.len()).map(|index| FrozenOperationId::new(self.identity, index))
    }

    pub fn regions(&self) -> impl Iterator<Item = FrozenRegionId> + '_ {
        (0..self.regions.len()).map(|index| FrozenRegionId::new(self.identity, index))
    }

    pub fn blocks(&self) -> impl Iterator<Item = FrozenBlockId> + '_ {
        (0..self.blocks.len()).map(|index| FrozenBlockId::new(self.identity, index))
    }

    pub fn values(&self) -> impl Iterator<Item = FrozenValueId> + '_ {
        (0..self.values.len()).map(|index| FrozenValueId::new(self.identity, index))
    }

    fn operation_index(&self, id: FrozenOperationId) -> usize {
        assert_eq!(
            id.arena, self.identity,
            "operation belongs to another frozen IR"
        );
        id.index()
    }

    fn region_index(&self, id: FrozenRegionId) -> usize {
        assert_eq!(
            id.arena, self.identity,
            "region belongs to another frozen IR"
        );
        id.index()
    }

    fn block_index(&self, id: FrozenBlockId) -> usize {
        assert_eq!(
            id.arena, self.identity,
            "block belongs to another frozen IR"
        );
        id.index()
    }

    fn value_index(&self, id: FrozenValueId) -> usize {
        assert_eq!(
            id.arena, self.identity,
            "value belongs to another frozen IR"
        );
        id.index()
    }
}

impl Operations for FrozenIr {
    fn operation_count(&self) -> usize {
        self.operations.len()
    }

    fn value_count(&self) -> usize {
        self.values.len()
    }

    fn is_live(&self, operation: usize) -> bool {
        operation < self.operations.len()
    }

    fn each_operand(&self, operation: usize, mut read: impl FnMut(usize)) {
        let operation = &self.operations[operation];
        for &value in &operation.operands {
            read(value.index());
        }
        for successor in &operation.successors {
            for &value in &successor.arguments {
                read(value.index());
            }
        }
    }

    fn each_result(&self, operation: usize, mut defined: impl FnMut(usize)) {
        for &value in &self.operations[operation].results {
            defined(value.index());
        }
    }
}

/// Maps handles of one consumed editable IR to its compact IDs.
pub struct FreezeMap {
    arenas: SourceArenas,
    operations: Vec<Option<FrozenOperationId>>,
    regions: Vec<Option<FrozenRegionId>>,
    blocks: Vec<Option<FrozenBlockId>>,
    values: Vec<Option<FrozenValueId>>,
}

pub(crate) struct SourceArenas {
    pub operation: u32,
    pub region: u32,
    pub block: u32,
    pub value: u32,
}

impl FreezeMap {
    pub fn operation(&self, id: Id<Operation>) -> Option<FrozenOperationId> {
        (id.arena() == self.arenas.operation)
            .then(|| self.operations.get(id.slot() as usize).copied().flatten())
            .flatten()
    }

    pub fn region(&self, id: Id<Region>) -> Option<FrozenRegionId> {
        (id.arena() == self.arenas.region)
            .then(|| self.regions.get(id.slot() as usize).copied().flatten())
            .flatten()
    }

    pub fn block(&self, id: Id<Block>) -> Option<FrozenBlockId> {
        (id.arena() == self.arenas.block)
            .then(|| self.blocks.get(id.slot() as usize).copied().flatten())
            .flatten()
    }

    pub fn value(&self, id: Id<Value>) -> Option<FrozenValueId> {
        (id.arena() == self.arenas.value)
            .then(|| self.values.get(id.slot() as usize).copied().flatten())
            .flatten()
    }

    pub(crate) fn new(
        arenas: SourceArenas,
        operations: Vec<Option<FrozenOperationId>>,
        regions: Vec<Option<FrozenRegionId>>,
        blocks: Vec<Option<FrozenBlockId>>,
        values: Vec<Option<FrozenValueId>>,
    ) -> Self {
        Self {
            arenas,
            operations,
            regions,
            blocks,
            values,
        }
    }
}
