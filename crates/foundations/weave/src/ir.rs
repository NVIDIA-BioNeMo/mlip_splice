// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::arena::{Arena, Id};
use crate::frozen::{
    FreezeMap, FrozenBlock, FrozenBlockId, FrozenIr, FrozenOperation, FrozenOperationId,
    FrozenRegion, FrozenRegionId, FrozenSuccessor, FrozenValue, FrozenValueDefinition,
    FrozenValueId, SourceArenas,
};
use crate::name::OperationName;
use crate::types::{Attribute, Type};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub type OperationId = Id<Operation>;
pub type RegionId = Id<Region>;
pub type BlockId = Id<Block>;
pub type ValueId = Id<Value>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ValueDefinition {
    OperationResult { operation: OperationId, index: u32 },
    BlockArgument { block: BlockId, index: u32 },
}

#[derive(Clone, Debug)]
pub struct Value {
    pub ty: Type,
    pub definition: ValueDefinition,
}

#[derive(Clone, Debug)]
pub struct Successor {
    pub block: BlockId,
    pub arguments: Vec<ValueId>,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub name: OperationName,
    pub parent: Option<BlockId>,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub successors: Vec<Successor>,
    pub regions: Vec<RegionId>,
    pub attributes: BTreeMap<String, Attribute>,
}

#[derive(Clone, Debug)]
pub struct Region {
    pub parent: OperationId,
    pub blocks: Vec<BlockId>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub parent: RegionId,
    pub arguments: Vec<ValueId>,
    pub operations: Vec<OperationId>,
}

#[derive(Default)]
pub struct Ir {
    operations: Arena<Operation>,
    regions: Arena<Region>,
    blocks: Arena<Block>,
    values: Arena<Value>,
    roots: Vec<OperationId>,
    uses: OnceLock<crate::Uses>,
}

pub type EditableIr = Ir;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IrError {
    MissingOperation,
    MissingRegion,
    MissingBlock,
    MissingValue,
    TypeMismatch,
    InvalidPosition,
    OperationAlreadyAttached,
    OperationNotAttached,
    OperationHasRegions,
    OperationHasUses,
    DetachedOperation,
    OwnershipCycle,
    RootHasParent,
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for IrError {}

impl Ir {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_operation(
        &mut self,
        name: OperationName,
        operands: impl IntoIterator<Item = ValueId>,
        result_types: impl IntoIterator<Item = Type>,
    ) -> Result<OperationId, IrError> {
        let operands: Vec<_> = operands.into_iter().collect();
        if operands.iter().any(|&value| !self.values.contains(value)) {
            return Err(IrError::MissingValue);
        }
        let result_types: Vec<_> = result_types.into_iter().collect();
        self.uses.take();
        let operation = self.operations.insert(Operation {
            name,
            parent: None,
            operands,
            results: Vec::new(),
            successors: Vec::new(),
            regions: Vec::new(),
            attributes: BTreeMap::new(),
        });
        let results = result_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                self.values.insert(Value {
                    ty,
                    definition: ValueDefinition::OperationResult {
                        operation,
                        index: index as u32,
                    },
                })
            })
            .collect();
        self.operations
            .get_mut(operation)
            .expect("new operation exists")
            .results = results;
        Ok(operation)
    }

    pub fn add_root(&mut self, operation: OperationId) -> Result<(), IrError> {
        let op = self
            .operations
            .get(operation)
            .ok_or(IrError::MissingOperation)?;
        if op.parent.is_some() {
            return Err(IrError::RootHasParent);
        }
        if !self.roots.contains(&operation) {
            self.roots.push(operation);
        }
        Ok(())
    }

    pub fn append_region(&mut self, operation: OperationId) -> Result<RegionId, IrError> {
        if !self.operations.contains(operation) {
            return Err(IrError::MissingOperation);
        }
        let region = self.regions.insert(Region {
            parent: operation,
            blocks: Vec::new(),
        });
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .regions
            .push(region);
        Ok(region)
    }

    pub fn append_block(
        &mut self,
        region: RegionId,
        argument_types: impl IntoIterator<Item = Type>,
    ) -> Result<BlockId, IrError> {
        if !self.regions.contains(region) {
            return Err(IrError::MissingRegion);
        }
        let argument_types: Vec<_> = argument_types.into_iter().collect();
        let block = self.blocks.insert(Block {
            parent: region,
            arguments: Vec::new(),
            operations: Vec::new(),
        });
        let arguments = argument_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                self.values.insert(Value {
                    ty,
                    definition: ValueDefinition::BlockArgument {
                        block,
                        index: index as u32,
                    },
                })
            })
            .collect();
        self.blocks
            .get_mut(block)
            .expect("new block exists")
            .arguments = arguments;
        self.regions
            .get_mut(region)
            .expect("checked region")
            .blocks
            .push(block);
        Ok(block)
    }

    pub fn append_operation(
        &mut self,
        block: BlockId,
        operation: OperationId,
    ) -> Result<(), IrError> {
        let position = self
            .blocks
            .get(block)
            .ok_or(IrError::MissingBlock)?
            .operations
            .len();
        self.insert_operation(block, position, operation)
    }

    pub fn set_attribute(
        &mut self,
        operation: OperationId,
        name: impl Into<String>,
        value: Attribute,
    ) -> Result<Option<Attribute>, IrError> {
        Ok(self
            .operations
            .get_mut(operation)
            .ok_or(IrError::MissingOperation)?
            .attributes
            .insert(name.into(), value))
    }

    pub fn set_operands(
        &mut self,
        operation: OperationId,
        operands: impl IntoIterator<Item = ValueId>,
    ) -> Result<(), IrError> {
        let operands: Vec<_> = operands.into_iter().collect();
        if operands.iter().any(|&value| !self.values.contains(value)) {
            return Err(IrError::MissingValue);
        }
        if !self.operations.contains(operation) {
            return Err(IrError::MissingOperation);
        }
        self.uses.take();
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .operands = operands;
        Ok(())
    }

    pub fn set_operand(
        &mut self,
        operation: OperationId,
        index: usize,
        value: ValueId,
    ) -> Result<ValueId, IrError> {
        if !self.values.contains(value) {
            return Err(IrError::MissingValue);
        }
        let operands = &mut self
            .operations
            .get_mut(operation)
            .ok_or(IrError::MissingOperation)?
            .operands;
        let Some(slot) = operands.get_mut(index) else {
            return Err(IrError::InvalidPosition);
        };
        self.uses.take();
        Ok(std::mem::replace(slot, value))
    }

    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) -> Result<usize, IrError> {
        let old_type = &self.values.get(old).ok_or(IrError::MissingValue)?.ty;
        let new_type = &self.values.get(new).ok_or(IrError::MissingValue)?.ty;
        if old_type != new_type {
            return Err(IrError::TypeMismatch);
        }
        let mut readers = self.uses().of_value(old.slot() as usize).to_vec();
        readers.dedup();
        let mut replaced = 0;
        for reader in readers {
            let id = self
                .operations
                .id_at(reader as usize)
                .expect("the use index contains live operations");
            let operation = self.operations.get_mut(id).expect("live operation");
            for operand in &mut operation.operands {
                if *operand == old {
                    *operand = new;
                    replaced += 1;
                }
            }
            for successor in &mut operation.successors {
                for argument in &mut successor.arguments {
                    if *argument == old {
                        *argument = new;
                        replaced += 1;
                    }
                }
            }
        }
        self.uses.take();
        Ok(replaced)
    }

    pub fn add_successor(
        &mut self,
        operation: OperationId,
        block: BlockId,
        arguments: impl IntoIterator<Item = ValueId>,
    ) -> Result<(), IrError> {
        if !self.blocks.contains(block) {
            return Err(IrError::MissingBlock);
        }
        let arguments: Vec<_> = arguments.into_iter().collect();
        if arguments.iter().any(|&value| !self.values.contains(value)) {
            return Err(IrError::MissingValue);
        }
        if !self.operations.contains(operation) {
            return Err(IrError::MissingOperation);
        }
        self.uses.take();
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .successors
            .push(Successor { block, arguments });
        Ok(())
    }

    pub fn insert_operation(
        &mut self,
        block: BlockId,
        position: usize,
        operation: OperationId,
    ) -> Result<(), IrError> {
        let held = self.blocks.get(block).ok_or(IrError::MissingBlock)?;
        if position > held.operations.len() {
            return Err(IrError::InvalidPosition);
        }
        if self.block_is_nested_in_operation(block, operation) {
            return Err(IrError::OwnershipCycle);
        }
        let node = self
            .operations
            .get(operation)
            .ok_or(IrError::MissingOperation)?;
        if node.parent.is_some() || self.roots.contains(&operation) {
            return Err(IrError::OperationAlreadyAttached);
        }
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .parent = Some(block);
        self.blocks
            .get_mut(block)
            .expect("checked block")
            .operations
            .insert(position, operation);
        Ok(())
    }

    pub fn detach_operation(&mut self, operation: OperationId) -> Result<(), IrError> {
        let block = self
            .operations
            .get(operation)
            .ok_or(IrError::MissingOperation)?
            .parent
            .ok_or(IrError::OperationNotAttached)?;
        let operations = &mut self
            .blocks
            .get_mut(block)
            .expect("recorded parent")
            .operations;
        let position = operations
            .iter()
            .position(|&held| held == operation)
            .expect("parent contains its operation");
        operations.remove(position);
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .parent = None;
        Ok(())
    }

    pub fn move_operation(
        &mut self,
        operation: OperationId,
        block: BlockId,
        mut position: usize,
    ) -> Result<(), IrError> {
        let old_block = self
            .operations
            .get(operation)
            .ok_or(IrError::MissingOperation)?
            .parent
            .ok_or(IrError::OperationNotAttached)?;
        let target = self.blocks.get(block).ok_or(IrError::MissingBlock)?;
        if position > target.operations.len() {
            return Err(IrError::InvalidPosition);
        }
        if self.block_is_nested_in_operation(block, operation) {
            return Err(IrError::OwnershipCycle);
        }
        let old_position = self
            .blocks
            .get(old_block)
            .expect("recorded parent")
            .operations
            .iter()
            .position(|&held| held == operation)
            .expect("parent contains its operation");
        if old_block == block && old_position < position {
            position -= 1;
        }
        self.blocks
            .get_mut(old_block)
            .expect("recorded parent")
            .operations
            .remove(old_position);
        self.blocks
            .get_mut(block)
            .expect("checked block")
            .operations
            .insert(position, operation);
        self.operations
            .get_mut(operation)
            .expect("checked operation")
            .parent = Some(block);
        Ok(())
    }

    pub fn erase_operation(&mut self, operation: OperationId) -> Result<(), IrError> {
        let node = self
            .operations
            .get(operation)
            .ok_or(IrError::MissingOperation)?;
        if !node.regions.is_empty() {
            return Err(IrError::OperationHasRegions);
        }
        let results = node.results.clone();
        let has_uses = results.iter().any(|result| {
            self.uses()
                .of_value(result.slot() as usize)
                .iter()
                .any(|&reader| reader != operation.slot())
        });
        if has_uses {
            return Err(IrError::OperationHasUses);
        }
        self.uses.take();
        if self
            .operations
            .get(operation)
            .expect("checked operation")
            .parent
            .is_some()
        {
            self.detach_operation(operation)?;
        }
        self.roots.retain(|&root| root != operation);
        for result in results {
            self.values
                .remove(result)
                .expect("operation owns its results");
        }
        self.operations
            .remove(operation)
            .expect("checked operation exists");
        Ok(())
    }

    /// Which operations read each value, built once and discarded by edits
    /// that change a use-def edge.
    pub fn uses(&self) -> &crate::Uses {
        self.uses.get_or_init(|| crate::Uses::of_arena(self))
    }

    /// Consumes the editable IR and closes every deleted slot.
    pub fn freeze(self) -> Result<FrozenIr, IrError> {
        self.freeze_with_map().map(|(ir, _)| ir)
    }

    /// Consumes the editable IR and also returns translations for surviving handles.
    pub fn freeze_with_map(self) -> Result<(FrozenIr, FreezeMap), IrError> {
        let mut rooted = std::collections::HashSet::new();
        let mut pending = self.roots.clone();
        while let Some(operation) = pending.pop() {
            if !rooted.insert(operation) {
                continue;
            }
            let operation = self
                .operations
                .get(operation)
                .ok_or(IrError::MissingOperation)?;
            for &region in &operation.regions {
                let region = self.regions.get(region).ok_or(IrError::MissingRegion)?;
                for &block in &region.blocks {
                    let block = self.blocks.get(block).ok_or(IrError::MissingBlock)?;
                    pending.extend(block.operations.iter().copied());
                }
            }
        }
        if rooted.len() != self.operations.len() {
            return Err(IrError::DetachedOperation);
        }

        let frozen_identity = crate::frozen::FrozenArenaId::mint();
        let operation_arena = self.operations.identity();
        let region_arena = self.regions.identity();
        let block_arena = self.blocks.identity();
        let value_arena = self.values.identity();

        let mut operation_map = vec![None; self.operations.slot_count()];
        for (dense, (id, _)) in self.operations.iter().enumerate() {
            operation_map[id.slot() as usize] =
                Some(FrozenOperationId::new(frozen_identity, dense));
        }
        let mut region_map = vec![None; self.regions.slot_count()];
        for (dense, (id, _)) in self.regions.iter().enumerate() {
            region_map[id.slot() as usize] = Some(FrozenRegionId::new(frozen_identity, dense));
        }
        let mut block_map = vec![None; self.blocks.slot_count()];
        for (dense, (id, _)) in self.blocks.iter().enumerate() {
            block_map[id.slot() as usize] = Some(FrozenBlockId::new(frozen_identity, dense));
        }
        let mut value_map = vec![None; self.values.slot_count()];
        for (dense, (id, _)) in self.values.iter().enumerate() {
            value_map[id.slot() as usize] = Some(FrozenValueId::new(frozen_identity, dense));
        }

        let map = FreezeMap::new(
            SourceArenas {
                operation: operation_arena,
                region: region_arena,
                block: block_arena,
                value: value_arena,
            },
            operation_map,
            region_map,
            block_map,
            value_map,
        );
        let map_operation = |id| map.operation(id).ok_or(IrError::MissingOperation);
        let map_region = |id| map.region(id).ok_or(IrError::MissingRegion);
        let map_block = |id| map.block(id).ok_or(IrError::MissingBlock);
        let map_value = |id| map.value(id).ok_or(IrError::MissingValue);

        let operations = self
            .operations
            .into_values()
            .map(|(_, operation)| {
                Ok(FrozenOperation {
                    name: operation.name,
                    parent: operation.parent.map(map_block).transpose()?,
                    operands: operation
                        .operands
                        .into_iter()
                        .map(map_value)
                        .collect::<Result<_, _>>()?,
                    results: operation
                        .results
                        .into_iter()
                        .map(map_value)
                        .collect::<Result<_, _>>()?,
                    successors: operation
                        .successors
                        .into_iter()
                        .map(|successor| {
                            Ok(FrozenSuccessor {
                                block: map_block(successor.block)?,
                                arguments: successor
                                    .arguments
                                    .into_iter()
                                    .map(map_value)
                                    .collect::<Result<_, _>>()?,
                            })
                        })
                        .collect::<Result<_, IrError>>()?,
                    regions: operation
                        .regions
                        .into_iter()
                        .map(map_region)
                        .collect::<Result<_, _>>()?,
                    attributes: operation.attributes,
                })
            })
            .collect::<Result<Vec<_>, IrError>>()?;
        let regions = self
            .regions
            .into_values()
            .map(|(_, region)| {
                Ok(FrozenRegion {
                    parent: map_operation(region.parent)?,
                    blocks: region
                        .blocks
                        .into_iter()
                        .map(map_block)
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, IrError>>()?;
        let blocks = self
            .blocks
            .into_values()
            .map(|(_, block)| {
                Ok(FrozenBlock {
                    parent: map_region(block.parent)?,
                    arguments: block
                        .arguments
                        .into_iter()
                        .map(map_value)
                        .collect::<Result<_, _>>()?,
                    operations: block
                        .operations
                        .into_iter()
                        .map(map_operation)
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, IrError>>()?;
        let values = self
            .values
            .into_values()
            .map(|(_, value)| {
                Ok(FrozenValue {
                    ty: value.ty,
                    definition: match value.definition {
                        ValueDefinition::OperationResult { operation, index } => {
                            FrozenValueDefinition::OperationResult {
                                operation: map_operation(operation)?,
                                index,
                            }
                        }
                        ValueDefinition::BlockArgument { block, index } => {
                            FrozenValueDefinition::BlockArgument {
                                block: map_block(block)?,
                                index,
                            }
                        }
                    },
                })
            })
            .collect::<Result<Vec<_>, IrError>>()?;
        let roots = self
            .roots
            .into_iter()
            .map(map_operation)
            .collect::<Result<_, _>>()?;

        Ok((
            FrozenIr::new(frozen_identity, operations, regions, blocks, values, roots),
            map,
        ))
    }

    fn block_is_nested_in_operation(&self, mut block: BlockId, ancestor: OperationId) -> bool {
        loop {
            let Some(region) = self.block(block).and_then(|node| self.region(node.parent)) else {
                return false;
            };
            if region.parent == ancestor {
                return true;
            }
            let Some(parent_block) = self.operation(region.parent).and_then(|op| op.parent) else {
                return false;
            };
            block = parent_block;
        }
    }

    pub fn roots(&self) -> &[OperationId] {
        &self.roots
    }
    pub fn operation(&self, id: OperationId) -> Option<&Operation> {
        self.operations.get(id)
    }
    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(id)
    }
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id)
    }
    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id)
    }
    pub fn operations(&self) -> impl Iterator<Item = (OperationId, &Operation)> {
        self.operations.iter()
    }
    pub fn regions(&self) -> impl Iterator<Item = (RegionId, &Region)> {
        self.regions.iter()
    }
    pub fn blocks(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        self.blocks.iter()
    }
    pub fn values(&self) -> impl Iterator<Item = (ValueId, &Value)> {
        self.values.iter()
    }
}

impl crate::Operations for Ir {
    fn operation_count(&self) -> usize {
        self.operations.slot_count()
    }

    fn value_count(&self) -> usize {
        self.values.slot_count()
    }

    fn is_live(&self, operation: usize) -> bool {
        self.operations.id_at(operation).is_some()
    }

    fn each_operand(&self, operation: usize, mut read: impl FnMut(usize)) {
        let Some(id) = self.operations.id_at(operation) else {
            return;
        };
        let operation = self.operations.get(id).expect("live operation");
        for value in &operation.operands {
            read(value.slot() as usize);
        }
        for successor in &operation.successors {
            for value in &successor.arguments {
                read(value.slot() as usize);
            }
        }
    }

    fn each_result(&self, operation: usize, mut defined: impl FnMut(usize)) {
        let Some(id) = self.operations.id_at(operation) else {
            return;
        };
        for value in &self.operations.get(id).expect("live operation").results {
            defined(value.slot() as usize);
        }
    }
}
