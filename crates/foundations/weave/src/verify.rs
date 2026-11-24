// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::diagnostic::Diagnostic;
use crate::ir::{BlockId, Ir, OperationId, RegionId, ValueDefinition, ValueId};
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};

pub fn verify(ir: &Ir, registry: &Registry) -> Vec<Diagnostic> {
    let mut verifier = Verifier {
        ir,
        registry,
        diagnostics: Vec::new(),
    };
    verifier.verify_ownership();
    verifier.verify_registered_types();
    for root in ir.roots() {
        verifier.verify_operation(*root, None);
    }
    verifier.diagnostics
}

struct Verifier<'a> {
    ir: &'a Ir,
    registry: &'a Registry,
    diagnostics: Vec<Diagnostic>,
}

impl Verifier<'_> {
    fn error(&mut self, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::error(message));
    }

    fn verify_registered_types(&mut self) {
        let unknown: Vec<_> = self
            .ir
            .values()
            .map(|(_, value)| &value.ty)
            .filter(|ty| !self.registry.contains_type(ty.syntax()))
            .map(ToString::to_string)
            .collect();
        for ty in unknown {
            self.error(format!("unknown type `{ty}`"));
        }
    }

    fn verify_ownership(&mut self) {
        let mut operation_owners = HashMap::new();
        for (block_id, block) in self.ir.blocks() {
            for operation in &block.operations {
                if operation_owners.insert(*operation, block_id).is_some() {
                    self.error(format!(
                        "operation {operation:?} has multiple owning blocks"
                    ));
                }
                match self.ir.operation(*operation) {
                    Some(op) if op.parent == Some(block_id) => {}
                    Some(_) => {
                        self.error(format!("operation {operation:?} has inconsistent parent"))
                    }
                    None => self.error(format!("block {block_id:?} contains a missing operation")),
                }
            }
        }
        let mut region_owners = HashMap::new();
        for (operation_id, operation) in self.ir.operations() {
            for region in &operation.regions {
                if region_owners.insert(*region, operation_id).is_some() {
                    self.error(format!("region {region:?} has multiple owners"));
                }
                match self.ir.region(*region) {
                    Some(node) if node.parent == operation_id => {}
                    Some(_) => self.error(format!("region {region:?} has inconsistent parent")),
                    None => self.error(format!(
                        "operation {operation_id:?} contains a missing region"
                    )),
                }
            }
        }
        for root in self.ir.roots() {
            match self.ir.operation(*root) {
                Some(op) if op.parent.is_none() => {}
                Some(_) => self.error(format!("root operation {root:?} has a block parent")),
                None => self.error(format!("missing root operation {root:?}")),
            }
        }
        let roots: HashSet<_> = self.ir.roots().iter().copied().collect();
        for (operation, node) in self.ir.operations() {
            if node.parent.is_some() && !operation_owners.contains_key(&operation) {
                self.error(format!(
                    "operation {operation:?} names an owner that does not contain it"
                ));
            }
            if !self.operation_is_rooted(operation, &roots) {
                self.error(format!("operation {operation:?} is detached"));
            }
        }
    }

    fn operation_is_rooted(
        &self,
        mut operation: OperationId,
        roots: &HashSet<OperationId>,
    ) -> bool {
        let mut visited = HashSet::new();
        loop {
            if !visited.insert(operation) {
                return false;
            }
            let Some(node) = self.ir.operation(operation) else {
                return false;
            };
            let Some(block) = node.parent else {
                return roots.contains(&operation);
            };
            let Some(parent) = self
                .ir
                .block(block)
                .and_then(|block| self.ir.region(block.parent))
                .map(|region| region.parent)
            else {
                return false;
            };
            operation = parent;
        }
    }

    fn verify_operation(&mut self, id: OperationId, expected_parent: Option<BlockId>) {
        let Some(operation) = self.ir.operation(id) else {
            return;
        };
        if operation.parent != expected_parent {
            self.error(format!("operation {id:?} appears under the wrong parent"));
        }
        if let Some(definition) = self.registry.operation(&operation.name) {
            definition.verify(self.ir, id, &mut self.diagnostics);
        } else {
            self.error(format!("unknown operation `{}`", operation.name));
        }
        if !self.registry.contains_dialect(operation.name.dialect()) {
            self.error(format!("unknown dialect `{}`", operation.name.dialect()));
        }
        for value in operation.operands.iter().chain(operation.results.iter()) {
            if self.ir.value(*value).is_none() {
                self.error(format!(
                    "operation {id:?} refers to missing value {value:?}"
                ));
            }
        }
        for (index, result) in operation.results.iter().enumerate() {
            if !matches!(self.ir.value(*result).map(|v| v.definition), Some(ValueDefinition::OperationResult { operation, index: result_index }) if operation == id && result_index == index as u32)
            {
                self.error(format!("result {result:?} has an inconsistent definition"));
            }
        }
        self.verify_extensions(id);
        for region in &operation.regions {
            self.verify_region(*region, id);
        }
    }

    fn verify_extensions(&mut self, id: OperationId) {
        let operation = self.ir.operation(id).expect("operation already checked");
        let unknown: Vec<_> = operation
            .attributes
            .values()
            .filter(|attr| !self.registry.contains_attribute(attr.syntax()))
            .map(ToString::to_string)
            .collect();
        for attribute in unknown {
            self.error(format!("unknown attribute `{attribute}`"));
        }
    }

    fn verify_region(&mut self, id: RegionId, parent: OperationId) {
        let Some(region) = self.ir.region(id) else {
            return;
        };
        if region.parent != parent {
            self.error(format!("region {id:?} appears under the wrong operation"));
        }
        let member_blocks: HashSet<_> = region.blocks.iter().copied().collect();
        let dominators = self.dominators(region.blocks.as_slice(), &member_blocks);
        for block in &region.blocks {
            self.verify_block(*block, id, &member_blocks, &dominators);
        }
    }

    fn verify_block(
        &mut self,
        id: BlockId,
        parent: RegionId,
        member_blocks: &HashSet<BlockId>,
        dominators: &HashMap<BlockId, HashSet<BlockId>>,
    ) {
        let Some(block) = self.ir.block(id) else {
            self.error(format!("missing block {id:?}"));
            return;
        };
        if block.parent != parent {
            self.error(format!("block {id:?} appears under the wrong region"));
        }
        for (index, argument) in block.arguments.iter().enumerate() {
            if !matches!(self.ir.value(*argument).map(|v| v.definition), Some(ValueDefinition::BlockArgument { block, index: argument_index }) if block == id && argument_index == index as u32)
            {
                self.error(format!(
                    "block argument {argument:?} has an inconsistent definition"
                ));
            }
        }
        let positions: HashMap<_, _> = block
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| (*op, i))
            .collect();
        for (position, operation_id) in block.operations.iter().enumerate() {
            self.verify_uses(
                *operation_id,
                id,
                position,
                &positions,
                member_blocks,
                dominators,
            );
            self.verify_operation(*operation_id, Some(id));
        }
        let needs_terminator = block.operations.last().is_none_or(|last| {
            self.ir
                .operation(*last)
                .and_then(|op| self.registry.operation(&op.name))
                .is_none_or(|def| !def.traits.terminator)
        });
        if needs_terminator {
            self.error(format!(
                "block {id:?} is empty or has no registered terminator"
            ));
        }
        for operation_id in &block.operations[..block.operations.len().saturating_sub(1)] {
            if self
                .ir
                .operation(*operation_id)
                .and_then(|op| self.registry.operation(&op.name))
                .is_some_and(|def| def.traits.terminator)
            {
                self.error(format!(
                    "terminator {operation_id:?} is not last in block {id:?}"
                ));
            }
        }
    }

    fn verify_uses(
        &mut self,
        operation_id: OperationId,
        block_id: BlockId,
        position: usize,
        positions: &HashMap<OperationId, usize>,
        member_blocks: &HashSet<BlockId>,
        dominators: &HashMap<BlockId, HashSet<BlockId>>,
    ) {
        let Some(operation) = self.ir.operation(operation_id) else {
            return;
        };
        for operand in operation
            .operands
            .iter()
            .chain(operation.successors.iter().flat_map(|s| &s.arguments))
        {
            if !self.value_dominates(*operand, block_id, position, positions, dominators) {
                self.error(format!(
                    "value {operand:?} does not dominate operation {operation_id:?}"
                ));
            }
        }
        for successor in &operation.successors {
            if !member_blocks.contains(&successor.block) {
                self.error(format!(
                    "successor {:?} leaves its owning region",
                    successor.block
                ));
                continue;
            }
            let Some(target) = self.ir.block(successor.block) else {
                continue;
            };
            if target.arguments.len() != successor.arguments.len() {
                self.error(format!(
                    "successor {:?} expects {} arguments, got {}",
                    successor.block,
                    target.arguments.len(),
                    successor.arguments.len()
                ));
            }
            for (argument, target_argument) in successor.arguments.iter().zip(&target.arguments) {
                if let (Some(actual), Some(expected)) =
                    (self.ir.value(*argument), self.ir.value(*target_argument))
                    && actual.ty != expected.ty
                {
                    self.error(format!(
                        "successor argument type `{}` does not match block argument type `{}`",
                        actual.ty, expected.ty
                    ));
                }
            }
        }
    }

    fn dominators(
        &self,
        blocks: &[BlockId],
        members: &HashSet<BlockId>,
    ) -> HashMap<BlockId, HashSet<BlockId>> {
        let Some(entry) = blocks.first().copied() else {
            return HashMap::new();
        };
        let mut predecessors: HashMap<_, HashSet<_>> = blocks
            .iter()
            .copied()
            .map(|block| (block, HashSet::new()))
            .collect();
        for source in blocks {
            let Some(block) = self.ir.block(*source) else {
                continue;
            };
            for operation in &block.operations {
                let Some(operation) = self.ir.operation(*operation) else {
                    continue;
                };
                for successor in &operation.successors {
                    if members.contains(&successor.block) {
                        predecessors
                            .entry(successor.block)
                            .or_default()
                            .insert(*source);
                    }
                }
            }
        }
        let mut result: HashMap<_, _> = blocks
            .iter()
            .copied()
            .map(|block| {
                let initial = if block == entry {
                    HashSet::from([entry])
                } else {
                    members.clone()
                };
                (block, initial)
            })
            .collect();
        loop {
            let mut changed = false;
            for block in blocks.iter().copied().filter(|block| *block != entry) {
                let predecessors = &predecessors[&block];
                let mut next = if predecessors.is_empty() {
                    HashSet::new()
                } else {
                    let mut iter = predecessors.iter();
                    let mut common = result[iter.next().expect("non-empty predecessors")].clone();
                    for predecessor in iter {
                        common.retain(|candidate| result[predecessor].contains(candidate));
                    }
                    common
                };
                next.insert(block);
                if result[&block] != next {
                    result.insert(block, next);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        result
    }

    fn value_dominates(
        &self,
        value: ValueId,
        use_block: BlockId,
        use_position: usize,
        positions: &HashMap<OperationId, usize>,
        dominators: &HashMap<BlockId, HashSet<BlockId>>,
    ) -> bool {
        let Some(value_data) = self.ir.value(value) else {
            return false;
        };
        let definition_block = match value_data.definition {
            ValueDefinition::BlockArgument { block, .. } => block,
            ValueDefinition::OperationResult { operation, .. } => {
                let Some(block) = self.ir.operation(operation).and_then(|op| op.parent) else {
                    return false;
                };
                block
            }
        };
        let Some(definition_region) = self.ir.block(definition_block).map(|block| block.parent)
        else {
            return false;
        };
        let Some(use_region) = self.ir.block(use_block).map(|block| block.parent) else {
            return false;
        };
        if definition_region == use_region {
            if definition_block == use_block {
                return match value_data.definition {
                    ValueDefinition::BlockArgument { .. } => true,
                    ValueDefinition::OperationResult { operation, .. } => positions
                        .get(&operation)
                        .is_some_and(|position| *position < use_position),
                };
            }
            return dominators
                .get(&use_block)
                .is_some_and(|blocks| blocks.contains(&definition_block));
        }
        let Some(parent_operation) = self.ir.region(use_region).map(|region| region.parent) else {
            return false;
        };
        let Some(parent_block) = self
            .ir
            .operation(parent_operation)
            .and_then(|operation| operation.parent)
        else {
            return false;
        };
        let Some(parent) = self.ir.block(parent_block) else {
            return false;
        };
        let Some(parent_position) = parent
            .operations
            .iter()
            .position(|operation| *operation == parent_operation)
        else {
            return false;
        };
        let Some(parent_region) = self.ir.region(parent.parent) else {
            return false;
        };
        let parent_members: HashSet<_> = parent_region.blocks.iter().copied().collect();
        let parent_dominators = self.dominators(&parent_region.blocks, &parent_members);
        let parent_positions: HashMap<_, _> = parent
            .operations
            .iter()
            .enumerate()
            .map(|(position, operation)| (*operation, position))
            .collect();
        self.value_dominates(
            value,
            parent_block,
            parent_position,
            &parent_positions,
            &parent_dominators,
        )
    }
}
