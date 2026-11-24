// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::SmallVec;

use crate::analysis;
use crate::ir::{Def, Func, Node, ValueDef, ValueId};
use crate::op::Op;
use crate::ops::Literal;
use crate::shape::Shape;

pub(super) struct Rebuilder {
    pub(super) func: Func,
    value_map: Vec<Option<ValueId>>,
}

/// How much of a function had been rebuilt at one moment. Speculative work —
/// unrolling a loop whose predicate turns out not to be decidable — is emitted
/// with [`Rebuilder::emit_fresh_results`], which touches nothing but the node
/// and value tables, so restoring their two lengths restores the rebuilder
/// exactly.
#[derive(Clone, Copy, Debug)]
pub(super) struct Mark {
    nodes: usize,
    values: usize,
}

impl Rebuilder {
    pub(super) fn new(old: &Func) -> Self {
        let mut func = Func {
            name: old.name.clone(),
            params: Vec::with_capacity(old.params.len()),
            nodes: Vec::with_capacity(old.nodes.len()),
            results: Vec::with_capacity(old.results.len()),
            values: Vec::with_capacity(old.values.len()),
        };
        let mut value_map = vec![None; old.values.len()];
        for (index, &param) in old.params.iter().enumerate() {
            let value = ValueId(func.values.len() as u32);
            func.values.push(ValueDef {
                shape: old.value_shape(param).clone(),
                def: Def::Param(index),
            });
            func.params.push(value);
            value_map[param.index()] = Some(value);
        }
        Self { func, value_map }
    }

    pub(super) fn mapped(&self, value: ValueId) -> Result<ValueId, String> {
        self.value_map
            .get(value.index())
            .and_then(|value| *value)
            .ok_or_else(|| format!("value %{} has not been rebuilt", value.index()))
    }

    pub(super) fn mapped_inputs(&self, node: &Node) -> Result<SmallVec<[ValueId; 4]>, String> {
        node.inputs
            .iter()
            .map(|&value| self.mapped(value))
            .collect()
    }

    pub(super) fn alias(
        &mut self,
        old_results: &[ValueId],
        new_results: &[ValueId],
    ) -> Result<(), String> {
        if old_results.len() != new_results.len() {
            return Err("cannot alias node results with different arity".into());
        }
        for (&old, &new) in old_results.iter().zip(new_results) {
            self.value_map[old.index()] = Some(new);
        }
        Ok(())
    }

    pub(super) fn emit(
        &mut self,
        old: &Func,
        old_results: &[ValueId],
        op: std::sync::Arc<dyn Op>,
        inputs: SmallVec<[ValueId; 4]>,
    ) -> Result<SmallVec<[ValueId; 1]>, String> {
        let node_index = self.func.nodes.len();
        let results = old_results
            .iter()
            .enumerate()
            .map(|(slot, &old_result)| {
                let result = ValueId(self.func.values.len() as u32);
                self.func.values.push(ValueDef {
                    shape: old.value_shape(old_result).clone(),
                    def: Def::NodeResult {
                        node: node_index,
                        slot,
                    },
                });
                self.value_map[old_result.index()] = Some(result);
                result
            })
            .collect::<SmallVec<_>>();
        self.func.nodes.push(Node {
            op,
            inputs,
            results: results.clone(),
        });
        Ok(results)
    }

    pub(super) fn emit_fresh(
        &mut self,
        shape: Shape,
        op: impl Op + 'static,
        inputs: impl IntoIterator<Item = ValueId>,
    ) -> ValueId {
        self.emit_fresh_op(shape, std::sync::Arc::new(op), inputs)
    }

    pub(super) fn emit_fresh_op(
        &mut self,
        shape: Shape,
        op: std::sync::Arc<dyn Op>,
        inputs: impl IntoIterator<Item = ValueId>,
    ) -> ValueId {
        self.emit_fresh_results(&[shape], op, inputs)[0]
    }

    /// Emit a node that no value of the old function stands for, with one result
    /// per shape given.
    pub(super) fn emit_fresh_results(
        &mut self,
        shapes: &[Shape],
        op: std::sync::Arc<dyn Op>,
        inputs: impl IntoIterator<Item = ValueId>,
    ) -> SmallVec<[ValueId; 1]> {
        let node = self.func.nodes.len();
        let results = shapes
            .iter()
            .enumerate()
            .map(|(slot, shape)| {
                let result = ValueId(self.func.values.len() as u32);
                self.func.values.push(ValueDef {
                    shape: shape.clone(),
                    def: Def::NodeResult { node, slot },
                });
                result
            })
            .collect::<SmallVec<[ValueId; 1]>>();
        self.func.nodes.push(Node {
            op,
            inputs: inputs.into_iter().collect(),
            results: results.clone(),
        });
        results
    }

    pub(super) fn mark(&self) -> Mark {
        Mark {
            nodes: self.func.nodes.len(),
            values: self.func.values.len(),
        }
    }

    /// Nodes emitted since `mark`.
    pub(super) fn nodes_since(&self, mark: Mark) -> usize {
        self.func.nodes.len() - mark.nodes
    }

    /// Discard everything emitted since `mark`.
    pub(super) fn rewind(&mut self, mark: Mark) {
        self.func.nodes.truncate(mark.nodes);
        self.func.values.truncate(mark.values);
    }

    pub(super) fn producer<T: Clone + 'static>(&self, value: ValueId) -> Option<(T, ValueId)> {
        analysis::unary_producer(&self.func, value)
    }

    pub(super) fn literal(&self, value: ValueId) -> Option<Literal> {
        analysis::constant_literal(&self.func, value).cloned()
    }

    pub(super) fn finish(mut self, old: &Func) -> Result<Func, String> {
        self.func.results = old
            .results
            .iter()
            .map(|&value| self.mapped(value))
            .collect::<Result<_, _>>()?;
        Ok(self.func)
    }
}
