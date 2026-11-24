// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SSA data model: a [`Module`] of [`Func`]s, each a topologically ordered
//! list of [`Node`]s over a per-function value table. Values and functions are
//! referenced by index newtypes ([`ValueId`], [`FuncId`]) into `Vec` arenas —
//! the idiom used throughout the workspace (`recipe`, `xir`).

use smallvec::SmallVec;

use crate::op::Op;
use crate::shape::Shape;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

impl FuncId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// How a value comes to exist: a function parameter, or the result of a node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Def {
    Param(usize),
    NodeResult { node: usize, slot: usize },
}

#[derive(Clone, Debug)]
pub struct ValueDef {
    pub shape: Shape,
    pub def: Def,
}

/// One operation: `results = op(inputs...)`. Multi-result is native (e.g.
/// `sort`, `while`), so there is no tuple / get-tuple-element clutter.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: std::sync::Arc<dyn Op>,
    pub inputs: SmallVec<[ValueId; 4]>,
    pub results: SmallVec<[ValueId; 1]>,
}

/// A function: a single-entry dataflow graph. `nodes` are in topological order
/// (a node's operands are defined by earlier params/nodes). `values` is the
/// value table giving every `ValueId` its shape and definition.
#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub params: Vec<ValueId>,
    pub nodes: Vec<Node>,
    pub results: Vec<ValueId>,
    pub values: Vec<ValueDef>,
}

impl Func {
    pub fn value_shape(&self, v: ValueId) -> &Shape {
        &self.values[v.index()].shape
    }

    pub fn num_values(&self) -> usize {
        self.values.len()
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub finite_map_graph: domain_index::FiniteMapGraph,
    pub funcs: Vec<Func>,
    pub entry: FuncId,
}

impl Module {
    pub fn func(&self, f: FuncId) -> &Func {
        &self.funcs[f.index()]
    }

    pub fn entry_func(&self) -> &Func {
        &self.funcs[self.entry.index()]
    }
}

impl weave::Operations for Func {
    fn operation_count(&self) -> usize {
        self.nodes.len()
    }

    fn value_count(&self) -> usize {
        self.values.len()
    }

    fn is_live(&self, operation: usize) -> bool {
        operation < self.nodes.len()
    }

    fn each_operand(&self, operation: usize, mut read: impl FnMut(usize)) {
        for &value in &self.nodes[operation].inputs {
            read(value.index());
        }
    }

    fn each_result(&self, operation: usize, mut defined: impl FnMut(usize)) {
        for &value in &self.nodes[operation].results {
            defined(value.index());
        }
    }
}
