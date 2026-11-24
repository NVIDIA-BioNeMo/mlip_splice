// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural isomorphism of scalar expression DAGs.
//!
//! Two scalar values compute "the same expression on different inputs" when the
//! trees they unfold to agree at every position except the leaves. That relation
//! is what makes a batch legal: one program over a lane axis evaluates every
//! member, and the members differ only in which leaf each lane reads.
//!
//! Shapes are interned rather than hashed, so equal ids mean equal structure
//! with no collision to rule out. Leaves are one id, which is exactly the
//! abstraction that lets `a*b` and `c*d` share a shape.
//!
//! The unfolding is a *tree*, not a DAG. Two values can share a shape while
//! sharing subexpressions differently — `f(g(x), g(x))` after common
//! subexpression elimination is one node where `f(g(x), g(y))` is two — and only
//! the tree view puts their leaves in correspondence. Replaying a tree can cost
//! more operations than the DAG it came from, so callers gate on [`Shape::size`].

use std::collections::HashMap;

use crate::ir::{Def, Func, Node, ValueId};

pub(super) type ShapeId = usize;

/// The shape every leaf shares. Abstracting leaf identity is what lets values
/// over different inputs land in one batch.
pub(super) const LEAF: ShapeId = 0;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ShapeKey {
    operation: String,
    children: Vec<ShapeId>,
}

/// Interned expression shapes, plus the tree measurements callers gate on.
pub(super) struct Shapes {
    keys: HashMap<ShapeKey, ShapeId>,
    children: Vec<Vec<ShapeId>>,
    /// Internal operations in the tree unfolding: what a batched replay costs.
    size: Vec<u64>,
    /// Leaves in the tree unfolding: how many gathers a batch needs.
    leaves: Vec<u64>,
}

impl Shapes {
    pub(super) fn new() -> Self {
        Self {
            keys: HashMap::new(),
            children: vec![Vec::new()],
            size: vec![0],
            leaves: vec![1],
        }
    }

    pub(super) fn size(&self, shape: ShapeId) -> u64 {
        self.size[shape]
    }

    #[cfg(test)]
    pub(super) fn leaves(&self, shape: ShapeId) -> u64 {
        self.leaves[shape]
    }

    pub(super) fn is_leaf(&self, shape: ShapeId) -> bool {
        shape == LEAF
    }

    /// Interns one operation over already-interned children. `limit` bounds the
    /// tree unfolding; a shape that would exceed it is refused, which keeps a
    /// deeply shared DAG from being replayed as an astronomically large tree.
    fn intern(&mut self, operation: String, children: Vec<ShapeId>, limit: u64) -> Option<ShapeId> {
        let key = ShapeKey {
            operation,
            children,
        };
        if let Some(&shape) = self.keys.get(&key) {
            return Some(shape);
        }
        let size = key
            .children
            .iter()
            .try_fold(1u64, |total, &child| total.checked_add(self.size[child]))?;
        let leaves = key
            .children
            .iter()
            .try_fold(0u64, |total, &child| total.checked_add(self.leaves[child]))?;
        if size > limit || leaves > limit {
            return None;
        }
        let shape = self.children.len();
        self.children.push(key.children.clone());
        self.size.push(size);
        self.leaves.push(leaves);
        self.keys.insert(key, shape);
        Some(shape)
    }
}

/// The key that decides whether two operations may occupy the same position of
/// one shape. Attributes are part of it, so a `Compare` on different directions
/// never shares a batch with one on another.
fn operation_key(func: &Func, value: ValueId) -> Option<String> {
    let node = producer(func, value)?;
    Some(format!("{}{:?}", node.op.mnemonic(), node.op.attrs()))
}

/// Assigns every value of `func` its shape, treating any value `is_leaf` accepts
/// as an abstract leaf. Returns `None` for values whose shape exceeded `limit`
/// or whose region membership is unresolved; those simply never batch.
///
/// Runs in one pass over nodes in definition order, so operands are interned
/// before the operations that read them.
pub(super) fn assign(
    func: &Func,
    limit: u64,
    is_leaf: impl Fn(ValueId) -> bool,
) -> (Shapes, Vec<Option<ShapeId>>) {
    let mut shapes = Shapes::new();
    let mut assigned = vec![None; func.values.len()];
    for value in func.params.iter() {
        assigned[value.index()] = Some(LEAF);
    }
    for node in &func.nodes {
        let Some(&result) = node.results.first() else {
            continue;
        };
        if node.results.len() != 1 {
            continue;
        }
        if is_leaf(result) {
            assigned[result.index()] = Some(LEAF);
            continue;
        }
        let Some(operation) = operation_key(func, result) else {
            continue;
        };
        let children = node
            .inputs
            .iter()
            .map(|input| assigned[input.index()])
            .collect::<Option<Vec<_>>>();
        let Some(children) = children else {
            continue;
        };
        assigned[result.index()] = shapes.intern(operation, children, limit);
    }
    (shapes, assigned)
}

/// Reads off one value's leaves in the order its shape's tree unfolding visits
/// them. Two values of the same shape produce lists of the same length whose
/// entries correspond position by position — that correspondence is what each
/// gather's index vector encodes.
pub(super) fn leaf_order(
    func: &Func,
    value: ValueId,
    shapes: &Shapes,
    assigned: &[Option<ShapeId>],
    leaves: &mut Vec<ValueId>,
) -> Option<()> {
    let shape = assigned[value.index()]?;
    if shapes.is_leaf(shape) {
        leaves.push(value);
        return Some(());
    }
    let node = producer(func, value)?;
    for &input in &node.inputs {
        leaf_order(func, input, shapes, assigned, leaves)?;
    }
    Some(())
}

#[cfg(test)]
mod tests;

pub(super) fn producer(func: &Func, value: ValueId) -> Option<&Node> {
    let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def else {
        return None;
    };
    func.nodes.get(node)
}
