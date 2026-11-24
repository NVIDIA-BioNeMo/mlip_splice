// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic admission of typed constructor DAGs, with distinct work and residency units.

use super::*;
use crate::op::{InferCtx, Op};
use crate::ops::Select;
/// Traversal is bounded by the materialized proof input, not by remaining proof
/// search fuel. A completed proof must survive unrelated later search exhaustion.
struct TraversalBudget {
    remaining: usize,
}

impl TraversalBudget {
    fn reserve(&mut self, work: usize) -> bool {
        if let Some(remaining) = self.remaining.checked_sub(work) {
            self.remaining = remaining;
            true
        } else {
            false
        }
    }
}

fn form_work(form: &Form) -> usize {
    let mut pending = vec![form];
    let mut work = 0usize;
    while let Some(form) = pending.pop() {
        work = work.saturating_add(1);
        match form {
            Form::Class(_) => {}
            Form::Coefficient(_, values) => work = work.saturating_add(values.len()),
            Form::Unary(_, _, x) | Form::Map { value: x, .. } => pending.push(x),
            Form::Binary(_, _, x, y) => pending.extend([x.as_ref(), y.as_ref()]),
            Form::Select(_, p, x, y) => pending.extend([p.as_ref(), x.as_ref(), y.as_ref()]),
        }
    }
    work
}

/// Children name snapshot classes, then topologically staged constructor keys.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    ty: usize,
    node: ExpressionNode,
}

pub(super) struct Admission<'a> {
    snapshot: &'a Snapshot,
    shapes: &'a [Shape],
    maps: &'a domain_index::FiniteMapGraph,
    keys: HashMap<Key, ExpressionId>,
    nodes: Vec<Key>,
    roots: Vec<(u32, ExpressionId)>,
    capacity: usize,
}

#[expect(
    clippy::mutable_key_type,
    reason = "AlgebraicReal caches preserve exact equality and hash"
)]
impl<'a> Admission<'a> {
    pub(super) fn new(
        snapshot: &'a Snapshot,
        shapes: &'a [Shape],
        coefficients: &[ExprTree],
        maps: &'a domain_index::FiniteMapGraph,
        capacity: usize,
    ) -> Self {
        let mut keys = HashMap::new();
        for (class, nodes) in snapshot.classes.iter().enumerate() {
            for &node in nodes {
                let child = |class: u32| ExpressionId(class as usize);
                let node = match node {
                    ENode::Value(value) => ExpressionNode::Value(value),
                    ENode::Coefficient(id) => {
                        let ExprTree::Coefficient { shape, values } = &coefficients[id] else {
                            unreachable!()
                        };
                        ExpressionNode::Coefficient {
                            shape: shape.clone(),
                            values: values.clone(),
                        }
                    }
                    ENode::Negate { value, .. } => ExpressionNode::Negate(child(value)),
                    ENode::Add { left, right, .. } => {
                        ExpressionNode::Add(child(left), child(right))
                    }
                    ENode::Multiply { left, right, .. } => {
                        ExpressionNode::Multiply(child(left), child(right))
                    }
                    ENode::Select {
                        predicate,
                        on_true,
                        on_false,
                        ..
                    } => ExpressionNode::Select(child(predicate), child(on_true), child(on_false)),
                    ENode::Gather { map, axis, value } => ExpressionNode::Gather {
                        map,
                        axis,
                        value: child(value),
                    },
                    ENode::Sum { map, axis, value } => ExpressionNode::Sum {
                        map,
                        axis,
                        value: child(value),
                    },
                };
                keys.insert(
                    Key {
                        ty: snapshot.types[class],
                        node,
                    },
                    ExpressionId(class),
                );
            }
        }
        Self {
            snapshot,
            shapes,
            maps,
            keys,
            nodes: Vec::new(),
            roots: Vec::new(),
            capacity,
        }
    }

    pub(super) fn plan(&mut self, proposal: Proposal) -> Result<bool, String> {
        let mut traversal = TraversalBudget {
            remaining: form_work(&proposal.form),
        };
        self.plan_with_work(proposal, &mut traversal)
    }

    /// Failure spends traversal work, but changes no constructor, coefficient, or DAG pool.
    fn plan_with_work(
        &mut self,
        proposal: Proposal,
        budget: &mut TraversalBudget,
    ) -> Result<bool, String> {
        if proposal.form.ty(self.snapshot) != self.snapshot.types[proposal.class as usize] {
            return Err("rewrite changed its e-class type".into());
        }
        let mut local = HashMap::new();
        let mut nodes = Vec::new();
        let Some(root) = self.visit(&proposal.form, &mut local, &mut nodes, budget)? else {
            return Ok(false);
        };
        self.keys.extend(local);
        self.nodes.extend(nodes);
        self.roots.push((proposal.class, root));
        Ok(true)
    }

    fn visit(
        &self,
        form: &Form,
        local: &mut HashMap<Key, ExpressionId>,
        nodes: &mut Vec<Key>,
        budget: &mut TraversalBudget,
    ) -> Result<Option<ExpressionId>, String> {
        if !budget.reserve(1) {
            return Ok(None);
        }
        let ty = form.ty(self.snapshot);
        let shape = self.shapes.get(ty).ok_or("unknown rewrite type")?;
        macro_rules! child {
            ($x:expr) => {
                match self.visit($x, local, nodes, budget)? {
                    Some(id) => id,
                    None => return Ok(None),
                }
            };
        }
        let same_type = |x: &Form| {
            if x.ty(self.snapshot) == ty {
                Ok(())
            } else {
                Err("typed rewrite operands disagree".to_string())
            }
        };
        let node = match form {
            Form::Class(class) => return Ok(Some(ExpressionId(*class as usize))),
            Form::Coefficient(_, values) => {
                if !budget.reserve(values.len()) {
                    return Ok(None);
                }
                if shape.element_count()? != values.len() {
                    return Err("rewrite coefficient shape disagrees".into());
                }
                ExpressionNode::Coefficient {
                    shape: shape.clone(),
                    values: values.clone().into(),
                }
            }
            Form::Unary("Negate", _, x) => {
                same_type(x)?;
                ExpressionNode::Negate(child!(x))
            }
            Form::Binary(name @ ("Add" | "Multiply"), _, x, y) => {
                same_type(x)?;
                same_type(y)?;
                let x = child!(x);
                let y = child!(y);
                if *name == "Add" {
                    ExpressionNode::Add(x, y)
                } else {
                    ExpressionNode::Multiply(x, y)
                }
            }
            Form::Select(_, p, x, y) => {
                let inputs = [p, x, y].map(|value| {
                    self.shapes
                        .get(value.ty(self.snapshot))
                        .cloned()
                        .ok_or("unknown select operand type")
                });
                let [p_shape, x_shape, y_shape] = inputs;
                let inputs = [p_shape?, x_shape?, y_shape?];
                let outputs = Select
                    .infer(&InferCtx::new(&inputs, &[], self.maps))
                    .map_err(|error| error.to_string())?;
                if outputs.as_slice() != [shape.clone()] {
                    return Err("rewrite select output disagrees".into());
                }
                ExpressionNode::Select(child!(p), child!(x), child!(y))
            }
            Form::Map {
                sum,
                map,
                axis,
                input,
                output: _,
                value,
            } => {
                if value.ty(self.snapshot) != *input {
                    return Err("rewrite map input disagrees".into());
                }
                let declared = self.maps.map(*map)?;
                let (source, target) = if *sum {
                    (declared.source, declared.target)
                } else {
                    (declared.target, declared.source)
                };
                let mut expected = self.shapes[*input].clone();
                if expected.domain_axes.get(*axis as usize) != Some(&source) {
                    return Err("rewrite map endpoint disagrees".into());
                }
                expected.domain_axes[*axis as usize] = target;
                if &expected != shape {
                    return Err("rewrite map output disagrees".into());
                }
                let value = child!(value);
                if *sum {
                    ExpressionNode::Sum {
                        map: *map,
                        axis: *axis,
                        value,
                    }
                } else {
                    ExpressionNode::Gather {
                        map: *map,
                        axis: *axis,
                        value,
                    }
                }
            }
            _ => return Err("unknown typed rewrite constructor".into()),
        };
        let key = Key { ty, node };
        if let Some(&id) = self.keys.get(&key).or_else(|| local.get(&key)) {
            return Ok(Some(id));
        }
        if self.nodes.len() + nodes.len() == self.capacity {
            return Ok(None);
        }
        let id = ExpressionId(self.snapshot.classes.len() + self.nodes.len() + nodes.len());
        local.insert(key.clone(), id);
        nodes.push(key);
        Ok(Some(id))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
    pub(super) fn reserved_keys(&self) -> usize {
        self.nodes.len()
    }
    pub(super) fn finish(self) -> Plan {
        Plan {
            nodes: self.nodes,
            roots: self.roots,
        }
    }
}

pub(super) struct Plan {
    nodes: Vec<Key>,
    roots: Vec<(u32, ExpressionId)>,
}

impl Plan {
    pub(super) fn commit(
        self,
        graph: &mut RewriteGraph<'_>,
        snapshot: &Snapshot,
        source_classes: &[u32],
        dag: &mut ExpressionDag,
    ) -> Result<Vec<ExpressionId>, String> {
        let mut witnesses = vec![None; snapshot.classes.len()];
        for (id, &class) in source_classes.iter().enumerate() {
            witnesses[class as usize].get_or_insert(ExpressionId(id));
        }
        let mut witnesses = witnesses
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or("e-class has no concrete witness")?;
        let mut values = snapshot
            .values
            .iter()
            .copied()
            .zip(snapshot.types.iter().copied())
            .collect::<Vec<_>>();
        let RewriteGraph {
            egraph,
            func,
            maps,
            coefficients,
            shapes,
            shape_ids,
            coefficient_ids,
        } = graph;
        egraph
            .update(|mut state| {
                let mut context = Context {
                    shapes,
                    shape_ids,
                    coefficients,
                    coefficient_ids,
                };
                for key in &self.nodes {
                    let value = import_node(
                        &mut state,
                        &mut context,
                        func,
                        maps,
                        &key.node,
                        &values,
                        false,
                    )?;
                    values.push(value);
                    let node = key.node.map_children(|id| witnesses[id.0]);
                    witnesses.push(dag.intern(node));
                }
                for &(class, root) in &self.roots {
                    state.union(snapshot.values[class as usize], values[root.0].0)?;
                }
                Ok(())
            })
            .map_err(|e| format!("typed rewrite proposal: {e}"))?;
        Ok(self
            .roots
            .iter()
            .map(|(_, root)| witnesses[root.0])
            .collect())
    }
}

#[cfg(test)]
mod tests;
