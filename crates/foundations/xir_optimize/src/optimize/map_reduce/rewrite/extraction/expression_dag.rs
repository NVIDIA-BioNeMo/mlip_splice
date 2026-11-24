// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interned expression DAGs and their topological SSA boundaries.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use algebraic_real::AlgebraicReal;

use super::super::ExprTree;
use crate::ir::{Def, Func, Node, ValueId};
use crate::ops::{
    Add, Compare, ComparisonDirection, Constant, GatherAlongMap, Literal, Multiply, Negate, Select,
    Subtract, SumAlongMap,
};
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::optimize::map_reduce::rewrite) struct ExpressionId(
    pub(in crate::optimize::map_reduce::rewrite) usize,
);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::optimize::map_reduce::rewrite) enum ExpressionNode {
    Value(ValueId),
    Coefficient {
        shape: Shape,
        values: Arc<[AlgebraicReal]>,
    },
    Negate(ExpressionId),
    Select(ExpressionId, ExpressionId, ExpressionId),
    Add(ExpressionId, ExpressionId),
    Multiply(ExpressionId, ExpressionId),
    Gather {
        map: domain_index::FiniteMapId,
        axis: u32,
        value: ExpressionId,
    },
    Sum {
        map: domain_index::FiniteMapId,
        axis: u32,
        value: ExpressionId,
    },
}

impl ExpressionNode {
    pub(in crate::optimize::map_reduce::rewrite) fn map_children(
        &self,
        mut child: impl FnMut(ExpressionId) -> ExpressionId,
    ) -> Self {
        match *self {
            Self::Value(_) | Self::Coefficient { .. } => self.clone(),
            Self::Negate(x) => Self::Negate(child(x)),
            Self::Add(x, y) => Self::Add(child(x), child(y)),
            Self::Multiply(x, y) => Self::Multiply(child(x), child(y)),
            Self::Select(p, x, y) => Self::Select(child(p), child(x), child(y)),
            Self::Gather { map, axis, value } => Self::Gather {
                map,
                axis,
                value: child(value),
            },
            Self::Sum { map, axis, value } => Self::Sum {
                map,
                axis,
                value: child(value),
            },
        }
    }

    pub(in crate::optimize::map_reduce::rewrite) fn children(
        &self,
    ) -> impl Iterator<Item = ExpressionId> {
        let children = match *self {
            Self::Value(_) | Self::Coefficient { .. } => [None, None, None],
            Self::Negate(x) | Self::Gather { value: x, .. } | Self::Sum { value: x, .. } => {
                [Some(x), None, None]
            }
            Self::Add(x, y) | Self::Multiply(x, y) => [Some(x), Some(y), None],
            Self::Select(p, x, y) => [Some(p), Some(x), Some(y)],
        };
        children.into_iter().flatten()
    }
}

#[derive(Clone, Debug, Default)]
pub(in crate::optimize::map_reduce::rewrite) struct ExpressionDag {
    nodes: Vec<ExpressionNode>,
    ids: HashMap<ExpressionNode, ExpressionId>,
}

impl ExpressionDag {
    pub(in crate::optimize::map_reduce::rewrite) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(in crate::optimize::map_reduce::rewrite) fn node(
        &self,
        id: ExpressionId,
    ) -> &ExpressionNode {
        &self.nodes[id.0]
    }

    pub(in crate::optimize::map_reduce::rewrite) fn nodes(
        &self,
    ) -> impl Iterator<Item = (ExpressionId, &ExpressionNode)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (ExpressionId(index), node))
    }

    pub(in crate::optimize::map_reduce::rewrite) fn intern(
        &mut self,
        node: ExpressionNode,
    ) -> ExpressionId {
        *self.ids.entry(node).or_insert_with_key(|node| {
            let id = ExpressionId(self.nodes.len());
            self.nodes.push(node.clone());
            id
        })
    }

    pub(in crate::optimize::map_reduce::rewrite) fn import(
        &mut self,
        tree: &ExprTree,
    ) -> ExpressionId {
        let node = match tree {
            ExprTree::Value(value) => ExpressionNode::Value(*value),
            ExprTree::Coefficient { shape, values } => ExpressionNode::Coefficient {
                shape: shape.clone(),
                values: Arc::clone(values),
            },
            ExprTree::Negate(x) => ExpressionNode::Negate(self.import(x)),
            ExprTree::Select(p, x, y) => {
                ExpressionNode::Select(self.import(p), self.import(x), self.import(y))
            }
            ExprTree::Add(x, y) => ExpressionNode::Add(self.import(x), self.import(y)),
            ExprTree::Multiply(x, y) => ExpressionNode::Multiply(self.import(x), self.import(y)),
            ExprTree::Gather { map, axis, value } => ExpressionNode::Gather {
                map: *map,
                axis: *axis,
                value: self.import(value),
            },
            ExprTree::Sum { map, axis, value } => ExpressionNode::Sum {
                map: *map,
                axis: *axis,
                value: self.import(value),
            },
        };
        self.intern(node)
    }

    #[cfg(test)]
    pub(in crate::optimize::map_reduce::rewrite) fn materialize(
        &self,
        id: ExpressionId,
    ) -> ExprTree {
        let child = |id| Box::new(self.materialize(id));
        match &self.nodes[id.0] {
            ExpressionNode::Value(value) => ExprTree::Value(*value),
            ExpressionNode::Coefficient { shape, values } => ExprTree::Coefficient {
                shape: shape.clone(),
                values: Arc::clone(values),
            },
            ExpressionNode::Negate(x) => ExprTree::Negate(child(*x)),
            ExpressionNode::Select(p, x, y) => ExprTree::Select(child(*p), child(*x), child(*y)),
            ExpressionNode::Add(x, y) => ExprTree::Add(child(*x), child(*y)),
            ExpressionNode::Multiply(x, y) => ExprTree::Multiply(child(*x), child(*y)),
            ExpressionNode::Gather { map, axis, value } => ExprTree::Gather {
                map: *map,
                axis: *axis,
                value: child(*value),
            },
            ExpressionNode::Sum { map, axis, value } => ExprTree::Sum {
                map: *map,
                axis: *axis,
                value: child(*value),
            },
        }
    }

    pub(in crate::optimize::map_reduce::rewrite) fn quality(
        &self,
        roots: impl IntoIterator<Item = ExpressionId>,
    ) -> (usize, usize, usize) {
        let mut seen = HashSet::new();
        let mut pending = roots.into_iter().collect::<Vec<_>>();
        let mut constants = 0usize;
        let mut selects = 0usize;
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            match &self.nodes[id.0] {
                ExpressionNode::Value(_) => {}
                ExpressionNode::Coefficient { values, .. } => {
                    constants = constants.saturating_add(
                        values
                            .iter()
                            .map(|value| value.structural_size().max(1))
                            .sum::<usize>(),
                    );
                }
                ExpressionNode::Negate(x)
                | ExpressionNode::Gather { value: x, .. }
                | ExpressionNode::Sum { value: x, .. } => pending.push(*x),
                ExpressionNode::Add(x, y) | ExpressionNode::Multiply(x, y) => {
                    pending.extend([*x, *y]);
                }
                ExpressionNode::Select(p, x, y) => {
                    selects += 1;
                    pending.extend([*p, *x, *y]);
                }
            }
        }
        (seen.len(), selects, constants)
    }
}

#[cfg(test)]
mod tests;

const MAX_SOURCE_NODES: usize = 1 << 20;

fn select_guard_value(func: &Func, select: &Node) -> Option<ValueId> {
    select.op.as_any().downcast_ref::<Select>()?;
    let Def::NodeResult { node, slot: 0 } = func.values[select.inputs[0].index()].def else {
        return None;
    };
    let compare = &func.nodes[node];
    if compare.results.len() != 1
        || !compare
            .op
            .as_any()
            .downcast_ref::<Compare>()
            .is_some_and(|op| {
                matches!(
                    op.comparison_direction,
                    ComparisonDirection::Lt
                        | ComparisonDirection::Le
                        | ComparisonDirection::Gt
                        | ComparisonDirection::Ge
                )
            })
    {
        return None;
    }
    let is_zero = |value: ValueId| {
        func.value_shape(value).dtype.is_float()
            && crate::analysis::constant_literal(func, value)
                .is_some_and(|literal| literal.data.iter().all(|&byte| byte == 0))
    };
    let (left, right) = (compare.inputs[0], compare.inputs[1]);
    if is_zero(right) && func.value_shape(left).dtype.is_float() {
        Some(left)
    } else if is_zero(left) && func.value_shape(right).dtype.is_float() {
        Some(right)
    } else {
        None
    }
}

pub(in crate::optimize::map_reduce::rewrite) struct SourceExpressions {
    pub(in crate::optimize::map_reduce::rewrite) dag: ExpressionDag,
    pub(in crate::optimize::map_reduce::rewrite) roots: Vec<(ValueId, ExpressionId)>,
    pub(in crate::optimize::map_reduce::rewrite) values: Vec<Option<ExpressionId>>,
}

pub(in crate::optimize::map_reduce::rewrite) fn import(
    func: &Func,
) -> Result<Option<SourceExpressions>, String> {
    if func.values.len() > MAX_SOURCE_NODES {
        return Ok(None);
    }
    let guards = func
        .nodes
        .iter()
        .filter_map(|node| select_guard_value(func, node))
        .collect::<HashSet<_>>();
    let supported = func
        .nodes
        .iter()
        .map(|node| {
            node.results.len() == 1
                && func.value_shape(node.results[0]).dtype.is_float()
                && (node.op.as_any().is::<Add>()
                    || node.op.as_any().is::<Multiply>()
                    || node.op.as_any().is::<Subtract>()
                    || node.op.as_any().is::<Negate>()
                    || node.op.as_any().is::<Select>()
                    || node.op.as_any().is::<GatherAlongMap>()
                    || node.op.as_any().is::<SumAlongMap>())
        })
        .collect::<Vec<_>>();
    let mut leaves = func.results.iter().copied().collect::<HashSet<_>>();
    for (node, &supported) in func.nodes.iter().zip(&supported) {
        if !supported {
            leaves.extend(node.inputs.iter().copied());
        }
    }
    let mut needed = vec![false; func.values.len()];
    let mut pending = func
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            let &result = node.results.first()?;
            (supported[index] && !guards.contains(&result) && leaves.contains(&result))
                .then_some(result)
        })
        .collect::<Vec<_>>();
    while let Some(value) = pending.pop() {
        if std::mem::replace(&mut needed[value.index()], true) {
            continue;
        }
        if let Def::NodeResult { node, .. } = func.values[value.index()].def
            && supported[node]
            && !guards.contains(&value)
        {
            pending.extend(func.nodes[node].inputs.iter().copied());
        }
    }
    let mut dag = ExpressionDag::default();
    let mut values = vec![None; func.values.len()];
    for &param in &func.params {
        if !needed[param.index()] {
            continue;
        }
        values[param.index()] = Some(dag.intern(ExpressionNode::Value(param)));
    }
    let mut roots = Vec::new();
    for (index, node) in func.nodes.iter().enumerate() {
        for &result in &node.results {
            if !needed[result.index()] {
                continue;
            }
            let input = |slot: usize| {
                values[node.inputs[slot].index()]
                    .ok_or_else(|| "finite-map input is not in SSA order".to_string())
            };
            let expression = if node.results.len() == 1 && !guards.contains(&result) {
                if let Some(constant) = node.op.as_any().downcast_ref::<Constant>()
                    && constant
                        .literal
                        .shape
                        .num_elements()
                        .is_some_and(|n| n <= 4096)
                    && let Some(values) = constant.literal.algebraic_values()
                {
                    ExpressionNode::Coefficient {
                        shape: constant.literal.shape.clone(),
                        values: values.into(),
                    }
                } else if !supported[index] {
                    ExpressionNode::Value(result)
                } else if node.op.as_any().is::<Negate>() {
                    ExpressionNode::Negate(input(0)?)
                } else if node.op.as_any().is::<Select>() {
                    let predicate = dag.intern(ExpressionNode::Value(node.inputs[0]));
                    ExpressionNode::Select(predicate, input(1)?, input(2)?)
                } else if node.op.as_any().is::<Subtract>() {
                    let right = dag.intern(ExpressionNode::Negate(input(1)?));
                    ExpressionNode::Add(input(0)?, right)
                } else if node.op.as_any().is::<Add>() {
                    ExpressionNode::Add(input(0)?, input(1)?)
                } else if node.op.as_any().is::<Multiply>() {
                    ExpressionNode::Multiply(input(0)?, input(1)?)
                } else if let Some(op) = node.op.as_any().downcast_ref::<GatherAlongMap>() {
                    ExpressionNode::Gather {
                        map: op.map,
                        axis: op.axis,
                        value: input(0)?,
                    }
                } else if let Some(op) = node.op.as_any().downcast_ref::<SumAlongMap>() {
                    ExpressionNode::Sum {
                        map: op.map,
                        axis: op.axis,
                        value: input(0)?,
                    }
                } else {
                    unreachable!()
                }
            } else {
                ExpressionNode::Value(result)
            };
            let id = dag.intern(expression);
            values[result.index()] = Some(id);
            if supported[index] && !guards.contains(&result) && leaves.contains(&result) {
                roots.push((result, id));
            }
            if dag.len() > MAX_SOURCE_NODES {
                return Ok(None);
            }
        }
    }
    Ok(Some(SourceExpressions { dag, roots, values }))
}

pub(in crate::optimize::map_reduce::rewrite) fn emit(
    rebuilt: &mut Rebuilder,
    graph: &domain_index::FiniteMapGraph,
    dag: &ExpressionDag,
    root: ExpressionId,
    emitted: &mut HashMap<ExpressionId, ValueId>,
) -> Result<ValueId, String> {
    let mut pending = vec![(root, false)];
    while let Some((id, ready)) = pending.pop() {
        if emitted.contains_key(&id) {
            continue;
        }
        let node = dag.node(id);
        if !ready {
            pending.push((id, true));
            pending.extend(node.children().map(|child| (child, false)));
            continue;
        }
        let child = |id: ExpressionId| emitted[&id];
        let value = match node {
            ExpressionNode::Value(value) => rebuilt.mapped(*value)?,
            ExpressionNode::Coefficient { shape, values } => rebuilt.emit_fresh(
                shape.clone(),
                Constant {
                    literal: Literal::exact(shape.clone(), values.to_vec())?,
                },
                [],
            ),
            ExpressionNode::Negate(value) => {
                let value = child(*value);
                rebuilt.emit_fresh(rebuilt.func.value_shape(value).clone(), Negate, [value])
            }
            ExpressionNode::Select(p, x, y) => {
                let (p, x, y) = (child(*p), child(*x), child(*y));
                let shape = rebuilt.func.value_shape(x).clone();
                if rebuilt.func.value_shape(y) != &shape {
                    return Err("finite-map extraction changed select arm shapes".into());
                }
                rebuilt.emit_fresh(shape, Select, [p, x, y])
            }
            ExpressionNode::Add(x, y) | ExpressionNode::Multiply(x, y) => {
                let (x, y) = (child(*x), child(*y));
                let shape = rebuilt.func.value_shape(x).clone();
                if rebuilt.func.value_shape(y) != &shape {
                    return Err(
                        "finite-map extraction combined values with different shapes".into(),
                    );
                }
                let op: Arc<dyn crate::op::Op> = if matches!(node, ExpressionNode::Add(..)) {
                    Arc::new(Add)
                } else {
                    Arc::new(Multiply)
                };
                rebuilt.emit_fresh_op(shape, op, [x, y])
            }
            ExpressionNode::Gather { map, axis, value }
            | ExpressionNode::Sum { map, axis, value } => {
                let value = child(*value);
                let declared = graph.map(*map)?;
                let mut shape = rebuilt.func.value_shape(value).clone();
                let sum = matches!(node, ExpressionNode::Sum { .. });
                let (input, output) = if sum {
                    (declared.source, declared.target)
                } else {
                    (declared.target, declared.source)
                };
                if shape.domain_axes.get(*axis as usize) != Some(&input) {
                    return Err("finite-map extraction reversed a map endpoint".into());
                }
                shape.domain_axes[*axis as usize] = output;
                if sum {
                    rebuilt.emit_fresh(
                        shape,
                        SumAlongMap {
                            map: *map,
                            axis: *axis,
                        },
                        [value],
                    )
                } else {
                    rebuilt.emit_fresh(
                        shape,
                        GatherAlongMap {
                            map: *map,
                            axis: *axis,
                        },
                        [value],
                    )
                }
            }
        };
        emitted.insert(id, value);
    }
    Ok(emitted[&root])
}

pub(in crate::optimize::map_reduce::rewrite) fn latest_definitions(
    dag: &ExpressionDag,
    func: &Func,
) -> Vec<usize> {
    let mut latest = Vec::with_capacity(dag.len());
    for (_, node) in dag.nodes() {
        let definition = match node {
            ExpressionNode::Value(value) => match func.values[value.index()].def {
                Def::Param(_) => 0,
                Def::NodeResult { node, .. } => node + 1,
            },
            _ => node.children().map(|id| latest[id.0]).max().unwrap_or(0),
        };
        latest.push(definition);
    }
    latest
}

pub(in crate::optimize::map_reduce::rewrite) fn live_nodes(
    func: &Func,
    dag: &ExpressionDag,
    replacements: &HashMap<ValueId, &ExpressionId>,
) -> Vec<bool> {
    enum Visit {
        Value(ValueId),
        Expression(ExpressionId),
    }
    let mut pending = func
        .results
        .iter()
        .copied()
        .map(Visit::Value)
        .collect::<Vec<_>>();
    let mut values = vec![false; func.values.len()];
    let mut expressions = HashSet::new();
    let mut nodes = vec![false; func.nodes.len()];
    while let Some(visit) = pending.pop() {
        match visit {
            Visit::Value(value) => {
                if std::mem::replace(&mut values[value.index()], true) {
                    continue;
                }
                let Def::NodeResult { node, .. } = func.values[value.index()].def else {
                    continue;
                };
                nodes[node] = true;
                if let Some(&&id) = replacements.get(&value) {
                    pending.push(Visit::Expression(id));
                } else {
                    pending.extend(func.nodes[node].inputs.iter().copied().map(Visit::Value));
                }
            }
            Visit::Expression(id) => {
                if !expressions.insert(id) {
                    continue;
                }
                match dag.node(id) {
                    ExpressionNode::Value(value) => pending.push(Visit::Value(*value)),
                    node => pending.extend(node.children().map(Visit::Expression)),
                }
            }
        }
    }
    nodes
}
