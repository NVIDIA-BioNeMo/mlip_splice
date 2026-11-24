// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One typed equivalence graph. Rust admits bounded proofs; egglog maintains congruence.

use std::collections::HashMap;

use algebraic_real::AlgebraicReal;
use egglog::Value;
use egglog::ast::{Command, Subdatatypes, Variant};
use egglog::prelude::*;

use super::ExprTree;
use super::extraction::expression_dag::{ExpressionDag, ExpressionId, ExpressionNode};

mod admission;
use super::extraction::{Classes, ENode};
use crate::ir::Func;
use crate::optimize::AlgebraicWorkBudget;
use crate::shape::Shape;

const MAX_ADDED_NODES: usize = 16_384;
const MAX_ROUNDS: usize = 8;

pub(super) struct RewriteGraph<'a> {
    egraph: EGraph,
    func: &'a Func,
    maps: &'a domain_index::FiniteMapGraph,
    pub(super) coefficients: Vec<ExprTree>,
    shapes: Vec<Shape>,
    shape_ids: HashMap<Shape, usize>,
    coefficient_ids: HashMap<ExprTree, usize>,
}

#[derive(PartialEq, Eq)]
pub(super) struct Snapshot {
    pub(super) classes: Classes,
    pub(super) types: Vec<usize>,
    pub(super) values: Vec<Value>,
}

pub(super) enum Form {
    Class(u32),
    Coefficient(usize, Vec<AlgebraicReal>),
    Unary(&'static str, usize, Box<Form>),
    Binary(&'static str, usize, Box<Form>, Box<Form>),
    Select(usize, Box<Form>, Box<Form>, Box<Form>),
    Map {
        sum: bool,
        map: domain_index::FiniteMapId,
        axis: u32,
        input: usize,
        output: usize,
        value: Box<Form>,
    },
}

impl Form {
    pub(super) fn binary(name: &'static str, ty: usize, left: Form, right: Form) -> Self {
        Self::Binary(name, ty, Box::new(left), Box::new(right))
    }

    fn ty(&self, snapshot: &Snapshot) -> usize {
        match self {
            Self::Class(class) => snapshot.types[*class as usize],
            Self::Coefficient(ty, _)
            | Self::Unary(_, ty, _)
            | Self::Binary(_, ty, _, _)
            | Self::Select(ty, ..) => *ty,
            Self::Map { output, .. } => *output,
        }
    }
}

pub(super) struct Proposal {
    pub(super) class: u32,
    pub(super) form: Form,
}

impl<'a> RewriteGraph<'a> {
    pub(super) fn new(
        func: &'a Func,
        maps: &'a domain_index::FiniteMapGraph,
    ) -> Result<Self, String> {
        let mut egraph = EGraph::default();
        macro_rules! variant {
            ($name:ident $($input:ident)*) => {
                Variant { span: span!(), name: stringify!($name).into(), types: vec![$(stringify!($input).into()),*], cost: None, unextractable: false }
            };
        }
        egraph
            .run_program(vec![Command::Datatypes {
                span: span!(),
                datatypes: vec![(
                    span!(),
                    "Expr".into(),
                    Subdatatypes::Variants(vec![
                        variant!(Value i64 i64),
                        variant!(Coefficient i64 i64),
                        variant!(Negate i64 Expr),
                        variant!(Add i64 Expr Expr),
                        variant!(Multiply i64 Expr Expr),
                        variant!(Select i64 Expr Expr Expr),
                        variant!(Gather i64 i64 i64 i64 Expr),
                        variant!(Sum i64 i64 i64 i64 Expr),
                    ]),
                )],
            }])
            .map_err(|e| e.to_string())?;
        Ok(Self {
            egraph,
            func,
            maps,
            coefficients: Vec::new(),
            shapes: Vec::new(),
            shape_ids: HashMap::new(),
            coefficient_ids: HashMap::new(),
        })
    }

    pub(super) fn import(&mut self, dag: &ExpressionDag) -> Result<(), String> {
        self.dag_values(dag, false).map(|_| ())
    }

    fn dag_values(&mut self, dag: &ExpressionDag, lookup: bool) -> Result<Vec<Value>, String> {
        let Self {
            egraph,
            func,
            maps,
            coefficients,
            shapes,
            shape_ids,
            coefficient_ids,
        } = self;
        egraph
            .update(|mut state| {
                let mut context = Context {
                    shapes,
                    shape_ids,
                    coefficients,
                    coefficient_ids,
                };
                let mut values = Vec::with_capacity(dag.len());
                for (_, node) in dag.nodes() {
                    let value =
                        import_node(&mut state, &mut context, func, maps, node, &values, lookup)?;
                    values.push(value);
                }
                Ok(values.into_iter().map(|(value, _)| value).collect())
            })
            .map_err(|e| e.to_string())
    }

    pub(super) fn snapshot_dag(
        &mut self,
        dag: &ExpressionDag,
    ) -> Result<(Snapshot, Vec<u32>), String> {
        let roots = self.dag_values(dag, true)?;
        let snapshot = self.snapshot()?;
        let index = snapshot
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect::<HashMap<_, _>>();
        let roots = roots
            .iter()
            .map(|root| {
                index
                    .get(root)
                    .copied()
                    .ok_or_else(|| "typed root has no e-class".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((snapshot, roots))
    }

    pub(super) fn instance_counts(&self, snapshot: &Snapshot) -> Result<Vec<u64>, String> {
        snapshot
            .types
            .iter()
            .map(|&ty| instances(&self.shapes[ty], self.maps))
            .collect()
    }

    fn snapshot(&mut self) -> Result<Snapshot, String> {
        let shapes = &self.shapes;
        let maps = self.maps;
        self.egraph
            .update(|state| read_classes(&state, shapes, maps))
            .map_err(|e| e.to_string())
    }

    pub(super) fn rewrite(
        &mut self,
        dag: &mut ExpressionDag,
        function: usize,
        max_terms: usize,
        budget: &mut AlgebraicWorkBudget,
    ) -> Result<Vec<ExpressionId>, String> {
        let mut reserved_keys = 0usize;
        let mut proofs = Vec::new();
        for _ in 0..MAX_ROUNDS {
            let (snapshot, source_classes) = self.snapshot_dag(dag)?;
            let mut proposals = super::proposals::proposals(
                &snapshot,
                &self.shapes,
                &self.coefficients,
                max_terms,
                budget,
            );
            proposals.extend(super::proposals::identity_proposals(
                &snapshot,
                &self.shapes,
                &self.coefficients,
                function,
                budget,
            ));
            let mut admission = admission::Admission::new(
                &snapshot,
                &self.shapes,
                &self.coefficients,
                self.maps,
                MAX_ADDED_NODES - reserved_keys,
            );
            for proposal in proposals {
                admission.plan(proposal)?;
            }
            if admission.is_empty() {
                break;
            }
            reserved_keys += admission.reserved_keys();
            let plan = admission.finish();
            proofs.extend(plan.commit(self, &snapshot, &source_classes, dag)?);
            if self.snapshot()? == snapshot {
                break;
            }
        }
        Ok(proofs)
    }
}

struct Context<'a> {
    shapes: &'a mut Vec<Shape>,
    shape_ids: &'a mut HashMap<Shape, usize>,
    coefficients: &'a mut Vec<ExprTree>,
    coefficient_ids: &'a mut HashMap<ExprTree, usize>,
}

impl Context<'_> {
    fn shape(&mut self, shape: Shape) -> usize {
        *self.shape_ids.entry(shape.clone()).or_insert_with(|| {
            let id = self.shapes.len();
            self.shapes.push(shape);
            id
        })
    }

    fn coefficient(&mut self, ty: usize, values: &[AlgebraicReal]) -> usize {
        let tree = ExprTree::Coefficient {
            shape: self.shapes[ty].clone(),
            values: values.into(),
        };
        *self.coefficient_ids.entry(tree.clone()).or_insert_with(|| {
            let id = self.coefficients.len();
            self.coefficients.push(tree);
            id
        })
    }
}

fn import_node(
    state: &mut FullState<'_, '_>,
    context: &mut Context<'_>,
    func: &Func,
    maps: &domain_index::FiniteMapGraph,
    node: &ExpressionNode,
    children: &[(Value, usize)],
    lookup: bool,
) -> Result<(Value, usize), egglog::Error> {
    macro_rules! apply {
        ($name:expr, $args:expr) => {
            if lookup {
                state.eclass_of($name, $args)?.ok_or_else(|| {
                    egglog::Error::BackendError("imported expression disappeared".into())
                })?
            } else {
                state.add($name, $args)?
            }
        };
    }
    Ok(match node {
        ExpressionNode::Value(value) => {
            let ty = context.shape(func.value_shape(*value).clone());
            (apply!("Value", (ty as i64, i64::from(value.0))), ty)
        }
        ExpressionNode::Coefficient { shape, values } => {
            let ty = context.shape(shape.clone());
            let id = context.coefficient(ty, values);
            (apply!("Coefficient", (ty as i64, id as i64)), ty)
        }
        ExpressionNode::Negate(x) => {
            let (x, ty) = children[x.0];
            (apply!("Negate", (ty as i64, x)), ty)
        }
        ExpressionNode::Add(x, y) | ExpressionNode::Multiply(x, y) => {
            let (x, ty) = children[x.0];
            let (y, right) = children[y.0];
            if ty != right {
                return Err(egglog::Error::BackendError(
                    "typed binary operands disagree".into(),
                ));
            }
            let name = if matches!(node, ExpressionNode::Add(..)) {
                "Add"
            } else {
                "Multiply"
            };
            (apply!(name, (ty as i64, x, y)), ty)
        }
        ExpressionNode::Select(p, x, y) => {
            let (p, _) = children[p.0];
            let (x, ty) = children[x.0];
            let (y, right) = children[y.0];
            if ty != right {
                return Err(egglog::Error::BackendError(
                    "typed select arms disagree".into(),
                ));
            }
            (apply!("Select", (ty as i64, p, x, y)), ty)
        }
        ExpressionNode::Gather { map, axis, value } | ExpressionNode::Sum { map, axis, value } => {
            let (value, input) = children[value.0];
            let declared = maps.map(*map).map_err(egglog::Error::BackendError)?;
            let sum = matches!(node, ExpressionNode::Sum { .. });
            let (source, target) = if sum {
                (declared.source, declared.target)
            } else {
                (declared.target, declared.source)
            };
            let mut output = context.shapes[input].clone();
            if output.domain_axes.get(*axis as usize) != Some(&source) {
                return Err(egglog::Error::BackendError(
                    "typed map endpoint disagrees".into(),
                ));
            }
            output.domain_axes[*axis as usize] = target;
            let output = context.shape(output);
            (
                apply!(
                    if sum { "Sum" } else { "Gather" },
                    (
                        map.index() as i64,
                        i64::from(*axis),
                        input as i64,
                        output as i64,
                        value
                    )
                ),
                output,
            )
        }
    })
}

pub(super) fn instances(
    shape: &Shape,
    graph: &domain_index::FiniteMapGraph,
) -> Result<u64, String> {
    let count = u64::try_from(shape.element_count()?).map_err(|e| e.to_string())?;
    shape.domain_axes.iter().try_fold(count, |count, &axis| {
        count
            .checked_mul(graph.set(axis)?.extent as u64)
            .ok_or_else(|| "operation instance count overflow".into())
    })
}

fn read_classes(
    state: &FullState<'_, '_>,
    shapes: &[Shape],
    maps: &domain_index::FiniteMapGraph,
) -> Result<Snapshot, egglog::Error> {
    let mut snapshot = Snapshot {
        classes: Vec::new(),
        types: Vec::new(),
        values: Vec::new(),
    };
    let mut index = HashMap::new();
    fn intern(snapshot: &mut Snapshot, index: &mut HashMap<Value, u32>, value: Value) -> u32 {
        *index.entry(value).or_insert_with(|| {
            let id = snapshot.classes.len() as u32;
            snapshot.classes.push(Vec::new());
            snapshot.types.push(usize::MAX);
            snapshot.values.push(value);
            id
        })
    }
    let mut error = None;
    for name in [
        "Value",
        "Coefficient",
        "Negate",
        "Add",
        "Multiply",
        "Select",
        "Gather",
        "Sum",
    ] {
        state.constructor_enodes(name, |enode| {
            let map = name == "Gather" || name == "Sum";
            let ty = state.value_to_base::<i64>(enode.children[if map { 3 } else { 0 }]) as usize;
            let class = intern(&mut snapshot, &mut index, enode.eclass);
            if snapshot.types[class as usize] != usize::MAX && snapshot.types[class as usize] != ty
            {
                error = Some("mixed types in an e-class".to_string());
            }
            snapshot.types[class as usize] = ty;
            let instances = match instances(&shapes[ty], maps) {
                Ok(n) => n,
                Err(e) => {
                    error = Some(e);
                    0
                }
            };
            let mut child = |slot| intern(&mut snapshot, &mut index, enode.children[slot]);
            let node = match name {
                "Value" => ENode::Value(crate::ir::ValueId(
                    state.value_to_base::<i64>(enode.children[1]) as u32,
                )),
                "Coefficient" => {
                    ENode::Coefficient(state.value_to_base::<i64>(enode.children[1]) as usize)
                }
                "Negate" => ENode::Negate {
                    instances,
                    value: child(1),
                },
                "Add" => ENode::Add {
                    instances,
                    left: child(1),
                    right: child(2),
                },
                "Multiply" => ENode::Multiply {
                    instances,
                    left: child(1),
                    right: child(2),
                },
                "Select" => ENode::Select {
                    instances,
                    predicate: child(1),
                    on_true: child(2),
                    on_false: child(3),
                },
                _ => {
                    let map = domain_index::FiniteMapId(
                        state.value_to_base::<i64>(enode.children[0]) as u32,
                    );
                    let axis = state.value_to_base::<i64>(enode.children[1]) as u32;
                    let value = child(4);
                    if name == "Sum" {
                        ENode::Sum { map, axis, value }
                    } else {
                        ENode::Gather { map, axis, value }
                    }
                }
            };
            snapshot.classes[class as usize].push(node);
        })?;
    }
    if let Some(error) = error {
        return Err(egglog::Error::BackendError(error));
    }
    Ok(snapshot)
}
