// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Laws of zero-extended pullback, fiberwise sum, and pointwise selection.

use algebraic_real::AlgebraicReal;
use domain_index::FiniteMapId;

use super::super::ExprTree;
use super::super::egraph::{Form, Proposal, Snapshot};
use super::super::extraction::ENode;
use crate::optimize::AlgebraicWorkBudget;
use crate::shape::Shape;

fn map_form(
    sum: bool,
    map: FiniteMapId,
    axis: u32,
    input: usize,
    output: usize,
    value: Form,
) -> Form {
    Form::Map {
        sum,
        map,
        axis,
        input,
        output,
        value: Box::new(value),
    }
}

fn negate(ty: usize, value: Form) -> Form {
    Form::Unary("Negate", ty, Box::new(value))
}

pub(in crate::optimize::map_reduce::rewrite) fn proposals(
    snapshot: &Snapshot,
    shapes: &[Shape],
    coefficients: &[ExprTree],
    function: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Vec<Proposal> {
    let mut proposals = Vec::new();
    let mut steps = 16_384usize;
    macro_rules! spend {
        ($class:expr) => {
            if proposals.len() >= 2048 || !take_step(budget, function, $class, &mut steps) {
                return proposals;
            }
        };
    }
    for (class, nodes) in snapshot.classes.iter().enumerate() {
        let class = class as u32;
        let ty = snapshot.types[class as usize];
        if !shapes[ty].dtype.is_float() {
            continue;
        }
        for &node in nodes {
            spend!(class);
            let mut push = |form| {
                if proposals.len() < 2048 {
                    proposals.push(Proposal { class, form });
                }
            };
            match node {
                ENode::Value(_) | ENode::Coefficient(_) => {}
                ENode::Gather { map, axis, value } | ENode::Sum { map, axis, value } => {
                    let sum = matches!(node, ENode::Sum { .. });
                    let input = snapshot.types[value as usize];
                    for &child in &snapshot.classes[value as usize] {
                        if !take_step(budget, function, class, &mut steps) {
                            return proposals;
                        }
                        let mapped =
                            |value| map_form(sum, map, axis, input, ty, Form::Class(value));
                        match child {
                            ENode::Add { left, right, .. } => {
                                push(Form::binary("Add", ty, mapped(left), mapped(right)))
                            }
                            ENode::Multiply { left, right, .. } if !sum => {
                                push(Form::binary("Multiply", ty, mapped(left), mapped(right)))
                            }
                            ENode::Negate { value, .. } => push(negate(ty, mapped(value))),
                            ENode::Coefficient(id) => {
                                let ExprTree::Coefficient { values, .. } = &coefficients[id] else {
                                    unreachable!()
                                };
                                if values.iter().all(AlgebraicReal::is_zero) {
                                    push(Form::Coefficient(ty, values.to_vec()));
                                }
                            }
                            ENode::Multiply { left, right, .. } => {
                                for (factor, residual) in [(left, right), (right, left)] {
                                    for &factor_node in &snapshot.classes[factor as usize] {
                                        if !take_step(budget, function, class, &mut steps) {
                                            return proposals;
                                        }
                                        match factor_node {
                                            ENode::Gather {
                                                map: pulled_map,
                                                axis: pulled_axis,
                                                value: coefficient,
                                            } if pulled_map == map
                                                && pulled_axis == axis
                                                && snapshot.types[coefficient as usize] == ty =>
                                            {
                                                push(Form::binary(
                                                    "Multiply",
                                                    ty,
                                                    Form::Class(coefficient),
                                                    mapped(residual),
                                                ));
                                            }
                                            ENode::Coefficient(id) => {
                                                let ExprTree::Coefficient { values, .. } =
                                                    &coefficients[id]
                                                else {
                                                    unreachable!()
                                                };
                                                // A domain-uniform dense coefficient commutes with a sum,
                                                // even when the arrow drops unmapped source cells.
                                                push(Form::binary(
                                                    "Multiply",
                                                    ty,
                                                    Form::Coefficient(ty, values.to_vec()),
                                                    mapped(residual),
                                                ));
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                ENode::Negate { value, .. } => {
                    for &child in &snapshot.classes[value as usize] {
                        if !take_step(budget, function, class, &mut steps) {
                            return proposals;
                        }
                        if let ENode::Select {
                            predicate,
                            on_true,
                            on_false,
                            ..
                        } = child
                        {
                            push(Form::Select(
                                ty,
                                Box::new(Form::Class(predicate)),
                                Box::new(negate(ty, Form::Class(on_true))),
                                Box::new(negate(ty, Form::Class(on_false))),
                            ));
                        }
                    }
                }
                ENode::Add { left, right, .. } | ENode::Multiply { left, right, .. } => {
                    let multiply = matches!(node, ENode::Multiply { .. });
                    let name = if multiply { "Multiply" } else { "Add" };
                    for &l in &snapshot.classes[left as usize] {
                        for &r in &snapshot.classes[right as usize] {
                            if !take_step(budget, function, class, &mut steps) {
                                return proposals;
                            }
                            match (l, r) {
                                (
                                    ENode::Gather {
                                        map,
                                        axis,
                                        value: a,
                                    },
                                    ENode::Gather {
                                        map: other,
                                        axis: other_axis,
                                        value: b,
                                    },
                                ) if map == other
                                    && axis == other_axis
                                    && snapshot.types[a as usize] == snapshot.types[b as usize] =>
                                {
                                    let input = snapshot.types[a as usize];
                                    push(map_form(
                                        false,
                                        map,
                                        axis,
                                        input,
                                        ty,
                                        Form::binary(name, input, Form::Class(a), Form::Class(b)),
                                    ));
                                }
                                (
                                    ENode::Sum {
                                        map,
                                        axis,
                                        value: a,
                                    },
                                    ENode::Sum {
                                        map: other,
                                        axis: other_axis,
                                        value: b,
                                    },
                                ) if !multiply
                                    && map == other
                                    && axis == other_axis
                                    && snapshot.types[a as usize] == snapshot.types[b as usize] =>
                                {
                                    let input = snapshot.types[a as usize];
                                    push(map_form(
                                        true,
                                        map,
                                        axis,
                                        input,
                                        ty,
                                        Form::binary("Add", input, Form::Class(a), Form::Class(b)),
                                    ));
                                }
                                (
                                    ENode::Multiply {
                                        left: a, right: b, ..
                                    },
                                    ENode::Multiply {
                                        left: c, right: d, ..
                                    },
                                ) if !multiply && a == c => {
                                    push(Form::binary(
                                        "Multiply",
                                        ty,
                                        Form::Class(a),
                                        Form::binary("Add", ty, Form::Class(b), Form::Class(d)),
                                    ));
                                }
                                (
                                    ENode::Select {
                                        predicate: p,
                                        on_true: a,
                                        on_false: b,
                                        ..
                                    },
                                    ENode::Select {
                                        predicate: q,
                                        on_true: c,
                                        on_false: d,
                                        ..
                                    },
                                ) if p == q => {
                                    push(Form::Select(
                                        ty,
                                        Box::new(Form::Class(p)),
                                        Box::new(Form::binary(
                                            name,
                                            ty,
                                            Form::Class(a),
                                            Form::Class(c),
                                        )),
                                        Box::new(Form::binary(
                                            name,
                                            ty,
                                            Form::Class(b),
                                            Form::Class(d),
                                        )),
                                    ));
                                }
                                _ => {}
                            }
                        }
                    }
                    if multiply {
                        for (factor, residual) in [(left, right), (right, left)] {
                            for &child in &snapshot.classes[residual as usize] {
                                if !take_step(budget, function, class, &mut steps) {
                                    return proposals;
                                }
                                if let ENode::Sum { map, axis, value } = child {
                                    let input = snapshot.types[value as usize];
                                    let constant = snapshot.classes[factor as usize]
                                        .iter()
                                        .find_map(|node| match node {
                                            ENode::Coefficient(id) => Some(*id),
                                            _ => None,
                                        });
                                    let pulled = if let Some(id) = constant {
                                        let ExprTree::Coefficient { values, .. } =
                                            &coefficients[id]
                                        else {
                                            unreachable!()
                                        };
                                        Form::Coefficient(input, values.to_vec())
                                    } else {
                                        map_form(false, map, axis, ty, input, Form::Class(factor))
                                    };
                                    push(map_form(
                                        true,
                                        map,
                                        axis,
                                        input,
                                        ty,
                                        Form::binary("Multiply", input, pulled, Form::Class(value)),
                                    ));
                                }
                            }
                        }
                    }
                }
                ENode::Select {
                    predicate,
                    on_true,
                    on_false,
                    ..
                } => {
                    if on_true == on_false {
                        push(Form::Class(on_true));
                    }
                    for &left in &snapshot.classes[on_true as usize] {
                        for &right in &snapshot.classes[on_false as usize] {
                            if !take_step(budget, function, class, &mut steps) {
                                return proposals;
                            }
                            let (name, a, b, c, d) = match (left, right) {
                                (
                                    ENode::Add {
                                        left: a, right: b, ..
                                    },
                                    ENode::Add {
                                        left: c, right: d, ..
                                    },
                                ) => ("Add", a, b, c, d),
                                (
                                    ENode::Multiply {
                                        left: a, right: b, ..
                                    },
                                    ENode::Multiply {
                                        left: c, right: d, ..
                                    },
                                ) => ("Multiply", a, b, c, d),
                                _ => continue,
                            };
                            for (common, x, y) in [
                                (a == c, (a, b), d),
                                (a == d, (a, b), c),
                                (b == c, (b, a), d),
                                (b == d, (b, a), c),
                            ] {
                                if common {
                                    push(Form::binary(
                                        name,
                                        ty,
                                        Form::Class(x.0),
                                        Form::Select(
                                            ty,
                                            Box::new(Form::Class(predicate)),
                                            Box::new(Form::Class(x.1)),
                                            Box::new(Form::Class(y)),
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    proposals
}

fn take_step(
    budget: &mut AlgebraicWorkBudget,
    function: usize,
    class: u32,
    steps: &mut usize,
) -> bool {
    let Some(remaining) = steps.checked_sub(1) else {
        return false;
    };
    *steps = remaining;
    budget.reserve((function, class), 1)
}
