// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::sync::Arc;

use super::arithmetic::ExactPolynomial;
use crate::analysis;
use crate::ir::{Def, Func, ValueId};
use crate::op::Op;
use crate::ops::ScalarOp;

pub(in crate::optimize) struct PolynomialExpansion {
    polynomial: ExactPolynomial,
    has_constant: bool,
    expanded_terms: usize,
}

impl PolynomialExpansion {
    pub(super) fn polynomial(&self) -> &ExactPolynomial {
        &self.polynomial
    }

    pub(super) fn has_constant(&self) -> bool {
        self.has_constant
    }
}

pub(super) fn polynomial_roots(func: &Func) -> Vec<bool> {
    let mut roots = vec![false; func.values.len()];
    for &result in &func.results {
        roots[result.index()] = true;
    }
    for node in &func.nodes {
        if !is_polynomial_op(node.op.as_ref()) {
            for &input in &node.inputs {
                roots[input.index()] = true;
            }
        }
    }
    roots
}

pub(super) fn is_polynomial_op(op: &dyn Op) -> bool {
    op.as_elementwise().is_some_and(|elementwise| {
        matches!(
            elementwise.scalar(),
            ScalarOp::Add | ScalarOp::Subtract | ScalarOp::Multiply | ScalarOp::Negate
        )
    })
}

pub(super) enum BoundedExpansion {
    Available(Arc<PolynomialExpansion>),
    Unavailable,
    Exhausted,
}

enum CachedExpansion {
    Unknown,
    Unavailable,
    Available(Arc<PolynomialExpansion>),
}

pub(super) struct BoundedPolynomialExpander<'a> {
    func: &'a Func,
    max_terms: usize,
    cache: Vec<CachedExpansion>,
}

impl<'a> BoundedPolynomialExpander<'a> {
    pub(super) fn new(func: &'a Func, max_terms: usize) -> Self {
        Self {
            func,
            max_terms,
            cache: (0..func.values.len())
                .map(|_| CachedExpansion::Unknown)
                .collect(),
        }
    }

    pub(super) fn expand(
        &mut self,
        value: ValueId,
        work: &mut crate::optimize::AlgebraicWorkBudget,
    ) -> BoundedExpansion {
        enum Frame {
            Visit(ValueId),
            Apply(ValueId, ScalarOp, Vec<ValueId>),
        }

        let mut frames = vec![Frame::Visit(value)];
        while let Some(frame) = frames.pop() {
            let mut budget = work.begin_attempt();
            if !budget.reserve(1) {
                return BoundedExpansion::Exhausted;
            }
            match frame {
                Frame::Visit(value) => {
                    if !matches!(self.cache[value.index()], CachedExpansion::Unknown) {
                        continue;
                    }
                    let shape = self.func.value_shape(value);
                    if !shape.dtype.is_float() {
                        self.cache[value.index()] = CachedExpansion::Unavailable;
                        continue;
                    }
                    if let Some(literal) = analysis::constant_literal(self.func, value) {
                        let Some(expansion) =
                            constant_expansion_bounded(shape, literal, self.max_terms, &mut budget)
                        else {
                            if budget.failed() {
                                return BoundedExpansion::Exhausted;
                            }
                            self.cache[value.index()] = CachedExpansion::Unavailable;
                            continue;
                        };
                        self.cache[value.index()] = CachedExpansion::Available(Arc::new(expansion));
                        continue;
                    }
                    let polynomial_node = match self.func.values[value.index()].def {
                        Def::NodeResult { node, slot: 0 } => {
                            let node = &self.func.nodes[node];
                            (node.results.len() == 1 && is_polynomial_op(node.op.as_ref()))
                                .then_some(node)
                        }
                        _ => None,
                    };
                    if let Some(node) = polynomial_node {
                        if !budget.reserve(node.inputs.len().saturating_mul(2)) {
                            return BoundedExpansion::Exhausted;
                        }
                        let scalar = node.op.as_elementwise().unwrap().scalar();
                        let inputs = node.inputs.to_vec();
                        frames.push(Frame::Apply(value, scalar, inputs.clone()));
                        frames.extend(inputs.into_iter().rev().map(Frame::Visit));
                    } else {
                        let Some(expansion) = variable_expansion_bounded(
                            value,
                            shape.num_elements(),
                            self.max_terms,
                            &mut budget,
                        ) else {
                            if budget.failed() {
                                return BoundedExpansion::Exhausted;
                            }
                            self.cache[value.index()] = CachedExpansion::Unavailable;
                            continue;
                        };
                        self.cache[value.index()] = CachedExpansion::Available(Arc::new(expansion));
                    }
                }
                Frame::Apply(value, op, inputs) => {
                    if !budget.reserve(inputs.len()) {
                        return BoundedExpansion::Exhausted;
                    }
                    let mut operands = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        match &self.cache[input.index()] {
                            CachedExpansion::Available(expansion) => {
                                operands.push(expansion.as_ref());
                            }
                            CachedExpansion::Unavailable => {
                                operands.clear();
                                break;
                            }
                            CachedExpansion::Unknown => {
                                unreachable!("postorder expansion visits inputs first")
                            }
                        }
                    }
                    if operands.is_empty() {
                        self.cache[value.index()] = CachedExpansion::Unavailable;
                        continue;
                    }
                    let Some(expansion) = apply(op, &operands, self.max_terms, &mut |cost| {
                        budget.reserve(cost)
                    }) else {
                        if budget.failed() {
                            return BoundedExpansion::Exhausted;
                        }
                        self.cache[value.index()] = CachedExpansion::Unavailable;
                        continue;
                    };
                    self.cache[value.index()] = CachedExpansion::Available(Arc::new(expansion));
                }
            }
        }
        match &self.cache[value.index()] {
            CachedExpansion::Available(expansion) => {
                let mut budget = work.begin_attempt();
                if budget.reserve(1) {
                    BoundedExpansion::Available(Arc::clone(expansion))
                } else {
                    BoundedExpansion::Exhausted
                }
            }
            CachedExpansion::Unavailable => BoundedExpansion::Unavailable,
            CachedExpansion::Unknown => unreachable!("requested expansion reached a cache state"),
        }
    }
}

#[cfg(test)]
pub(super) fn full_expansion(
    func: &Func,
    value: ValueId,
    max_terms: usize,
) -> Option<Arc<PolynomialExpansion>> {
    let mut expander = BoundedPolynomialExpander::new(func, max_terms);
    let mut work = crate::optimize::AlgebraicWorkBudget::new(usize::MAX, usize::MAX);
    match expander.expand(value, &mut work) {
        BoundedExpansion::Available(expansion) => Some(expansion),
        BoundedExpansion::Unavailable | BoundedExpansion::Exhausted => None,
    }
}

fn constant_expansion_bounded(
    shape: &crate::shape::Shape,
    literal: &crate::ops::Literal,
    max_terms: usize,
    budget: &mut crate::optimize::AlgebraicProofBudget<'_>,
) -> Option<PolynomialExpansion> {
    let elements = shape.num_elements()?;
    let coefficient_work = match &literal.exact {
        Some(exact) if exact.len() == elements => {
            if !budget.reserve(elements) {
                return None;
            }
            let mut work = 0usize;
            for value in exact {
                if !budget.reserve(1) {
                    return None;
                }
                work = work.saturating_add(value.structural_size().max(1));
            }
            work
        }
        Some(_) => return None,
        None if literal.data.len() == shape.byte_size().ok()? => elements,
        None => return None,
    };
    if coefficient_work.saturating_add(1) > max_terms
        || !budget.reserve(coefficient_work.saturating_add(elements).saturating_add(1))
    {
        return None;
    }
    let exact = literal.algebraic_values()?;
    Some(PolynomialExpansion {
        polynomial: ExactPolynomial::constant(exact, max_terms)?,
        has_constant: true,
        expanded_terms: 1,
    })
}

fn variable_expansion_bounded(
    value: ValueId,
    elements: Option<usize>,
    max_terms: usize,
    budget: &mut crate::optimize::AlgebraicProofBudget<'_>,
) -> Option<PolynomialExpansion> {
    let elements = elements?;
    if !budget.reserve(elements.saturating_add(2)) {
        return None;
    }
    Some(PolynomialExpansion {
        polynomial: ExactPolynomial::variable(value, elements, max_terms)?,
        has_constant: false,
        expanded_terms: 1,
    })
}

fn apply(
    op: ScalarOp,
    operands: &[impl Borrow<PolynomialExpansion>],
    max_terms: usize,
    charge: &mut impl FnMut(usize) -> bool,
) -> Option<PolynomialExpansion> {
    let left = operands.first()?.borrow();
    let has_constant = operands.iter().any(|operand| operand.borrow().has_constant);
    let (polynomial, expanded_terms) = match op {
        ScalarOp::Add | ScalarOp::Subtract | ScalarOp::Multiply => {
            let right = operands.get(1)?.borrow();
            let expanded_terms = if op == ScalarOp::Multiply {
                left.expanded_terms.checked_mul(right.expanded_terms)?
            } else {
                left.expanded_terms.checked_add(right.expanded_terms)?
            };
            if expanded_terms > max_terms {
                return None;
            }
            let polynomial = match op {
                ScalarOp::Add => left.polynomial.add(&right.polynomial, max_terms, charge)?,
                ScalarOp::Subtract => {
                    left.polynomial
                        .subtract(&right.polynomial, max_terms, charge)?
                }
                ScalarOp::Multiply => {
                    left.polynomial
                        .multiply(&right.polynomial, max_terms, charge)?
                }
                _ => unreachable!(),
            };
            (polynomial, expanded_terms)
        }
        ScalarOp::Negate => (left.polynomial.negate(charge)?, left.expanded_terms),
        _ => return None,
    };
    Some(PolynomialExpansion {
        polynomial,
        has_constant,
        expanded_terms,
    })
}

#[cfg(test)]
mod tests;
