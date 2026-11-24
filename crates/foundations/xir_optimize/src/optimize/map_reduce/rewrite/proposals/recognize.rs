// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded recognition of polynomial witnesses already present in an e-class.

use super::super::ExprTree;
use super::super::egraph::{Form, Snapshot};
use super::super::extraction::ENode;
use super::factor::Polynomial;
use crate::optimize::AlgebraicProofBudget;
use algebraic_real::AlgebraicReal;

pub(super) fn contains(
    snapshot: &Snapshot,
    coefficients: &[ExprTree],
    class: u32,
    form: &Form,
    budget: &mut AlgebraicProofBudget<'_>,
) -> Option<bool> {
    if !budget.reserve(1) {
        return None;
    }
    if let Form::Class(atom) = form {
        return Some(*atom == class);
    }
    for &node in &snapshot.classes[class as usize] {
        if !budget.reserve(1) {
            return None;
        }
        let matched = match (form, node) {
            (Form::Coefficient(ty, values), ENode::Coefficient(id))
                if *ty == snapshot.types[class as usize] =>
            {
                let ExprTree::Coefficient {
                    values: present, ..
                } = &coefficients[id]
                else {
                    unreachable!()
                };
                if !budget.reserve(values.len().saturating_add(present.len())) {
                    return None;
                }
                values.as_slice() == present.as_ref()
            }
            (Form::Binary(name, ty, left, right), node)
                if *ty == snapshot.types[class as usize] =>
            {
                let children = match (*name, node) {
                    ("Add", ENode::Add { left, right, .. })
                    | ("Multiply", ENode::Multiply { left, right, .. }) => Some((left, right)),
                    _ => None,
                };
                if let Some((a, b)) = children {
                    contains(snapshot, coefficients, a, left, budget)?
                        && contains(snapshot, coefficients, b, right, budget)?
                } else {
                    false
                }
            }
            _ => false,
        };
        if matched {
            return Some(true);
        }
    }
    Some(false)
}

pub(super) fn equal(
    left: &Form,
    right: &Form,
    budget: &mut AlgebraicProofBudget<'_>,
) -> Option<bool> {
    if !budget.reserve(1) {
        return None;
    }
    Some(match (left, right) {
        (Form::Class(left), Form::Class(right)) => left == right,
        (Form::Coefficient(lt, left), Form::Coefficient(rt, right)) => {
            if !budget.reserve(left.len().saturating_add(right.len())) {
                return None;
            }
            lt == rt && left == right
        }
        (Form::Binary(ln, lt, la, lb), Form::Binary(rn, rt, ra, rb)) => {
            ln == rn && lt == rt && equal(la, ra, budget)? && equal(lb, rb, budget)?
        }
        _ => false,
    })
}

/// Recognize the balanced normal form without allocating its tree. Class
/// leaves compare IDs; each inspected constructor and coefficient cell is paid.
pub(super) fn contains_normal_form(
    snapshot: &Snapshot,
    coefficients: &[ExprTree],
    class: u32,
    polynomial: &Polynomial,
    budget: &mut AlgebraicProofBudget<'_>,
) -> Option<bool> {
    contains_canonical(
        snapshot,
        coefficients,
        class,
        snapshot.types[class as usize],
        Canonical::Sum(polynomial, 0, polynomial.len()),
        budget,
    )
}

#[derive(Clone, Copy)]
enum Canonical<'a> {
    Sum(&'a Polynomial, usize, usize),
    Product(&'a [u32], Option<&'a [AlgebraicReal]>),
}

/// Pairwise reduction splits at the largest power of two below the length.
fn split(length: usize) -> usize {
    1 << (length - 1).ilog2()
}

fn contains_canonical(
    snapshot: &Snapshot,
    coefficients: &[ExprTree],
    class: u32,
    ty: usize,
    mut form: Canonical<'_>,
    budget: &mut AlgebraicProofBudget<'_>,
) -> Option<bool> {
    if let Canonical::Sum(polynomial, begin, end) = form
        && end - begin == 1
    {
        let term = &polynomial.terms()[begin];
        if !budget.reserve(term.coefficient().len()) {
            return None;
        }
        let coefficient = (term.factors().is_empty()
            || !term
                .coefficient()
                .iter()
                .all(|value| value == &AlgebraicReal::one()))
        .then_some(term.coefficient());
        form = Canonical::Product(term.factors(), coefficient);
    }
    if let Canonical::Product([atom], None) = form {
        return Some(*atom == class);
    }
    if snapshot.types[class as usize] != ty {
        return Some(false);
    }
    for &node in &snapshot.classes[class as usize] {
        if !budget.reserve(1) {
            return None;
        }
        let children = match (form, node) {
            (Canonical::Sum(_, begin, end), ENode::Coefficient(id)) if begin == end => {
                let ExprTree::Coefficient { values, .. } = &coefficients[id] else {
                    unreachable!()
                };
                if !budget.reserve(values.len()) {
                    return None;
                }
                if values.iter().all(AlgebraicReal::is_zero) {
                    return Some(true);
                }
                None
            }
            (Canonical::Product([], Some(values)), ENode::Coefficient(id)) => {
                let ExprTree::Coefficient {
                    values: present, ..
                } = &coefficients[id]
                else {
                    unreachable!()
                };
                if !budget.reserve(values.len().saturating_add(present.len())) {
                    return None;
                }
                if values == present.as_ref() {
                    return Some(true);
                }
                None
            }
            (Canonical::Sum(polynomial, begin, end), ENode::Add { left, right, .. })
                if end - begin > 1 =>
            {
                let middle = begin + split(end - begin);
                Some((
                    left,
                    right,
                    Canonical::Sum(polynomial, begin, middle),
                    Canonical::Sum(polynomial, middle, end),
                ))
            }
            (Canonical::Product(factors, coefficient), ENode::Multiply { left, right, .. })
                if factors.len() + usize::from(coefficient.is_some()) > 1 =>
            {
                let (a, b) =
                    factors.split_at(split(factors.len() + usize::from(coefficient.is_some())));
                Some((
                    left,
                    right,
                    Canonical::Product(a, None),
                    Canonical::Product(b, coefficient),
                ))
            }
            _ => None,
        };
        if let Some((left, right, a, b)) = children
            && contains_canonical(snapshot, coefficients, left, ty, a, budget)?
            && contains_canonical(snapshot, coefficients, right, ty, b, budget)?
        {
            return Some(true);
        }
    }
    Some(false)
}
