// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded polynomial witnesses over canonical, typed e-class atoms.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use algebraic_real::AlgebraicReal;

use super::super::ExprTree;
use super::super::egraph::{Form, Proposal, Snapshot};
use super::super::extraction::ENode;
use super::recognize;
use crate::optimize::exact_polynomial::ExactPolynomial;
use crate::optimize::{AlgebraicProofBudget, AlgebraicWorkBudget};
use crate::shape::Shape;

pub(super) type Polynomial = ExactPolynomial<u32>;
const MAX_POLYNOMIAL_SIZE: usize = 4096;
const MAX_PROPOSALS: usize = 2048;

const MAX_DEPTH: usize = 128;

pub(in crate::optimize::map_reduce::rewrite) fn proposals(
    snapshot: &Snapshot,
    shapes: &[Shape],
    coefficients: &[ExprTree],
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Vec<Proposal> {
    let mut expansion = Expansion {
        snapshot,
        shapes,
        coefficients,
        max_terms: max_terms.min(MAX_POLYNOMIAL_SIZE),
        active: vec![false; snapshot.classes.len()],
        cache: vec![None; snapshot.classes.len()],
    };
    // Keep only checked scopes. Larger normal forms have priority, but work
    // spent on later classes cannot revoke a completed witness's reserved slot.
    let mut scopes = BTreeMap::new();
    for class in 0..snapshot.classes.len() as u32 {
        if budget.remaining == 0 {
            break;
        }
        let ty = snapshot.types[class as usize];
        let shape = &shapes[ty];
        if !shape.dtype.is_float() || shape.num_elements().is_none() {
            continue;
        }
        let polynomial = expansion.expand(class, 0, &mut budget.begin_attempt());
        let Some(polynomial) = polynomial.filter(|p| !is_atom(p, class)) else {
            continue;
        };
        let elements = shape.num_elements().expect("expanded shape is finite");
        let priority = (Reverse(materialization_work(&polynomial, elements)), class);
        if scopes.len() == MAX_PROPOSALS
            && scopes
                .last_key_value()
                .is_some_and(|(last, _)| priority >= *last)
        {
            continue;
        }
        let schedules = Schedules::new(&polynomial, ty, elements, budget);
        let mut scope = FactorScope {
            class,
            polynomial,
            elements,
            schedules,
            scalar_pending: true,
            factors: Vec::new(),
            flat: None,
        };
        if let Some(form) = scope.next_factor(snapshot, coefficients, expansion.max_terms, budget) {
            scope.factors.push(form);
        } else {
            scope.flat = scope.build_normal_form(snapshot, coefficients, budget);
            if scope.flat.is_none() {
                continue;
            }
        }
        scopes.insert(priority, scope);
        if scopes.len() > MAX_PROPOSALS {
            scopes.pop_last();
        }
    }
    let mut scopes = scopes.into_values().collect::<Vec<_>>();
    // Every retained scope already owns its first slot, including scopes whose
    // only witness is flat. Extra factor variants share the remaining slots.
    let mut slots = scopes.len();
    loop {
        let before = slots;
        for scope in &mut scopes {
            if slots >= MAX_PROPOSALS || budget.remaining == 0 {
                break;
            }
            if let Some(form) =
                scope.next_factor(snapshot, coefficients, expansion.max_terms, budget)
            {
                scope.factors.push(form);
                slots += 1;
            }
        }
        if slots == before || slots >= MAX_PROPOSALS || budget.remaining == 0 {
            break;
        }
    }
    for scope in &mut scopes {
        if slots >= MAX_PROPOSALS || budget.remaining == 0 {
            break;
        }
        if scope.flat.is_none() {
            scope.flat = scope.build_normal_form(snapshot, coefficients, budget);
            slots += usize::from(scope.flat.is_some());
        }
    }
    let mut factors = Vec::new();
    let mut flats = Vec::new();
    for (rank, scope) in scopes.into_iter().enumerate() {
        for (round, form) in scope.factors.into_iter().enumerate() {
            factors.push((
                round,
                rank,
                Proposal {
                    class: scope.class,
                    form,
                },
            ));
        }
        if let Some(form) = scope.flat {
            flats.push(Proposal {
                class: scope.class,
                form,
            });
        }
    }
    factors.sort_unstable_by_key(|(round, rank, _)| (*round, *rank));
    factors
        .into_iter()
        .map(|(_, _, proposal)| proposal)
        .chain(flats)
        .collect()
}

/// Each visit can admit one fully checked witness. Later visits cannot consume
/// the work or proposal slot needed to retain an already completed proof.
struct FactorScope {
    class: u32,
    polynomial: Arc<Polynomial>,
    elements: usize,
    schedules: Schedules,
    scalar_pending: bool,
    factors: Vec<Form>,
    flat: Option<Form>,
}

impl FactorScope {
    fn next_factor(
        &mut self,
        snapshot: &Snapshot,
        coefficients: &[ExprTree],
        max_terms: usize,
        budget: &mut AlgebraicWorkBudget,
    ) -> Option<Form> {
        let ty = snapshot.types[self.class as usize];
        while !self.schedules.exhausted() {
            let mut attempt = budget.begin_attempt();
            if let Some(witness) = self.schedules.next(&self.polynomial, &mut attempt)
                && novel_witness(
                    snapshot,
                    coefficients,
                    self.class,
                    &witness.form,
                    &self.factors,
                    &mut attempt,
                )
            {
                return Some(witness.form);
            }
        }
        if std::mem::take(&mut self.scalar_pending) && self.polynomial.len() >= 2 {
            let mut attempt = budget.begin_attempt();
            if let Some(form) =
                factored_form(&self.polynomial, ty, self.elements, max_terms, &mut attempt)
                && novel_witness(
                    snapshot,
                    coefficients,
                    self.class,
                    &form,
                    &self.factors,
                    &mut attempt,
                )
            {
                return Some(form);
            }
        }
        None
    }

    fn build_normal_form(
        &self,
        snapshot: &Snapshot,
        coefficients: &[ExprTree],
        budget: &mut AlgebraicWorkBudget,
    ) -> Option<Form> {
        let mut attempt = budget.begin_attempt();
        if recognize::contains_normal_form(
            snapshot,
            coefficients,
            self.class,
            &self.polynomial,
            &mut attempt,
        ) != Some(false)
        {
            return None;
        }
        let work = materialization_work(&self.polynomial, self.elements);
        if work > MAX_FORM_WORK || !attempt.reserve(work) {
            return None;
        }
        Some(materialize(
            &self.polynomial,
            snapshot.types[self.class as usize],
            self.elements,
        ))
    }
}

fn novel_witness(
    snapshot: &Snapshot,
    coefficients: &[ExprTree],
    class: u32,
    form: &Form,
    retained: &[Form],
    budget: &mut AlgebraicProofBudget<'_>,
) -> bool {
    recognize::contains(snapshot, coefficients, class, form, budget) == Some(false)
        && retained
            .iter()
            .all(|present| recognize::equal(present, form, budget) == Some(false))
}

fn atom(polynomial: &Polynomial) -> Option<u32> {
    let [term] = polynomial.terms() else {
        return None;
    };
    let [class] = term.factors() else {
        return None;
    };
    term.coefficient()
        .iter()
        .all(|x| x == &AlgebraicReal::one())
        .then_some(*class)
}

fn is_atom(polynomial: &Polynomial, class: u32) -> bool {
    atom(polynomial) == Some(class)
}

fn factored_form(
    polynomial: &Polynomial,
    ty: usize,
    elements: usize,
    max_terms: usize,
    budget: &mut AlgebraicProofBudget<'_>,
) -> Option<Form> {
    let cells = polynomial.terms().iter().fold(0usize, |count, term| {
        count.saturating_add(term.coefficient().len())
    });
    // Pivot support compares every coefficient against every candidate.
    if !budget.reserve(cells.saturating_mul(cells.saturating_add(1))) {
        return None;
    }
    let factor = polynomial.common_scalar_factor_with_budget(&mut |cost| budget.reserve(cost))?;
    if !budget.reserve(factor.estimated_work()) {
        return None;
    }
    let inverse = factor.recip()?;
    if !budget.reserve(elements.saturating_add(1)) {
        return None;
    }
    let constant = Polynomial::constant(vec![inverse; elements], max_terms)?;
    let quotient = polynomial.multiply(&constant, max_terms, &mut |cost| budget.reserve(cost))?;
    let work = materialization_work(&quotient, elements)
        .saturating_add(elements)
        .saturating_add(2);
    if work > MAX_FORM_WORK || !budget.reserve(work) {
        return None;
    }
    Some(Form::binary(
        "Multiply",
        ty,
        Form::Coefficient(ty, vec![factor; elements]),
        materialize(&quotient, ty, elements),
    ))
}

struct Expansion<'a> {
    snapshot: &'a Snapshot,
    shapes: &'a [Shape],
    coefficients: &'a [ExprTree],
    max_terms: usize,
    active: Vec<bool>,
    cache: Vec<Option<Arc<Polynomial>>>,
}

impl Expansion<'_> {
    fn expand(
        &mut self,
        class: u32,
        depth: usize,
        budget: &mut AlgebraicProofBudget<'_>,
    ) -> Option<Arc<Polynomial>> {
        if let Some(polynomial) = &self.cache[class as usize] {
            return Some(Arc::clone(polynomial));
        }
        if depth >= MAX_DEPTH || self.active[class as usize] || !budget.reserve(1) {
            return None;
        }
        let shape = &self.shapes[self.snapshot.types[class as usize]];
        let elements = shape.num_elements()?;
        if !shape.dtype.is_float() || elements > self.max_terms {
            return None;
        }
        self.active[class as usize] = true;
        let result = self.expand_node(class, depth, elements, budget);
        self.active[class as usize] = false;
        if budget.failed() {
            return None;
        }
        let result = result.or_else(|| {
            budget
                .reserve(elements.saturating_add(1))
                .then(|| Polynomial::variable(class, elements, self.max_terms))
                .flatten()
        })?;
        let result = Arc::new(result);
        self.cache[class as usize] = Some(Arc::clone(&result));
        Some(result)
    }

    fn expand_node(
        &mut self,
        class: u32,
        depth: usize,
        elements: usize,
        budget: &mut AlgebraicProofBudget<'_>,
    ) -> Option<Polynomial> {
        for &node in &self.snapshot.classes[class as usize] {
            if let ENode::Coefficient(id) = node {
                let ExprTree::Coefficient { values, .. } = &self.coefficients[id] else {
                    unreachable!()
                };
                if !budget.reserve(elements.saturating_add(1)) {
                    return None;
                }
                return Polynomial::constant(values.to_vec(), self.max_terms);
            }
        }
        for index in 0..self.snapshot.classes[class as usize].len() {
            let node = self.snapshot.classes[class as usize][index];
            let result = match node {
                ENode::Negate { value, .. } => self
                    .expand(value, depth + 1, budget)
                    .and_then(|polynomial| polynomial.negate(&mut |cost| budget.reserve(cost))),
                ENode::Add { left, right, .. } | ENode::Multiply { left, right, .. } => {
                    let left = self.expand(left, depth + 1, budget);
                    let right = if budget.failed() {
                        None
                    } else {
                        self.expand(right, depth + 1, budget)
                    };
                    match (left, right) {
                        (Some(left), Some(right)) => {
                            let multiply = matches!(node, ENode::Multiply { .. });
                            if multiply && left.len().saturating_mul(right.len()) > self.max_terms {
                                None
                            } else if multiply {
                                left.multiply(&right, self.max_terms, &mut |cost| {
                                    budget.reserve(cost)
                                })
                            } else {
                                left.add(&right, self.max_terms, &mut |cost| budget.reserve(cost))
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if result.is_some() || budget.failed() {
                return result;
            }
        }
        None
    }
}

fn storage_work(polynomial: &Polynomial) -> usize {
    polynomial.terms().iter().fold(0usize, |work, term| {
        work.saturating_add(1)
            .saturating_add(term.factors().len())
            .saturating_add(term.coefficient().len())
    })
}

/// Reserve tree nodes and coefficient cells before allocating the graph form.
fn materialization_work(polynomial: &Polynomial, elements: usize) -> usize {
    if polynomial.terms().is_empty() {
        return elements.saturating_add(1);
    }
    polynomial
        .terms()
        .iter()
        .fold(polynomial.len() - 1, |work, term| {
            let coefficient = term.factors().is_empty()
                || !term
                    .coefficient()
                    .iter()
                    .all(|x| x == &AlgebraicReal::one());
            let factors = term.factors().len() + usize::from(coefficient);
            work.saturating_add(factors)
                .saturating_add(factors - 1)
                .saturating_add(if coefficient {
                    term.coefficient().len()
                } else {
                    0
                })
        })
}

fn materialize(polynomial: &Polynomial, ty: usize, elements: usize) -> Form {
    let mut terms = Vec::new();
    for term in polynomial.terms() {
        let mut factors = term
            .factors()
            .iter()
            .copied()
            .map(Form::Class)
            .collect::<Vec<_>>();
        if factors.is_empty()
            || !term
                .coefficient()
                .iter()
                .all(|x| x == &AlgebraicReal::one())
        {
            factors.push(Form::Coefficient(ty, term.coefficient().to_vec()));
        }
        while factors.len() > 1 {
            let mut next = Vec::with_capacity(factors.len().div_ceil(2));
            let mut iter = factors.into_iter();
            while let Some(left) = iter.next() {
                next.push(match iter.next() {
                    Some(right) => Form::binary("Multiply", ty, left, right),
                    None => left,
                });
            }
            factors = next;
        }
        terms.push(factors.pop().unwrap());
    }
    if terms.is_empty() {
        return Form::Coefficient(ty, vec![AlgebraicReal::zero(); elements]);
    }
    while terms.len() > 1 {
        let mut next = Vec::new();
        let mut iter = terms.into_iter();
        while let Some(left) = iter.next() {
            next.push(match iter.next() {
                Some(right) => Form::binary("Add", ty, left, right),
                None => left,
            });
        }
        terms = next;
    }
    terms.pop().unwrap()
}

const MAX_GREEDY_VARIANTS: usize = 8;
const MAX_PLAN_DEPTH: usize = 128;
pub(super) const MAX_FORM_WORK: usize = 4096;

#[derive(Clone)]
struct Term<'a> {
    factors: Vec<u32>,
    coefficient: &'a [AlgebraicReal],
}

#[derive(Clone, Copy)]
enum Order {
    Support,
    Atom,
}

pub(super) struct Witness {
    pub(super) form: Form,
    pub(super) work: usize,
}

/// A bounded sequence of independent factor proofs. The caller validates and
/// retains each yielded witness before asking for another proof.
pub(super) struct Schedules {
    ty: usize,
    flat_work: usize,
    orders: VecDeque<(u32, Order)>,
}

impl Schedules {
    pub(super) fn new(
        polynomial: &Polynomial,
        ty: usize,
        elements: usize,
        budget: &mut AlgebraicWorkBudget,
    ) -> Self {
        let mut schedules = Self {
            ty,
            flat_work: materialization_work(polynomial, elements),
            orders: VecDeque::new(),
        };
        if polynomial.len() < 2 {
            return schedules;
        }
        let mut attempt = budget.begin_attempt();
        if !attempt.reserve(storage_work(polynomial)) {
            return schedules;
        }
        let mut support = HashMap::<u32, usize>::new();
        for term in polynomial.terms() {
            let mut previous = None;
            for &factor in term.factors() {
                if previous != Some(factor) {
                    *support.entry(factor).or_default() += 1;
                    previous = Some(factor);
                }
            }
        }
        let mut candidates = support
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect::<Vec<_>>();
        let sort_work = candidates
            .len()
            .saturating_mul(candidates.len().checked_ilog2().unwrap_or(0) as usize + 1);
        if !attempt.reserve(sort_work) {
            return schedules;
        }
        candidates.sort_unstable_by_key(|&(factor, count)| (usize::MAX - count, factor));
        if let Some(&(first, _)) = candidates.first() {
            let atom = candidates.iter().map(|&(factor, _)| factor).min().unwrap();
            schedules.orders.extend(
                std::iter::once((first, Order::Support))
                    .chain(std::iter::once((atom, Order::Atom)))
                    .chain(
                        candidates
                            .iter()
                            .take(MAX_GREEDY_VARIANTS)
                            .skip(1)
                            .map(|&(factor, _)| (factor, Order::Support)),
                    ),
            );
        }
        schedules
    }

    pub(super) fn exhausted(&self) -> bool {
        self.orders.is_empty()
    }

    pub(super) fn next(
        &mut self,
        polynomial: &Polynomial,
        budget: &mut AlgebraicProofBudget<'_>,
    ) -> Option<Witness> {
        let (factor, order) = self.orders.pop_front()?;
        if !budget.reserve(storage_work(polynomial)) {
            return None;
        }
        let terms = polynomial
            .terms()
            .iter()
            .map(|term| Term {
                factors: term.factors().to_vec(),
                coefficient: term.coefficient(),
            })
            .collect::<Vec<_>>();
        let mut planner = Planner {
            ty: self.ty,
            budget,
            work: 0,
        };
        let form = planner.split(terms, factor, order, 0)?;
        let witness = Witness {
            form,
            work: planner.work,
        };
        (witness.work < self.flat_work).then_some(witness)
    }
}

#[cfg(test)]
fn witnesses(
    polynomial: &Polynomial,
    ty: usize,
    elements: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Vec<Witness> {
    let mut schedules = Schedules::new(polynomial, ty, elements, budget);
    let mut result = Vec::new();
    while !schedules.exhausted() {
        if let Some(witness) = schedules.next(polynomial, &mut budget.begin_attempt()) {
            result.push(witness);
        }
    }
    result
}

struct Planner<'a, 'b> {
    ty: usize,
    budget: &'a mut AlgebraicProofBudget<'b>,
    work: usize,
}

impl Planner<'_, '_> {
    fn reserve_form(&mut self, work: usize) -> Option<()> {
        let total = self.work.checked_add(work)?;
        if total > MAX_FORM_WORK || !self.budget.reserve(work) {
            return None;
        }
        self.work = total;
        Some(())
    }

    fn class(&mut self, class: u32) -> Option<Form> {
        self.reserve_form(1)?;
        Some(Form::Class(class))
    }

    fn binary(&mut self, name: &'static str, left: Form, right: Form) -> Option<Form> {
        self.reserve_form(1)?;
        Some(Form::binary(name, self.ty, left, right))
    }

    fn combine(&mut self, name: &'static str, mut forms: Vec<Form>) -> Option<Form> {
        while forms.len() > 1 {
            let mut next = Vec::with_capacity(forms.len().div_ceil(2));
            let mut iter = forms.into_iter();
            while let Some(left) = iter.next() {
                next.push(match iter.next() {
                    Some(right) => self.binary(name, left, right)?,
                    None => left,
                });
            }
            forms = next;
        }
        forms.pop()
    }

    fn flat(&mut self, terms: Vec<Term<'_>>) -> Option<Form> {
        let mut sum = Vec::with_capacity(terms.len());
        for term in terms {
            if !self.budget.reserve(term.coefficient.len()) {
                return None;
            }
            let mut factors = Vec::with_capacity(term.factors.len().saturating_add(1));
            for factor in term.factors {
                factors.push(self.class(factor)?);
            }
            if factors.is_empty()
                || !term
                    .coefficient
                    .iter()
                    .all(|value| value == &AlgebraicReal::one())
            {
                self.reserve_form(term.coefficient.len().saturating_add(1))?;
                factors.push(Form::Coefficient(self.ty, term.coefficient.to_vec()));
            }
            sum.push(self.combine("Multiply", factors)?);
        }
        self.combine("Add", sum)
    }

    fn plan(&mut self, terms: Vec<Term<'_>>, order: Order, depth: usize) -> Option<Form> {
        if depth >= MAX_PLAN_DEPTH || terms.len() < 2 {
            return self.flat(terms);
        }
        let scan = terms.iter().fold(terms.len(), |work, term| {
            work.saturating_add(term.factors.len())
        });
        if !self.budget.reserve(scan) {
            return None;
        }
        let mut support = HashMap::<u32, usize>::new();
        for term in &terms {
            let mut previous = None;
            for &factor in &term.factors {
                if previous != Some(factor) {
                    *support.entry(factor).or_default() += 1;
                    previous = Some(factor);
                }
            }
        }
        let mut candidates = support
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .collect::<Vec<_>>();
        let sort_work = candidates
            .len()
            .saturating_mul(candidates.len().checked_ilog2().unwrap_or(0) as usize + 1);
        if !self.budget.reserve(sort_work) {
            return None;
        }
        candidates.sort_unstable_by_key(|&(factor, count)| match order {
            Order::Support => (usize::MAX - count, factor),
            Order::Atom => (0, factor),
        });
        let Some(&(factor, _)) = candidates.first() else {
            return self.flat(terms);
        };
        self.split(terms, factor, order, depth)
    }

    fn split(
        &mut self,
        terms: Vec<Term<'_>>,
        factor: u32,
        order: Order,
        depth: usize,
    ) -> Option<Form> {
        let scan = terms.iter().fold(terms.len(), |work, term| {
            work.saturating_add(term.factors.len())
        });
        if !self.budget.reserve(scan) {
            return None;
        }
        let mut quotient = Vec::new();
        let mut remainder = Vec::new();
        for mut term in terms {
            match term.factors.binary_search(&factor) {
                Ok(position) => {
                    term.factors.remove(position);
                    quotient.push(term);
                }
                Err(_) => remainder.push(term),
            }
        }
        let left = self.class(factor)?;
        let right = self.plan(quotient, order, depth + 1)?;
        let product = self.binary("Multiply", left, right)?;
        if remainder.is_empty() {
            Some(product)
        } else {
            let rest = self.plan(remainder, order, depth + 1)?;
            self.binary("Add", product, rest)
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "factor/proposal_tests.rs"]
mod proposal_tests;
