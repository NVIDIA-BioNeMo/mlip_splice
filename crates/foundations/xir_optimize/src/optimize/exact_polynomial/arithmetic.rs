// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use algebraic_real::AlgebraicReal;

use crate::ir::ValueId;

pub(in crate::optimize) trait PolynomialAtom: Copy + Eq + Hash {
    fn ordinal(self) -> u32;
}

impl PolynomialAtom for ValueId {
    fn ordinal(self) -> u32 {
        self.0
    }
}

impl PolynomialAtom for u32 {
    fn ordinal(self) -> u32 {
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::optimize) struct ExactPolynomial<A = ValueId> {
    terms: Vec<ExactMonomial<A>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::optimize) struct ExactMonomial<A = ValueId> {
    factors: Vec<A>,
    coefficient: Vec<AlgebraicReal>,
}

impl<A: PolynomialAtom> ExactMonomial<A> {
    pub(in crate::optimize) fn new(mut factors: Vec<A>, coefficient: Vec<AlgebraicReal>) -> Self {
        factors.sort_unstable_by_key(|value| value.ordinal());
        Self {
            factors,
            coefficient,
        }
    }

    pub(in crate::optimize) fn factors(&self) -> &[A] {
        &self.factors
    }

    pub(in crate::optimize) fn coefficient(&self) -> &[AlgebraicReal] {
        &self.coefficient
    }

    pub(in crate::optimize) fn into_parts(self) -> (Vec<A>, Vec<AlgebraicReal>) {
        (self.factors, self.coefficient)
    }
}

impl<A: PolynomialAtom> ExactPolynomial<A> {
    pub(in crate::optimize) fn constant(
        coefficient: Vec<AlgebraicReal>,
        max_terms: usize,
    ) -> Option<Self> {
        Self::from_terms(vec![ExactMonomial::new(Vec::new(), coefficient)], max_terms)
    }

    pub(in crate::optimize) fn variable(
        value: A,
        elements: usize,
        max_terms: usize,
    ) -> Option<Self> {
        Self::from_terms(
            vec![ExactMonomial::new(
                vec![value],
                vec![AlgebraicReal::one(); elements],
            )],
            max_terms,
        )
    }

    pub(in crate::optimize) fn from_terms(
        terms: Vec<ExactMonomial<A>>,
        max_terms: usize,
    ) -> Option<Self> {
        Self::from_terms_with_budget(terms, max_terms, &mut |_| true)
    }

    fn from_terms_with_budget(
        terms: Vec<ExactMonomial<A>>,
        max_terms: usize,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        if !charge(terms.len().saturating_mul(2)) {
            return None;
        }
        let mut merged: HashMap<Vec<A>, Vec<AlgebraicReal>> = HashMap::with_capacity(terms.len());
        for term in terms {
            let (factors, coefficient) = term.into_parts();
            match merged.entry(factors) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let accumulator = entry.get_mut();
                    if accumulator.len() != coefficient.len() {
                        return None;
                    }
                    for (accumulator, value) in accumulator.iter_mut().zip(coefficient) {
                        if !charge(
                            accumulator
                                .estimated_work()
                                .saturating_add(value.estimated_work()),
                        ) {
                            return None;
                        }
                        *accumulator = &*accumulator + &value;
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(coefficient);
                }
            }
            if merged.len() > max_terms {
                return None;
            }
        }
        let mut terms = merged
            .into_iter()
            .filter(|(_, coefficient)| coefficient.iter().any(|value| !value.is_zero()))
            .map(|(factors, coefficient)| ExactMonomial::new(factors, coefficient))
            .collect::<Vec<_>>();
        terms.sort_unstable_by(|left, right| {
            left.factors
                .iter()
                .map(|value| value.ordinal())
                .cmp(right.factors.iter().map(|value| value.ordinal()))
        });
        let cost = terms.iter().try_fold(0usize, |cost, term| {
            let coefficient = term.coefficient.iter().try_fold(0usize, |cost, value| {
                cost.checked_add(value.structural_size().max(1))
            })?;
            cost.checked_add(term.factors.len())?
                .checked_add(coefficient)
        })?;
        (terms.len() <= max_terms && cost <= max_terms).then_some(Self { terms })
    }

    pub(in crate::optimize) fn terms(&self) -> &[ExactMonomial<A>] {
        &self.terms
    }

    pub(in crate::optimize) fn len(&self) -> usize {
        self.terms.len()
    }

    pub(in crate::optimize) fn is_constant(&self) -> bool {
        self.terms.iter().all(|term| term.factors.is_empty())
    }

    #[cfg(test)]
    pub(in crate::optimize) fn common_scalar_factor(&self) -> Option<AlgebraicReal> {
        self.common_scalar_factor_with_budget(&mut |_| true)
    }

    pub(in crate::optimize) fn common_scalar_factor_with_budget(
        &self,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<AlgebraicReal> {
        if self.is_constant() {
            return None;
        }
        let coefficients = self
            .terms
            .iter()
            .flat_map(|term| &term.coefficient)
            .filter(|coefficient| !coefficient.is_zero())
            .collect::<Vec<_>>();
        let units = |pivot: &AlgebraicReal| {
            coefficients
                .iter()
                .filter(|coefficient| **coefficient == pivot)
                .count()
        };
        let mut best = (AlgebraicReal::one(), units(&AlgebraicReal::one()));
        for &pivot in &coefficients {
            let count = units(pivot);
            if count <= best.1 {
                continue;
            }
            let mut rational_line = true;
            for coefficient in &coefficients {
                if !charge(
                    coefficient
                        .estimated_work()
                        .saturating_mul(pivot.estimated_work()),
                ) {
                    return None;
                }
                if (*coefficient / pivot).as_fraction().is_none() {
                    rational_line = false;
                    break;
                }
            }
            if !rational_line {
                continue;
            }
            best = (pivot.clone(), count);
        }
        (best.0 != AlgebraicReal::one()).then_some(best.0)
    }
}

fn merge_sorted<A: PolynomialAtom>(left: &[A], right: &[A]) -> Vec<A> {
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        if left[left_index].ordinal() <= right[right_index].ordinal() {
            merged.push(left[left_index]);
            left_index += 1;
        } else {
            merged.push(right[right_index]);
            right_index += 1;
        }
    }
    merged.extend_from_slice(&left[left_index..]);
    merged.extend_from_slice(&right[right_index..]);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(factors: &[u32], coefficient: AlgebraicReal) -> ExactMonomial {
        ExactMonomial::new(
            factors.iter().copied().map(ValueId).collect(),
            vec![coefficient],
        )
    }

    fn polynomial(terms: Vec<ExactMonomial>) -> ExactPolynomial {
        ExactPolynomial::from_terms(terms, 100).unwrap()
    }

    #[test]
    fn maintains_canonical_form() {
        let value = polynomial(vec![
            term(&[2, 1], AlgebraicReal::integer(3)),
            term(&[1, 2], -&AlgebraicReal::one()),
            term(&[3], AlgebraicReal::zero()),
        ]);
        assert_eq!(value.terms.len(), 1);
        assert_eq!(value.terms[0].factors, [ValueId(1), ValueId(2)]);
        assert_eq!(value.terms[0].coefficient, [AlgebraicReal::integer(2)]);
    }

    #[test]
    fn finds_the_scalar_that_leaves_the_most_unit_coefficients() {
        let root = AlgebraicReal::sqrt(3u32);
        let value = polynomial(vec![
            term(&[1], root.clone()),
            term(&[2], &root * &AlgebraicReal::integer(3)),
        ]);
        assert_eq!(value.common_scalar_factor(), Some(root));
    }

    #[test]
    fn leaves_coefficients_on_different_rational_lines_unfactored() {
        let value = polynomial(vec![
            term(&[1], AlgebraicReal::sqrt(2u32)),
            term(&[2], AlgebraicReal::sqrt(3u32)),
        ]);
        assert!(value.common_scalar_factor().is_none());
    }

    #[test]
    fn leaves_a_polynomial_with_unit_coefficients_unfactored() {
        let value = polynomial(vec![
            term(&[1], AlgebraicReal::one()),
            term(&[2], AlgebraicReal::integer(3)),
        ]);
        assert!(value.common_scalar_factor().is_none());
    }
}

impl<A: PolynomialAtom> ExactPolynomial<A> {
    pub(in crate::optimize) fn add(
        &self,
        right: &Self,
        max_terms: usize,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        self.merge_sum(right, false, max_terms, charge)
    }

    pub(in crate::optimize) fn subtract(
        &self,
        right: &Self,
        max_terms: usize,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        self.merge_sum(right, true, max_terms, charge)
    }

    pub(in crate::optimize) fn negate(
        &self,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        Self { terms: Vec::new() }.merge_sum(self, true, usize::MAX, charge)
    }

    fn merge_sum(
        &self,
        right: &Self,
        negate_right: bool,
        max_terms: usize,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        let capacity = self.len().checked_add(right.len())?;
        if !charge(capacity) {
            return None;
        }
        let mut terms = Vec::with_capacity(capacity);
        let (mut left_index, mut right_index) = (0, 0);
        let mut size = 0usize;
        while left_index < self.len() || right_index < right.len() {
            let left = self.terms.get(left_index);
            let right = right.terms.get(right_index);
            let order = match (left, right) {
                (Some(left), Some(right)) => {
                    if !charge(
                        left.factors
                            .len()
                            .min(right.factors.len())
                            .saturating_add(1),
                    ) {
                        return None;
                    }
                    left.factors
                        .iter()
                        .map(|a| a.ordinal())
                        .cmp(right.factors.iter().map(|a| a.ordinal()))
                }
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => unreachable!(),
            };
            let (term, other, negate) = match order {
                Ordering::Less => {
                    left_index += 1;
                    (left?, None, false)
                }
                Ordering::Greater => {
                    right_index += 1;
                    (right?, None, negate_right)
                }
                Ordering::Equal => {
                    left_index += 1;
                    right_index += 1;
                    (left?, right, false)
                }
            };
            if other.is_some_and(|other| other.coefficient.len() != term.coefficient.len())
                || !charge(term.coefficient.len())
            {
                return None;
            }
            let mut coefficient = Vec::with_capacity(term.coefficient.len());
            for (index, value) in term.coefficient.iter().enumerate() {
                let value = if let Some(other) = other {
                    let right = &other.coefficient[index];
                    if !charge(
                        value
                            .estimated_work()
                            .saturating_add(right.estimated_work()),
                    ) {
                        return None;
                    }
                    if negate_right {
                        value - right
                    } else {
                        value + right
                    }
                } else if negate {
                    if !charge(value.estimated_work()) {
                        return None;
                    }
                    -value
                } else {
                    if !charge(1) || !charge(value.structural_size().max(1)) {
                        return None;
                    }
                    value.clone()
                };
                coefficient.push(value);
            }
            if coefficient.iter().all(|value| value.is_zero()) {
                continue;
            }
            size = size.checked_add(term.factors.len())?;
            for value in &coefficient {
                if !charge(1) {
                    return None;
                }
                size = size.checked_add(value.structural_size().max(1))?;
            }
            if size > max_terms || terms.len() == max_terms || !charge(term.factors.len()) {
                return None;
            }
            terms.push(ExactMonomial {
                factors: term.factors.clone(),
                coefficient,
            });
        }
        Some(Self { terms })
    }

    pub(in crate::optimize) fn multiply(
        &self,
        right: &Self,
        max_terms: usize,
        charge: &mut impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        let term_count = self.len().checked_mul(right.len())?;
        if term_count > max_terms || !charge(term_count) {
            return None;
        }
        let mut terms = Vec::with_capacity(term_count);
        for left_term in &self.terms {
            for right_term in &right.terms {
                if left_term.coefficient.len() != right_term.coefficient.len()
                    || !charge(left_term.coefficient.len())
                {
                    return None;
                }
                let mut coefficient = Vec::with_capacity(left_term.coefficient.len());
                for (left, right) in left_term.coefficient.iter().zip(&right_term.coefficient) {
                    if !charge(left.estimated_work().saturating_mul(right.estimated_work())) {
                        return None;
                    }
                    coefficient.push(left * right);
                }
                if !charge(
                    left_term
                        .factors
                        .len()
                        .saturating_add(right_term.factors.len()),
                ) {
                    return None;
                }
                terms.push(ExactMonomial {
                    factors: merge_sorted(&left_term.factors, &right_term.factors),
                    coefficient,
                });
            }
        }
        Self::from_terms_with_budget(terms, max_terms, charge)
    }
}

#[cfg(test)]
#[path = "arithmetic/operations_tests.rs"]
mod operations_tests;
