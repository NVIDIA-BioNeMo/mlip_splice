// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::algebraic_real::AlgebraicReal;
use crate::canonical::{self, select_root};
use crate::multiquadratic::Multiquadratic;
use crate::number_field::NumberFieldElement;
use crate::polynomial::Polynomial;
use crate::representation::Representation;
use crate::root_composition;
use crate::selected_root::SelectedRoot;

const MAX_LAZY_COMPOSITION_DEGREE: usize = 256;

impl Neg for &AlgebraicReal {
    type Output = AlgebraicReal;
    fn neg(self) -> Self::Output {
        match &self.representation {
            Representation::NumberField(value) => AlgebraicReal::number_field(value.neg()),
            Representation::IsolatedRoot(_) => {
                let mut polynomial = self.annihilator();
                for (index, coefficient) in polynomial.iter_mut().enumerate() {
                    if index % 2 == 1 {
                        *coefficient = -&*coefficient;
                    }
                }
                select_root(polynomial, |bits| self.interval(bits).neg())
                    .expect("negation has a real algebraic result")
            }
            Representation::Multiquadratic(value) => AlgebraicReal::multiquadratic(-value),
        }
    }
}
impl Neg for AlgebraicReal {
    type Output = Self;
    fn neg(self) -> Self {
        -&self
    }
}

fn element_in_field(
    value: &AlgebraicReal,
    field: &Arc<SelectedRoot>,
) -> Option<NumberFieldElement> {
    match &value.representation {
        Representation::NumberField(element)
            if Arc::ptr_eq(element.field(), field) || element.field().same_embedding(field) =>
        {
            Some(element.clone())
        }
        Representation::IsolatedRoot(root) => canonical::cached_selected_root(root)
            .filter(|root| Arc::ptr_eq(root, field) || root.same_embedding(field))
            .map(NumberFieldElement::generator),
        Representation::Multiquadratic(value) => value
            .rational_value()
            .map(|constant| NumberFieldElement::constant(field.clone(), constant)),
        Representation::NumberField(_) => None,
    }
}

fn shared_number_field_operands(
    left: &AlgebraicReal,
    right: &AlgebraicReal,
) -> Option<(NumberFieldElement, NumberFieldElement)> {
    match (&left.representation, &right.representation) {
        (Representation::NumberField(left), _) => {
            element_in_field(right, left.field()).map(|right| (left.clone(), right))
        }
        (_, Representation::NumberField(right)) => {
            element_in_field(left, right.field()).map(|left| (left, right.clone()))
        }
        (Representation::IsolatedRoot(left), Representation::IsolatedRoot(right))
            if left.same_origin(right) =>
        {
            let generator = NumberFieldElement::generator(canonical::selected_root(left));
            Some((generator.clone(), generator))
        }
        (Representation::IsolatedRoot(root), Representation::Multiquadratic(value)) => {
            let constant = value.rational_value()?;
            let field = canonical::selected_root(root);
            Some((
                NumberFieldElement::generator(field.clone()),
                NumberFieldElement::constant(field, constant),
            ))
        }
        (Representation::Multiquadratic(value), Representation::IsolatedRoot(root)) => {
            let constant = value.rational_value()?;
            let field = canonical::selected_root(root);
            Some((
                NumberFieldElement::constant(field.clone(), constant),
                NumberFieldElement::generator(field),
            ))
        }
        (Representation::IsolatedRoot(_), Representation::IsolatedRoot(_))
        | (Representation::Multiquadratic(_), Representation::Multiquadratic(_)) => None,
    }
}

fn multiquadratic_pair<'a>(
    left: &'a AlgebraicReal,
    right: &'a AlgebraicReal,
) -> Option<(&'a Multiquadratic, &'a Multiquadratic)> {
    match (&left.representation, &right.representation) {
        (Representation::Multiquadratic(left), Representation::Multiquadratic(right)) => {
            Some((left, right))
        }
        _ => None,
    }
}

fn compact_composition_inputs(
    left: &AlgebraicReal,
    right: &AlgebraicReal,
) -> (Polynomial, Polynomial) {
    let annihilators = (left.annihilator(), right.annihilator());
    let degree = (annihilators.0.len() - 1).checked_mul(annihilators.1.len() - 1);
    if degree.is_some_and(|degree| degree <= MAX_LAZY_COMPOSITION_DEGREE) {
        annihilators
    } else {
        (left.canonical_root().0, right.canonical_root().0)
    }
}

fn add_values(left: &AlgebraicReal, right: &AlgebraicReal) -> AlgebraicReal {
    if let Some((left, right)) = multiquadratic_pair(left, right) {
        return AlgebraicReal::multiquadratic(left + right);
    }
    if let Some((left, right)) = shared_number_field_operands(left, right) {
        return AlgebraicReal::number_field(left.add(&right));
    }
    let (left_polynomial, right_polynomial) = compact_composition_inputs(left, right);
    let polynomial = root_composition::sum_polynomial(&left_polynomial, &right_polynomial);
    select_root(polynomial, |bits| {
        left.interval(bits).add(&right.interval(bits))
    })
    .expect("sum has a real algebraic result")
}

fn subtract_values(left: &AlgebraicReal, right: &AlgebraicReal) -> AlgebraicReal {
    if let Some((left, right)) = multiquadratic_pair(left, right) {
        return AlgebraicReal::multiquadratic(left - right);
    }
    if let Some((left, right)) = shared_number_field_operands(left, right) {
        return AlgebraicReal::number_field(left.add(&right.neg()));
    }
    add_values(left, &-right)
}

fn multiply_values(left: &AlgebraicReal, right: &AlgebraicReal) -> AlgebraicReal {
    if let Some((left, right)) = multiquadratic_pair(left, right) {
        return AlgebraicReal::multiquadratic(left * right);
    }
    if let Some((left, right)) = shared_number_field_operands(left, right) {
        return AlgebraicReal::number_field(left.mul(&right));
    }
    let (left_polynomial, right_polynomial) = compact_composition_inputs(left, right);
    let polynomial = root_composition::product_polynomial(&left_polynomial, &right_polynomial);
    select_root(polynomial, |bits| {
        left.interval(bits).mul(&right.interval(bits))
    })
    .expect("product has a real algebraic result")
}

fn divide_values(left: &AlgebraicReal, right: &AlgebraicReal) -> AlgebraicReal {
    assert!(!right.is_zero(), "AlgebraicReal division by zero");
    if let Some((left, right)) = multiquadratic_pair(left, right) {
        return AlgebraicReal::multiquadratic(left / right);
    }
    if let Some((left, right)) = shared_number_field_operands(left, right) {
        return AlgebraicReal::number_field(left.mul(&right.recip()));
    }
    let mut reciprocal = right.annihilator();
    reciprocal.reverse();
    let inverse = select_root(reciprocal, |bits| {
        let mut precision = bits;
        loop {
            if let Some(interval) = right.interval(precision).recip() {
                break interval;
            }
            precision = precision
                .checked_mul(2)
                .expect("nonzero algebraic interval precision overflow");
        }
    })
    .expect("reciprocal has a real algebraic result");
    multiply_values(left, &inverse)
}

macro_rules! binary_operation {
    ($trait:ident, $method:ident, $function:ident) => {
        impl $trait for &AlgebraicReal {
            type Output = AlgebraicReal;
            fn $method(self, rhs: Self) -> Self::Output {
                $function(self, rhs)
            }
        }
        impl $trait<AlgebraicReal> for AlgebraicReal {
            type Output = AlgebraicReal;
            fn $method(self, rhs: AlgebraicReal) -> Self::Output {
                $function(&self, &rhs)
            }
        }
        impl $trait<&AlgebraicReal> for AlgebraicReal {
            type Output = AlgebraicReal;
            fn $method(self, rhs: &AlgebraicReal) -> Self::Output {
                $function(&self, rhs)
            }
        }
        impl $trait<AlgebraicReal> for &AlgebraicReal {
            type Output = AlgebraicReal;
            fn $method(self, rhs: AlgebraicReal) -> Self::Output {
                $function(self, &rhs)
            }
        }
    };
}
binary_operation!(Add, add, add_values);
binary_operation!(Sub, sub, subtract_values);
binary_operation!(Mul, mul, multiply_values);
binary_operation!(Div, div, divide_values);
