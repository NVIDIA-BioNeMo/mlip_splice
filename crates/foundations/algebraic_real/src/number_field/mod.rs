// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::polynomial::{self, Interval, Polynomial};
use crate::rational_polynomial::{RationalPolynomial, add, inverse_mod, multiply, remainder, trim};
use crate::selected_root::SelectedRoot;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReducedPolynomial(RationalPolynomial);

impl ReducedPolynomial {
    fn new(coefficients: RationalPolynomial, modulus_degree: usize) -> Self {
        let coefficients = trim(coefficients);
        assert!(
            coefficients.len() <= modulus_degree,
            "number-field coefficients are reduced"
        );
        Self(coefficients)
    }
}

impl Deref for ReducedPolynomial {
    type Target = [BigRational];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NumberFieldElement {
    field: Arc<SelectedRoot>,
    coefficients: ReducedPolynomial,
    canonical: Arc<OnceLock<Arc<SelectedRoot>>>,
}

fn interval_polynomial(coefficients: &[BigRational], argument: &Interval) -> Interval {
    coefficients.iter().rev().fold(
        Interval::point(BigRational::zero()),
        |value, coefficient| {
            value
                .mul(argument)
                .add(&Interval::point(coefficient.clone()))
        },
    )
}

fn power_sums(polynomial: &Polynomial) -> Vec<BigRational> {
    let degree = polynomial.len() - 1;
    let leading = BigRational::from_integer(polynomial[degree].clone());
    let monic: Vec<_> = polynomial
        .iter()
        .map(|coefficient| BigRational::from_integer(coefficient.clone()) / &leading)
        .collect();
    let mut sums = Vec::with_capacity(degree);
    sums.push(BigRational::from_integer(BigInt::from(degree)));
    for power in 1..degree {
        let mut sum = BigRational::zero();
        for offset in 1..power {
            sum += &monic[degree - offset] * &sums[power - offset];
        }
        sum += &monic[degree - power] * BigInt::from(power);
        sums.push(-sum);
    }
    sums
}

fn characteristic_polynomial(element: &NumberFieldElement) -> Polynomial {
    let degree = element.field.polynomial().len() - 1;
    let defining: Vec<_> = element
        .field
        .polynomial()
        .iter()
        .cloned()
        .map(BigRational::from_integer)
        .collect();
    let generator_sums = power_sums(element.field.polynomial());
    let mut element_power = vec![BigRational::one()];
    let mut sums = Vec::with_capacity(degree + 1);
    for _ in 0..=degree {
        sums.push(
            element_power
                .iter()
                .zip(&generator_sums)
                .fold(BigRational::zero(), |sum, (coefficient, trace)| {
                    sum + coefficient * trace
                }),
        );
        element_power = remainder(&multiply(&element_power, &element.coefficients), &defining);
    }

    let mut polynomial = vec![BigRational::zero(); degree + 1];
    polynomial[degree] = BigRational::one();
    for offset in 1..=degree {
        let mut sum = sums[offset].clone();
        for previous in 1..offset {
            sum += &polynomial[degree - previous] * &sums[offset - previous];
        }
        polynomial[degree - offset] = -sum / BigInt::from(offset);
    }
    let characteristic =
        polynomial::primitive(polynomial).expect("a characteristic polynomial is nonconstant");
    polynomial::square_free(&characteristic)
}

impl Hash for NumberFieldElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.field.polynomial().hash(state);
        self.field.root_index().hash(state);
        self.coefficients.hash(state);
    }
}

impl NumberFieldElement {
    pub(crate) fn generator(field: Arc<SelectedRoot>) -> Self {
        Self::from_coefficients(field, vec![BigRational::zero(), BigRational::one()])
    }

    pub(crate) fn constant(field: Arc<SelectedRoot>, value: BigRational) -> Self {
        Self::from_reduced_coefficients(field, vec![value])
    }

    fn from_coefficients(field: Arc<SelectedRoot>, coefficients: RationalPolynomial) -> Self {
        let defining: Vec<_> = field
            .polynomial()
            .iter()
            .cloned()
            .map(BigRational::from_integer)
            .collect();
        let coefficients = remainder(&coefficients, &defining);
        Self::from_reduced_coefficients(field, coefficients)
    }

    fn from_reduced_coefficients(
        field: Arc<SelectedRoot>,
        coefficients: RationalPolynomial,
    ) -> Self {
        let coefficients = ReducedPolynomial::new(coefficients, field.polynomial().len() - 1);
        Self {
            coefficients,
            field,
            canonical: Arc::new(OnceLock::new()),
        }
    }

    pub(crate) fn same_field(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.field, &other.field) || self.field.same_embedding(&other.field)
    }

    pub(crate) fn same_field_element(&self, other: &Self) -> bool {
        self.same_field(other) && self.coefficients == other.coefficients
    }

    pub(crate) fn field(&self) -> &Arc<SelectedRoot> {
        &self.field
    }

    pub(crate) fn as_fraction(&self) -> Option<BigRational> {
        (self.coefficients.len() == 1).then(|| self.coefficients[0].clone())
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        assert!(self.same_field(other));
        Self::from_reduced_coefficients(
            self.field.clone(),
            add(&self.coefficients, &other.coefficients),
        )
    }

    pub(crate) fn neg(&self) -> Self {
        Self::from_reduced_coefficients(
            self.field.clone(),
            self.coefficients.iter().map(|value| -value).collect(),
        )
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        assert!(self.same_field(other));
        Self::from_coefficients(
            self.field.clone(),
            multiply(&self.coefficients, &other.coefficients),
        )
    }

    pub(crate) fn recip(&self) -> Self {
        let defining: Vec<_> = self
            .field
            .polynomial()
            .iter()
            .cloned()
            .map(BigRational::from_integer)
            .collect();
        Self::from_reduced_coefficients(
            self.field.clone(),
            inverse_mod(&self.coefficients, &defining),
        )
    }

    pub(crate) fn interval(&self, bits: usize) -> Interval {
        let generator = self
            .field
            .isolator()
            .isolate_to_bits(self.field.root_index(), bits);
        interval_polynomial(&self.coefficients, &generator)
    }

    pub(crate) fn canonical(&self) -> Arc<SelectedRoot> {
        self.canonical
            .get_or_init(|| {
                let polynomial = characteristic_polynomial(self);
                let roots = Arc::new(polynomial::RootIsolator::new(&polynomial));
                let (root_index, _) =
                    polynomial::select_overlapping_root(&roots, |bits| self.interval(bits))
                        .expect("number-field root isolation precision overflow");
                SelectedRoot::from_isolator(root_index, roots)
            })
            .clone()
    }

    pub(crate) fn structural_size(&self) -> usize {
        self.field.polynomial().len() + self.coefficients.len()
    }

    pub(crate) fn estimated_work(&self) -> usize {
        let field_work = self.field.polynomial().len().saturating_mul(
            self.field
                .polynomial()
                .iter()
                .map(|coefficient| coefficient.bits() as usize + 1)
                .sum::<usize>(),
        );
        self.coefficients
            .iter()
            .fold(field_work, |work, coefficient| {
                work.saturating_add(coefficient.numer().bits() as usize)
                    .saturating_add(coefficient.denom().bits() as usize)
                    .saturating_add(1)
            })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn z(coefficients: &[i64]) -> Polynomial {
        coefficients.iter().copied().map(BigInt::from).collect()
    }

    #[test]
    #[should_panic(expected = "number-field coefficients are reduced")]
    fn reduced_polynomial_rejects_degree_overflow() {
        ReducedPolynomial::new(
            vec![BigRational::zero(), BigRational::zero(), BigRational::one()],
            2,
        );
    }

    #[test]
    fn field_identity_uses_the_selected_root_not_allocation() {
        let polynomial = z(&[-2, 0, 1]);
        let left = NumberFieldElement::generator(SelectedRoot::new(polynomial.clone(), 1));
        let right = NumberFieldElement::generator(SelectedRoot::new(polynomial, 1));
        assert!(left.same_field(&right));
        assert!(left.same_field_element(&right));

        let hash = |value: &NumberFieldElement| {
            let mut state = DefaultHasher::new();
            value.hash(&mut state);
            state.finish()
        };
        assert_eq!(hash(&left), hash(&right));
    }

    #[test]
    fn powers_stay_in_the_defining_field() {
        let field = SelectedRoot::new(z(&[-1, -1, 0, 0, 0, 1]), 0);
        let alpha = NumberFieldElement::generator(field.clone());
        let square = alpha.mul(&alpha);
        let fourth = square.mul(&square);
        assert_eq!(fourth.coefficients.len(), 5);
        assert_eq!(fourth.canonical().polynomial(), &z(&[-1, 1, -4, 6, -4, 1]));
    }

    #[test]
    fn reciprocal_multiplies_to_one() {
        let field = SelectedRoot::new(z(&[-2, 0, 0, 1]), 0);
        let alpha = NumberFieldElement::generator(field.clone());
        assert_eq!(
            alpha.mul(&alpha.recip()).as_fraction(),
            Some(BigRational::one())
        );
        let shifted = alpha.add(&NumberFieldElement::constant(
            field.clone(),
            BigRational::one(),
        ));
        assert_eq!(
            shifted.mul(&shifted.recip()).as_fraction(),
            Some(BigRational::one())
        );
    }

    #[test]
    fn nonmonic_fields_reduce_exactly() {
        let field = SelectedRoot::new(z(&[-1, 0, 2]), 1);
        let alpha = NumberFieldElement::generator(field.clone());
        assert_eq!(
            alpha.mul(&alpha).as_fraction(),
            Some(BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn characteristic_square_free_part_finds_subfields() {
        let field = SelectedRoot::new(z(&[-2, 0, 0, 0, 1]), 1);
        let alpha = NumberFieldElement::generator(field.clone());
        let square = alpha.mul(&alpha);
        let canonical = square.canonical();
        assert_eq!(canonical.polynomial(), &z(&[-2, 0, 1]));
        assert_eq!(canonical.root_index(), 1);
    }
}
