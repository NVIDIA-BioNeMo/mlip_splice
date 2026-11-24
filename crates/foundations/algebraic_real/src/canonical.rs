// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::AlgebraicError;
use crate::algebraic_real::AlgebraicReal;
use crate::isolated_root::IsolatedRoot;
use crate::multiquadratic::{self, Multiquadratic};
use crate::number_field::NumberFieldElement;
use crate::polynomial::{self, Interval, Polynomial, RootIsolator};
use crate::representation::Representation;
use crate::selected_root::SelectedRoot;

fn isolated_root_data(root: &IsolatedRoot) -> Arc<SelectedRoot> {
    root.canonical_with(|| {
        let roots = polynomial::canonical_roots(root.polynomial(), 64);
        let (polynomial, root_index, _) = &roots[root.root_index()];
        let isolator = if polynomial == root.polynomial() && *root_index == root.root_index() {
            root.isolator().clone()
        } else {
            Arc::new(RootIsolator::new(polynomial))
        };
        SelectedRoot::from_isolator(*root_index, isolator)
    })
}

pub(crate) fn isolated_root(root: &IsolatedRoot) -> (Polynomial, usize) {
    let selected = isolated_root_data(root);
    (selected.polynomial().clone(), selected.root_index())
}

pub(crate) fn selected_root(root: &IsolatedRoot) -> Arc<SelectedRoot> {
    isolated_root_data(root)
}

pub(crate) fn cached_selected_root(root: &IsolatedRoot) -> Option<Arc<SelectedRoot>> {
    root.cached_canonical().cloned()
}

impl AlgebraicReal {
    pub(crate) fn as_number_field_element(&self) -> Option<NumberFieldElement> {
        match &self.representation {
            Representation::NumberField(element) => Some(element.clone()),
            Representation::IsolatedRoot(root) => {
                Some(NumberFieldElement::generator(selected_root(root)))
            }
            Representation::Multiquadratic(_) => None,
        }
    }

    pub(crate) fn from_isolated_polynomial(
        polynomial: Polynomial,
        root_index: usize,
        canonical: Option<Arc<SelectedRoot>>,
        isolator: Arc<polynomial::RootIsolator>,
    ) -> Self {
        if let [constant, linear] = polynomial.as_slice() {
            return Self::rational(BigRational::new(-constant, linear.clone()));
        }
        if polynomial[0].is_zero() {
            let nonzero_roots = polynomial::isolate(&polynomial[1..].to_vec());
            let zero_index = nonzero_roots
                .iter()
                .filter(|interval| interval.high() <= &BigRational::zero())
                .count();
            if root_index == zero_index {
                return Self::zero();
            }
        }
        debug_assert!(polynomial.len() > 2);
        let selected = canonical
            .as_ref()
            .filter(|root| root.polynomial() == &polynomial && root.root_index() == root_index)
            .cloned()
            .unwrap_or_else(|| SelectedRoot::from_isolator(root_index, isolator));
        Self {
            representation: Representation::IsolatedRoot(IsolatedRoot::new(selected, canonical)),
        }
    }

    pub(crate) fn from_irreducible_polynomial(polynomial: Polynomial, root_index: usize) -> Self {
        let isolator = Arc::new(polynomial::RootIsolator::new(&polynomial));
        let canonical = SelectedRoot::from_isolator(root_index, isolator.clone());
        Self::from_isolated_polynomial(polynomial, root_index, Some(canonical), isolator)
    }

    pub(crate) fn canonical_root(&self) -> (Polynomial, usize) {
        match &self.representation {
            Representation::NumberField(value) => {
                let field = value.canonical();
                (field.polynomial().clone(), field.root_index())
            }
            Representation::IsolatedRoot(root) => crate::canonical::isolated_root(root),
            Representation::Multiquadratic(value) => value.canonical_with(|| {
                let polynomial = value.minimal_polynomial();
                let roots = polynomial::isolate(&polynomial);
                let root_index = roots
                    .iter()
                    .position(|interval| {
                        (value - &Multiquadratic::rational(interval.low().clone())).signum() >= 0
                            && (value - &Multiquadratic::rational(interval.high().clone())).signum()
                                <= 0
                    })
                    .expect("minimal polynomial must isolate its value");
                (polynomial, root_index)
            }),
        }
    }

    pub(crate) fn annihilator(&self) -> Polynomial {
        match &self.representation {
            Representation::IsolatedRoot(root) => root.polynomial().clone(),
            Representation::NumberField(_) | Representation::Multiquadratic(_) => {
                self.canonical_root().0
            }
        }
    }

    pub(crate) fn isolating_root(&self) -> Arc<SelectedRoot> {
        match &self.representation {
            Representation::IsolatedRoot(root) => root.selected().clone(),
            Representation::NumberField(value) => value.canonical(),
            Representation::Multiquadratic(_) => {
                let (polynomial, root_index) = self.canonical_root();
                let isolator = Arc::new(polynomial::RootIsolator::new(&polynomial));
                SelectedRoot::from_isolator(root_index, isolator)
            }
        }
    }

    pub(crate) fn interval(&self, bits: usize) -> Interval {
        match &self.representation {
            Representation::NumberField(value) => value.interval(bits),
            Representation::IsolatedRoot(root) => root.interval(bits),
            Representation::Multiquadratic(value) => multiquadratic::interval(value, bits),
        }
    }
}

fn nonnegative_root_bounds(
    value: &BigRational,
    degree: u32,
    bits: usize,
) -> (BigRational, BigRational) {
    let value = value.clone().max(BigRational::zero());
    let shift = bits
        .checked_mul(degree as usize)
        .expect("root precision overflow");
    let scaled_numerator = value.numer().magnitude() << shift;
    let scaled = &scaled_numerator / value.denom().magnitude();
    let lower_integer = scaled.nth_root(degree);
    let denominator = BigInt::one() << bits;
    let low = BigRational::new(BigInt::from(lower_integer.clone()), denominator.clone());
    let exact = lower_integer.pow(degree) * value.denom().magnitude() == scaled_numerator;
    let high_integer = if exact {
        lower_integer
    } else {
        lower_integer + 1u32
    };
    let high = BigRational::new(BigInt::from(high_integer), denominator);
    (low, high)
}

pub(crate) fn root_interval(
    source: &Interval,
    degree: u32,
    positive: bool,
    bits: usize,
) -> Interval {
    if positive {
        let (low, _) = nonnegative_root_bounds(source.low(), degree, bits);
        let (_, high) = nonnegative_root_bounds(source.high(), degree, bits);
        Interval::new(low, high)
    } else {
        let (_, magnitude_high) = nonnegative_root_bounds(&(-source.low()), degree, bits);
        let (magnitude_low, _) = nonnegative_root_bounds(&(-source.high()), degree, bits);
        Interval::new(-magnitude_high, -magnitude_low)
    }
}

pub(crate) fn select_root(
    mut annihilator: Polynomial,
    interval: impl FnMut(usize) -> Interval,
) -> Result<AlgebraicReal, AlgebraicError> {
    annihilator = polynomial::square_free(&polynomial::primitive_integer(annihilator));
    let isolator = Arc::new(polynomial::RootIsolator::new(&annihilator));
    let (root_index, _) = polynomial::select_overlapping_root(&isolator, interval)?;
    Ok(AlgebraicReal::from_isolated_polynomial(
        annihilator,
        root_index,
        None,
        isolator,
    ))
}

pub(crate) fn equal(left: &AlgebraicReal, right: &AlgebraicReal) -> bool {
    let left = left.isolating_root();
    let right = right.isolating_root();
    if left.polynomial() == right.polynomial() {
        return left.root_index() == right.root_index();
    }

    let common = polynomial::gcd(left.polynomial(), right.polynomial());
    if common.len() == 1 {
        return false;
    }
    let common_roots = RootIsolator::new(&common);
    let left_interval = left.isolator().strict_isolation(left.root_index(), 32);
    let right_interval = right.isolator().strict_isolation(right.root_index(), 32);
    match (
        common_roots.unique_root_index_in(&left_interval),
        common_roots.unique_root_index_in(&right_interval),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}
