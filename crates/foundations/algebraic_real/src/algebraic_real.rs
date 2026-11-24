// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::AlgebraicError;
use crate::canonical::{root_interval, select_root};
use crate::float::round_rational_to_f64;
use crate::multiquadratic::Multiquadratic;
use crate::number_field::NumberFieldElement;
use crate::polynomial::{self, Polynomial};
use crate::rational_root;
use crate::representation::Representation;

const MAX_ROOT_DEGREE: u32 = 4_096;

/// An exact real algebraic number.
///
/// Values made from square roots retain a private multiquadratic fast path.
/// Every other value is an isolated root of a square-free integer annihilator;
/// its irreducible polynomial is computed only when a canonical boundary needs it.
#[derive(Clone, Debug)]
pub struct AlgebraicReal {
    pub(crate) representation: Representation,
}

impl AlgebraicReal {
    pub(crate) fn multiquadratic(value: Multiquadratic) -> Self {
        Self {
            representation: Representation::Multiquadratic(value),
        }
    }

    pub(crate) fn number_field(element: NumberFieldElement) -> Self {
        if let Some(value) = element.as_fraction() {
            Self::rational(value)
        } else {
            Self {
                representation: Representation::NumberField(element),
            }
        }
    }

    pub fn from_polynomial(
        coefficients: Vec<BigRational>,
        root_index: usize,
    ) -> Result<Self, AlgebraicError> {
        let polynomial = polynomial::square_free(&polynomial::primitive(coefficients)?);
        let isolator = Arc::new(polynomial::RootIsolator::new(&polynomial));
        if root_index >= isolator.root_count() {
            return Err(AlgebraicError::RootIndex {
                index: root_index,
                roots: isolator.root_count(),
            });
        }
        Ok(Self::from_isolated_polynomial(
            polynomial, root_index, None, isolator,
        ))
    }
    pub fn zero() -> Self {
        Self::multiquadratic(Multiquadratic::zero())
    }
    pub fn one() -> Self {
        Self::multiquadratic(Multiquadratic::one())
    }
    pub fn integer(value: impl Into<BigInt>) -> Self {
        Self::multiquadratic(Multiquadratic::integer(value))
    }
    pub fn rational(value: BigRational) -> Self {
        Self::multiquadratic(Multiquadratic::rational(value))
    }
    pub fn sqrt(value: impl Into<BigUint>) -> Self {
        Self::multiquadratic(Multiquadratic::sqrt(value))
    }

    pub fn sqrt_rational(value: BigRational) -> Result<Self, AlgebraicError> {
        if value.is_negative() {
            return Err(AlgebraicError::EvenRootOfNegative);
        }
        Ok(Self::multiquadratic(Multiquadratic::sqrt_rational(value)))
    }

    pub fn term(coefficient: BigRational, radicand: impl Into<BigUint>) -> Self {
        Self::multiquadratic(Multiquadratic::term(coefficient, radicand))
    }

    pub fn is_zero(&self) -> bool {
        match &self.representation {
            Representation::NumberField(value) => value.as_fraction().is_some_and(|x| x.is_zero()),
            Representation::IsolatedRoot(_) => false,
            Representation::Multiquadratic(value) => value.is_zero(),
        }
    }

    pub fn as_fraction(&self) -> Option<BigRational> {
        match &self.representation {
            Representation::NumberField(value) => value.as_fraction(),
            Representation::IsolatedRoot(_) => {
                let (polynomial, _) = self.canonical_root();
                let [constant, linear] = polynomial.as_slice() else {
                    return None;
                };
                Some(BigRational::new(-constant, linear.clone()))
            }
            Representation::Multiquadratic(value) => value.rational_value(),
        }
    }

    pub fn minimal_polynomial_and_root_index(&self) -> (Polynomial, usize) {
        self.canonical_root()
    }
    pub fn minimal_polynomial(&self) -> Polynomial {
        self.minimal_polynomial_and_root_index().0
    }
    pub fn degree(&self) -> usize {
        self.canonical_root().0.len() - 1
    }
    pub fn trace(&self) -> BigRational {
        let (polynomial, _) = self.canonical_root();
        let degree = polynomial.len() - 1;
        BigRational::new(-&polynomial[degree - 1], polynomial[degree].clone())
    }

    pub fn norm(&self) -> BigRational {
        let (polynomial, _) = self.canonical_root();
        let degree = polynomial.len() - 1;
        let numerator = if degree.is_multiple_of(2) {
            polynomial[0].clone()
        } else {
            -&polynomial[0]
        };
        BigRational::new(numerator, polynomial[degree].clone())
    }

    pub fn signum(&self) -> i32 {
        match &self.representation {
            Representation::NumberField(value) => {
                debug_assert!(value.as_fraction().is_none());
                for bits in (64..).step_by(64) {
                    let interval = value.interval(bits);
                    if interval.high() < &BigRational::zero() {
                        return -1;
                    }
                    if interval.low() > &BigRational::zero() {
                        return 1;
                    }
                }
                unreachable!()
            }
            Representation::IsolatedRoot(root) => root.signum(),
            Representation::Multiquadratic(value) => value.signum(),
        }
    }

    pub fn abs(&self) -> Self {
        if self.signum() < 0 {
            -self
        } else {
            self.clone()
        }
    }

    pub fn floor(&self) -> BigInt {
        if let Some(value) = self.as_fraction() {
            return value.numer().div_floor(value.denom());
        }
        let mut bits = 32usize;
        loop {
            let interval = self.interval(bits);
            let low = interval.low().numer().div_floor(interval.low().denom());
            let high = interval.high().numer().div_floor(interval.high().denom());
            if low == high {
                return low;
            }
            bits = bits.checked_add(32).expect("floor precision overflow");
        }
    }

    pub fn ceil(&self) -> BigInt {
        -(-self).floor()
    }
    pub fn trunc(&self) -> BigInt {
        if self.signum() < 0 {
            self.ceil()
        } else {
            self.floor()
        }
    }

    pub fn checked_div(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() {
            None
        } else {
            Some(self / divisor)
        }
    }

    pub fn recip(&self) -> Option<Self> {
        Self::one().checked_div(self)
    }

    pub(crate) fn nth_root(&self, degree: u32) -> Result<Self, AlgebraicError> {
        if degree == 0 {
            return Err(AlgebraicError::ZeroRootDegree);
        }
        if degree > MAX_ROOT_DEGREE {
            return Err(AlgebraicError::RootDegreeTooLarge);
        }
        if degree.is_multiple_of(2) && self.signum() < 0 {
            return Err(AlgebraicError::EvenRootOfNegative);
        }
        if degree == 1 {
            return Ok(self.clone());
        }
        if degree == 2
            && let Representation::Multiquadratic(value) = &self.representation
            && let Some(root) = value.sqrt_exact()
        {
            return Ok(Self::multiquadratic(root));
        }
        if let Some(value) = self.as_fraction() {
            return Ok(rational_root::nth_root(value, degree));
        }
        let (source, _) = self.canonical_root();
        let source_degree = source.len() - 1;
        if source[1..source_degree].iter().all(Zero::is_zero) {
            let source_degree =
                u32::try_from(source_degree).map_err(|_| AlgebraicError::RootDegreeTooLarge)?;
            let combined_degree = source_degree
                .checked_mul(degree)
                .filter(|&degree| degree <= MAX_ROOT_DEGREE)
                .ok_or(AlgebraicError::RootDegreeTooLarge)?;
            let radicand = BigRational::new(-&source[0], source[source_degree as usize].clone());
            let mut root = rational_root::nth_root(radicand, combined_degree);
            if source_degree.is_multiple_of(2) && self.signum() < 0 {
                root = -root;
            }
            return Ok(root);
        }
        let result_degree = source_degree
            .checked_mul(degree as usize)
            .filter(|&degree| degree <= MAX_ROOT_DEGREE as usize)
            .ok_or(AlgebraicError::RootDegreeTooLarge)?;
        let mut annihilator = vec![BigInt::zero(); result_degree + 1];
        for (index, coefficient) in source.iter().enumerate() {
            annihilator[index * degree as usize] = coefficient.clone();
        }
        let positive = self.signum() >= 0;
        select_root(annihilator, |bits| {
            let interval = self.interval(bits);
            root_interval(&interval, degree, positive, bits)
        })
    }

    pub fn pow_rational(
        &self,
        numerator: &BigInt,
        denominator: &BigUint,
    ) -> Result<Self, AlgebraicError> {
        if denominator.is_zero() {
            return Err(AlgebraicError::ZeroRootDegree);
        }
        let divisor = BigInt::from(denominator.clone());
        let gcd = numerator.gcd(&divisor);
        let numerator = numerator / &gcd;
        let denominator = (divisor / gcd)
            .to_u32()
            .ok_or(AlgebraicError::RootDegreeTooLarge)?;
        if denominator.is_multiple_of(2) && self.signum() < 0 {
            return Err(AlgebraicError::EvenRootOfNegative);
        }
        self.pow_integer(&numerator)?.nth_root(denominator)
    }

    pub fn pow_integer(&self, exponent: &BigInt) -> Result<Self, AlgebraicError> {
        if exponent.is_zero() {
            return Ok(Self::one());
        }
        if exponent.is_one() {
            return Ok(self.clone());
        }
        if self.is_zero() {
            return if exponent.is_negative() {
                Err(AlgebraicError::DivisionByZero)
            } else {
                Ok(Self::zero())
            };
        }
        if let Some(value) = self.as_fraction() {
            if value.is_one() {
                return Ok(Self::one());
            }
            if value == -BigRational::one() {
                return Ok(if exponent.magnitude().bit(0) {
                    Self::integer(-1)
                } else {
                    Self::one()
                });
            }
        }
        let mut power = exponent.magnitude().clone();
        let mut factor = match &self.representation {
            Representation::IsolatedRoot(_) | Representation::NumberField(_) => {
                Self::number_field(self.as_number_field_element().unwrap())
            }
            Representation::Multiquadratic(_) => self.clone(),
        };
        let mut result = Self::one();
        while !power.is_zero() {
            if power.bit(0) {
                result = &result * &factor;
            }
            power >>= 1usize;
            if !power.is_zero() {
                factor = &factor * &factor;
            }
        }
        if exponent.is_negative() {
            Ok(Self::one()
                .checked_div(&result)
                .expect("nonzero power has a reciprocal"))
        } else {
            Ok(result)
        }
    }

    pub fn to_f64(&self) -> f64 {
        if let Some(value) = self.as_fraction() {
            return round_rational_to_f64(&value);
        }
        for bits in (64..).step_by(64) {
            let interval = self.interval(bits);
            let low = round_rational_to_f64(interval.low());
            let high = round_rational_to_f64(interval.high());
            if low.to_bits() == high.to_bits() {
                return low;
            }
        }
        unreachable!()
    }

    pub fn canonical_rational_scale(&self) -> Option<BigRational> {
        match &self.representation {
            Representation::NumberField(_) | Representation::IsolatedRoot(_) => {
                self.as_fraction().or(Some(BigRational::one()))
            }
            Representation::Multiquadratic(value) => value.rational_content(),
        }
    }

    pub fn structural_size(&self) -> usize {
        match &self.representation {
            Representation::NumberField(value) => value.structural_size(),
            Representation::IsolatedRoot(root) => root.polynomial().len(),
            Representation::Multiquadratic(value) => value.num_terms().max(1),
        }
    }

    pub fn estimated_work(&self) -> usize {
        match &self.representation {
            Representation::NumberField(value) => value.estimated_work(),
            Representation::IsolatedRoot(root) => root.polynomial().len().saturating_mul(
                root.polynomial()
                    .iter()
                    .map(|x| x.bits() as usize + 1)
                    .sum::<usize>(),
            ),
            Representation::Multiquadratic(value) => value
                .terms()
                .into_iter()
                .fold(0usize, |cost, (coefficient, radicand)| {
                    cost.saturating_add(coefficient.numer().bits() as usize)
                        .saturating_add(coefficient.denom().bits() as usize)
                        .saturating_add(radicand.bits() as usize)
                        .saturating_add(1)
                })
                .max(1),
        }
    }

    pub fn to_data(&self) -> crate::AlgebraicRealData {
        match &self.representation {
            Representation::NumberField(_) | Representation::IsolatedRoot(_) => {
                let (polynomial, root_index) = self.canonical_root();
                if let [constant, linear] = polynomial.as_slice() {
                    return Self::rational(BigRational::new(-constant, linear.clone())).to_data();
                }
                crate::AlgebraicRealData::Polynomial {
                    coefficients: polynomial,
                    root_index,
                }
            }
            Representation::Multiquadratic(value) => crate::AlgebraicRealData::Multiquadratic(
                value
                    .terms()
                    .into_iter()
                    .map(|(coefficient, radicand)| {
                        (
                            coefficient.numer().clone(),
                            coefficient.denom().clone(),
                            radicand,
                        )
                    })
                    .collect(),
            ),
        }
    }

    pub fn from_data(data: crate::AlgebraicRealData) -> Result<Self, AlgebraicError> {
        match data {
            crate::AlgebraicRealData::Multiquadratic(terms) => {
                let mut value = Multiquadratic::zero();
                for (numerator, denominator, radicand) in terms {
                    if denominator.is_zero() {
                        return Err(AlgebraicError::ZeroDenominator);
                    }
                    if radicand.is_zero() {
                        return Err(AlgebraicError::InvalidEncoding);
                    }
                    value = value
                        + Multiquadratic::term(BigRational::new(numerator, denominator), radicand);
                }
                Ok(Self::multiquadratic(value))
            }
            crate::AlgebraicRealData::Polynomial {
                coefficients,
                root_index,
            } => Self::from_polynomial(
                coefficients
                    .into_iter()
                    .map(BigRational::from_integer)
                    .collect(),
                root_index,
            ),
        }
    }
}

impl PartialEq for AlgebraicReal {
    fn eq(&self, other: &Self) -> bool {
        if let (Representation::Multiquadratic(left), Representation::Multiquadratic(right)) =
            (&self.representation, &other.representation)
        {
            return left == right;
        }
        if let (Representation::NumberField(left), Representation::NumberField(right)) =
            (&self.representation, &other.representation)
            && left.same_field(right)
        {
            return left.same_field_element(right);
        }
        crate::canonical::equal(self, other)
    }
}
impl Eq for AlgebraicReal {}

impl Hash for AlgebraicReal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let scale = BigInt::from(256);
        for bits in (8..).step_by(8) {
            let interval = self.interval(bits);
            let low = (interval.low() * &scale)
                .numer()
                .div_floor((interval.low() * &scale).denom());
            let high = (interval.high() * &scale)
                .numer()
                .div_floor((interval.high() * &scale).denom());
            if low == high {
                low.hash(state);
                return;
            }
            if bits == 256
                && let Some(value) = self.as_fraction()
            {
                let scaled = value * &scale;
                scaled.numer().div_floor(scaled.denom()).hash(state);
                return;
            }
        }
        unreachable!()
    }
}

impl PartialOrd for AlgebraicReal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AlgebraicReal {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.representation.same_stored_value(&other.representation) {
            return Ordering::Equal;
        }
        let left = self.interval(32);
        let right = other.interval(32);
        if let Some(order) = left.separated_order(&right) {
            return order;
        }
        if self == other {
            return Ordering::Equal;
        }
        for bits in (64..).step_by(32) {
            let left = self.interval(bits);
            let right = other.interval(bits);
            if let Some(order) = left.separated_order(&right) {
                return order;
            }
        }
        unreachable!()
    }
}

impl fmt::Display for AlgebraicReal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.representation {
            Representation::NumberField(_) | Representation::IsolatedRoot(_) => {
                let (polynomial, root_index) = self.canonical_root();
                write!(formatter, "root(")?;
                write_polynomial(formatter, &polynomial)?;
                write!(formatter, ", {root_index})")
            }
            Representation::Multiquadratic(value) => write!(formatter, "{value}"),
        }
    }
}

fn write_polynomial(formatter: &mut fmt::Formatter<'_>, polynomial: &[BigInt]) -> fmt::Result {
    let mut wrote = false;
    for (degree, coefficient) in polynomial.iter().enumerate().rev() {
        if coefficient.is_zero() {
            continue;
        }
        let negative = coefficient.is_negative();
        if wrote {
            write!(formatter, " {} ", if negative { "-" } else { "+" })?;
        } else if negative {
            write!(formatter, "-")?;
        }
        let magnitude = coefficient.magnitude();
        if degree == 0 || !magnitude.is_one() {
            write!(formatter, "{magnitude}")?;
            if degree != 0 {
                write!(formatter, "*")?;
            }
        }
        if degree != 0 {
            write!(formatter, "x")?;
            if degree != 1 {
                write!(formatter, "^{degree}")?;
            }
        }
        wrote = true;
    }
    if !wrote {
        write!(formatter, "0")?;
    }
    Ok(())
}

impl Zero for AlgebraicReal {
    fn zero() -> Self {
        Self::zero()
    }
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}
impl One for AlgebraicReal {
    fn one() -> Self {
        Self::one()
    }
}
