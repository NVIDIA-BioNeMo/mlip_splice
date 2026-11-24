// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::AlgebraicError;
use crate::integer_polynomial_factorization;
use crate::rational_polynomial::{
    RationalPolynomial, div_rem as div_rem_q, monic_gcd as gcd_q, trim as trim_q,
};

pub(crate) type Polynomial = Vec<BigInt>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Interval {
    low: BigRational,
    high: BigRational,
}

impl Interval {
    pub(crate) fn new(low: BigRational, high: BigRational) -> Self {
        assert!(low <= high, "interval bounds are ordered");
        Self { low, high }
    }

    pub(crate) fn point(value: BigRational) -> Self {
        Self::new(value.clone(), value)
    }

    pub(crate) fn low(&self) -> &BigRational {
        &self.low
    }

    pub(crate) fn high(&self) -> &BigRational {
        &self.high
    }

    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    pub(crate) fn separated_order(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.high < other.low {
            Some(std::cmp::Ordering::Less)
        } else if other.high < self.low {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        Self::new(&self.low + &other.low, &self.high + &other.high)
    }

    pub(crate) fn neg(&self) -> Self {
        Self::new(-&self.high, -&self.low)
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        let values = [
            &self.low * &other.low,
            &self.low * &other.high,
            &self.high * &other.low,
            &self.high * &other.high,
        ];
        Self::new(
            values.iter().min().unwrap().clone(),
            values.iter().max().unwrap().clone(),
        )
    }

    pub(crate) fn recip(&self) -> Option<Self> {
        if self.low <= BigRational::zero() && self.high >= BigRational::zero() {
            None
        } else {
            let a = self.low.recip();
            let b = self.high.recip();
            Some(Self::new(a.clone().min(b.clone()), a.max(b)))
        }
    }
}

pub(crate) fn trim(mut polynomial: Polynomial) -> Polynomial {
    while polynomial.len() > 1 && polynomial.last().is_some_and(Zero::is_zero) {
        polynomial.pop();
    }
    if polynomial.is_empty() {
        polynomial.push(BigInt::zero());
    }
    polynomial
}

pub(crate) fn primitive(polynomial: RationalPolynomial) -> Result<Polynomial, AlgebraicError> {
    let polynomial = trim_q(polynomial);
    if polynomial.len() < 2 {
        return Err(AlgebraicError::ConstantPolynomial);
    }
    let denominator = polynomial
        .iter()
        .fold(BigInt::one(), |lcm, value| lcm.lcm(value.denom()));
    let integers = polynomial
        .into_iter()
        .map(|value| value.numer() * (&denominator / value.denom()))
        .collect();
    Ok(primitive_integer(integers))
}

pub(crate) fn primitive_integer(polynomial: Polynomial) -> Polynomial {
    let mut polynomial = trim(polynomial);
    let content = polynomial
        .iter()
        .fold(BigInt::zero(), |gcd, value| gcd.gcd(value));
    if !content.is_zero() && !content.is_one() {
        for value in &mut polynomial {
            *value /= &content;
        }
    }
    if polynomial.last().is_some_and(Signed::is_negative) {
        for value in &mut polynomial {
            *value = -&*value;
        }
    }
    polynomial
}

fn as_rational(polynomial: &Polynomial) -> RationalPolynomial {
    polynomial
        .iter()
        .cloned()
        .map(BigRational::from_integer)
        .collect()
}

fn derivative(polynomial: &RationalPolynomial) -> RationalPolynomial {
    trim_q(
        polynomial
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, value)| value * BigInt::from(i))
            .collect(),
    )
}

fn pseudo_remainder(numerator: &Polynomial, denominator: &Polynomial) -> Polynomial {
    let denominator = trim(denominator.clone());
    assert!(denominator.iter().any(|coefficient| !coefficient.is_zero()));
    let mut remainder = trim(numerator.clone());
    while remainder.len() >= denominator.len()
        && remainder.iter().any(|coefficient| !coefficient.is_zero())
    {
        let shift = remainder.len() - denominator.len();
        let leading = remainder.last().unwrap().clone();
        let denominator_leading = denominator.last().unwrap();
        for coefficient in &mut remainder {
            *coefficient *= denominator_leading;
        }
        for (index, coefficient) in denominator.iter().enumerate() {
            remainder[index + shift] -= &leading * coefficient;
        }
        remainder = trim(remainder);
    }
    remainder
}

pub(crate) fn gcd(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut left = primitive_integer(left.clone());
    let mut right = primitive_integer(right.clone());
    while right.iter().any(|coefficient| !coefficient.is_zero()) {
        let remainder = primitive_integer(pseudo_remainder(&left, &right));
        left = right;
        right = remainder;
    }
    primitive_integer(left)
}

pub(crate) fn square_free(polynomial: &Polynomial) -> Polynomial {
    let rational = as_rational(polynomial);
    let gcd = gcd_q(rational.clone(), derivative(&rational));
    let (quotient, remainder) = div_rem_q(&rational, &gcd);
    assert!(remainder.iter().all(Zero::is_zero));
    primitive(quotient).expect("square-free quotient is nonconstant")
}

fn derivative_integer(polynomial: &Polynomial) -> Polynomial {
    trim(
        polynomial
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, value)| value * BigInt::from(i))
            .collect(),
    )
}

fn primitive_integer_preserving_sign(polynomial: RationalPolynomial) -> Polynomial {
    let polynomial = trim_q(polynomial);
    let denominator = polynomial
        .iter()
        .fold(BigInt::one(), |lcm, value| lcm.lcm(value.denom()));
    let mut integers: Polynomial = polynomial
        .into_iter()
        .map(|value| value.numer() * (&denominator / value.denom()))
        .collect();
    let content = integers
        .iter()
        .fold(BigInt::zero(), |gcd, value| gcd.gcd(value));
    if !content.is_zero() && !content.is_one() {
        for value in &mut integers {
            *value /= &content;
        }
    }
    trim(integers)
}

fn sturm(polynomial: &Polynomial) -> Vec<Polynomial> {
    let first = polynomial.clone();
    let second = derivative_integer(&first);
    let mut sequence = vec![first, second];
    while sequence.last().unwrap().len() > 1 {
        let length = sequence.len();
        let (_, remainder) = div_rem_q(
            &as_rational(&sequence[length - 2]),
            &as_rational(&sequence[length - 1]),
        );
        if remainder.iter().all(Zero::is_zero) {
            break;
        }
        sequence.push(primitive_integer_preserving_sign(
            remainder.into_iter().map(|value| -value).collect(),
        ));
    }
    sequence
}

fn sign_at(polynomial: &Polynomial, value: &BigRational) -> i8 {
    let numerator = value.numer();
    let denominator = value.denom();
    let mut scaled = polynomial.last().unwrap().clone();
    let mut denominator_power = BigInt::one();
    for coefficient in polynomial.iter().rev().skip(1) {
        denominator_power *= denominator;
        scaled = scaled * numerator + coefficient * &denominator_power;
    }
    if scaled.is_positive() {
        1
    } else if scaled.is_negative() {
        -1
    } else {
        0
    }
}

fn variations(sequence: &[Polynomial], value: &BigRational) -> usize {
    let mut previous = 0i8;
    let mut changes = 0;
    for polynomial in sequence {
        let sign = sign_at(polynomial, value);
        if sign != 0 {
            if previous != 0 && sign != previous {
                changes += 1;
            }
            previous = sign;
        }
    }
    changes
}

fn root_count(sequence: &[Polynomial], low: &BigRational, high: &BigRational) -> usize {
    variations(sequence, low)
        .checked_sub(variations(sequence, high))
        .expect("Sturm variations must decrease across an interval")
}

fn root_bound(polynomial: &Polynomial) -> BigInt {
    let lead = polynomial.last().unwrap().abs();
    let maximum = polynomial[..polynomial.len() - 1]
        .iter()
        .map(|value| {
            let (q, r) = value.abs().div_rem(&lead);
            q + if r.is_zero() { 0 } else { 1 }
        })
        .max()
        .unwrap_or_else(BigInt::zero);
    maximum + 1
}

fn isolate_recursive(
    sequence: &[Polynomial],
    interval: Interval,
    low_variations: usize,
    high_variations: usize,
    output: &mut Vec<Interval>,
) {
    let count = low_variations
        .checked_sub(high_variations)
        .expect("Sturm variations must decrease across an interval");
    if count == 0 {
        return;
    }
    if count == 1 {
        output.push(interval);
        return;
    }
    let Interval { low, high } = interval;
    let midpoint = (&low + &high) / BigInt::from(2);
    let middle_variations = variations(sequence, &midpoint);
    isolate_recursive(
        sequence,
        Interval::new(low, midpoint.clone()),
        low_variations,
        middle_variations,
        output,
    );
    isolate_recursive(
        sequence,
        Interval::new(midpoint, high),
        middle_variations,
        high_variations,
        output,
    );
}

#[derive(Debug)]
struct RootIsolation {
    initial: Interval,
    refined: Mutex<Interval>,
}

#[derive(Debug)]
pub(crate) struct RootIsolator {
    sequence: Vec<Polynomial>,
    roots: Vec<RootIsolation>,
}

impl RootIsolator {
    pub(crate) fn new(polynomial: &Polynomial) -> Self {
        let sequence = sturm(polynomial);
        assert_eq!(
            sequence.last().unwrap().len(),
            1,
            "root isolation requires a square-free polynomial"
        );
        let roots = if polynomial.len() == 2 {
            vec![Interval::point(BigRational::new(
                -&polynomial[0],
                polynomial[1].clone(),
            ))]
        } else {
            let bound = root_bound(polynomial);
            let low = BigRational::from_integer(-&bound);
            let high = BigRational::from_integer(bound);
            let low_variations = variations(&sequence, &low);
            let high_variations = variations(&sequence, &high);
            let count = low_variations
                .checked_sub(high_variations)
                .expect("Sturm variations must decrease across an interval");
            let mut roots = Vec::with_capacity(count);
            isolate_recursive(
                &sequence,
                Interval::new(low, high),
                low_variations,
                high_variations,
                &mut roots,
            );
            roots
        };
        let roots = roots
            .into_iter()
            .map(|initial| RootIsolation {
                refined: Mutex::new(initial.clone()),
                initial,
            })
            .collect();
        Self { sequence, roots }
    }

    pub(crate) fn polynomial(&self) -> &Polynomial {
        &self.sequence[0]
    }

    pub(crate) fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub(crate) fn initial_interval(&self, root_index: usize) -> &Interval {
        &self.roots[root_index].initial
    }

    pub(crate) fn root_count_in(&self, interval: &Interval) -> usize {
        root_count(&self.sequence, &interval.low, &interval.high)
    }

    pub(crate) fn unique_root_index_in(&self, interval: &Interval) -> Option<usize> {
        let global_low = BigRational::from_integer(-root_bound(&self.sequence[0]));
        let counting_low = global_low.min(&interval.low - BigInt::one());
        if interval.low == interval.high {
            return (sign_at(&self.sequence[0], &interval.low) == 0).then(|| {
                root_count(&self.sequence, &counting_low, &interval.low)
                    .checked_sub(1)
                    .expect("the selected point is a root")
            });
        }
        debug_assert_ne!(sign_at(&self.sequence[0], &interval.low), 0);
        debug_assert_ne!(sign_at(&self.sequence[0], &interval.high), 0);
        (self.root_count_in(interval) == 1)
            .then(|| root_count(&self.sequence, &counting_low, &interval.low))
    }

    pub(crate) fn strict_isolation(&self, root_index: usize, bits: usize) -> Interval {
        let mut precision = bits;
        loop {
            let interval = self.isolate_to_bits(root_index, precision);
            let margin = BigRational::new(BigInt::one(), BigInt::one() << precision);
            let expanded = Interval::new(&interval.low - &margin, &interval.high + &margin);
            if self.root_count_in(&expanded) == 1
                && sign_at(&self.sequence[0], &expanded.low) != 0
                && sign_at(&self.sequence[0], &expanded.high) != 0
            {
                return expanded;
            }
            precision = precision
                .checked_mul(2)
                .expect("root isolation precision overflow");
        }
    }

    pub(crate) fn isolate_to_bits(&self, root_index: usize, bits: usize) -> Interval {
        let target = BigRational::new(BigInt::one(), BigInt::one() << bits);
        let mut interval = self.roots[root_index]
            .refined
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let high_sign = sign_at(&self.sequence[0], &interval.high);
        if high_sign == 0 {
            *interval = Interval::point(interval.high.clone());
            return interval.clone();
        }
        while &interval.high - &interval.low > target {
            let midpoint = (&interval.low + &interval.high) / BigInt::from(2);
            let middle_sign = sign_at(&self.sequence[0], &midpoint);
            if middle_sign == 0 {
                *interval = Interval::point(midpoint);
                return interval.clone();
            }
            if middle_sign != high_sign {
                interval.low = midpoint;
            } else {
                interval.high = midpoint;
            }
        }
        interval.clone()
    }
}

pub(crate) fn select_overlapping_root(
    isolator: &RootIsolator,
    mut target: impl FnMut(usize) -> Interval,
) -> Result<(usize, Interval), AlgebraicError> {
    let mut bits = 32;
    loop {
        let target = target(bits);
        let matches: Vec<_> = (0..isolator.root_count())
            .filter(|&root_index| isolator.initial_interval(root_index).overlaps(&target))
            .filter_map(|root_index| {
                let isolation = isolator.isolate_to_bits(root_index, bits);
                isolation
                    .overlaps(&target)
                    .then_some((root_index, isolation))
            })
            .collect();
        if let [(root_index, isolation)] = matches.as_slice() {
            return Ok((*root_index, isolation.clone()));
        }
        bits = bits.checked_mul(2).ok_or(AlgebraicError::IsolationFailed)?;
    }
}

pub(crate) fn isolate(polynomial: &Polynomial) -> Vec<Interval> {
    RootIsolator::new(polynomial)
        .roots
        .into_iter()
        .map(|root| root.initial)
        .collect()
}

struct CanonicalRootCandidate {
    polynomial: Polynomial,
    root_index: usize,
    interval: Interval,
    isolator: Arc<RootIsolator>,
}

pub(crate) fn canonical_roots(
    polynomial: &Polynomial,
    bits: usize,
) -> Vec<(Polynomial, usize, Interval)> {
    let mut roots: Vec<_> =
        integer_polynomial_factorization::factor_square_free(&square_free(polynomial))
            .into_iter()
            .flat_map(|polynomial| {
                let isolator = Arc::new(RootIsolator::new(&polynomial));
                (0..isolator.root_count())
                    .map(|root_index| CanonicalRootCandidate {
                        interval: isolator.isolate_to_bits(root_index, bits),
                        polynomial: polynomial.clone(),
                        root_index,
                        isolator: isolator.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
    let mut precision = bits;
    loop {
        roots.sort_by(|left, right| {
            left.interval
                .low
                .cmp(&right.interval.low)
                .then_with(|| left.interval.high.cmp(&right.interval.high))
                .then_with(|| left.polynomial.cmp(&right.polynomial))
                .then_with(|| left.root_index.cmp(&right.root_index))
        });
        if roots
            .windows(2)
            .all(|pair| pair[0].interval.high < pair[1].interval.low)
        {
            return roots
                .into_iter()
                .map(|root| (root.polynomial, root.root_index, root.interval))
                .collect();
        }
        precision = precision
            .checked_mul(2)
            .expect("canonical root precision overflow");
        for root in &mut roots {
            root.interval = root.isolator.isolate_to_bits(root.root_index, precision);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn rational_sturm(polynomial: &Polynomial) -> Vec<RationalPolynomial> {
        let first = as_rational(polynomial);
        let second = derivative(&first);
        let mut sequence = vec![first, second];
        while sequence.last().unwrap().len() > 1 {
            let length = sequence.len();
            let (_, remainder) = div_rem_q(&sequence[length - 2], &sequence[length - 1]);
            if remainder.iter().all(Zero::is_zero) {
                break;
            }
            sequence.push(remainder.into_iter().map(|value| -value).collect());
        }
        sequence
    }

    fn rational_sign_at(polynomial: &RationalPolynomial, value: &BigRational) -> i8 {
        let evaluated = polynomial
            .iter()
            .rev()
            .fold(BigRational::zero(), |sum, coefficient| {
                sum * value + coefficient
            });
        if evaluated.is_positive() {
            1
        } else if evaluated.is_negative() {
            -1
        } else {
            0
        }
    }

    fn rational_variations(sequence: &[RationalPolynomial], value: &BigRational) -> usize {
        let mut previous = 0i8;
        let mut changes = 0;
        for polynomial in sequence {
            let sign = rational_sign_at(polynomial, value);
            if sign != 0 {
                if previous != 0 && sign != previous {
                    changes += 1;
                }
                previous = sign;
            }
        }
        changes
    }

    #[test]
    #[should_panic(expected = "interval bounds are ordered")]
    fn interval_rejects_reversed_bounds() {
        Interval::new(BigRational::one(), BigRational::zero());
    }

    #[test]
    fn integer_sign_evaluation_handles_large_denominators() {
        let polynomial = vec![(-17).into(), 4.into(), 0.into(), (-9).into(), 2.into()];
        let denominator = BigInt::one() << 200;
        let point = BigRational::new((&denominator << 1) + 1, denominator);
        assert_eq!(
            sign_at(&polynomial, &point),
            rational_sign_at(&as_rational(&polynomial), &point)
        );
    }

    #[test]
    fn refinement_handles_roots_on_interval_boundaries() {
        let rational_roots = RootIsolator::new(&vec![0.into(), (-1).into(), 0.into(), 1.into()]);
        let expected = [-1, 0, 1].map(|value| BigRational::from_integer(value.into()));
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                rational_roots.isolate_to_bits(index, 256),
                Interval::point(expected.clone())
            );
            assert_eq!(
                rational_roots.unique_root_index_in(&Interval::point(expected)),
                Some(index)
            );
        }

        let mixed_roots = RootIsolator::new(&vec![3.into(), (-3).into(), (-1).into(), 1.into()]);
        let positive_sqrt_three = mixed_roots.isolate_to_bits(2, 256);
        assert!(positive_sqrt_three.low > BigRational::one());
        assert!(positive_sqrt_three.high < BigRational::from_integer(2.into()));
        assert_eq!(mixed_roots.root_count_in(&positive_sqrt_three), 1);
    }

    #[test]
    fn refinement_reuses_the_narrowest_certified_interval() {
        let isolator = RootIsolator::new(&vec![(-2).into(), 0.into(), 1.into()]);
        let fine = isolator.isolate_to_bits(1, 256);
        assert_eq!(isolator.isolate_to_bits(1, 32), fine);
        assert_eq!(isolator.root_count_in(&fine), 1);
    }

    #[test]
    fn concurrent_refinement_keeps_one_certified_root() {
        let isolator = std::sync::Arc::new(RootIsolator::new(&vec![
            (-2).into(),
            0.into(),
            0.into(),
            1.into(),
        ]));
        let handles: Vec<_> = [32usize, 64, 128, 256]
            .into_iter()
            .map(|bits| {
                let isolator = isolator.clone();
                std::thread::spawn(move || (bits, isolator.isolate_to_bits(0, bits)))
            })
            .collect();
        for handle in handles {
            let (bits, interval) = handle.join().unwrap();
            assert!(
                &interval.high - &interval.low
                    <= BigRational::new(BigInt::one(), BigInt::one() << bits)
            );
            assert_eq!(isolator.root_count_in(&interval), 1);
        }
    }

    #[test]
    fn refinement_recovers_from_a_poisoned_cache_lock() {
        let isolator =
            std::sync::Arc::new(RootIsolator::new(&vec![(-2).into(), 0.into(), 1.into()]));
        let poisoning_thread = {
            let isolator = isolator.clone();
            std::thread::spawn(move || {
                let _guard = isolator.roots[1].refined.lock().unwrap();
                panic!("poison refinement cache for the recovery test");
            })
        };
        assert!(poisoning_thread.join().is_err());
        let interval = isolator.isolate_to_bits(1, 128);
        assert_eq!(isolator.root_count_in(&interval), 1);
    }

    #[test]
    #[should_panic(expected = "root isolation requires a square-free polynomial")]
    fn root_isolation_rejects_repeated_roots() {
        RootIsolator::new(&vec![(-3).into(), 7.into(), (-5).into(), 1.into()]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn integer_sturm_variations_match_rational_sturm(
            mut coefficients in prop::collection::vec(-8i32..=8, 2..=7),
            numerator in -16i32..=16,
            denominator in 1i32..=16,
        ) {
            if coefficients.last() == Some(&0) {
                *coefficients.last_mut().unwrap() = 1;
            }
            let polynomial: Polynomial = coefficients.into_iter().map(BigInt::from).collect();
            let point = BigRational::new(numerator.into(), denominator.into());
            prop_assert_eq!(
                variations(&sturm(&polynomial), &point),
                rational_variations(&rational_sturm(&polynomial), &point),
            );
        }

        #[test]
        fn refined_intervals_keep_their_selected_roots(
            mut coefficients in prop::collection::vec(-5i32..=5, 2..=7),
        ) {
            if coefficients.last() == Some(&0) {
                *coefficients.last_mut().unwrap() = 1;
            }
            let polynomial = square_free(
                &coefficients.into_iter().map(BigInt::from).collect::<Polynomial>()
            );
            let isolator = RootIsolator::new(&polynomial);
            let global_low = BigRational::from_integer(-root_bound(&polynomial));
            let target_width = BigRational::new(BigInt::one(), BigInt::one() << 64);
            for root_index in 0..isolator.root_count() {
                let coarse = isolator.initial_interval(root_index);
                let interval = isolator.isolate_to_bits(root_index, 64);
                let narrower = isolator.isolate_to_bits(root_index, 128);
                let strict = isolator.strict_isolation(root_index, 64);
                prop_assert_eq!(isolator.unique_root_index_in(&strict), Some(root_index));
                prop_assert!(interval.low >= coarse.low);
                prop_assert!(interval.high <= coarse.high);
                prop_assert!(narrower.low >= interval.low);
                prop_assert!(narrower.high <= interval.high);
                prop_assert!(&interval.high - &interval.low <= target_width);
                prop_assert_eq!(
                    root_count(&isolator.sequence, &global_low, &narrower.high),
                    root_index + 1,
                );
                if narrower.low == narrower.high {
                    prop_assert_eq!(sign_at(&polynomial, &narrower.low), 0);
                } else {
                    prop_assert_eq!(isolator.root_count_in(&narrower), 1);
                    prop_assert_ne!(sign_at(&polynomial, &narrower.low), 0);
                    prop_assert_ne!(sign_at(&polynomial, &narrower.high), 0);
                }
            }
        }

        #[test]
        fn primitive_pseudo_remainder_gcd_matches_rational_euclid(
            mut left in prop::collection::vec(-8i32..=8, 1..=6),
            mut right in prop::collection::vec(-8i32..=8, 1..=6),
        ) {
            if left.last() == Some(&0) {
                *left.last_mut().unwrap() = 1;
            }
            if right.last() == Some(&0) {
                *right.last_mut().unwrap() = 1;
            }
            let left: Polynomial = left.into_iter().map(BigInt::from).collect();
            let right: Polynomial = right.into_iter().map(BigInt::from).collect();
            let rational = gcd_q(as_rational(&left), as_rational(&right));
            let expected = if rational.len() == 1 {
                vec![BigInt::one()]
            } else {
                primitive(rational).unwrap()
            };
            prop_assert_eq!(gcd(&left, &right), expected);
        }
    }
}
