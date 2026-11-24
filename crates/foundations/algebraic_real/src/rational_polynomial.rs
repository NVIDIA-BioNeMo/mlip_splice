// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_rational::BigRational;
use num_traits::{One, Zero};

pub(crate) type RationalPolynomial = Vec<BigRational>;

pub(crate) fn trim(mut polynomial: RationalPolynomial) -> RationalPolynomial {
    while polynomial.len() > 1 && polynomial.last().is_some_and(Zero::is_zero) {
        polynomial.pop();
    }
    if polynomial.is_empty() {
        polynomial.push(BigRational::zero());
    }
    polynomial
}

pub(crate) fn add(left: &[BigRational], right: &[BigRational]) -> RationalPolynomial {
    let mut sum = vec![BigRational::zero(); left.len().max(right.len())];
    for (degree, coefficient) in left.iter().enumerate() {
        sum[degree] += coefficient;
    }
    for (degree, coefficient) in right.iter().enumerate() {
        sum[degree] += coefficient;
    }
    trim(sum)
}

pub(crate) fn subtract(left: &[BigRational], right: &[BigRational]) -> RationalPolynomial {
    let mut difference = vec![BigRational::zero(); left.len().max(right.len())];
    for (degree, coefficient) in left.iter().enumerate() {
        difference[degree] += coefficient;
    }
    for (degree, coefficient) in right.iter().enumerate() {
        difference[degree] -= coefficient;
    }
    trim(difference)
}

pub(crate) fn multiply(left: &[BigRational], right: &[BigRational]) -> RationalPolynomial {
    if left.is_empty() || right.is_empty() {
        return vec![BigRational::zero()];
    }
    let mut product = vec![BigRational::zero(); left.len() + right.len() - 1];
    for (left_degree, left_coefficient) in left.iter().enumerate() {
        for (right_degree, right_coefficient) in right.iter().enumerate() {
            product[left_degree + right_degree] += left_coefficient * right_coefficient;
        }
    }
    trim(product)
}

pub(crate) fn div_rem(
    numerator: &[BigRational],
    denominator: &[BigRational],
) -> (RationalPolynomial, RationalPolynomial) {
    let mut remainder = trim(numerator.to_vec());
    let denominator = trim(denominator.to_vec());
    assert_ne!(denominator, [BigRational::zero()]);
    if remainder.len() < denominator.len() {
        return (vec![BigRational::zero()], remainder);
    }
    let mut quotient = vec![BigRational::zero(); remainder.len() - denominator.len() + 1];
    while remainder.len() >= denominator.len() && remainder != [BigRational::zero()] {
        let shift = remainder.len() - denominator.len();
        let coefficient = remainder.last().unwrap() / denominator.last().unwrap();
        quotient[shift] = coefficient.clone();
        for (degree, value) in denominator.iter().enumerate() {
            remainder[degree + shift] -= &coefficient * value;
        }
        remainder = trim(remainder);
    }
    (trim(quotient), remainder)
}

pub(crate) fn remainder(
    numerator: &[BigRational],
    denominator: &[BigRational],
) -> RationalPolynomial {
    div_rem(numerator, denominator).1
}

pub(crate) fn monic_gcd(
    mut left: RationalPolynomial,
    mut right: RationalPolynomial,
) -> RationalPolynomial {
    left = trim(left);
    right = trim(right);
    while right != [BigRational::zero()] {
        let next = remainder(&left, &right);
        left = right;
        right = next;
    }
    if left == [BigRational::zero()] {
        return left;
    }
    let leading = left.last().unwrap().clone();
    trim(left.into_iter().map(|value| value / &leading).collect())
}

pub(crate) fn inverse_mod(value: &[BigRational], modulus: &[BigRational]) -> RationalPolynomial {
    let (mut old_remainder, mut new_remainder) = (trim(modulus.to_vec()), trim(value.to_vec()));
    let (mut old_coefficient, mut new_coefficient) =
        (vec![BigRational::zero()], vec![BigRational::one()]);
    while new_remainder != [BigRational::zero()] {
        let (quotient, remainder) = div_rem(&old_remainder, &new_remainder);
        let coefficient = subtract(&old_coefficient, &multiply(&quotient, &new_coefficient));
        (old_remainder, new_remainder) = (new_remainder, remainder);
        (old_coefficient, new_coefficient) = (new_coefficient, coefficient);
    }
    assert_eq!(
        old_remainder.len(),
        1,
        "every nonzero element of a number field is invertible"
    );
    let scale = old_remainder[0].recip();
    remainder(
        &old_coefficient
            .into_iter()
            .map(|coefficient| coefficient * &scale)
            .collect::<Vec<_>>(),
        modulus,
    )
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use proptest::prelude::*;

    use super::*;

    fn integers(coefficients: Vec<i16>) -> RationalPolynomial {
        coefficients
            .into_iter()
            .map(|value| BigRational::from_integer(BigInt::from(value)))
            .collect()
    }

    #[test]
    fn empty_coefficient_slices_are_canonical_zero() {
        assert_eq!(
            multiply(&[], &[BigRational::one()]),
            vec![BigRational::zero()]
        );
        assert_eq!(
            multiply(&[BigRational::one()], &[]),
            vec![BigRational::zero()]
        );
        assert_eq!(monic_gcd(vec![], vec![]), vec![BigRational::zero()]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn division_reconstructs_the_dividend(
            numerator in prop::collection::vec(-20i16..=20, 0..10),
            mut denominator in prop::collection::vec(-20i16..=20, 1..7),
        ) {
            if denominator.last() == Some(&0) {
                *denominator.last_mut().unwrap() = 1;
            }
            let numerator = trim(integers(numerator));
            let denominator = trim(integers(denominator));
            let (quotient, remainder) = div_rem(&numerator, &denominator);
            prop_assert_eq!(
                add(&multiply(&quotient, &denominator), &remainder),
                numerator,
            );
            prop_assert!(
                remainder == [BigRational::zero()] || remainder.len() < denominator.len()
            );
        }

        #[test]
        fn addition_and_subtraction_are_inverse(
            left in prop::collection::vec(-20i16..=20, 0..10),
            right in prop::collection::vec(-20i16..=20, 0..10),
        ) {
            let left = trim(integers(left));
            let right = trim(integers(right));
            prop_assert_eq!(subtract(&add(&left, &right), &right), left);
        }

        #[test]
        fn modular_inverse_multiplies_to_one(
            constant in -20i16..=20,
            linear in -20i16..=20,
        ) {
            prop_assume!(constant != 0 || linear != 0);
            let value = integers(vec![constant, linear]);
            let modulus = integers(vec![1, 0, 1]);
            let inverse = inverse_mod(&value, &modulus);
            prop_assert_eq!(
                remainder(&multiply(&value, &inverse), &modulus),
                vec![BigRational::one()],
            );
        }
    }
}
