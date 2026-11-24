// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::polynomial::{Polynomial, primitive_integer};

/// Make a monic integer polynomial whose roots are `scale` times the input roots.
fn scale_roots_to_monic(polynomial: &Polynomial, scale: &BigInt) -> Polynomial {
    let leading = polynomial.last().expect("a polynomial is nonempty");
    assert!(
        scale.is_multiple_of(leading),
        "root scale must be a multiple of the leading coefficient"
    );
    let mut factor = BigInt::one();
    let mut scaled = Vec::with_capacity(polynomial.len());
    for coefficient in polynomial.iter().rev() {
        scaled.push(coefficient * &factor / leading);
        factor *= scale;
    }
    scaled.reverse();
    scaled
}

/// Return a monic polynomial's root power sums through `count`.
///
/// Index zero is the number of roots. Newton's identities compute every later
/// sum without constructing a splitting field or a multiplication matrix.
fn power_sums(polynomial: &Polynomial, count: usize) -> Vec<BigInt> {
    debug_assert_eq!(polynomial.last(), Some(&BigInt::one()));
    let degree = polynomial.len() - 1;
    let mut sums = Vec::with_capacity(count + 1);
    sums.push(BigInt::from(degree));
    for power in 1..=count {
        let mut sum = BigInt::zero();
        if power <= degree {
            for offset in 1..power {
                sum += &polynomial[degree - offset] * &sums[power - offset];
            }
            sum += &polynomial[degree - power] * power;
        } else {
            for offset in 1..=degree {
                sum += &polynomial[degree - offset] * &sums[power - offset];
            }
        }
        sums.push(-sum);
    }
    sums
}

/// Reconstruct the monic polynomial whose roots have the given power sums.
fn polynomial_from_power_sums(sums: &[BigInt]) -> Polynomial {
    let degree = sums.len() - 1;
    let mut polynomial = vec![BigInt::zero(); degree + 1];
    polynomial[degree] = BigInt::one();
    for offset in 1..=degree {
        let mut sum = sums[offset].clone();
        for previous in 1..offset {
            sum += &polynomial[degree - previous] * &sums[offset - previous];
        }
        let (coefficient, remainder) = (-sum).div_rem(&BigInt::from(offset));
        assert!(
            remainder.is_zero(),
            "root power sums must reconstruct an integer polynomial"
        );
        polynomial[degree - offset] = coefficient;
    }
    polynomial
}

/// Replace roots `r` by `r / scale` and return the primitive integer associate.
fn unscale_roots(mut polynomial: Polynomial, scale: &BigInt) -> Polynomial {
    let mut factor = BigInt::one();
    for coefficient in &mut polynomial {
        *coefficient *= &factor;
        factor *= scale;
    }
    primitive_integer(polynomial)
}

fn polynomial_from_composed_power_sums(
    left: &Polynomial,
    right: &Polynomial,
    result_scale: &BigInt,
    compose: impl FnOnce(&[BigInt], &[BigInt], usize) -> Vec<BigInt>,
) -> Polynomial {
    let degree = (left.len() - 1)
        .checked_mul(right.len() - 1)
        .expect("composed polynomial degree overflow");
    let left_sums = power_sums(left, degree);
    let right_sums = power_sums(right, degree);
    let polynomial = polynomial_from_power_sums(&compose(&left_sums, &right_sums, degree));
    unscale_roots(polynomial, result_scale)
}

/// Return a polynomial whose roots, with multiplicity, are all `a + b`.
pub(crate) fn sum_polynomial(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let scale = left.last().unwrap().lcm(right.last().unwrap());
    let left = scale_roots_to_monic(left, &scale);
    let right = scale_roots_to_monic(right, &scale);
    polynomial_from_composed_power_sums(&left, &right, &scale, |left, right, degree| {
        let mut sums = Vec::with_capacity(degree + 1);
        for power in 0..=degree {
            let mut binomial = BigInt::one();
            let mut sum = BigInt::zero();
            for left_power in 0..=power {
                sum += &left[left_power] * &right[power - left_power] * &binomial;
                if left_power != power {
                    binomial = binomial * (power - left_power) / BigInt::from(left_power + 1);
                }
            }
            sums.push(sum);
        }
        sums
    })
}

/// Return a polynomial whose roots, with multiplicity, are all `a * b`.
pub(crate) fn product_polynomial(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let left_scale = left.last().unwrap();
    let right_scale = right.last().unwrap();
    let result_scale = left_scale * right_scale;
    let left = scale_roots_to_monic(left, left_scale);
    let right = scale_roots_to_monic(right, right_scale);
    polynomial_from_composed_power_sums(&left, &right, &result_scale, |left, right, degree| {
        (0..=degree)
            .map(|power| &left[power] * &right[power])
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn z(values: &[i64]) -> Polynomial {
        values.iter().copied().map(BigInt::from).collect()
    }

    fn polynomial_from_roots(roots: impl IntoIterator<Item = i32>) -> Polynomial {
        roots
            .into_iter()
            .fold(vec![BigInt::one()], |polynomial, root| {
                let mut product = vec![BigInt::zero(); polynomial.len() + 1];
                for (degree, coefficient) in polynomial.into_iter().enumerate() {
                    product[degree] -= &coefficient * root;
                    product[degree + 1] += coefficient;
                }
                product
            })
    }

    fn polynomial_from_rational_roots(roots: impl IntoIterator<Item = (i32, i32)>) -> Polynomial {
        let polynomial = roots.into_iter().fold(
            vec![BigInt::one()],
            |polynomial, (numerator, denominator)| {
                let mut product = vec![BigInt::zero(); polynomial.len() + 1];
                for (degree, coefficient) in polynomial.into_iter().enumerate() {
                    product[degree] -= &coefficient * numerator;
                    product[degree + 1] += coefficient * denominator;
                }
                product
            },
        );
        primitive_integer(polynomial)
    }

    #[test]
    fn composes_quadratic_root_multisets() {
        let root_two = z(&[-2, 0, 1]);
        let root_three = z(&[-3, 0, 1]);
        assert_eq!(
            sum_polynomial(&root_two, &root_three),
            z(&[1, 0, -10, 0, 1])
        );
        assert_eq!(
            product_polynomial(&root_two, &root_three),
            z(&[36, 0, -12, 0, 1])
        );

        let imaginary_unit = z(&[1, 0, 1]);
        assert_eq!(
            sum_polynomial(&imaginary_unit, &imaginary_unit),
            z(&[0, 0, 4, 0, 1])
        );
        assert_eq!(
            product_polynomial(&imaginary_unit, &imaginary_unit),
            z(&[1, 0, -2, 0, 1])
        );
    }

    #[test]
    fn nonmonic_inputs_stay_primitive_and_integral() {
        let half = z(&[-1, 2]);
        let third = z(&[-1, 3]);
        assert_eq!(sum_polynomial(&half, &third), z(&[-5, 6]));
        assert_eq!(product_polynomial(&half, &third), z(&[-1, 6]));

        let inverse_root_two = z(&[-1, 0, 2]);
        let inverse_root_three = z(&[-1, 0, 3]);
        assert_eq!(
            sum_polynomial(&inverse_root_two, &inverse_root_three),
            z(&[1, 0, -60, 0, 36])
        );
        assert_eq!(
            product_polynomial(&inverse_root_two, &inverse_root_three),
            z(&[1, 0, -12, 0, 36])
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn composition_matches_explicit_integer_roots(
            left_roots in prop::collection::vec(-5i32..=5, 1..=4),
            right_roots in prop::collection::vec(-5i32..=5, 1..=4),
        ) {
            let left = polynomial_from_roots(left_roots.iter().copied());
            let right = polynomial_from_roots(right_roots.iter().copied());
            let sums = left_roots
                .iter()
                .flat_map(|left| right_roots.iter().map(move |right| left + right));
            let products = left_roots
                .iter()
                .flat_map(|left| right_roots.iter().map(move |right| left * right));
            prop_assert_eq!(sum_polynomial(&left, &right), polynomial_from_roots(sums));
            prop_assert_eq!(
                product_polynomial(&left, &right),
                polynomial_from_roots(products),
            );
        }

        #[test]
        fn composition_matches_explicit_rational_roots(
            left_roots in prop::collection::vec((-5i32..=5, 1i32..=5), 1..=3),
            right_roots in prop::collection::vec((-5i32..=5, 1i32..=5), 1..=3),
        ) {
            let left = polynomial_from_rational_roots(left_roots.iter().copied());
            let right = polynomial_from_rational_roots(right_roots.iter().copied());
            let sums = left_roots.iter().flat_map(|&(left_numerator, left_denominator)| {
                right_roots.iter().map(move |&(right_numerator, right_denominator)| {
                    (
                        left_numerator * right_denominator
                            + right_numerator * left_denominator,
                        left_denominator * right_denominator,
                    )
                })
            });
            let products = left_roots.iter().flat_map(|&(left_numerator, left_denominator)| {
                right_roots.iter().map(move |&(right_numerator, right_denominator)| {
                    (
                        left_numerator * right_numerator,
                        left_denominator * right_denominator,
                    )
                })
            });
            prop_assert_eq!(
                sum_polynomial(&left, &right),
                polynomial_from_rational_roots(sums),
            );
            prop_assert_eq!(
                product_polynomial(&left, &right),
                polynomial_from_rational_roots(products),
            );
        }
    }
}
