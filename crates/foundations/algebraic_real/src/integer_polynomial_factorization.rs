// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};

use crate::polynomial::{self, Polynomial, primitive_integer, trim};
use crate::prime_field_polynomial;

fn multiply(left: &[BigInt], right: &[BigInt]) -> Polynomial {
    let mut product = vec![BigInt::zero(); left.len() + right.len() - 1];
    for (left_degree, left_coefficient) in left.iter().enumerate() {
        for (right_degree, right_coefficient) in right.iter().enumerate() {
            product[left_degree + right_degree] += left_coefficient * right_coefficient;
        }
    }
    trim(product)
}

fn factor_coefficient_bound(polynomial: &Polynomial) -> BigInt {
    let norm_squared = polynomial.iter().fold(BigUint::ZERO, |sum, coefficient| {
        sum + coefficient.magnitude() * coefficient.magnitude()
    });
    let norm_floor = norm_squared.sqrt();
    let norm_ceiling = if &norm_floor * &norm_floor == norm_squared {
        norm_floor
    } else {
        norm_floor + 1u32
    };
    let top = polynomial.len() - 2;
    let bottom = top / 2;
    let binomial = (1..=bottom).fold(BigInt::one(), |value, index| {
        value * BigInt::from(top + 1 - index) / BigInt::from(index)
    });
    BigInt::from(polynomial.last().unwrap().magnitude().clone())
        * binomial
        * BigInt::from(norm_ceiling)
}

fn product(factors: &[Polynomial]) -> Polynomial {
    factors.iter().fold(vec![BigInt::one()], |product, factor| {
        multiply(&product, factor)
    })
}

fn product_mod(factors: &[Polynomial], modulus: &BigInt) -> Polynomial {
    factors.iter().fold(vec![BigInt::one()], |product, factor| {
        multiply(&product, factor)
            .into_iter()
            .map(|coefficient| coefficient.mod_floor(modulus))
            .collect()
    })
}

fn lift_factors(
    polynomial: &Polynomial,
    prime_factors: &prime_field_polynomial::PrimeFieldFactors,
    bound: &BigInt,
) -> (Vec<Polynomial>, BigInt) {
    let prime = BigInt::from(prime_factors.prime());
    let mut final_modulus = prime.clone();
    let target_modulus = bound * 2 + 1;
    while final_modulus < target_modulus {
        final_modulus *= &prime;
    }
    let leading = polynomial.last().unwrap();
    let inverse = leading.extended_gcd(&final_modulus);
    assert!(
        inverse.gcd.is_one(),
        "the lifting prime must not divide the leading coefficient"
    );
    let leading_inverse = inverse.x.mod_floor(&final_modulus);

    let mut modulus = prime.clone();
    let mut factors: Vec<Polynomial> = prime_factors
        .factors()
        .iter()
        .map(|factor| factor.iter().copied().map(BigInt::from).collect())
        .collect();
    while modulus < final_modulus {
        let next_modulus = &modulus * &prime;
        let approximate = product_mod(&factors, &next_modulus);
        let mut error = Vec::with_capacity(polynomial.len());
        for (degree, coefficient) in polynomial.iter().enumerate() {
            let target = (coefficient * &leading_inverse).mod_floor(&next_modulus);
            let difference = (target - &approximate[degree]).mod_floor(&next_modulus);
            let (quotient, remainder) = difference.div_rem(&modulus);
            assert!(
                remainder.is_zero(),
                "Hensel factors must multiply to the normalized source modulo the current modulus"
            );
            error.push(quotient);
        }
        let corrections = prime_factors.corrections(&trim(error));
        for (factor, correction) in factors.iter_mut().zip(corrections) {
            for (coefficient, correction) in factor.iter_mut().zip(correction) {
                *coefficient += &modulus * correction;
            }
        }
        modulus = next_modulus;
    }
    assert!(
        product_mod(&factors, &final_modulus)
            .iter()
            .zip(polynomial)
            .all(|(actual, expected)| {
                (actual - expected * &leading_inverse).is_multiple_of(&final_modulus)
            }),
        "lifted factors must reconstruct the normalized source"
    );
    (factors, final_modulus)
}

fn centered_mod(value: BigInt, modulus: &BigInt) -> BigInt {
    let residue = value.mod_floor(modulus);
    if &residue * 2 > *modulus {
        residue - modulus
    } else {
        residue
    }
}

fn multiply_centered_mod(left: &[BigInt], right: &[BigInt], modulus: &BigInt) -> Polynomial {
    multiply(left, right)
        .into_iter()
        .map(|coefficient| centered_mod(coefficient, modulus))
        .collect()
}

fn exact_division(numerator: &Polynomial, denominator: &Polynomial) -> Option<Polynomial> {
    let mut remainder = numerator.clone();
    if remainder.len() < denominator.len() {
        return None;
    }
    let mut quotient = vec![BigInt::zero(); remainder.len() - denominator.len() + 1];
    while remainder.len() >= denominator.len() && remainder != [BigInt::zero()] {
        let shift = remainder.len() - denominator.len();
        let (coefficient, residual) = remainder
            .last()
            .unwrap()
            .div_rem(denominator.last().unwrap());
        if !residual.is_zero() {
            return None;
        }
        quotient[shift] = coefficient.clone();
        for (degree, value) in denominator.iter().enumerate() {
            remainder[degree + shift] -= &coefficient * value;
        }
        remainder = trim(remainder);
    }
    (remainder == [BigInt::zero()]).then(|| primitive_integer(quotient))
}

fn candidate_from_positions(
    lifted: &[Polynomial],
    remaining: &[usize],
    positions: &[usize],
    modulus: &BigInt,
    scale: &BigInt,
) -> Polynomial {
    positions
        .iter()
        .fold(vec![scale.clone()], |candidate, position| {
            multiply_centered_mod(&candidate, &lifted[remaining[*position]], modulus)
        })
}

fn find_divisor_with_size(
    current: &Polynomial,
    lifted: &[Polynomial],
    remaining: &[usize],
    modulus: &BigInt,
    size: usize,
    start: usize,
    positions: &mut Vec<usize>,
) -> Option<(Polynomial, Polynomial, Vec<usize>)> {
    if positions.len() == size {
        let candidate = primitive_integer(candidate_from_positions(
            lifted,
            remaining,
            positions,
            modulus,
            current.last().unwrap(),
        ));
        if candidate.len() <= 1 || candidate.len() >= current.len() {
            return None;
        }
        return exact_division(current, &candidate)
            .map(|quotient| (candidate, quotient, positions.clone()));
    }
    let needed = size - positions.len();
    for position in start..=remaining.len() - needed {
        positions.push(position);
        if let Some(divisor) = find_divisor_with_size(
            current,
            lifted,
            remaining,
            modulus,
            size,
            position + 1,
            positions,
        ) {
            return Some(divisor);
        }
        positions.pop();
    }
    None
}

fn reconstruct_factors(
    polynomial: Polynomial,
    lifted: &[Polynomial],
    modulus: &BigInt,
) -> Vec<Polynomial> {
    let mut current = polynomial;
    let mut remaining: Vec<_> = (0..lifted.len()).collect();
    let mut factors = Vec::new();
    while remaining.len() > 1 {
        let mut found = None;
        for size in 1..=remaining.len() / 2 {
            found = find_divisor_with_size(
                &current,
                lifted,
                &remaining,
                modulus,
                size,
                0,
                &mut Vec::with_capacity(size),
            );
            if found.is_some() {
                break;
            }
        }
        let Some((factor, quotient, positions)) = found else {
            break;
        };
        for position in positions.into_iter().rev() {
            remaining.remove(position);
        }
        factors.push(factor);
        current = quotient;
    }
    factors.push(current);
    factors.sort();
    factors
}

/// Factor a square-free primitive integer polynomial into primitive irreducibles.
pub(crate) fn factor_square_free(polynomial: &Polynomial) -> Vec<Polynomial> {
    let polynomial = primitive_integer(polynomial.clone());
    assert!(
        polynomial.len() >= 2,
        "factorization needs a nonconstant polynomial"
    );
    assert_eq!(
        polynomial::square_free(&polynomial),
        polynomial,
        "factor_square_free requires a square-free polynomial"
    );
    let prime_factors = prime_field_polynomial::factor_square_free_reduction(&polynomial);
    if prime_factors.factors().len() == 1 {
        return vec![polynomial];
    }
    let bound = factor_coefficient_bound(&polynomial);
    let (lifted, modulus) = lift_factors(&polynomial, &prime_factors, &bound);
    let factors = reconstruct_factors(polynomial.clone(), &lifted, &modulus);
    assert_eq!(
        product(&factors),
        polynomial,
        "reconstructed factors must multiply to the source"
    );
    factors
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn z(coefficients: &[i64]) -> Polynomial {
        coefficients.iter().copied().map(BigInt::from).collect()
    }

    #[test]
    fn factors_monic_polynomials_after_multifactor_lifting() {
        let factors = [z(&[1, 17, 1]), z(&[-3, 0, 1]), z(&[5, -2, 0, 1])];
        let polynomial = product(&factors);
        assert_eq!(factor_square_free(&polynomial), {
            let mut expected = factors.to_vec();
            expected.sort();
            expected
        });
    }

    #[test]
    fn lifting_recovers_nonmonic_primitive_factors() {
        let factors = [z(&[1, 2]), z(&[-2, 3]), z(&[7, -1, 4])];
        let polynomial = product(&factors);
        assert_eq!(factor_square_free(&polynomial), {
            let mut expected = factors.map(primitive_integer).to_vec();
            expected.sort();
            expected
        });
    }

    #[test]
    fn irreducible_polynomial_needs_no_lifting() {
        let polynomial = z(&[-1, -1, 0, 0, 0, 1]);
        assert_eq!(factor_square_free(&polynomial), [polynomial]);
    }

    #[test]
    fn reconstruction_handles_large_coefficients() {
        let factors = [z(&[1_000_003, -77, 1]), z(&[-999_983, 41, 0, 1])];
        let polynomial = product(&factors);
        assert_eq!(factor_square_free(&polynomial), {
            let mut expected = factors.to_vec();
            expected.sort();
            expected
        });
    }

    #[test]
    fn factors_inputs_that_defeated_kronecker_search() {
        let swinnerton_dyer = z(&[576, 0, -960, 0, 352, 0, -40, 0, 1]);
        assert_eq!(
            factor_square_free(&swinnerton_dyer).as_slice(),
            std::slice::from_ref(&swinnerton_dyer)
        );

        let cyclotomic = z(&[1, 0, 0, 0, 0, 0, 0, 0, 1]);
        let polynomial = multiply(&swinnerton_dyer, &cyclotomic);
        let mut expected = vec![swinnerton_dyer, cyclotomic];
        expected.sort();
        assert_eq!(factor_square_free(&polynomial), expected);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn recovers_distinct_integer_roots(
            roots in prop::collection::btree_set(-20i32..=20, 1..=8),
        ) {
            let factors: Vec<_> = roots
                .into_iter()
                .map(|root| vec![BigInt::from(-root), BigInt::one()])
                .collect();
            let polynomial = product(&factors);
            let mut expected = factors;
            expected.sort();
            prop_assert_eq!(factor_square_free(&polynomial), expected);
        }
    }
}
