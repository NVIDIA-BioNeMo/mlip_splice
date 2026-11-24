// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use crate::polynomial::Polynomial;

type Residues = Vec<u64>;

#[derive(Clone, Debug)]
pub(crate) struct PrimeFieldFactors {
    prime: u64,
    factors: Vec<Residues>,
    cofactor_inverses: Vec<Residues>,
}

fn trim(mut polynomial: Residues) -> Residues {
    while polynomial.len() > 1 && polynomial.last() == Some(&0) {
        polynomial.pop();
    }
    if polynomial.is_empty() {
        polynomial.push(0);
    }
    polynomial
}

fn add_product(value: u64, left: u64, right: u64, prime: u64) -> u64 {
    ((u128::from(value) + u128::from(left) * u128::from(right)) % u128::from(prime)) as u64
}

fn subtract_product(value: u64, left: u64, right: u64, prime: u64) -> u64 {
    let product = ((u128::from(left) * u128::from(right)) % u128::from(prime)) as u64;
    if value >= product {
        value - product
    } else {
        prime - (product - value)
    }
}

fn integer_power(mut base: u64, mut exponent: u64, prime: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = add_product(0, result, base, prime);
        }
        base = add_product(0, base, base, prime);
        exponent >>= 1;
    }
    result
}

fn scale(mut polynomial: Residues, factor: u64, prime: u64) -> Residues {
    for coefficient in &mut polynomial {
        *coefficient = add_product(0, *coefficient, factor, prime);
    }
    trim(polynomial)
}

fn monic(polynomial: Residues, prime: u64) -> Residues {
    let inverse = integer_power(*polynomial.last().unwrap(), prime - 2, prime);
    scale(polynomial, inverse, prime)
}

fn subtract(left: &[u64], right: &[u64], prime: u64) -> Residues {
    let mut difference = vec![0; left.len().max(right.len())];
    for (index, coefficient) in difference.iter_mut().enumerate() {
        let left = left.get(index).copied().unwrap_or(0);
        let right = right.get(index).copied().unwrap_or(0);
        *coefficient = if left >= right {
            left - right
        } else {
            prime - (right - left)
        };
    }
    trim(difference)
}

fn multiply(left: &[u64], right: &[u64], prime: u64) -> Residues {
    let mut product = vec![0; left.len() + right.len() - 1];
    for (left_degree, left_coefficient) in left.iter().enumerate() {
        for (right_degree, right_coefficient) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            product[degree] = add_product(
                product[degree],
                *left_coefficient,
                *right_coefficient,
                prime,
            );
        }
    }
    trim(product)
}

fn div_rem(numerator: &[u64], denominator: &[u64], prime: u64) -> (Residues, Residues) {
    let mut remainder = trim(numerator.to_vec());
    let denominator = trim(denominator.to_vec());
    assert_ne!(denominator, [0]);
    if remainder.len() < denominator.len() {
        return (vec![0], remainder);
    }
    let inverse = integer_power(*denominator.last().unwrap(), prime - 2, prime);
    let mut quotient = vec![0; remainder.len() - denominator.len() + 1];
    while remainder.len() >= denominator.len() && remainder != [0] {
        let shift = remainder.len() - denominator.len();
        let coefficient = add_product(0, *remainder.last().unwrap(), inverse, prime);
        quotient[shift] = coefficient;
        for (index, value) in denominator.iter().enumerate() {
            remainder[index + shift] =
                subtract_product(remainder[index + shift], coefficient, *value, prime);
        }
        remainder = trim(remainder);
    }
    (trim(quotient), remainder)
}

fn remainder(numerator: &[u64], denominator: &[u64], prime: u64) -> Residues {
    div_rem(numerator, denominator, prime).1
}

fn reduced_product(left: &[u64], right: &[u64], polynomial: &[u64], prime: u64) -> Residues {
    remainder(&multiply(left, right, prime), polynomial, prime)
}

fn polynomial_power(
    mut base: Residues,
    mut exponent: u64,
    polynomial: &[u64],
    prime: u64,
) -> Residues {
    let mut result = vec![1];
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = reduced_product(&result, &base, polynomial, prime);
        }
        base = reduced_product(&base, &base, polynomial, prime);
        exponent >>= 1;
    }
    result
}

fn gcd(mut left: Residues, mut right: Residues, prime: u64) -> Residues {
    while right != [0] {
        let next = remainder(&left, &right, prime);
        left = right;
        right = next;
    }
    monic(left, prime)
}

fn derivative(polynomial: &[u64], prime: u64) -> Residues {
    trim(
        polynomial
            .iter()
            .enumerate()
            .skip(1)
            .map(|(degree, coefficient)| add_product(0, *coefficient, degree as u64 % prime, prime))
            .collect(),
    )
}

fn inverse_mod(value: &[u64], modulus: &[u64], prime: u64) -> Residues {
    let (mut old_remainder, mut new_remainder) = (modulus.to_vec(), value.to_vec());
    let (mut old_coefficient, mut new_coefficient) = (vec![0], vec![1]);
    while new_remainder != [0] {
        let (quotient, remainder) = div_rem(&old_remainder, &new_remainder, prime);
        let coefficient = subtract(
            &old_coefficient,
            &multiply(&quotient, &new_coefficient, prime),
            prime,
        );
        (old_remainder, new_remainder) = (new_remainder, remainder);
        (old_coefficient, new_coefficient) = (new_coefficient, coefficient);
    }
    assert_eq!(old_remainder.len(), 1, "modular factors must be coprime");
    let inverse = integer_power(old_remainder[0], prime - 2, prime);
    remainder(&scale(old_coefficient, inverse, prime), modulus, prime)
}

fn modulo(polynomial: &Polynomial, prime: u64) -> Residues {
    let prime_big = BigInt::from(prime);
    trim(
        polynomial
            .iter()
            .map(|coefficient| coefficient.mod_floor(&prime_big).to_u64().unwrap())
            .collect(),
    )
}

fn nullspace(mut matrix: Vec<Vec<u64>>, prime: u64) -> Vec<Residues> {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut pivot_columns = Vec::new();
    for column in 0..columns {
        let Some(pivot) = (pivot_columns.len()..rows).find(|&row| matrix[row][column] != 0) else {
            continue;
        };
        matrix.swap(pivot_columns.len(), pivot);
        let pivot_row = pivot_columns.len();
        let inverse = integer_power(matrix[pivot_row][column], prime - 2, prime);
        for value in &mut matrix[pivot_row] {
            *value = add_product(0, *value, inverse, prime);
        }
        for row in 0..rows {
            if row == pivot_row || matrix[row][column] == 0 {
                continue;
            }
            let factor = matrix[row][column];
            let pivot_values = matrix[pivot_row].clone();
            for (entry, value) in matrix[row].iter_mut().enumerate().skip(column) {
                *value = subtract_product(*value, factor, pivot_values[entry], prime);
            }
        }
        pivot_columns.push(column);
    }

    let free_columns: Vec<_> = (0..columns)
        .filter(|column| !pivot_columns.contains(column))
        .collect();
    free_columns
        .into_iter()
        .map(|free| {
            let mut vector = vec![0; columns];
            vector[free] = 1;
            for (row, pivot) in pivot_columns.iter().copied().enumerate() {
                let coefficient = matrix[row][free];
                vector[pivot] = if coefficient == 0 {
                    0
                } else {
                    prime - coefficient
                };
            }
            trim(vector)
        })
        .collect()
}

fn berlekamp_factor_square_free(polynomial: Residues, prime: u64) -> Vec<Residues> {
    let polynomial = monic(polynomial, prime);
    let degree = polynomial.len() - 1;
    if degree <= 1 {
        return vec![polynomial];
    }

    let x = vec![0, 1];
    let frobenius = polynomial_power(x, prime, &polynomial, prime);
    let mut column = vec![1];
    let mut matrix = vec![vec![0; degree]; degree];
    let mut power = 0;
    while power < degree {
        for (row, coefficient) in column.iter().copied().enumerate() {
            matrix[row][power] = coefficient;
        }
        matrix[power][power] = if matrix[power][power] == 0 {
            prime - 1
        } else {
            matrix[power][power] - 1
        };
        column = reduced_product(&column, &frobenius, &polynomial, prime);
        power += 1;
    }
    let basis = nullspace(matrix, prime);
    let factor_count = basis.len();
    let mut factors = vec![polynomial];
    'basis: for vector in basis {
        for constant in 0..prime {
            let mut shifted = vector.clone();
            if shifted.is_empty() {
                shifted.push(0);
            }
            shifted[0] = if shifted[0] >= constant {
                shifted[0] - constant
            } else {
                prime - (constant - shifted[0])
            };
            shifted = trim(shifted);
            let mut split = Vec::new();
            for factor in factors {
                if factor.len() <= 2 {
                    split.push(factor);
                    continue;
                }
                let divisor = gcd(factor.clone(), shifted.clone(), prime);
                if divisor.len() == 1 || divisor.len() == factor.len() {
                    split.push(factor);
                } else {
                    let (quotient, remainder) = div_rem(&factor, &divisor, prime);
                    assert_eq!(remainder, [0]);
                    split.push(monic(divisor, prime));
                    split.push(monic(quotient, prime));
                }
            }
            factors = split;
            if factors.len() == factor_count {
                break 'basis;
            }
        }
    }
    assert_eq!(
        factors.len(),
        factor_count,
        "Berlekamp basis must separate every modular factor"
    );
    factors.sort();
    factors
}

fn is_prime(value: u64) -> bool {
    value >= 2 && (2..=value.isqrt()).all(|divisor| !value.is_multiple_of(divisor))
}

pub(crate) fn factor_square_free_reduction(polynomial: &Polynomial) -> PrimeFieldFactors {
    let mut prime = 2;
    let mut good_primes = 0;
    let mut best: Option<(u64, Vec<Residues>)> = None;
    while good_primes < 8 {
        if is_prime(prime) {
            let reduced = modulo(polynomial, prime);
            if reduced.len() == polynomial.len()
                && gcd(reduced.clone(), derivative(&reduced, prime), prime).len() == 1
            {
                good_primes += 1;
                let factors = berlekamp_factor_square_free(monic(reduced, prime), prime);
                if factors.len() == 1 {
                    best = Some((prime, factors));
                    break;
                }
                if best
                    .as_ref()
                    .is_none_or(|(_, previous)| factors.len() < previous.len())
                {
                    best = Some((prime, factors));
                }
            }
        }
        prime = prime
            .checked_add(1)
            .expect("ran out of machine-sized primes");
    }
    let (prime, factors) = best.expect("a square-free integer polynomial has a good prime");
    let complete_product = factors
        .iter()
        .fold(vec![1], |product, factor| multiply(&product, factor, prime));
    let cofactor_inverses = factors
        .iter()
        .map(|factor| {
            let (cofactor, remainder) = div_rem(&complete_product, factor, prime);
            assert_eq!(remainder, [0]);
            inverse_mod(&cofactor, factor, prime)
        })
        .collect();
    PrimeFieldFactors {
        prime,
        factors,
        cofactor_inverses,
    }
}

impl PrimeFieldFactors {
    pub(crate) fn prime(&self) -> u64 {
        self.prime
    }

    pub(crate) fn factors(&self) -> &[Residues] {
        &self.factors
    }

    pub(crate) fn corrections(&self, error: &Polynomial) -> Vec<Residues> {
        let error = modulo(error, self.prime);
        self.factors
            .iter()
            .zip(&self.cofactor_inverses)
            .map(|(factor, inverse)| reduced_product(&error, inverse, factor, self.prime))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn z(coefficients: &[i64]) -> Polynomial {
        coefficients.iter().copied().map(BigInt::from).collect()
    }

    fn product(factors: &[Residues], prime: u64) -> Residues {
        factors
            .iter()
            .fold(vec![1], |product, factor| multiply(&product, factor, prime))
    }

    #[test]
    fn berlekamp_separates_irreducible_factors() {
        let polynomial = z(&[-6, 11, -6, 1]);
        let factors = factor_square_free_reduction(&polynomial);
        assert_eq!(
            product(factors.factors(), factors.prime()),
            modulo(&polynomial, factors.prime())
        );
        assert_eq!(factors.factors().len(), 3);
    }

    #[test]
    fn irreducible_reduction_is_a_certificate() {
        let polynomial = z(&[-1, -1, 0, 0, 0, 1]);
        let factors = factor_square_free_reduction(&polynomial);
        assert_eq!(factors.factors().len(), 1);
    }

    #[test]
    fn cofactor_inverses_solve_multifactor_correction() {
        let polynomial = z(&[-6, 11, -6, 1]);
        let factors = factor_square_free_reduction(&polynomial);
        let error = z(&[7, -4, 9]);
        let corrections = factors.corrections(&error);
        let reconstructed =
            corrections
                .iter()
                .enumerate()
                .fold(vec![0], |sum, (index, correction)| {
                    let cofactor = factors
                        .factors()
                        .iter()
                        .enumerate()
                        .filter(|(other, _)| *other != index)
                        .fold(vec![1], |product, (_, factor)| {
                            multiply(&product, factor, factors.prime())
                        });
                    let term = multiply(correction, &cofactor, factors.prime());
                    let mut result = sum;
                    result.resize(result.len().max(term.len()), 0);
                    for (degree, coefficient) in term.into_iter().enumerate() {
                        result[degree] = (result[degree] + coefficient) % factors.prime();
                    }
                    trim(result)
                });
        assert_eq!(reconstructed, modulo(&error, factors.prime()));
    }

    fn is_reducible_by_enumeration(polynomial: &[u64], prime: u64) -> bool {
        let degree = polynomial.len() - 1;
        for factor_degree in 1..=degree / 2 {
            for mut encoded in 0..prime.pow(factor_degree as u32) {
                let mut factor = Vec::with_capacity(factor_degree + 1);
                for _ in 0..factor_degree {
                    factor.push(encoded % prime);
                    encoded /= prime;
                }
                factor.push(1);
                if remainder(polynomial, &factor, prime) == [0] {
                    return true;
                }
            }
        }
        false
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn berlekamp_matches_exhaustive_search(
            coefficients in prop::collection::vec(0u8..=4, 2..=7),
            prime in prop::sample::select(vec![2u64, 3, 5]),
        ) {
            let mut residues: Vec<_> = coefficients
                .into_iter()
                .map(|coefficient| u64::from(coefficient) % prime)
                .collect();
            *residues.last_mut().unwrap() = 1;
            residues = trim(residues);
            let derivative = derivative(&residues, prime);
            prop_assume!(gcd(residues.clone(), derivative, prime).len() == 1);
            let factors = berlekamp_factor_square_free(residues.clone(), prime);
            prop_assert_eq!(product(&factors, prime), residues.clone());
            prop_assert_eq!(factors.len() > 1, is_reducible_by_enumeration(&residues, prime));
        }
    }
}
