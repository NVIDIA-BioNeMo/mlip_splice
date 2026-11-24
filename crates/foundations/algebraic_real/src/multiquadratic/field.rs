// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeMap;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use super::Multiquadratic;

/// One echelon row of the F₂-vector space of rational square classes.
///
/// The first mask is a square class: bit `i` is set when atom `i` occurs to an
/// odd power. The second expresses that class in the independent basis
/// discovered from the input radicands. Both are `BigUint`, so neither the
/// number of atoms nor the rank is bounded by a machine word.
type EchelonRow = (BigUint, BigUint);

/// Refine the input radicands into pairwise-coprime square-class atoms.
///
/// An atom may be composite when all its primes always occur together. Prime
/// factorization is unnecessary: exact gcd splitting exposes precisely the
/// independent columns needed for F₂ row reduction.
fn square_class_atoms(radicands: impl Iterator<Item = BigUint>) -> Vec<BigUint> {
    let mut atoms: Vec<BigUint> = Vec::new();
    for mut remaining in radicands.filter(|radicand| !radicand.is_one()) {
        let mut index = 0;
        while index < atoms.len() && !remaining.is_one() {
            let common = atoms[index].gcd(&remaining);
            if !common.is_one() {
                let complement = &atoms[index] / &common;
                atoms[index] = common.clone();
                remaining /= common;
                if !complement.is_one() {
                    atoms.push(complement);
                }
            }
            index += 1;
        }
        if !remaining.is_one() {
            atoms.push(remaining);
        }
    }
    atoms.sort();
    atoms
}

/// The square class of `radicand` as a bitmask over `atoms`.
///
/// Bit `i` is set when `atoms[i]` divides the squarefree radicand, that is when
/// that atom appears to an odd power.
fn atom_support(radicand: &BigUint, atoms: &[BigUint]) -> BigUint {
    let mut support = BigUint::ZERO;
    for (index, atom) in atoms.iter().enumerate() {
        if (radicand % atom).is_zero() {
            support.set_bit(index as u64, true);
        }
    }
    support
}

/// Index of the highest set bit: the pivot atom of a nonzero square class.
fn highest_set_bit(mask: &BigUint) -> Option<u64> {
    mask.bits().checked_sub(1)
}

/// Add echelon rows into `support` while its pivot has a row, tracking coordinates.
///
/// Bitwise xor is addition over F₂. Each step clears the current pivot bit, so
/// the pivot strictly decreases and the loop terminates.
fn reduce_by_echelon(
    rows: &BTreeMap<u64, EchelonRow>,
    support: &mut BigUint,
    coordinates: &mut BigUint,
) {
    while let Some(pivot) = highest_set_bit(support) {
        let Some((row_support, row_coordinates)) = rows.get(&pivot) else {
            break;
        };
        *support ^= row_support;
        *coordinates ^= row_coordinates;
    }
}

/// Row-reduce `supports` into an echelon basis keyed by pivot atom.
///
/// The row created for the `n`-th independent support carries generator bit `n`,
/// so the rank is the number of rows and the generators are independent.
fn square_class_echelon(supports: &[BigUint]) -> BTreeMap<u64, EchelonRow> {
    let mut rows: BTreeMap<u64, EchelonRow> = BTreeMap::new();
    for support in supports {
        let mut residue = support.clone();
        let mut coordinates = BigUint::one() << rows.len();
        reduce_by_echelon(&rows, &mut residue, &mut coordinates);
        if let Some(pivot) = highest_set_bit(&residue) {
            rows.insert(pivot, (residue, coordinates));
        }
    }
    rows
}

/// Express one square class in the generator basis of `rows`.
fn echelon_coordinates(rows: &BTreeMap<u64, EchelonRow>, support: &BigUint) -> BigUint {
    let mut residue = support.clone();
    let mut coordinates = BigUint::ZERO;
    reduce_by_echelon(rows, &mut residue, &mut coordinates);
    assert!(
        residue.is_zero(),
        "square class was not in its own generated span"
    );
    coordinates
}

/// The rank and the per-term generator coordinates of the square classes in `terms`.
///
/// Bit `j` of the `i`-th mask is set exactly when flipping the sign of generator
/// `j` negates term `i`.
fn square_class_coordinates(terms: &[(BigRational, BigUint)]) -> (usize, Vec<BigUint>) {
    let atoms = square_class_atoms(terms.iter().map(|(_, radicand)| radicand.clone()));
    let supports: Vec<BigUint> = terms
        .iter()
        .map(|(_, radicand)| atom_support(radicand, &atoms))
        .collect();
    let rows = square_class_echelon(&supports);
    let coordinates = supports
        .iter()
        .map(|support| echelon_coordinates(&rows, support))
        .collect();
    (rows.len(), coordinates)
}

/// The number of independent square classes in `value`.
#[cfg(test)]
fn square_class_rank(value: &Multiquadratic) -> usize {
    square_class_coordinates(&value.terms()).0
}

/// Rebuild `terms` with the signs selected by one generator-flip mask.
fn conjugate_terms(
    terms: &[(BigRational, BigUint)],
    coordinates: &[usize],
    flips: usize,
) -> Multiquadratic {
    Multiquadratic::from_canonical_terms(terms.iter().zip(coordinates).map(
        |((coefficient, radicand), coordinate)| {
            let coefficient = if (coordinate & flips).count_ones().is_odd() {
                -coefficient
            } else {
                coefficient.clone()
            };
            (coefficient, radicand.clone())
        },
    ))
}

#[cfg(test)]
fn degree_from_rank(rank: usize) -> BigUint {
    BigUint::one() << rank
}

fn as_rational(value: &Multiquadratic) -> BigRational {
    if value.is_zero() {
        return BigRational::zero();
    }
    match value.as_single_term() {
        Some((coefficient, radicand)) if radicand.is_one() => coefficient,
        _ => panic!("a Galois-invariant result was not rational: {value}"),
    }
}

fn primitive_integer_coefficients(coefficients: Vec<BigRational>) -> Vec<BigInt> {
    let denominator_lcm = coefficients.iter().fold(BigInt::one(), |lcm, coefficient| {
        lcm.lcm(coefficient.denom())
    });
    let mut integers: Vec<_> = coefficients
        .iter()
        .map(|coefficient| coefficient.numer() * (&denominator_lcm / coefficient.denom()))
        .collect();
    let content = integers
        .iter()
        .fold(BigInt::zero(), |gcd, coefficient| gcd.gcd(coefficient));
    if !content.is_zero() && !content.is_one() {
        for coefficient in &mut integers {
            *coefficient /= &content;
        }
    }
    if integers.last().is_some_and(Signed::is_negative) {
        for coefficient in &mut integers {
            *coefficient = -&*coefficient;
        }
    }
    integers
}

impl Multiquadratic {
    /// Return all embeddings of `self` into the reals, in increasing order.
    ///
    /// They are obtained by independently changing signs on a basis of the
    /// rational square classes appearing in `self`. The values are distinct;
    /// their number is `[Q(self) : Q]`.
    pub(crate) fn conjugates(&self) -> Vec<Self> {
        let terms = self.terms();
        let (rank, coordinates) = square_class_coordinates(&terms);
        let count = 1usize
            .checked_shl(u32::try_from(rank).expect("square-class rank fits u32"))
            .expect("algebraic degree exceeds addressable memory");
        let coordinates: Vec<_> = coordinates
            .into_iter()
            .map(|coordinate| {
                coordinate
                    .to_usize()
                    .expect("addressable conjugate coordinates fit usize")
            })
            .collect();
        let mut conjugates: Vec<_> = (0..count)
            .map(|flips| conjugate_terms(&terms, &coordinates, flips))
            .collect();
        conjugates.sort_by(|left, right| match (left - right).signum() {
            sign if sign < 0 => Ordering::Less,
            0 => Ordering::Equal,
            _ => Ordering::Greater,
        });
        conjugates
    }

    /// Return the primitive integer associate of the minimal polynomial over Q.
    ///
    /// Coefficients are integers in ascending power order, have gcd one, and
    /// have positive leading coefficient. Thus `[-2, 0, 1]` means `t² - 2`.
    pub(crate) fn minimal_polynomial(&self) -> Vec<BigInt> {
        let mut polynomial = vec![Multiquadratic::one()];
        for conjugate in self.conjugates() {
            let mut product = vec![Multiquadratic::zero(); polynomial.len() + 1];
            for (degree, coefficient) in polynomial.into_iter().enumerate() {
                product[degree] = &product[degree] - &(&coefficient * &conjugate);
                product[degree + 1] = &product[degree + 1] + &coefficient;
            }
            polynomial = product;
        }
        primitive_integer_coefficients(polynomial.iter().map(as_rational).collect())
    }

    /// Return the field trace `Tr_{Q(self)/Q}(self)`.
    #[cfg(test)]
    pub(crate) fn trace(&self) -> BigRational {
        let rational_part = self
            .terms()
            .into_iter()
            .find_map(|(coefficient, radicand)| radicand.is_one().then_some(coefficient))
            .unwrap_or_else(BigRational::zero);
        rational_part * BigInt::from(degree_from_rank(square_class_rank(self)))
    }

    /// Return the field norm `N_{Q(self)/Q}(self)`.
    #[cfg(test)]
    pub(crate) fn norm(&self) -> BigRational {
        let product = self
            .conjugates()
            .into_iter()
            .fold(Multiquadratic::one(), |product, conjugate| {
                product * conjugate
            });
        as_rational(&product)
    }

    /// Return the extension degree `[Q(self) : Q]`.
    #[cfg(test)]
    pub(crate) fn field_degree(&self) -> BigUint {
        degree_from_rank(square_class_rank(self))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use proptest::prelude::*;

    fn q(numerator: i64, denominator: i64) -> BigRational {
        BigRational::new(numerator.into(), denominator.into())
    }

    fn independent_binary_rank(masks: impl IntoIterator<Item = u8>) -> usize {
        let mut rows = [0u8; 8];
        let mut rank = 0;
        for mut mask in masks {
            while mask != 0 {
                let pivot = (u8::BITS - 1 - mask.leading_zeros()) as usize;
                if rows[pivot] == 0 {
                    rows[pivot] = mask;
                    rank += 1;
                    break;
                }
                mask ^= rows[pivot];
            }
        }
        rank
    }

    #[test]
    fn rational_field_data() {
        for (value, polynomial) in [
            (Multiquadratic::zero(), vec![0.into(), 1.into()]),
            (Multiquadratic::integer(3), vec![(-3).into(), 1.into()]),
            (
                Multiquadratic::rational(q(1, 2)),
                vec![(-1).into(), 2.into()],
            ),
        ] {
            assert_eq!(value.conjugates(), vec![value.clone()]);
            assert_eq!(value.minimal_polynomial(), polynomial);
            assert_eq!(value.trace(), as_rational(&value));
            assert_eq!(value.norm(), as_rational(&value));
            assert_eq!(value.field_degree(), BigUint::one());
        }
    }

    #[test]
    fn quadratic_field_data() {
        let root = Multiquadratic::sqrt(2u32);
        assert_eq!(root.conjugates(), vec![-&root, root.clone()]);
        assert_eq!(
            root.minimal_polynomial(),
            vec![(-2).into(), 0.into(), 1.into()]
        );
        assert_eq!(root.trace(), BigRational::zero());
        assert_eq!(root.norm(), BigRational::from_integer((-2).into()));
        assert_eq!(root.field_degree(), BigUint::from(2u32));
    }

    #[test]
    fn quartic_field_data() {
        let value =
            Multiquadratic::integer(1) + Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32);
        assert_eq!(
            value.minimal_polynomial(),
            vec![(-8).into(), 16.into(), (-4).into(), (-4).into(), 1.into()]
        );
        assert_eq!(value.trace(), BigRational::from_integer(4.into()));
        assert_eq!(value.norm(), BigRational::from_integer((-8).into()));
        assert_eq!(value.field_degree(), BigUint::from(4u32));
        assert_eq!(value.conjugates().len(), 4);
    }

    #[test]
    fn dependent_prime_generators_do_not_inflate_degree() {
        let root6 = Multiquadratic::sqrt(6u32);
        assert_eq!(root6.field_degree(), BigUint::from(2u32));
        assert_eq!(root6.conjugates(), vec![-&root6, root6.clone()]);

        let value = Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32) + &root6;
        assert_eq!(value.field_degree(), BigUint::from(4u32));
        assert_eq!(value.conjugates().len(), 4);

        let dependent =
            Multiquadratic::sqrt(6u32) + Multiquadratic::sqrt(10u32) + Multiquadratic::sqrt(15u32);
        assert_eq!(dependent.field_degree(), BigUint::from(4u32));
        assert_eq!(dependent.conjugates().len(), 4);
    }

    #[test]
    fn polynomial_vanishes_exactly() {
        let samples = [
            Multiquadratic::rational(q(-7, 3)),
            Multiquadratic::sqrt(6u32),
            Multiquadratic::integer(1) + Multiquadratic::sqrt(2u32),
            Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32),
            Multiquadratic::integer(1) + Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32),
        ];
        for value in samples {
            let evaluation = value
                .minimal_polynomial()
                .into_iter()
                .rev()
                .fold(Multiquadratic::zero(), |acc, coefficient| {
                    acc * &value + Multiquadratic::integer(coefficient)
                });
            assert!(evaluation.is_zero(), "polynomial did not vanish at {value}");
        }
    }

    #[test]
    fn fractional_polynomial_is_primitive_integral() {
        let value =
            (Multiquadratic::integer(1) + Multiquadratic::sqrt(2u32)) / Multiquadratic::integer(3);
        assert_eq!(
            value.minimal_polynomial(),
            vec![(-1).into(), (-6).into(), 9.into()]
        );
    }

    #[test]
    fn trace_and_degree_support_more_than_machine_word_generators() {
        let primes = [
            2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
            83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
            179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
            271, 277, 281, 283, 293, 307, 311,
        ];
        let value = primes
            .iter()
            .fold(Multiquadratic::integer(7), |sum, &prime| {
                sum + Multiquadratic::sqrt(prime)
            });
        let degree = BigUint::one() << primes.len();
        assert_eq!(value.field_degree(), degree);
        assert_eq!(
            value.trace(),
            BigRational::from_integer(BigInt::from(7) * BigInt::from(degree))
        );
    }

    #[test]
    fn composite_atom_does_not_require_large_factorization() {
        let p = BigUint::from(2_305_843_009_213_693_951u64);
        let q = BigUint::from(2_147_483_647u64);
        let value = Multiquadratic::sqrt(p.clone()) * Multiquadratic::sqrt(q.clone());
        assert_eq!(value.field_degree(), BigUint::from(2u32));
        assert_eq!(value.conjugates(), vec![-&value, value.clone()]);
        assert_eq!(
            value.minimal_polynomial(),
            vec![-BigInt::from(p * q), BigInt::zero(), BigInt::one()]
        );
    }

    #[test]
    fn polynomial_coefficients_grow_beyond_i128() {
        let integer = BigInt::one() << 200usize;
        let value = Multiquadratic::integer(integer.clone()) + Multiquadratic::sqrt(2u32);
        assert_eq!(
            value.minimal_polynomial(),
            vec![&integer * &integer - 2, -2 * integer, BigInt::one()]
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn field_invariants_agree_exactly(
            a in -5i32..=5,
            b in -5i32..=5,
            c in -5i32..=5,
            d in -5i32..=5,
            denominator in 1i32..=5,
        ) {
            let coefficient = |n: i32| {
                Multiquadratic::rational(BigRational::new(n.into(), denominator.into()))
            };
            let value = coefficient(a)
                + coefficient(b) * Multiquadratic::sqrt(2u32)
                + coefficient(c) * Multiquadratic::sqrt(3u32)
                + coefficient(d) * Multiquadratic::sqrt(6u32);
            let conjugates = value.conjugates();
            prop_assert_eq!(BigUint::from(conjugates.len()), value.field_degree());
            prop_assert_eq!(
                conjugates
                    .iter()
                    .map(Multiquadratic::terms)
                    .collect::<HashSet<_>>()
                    .len(),
                conjugates.len(),
            );
            for pair in conjugates.windows(2) {
                prop_assert!((&pair[0] - &pair[1]).signum() < 0);
            }

            let explicit_trace = conjugates
                .iter()
                .fold(Multiquadratic::zero(), |sum, conjugate| sum + conjugate);
            prop_assert_eq!(as_rational(&explicit_trace), value.trace());

            let explicit_norm = conjugates
                .iter()
                .fold(Multiquadratic::one(), |product, conjugate| product * conjugate);
            prop_assert_eq!(as_rational(&explicit_norm), value.norm());

            let polynomial = value.minimal_polynomial();
            prop_assert_eq!(BigUint::from(polynomial.len()), value.field_degree() + 1u32);
            for conjugate in conjugates {
                let evaluation = polynomial.iter().rev().fold(
                    Multiquadratic::zero(),
                    |acc, coefficient| acc * &conjugate + Multiquadratic::integer(coefficient.clone()),
                );
                prop_assert!(evaluation.is_zero());
            }

            let degree = polynomial.len() - 1;
            prop_assert_eq!(
                BigRational::new(-polynomial[degree - 1].clone(), polynomial[degree].clone()),
                value.trace(),
            );
            let norm_sign = if degree.is_multiple_of(2) { 1 } else { -1 };
            prop_assert_eq!(
                BigRational::new(
                    BigInt::from(norm_sign) * &polynomial[0],
                    polynomial[degree].clone(),
                ),
                value.norm(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn rank_four_stress_against_independent_oracle(
            coefficients in prop::collection::vec(-2i32..=2, 12),
            denominator in 1i32..=4,
        ) {
            // Masks independently encode the prime classes 2, 3, 5, and 7.
            let radicands = [1u32, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35];
            let masks = [0u8, 1, 2, 4, 3, 8, 5, 9, 6, 10, 7, 12];
            let value = coefficients
                .iter()
                .zip(radicands)
                .fold(Multiquadratic::zero(), |sum, (&coefficient, radicand)| {
                    sum + Multiquadratic::term(
                        BigRational::new(coefficient.into(), denominator.into()),
                        radicand,
                    )
                });

            let rank = independent_binary_rank(
                coefficients
                    .iter()
                    .zip(masks)
                    .filter_map(|(&coefficient, mask)| (coefficient != 0).then_some(mask)),
            );
            prop_assert_eq!(value.field_degree(), BigUint::one() << rank);

            let conjugates = value.conjugates();
            prop_assert_eq!(conjugates.len(), 1usize << rank);
            prop_assert_eq!(
                conjugates
                    .iter()
                    .map(Multiquadratic::terms)
                    .collect::<HashSet<_>>()
                    .len(),
                conjugates.len(),
            );
            for pair in conjugates.windows(2) {
                prop_assert!((&pair[0] - &pair[1]).signum() < 0);
            }

            let explicit_trace = conjugates
                .iter()
                .fold(Multiquadratic::zero(), |sum, conjugate| sum + conjugate);
            let explicit_norm = conjugates
                .iter()
                .fold(Multiquadratic::one(), |product, conjugate| product * conjugate);
            prop_assert_eq!(as_rational(&explicit_trace), value.trace());
            prop_assert_eq!(as_rational(&explicit_norm), value.norm());

            let polynomial = value.minimal_polynomial();
            prop_assert_eq!(polynomial.len(), conjugates.len() + 1);
            for conjugate in &conjugates {
                let evaluation = polynomial.iter().rev().fold(
                    Multiquadratic::zero(),
                    |acc, coefficient| acc * conjugate + Multiquadratic::integer(coefficient.clone()),
                );
                prop_assert!(evaluation.is_zero());
            }

            // Exercise division and square-root extraction in the same rank-four field.
            let divisor = Multiquadratic::one()
                + Multiquadratic::sqrt(2u32)
                + Multiquadratic::sqrt(3u32)
                + Multiquadratic::sqrt(5u32)
                + Multiquadratic::sqrt(7u32);
            prop_assert_eq!((&value / &divisor) * &divisor, value.clone());
            let square = &value * &value;
            let nonnegative = if value.signum() < 0 { -value } else { value };
            prop_assert_eq!(square.sqrt_exact(), Some(nonnegative));
        }
    }
}
