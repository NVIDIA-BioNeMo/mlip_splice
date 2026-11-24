// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use crate::AlgebraicReal;

fn exact_nth_root(value: &BigRational, degree: u32) -> Option<BigRational> {
    let magnitude = value.numer().magnitude().nth_root(degree);
    let numerator = if value.is_negative() {
        -BigInt::from(magnitude)
    } else {
        BigInt::from(magnitude)
    };
    let denominator = BigInt::from(value.denom().magnitude().nth_root(degree));
    (&numerator.pow(degree) == value.numer() && &denominator.pow(degree) == value.denom())
        .then(|| BigRational::new(numerator, denominator))
}

pub(crate) fn nth_root(mut value: BigRational, mut degree: u32) -> AlgebraicReal {
    if value.is_zero() {
        return AlgebraicReal::zero();
    }
    let mut unfactored = degree;
    let mut prime = 2;
    let mut prime_divisors = Vec::new();
    while prime <= unfactored / prime {
        if unfactored.is_multiple_of(prime) {
            prime_divisors.push(prime);
            while unfactored.is_multiple_of(prime) {
                unfactored /= prime;
            }
        }
        prime += if prime == 2 { 1 } else { 2 };
    }
    if unfactored > 1 {
        prime_divisors.push(unfactored);
    }
    for prime in prime_divisors {
        while degree.is_multiple_of(prime) {
            let Some(root) = exact_nth_root(&value, prime) else {
                break;
            };
            value = root;
            degree /= prime;
        }
    }
    if degree == 1 {
        return AlgebraicReal::rational(value);
    }
    if degree == 2 {
        return AlgebraicReal::sqrt_rational(value)
            .expect("an even-degree rational root must have a nonnegative radicand");
    }

    // Capelli's theorem makes this binomial irreducible because its radicand
    // is not a p-th power for any prime p dividing the degree. The exceptional
    // -4 Q^4 case cannot apply: a degree divisible by four requires a positive radicand.
    let mut polynomial = vec![BigInt::ZERO; degree as usize + 1];
    polynomial[0] = -value.numer();
    polynomial[degree as usize] = value.denom().clone();
    let root_index = usize::from(degree.is_multiple_of(2));
    AlgebraicReal::from_irreducible_polynomial(polynomial, root_index)
}
