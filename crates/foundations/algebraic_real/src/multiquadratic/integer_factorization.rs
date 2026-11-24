// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

/// Return the prime factors of a machine-sized integer, with multiplicity.
///
/// Miller--Rabin with these seven witnesses is deterministic on all `u64`
/// inputs. Pollard's rho only finds divisors; primality is decided separately,
/// so no probabilistic result can enter the canonical number representation.
fn prime_factors_u64(n: u64) -> Vec<u64> {
    fn collect(n: u64, factors: &mut Vec<u64>) {
        if n == 1 {
            return;
        }
        if is_prime_u64(n) {
            factors.push(n);
            return;
        }
        let d = pollard_rho(n);
        collect(d, factors);
        collect(n / d, factors);
    }

    let mut factors = Vec::new();
    collect(n, &mut factors);
    factors.sort_unstable();
    factors
}

fn is_prime_u64(n: u64) -> bool {
    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    const WITNESSES: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

    if n < 2 {
        return false;
    }
    for p in SMALL_PRIMES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }

    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'witness: for a in WITNESSES {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(modulus)) as u64
}

fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((u128::from(a) + u128::from(b)) % u128::from(modulus)) as u64
}

fn pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn pollard_rho(n: u64) -> u64 {
    if n.is_multiple_of(2) {
        return 2;
    }
    if n.is_multiple_of(3) {
        return 3;
    }

    // Changing c deterministically gives a fresh polynomial when a rho walk
    // happens to cycle without exposing a factor.
    for c in 1.. {
        let mut x = 2;
        let mut y = 2;
        let mut divisor = 1;
        while divisor == 1 {
            x = add_mod(mul_mod(x, x, n), c, n);
            y = add_mod(mul_mod(y, y, n), c, n);
            y = add_mod(mul_mod(y, y, n), c, n);
            divisor = gcd_u64(x.abs_diff(y), n);
        }
        if divisor != n {
            return divisor;
        }
    }
    unreachable!()
}

fn include_factor(coeff: &mut BigUint, radicand: &mut BigUint, p: BigUint, exponent: u32) {
    *coeff *= p.pow(exponent / 2);
    if exponent % 2 == 1 {
        *radicand *= p;
    }
}

fn include_u64_factors(coeff: &mut BigUint, radicand: &mut BigUint, n: u64) {
    let factors = prime_factors_u64(n);
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut end = i + 1;
        while end < factors.len() && factors[end] == p {
            end += 1;
        }
        include_factor(coeff, radicand, BigUint::from(p), (end - i) as u32);
        i = end;
    }
}

fn is_mersenne_prime(value: &BigUint) -> bool {
    let power_of_two = value + 1u32;
    let exponent = power_of_two.bits().saturating_sub(1);
    if exponent < 2
        || BigUint::one() << usize::try_from(exponent).expect("BigUint bit count fits usize")
            != power_of_two
        || !is_prime_u64(exponent)
    {
        return false;
    }
    if exponent == 2 {
        return true;
    }
    let mut lucas_lehmer = BigUint::from(4u32);
    for _ in 0..exponent - 2 {
        lucas_lehmer = (&lucas_lehmer * &lucas_lehmer + value - 2u32) % value;
    }
    lucas_lehmer.is_zero()
}

pub(crate) fn squarefree_decompose(mut n: BigUint) -> (BigUint, BigUint) {
    if n.is_zero() {
        return (BigUint::zero(), BigUint::zero());
    }

    // Recursive square-root extraction reaches large rational perfect squares
    // very often. Recognize them before factorization: integer square root is
    // polynomial-time, while trial division of a large square can be ruinous.
    let root = n.sqrt();
    if &root * &root == n {
        return (root, BigUint::one());
    }
    if is_mersenne_prime(&n) {
        return (BigUint::one(), n);
    }

    let mut coeff = BigUint::one();
    let mut radicand = BigUint::one();
    if let Some(n) = n.to_u64() {
        include_u64_factors(&mut coeff, &mut radicand, n);
        return (coeff, radicand);
    }

    // Above `u64`, keep the exact trial-division fallback. As soon as the
    // residual fits in a machine word, finish it with Miller--Rabin/Pollard rho.
    // This is slower than probable-prime BigUint libraries, but it cannot
    // misclassify a composite and corrupt the canonical representation.
    let mut d = BigUint::from(2u32);
    while &d * &d <= n {
        let mut exponent = 0;
        while (&n % &d).is_zero() {
            n /= &d;
            exponent += 1;
        }
        if exponent != 0 {
            include_factor(&mut coeff, &mut radicand, d.clone(), exponent);
            if let Some(residual) = n.to_u64() {
                include_u64_factors(&mut coeff, &mut radicand, residual);
                return (coeff, radicand);
            }
        }
        d += if d == BigUint::from(2u32) { 1u32 } else { 2u32 };
    }
    if n > BigUint::one() {
        radicand *= n;
    }
    (coeff, radicand)
}

pub(crate) fn smallest_prime_factor(n: &BigUint) -> BigUint {
    assert!(n > &BigUint::one(), "smallest prime factor requires n > 1");
    if let Some(n) = n.to_u64() {
        return BigUint::from(prime_factors_u64(n)[0]);
    }

    if (n % 2u32).is_zero() {
        return BigUint::from(2u32);
    }
    let mut d = BigUint::from(3u32);
    while &d * &d <= *n {
        if (n % &d).is_zero() {
            return d;
        }
        d += 2u32;
    }
    n.clone()
}

pub(crate) fn distinct_prime_factors(n: &BigUint) -> Vec<BigUint> {
    let mut residual = n.clone();
    let mut factors = Vec::new();
    while residual > BigUint::one() {
        let p = smallest_prime_factor(&residual);
        factors.push(p.clone());
        while (&residual % &p).is_zero() {
            residual /= &p;
        }
    }
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primality_is_deterministic_across_u64() {
        assert!(is_prime_u64(2_305_843_009_213_693_951)); // 2^61 - 1
        assert!(is_prime_u64(18_446_744_073_709_551_557));
        assert!(!is_prime_u64(u64::MAX));
        assert!(!is_prime_u64(341_550_071_728_321));
    }

    #[test]
    fn lucas_lehmer_certifies_large_mersenne_primes() {
        assert!(is_mersenne_prime(&((BigUint::one() << 127usize) - 1u32)));
        assert!(!is_mersenne_prime(&BigUint::from(2_047u32)));
    }

    #[test]
    fn pollard_rho_factors_machine_sized_semiprimes() {
        assert_eq!(prime_factors_u64(600_851_475_143), [71, 839, 1_471, 6_857]);
        let p = 2_147_483_647;
        assert_eq!(prime_factors_u64(p * p), [p, p]);
    }

    #[test]
    fn squarefree_decomposition_handles_large_machine_values() {
        let prime = 2_305_843_009_213_693_951u64;
        assert_eq!(
            squarefree_decompose(BigUint::from(prime)),
            (BigUint::one(), BigUint::from(prime))
        );

        let p = 2_147_483_647u64;
        assert_eq!(
            squarefree_decompose(BigUint::from(p * p)),
            (BigUint::from(p), BigUint::one())
        );
    }

    #[test]
    fn distinct_factors_are_sorted_and_exact() {
        assert_eq!(
            distinct_prime_factors(&BigUint::from(2u32 * 3 * 5 * 11)),
            [2u32, 3, 5, 11].map(BigUint::from)
        );
        assert!(distinct_prime_factors(&BigUint::one()).is_empty());
    }
}
