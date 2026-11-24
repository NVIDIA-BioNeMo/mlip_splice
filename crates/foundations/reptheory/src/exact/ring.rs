// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::{Add, Mul, Neg, Sub};

use algebraic_real::AlgebraicReal;

use super::ComplexAlgebraicReal;

/// A commutative ring with the four reference-taking arithmetic operators, so
/// `Mat<K>` can be generic over the real `AlgebraicReal` and its complexification.
pub trait Ring: Clone + PartialEq
where
    for<'a> &'a Self: Add<&'a Self, Output = Self>
        + Sub<&'a Self, Output = Self>
        + Mul<&'a Self, Output = Self>
        + Neg<Output = Self>,
{
    fn zero() -> Self;

    fn one() -> Self;

    fn is_zero(&self) -> bool;
}

pub trait Conjugate {
    fn conj(&self) -> Self;
}

impl Ring for AlgebraicReal {
    fn zero() -> Self {
        AlgebraicReal::zero()
    }

    fn one() -> Self {
        AlgebraicReal::one()
    }

    fn is_zero(&self) -> bool {
        AlgebraicReal::is_zero(self)
    }
}

impl Conjugate for AlgebraicReal {
    fn conj(&self) -> Self {
        self.clone()
    }
}

impl Ring for ComplexAlgebraicReal {
    fn zero() -> Self {
        ComplexAlgebraicReal::zero()
    }

    fn one() -> Self {
        ComplexAlgebraicReal::one()
    }

    fn is_zero(&self) -> bool {
        ComplexAlgebraicReal::is_zero(self)
    }
}

impl Conjugate for ComplexAlgebraicReal {
    fn conj(&self) -> Self {
        ComplexAlgebraicReal::conj(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_sum<K: Ring>(xs: &[K]) -> K
    where
        for<'a> &'a K: Add<&'a K, Output = K>
            + Sub<&'a K, Output = K>
            + Mul<&'a K, Output = K>
            + Neg<Output = K>,
    {
        let mut acc = K::zero();
        for x in xs {
            acc = &acc + x;
        }
        acc
    }

    #[test]
    fn ring_sum_over_real() {
        let xs = [
            AlgebraicReal::sqrt(2u32),
            AlgebraicReal::one(),
            AlgebraicReal::sqrt(2u32),
        ];
        let two_sqrt2 = &AlgebraicReal::sqrt(2u32) + &AlgebraicReal::sqrt(2u32);
        assert_eq!(ring_sum(&xs), &two_sqrt2 + &AlgebraicReal::one());
    }

    #[test]
    fn ring_sum_over_complex() {
        let xs = [
            ComplexAlgebraicReal::imag(AlgebraicReal::one()),
            ComplexAlgebraicReal::real(AlgebraicReal::one()),
            ComplexAlgebraicReal::imag(AlgebraicReal::one()),
        ];
        let expected = ComplexAlgebraicReal {
            re: AlgebraicReal::one(),
            im: &AlgebraicReal::one() + &AlgebraicReal::one(),
        };
        assert_eq!(ring_sum(&xs), expected);
    }

    #[test]
    fn ring_identities() {
        assert!(<AlgebraicReal as Ring>::zero().is_zero());
        assert!(!<AlgebraicReal as Ring>::one().is_zero());
        assert!(<ComplexAlgebraicReal as Ring>::zero().is_zero());
        assert!(!<ComplexAlgebraicReal as Ring>::one().is_zero());
    }

    #[test]
    fn conjugate_is_identity_on_real() {
        let x = &AlgebraicReal::sqrt(3u32) + &AlgebraicReal::one();
        assert_eq!(Conjugate::conj(&x), x);
    }

    #[test]
    fn conjugate_flips_imaginary_on_complex() {
        let z = ComplexAlgebraicReal {
            re: AlgebraicReal::sqrt(2u32),
            im: AlgebraicReal::integer(3),
        };
        assert_eq!(
            Conjugate::conj(&z),
            ComplexAlgebraicReal {
                re: AlgebraicReal::sqrt(2u32),
                im: AlgebraicReal::integer(-3),
            }
        );
    }
}
