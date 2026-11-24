// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_rational::{BigRational, Ratio};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Signed, ToPrimitive, Zero};

#[derive(Clone, Debug)]
pub(crate) enum Rational {
    Small(Ratio<i128>),
    Big(BigRational),
}

impl Rational {
    pub(crate) fn one() -> Self {
        Rational::Small(Ratio::one())
    }

    pub(crate) fn from_bigrational(b: BigRational) -> Self {
        match (b.numer().to_i128(), b.denom().to_i128()) {
            (Some(n), Some(d)) => Rational::Small(Ratio::new_raw(n, d)),
            _ => Rational::Big(b),
        }
    }

    pub(crate) fn from_bigint(n: BigInt) -> Self {
        match n.to_i128() {
            Some(v) => Rational::Small(Ratio::from_integer(v)),
            None => Rational::Big(BigRational::from_integer(n)),
        }
    }

    pub(crate) fn from_i128(n: i128) -> Self {
        Rational::Small(Ratio::from_integer(n))
    }

    pub(crate) fn to_bigrational(&self) -> BigRational {
        match self {
            Rational::Small(r) => {
                BigRational::new_raw(BigInt::from(*r.numer()), BigInt::from(*r.denom()))
            }
            Rational::Big(b) => b.clone(),
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        match self {
            Rational::Small(r) => r.is_zero(),
            Rational::Big(b) => b.is_zero(),
        }
    }

    pub(crate) fn is_negative(&self) -> bool {
        match self {
            Rational::Small(r) => r.is_negative(),
            Rational::Big(b) => b.is_negative(),
        }
    }

    pub(crate) fn recip(&self) -> Rational {
        match self {
            Rational::Small(r) if r.numer().checked_neg().is_some() => Rational::Small(r.recip()),
            Rational::Small(r) => Rational::from_bigrational(widen(r).recip()),
            Rational::Big(b) => Rational::from_bigrational(b.recip()),
        }
    }

    #[cfg(test)]
    pub(crate) fn to_f64(&self) -> f64 {
        match self {
            Rational::Small(r) => *r.numer() as f64 / *r.denom() as f64,
            Rational::Big(b) => {
                let to = |n: &BigInt| n.to_string().parse::<f64>().unwrap();
                to(b.numer()) / to(b.denom())
            }
        }
    }
}

fn widen(r: &Ratio<i128>) -> BigRational {
    BigRational::new_raw(BigInt::from(*r.numer()), BigInt::from(*r.denom()))
}

impl Neg for &Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        match self {
            Rational::Small(r) if r.numer().checked_neg().is_some() => Rational::Small(-*r),
            _ => Rational::from_bigrational(-self.to_bigrational()),
        }
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        -&self
    }
}

macro_rules! binop {
    ($trait:ident, $method:ident, $checked:ident) => {
        impl $trait for &Rational {
            type Output = Rational;
            fn $method(self, rhs: &Rational) -> Rational {
                match (self, rhs) {
                    (Rational::Small(a), Rational::Small(b)) => match a.$checked(b) {
                        Some(r) => Rational::Small(r),
                        None => Rational::from_bigrational($trait::$method(widen(a), widen(b))),
                    },
                    _ => Rational::from_bigrational($trait::$method(
                        self.to_bigrational(),
                        rhs.to_bigrational(),
                    )),
                }
            }
        }
        impl $trait for Rational {
            type Output = Rational;
            fn $method(self, rhs: Rational) -> Rational {
                $trait::$method(&self, &rhs)
            }
        }
    };
}
binop!(Add, add, checked_add);
binop!(Sub, sub, checked_sub);
binop!(Mul, mul, checked_mul);

// `Big` never holds an `i128`-representable value, so the two variants cover
// disjoint values: equality and hashing compare within a variant.
impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Rational::Small(a), Rational::Small(b)) => a == b,
            (Rational::Big(a), Rational::Big(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Rational {}

impl Hash for Rational {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Rational::Small(r) => {
                0u8.hash(state);
                r.numer().hash(state);
                r.denom().hash(state);
            }
            Rational::Big(b) => {
                1u8.hash(state);
                b.numer().hash(state);
                b.denom().hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(n: i128, d: i128) -> Rational {
        Rational::Small(Ratio::new(n, d))
    }

    #[test]
    fn arithmetic_matches_bigrational() {
        let a = small(3, 4);
        let b = small(-2, 6);
        assert_eq!(
            (&a + &b).to_bigrational(),
            a.to_bigrational() + b.to_bigrational()
        );
        assert_eq!(
            (&a - &b).to_bigrational(),
            a.to_bigrational() - b.to_bigrational()
        );
        assert_eq!(
            (&a * &b).to_bigrational(),
            a.to_bigrational() * b.to_bigrational()
        );
        assert_eq!((-&a).to_bigrational(), -a.to_bigrational());
    }

    #[test]
    fn overflow_falls_back_to_big_and_shrinks() {
        let huge = small(i128::MAX / 2, 1);
        let prod = &huge * &small(4, 1);
        assert!(matches!(prod, Rational::Big(_)));
        let back = &prod * &small(1, 4);
        assert_eq!(back, huge);
        assert!(matches!(back, Rational::Small(_)));
    }

    #[test]
    fn small_and_big_of_equal_value_never_coexist() {
        let direct = small(1, 2);
        let viabig = Rational::from_bigrational(BigRational::new(1.into(), 2.into()));
        assert_eq!(direct, viabig);
        assert!(matches!(viabig, Rational::Small(_)));
    }

    #[test]
    fn reciprocal_of_minimum_small_integer_is_exact() {
        let minimum = Rational::from_i128(i128::MIN);
        assert_eq!(
            minimum.recip().to_bigrational(),
            BigRational::new(BigInt::one(), BigInt::from(i128::MIN))
        );
    }
}
