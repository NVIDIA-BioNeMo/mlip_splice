// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::ToPrimitive;

#[derive(Clone, Debug)]
pub(crate) enum Radicand {
    Small(u64),
    Big(BigUint),
}

impl Radicand {
    pub(crate) fn one() -> Self {
        Radicand::Small(1)
    }

    pub(crate) fn from_biguint(m: BigUint) -> Self {
        match m.to_u64() {
            Some(v) => Radicand::Small(v),
            None => Radicand::Big(m),
        }
    }

    pub(crate) fn is_one(&self) -> bool {
        matches!(self, Radicand::Small(1))
    }

    pub(crate) fn to_biguint(&self) -> BigUint {
        match self {
            Radicand::Small(v) => BigUint::from(*v),
            Radicand::Big(b) => b.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn to_f64(&self) -> f64 {
        match self {
            Radicand::Small(v) => *v as f64,
            Radicand::Big(b) => b.to_string().parse::<f64>().unwrap(),
        }
    }

    pub(crate) fn to_rational(&self) -> super::Rational {
        match self {
            Radicand::Small(v) => super::Rational::from_i128(*v as i128),
            Radicand::Big(b) => super::Rational::from_bigint(BigInt::from(b.clone())),
        }
    }

    pub(crate) fn gcd(&self, other: &Radicand) -> Radicand {
        match (self, other) {
            (Radicand::Small(a), Radicand::Small(b)) => Radicand::Small(a.gcd(b)),
            _ => Radicand::from_biguint(self.to_biguint().gcd(&other.to_biguint())),
        }
    }

    pub(crate) fn div_exact(&self, rhs: &Radicand) -> Radicand {
        match (self, rhs) {
            (Radicand::Small(a), Radicand::Small(b)) => Radicand::Small(a / b),
            _ => Radicand::from_biguint(self.to_biguint() / rhs.to_biguint()),
        }
    }

    pub(crate) fn mul(&self, rhs: &Radicand) -> Radicand {
        match (self, rhs) {
            (Radicand::Small(a), Radicand::Small(b)) => match a.checked_mul(*b) {
                Some(v) => Radicand::Small(v),
                None => Radicand::Big(BigUint::from(*a) * BigUint::from(*b)),
            },
            _ => Radicand::from_biguint(self.to_biguint() * rhs.to_biguint()),
        }
    }
}

impl fmt::Display for Radicand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Radicand::Small(v) => write!(f, "{v}"),
            Radicand::Big(b) => write!(f, "{b}"),
        }
    }
}

// `Big` only holds values above `u64::MAX`, so a `Small` is always less than a
// `Big` and the variants cover disjoint values.
impl Ord for Radicand {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Radicand::Small(a), Radicand::Small(b)) => a.cmp(b),
            (Radicand::Big(a), Radicand::Big(b)) => a.cmp(b),
            (Radicand::Small(_), Radicand::Big(_)) => Ordering::Less,
            (Radicand::Big(_), Radicand::Small(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for Radicand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Radicand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Radicand {}

impl std::hash::Hash for Radicand {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Radicand::Small(v) => {
                0u8.hash(state);
                v.hash(state);
            }
            Radicand::Big(b) => {
                1u8.hash(state);
                b.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_div_mul_match_biguint() {
        let a = Radicand::Small(30);
        let b = Radicand::Small(42);
        let g = a.gcd(&b);
        assert_eq!(g, Radicand::Small(6));
        assert_eq!(a.div_exact(&g), Radicand::Small(5));
        assert_eq!(a.mul(&b), Radicand::Small(1260));
    }

    #[test]
    fn overflow_promotes_then_shrinks() {
        let big = Radicand::Small(u64::MAX);
        let prod = big.mul(&Radicand::Small(3));
        assert!(matches!(prod, Radicand::Big(_)));
        assert_eq!(prod.div_exact(&Radicand::Small(3)), big);
    }

    #[test]
    fn small_orders_below_big_and_equals_canonically() {
        let small = Radicand::Small(7);
        let big = Radicand::from_biguint(BigUint::from(u128::from(u64::MAX) + 2));
        assert!(small < big);
        assert_eq!(
            Radicand::from_biguint(BigUint::from(7u32)),
            Radicand::Small(7)
        );
    }
}
