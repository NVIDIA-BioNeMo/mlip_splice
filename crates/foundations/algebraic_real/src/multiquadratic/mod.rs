// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod field;
mod integer_factorization;
mod radicand;
mod rational;
mod square_root;

use radicand::Radicand;
use rational::Rational;

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use self::integer_factorization::{smallest_prime_factor, squarefree_decompose};
use crate::polynomial::Interval;

type CanonicalRoot = (crate::polynomial::Polynomial, usize);

const MAX_SHARED_CANONICAL_ROOTS: usize = 256;
const MAX_SHARED_CANONICAL_BYTES: usize = 64 * 1024 * 1024;

fn magnitude_bytes(bits: u64) -> usize {
    usize::try_from(bits.div_ceil(8)).unwrap_or(usize::MAX)
}

fn compact_storage_bytes(value: &Compact) -> usize {
    value.iter().fold(0usize, |bytes, (radicand, coefficient)| {
        let coefficient = coefficient.to_bigrational();
        bytes
            .saturating_add(std::mem::size_of::<(Radicand, Rational)>())
            .saturating_add(magnitude_bytes(coefficient.numer().bits()))
            .saturating_add(magnitude_bytes(coefficient.denom().bits()))
            .saturating_add(magnitude_bytes(radicand.to_biguint().bits()))
    })
}

fn canonical_storage_bytes(value: &Compact, root: &CanonicalRoot) -> usize {
    root.0
        .iter()
        .fold(compact_storage_bytes(value), |bytes, coefficient| {
            bytes
                .saturating_add(std::mem::size_of::<BigInt>())
                .saturating_add(magnitude_bytes(coefficient.bits()))
        })
}

#[derive(Default)]
struct BoundedCanonicalRoots {
    roots: HashMap<Compact, CanonicalRoot>,
    stored_bytes: usize,
}

impl BoundedCanonicalRoots {
    fn get(&self, value: &Compact) -> Option<CanonicalRoot> {
        self.roots.get(value).cloned()
    }

    fn insert(&mut self, value: Compact, root: CanonicalRoot) {
        let stored_bytes = canonical_storage_bytes(&value, &root);
        if stored_bytes > MAX_SHARED_CANONICAL_BYTES {
            return;
        }
        if self.roots.len() >= MAX_SHARED_CANONICAL_ROOTS
            || self.stored_bytes.saturating_add(stored_bytes) > MAX_SHARED_CANONICAL_BYTES
        {
            self.roots.clear();
            self.stored_bytes = 0;
        }
        if let Some(previous) = self.roots.insert(value.clone(), root) {
            self.stored_bytes -= canonical_storage_bytes(&value, &previous);
        }
        self.stored_bytes += stored_bytes;
    }
}

static SHARED_CANONICAL_ROOTS: LazyLock<Mutex<BoundedCanonicalRoots>> =
    LazyLock::new(|| Mutex::new(BoundedCanonicalRoots::default()));

#[derive(Clone, Debug)]
pub(crate) struct Multiquadratic(Arc<MultiquadraticData>);

#[derive(Debug)]
struct MultiquadraticData {
    compact: Compact,
    canonical: OnceLock<CanonicalRoot>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Compact {
    Zero,
    Term(Rational, Radicand),
    Sum(BTreeMap<Radicand, Rational>),
}

impl Compact {
    fn iter(&self) -> impl Iterator<Item = (&Radicand, &Rational)> {
        let (single, multiple) = match self {
            Self::Zero => (None, None),
            Self::Term(coefficient, radicand) => (Some((radicand, coefficient)), None),
            Self::Sum(terms) => (None, Some(terms)),
        };
        single
            .into_iter()
            .chain(multiple.into_iter().flat_map(BTreeMap::iter))
    }
}

impl PartialEq for Multiquadratic {
    fn eq(&self, other: &Self) -> bool {
        self.0.compact == other.0.compact
    }
}

impl Eq for Multiquadratic {}

impl Hash for Multiquadratic {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.compact.hash(state);
    }
}

/// Exact arithmetic for one multiquadratic number.
///
/// Constructors keep radicands squarefree, coefficients nonzero, and like terms
/// combined. Number-field and root variants never enter these operations.
impl Multiquadratic {
    fn from_compact(compact: Compact) -> Self {
        Self(Arc::new(MultiquadraticData {
            compact,
            canonical: OnceLock::new(),
        }))
    }

    pub(crate) fn canonical_with(&self, compute: impl FnOnce() -> CanonicalRoot) -> CanonicalRoot {
        if let Some(root) = self.0.canonical.get() {
            return root.clone();
        }
        let shared = SHARED_CANONICAL_ROOTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&self.0.compact);
        if let Some(root) = shared {
            let _ = self.0.canonical.set(root.clone());
            return root;
        }

        let computed = compute();
        let root = match self.0.canonical.set(computed) {
            Ok(()) => self
                .0
                .canonical
                .get()
                .expect("the canonical root was just stored")
                .clone(),
            Err(_) => self
                .0
                .canonical
                .get()
                .expect("another caller stored the canonical root")
                .clone(),
        };
        SHARED_CANONICAL_ROOTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(self.0.compact.clone(), root.clone());
        root
    }

    pub(crate) fn zero() -> Self {
        Self::from_compact(Compact::Zero)
    }

    pub(crate) fn one() -> Self {
        Self::from_compact(Compact::Term(Rational::one(), Radicand::one()))
    }

    pub(crate) fn rational(q: BigRational) -> Self {
        if q.is_zero() {
            Self::from_compact(Compact::Zero)
        } else {
            Self::from_compact(Compact::Term(
                Rational::from_bigrational(q),
                Radicand::one(),
            ))
        }
    }

    pub(crate) fn integer(n: impl Into<BigInt>) -> Self {
        Multiquadratic::rational(BigRational::from_integer(n.into()))
    }

    pub(crate) fn sqrt(n: impl Into<BigUint>) -> Self {
        let (coeff, radicand) = squarefree_decompose(n.into());
        if radicand.is_zero() {
            Self::from_compact(Compact::Zero)
        } else {
            Self::from_compact(Compact::Term(
                Rational::from_bigint(coeff.into()),
                Radicand::from_biguint(radicand),
            ))
        }
    }

    pub(crate) fn sqrt_rational(q: BigRational) -> Self {
        assert!(
            !q.is_negative(),
            "negative input to Multiquadratic::sqrt_rational"
        );
        if q.is_zero() {
            return Self::from_compact(Compact::Zero);
        }
        let a = q.numer().magnitude().clone();
        let b = q.denom().magnitude().clone();
        // √(a/b) = √(a·b) / b
        Multiquadratic::sqrt(&a * &b)
            * Multiquadratic::rational(BigRational::new(BigInt::one(), b.into()))
    }

    pub(crate) fn term(q: BigRational, m: impl Into<BigUint>) -> Self {
        Multiquadratic::rational(q) * Multiquadratic::sqrt(m)
    }

    pub(super) fn from_canonical_terms(
        terms: impl IntoIterator<Item = (BigRational, BigUint)>,
    ) -> Self {
        let mut canonical = BTreeMap::new();
        for (coefficient, radicand) in terms {
            add_term(
                &mut canonical,
                Radicand::from_biguint(radicand),
                Rational::from_bigrational(coefficient),
            );
        }
        Self::from_map(canonical)
    }

    pub(crate) fn is_zero(&self) -> bool {
        matches!(&self.0.compact, Compact::Zero)
    }

    pub(crate) fn signum(&self) -> i32 {
        match &self.0.compact {
            Compact::Zero => 0,
            Compact::Term(q, _) => {
                if q.is_negative() {
                    -1
                } else {
                    1
                }
            }
            Compact::Sum(_) => self.sign_of_sum(),
        }
    }

    fn sign_of_sum(&self) -> i32 {
        let p = self
            .iter()
            .filter(|(m, _)| !m.is_one())
            .map(|(m, _)| smallest_prime_factor(&m.to_biguint()))
            .min()
            .expect("a multi-term sum carries at least one radical");
        let (u, v) = self.split_on_prime(&p);
        let (su, sv) = (u.signum(), v.signum());
        match () {
            _ if su == 0 => sv, // u + v√p = v√p, and √p > 0
            _ if sv == 0 => su,
            _ if su == sv => su,
            // Opposite signs: sign(u + v√p) = sign(u) · sign(u² − p·v²).
            _ => {
                let p = Multiquadratic::integer(BigInt::from(p));
                su * (&(&u * &u) - &(&p * &(&v * &v))).signum()
            }
        }
    }

    /// Split `self = u + v√p`, where neither `u` nor `v` contains `√p`.
    pub(crate) fn split_on_prime(&self, p: &BigUint) -> (Self, Self) {
        let mut u = BTreeMap::new();
        let mut v = BTreeMap::new();
        for (m, q) in self.iter() {
            let mb = m.to_biguint();
            if (&mb % p).is_zero() {
                add_term(&mut v, Radicand::from_biguint(&mb / p), q.clone());
            } else {
                add_term(&mut u, m.clone(), q.clone());
            }
        }
        (Multiquadratic::from_map(u), Multiquadratic::from_map(v))
    }

    pub(crate) fn num_terms(&self) -> usize {
        match &self.0.compact {
            Compact::Zero => 0,
            Compact::Term(..) => 1,
            Compact::Sum(terms) => terms.len(),
        }
    }

    pub(crate) fn as_single_term(&self) -> Option<(BigRational, BigUint)> {
        match &self.0.compact {
            Compact::Term(q, m) => Some((q.to_bigrational(), m.to_biguint())),
            _ => None,
        }
    }

    pub(crate) fn rational_value(&self) -> Option<BigRational> {
        match &self.0.compact {
            Compact::Zero => Some(BigRational::zero()),
            Compact::Term(coefficient, radicand) if radicand.is_one() => {
                Some(coefficient.to_bigrational())
            }
            Compact::Term(..) | Compact::Sum(_) => None,
        }
    }

    /// Return the reciprocal of this value, or `None` for zero.
    ///
    /// For `x = u + v√p`, conjugation gives
    /// `x⁻¹ = (u - v√p) (u² - pv²)⁻¹`. The norm `u² - pv²` contains one fewer
    /// prime generator, so recursively applying the identity reaches a rational.
    pub(crate) fn recip(&self) -> Option<Self> {
        match &self.0.compact {
            Compact::Zero => None,
            Compact::Term(q, m) => {
                let denom = q * &m.to_rational();
                Some(Self::from_compact(Compact::Term(denom.recip(), m.clone())))
            }
            Compact::Sum(_) => {
                let p = self
                    .iter()
                    .filter(|(m, _)| !m.is_one())
                    .map(|(m, _)| smallest_prime_factor(&m.to_biguint()))
                    .min()
                    .expect("a multi-term sum carries at least one radical");
                let (u, v) = self.split_on_prime(&p);
                let p_value = Multiquadratic::integer(BigInt::from(p.clone()));
                let norm = &(&u * &u) - &(&p_value * &(&v * &v));
                let inverse_norm = norm.recip()?;
                let conjugate = u - v * Multiquadratic::sqrt(p);
                Some(conjugate * inverse_norm)
            }
        }
    }

    pub(crate) fn rational_content(&self) -> Option<BigRational> {
        let mut num_gcd = BigInt::zero();
        let mut den_lcm = BigInt::one();
        let mut any = false;
        for (_, q) in self.iter() {
            any = true;
            let q = q.to_bigrational();
            num_gcd = num_gcd.gcd(q.numer());
            den_lcm = den_lcm.lcm(q.denom());
        }
        any.then(|| BigRational::new(num_gcd, den_lcm))
    }

    /// The canonical terms as `(coefficient, radicand)` pairs with squarefree
    /// radicands (a radicand of 1 is the rational part); empty for zero.
    pub(crate) fn terms(&self) -> Vec<(BigRational, BigUint)> {
        self.iter()
            .map(|(m, q)| (q.to_bigrational(), m.to_biguint()))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn to_f64(&self) -> f64 {
        self.iter()
            .map(|(m, q)| q.to_f64() * m.to_f64().sqrt())
            .sum()
    }

    fn iter(&self) -> impl Iterator<Item = (&Radicand, &Rational)> {
        self.0.compact.iter()
    }

    fn from_map(mut terms: BTreeMap<Radicand, Rational>) -> Self {
        Self::from_compact(match terms.len() {
            0 => Compact::Zero,
            1 => {
                let (m, q) = terms.pop_first().unwrap();
                Compact::Term(q, m)
            }
            _ => Compact::Sum(terms),
        })
    }
}

impl fmt::Display for Multiquadratic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        for (i, (m, q)) in self.iter().enumerate() {
            let q = q.to_bigrational();
            let negative = q.is_negative();
            if i == 0 {
                if negative {
                    write!(f, "-")?;
                }
            } else {
                write!(f, " {} ", if negative { "-" } else { "+" })?;
            }
            let mag = q.abs();
            write_magnitude(f, mag.numer(), q.denom(), m)?;
        }
        Ok(())
    }
}

fn write_magnitude(
    f: &mut fmt::Formatter<'_>,
    num: &BigInt,
    den: &BigInt,
    m: &Radicand,
) -> fmt::Result {
    if m.is_one() {
        if den.is_one() {
            write!(f, "{num}")
        } else {
            write!(f, "{num}/{den}")
        }
    } else {
        if num.is_one() {
            write!(f, "√{m}")?;
        } else {
            write!(f, "{num}·√{m}")?;
        }
        if !den.is_one() {
            write!(f, "/{den}")?;
        }
        Ok(())
    }
}

fn add_term(terms: &mut BTreeMap<Radicand, Rational>, radicand: Radicand, coeff: Rational) {
    if coeff.is_zero() {
        return;
    }
    match terms.entry(radicand) {
        Entry::Vacant(e) => {
            e.insert(coeff);
        }
        Entry::Occupied(mut e) => {
            let sum = e.get() + &coeff;
            if sum.is_zero() {
                e.remove();
            } else {
                *e.get_mut() = sum;
            }
        }
    }
}

fn mul_terms(qa: &Rational, a: &Radicand, qb: &Rational, b: &Radicand) -> (Rational, Radicand) {
    let g = a.gcd(b);
    let radicand = a.div_exact(&g).mul(&b.div_exact(&g));
    let coeff = &(qa * qb) * &g.to_rational();
    (coeff, radicand)
}

impl Neg for &Multiquadratic {
    type Output = Multiquadratic;
    fn neg(self) -> Multiquadratic {
        Multiquadratic::from_compact(match &self.0.compact {
            Compact::Zero => Compact::Zero,
            Compact::Term(q, m) => Compact::Term(-q, m.clone()),
            Compact::Sum(terms) => {
                Compact::Sum(terms.iter().map(|(m, q)| (m.clone(), -q)).collect())
            }
        })
    }
}

impl Neg for Multiquadratic {
    type Output = Multiquadratic;
    fn neg(self) -> Multiquadratic {
        -&self
    }
}

impl Add for &Multiquadratic {
    type Output = Multiquadratic;
    fn add(self, rhs: &Multiquadratic) -> Multiquadratic {
        match (&self.0.compact, &rhs.0.compact) {
            (Compact::Zero, _) => rhs.clone(),
            (_, Compact::Zero) => self.clone(),
            (Compact::Term(qa, a), Compact::Term(qb, b)) if a == b => {
                Multiquadratic::rational_at(qa + qb, a.clone())
            }
            _ => {
                let mut terms: BTreeMap<Radicand, Rational> =
                    self.iter().map(|(m, q)| (m.clone(), q.clone())).collect();
                for (m, q) in rhs.iter() {
                    add_term(&mut terms, m.clone(), q.clone());
                }
                Multiquadratic::from_map(terms)
            }
        }
    }
}

impl Sub for &Multiquadratic {
    type Output = Multiquadratic;
    fn sub(self, rhs: &Multiquadratic) -> Multiquadratic {
        match (&self.0.compact, &rhs.0.compact) {
            (_, Compact::Zero) => self.clone(),
            (Compact::Zero, _) => -rhs,
            (Compact::Term(qa, a), Compact::Term(qb, b)) if a == b => {
                Multiquadratic::rational_at(qa - qb, a.clone())
            }
            _ => {
                let mut terms: BTreeMap<Radicand, Rational> =
                    self.iter().map(|(m, q)| (m.clone(), q.clone())).collect();
                for (m, q) in rhs.iter() {
                    add_term(&mut terms, m.clone(), -q);
                }
                Multiquadratic::from_map(terms)
            }
        }
    }
}

impl Mul for &Multiquadratic {
    type Output = Multiquadratic;
    fn mul(self, rhs: &Multiquadratic) -> Multiquadratic {
        match (&self.0.compact, &rhs.0.compact) {
            (Compact::Zero, _) | (_, Compact::Zero) => Multiquadratic::from_compact(Compact::Zero),
            (Compact::Term(qa, a), Compact::Term(qb, b)) => {
                let (coeff, radicand) = mul_terms(qa, a, qb, b);
                Multiquadratic::from_compact(Compact::Term(coeff, radicand))
            }
            _ => {
                let mut terms = BTreeMap::new();
                for (a, qa) in self.iter() {
                    for (b, qb) in rhs.iter() {
                        let (coeff, radicand) = mul_terms(qa, a, qb, b);
                        add_term(&mut terms, radicand, coeff);
                    }
                }
                Multiquadratic::from_map(terms)
            }
        }
    }
}

// Division in a field is multiplication by the reciprocal.
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for &Multiquadratic {
    type Output = Multiquadratic;
    fn div(self, rhs: &Multiquadratic) -> Multiquadratic {
        let inverse = rhs.recip().expect("Multiquadratic division by zero");
        self * &inverse
    }
}

impl Multiquadratic {
    fn rational_at(q: Rational, m: Radicand) -> Self {
        if q.is_zero() {
            Self::from_compact(Compact::Zero)
        } else {
            Self::from_compact(Compact::Term(q, m))
        }
    }
}

macro_rules! forward_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<Multiquadratic> for Multiquadratic {
            type Output = Multiquadratic;
            fn $method(self, rhs: Multiquadratic) -> Multiquadratic {
                $trait::$method(&self, &rhs)
            }
        }
        impl $trait<&Multiquadratic> for Multiquadratic {
            type Output = Multiquadratic;
            fn $method(self, rhs: &Multiquadratic) -> Multiquadratic {
                $trait::$method(&self, rhs)
            }
        }
        impl $trait<Multiquadratic> for &Multiquadratic {
            type Output = Multiquadratic;
            fn $method(self, rhs: Multiquadratic) -> Multiquadratic {
                $trait::$method(self, &rhs)
            }
        }
    };
}
forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);
forward_binop!(Div, div);

pub(crate) fn interval(value: &Multiquadratic, bits: usize) -> Interval {
    let denominator = BigInt::one() << bits;
    value.terms().into_iter().fold(
        Interval::point(BigRational::zero()),
        |sum, (coefficient, radicand)| {
            let scaled = radicand << (bits * 2);
            let lower_integer = scaled.sqrt();
            let exact = &lower_integer * &lower_integer == scaled;
            let low = BigRational::new(BigInt::from(lower_integer.clone()), denominator.clone());
            let high = if exact {
                low.clone()
            } else {
                BigRational::new(BigInt::from(lower_integer + 1u32), denominator.clone())
            };
            let term = if coefficient.is_negative() {
                Interval::new(&coefficient * high, &coefficient * low)
            } else {
                Interval::new(&coefficient * low, &coefficient * high)
            };
            sum.add(&term)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn r(n: i64) -> Multiquadratic {
        Multiquadratic::rational(BigRational::from_integer(BigInt::from(n)))
    }

    #[test]
    fn canonical_builder_combines_duplicates_and_drops_zeroes() {
        let value = Multiquadratic::from_canonical_terms([
            (q(1, 1), 2u32.into()),
            (q(2, 1), 2u32.into()),
            (q(-3, 1), 2u32.into()),
            (q(4, 1), 3u32.into()),
            (q(0, 1), 5u32.into()),
        ]);
        assert_eq!(value, Multiquadratic::term(q(4, 1), 3u32));
    }

    #[test]
    fn separately_built_equal_values_share_canonical_work() {
        let build = || {
            Multiquadratic::sqrt(101u32)
                + Multiquadratic::sqrt(103u32)
                + Multiquadratic::sqrt(107u32)
                + Multiquadratic::sqrt(109u32)
        };
        let expected =
            crate::algebraic_real::AlgebraicReal::multiquadratic(build()).canonical_root();
        assert_eq!(
            expected,
            build().canonical_with(|| unreachable!("an equal value populated the shared cache"))
        );
    }

    #[test]
    fn clones_share_immutable_terms_and_canonical_cache() {
        let value = Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32);
        let clone = value.clone();
        assert!(Arc::ptr_eq(&value.0, &clone.0));
        let expected =
            crate::algebraic_real::AlgebraicReal::multiquadratic(value.clone()).canonical_root();
        assert_eq!(
            expected,
            clone.canonical_with(|| unreachable!("clone shares the populated cache"))
        );
    }

    #[test]
    fn sqrt_squares_to_rational() {
        let s2 = Multiquadratic::sqrt(2u32);
        assert_eq!(&s2 * &s2, r(2));
        assert!((&s2 * &s2).as_single_term().is_some());
    }

    #[test]
    fn sqrt_product_combines() {
        assert_eq!(
            Multiquadratic::sqrt(2u32) * Multiquadratic::sqrt(3u32),
            Multiquadratic::sqrt(6u32)
        );
        assert_eq!(
            Multiquadratic::sqrt(6u32) * Multiquadratic::sqrt(15u32),
            Multiquadratic::term(q(3, 1), 10u32)
        );
    }

    #[test]
    fn radicands_are_reduced() {
        assert_eq!(
            Multiquadratic::sqrt(8u32),
            Multiquadratic::term(q(2, 1), 2u32)
        );
        assert_eq!(
            Multiquadratic::sqrt(12u32),
            Multiquadratic::term(q(2, 1), 3u32)
        );
        assert_eq!(Multiquadratic::sqrt(1u32), r(1));
        assert_eq!(
            Multiquadratic::sqrt(50u32),
            Multiquadratic::term(q(5, 1), 2u32)
        );
        assert!((Multiquadratic::term(q(2, 1), 2u32) - Multiquadratic::sqrt(8u32)).is_zero());
    }

    #[test]
    fn binomial_square() {
        let x = Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32);
        let expected = r(5) + Multiquadratic::term(q(2, 1), 6u32);
        assert_eq!(&x * &x, expected);
    }

    #[test]
    fn sqrt_of_rational() {
        let h = Multiquadratic::sqrt_rational(q(1, 2));
        assert_eq!(h, Multiquadratic::term(q(1, 2), 2u32));
        assert_eq!(&h * &h, Multiquadratic::rational(q(1, 2)));
        let t = Multiquadratic::sqrt_rational(q(3, 5));
        assert_eq!(&t * &t, Multiquadratic::rational(q(3, 5)));
    }

    #[test]
    #[should_panic(expected = "negative input to Multiquadratic::sqrt_rational")]
    fn sqrt_rational_rejects_negative_input_clearly() {
        let _ = Multiquadratic::sqrt_rational(q(-1, 2));
    }

    #[test]
    fn linear_independence_zero_test() {
        let v = Multiquadratic::term(q(3, 1), 2u32) - Multiquadratic::term(q(2, 1), 3u32);
        assert!(!v.is_zero());
        assert!((&v - &v).is_zero());
        assert!(!(r(1) + Multiquadratic::sqrt(2u32)).is_zero());
    }

    #[test]
    fn additive_and_multiplicative_identities() {
        let x = Multiquadratic::sqrt(7u32) + r(3);
        assert_eq!(&x + &Multiquadratic::zero(), x);
        assert_eq!(&x * &Multiquadratic::one(), x);
        assert!((&x * &Multiquadratic::zero()).is_zero());
    }

    #[test]
    fn general_division() {
        let div = |a: &Multiquadratic, b: &Multiquadratic| a * &b.recip().unwrap();
        assert_eq!(
            div(&Multiquadratic::sqrt(6u32), &Multiquadratic::sqrt(2u32)),
            Multiquadratic::sqrt(3u32)
        );
        let inv = Multiquadratic::sqrt(2u32).recip().unwrap();
        assert_eq!(inv, Multiquadratic::term(q(1, 2), 2u32));
        assert_eq!(Multiquadratic::sqrt(2u32) * &inv, r(1));
        let a = Multiquadratic::term(q(3, 4), 6u32);
        let b = Multiquadratic::term(q(1, 2), 2u32);
        assert_eq!(div(&a, &b), Multiquadratic::term(q(3, 2), 3u32));
        let binomial = Multiquadratic::sqrt(2u32) + r(1);
        assert_eq!(binomial.recip().unwrap(), Multiquadratic::sqrt(2u32) - r(1));

        let two_roots = Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32);
        assert_eq!(
            two_roots.recip().unwrap(),
            Multiquadratic::sqrt(3u32) - Multiquadratic::sqrt(2u32)
        );

        let three_terms = r(1) + Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32);
        assert_eq!(
            &three_terms * &three_terms.recip().unwrap(),
            Multiquadratic::one()
        );
        assert_eq!(&three_terms / &three_terms, Multiquadratic::one());
        assert!(Multiquadratic::zero().recip().is_none());
    }

    #[test]
    #[should_panic(expected = "Multiquadratic division by zero")]
    fn division_by_zero_panics() {
        let _ = Multiquadratic::one() / Multiquadratic::zero();
    }

    #[test]
    fn float_value() {
        let x = Multiquadratic::term(q(1, 2), 2u32);
        assert!((x.to_f64() - 0.5f64.sqrt()).abs() < 1e-12);
        let y = r(5) + Multiquadratic::term(q(2, 1), 6u32);
        assert!((y.to_f64() - (5.0 + 2.0 * 6.0f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn display_reads_as_math() {
        assert_eq!(Multiquadratic::zero().to_string(), "0");
        assert_eq!(r(3).to_string(), "3");
        assert_eq!(Multiquadratic::rational(q(1, 2)).to_string(), "1/2");
        assert_eq!(Multiquadratic::sqrt(5u32).to_string(), "√5");
        assert_eq!((-Multiquadratic::sqrt(2u32)).to_string(), "-√2");
        assert_eq!(Multiquadratic::term(q(1, 2), 3u32).to_string(), "√3/2");
        assert_eq!(Multiquadratic::term(q(3, 4), 6u32).to_string(), "3·√6/4");
        assert_eq!(Multiquadratic::term(q(2, 1), 6u32).to_string(), "2·√6");
        let s = Multiquadratic::term(q(1, 2), 3u32) + Multiquadratic::sqrt(5u32);
        assert_eq!(s.to_string(), "√3/2 + √5");
        let t = r(5) - Multiquadratic::term(q(2, 1), 6u32);
        assert_eq!(t.to_string(), "5 - 2·√6");
    }

    #[test]
    fn signum_matches_the_real_value() {
        let cases = [
            r(0),
            r(3),
            -r(3),
            Multiquadratic::sqrt(2u32),
            -Multiquadratic::sqrt(2u32),
            // 3 − 2√2 ≈ 0.17 > 0; 2√2 − 3 < 0 (close calls a float could miss)
            r(3) - Multiquadratic::term(q(2, 1), 2u32),
            Multiquadratic::term(q(2, 1), 2u32) - r(3),
            // √2 + √3 − √5 ≈ 0.91 > 0; flip one term → < 0
            Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32) - Multiquadratic::sqrt(5u32),
            Multiquadratic::sqrt(2u32) - Multiquadratic::sqrt(3u32) - Multiquadratic::sqrt(5u32),
            // 7√6 − 17 ≈ 0.14 > 0  (17² = 289, (7√6)² = 294)
            Multiquadratic::term(q(7, 1), 6u32) - r(17),
        ];
        for x in &cases {
            let want = x.to_f64().partial_cmp(&0.0).unwrap() as i32;
            assert_eq!(x.signum(), want, "signum of {x}");
            assert_eq!((-x).signum(), -x.signum(), "antisymmetry on {x}");
        }
    }

    #[test]
    fn distributive() {
        let a = Multiquadratic::sqrt(2u32) + r(1);
        let b = Multiquadratic::sqrt(3u32);
        let c = Multiquadratic::term(q(1, 3), 6u32) - r(2);
        assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
    }

    #[test]
    fn canonical_independent_of_construction() {
        let collapsed =
            (Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32)) - Multiquadratic::sqrt(3u32);
        let direct = Multiquadratic::sqrt(2u32);
        assert_eq!(collapsed, direct);
        assert_eq!(collapsed.num_terms(), 1);

        let to_zero = Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32)
            - Multiquadratic::sqrt(2u32)
            - Multiquadratic::sqrt(3u32);
        assert_eq!(to_zero, Multiquadratic::zero());
        assert!(to_zero.is_zero());

        let prod = (Multiquadratic::sqrt(2u32) + r(1)) * (Multiquadratic::sqrt(2u32) - r(1)); // = 1
        assert_eq!(prod, r(1));
        assert_eq!(prod.as_single_term().unwrap().1, BigUint::one());
    }

    #[test]
    fn equality_iff_difference_is_zero() {
        let samples = [
            r(0),
            r(3),
            Multiquadratic::sqrt(2u32),
            Multiquadratic::term(q(3, 4), 6u32),
            r(5) + Multiquadratic::term(q(2, 1), 6u32),
            Multiquadratic::sqrt(2u32) + Multiquadratic::sqrt(3u32) + Multiquadratic::sqrt(5u32),
        ];
        for a in &samples {
            for b in &samples {
                assert_eq!(a == b, (a - b).is_zero(), "{a:?} vs {b:?}");
            }
        }
    }
}
