// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;
use std::sync::{Arc, Mutex};

use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::CgCache;
use crate::exact::ComplexAlgebraicReal;
use crate::rep::{Irrep, Mat, Rep};
use algebraic_real::AlgebraicReal;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Su2Irrep {
    pub twice_j: u32,
}

impl Su2Irrep {
    pub fn new(twice_j: u32) -> Self {
        Su2Irrep { twice_j }
    }
}

impl Rep for Su2Irrep {
    fn dim(&self) -> usize {
        self.twice_j as usize + 1
    }

    fn generators(&self) -> Vec<Mat<ComplexAlgebraicReal>> {
        su2_generators(self.twice_j)
    }
}

impl Irrep for Su2Irrep {
    fn trivial(&self) -> Self {
        Su2Irrep::new(0)
    }

    fn tensor_product(&self, other: &Self) -> Vec<Self> {
        let lo = self.twice_j.abs_diff(other.twice_j);
        let hi = self.twice_j + other.twice_j;
        (lo..=hi).step_by(2).map(Su2Irrep::new).collect()
    }

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>> {
        let lo = a.twice_j.abs_diff(b.twice_j);
        let hi = a.twice_j + b.twice_j;
        if c.twice_j < lo || c.twice_j > hi || !(c.twice_j - lo).is_multiple_of(2) {
            return Vec::new();
        }
        vec![
            su2_clebsch_gordan(a.twice_j, b.twice_j, c.twice_j)
                .as_ref()
                .clone(),
        ]
    }

    fn cartan_product(a: &Self, b: &Self) -> Self {
        Su2Irrep::new(a.twice_j + b.twice_j)
    }

    fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        match s.split_once('/') {
            Some((num, "2")) => {
                let twice_j = num.parse::<u32>().ok()?;
                // A reduced half-integer has an odd numerator over 2.
                (twice_j % 2 == 1).then(|| Su2Irrep::new(twice_j))
            }
            Some(_) => None,
            None => {
                let j = s.parse::<u32>().ok()?;
                Some(Su2Irrep::new(2 * j))
            }
        }
    }
}

impl fmt::Display for Su2Irrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.twice_j.is_multiple_of(2) {
            write!(f, "{}", self.twice_j / 2)
        } else {
            write!(f, "{}/2", self.twice_j)
        }
    }
}

impl Ord for Su2Irrep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.twice_j.cmp(&other.twice_j)
    }
}

impl PartialOrd for Su2Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn m_values(twice_j: u32) -> Vec<BigRational> {
    let t = twice_j as i64;
    let two = BigInt::from(2);
    (-t..=t)
        .step_by(2)
        .map(|twice_m| BigRational::new(BigInt::from(twice_m), two.clone()))
        .collect()
}

fn factorial_table(max: u64) -> Vec<BigUint> {
    let mut table = vec![BigUint::one()];
    for k in 1..=max {
        let next = table.last().unwrap() * BigUint::from(k);
        table.push(next);
    }
    table
}

fn factorial_of(value: &BigRational, table: &[BigUint]) -> BigUint {
    assert!(value.is_integer(), "factorial of a non-integer");
    let n = value.to_integer();
    assert!(n >= BigInt::zero(), "factorial of a negative integer");
    table[usize::try_from(n).expect("factorial argument too large")].clone()
}

fn ratio(num: BigUint, den: BigUint) -> BigRational {
    BigRational::new(num.into(), den.into())
}

fn su2_coefficient(
    j1: &BigRational,
    m1: &BigRational,
    j2: &BigRational,
    m2: &BigRational,
    j3: &BigRational,
    m3: &BigRational,
    fact: &[BigUint],
) -> AlgebraicReal {
    if m3 != &(m1 + m2) {
        return AlgebraicReal::zero();
    }

    let two = BigRational::from_integer(BigInt::from(2));
    let one = BigRational::one();

    let vmin = [j2 - j1 + m3, m1 - j1, BigRational::zero()]
        .into_iter()
        .max()
        .unwrap();
    let vmax = [j2 + j3 + m1, j2 + j3 - j1, j3 + m3]
        .into_iter()
        .min()
        .unwrap();

    // C = (2j₃+1) · (j₃+j₁−j₂)!(j₃−j₁+j₂)!(j₁+j₂−j₃)!(j₃+m₃)!(j₃−m₃)!
    //               / [ (j₁+j₂+j₃+1)!(j₁−m₁)!(j₁+m₁)!(j₂−m₂)!(j₂+m₂)! ]
    let c_num = factorial_of(&(j3 + j1 - j2), fact)
        * factorial_of(&(j3 - j1 + j2), fact)
        * factorial_of(&(j1 + j2 - j3), fact)
        * factorial_of(&(j3 + m3), fact)
        * factorial_of(&(j3 - m3), fact);
    let c_den = factorial_of(&(j1 + j2 + j3 + &one), fact)
        * factorial_of(&(j1 - m1), fact)
        * factorial_of(&(j1 + m1), fact)
        * factorial_of(&(j2 - m2), fact)
        * factorial_of(&(j2 + m2), fact);
    let c = (&two * j3 + &one) * ratio(c_num, c_den);

    // S = Σ_v (−1)^(v + j₂ + m₂) · (j₂+j₃+m₁−v)!(j₁−m₁+v)!
    //          / [ v!(j₃−j₁+j₂−v)!(j₃+m₃−v)!(v+j₁−j₂−m₃)! ]
    let mut v = vmin.clone();
    let mut s = BigRational::zero();
    while v <= vmax {
        let term_num =
            factorial_of(&(j2 + j3 + m1 - &v), fact) * factorial_of(&(j1 - m1 + &v), fact);
        let term_den = factorial_of(&v, fact)
            * factorial_of(&(j3 - j1 + j2 - &v), fact)
            * factorial_of(&(j3 + m3 - &v), fact)
            * factorial_of(&(&v + j1 - j2 - m3), fact);
        let magnitude = ratio(term_num, term_den);
        let exponent = &v + j2 + m2;
        debug_assert!(exponent.is_integer());
        let sign = if exponent.to_integer().bit(0) {
            -BigRational::one()
        } else {
            BigRational::one()
        };
        s += sign * magnitude;
        v += &one;
    }

    // S·√C: pull |S| inside the root, keep S's sign — a single √-rational term.
    AlgebraicReal::rational(s)
        * AlgebraicReal::sqrt_rational(c).expect("Clebsch-Gordan radicand is nonnegative")
}

/// Exact SU(2) Clebsch-Gordan, memoized on the spin triple. The coefficients are
/// a deterministic function of `(t1, t2, t3)` (twice the spins), so a process-wide
/// memo returns the identical matrix every caller would have recomputed.
pub(crate) fn su2_clebsch_gordan(t1: u32, t2: u32, t3: u32) -> Arc<Mat<AlgebraicReal>> {
    static CACHE: CgCache = CgCache::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Default::default()));
    if let Some(hit) = cache.lock().unwrap().get(&(t1, t2, t3)) {
        return hit.clone();
    }
    let mat = Arc::new(compute_su2_clebsch_gordan(t1, t2, t3));
    cache
        .lock()
        .unwrap()
        .entry((t1, t2, t3))
        .or_insert(mat)
        .clone()
}

fn compute_su2_clebsch_gordan(t1: u32, t2: u32, t3: u32) -> Mat<AlgebraicReal> {
    let half = BigRational::new(BigInt::one(), BigInt::from(2));
    let (j1, j2, j3) = (
        &half * BigInt::from(t1),
        &half * BigInt::from(t2),
        &half * BigInt::from(t3),
    );
    let (m1s, m2s, m3s) = (m_values(t1), m_values(t2), m_values(t3));
    let (d2, d3) = (m2s.len(), m3s.len());
    // Every factorial argument is bounded by j₁+j₂+j₃+1.
    let fact = factorial_table((t1 + t2 + t3) as u64 / 2 + 1);

    let rows = m1s.len() * d2;
    let mut mat = Mat {
        rows,
        cols: d3,
        entries: vec![AlgebraicReal::zero(); rows * d3],
    };
    for (i1, m1) in m1s.iter().enumerate() {
        for (i2, m2) in m2s.iter().enumerate() {
            for (i3, m3) in m3s.iter().enumerate() {
                let coeff = su2_coefficient(&j1, m1, &j2, m2, &j3, m3, &fact);
                if !coeff.is_zero() {
                    mat.set(i1 * d2 + i2, i3, coeff);
                }
            }
        }
    }
    mat
}

pub(crate) fn su2_generators(twice_j: u32) -> Vec<Mat<ComplexAlgebraicReal>> {
    let d = twice_j as usize + 1;
    let half = BigRational::new(BigInt::one(), BigInt::from(2));
    let j = &half * BigInt::from(twice_j);
    let jj1 = &j * (&j + BigRational::one()); // j(j+1)

    let surd = |m: &BigRational, shift: i64| -> AlgebraicReal {
        // √(j(j+1) − m(m+shift)); the radicand is a non-negative rational.
        let mp = m + BigRational::from_integer(BigInt::from(shift));
        let radicand = &jj1 - m * mp;
        assert!(!radicand.is_negative());
        AlgebraicReal::sqrt_rational(radicand).expect("Clebsch-Gordan radicand is nonnegative")
    };

    let ms = m_values(twice_j);

    // raising[a+1, a] = −√(j(j+1) − m_a(m_a+1));  lowering[a, a+1] = √(j(j+1) − m_b(m_b−1)).
    let mut raising = vec![vec![AlgebraicReal::zero(); d]; d];
    let mut lowering = vec![vec![AlgebraicReal::zero(); d]; d];
    for a in 0..d - 1 {
        raising[a + 1][a] = -&surd(&ms[a], 1);
        lowering[a][a + 1] = surd(&ms[a + 1], -1);
    }

    let half = AlgebraicReal::rational(BigRational::new(BigInt::one(), BigInt::from(2)));

    let mut lx = Mat::zeros(d, d);
    let mut lz = Mat::zeros(d, d);
    let mut neg_ly = Mat::zeros(d, d);
    for r in 0..d {
        for c in 0..d {
            let raise = &raising[r][c];
            let lower = &lowering[r][c];
            // x = ½(raising + lowering)
            lx.set(r, c, ComplexAlgebraicReal::real(&half * &(raise + lower)));
            // −y = −½i(raising − lowering)
            neg_ly.set(
                r,
                c,
                ComplexAlgebraicReal::imag(&(-&half) * &(raise - lower)),
            );
        }
        // z = diag(i·m)
        lz.set(
            r,
            r,
            ComplexAlgebraicReal::imag(AlgebraicReal::rational(ms[r].clone())),
        );
    }

    vec![lx, lz, neg_ly]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irrep(twice_j: u32) -> Su2Irrep {
        Su2Irrep::new(twice_j)
    }

    fn commutator(
        a: &Mat<ComplexAlgebraicReal>,
        b: &Mat<ComplexAlgebraicReal>,
    ) -> Mat<ComplexAlgebraicReal> {
        let ab = a.mul(b);
        let ba = b.mul(a);
        let mut out = Mat::zeros(ab.rows, ab.cols);
        for i in 0..ab.rows {
            for j in 0..ab.cols {
                out.set(i, j, ab.get(i, j) - ba.get(i, j));
            }
        }
        out
    }

    #[test]
    fn su2_commutators_exact() {
        for t in [0u32, 1, 2, 3, 4, 5] {
            let g = irrep(t).generators();
            let (lx, lz, neg_ly) = (&g[0], &g[1], &g[2]);
            let mut ly = Mat::zeros(neg_ly.rows, neg_ly.cols);
            for i in 0..ly.rows {
                for j in 0..ly.cols {
                    ly.set(i, j, -neg_ly.get(i, j));
                }
            }
            assert_eq!(commutator(lx, &ly), *lz, "[Lx,Ly]=Lz at 2j={t}");
            assert_eq!(commutator(&ly, lz), *lx, "[Ly,Lz]=Lx at 2j={t}");
            assert_eq!(commutator(lz, lx), ly, "[Lz,Lx]=Ly at 2j={t}");
        }
    }

    fn assert_intertwines(a: &Su2Irrep, b: &Su2Irrep, c: &Su2Irrep) {
        let paths = Su2Irrep::clebsch_gordan(a, b, c);
        let (da, db, dc) = (a.dim(), b.dim(), c.dim());
        let ga = a.generators();
        let gb = b.generators();
        let gc = c.generators();

        for path in &paths {
            assert_eq!(path.rows, da * db);
            assert_eq!(path.cols, dc);
            let entry = |j: usize, l: usize, m: usize| {
                ComplexAlgebraicReal::real(path.get(j * db + l, m).clone())
            };

            for component in 0..3 {
                let (la, lb, lc) = (&ga[component], &gb[component], &gc[component]);
                for j in 0..da {
                    for l in 0..db {
                        for m in 0..dc {
                            let mut left = ComplexAlgebraicReal::zero();
                            for jp in 0..da {
                                left = &left + &(la.get(j, jp) * &entry(jp, l, m));
                            }
                            for lp in 0..db {
                                left = &left + &(lb.get(l, lp) * &entry(j, lp, m));
                            }
                            let mut right = ComplexAlgebraicReal::zero();
                            for mp in 0..dc {
                                right = &right + &(lc.get(mp, m) * &entry(j, l, mp));
                            }
                            assert_eq!(
                                left, right,
                                "intertwiner {a} {b} {c} comp {component} at ({j},{l},{m})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn clebsch_gordan_intertwines() {
        let cases = [
            (irrep(1), irrep(1), irrep(0)),
            (irrep(1), irrep(1), irrep(2)),
            (irrep(1), irrep(2), irrep(1)),
            (irrep(1), irrep(2), irrep(3)),
            (irrep(2), irrep(2), irrep(0)),
            (irrep(2), irrep(2), irrep(2)),
            (irrep(2), irrep(2), irrep(4)),
            (irrep(3), irrep(2), irrep(1)),
            (irrep(3), irrep(3), irrep(0)),
            (irrep(0), irrep(3), irrep(3)),
        ];
        for (a, b, c) in cases {
            assert_intertwines(&a, &b, &c);
        }
    }

    #[test]
    fn clebsch_gordan_path_orthonormal() {
        let cases = [
            (irrep(1), irrep(1), irrep(0)),
            (irrep(1), irrep(1), irrep(2)),
            (irrep(1), irrep(2), irrep(3)),
            (irrep(2), irrep(2), irrep(2)),
            (irrep(3), irrep(2), irrep(3)),
        ];
        for (a, b, c) in cases {
            let paths = Su2Irrep::clebsch_gordan(&a, &b, &c);
            let [path] = <[_; 1]>::try_from(paths).ok().unwrap();
            let dc = c.dim();
            for m in 0..dc {
                for mp in 0..dc {
                    let mut acc = AlgebraicReal::zero();
                    for row in 0..path.rows {
                        acc = &acc + &(path.get(row, m) * path.get(row, mp));
                    }
                    let expected = if m == mp {
                        AlgebraicReal::one()
                    } else {
                        AlgebraicReal::zero()
                    };
                    assert_eq!(acc, expected, "orthonormal {a} {b} {c} ({m},{mp})");
                }
            }
        }
    }

    #[test]
    fn tensor_product_selection_rule() {
        assert_eq!(irrep(1).tensor_product(&irrep(1)), vec![irrep(0), irrep(2)]);
        assert_eq!(irrep(2).tensor_product(&irrep(1)), vec![irrep(1), irrep(3)]);
        // ½ ⊗ ½ contains no ½: the parity of 2j forbids it.
        assert!(Su2Irrep::clebsch_gordan(&irrep(1), &irrep(1), &irrep(1)).is_empty());
        assert!(Su2Irrep::clebsch_gordan(&irrep(1), &irrep(1), &irrep(4)).is_empty());
    }

    #[test]
    fn cartan_product_is_highest_weight() {
        let cases = [
            (irrep(1), irrep(1)),
            (irrep(1), irrep(2)),
            (irrep(2), irrep(2)),
            (irrep(3), irrep(2)),
            (irrep(0), irrep(5)),
        ];
        for (a, b) in cases {
            let top = *a.tensor_product(&b).last().unwrap();
            assert_eq!(Su2Irrep::cartan_product(&a, &b), top, "cartan {a} ⊗ {b}");
        }
    }

    #[test]
    fn string_round_trip() {
        for (s, t) in [("0", 0), ("1/2", 1), ("1", 2), ("3/2", 3), ("2", 4)] {
            let r = irrep(t);
            assert_eq!(r.to_string(), s);
            assert_eq!(Su2Irrep::parse(s), Some(r));
        }
        assert_eq!(Su2Irrep::parse("5/2"), Some(irrep(5)));
        // An even numerator over 2 is not the reduced form of a half-integer.
        assert_eq!(Su2Irrep::parse("2/2"), None);
        assert_eq!(Su2Irrep::parse("x"), None);
        assert_eq!(Su2Irrep::parse(""), None);
    }
}
