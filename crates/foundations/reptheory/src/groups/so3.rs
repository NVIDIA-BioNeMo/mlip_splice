// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;
use std::sync::{Arc, Mutex};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;

use super::CgCache;
use super::su2::{su2_clebsch_gordan, su2_generators};
use crate::exact::ComplexAlgebraicReal;
use crate::rep::{Irrep, Mat, Rep};
use algebraic_real::AlgebraicReal;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct So3Irrep {
    pub l: u32,
}

impl So3Irrep {
    pub fn new(l: u32) -> Self {
        So3Irrep { l }
    }
}

impl Rep for So3Irrep {
    fn dim(&self) -> usize {
        2 * self.l as usize + 1
    }

    fn generators(&self) -> Vec<Mat<ComplexAlgebraicReal>> {
        real_generators(self.l)
    }
}

impl Irrep for So3Irrep {
    fn trivial(&self) -> Self {
        So3Irrep::new(0)
    }

    fn tensor_product(&self, other: &Self) -> Vec<Self> {
        let lo = self.l.abs_diff(other.l);
        let hi = self.l + other.l;
        (lo..=hi).map(So3Irrep::new).collect()
    }

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>> {
        if c.l < a.l.abs_diff(b.l) || c.l > a.l + b.l {
            return Vec::new();
        }
        vec![real_clebsch_gordan(a.l, b.l, c.l).as_ref().clone()]
    }

    fn cartan_product(a: &Self, b: &Self) -> Self {
        So3Irrep::new(a.l + b.l)
    }

    fn parse(s: &str) -> Option<Self> {
        s.trim().parse::<u32>().ok().map(So3Irrep::new)
    }
}

impl fmt::Display for So3Irrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.l)
    }
}

impl Ord for So3Irrep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.l.cmp(&other.l)
    }
}

impl PartialOrd for So3Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn real_generators(l: u32) -> Vec<Mat<ComplexAlgebraicReal>> {
    let q = change_basis_real_to_complex(l);
    let q_dagger = q.conjugate_transpose();
    su2_generators(2 * l)
        .iter()
        .map(|x| q_dagger.mul(&x.mul(&q)))
        .collect()
}

/// Exact real (SO(3)) Clebsch-Gordan, memoized on `(la, lb, lc)`. The real-basis
/// change of variables over the SU(2) coefficients is deterministic, so the memo
/// returns the identical matrix; both SO3 and O3 route through here.
pub(crate) fn real_clebsch_gordan(la: u32, lb: u32, lc: u32) -> Arc<Mat<AlgebraicReal>> {
    static CACHE: CgCache = CgCache::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Default::default()));
    if let Some(hit) = cache.lock().unwrap().get(&(la, lb, lc)) {
        return hit.clone();
    }
    let mat = Arc::new(compute_real_clebsch_gordan(la, lb, lc));
    cache
        .lock()
        .unwrap()
        .entry((la, lb, lc))
        .or_insert(mat)
        .clone()
}

fn sparse_rows(mat: &Mat<ComplexAlgebraicReal>) -> Vec<Vec<(usize, &ComplexAlgebraicReal)>> {
    (0..mat.rows)
        .map(|i| {
            (0..mat.cols)
                .filter_map(|j| {
                    let v = mat.get(i, j);
                    (!v.is_zero()).then_some((j, v))
                })
                .collect()
        })
        .collect()
}

fn compute_real_clebsch_gordan(la: u32, lb: u32, lc: u32) -> Mat<AlgebraicReal> {
    let su2 = su2_clebsch_gordan(2 * la, 2 * lb, 2 * lc);
    let q1 = change_basis_real_to_complex(la);
    let q2 = change_basis_real_to_complex(lb);
    // Rows of conj(Q3): q3_conj[n][m] = conj(Q3[n,m]).
    let q3_conj = change_basis_real_to_complex(lc).conjugate();
    // Each row of a change-of-basis matrix has at most two nonzeros, so the
    // rotation touches ≤8 real cells per spin-basis entry.
    let (q1_rows, q2_rows, q3_rows) = (sparse_rows(&q1), sparse_rows(&q2), sparse_rows(&q3_conj));

    let (da, db, dc) = (
        2 * la as usize + 1,
        2 * lb as usize + 1,
        2 * lc as usize + 1,
    );

    // Cᵣ[j,l,m] = Σ_{i,k,n} Q1[i,j] Q2[k,l] conj(Q3)[n,m] Csu2[i,k,n].
    let mut acc = vec![ComplexAlgebraicReal::zero(); da * db * dc];
    for (i, q1_row) in q1_rows.iter().enumerate() {
        for (k, q2_row) in q2_rows.iter().enumerate() {
            for (n, q3_row) in q3_rows.iter().enumerate() {
                let coeff = su2.get(i * db + k, n);
                if coeff.is_zero() {
                    continue;
                }
                let csu2 = ComplexAlgebraicReal::real(coeff.clone());
                for &(j, q1v) in q1_row {
                    let w_ij = q1v * &csu2;
                    for &(l, q2v) in q2_row {
                        let w_ijl = &w_ij * q2v;
                        for &(m, q3v) in q3_row {
                            let cell = &mut acc[(j * db + l) * dc + m];
                            *cell = &*cell + &(&w_ijl * q3v);
                        }
                    }
                }
            }
        }
    }

    let mut path = Mat {
        rows: da * db,
        cols: dc,
        entries: vec![AlgebraicReal::zero(); da * db * dc],
    };
    for (cell, value) in acc.into_iter().enumerate() {
        let real = value.into_real();
        if !real.is_zero() {
            path.entries[cell] = real;
        }
    }
    path
}

pub(crate) fn change_basis_real_to_complex(l: u32) -> Mat<ComplexAlgebraicReal> {
    let d = 2 * l as usize + 1;
    let ell = l as i64;
    let inv_sqrt2 = AlgebraicReal::sqrt_rational(BigRational::new(BigInt::one(), BigInt::from(2)))
        .expect("one half is nonnegative");

    let mut q = Mat::zeros(d, d);
    for m in -ell..0 {
        let row = (ell + m) as usize;
        let am = m.unsigned_abs() as usize;
        q.set(
            row,
            ell as usize + am,
            ComplexAlgebraicReal::real(inv_sqrt2.clone()),
        );
        q.set(
            row,
            ell as usize - am,
            ComplexAlgebraicReal::imag(-&inv_sqrt2),
        );
    }
    q.set(
        ell as usize,
        ell as usize,
        ComplexAlgebraicReal::real(AlgebraicReal::one()),
    );
    for m in 1..=ell {
        let row = (ell + m) as usize;
        let am = m as usize;
        let sign = if m % 2 == 0 {
            AlgebraicReal::one()
        } else {
            -AlgebraicReal::one()
        };
        q.set(
            row,
            ell as usize + am,
            ComplexAlgebraicReal::real(&sign * &inv_sqrt2),
        );
        q.set(
            row,
            ell as usize - am,
            ComplexAlgebraicReal::imag(&sign * &inv_sqrt2),
        );
    }

    // Multiply by (-i)^l = {1, -i, -1, i} cycling with l mod 4.
    let phase = match l % 4 {
        0 => ComplexAlgebraicReal::real(AlgebraicReal::one()),
        1 => ComplexAlgebraicReal::imag(-AlgebraicReal::one()),
        2 => ComplexAlgebraicReal::real(-AlgebraicReal::one()),
        _ => ComplexAlgebraicReal::imag(AlgebraicReal::one()),
    };
    for entry in q.entries.iter_mut() {
        *entry = &*entry * &phase;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irrep(l: u32) -> So3Irrep {
        So3Irrep::new(l)
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
    fn so3_commutators_exact() {
        for l in [0u32, 1, 2, 3, 4] {
            let g = irrep(l).generators();
            let (lx, lz, neg_ly) = (&g[0], &g[1], &g[2]);
            let mut ly = Mat::zeros(neg_ly.rows, neg_ly.cols);
            for i in 0..ly.rows {
                for j in 0..ly.cols {
                    ly.set(i, j, -neg_ly.get(i, j));
                }
            }
            assert_eq!(commutator(lx, &ly), *lz, "[Lx,Ly]=Lz at l={l}");
            assert_eq!(commutator(&ly, lz), *lx, "[Ly,Lz]=Lx at l={l}");
            assert_eq!(commutator(lz, lx), ly, "[Lz,Lx]=Ly at l={l}");
        }
    }

    #[test]
    fn generators_are_antisymmetric() {
        for l in [0u32, 1, 2, 3] {
            for g in irrep(l).generators() {
                for i in 0..g.rows {
                    for j in 0..g.cols {
                        assert_eq!(
                            *g.get(i, j),
                            -g.get(j, i),
                            "generator antisymmetric at l={l} ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    fn assert_intertwines(a: &So3Irrep, b: &So3Irrep, c: &So3Irrep) {
        let paths = So3Irrep::clebsch_gordan(a, b, c);
        let (da, db, dc) = (a.dim(), b.dim(), c.dim());
        let ga = a.generators();
        let gb = b.generators();
        let gc = c.generators();

        for path in &paths {
            assert_eq!(path.rows, da * db);
            assert_eq!(path.cols, dc);
            let entry = |j: usize, l: usize, m: usize| path.get(j * db + l, m);
            let real =
                |g: &Mat<ComplexAlgebraicReal>, i: usize, j: usize| g.get(i, j).clone().into_real();

            for component in 0..3 {
                let (la, lb, lc) = (&ga[component], &gb[component], &gc[component]);
                for j in 0..da {
                    for l in 0..db {
                        for m in 0..dc {
                            let mut left = AlgebraicReal::zero();
                            for jp in 0..da {
                                left = &left + &(&real(la, j, jp) * entry(jp, l, m));
                            }
                            for lp in 0..db {
                                left = &left + &(&real(lb, l, lp) * entry(j, lp, m));
                            }
                            let mut right = AlgebraicReal::zero();
                            for mp in 0..dc {
                                right = &right + &(&real(lc, mp, m) * entry(j, l, mp));
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
            (irrep(1), irrep(1), irrep(1)),
            (irrep(1), irrep(1), irrep(2)),
            (irrep(1), irrep(2), irrep(1)),
            (irrep(1), irrep(2), irrep(2)),
            (irrep(1), irrep(2), irrep(3)),
            (irrep(2), irrep(2), irrep(0)),
            (irrep(2), irrep(2), irrep(2)),
            (irrep(2), irrep(2), irrep(4)),
            (irrep(0), irrep(1), irrep(1)),
            (irrep(3), irrep(2), irrep(2)),
        ];
        for (a, b, c) in cases {
            assert_intertwines(&a, &b, &c);
        }
    }

    #[test]
    fn clebsch_gordan_path_orthonormal() {
        let cases = [
            (irrep(1), irrep(1), irrep(0)),
            (irrep(1), irrep(1), irrep(1)),
            (irrep(1), irrep(1), irrep(2)),
            (irrep(1), irrep(2), irrep(2)),
            (irrep(2), irrep(2), irrep(2)),
        ];
        for (a, b, c) in cases {
            let paths = So3Irrep::clebsch_gordan(&a, &b, &c);
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
        assert_eq!(
            irrep(1).tensor_product(&irrep(2)),
            vec![irrep(1), irrep(2), irrep(3)]
        );
        assert_eq!(
            irrep(2).tensor_product(&irrep(2)),
            (0..=4).map(irrep).collect::<Vec<_>>()
        );
        assert!(So3Irrep::clebsch_gordan(&irrep(1), &irrep(1), &irrep(3)).is_empty());
    }

    #[test]
    fn cartan_product_is_highest_weight() {
        let cases = [
            (irrep(1), irrep(1)),
            (irrep(1), irrep(2)),
            (irrep(2), irrep(2)),
            (irrep(3), irrep(2)),
            (irrep(0), irrep(4)),
        ];
        for (a, b) in cases {
            let top = *a.tensor_product(&b).last().unwrap();
            assert_eq!(So3Irrep::cartan_product(&a, &b), top, "cartan {a} ⊗ {b}");
        }
    }

    #[test]
    fn string_round_trip() {
        for (s, l) in [("0", 0), ("1", 1), ("2", 2), ("12", 12)] {
            let r = irrep(l);
            assert_eq!(r.to_string(), s);
            assert_eq!(So3Irrep::parse(s), Some(r));
        }
        assert_eq!(So3Irrep::parse("x"), None);
        assert_eq!(So3Irrep::parse(""), None);
    }
}
