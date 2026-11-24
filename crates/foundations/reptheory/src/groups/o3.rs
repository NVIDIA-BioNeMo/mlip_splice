// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::fmt;

use super::so3::{real_clebsch_gordan, real_generators};
use crate::exact::ComplexAlgebraicReal;
use crate::rep::{Irrep, Mat, Rep};
use algebraic_real::AlgebraicReal;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct O3Irrep {
    pub l: u32,
    pub parity: i8,
}

impl O3Irrep {
    pub fn new(l: u32, parity: i8) -> Self {
        assert!(parity == 1 || parity == -1, "parity must be ±1");
        O3Irrep { l, parity }
    }
}

impl Rep for O3Irrep {
    fn dim(&self) -> usize {
        2 * self.l as usize + 1
    }

    fn generators(&self) -> Vec<Mat<ComplexAlgebraicReal>> {
        real_generators(self.l)
    }
}

impl Irrep for O3Irrep {
    fn trivial(&self) -> Self {
        O3Irrep::new(0, 1)
    }

    fn tensor_product(&self, other: &Self) -> Vec<Self> {
        let parity = self.parity * other.parity;
        let lo = self.l.abs_diff(other.l);
        let hi = self.l + other.l;
        (lo..=hi).map(|l| O3Irrep::new(l, parity)).collect()
    }

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>> {
        if c.parity != a.parity * b.parity {
            return Vec::new();
        }
        if c.l < a.l.abs_diff(b.l) || c.l > a.l + b.l {
            return Vec::new();
        }
        vec![real_clebsch_gordan(a.l, b.l, c.l).as_ref().clone()]
    }

    fn cartan_product(a: &Self, b: &Self) -> Self {
        O3Irrep::new(a.l + b.l, a.parity * b.parity)
    }

    fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        let (digits, tag) = s.split_at(s.len().checked_sub(1)?);
        let parity = match tag {
            "e" => 1,
            "o" => -1,
            _ => return None,
        };
        let l = digits.parse::<u32>().ok()?;
        Some(O3Irrep::new(l, parity))
    }
}

impl fmt::Display for O3Irrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = if self.parity > 0 { 'e' } else { 'o' };
        write!(f, "{}{}", self.l, tag)
    }
}

impl Ord for O3Irrep {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by degree, then by natural-parity agreement.
        let key = |r: &O3Irrep| {
            let sign = if r.l.is_multiple_of(2) { 1 } else { -1 };
            (r.l, -(r.parity as i32) * sign)
        };
        key(self).cmp(&key(other))
    }
}

impl PartialOrd for O3Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irrep(l: u32, parity: i8) -> O3Irrep {
        O3Irrep::new(l, parity)
    }

    fn assert_intertwines(a: &O3Irrep, b: &O3Irrep, c: &O3Irrep) {
        let paths = O3Irrep::clebsch_gordan(a, b, c);
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
            (irrep(1, -1), irrep(1, -1), irrep(0, 1)),
            (irrep(1, -1), irrep(1, -1), irrep(1, 1)),
            (irrep(1, -1), irrep(1, -1), irrep(2, 1)),
            (irrep(1, -1), irrep(2, 1), irrep(1, -1)),
            (irrep(1, -1), irrep(2, 1), irrep(2, -1)),
            (irrep(1, -1), irrep(2, 1), irrep(3, -1)),
            (irrep(2, 1), irrep(2, 1), irrep(0, 1)),
            (irrep(2, 1), irrep(2, 1), irrep(2, 1)),
            (irrep(2, 1), irrep(2, 1), irrep(4, 1)),
            (irrep(0, 1), irrep(1, -1), irrep(1, -1)),
            (irrep(3, -1), irrep(2, 1), irrep(2, -1)),
            (irrep(2, -1), irrep(2, -1), irrep(3, 1)),
        ];
        for (a, b, c) in cases {
            assert_intertwines(&a, &b, &c);
        }
    }

    #[test]
    fn clebsch_gordan_path_orthonormal() {
        // the single path is normalized so
        // that Σ_{j,l} C[j,l,m] C[j,l,m'] = δ_{m,m'} exactly for each (a,b,c)
        // where a path exists.
        let cases = [
            (irrep(1, -1), irrep(1, -1), irrep(0, 1)),
            (irrep(1, -1), irrep(1, -1), irrep(1, 1)),
            (irrep(1, -1), irrep(1, -1), irrep(2, 1)),
            (irrep(1, -1), irrep(2, 1), irrep(2, -1)),
            (irrep(2, 1), irrep(2, 1), irrep(2, 1)),
        ];
        for (a, b, c) in cases {
            let paths = O3Irrep::clebsch_gordan(&a, &b, &c);
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
        let prod = irrep(1, -1).tensor_product(&irrep(2, 1));
        assert_eq!(
            prod,
            vec![irrep(1, -1), irrep(2, -1), irrep(3, -1)],
            "l ∈ |1−2|..=1+2, parity (−1)(+1)=−1"
        );
        let prod = irrep(2, 1).tensor_product(&irrep(2, 1));
        assert_eq!(prod, (0..=4).map(|l| irrep(l, 1)).collect::<Vec<_>>());
        assert!(O3Irrep::clebsch_gordan(&irrep(1, -1), &irrep(1, -1), &irrep(1, -1)).is_empty());
        assert!(O3Irrep::clebsch_gordan(&irrep(1, -1), &irrep(1, -1), &irrep(3, 1)).is_empty());
    }

    #[test]
    fn cartan_product_is_highest_weight() {
        let cases = [
            (irrep(1, -1), irrep(1, -1)),
            (irrep(1, -1), irrep(2, 1)),
            (irrep(2, 1), irrep(2, 1)),
            (irrep(3, -1), irrep(2, 1)),
            (irrep(0, 1), irrep(4, -1)),
        ];
        for (a, b) in cases {
            let top = *a.tensor_product(&b).last().unwrap();
            assert_eq!(O3Irrep::cartan_product(&a, &b), top, "cartan {a} ⊗ {b}");
        }
    }

    #[test]
    fn string_round_trip() {
        for (s, l, p) in [("0e", 0, 1), ("1o", 1, -1), ("2e", 2, 1), ("3o", 3, -1)] {
            let r = irrep(l, p);
            assert_eq!(r.to_string(), s);
            assert_eq!(O3Irrep::parse(s), Some(r));
        }
        assert_eq!(O3Irrep::parse("12e"), Some(irrep(12, 1)));
        assert_eq!(O3Irrep::parse("2x"), None);
        assert_eq!(O3Irrep::parse(""), None);
    }
}
