// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::HashSet;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use rayon::prelude::*;

use super::basis::{Basis, Segment};
use super::linalg::rational_content_of;
use crate::rep::CouplingIrrep;
use algebraic_real::AlgebraicReal;

type Row = BTreeMap<usize, AlgebraicReal>;

pub(crate) fn sparsify_basis<I: CouplingIrrep>(basis: Basis<I>) -> Basis<I> {
    let segments = basis
        .segments
        .into_par_iter()
        .map(sparsify_segment)
        .collect();
    Basis {
        irreps: basis.irreps,
        segments,
    }
}

fn sparsify_segment(seg: Segment) -> Segment {
    let ir_dim = seg.ir_dim;
    let mut rows: Vec<Row> = vec![BTreeMap::new(); seg.mul];
    for ((flat, m, i), v) in seg.nonzeros() {
        rows[m].insert(flat * ir_dim + i, v.clone());
    }

    let rows = minimize_term_count(rows);

    let mut out = Segment::zeros(seg.input_dims.clone(), seg.mul, ir_dim);
    for (m, row) in rows.iter().enumerate() {
        for (col, v) in normalize_pow2(row) {
            out.set(col / ir_dim, m, col % ir_dim, v);
        }
    }
    out
}

fn cost(row: &Row) -> usize {
    row.values().map(AlgebraicReal::structural_size).sum()
}

fn minimize_term_count(rows: Vec<Row>) -> Vec<Row> {
    let primitive: Vec<Row> = rows.iter().map(primitive_row).collect();
    let n = primitive.len();
    let orders: Vec<Vec<usize>> = restart_orders(n).collect();
    orders
        .into_par_iter()
        .enumerate()
        .map(|(k, order)| {
            let rows = sweep_descent(order.iter().map(|&i| primitive[i].clone()).collect());
            let total = rows.iter().map(cost).sum::<usize>();
            (rows, total, k)
        })
        .min_by_key(|&(_, total, k)| (total, k))
        .map(|(rows, _, _)| rows)
        .unwrap_or(primitive)
}

fn restart_orders(n: usize) -> impl Iterator<Item = Vec<usize>> {
    let identity: Vec<usize> = (0..n).collect();
    let reversed: Vec<usize> = (0..n).rev().collect();
    let interleaved: Vec<usize> = (0..n).step_by(2).chain((1..n).step_by(2)).collect();
    let extra = (n > 1)
        .then_some([reversed, interleaved])
        .into_iter()
        .flatten();
    std::iter::once(identity).chain(extra)
}

fn sweep_descent(mut rows: Vec<Row>) -> Vec<Row> {
    loop {
        let mut improved = false;
        for a in 0..rows.len() {
            for b in 0..rows.len() {
                if a == b {
                    continue;
                }
                if let Some((delta, new_a)) = best_pair_reduction(&rows[a], &rows[b])
                    && delta < 0
                {
                    rows[a] = primitive_row(&new_a);
                    improved = true;
                }
            }
        }
        if !improved {
            return rows;
        }
    }
}

#[allow(clippy::mutable_key_type)]
fn best_pair_reduction(a: &Row, b: &Row) -> Option<(isize, Row)> {
    let base = cost(a) as isize;
    let mut seen: HashSet<AlgebraicReal> = HashSet::new();
    let mut best: Option<(isize, Row)> = None;
    for (&col, bv) in b {
        // ρ = a[c]/b[c] exists exactly only where b[c] is a single surd.
        let Some(inv) = bv.recip() else {
            continue;
        };
        let Some(av) = a.get(&col) else { continue };
        let rho = av * &inv;
        if !seen.insert(rho.clone()) {
            continue;
        }
        let new_a = combine(a, b, &rho);
        let delta = cost(&new_a) as isize - base;
        if best.as_ref().is_none_or(|(d, _)| delta < *d) {
            best = Some((delta, new_a));
        }
    }
    best
}

fn combine(a: &Row, b: &Row, rho: &AlgebraicReal) -> Row {
    let mut out = a.clone();
    for (&col, bv) in b {
        let term = rho * bv;
        match out.entry(col) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(-&term);
            }
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let diff = e.get() - &term;
                if diff.is_zero() {
                    e.remove();
                } else {
                    *e.get_mut() = diff;
                }
            }
        }
    }
    out
}

fn primitive_row(row: &Row) -> Row {
    let Some(content) = rational_content_of(row.values()) else {
        return row.clone();
    };
    let inv = AlgebraicReal::rational(content.recip());
    let flip = row.values().next().is_some_and(|value| value.signum() < 0);
    row.iter()
        .map(|(&col, v)| {
            let scaled = v * &inv;
            (col, if flip { -scaled } else { scaled })
        })
        .collect()
}

fn normalize_pow2(row: &Row) -> Row {
    let norm_sq = row
        .values()
        .fold(AlgebraicReal::zero(), |acc, v| &acc + &(v * v));
    if norm_sq.is_zero() {
        return row.clone();
    }
    let k = (0.5 * norm_sq.to_f64().log2()).round() as i32;
    let scale = if k >= 0 {
        AlgebraicReal::rational(BigRational::new(BigInt::one(), BigInt::one() << (k as u32)))
    } else {
        AlgebraicReal::rational(BigRational::from_integer(BigInt::one() << ((-k) as u32)))
    };
    row.iter().map(|(&col, v)| (col, v * &scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(entries: &[(usize, i64)]) -> Row {
        entries
            .iter()
            .map(|&(c, v)| (c, AlgebraicReal::integer(v)))
            .collect()
    }

    #[test]
    fn cancels_a_shared_block() {
        let rows = vec![
            row(&[(0, 1), (1, 1), (2, 1)]),
            row(&[(1, 1), (2, 1), (3, 1)]),
        ];
        let out = minimize_term_count(rows);
        assert_eq!(
            out,
            vec![row(&[(0, 1), (3, -1)]), row(&[(1, 1), (2, 1), (3, 1)])]
        );
        assert_eq!(out.iter().map(cost).sum::<usize>(), 5);
    }

    #[test]
    fn already_sparse_is_untouched() {
        let rows = vec![row(&[(0, -2)]), row(&[(1, 3)])];
        let out = minimize_term_count(rows);
        assert_eq!(out, vec![row(&[(0, 1)]), row(&[(1, 1)])]);
    }
}
