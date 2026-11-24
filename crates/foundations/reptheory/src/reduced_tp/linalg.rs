// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::Zero;

use algebraic_real::AlgebraicReal;

pub(super) fn rational_content_of<'a>(
    values: impl Iterator<Item = &'a AlgebraicReal>,
) -> Option<BigRational> {
    let mut num_gcd = BigInt::zero();
    let mut den_lcm = BigInt::from(1);
    for value in values {
        if let Some(coefficient) = value.canonical_rational_scale() {
            num_gcd = num_gcd.gcd(coefficient.numer());
            den_lcm = den_lcm.lcm(coefficient.denom());
        }
    }
    (!num_gcd.is_zero()).then(|| BigRational::new(num_gcd, den_lcm))
}

pub(super) fn primitive(row: &[AlgebraicReal]) -> Vec<AlgebraicReal> {
    let Some(content) = rational_content_of(row.iter()) else {
        return row.to_vec();
    };
    let inv = AlgebraicReal::rational(content.recip());
    let flip = row
        .iter()
        .find(|e| !e.is_zero())
        .is_some_and(|value| value.signum() < 0);
    row.iter()
        .map(|e| {
            let scaled = e * &inv;
            if flip { -scaled } else { scaled }
        })
        .collect()
}

fn row_echelon(a: &mut [Vec<AlgebraicReal>], cols: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    let mut pivot_row = 0;
    for col in 0..cols {
        let Some(sel) = (pivot_row..a.len()).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };
        a.swap(pivot_row, sel);
        let pivot = a[pivot_row].clone();
        let pivot_val = pivot[col].clone();
        // Eliminate fraction-free: rowᵣ ← pivot_val·rowᵣ − rowᵣ[col]·pivot.
        for (r, target) in a.iter_mut().enumerate() {
            if r == pivot_row || target[col].is_zero() {
                continue;
            }
            let factor = target[col].clone();
            for (t, p) in target.iter_mut().zip(&pivot) {
                *t = &(&pivot_val * &*t) - &(&factor * p);
            }
        }
        pivots.push(col);
        pivot_row += 1;
        if pivot_row == a.len() {
            break;
        }
    }
    pivots
}

pub(super) fn null_space(m: &[Vec<AlgebraicReal>], cols: usize) -> Vec<Vec<AlgebraicReal>> {
    let mut echelon: Vec<Vec<AlgebraicReal>> = m.to_vec();
    let pivots = row_echelon(&mut echelon, cols);
    let pivot_set: std::collections::BTreeSet<usize> = pivots.iter().copied().collect();
    let free: Vec<usize> = (0..cols).filter(|c| !pivot_set.contains(c)).collect();

    // For each free column f: set x[f] = ∏ pivot_vals and solve each pivot
    // fraction-free via x[p] = −coeff · (∏ other pivot_vals). Multiplying
    // through by the full product clears every division — only multiplication.
    let pivot_vals: Vec<AlgebraicReal> = pivots
        .iter()
        .enumerate()
        .map(|(row_idx, &p)| echelon[row_idx][p].clone())
        .collect();
    let prod = pivot_vals
        .iter()
        .fold(AlgebraicReal::one(), |acc, v| &acc * v);

    free.iter()
        .map(|&f| {
            let mut x = vec![AlgebraicReal::zero(); cols];
            x[f] = prod.clone();
            for (row_idx, &p) in pivots.iter().enumerate() {
                // x[p] = −coeff · (∏ pivot_vals) / pivot_val[p]; pivot_val[p]
                // divides the product exactly, so this is division-free.
                let coeff = &echelon[row_idx][f];
                let others = pivots
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != row_idx)
                    .fold(AlgebraicReal::one(), |acc, (j, _)| &acc * &pivot_vals[j]);
                x[p] = -&(coeff * &others);
            }
            primitive(&x)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(vals: &[i64]) -> Vec<AlgebraicReal> {
        vals.iter().map(|&v| AlgebraicReal::integer(v)).collect()
    }

    fn dot(a: &[AlgebraicReal], b: &[AlgebraicReal]) -> AlgebraicReal {
        a.iter()
            .zip(b)
            .fold(AlgebraicReal::zero(), |acc, (x, y)| &acc + &(x * y))
    }

    #[test]
    fn null_space_spans_the_kernel() {
        let eqs = vec![row(&[1, 1, 1]), row(&[1, 0, -1])];
        let ns = null_space(&eqs, 3);
        assert_eq!(ns.len(), 1);
        for eq in &eqs {
            assert!(dot(eq, &ns[0]).is_zero());
        }
        assert_eq!(ns[0], row(&[1, -2, 1]));
    }

    #[test]
    fn null_space_is_primitive_and_sign_canonical() {
        let ns = null_space(&[row(&[2, 4])], 2);
        assert_eq!(ns.len(), 1);
        assert_eq!(ns[0], row(&[2, -1]));
    }

    #[test]
    fn primitive_clears_content_and_sign() {
        let q = |n, d| AlgebraicReal::rational(BigRational::new(BigInt::from(n), BigInt::from(d)));
        let p = primitive(&[q(-1, 2), q(1, 4), AlgebraicReal::zero()]);
        assert_eq!(
            p,
            vec![
                AlgebraicReal::integer(2),
                AlgebraicReal::integer(-1),
                AlgebraicReal::zero()
            ]
        );
    }
}
