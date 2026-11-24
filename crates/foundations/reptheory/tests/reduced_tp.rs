// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use reptheory::basis::Basis;
use reptheory::exact::ComplexAlgebraicReal;
use reptheory::irreps::Irreps;
use reptheory::perm::SignedPermGroup;
use reptheory::rep::Mat;
use reptheory::{Irrep, O3Irrep, So3Irrep, Su2Irrep};

fn irreps_generators<I: Irrep>(irreps: &Irreps<I>) -> [Mat<ComplexAlgebraicReal>; 3] {
    let dim = irreps.dim();
    let mut gens = [
        Mat::zeros(dim, dim),
        Mat::zeros(dim, dim),
        Mat::zeros(dim, dim),
    ];
    let mut base = 0;
    for (mul, ir) in irreps.iter() {
        let ir_gen = ir.generators();
        let d = ir.dim();
        for _ in 0..*mul {
            for (comp, g) in ir_gen.iter().enumerate() {
                for i in 0..d {
                    for j in 0..d {
                        gens[comp].set(base + i, base + j, g.get(i, j).clone());
                    }
                }
            }
            base += d;
        }
    }
    gens
}

fn columns_equivariant<I: Irrep>(basis: &Basis<I>, input_dims_irreps: &[Irreps<I>]) -> bool {
    let q = basis.to_columns();
    let input_dims: Vec<usize> = input_dims_irreps.iter().map(Irreps::dim).collect();
    let per_axis_gens: Vec<[Mat<ComplexAlgebraicReal>; 3]> =
        input_dims_irreps.iter().map(irreps_generators).collect();
    let out_gens = irreps_generators(&basis.irreps);
    let column = |r: usize, c: usize| ComplexAlgebraicReal::real(q.get(r, c).clone());

    for comp in 0..3 {
        for out in 0..q.cols {
            for input_flat in 0..q.rows {
                let coords = unravel(input_flat, &input_dims);
                let mut left = ComplexAlgebraicReal::zero();
                for (axis, axis_gens) in per_axis_gens.iter().enumerate() {
                    let g = &axis_gens[comp];
                    for xp in 0..input_dims[axis] {
                        let mut coords2 = coords.clone();
                        coords2[axis] = xp;
                        let flat2 = ravel(&coords2, &input_dims);
                        left = &left + &(g.get(coords[axis], xp) * &column(flat2, out));
                    }
                }
                let mut right = ComplexAlgebraicReal::zero();
                let gout = &out_gens[comp];
                for outp in 0..q.cols {
                    right = &right + &(gout.get(outp, out) * &column(input_flat, outp));
                }
                if !(&left - &right).is_zero() {
                    return false;
                }
            }
        }
    }
    true
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for (c, &d) in coords.iter_mut().zip(shape).rev() {
        *c = flat % d;
        flat /= d;
    }
    coords
}

fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    for (&c, &d) in coords.iter().zip(shape) {
        flat = flat * d + c;
    }
    flat
}

fn columns_permutation_symmetric<I: Irrep>(basis: &Basis<I>, group: &SignedPermGroup) -> bool {
    let input_dims = basis.input_dims().to_vec();
    let q = basis.to_columns();
    for element in &group.elements {
        for out in 0..q.cols {
            for input_flat in 0..q.rows {
                let coords = unravel(input_flat, &input_dims);
                let permuted: Vec<usize> = element.perm.iter().map(|&p| coords[p]).collect();
                let pf = ravel(&permuted, &input_dims);
                let lhs = q.get(pf, out);
                let rhs = if element.sign < 0 {
                    -q.get(input_flat, out)
                } else {
                    q.get(input_flat, out).clone()
                };
                if !(lhs - &rhs).is_zero() {
                    return false;
                }
            }
        }
    }
    true
}

fn columns_independent<I: Irrep>(basis: &Basis<I>) -> bool {
    let q = basis.to_columns();
    let mut gram: Vec<Vec<AlgebraicReal>> = vec![vec![AlgebraicReal::zero(); q.cols]; q.cols];
    for (a, ga) in gram.iter_mut().enumerate() {
        for (b, gab) in ga.iter_mut().enumerate() {
            let mut acc = AlgebraicReal::zero();
            for r in 0..q.rows {
                acc = &acc + &(q.get(r, a) * q.get(r, b));
            }
            *gab = acc;
        }
    }
    rank(&mut gram, q.cols) == q.cols
}

fn rank(a: &mut [Vec<AlgebraicReal>], cols: usize) -> usize {
    let mut pivot_row = 0;
    for col in 0..cols {
        let Some(sel) = (pivot_row..a.len()).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };
        a.swap(pivot_row, sel);
        let pivot = a[pivot_row].clone();
        let pivot_val = pivot[col].clone();
        for (r, target) in a.iter_mut().enumerate() {
            if r == pivot_row || target[col].is_zero() {
                continue;
            }
            let factor = target[col].clone();
            for (t, p) in target.iter_mut().zip(&pivot) {
                *t = &(&pivot_val * &*t) - &(&factor * p);
            }
        }
        pivot_row += 1;
        if pivot_row == a.len() {
            break;
        }
    }
    pivot_row
}

fn invariant_dimension(input_dims: &[usize], group: &SignedPermGroup) -> i64 {
    let mut total: i64 = 0;
    for element in &group.elements {
        let mut seen = vec![false; element.perm.len()];
        let mut fixed: i64 = 1;
        for start in 0..element.perm.len() {
            if seen[start] {
                continue;
            }
            let mut i = start;
            while !seen[i] {
                seen[i] = true;
                i = element.perm[i];
            }
            fixed *= input_dims[start] as i64;
        }
        total += element.sign as i64 * fixed;
    }
    let order = group.order() as i64;
    assert_eq!(total % order, 0, "character sum not divisible by |G|");
    total / order
}

fn assert_valid_change_of_basis<I: Irrep>(
    b: &Basis<I>,
    per_index: &[Irreps<I>],
    group: &SignedPermGroup,
    label: &str,
) {
    assert!(columns_equivariant(b, per_index), "equivariance {label}");
    assert!(
        columns_permutation_symmetric(b, group),
        "permutation symmetry {label}"
    );
    assert!(columns_independent(b), "independence {label}");
    let input_dims: Vec<usize> = per_index.iter().map(Irreps::dim).collect();
    assert_eq!(
        b.irreps.dim() as i64,
        invariant_dimension(&input_dims, group),
        "dimension {label}"
    );
}

fn assert_reduced_tensor_product<I: Irrep>(
    per_index: &[Irreps<I>],
    group: &SignedPermGroup,
    label: &str,
) {
    let b = reptheory::reduced_tensor_product(per_index, group, None);
    assert_valid_change_of_basis(&b, per_index, group, label);
}

fn assert_symmetric_path<I: Irrep + std::fmt::Display>(ir: &Irreps<I>, degree: usize) {
    let b = reptheory::reduced_symmetric_tensor_product(ir, degree, None);
    let per_index = vec![ir.clone(); degree];
    let group = SignedPermGroup::symmetric(degree);
    let label = format!("Sym^{degree}({ir})");
    assert_valid_change_of_basis(&b, &per_index, &group, &label);
}

fn assert_dimension_matches<I: Irrep>(
    per_index: &[Irreps<I>],
    group: &SignedPermGroup,
    label: &str,
) {
    let b = reptheory::reduced_tensor_product(per_index, group, None);
    let input_dims: Vec<usize> = per_index.iter().map(Irreps::dim).collect();
    assert_eq!(
        b.irreps.dim() as i64,
        invariant_dimension(&input_dims, group),
        "output dimension != invariant-subspace dimension for {label}"
    );
}

fn o3_cases() -> Vec<(Vec<Irreps<O3Irrep>>, SignedPermGroup, String)> {
    group_cases(&[
        ("1x1o", " S2"),
        ("1x1o", "A2"),
        ("1x1o", "S3"),
        ("1x1o", "A3"),
        ("1x2e", "S3"),
        ("1x1o+1x2e", "S2"),
        ("1x1o+1x2e", "A2"),
        ("1x0e+1x1o", "S3"),
        ("1x1o", "ijk=jik"),
        ("1x1o", "ijk=jik=kij"),
        ("1x1o+1x2e", "ijk=jik"),
    ])
}

fn so3_cases() -> Vec<(Vec<Irreps<So3Irrep>>, SignedPermGroup, String)> {
    group_cases(&[
        ("1x1", "S2"),
        ("1x1", "A2"),
        ("1x1", "S3"),
        ("1x2", "S3"),
        ("1x0+1x1", "S3"),
        ("1x1", "ijk=jik"),
    ])
}

fn su2_cases() -> Vec<(Vec<Irreps<Su2Irrep>>, SignedPermGroup, String)> {
    group_cases(&[
        ("1x1/2", "S2"),
        ("1x1/2", "A2"),
        ("1x1", "S2"),
        ("1x1/2", "S3"),
        ("1x1/2+1x1", "S2"),
        ("1x1/2+1x1", "A2"),
    ])
}

fn group_cases<I: Irrep>(specs: &[(&str, &str)]) -> Vec<(Vec<Irreps<I>>, SignedPermGroup, String)> {
    specs
        .iter()
        .map(|&(ir_spec, group_spec)| {
            let ir: Irreps<I> = Irreps::parse(ir_spec).unwrap();
            let (degree, group) = parse_group(group_spec);
            let label = format!("{group_spec} of {ir_spec}");
            (vec![ir; degree], group, label)
        })
        .collect()
}

fn parse_group(spec: &str) -> (usize, SignedPermGroup) {
    if let Some(n) = spec.strip_prefix('S') {
        let n: usize = n.trim().parse().unwrap();
        (n, SignedPermGroup::symmetric(n))
    } else if let Some(n) = spec.strip_prefix('A') {
        let n: usize = n.trim().parse().unwrap();
        (n, SignedPermGroup::antisymmetric(n))
    } else {
        let (indices, group) = reptheory::perm::parse_formula(spec).unwrap();
        (indices.len(), group)
    }
}

#[test]
fn exact_equivariance() {
    for (per_index, group, label) in o3_cases() {
        assert_reduced_tensor_product(&per_index, &group, &label);
    }
    for (per_index, group, label) in so3_cases() {
        assert_reduced_tensor_product(&per_index, &group, &label);
    }
    for (per_index, group, label) in su2_cases() {
        assert_reduced_tensor_product(&per_index, &group, &label);
    }
}

#[test]
fn symmetric_path_is_a_valid_change_of_basis() {
    let o3: [(&str, usize); 4] = [
        ("1x1o+1x2e", 2),
        ("1x0e+1x1o", 3),
        ("1x1o+1x2e", 3),
        ("1x0e+1x1o", 4),
    ];
    for (ir, degree) in o3 {
        assert_symmetric_path::<O3Irrep>(&Irreps::parse(ir).unwrap(), degree);
    }
    for (ir, degree) in [("1x0+1x1", 3), ("1x1+1x2", 2)] {
        assert_symmetric_path::<So3Irrep>(&Irreps::parse(ir).unwrap(), degree);
    }
    for (ir, degree) in [("1x1/2+1x1", 2), ("1x1/2+1x1", 3)] {
        assert_symmetric_path::<Su2Irrep>(&Irreps::parse(ir).unwrap(), degree);
    }
}

#[test]
fn dimension_matches_character_formula() {
    for (per_index, group, label) in o3_cases() {
        assert_dimension_matches(&per_index, &group, &label);
    }
    for (per_index, group, label) in so3_cases() {
        assert_dimension_matches(&per_index, &group, &label);
    }
    for (per_index, group, label) in su2_cases() {
        assert_dimension_matches(&per_index, &group, &label);
    }
}
