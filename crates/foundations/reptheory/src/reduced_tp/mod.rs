// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use rayon::prelude::*;

pub mod basis;
mod fusion;
mod linalg;
mod sparsify;

use crate::irreps::{Irreps, selection_rule_power, selection_rule_product};
use crate::perm::{SignedPerm, SignedPermGroup};
use crate::rep::{CouplingIrrep, Mat};
use algebraic_real::AlgebraicReal;
use basis::{Basis, Segment, ravel, unravel};
use fusion::reduce_basis_product;

pub fn reduced_tensor_product<I: CouplingIrrep>(
    per_index_irreps: &[Irreps<I>],
    group: &SignedPermGroup,
    keep_ir: Option<&[I]>,
) -> Basis<I> {
    assert_eq!(
        per_index_irreps.len(),
        group.degree,
        "one irreps list per group index"
    );
    let keep: Option<BTreeSet<I>> = keep_ir.map(|k| k.iter().cloned().collect());
    project_invariant_subspace(per_index_irreps, group, keep.as_ref())
}

pub fn reduced_symmetric_tensor_product<I: CouplingIrrep>(
    irreps: &Irreps<I>,
    degree: usize,
    keep_ir: Option<&[I]>,
) -> Basis<I> {
    if degree > 1 && irreps.num_irreps() > 1 {
        let keep: Option<BTreeSet<I>> = keep_ir.map(|k| k.iter().cloned().collect());
        return multinomial_symmetric_tensor_product(irreps, degree, keep.as_ref());
    }
    let per_index = vec![irreps.clone(); degree];
    reduced_tensor_product(&per_index, &SignedPermGroup::symmetric(degree), keep_ir)
}

pub fn reduced_antisymmetric_tensor_product<I: CouplingIrrep>(
    irreps: &Irreps<I>,
    degree: usize,
    keep_ir: Option<&[I]>,
) -> Basis<I> {
    let per_index = vec![irreps.clone(); degree];
    reduced_tensor_product(&per_index, &SignedPermGroup::antisymmetric(degree), keep_ir)
}

fn project_invariant_subspace<I: CouplingIrrep>(
    per_index_irreps: &[Irreps<I>],
    group: &SignedPermGroup,
    keep: Option<&BTreeSet<I>>,
) -> Basis<I> {
    let degree = per_index_irreps.len();
    assert!(degree > 0, "tensor product of zero indices is undefined");

    let rotation = unconstrained_rotation_basis(per_index_irreps, keep);
    if rotation.segments.is_empty() {
        return rotation;
    }
    let dims = rotation.input_dims().to_vec();
    let scatters = index_scatters(&group.generators, &dims);

    let (entries, segments) = rotation
        .irreps
        .iter()
        .zip(&rotation.segments)
        .par_bridge()
        .filter_map(|((_, ir), seg)| {
            let basis = invariant_basis_coefficients(seg, &scatters);
            (!basis.is_empty()).then(|| {
                (
                    (basis.len(), ir.clone()),
                    combine_multiplicities(seg, &basis),
                )
            })
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    sparsify::sparsify_basis(
        Basis {
            irreps: Irreps::new(entries),
            segments,
        }
        .regroup(),
    )
}

fn unconstrained_rotation_basis<I: CouplingIrrep>(
    per_index_irreps: &[Irreps<I>],
    keep: Option<&BTreeSet<I>>,
) -> Basis<I> {
    let degree = per_index_irreps.len();
    let mut cache = FusionCache::new();
    fuse_range(per_index_irreps, 0, degree, keep, &mut cache)
}

type FusionCache<I> = BTreeMap<(Vec<Vec<(usize, I)>>, Option<Vec<I>>), Basis<I>>;

fn fuse_range<I: CouplingIrrep>(
    per_index_irreps: &[Irreps<I>],
    lo: usize,
    hi: usize,
    keep: Option<&BTreeSet<I>>,
    cache: &mut FusionCache<I>,
) -> Basis<I> {
    let sub_keep = restrict_keep(keep, per_index_irreps, lo, hi);
    let key = (
        per_index_irreps[lo..hi]
            .iter()
            .map(|ir| ir.entries.clone())
            .collect(),
        sub_keep.as_ref().map(|k| k.iter().cloned().collect()),
    );
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }

    let basis = if hi - lo == 1 {
        filter_basis(
            identity_basis(&per_index_irreps[lo], 0, 1),
            sub_keep.as_ref(),
        )
    } else {
        let mid = lo + (hi - lo) / 2;
        let left = fuse_range(per_index_irreps, lo, mid, keep, cache);
        let right = fuse_range(per_index_irreps, mid, hi, keep, cache);
        let (k_left, k_right) = (mid - lo, hi - mid);
        let left = embed_range(&left, 0, k_left + k_right);
        let right = embed_range(&right, k_left, k_left + k_right);
        reduce_basis_product(&left, &right, sub_keep.as_ref())
    };

    cache.insert(key, basis.clone());
    basis
}

fn embed_range<I: CouplingIrrep>(basis: &Basis<I>, offset: usize, rank: usize) -> Basis<I> {
    let segments = basis
        .segments
        .iter()
        .map(|seg| {
            let mut dims = vec![1usize; rank];
            dims[offset..offset + seg.input_dims.len()].copy_from_slice(&seg.input_dims);
            seg.with_input_dims(dims)
        })
        .collect();
    Basis {
        irreps: basis.irreps.clone(),
        segments,
    }
}

fn index_scatters(perms: &[SignedPerm], dims: &[usize]) -> Vec<(i8, Vec<usize>)> {
    let input_size: usize = dims.iter().product();
    perms
        .iter()
        .map(|element| {
            let mut dest = vec![0usize; input_size];
            for flat in 0..input_size {
                let coords = unravel(flat, dims);
                let permuted: Vec<usize> = element.perm.iter().map(|&p| coords[p]).collect();
                dest[ravel(&permuted, dims)] = flat;
            }
            (element.sign, dest)
        })
        .collect()
}

struct MultiplicitySupport<'a> {
    mul: usize,
    ir_dim: usize,
    occupants_of: Vec<Vec<(usize, &'a AlgebraicReal)>>,
    nonzeros_of: Vec<Vec<(usize, usize, &'a AlgebraicReal)>>,
}

impl<'a> MultiplicitySupport<'a> {
    fn new(seg: &'a Segment) -> Self {
        let ir_dim = seg.ir_dim;
        let cells = seg.input_size() * ir_dim;
        let mut occupants_of: Vec<Vec<(usize, &AlgebraicReal)>> = vec![Vec::new(); cells];
        let mut nonzeros_of: Vec<Vec<(usize, usize, &AlgebraicReal)>> = vec![Vec::new(); seg.mul];
        for ((src, a, i), v) in seg.nonzeros() {
            occupants_of[src * ir_dim + i].push((a, v));
            nonzeros_of[a].push((src, i, v));
        }
        MultiplicitySupport {
            mul: seg.mul,
            ir_dim,
            occupants_of,
            nonzeros_of,
        }
    }

    fn overlaps(&self, dest: &[usize]) -> Mat<AlgebraicReal> {
        let mut m = Mat::zeros(self.mul, self.mul);
        for (a, ra_nonzeros) in self.nonzeros_of.iter().enumerate() {
            for &(src, i, va) in ra_nonzeros {
                let cell = dest[src] * self.ir_dim + i;
                for &(b, rb) in &self.occupants_of[cell] {
                    let acc = m.get(b, a) + &(rb * va);
                    m.set(b, a, acc);
                }
            }
        }
        m
    }
}

fn invariant_basis_coefficients(
    seg: &Segment,
    scatters: &[(i8, Vec<usize>)],
) -> Vec<Vec<AlgebraicReal>> {
    let mul = seg.mul;
    let support = MultiplicitySupport::new(seg);
    let identity: Vec<usize> = (0..seg.input_size()).collect();
    let gram = support.overlaps(&identity);

    let equations: Vec<Vec<AlgebraicReal>> = scatters
        .par_iter()
        .flat_map_iter(|(sign, dest)| {
            let sign = *sign;
            let rho = support.overlaps(dest);
            let gram = &gram;
            (0..mul).map(move |b| {
                (0..mul)
                    .map(|a| {
                        let m = rho.get(b, a);
                        let g = gram.get(b, a);
                        if sign > 0 { m - g } else { m + g }
                    })
                    .collect()
            })
        })
        .collect();
    linalg::null_space(&equations, mul)
}

fn combine_multiplicities(seg: &Segment, coeffs: &[Vec<AlgebraicReal>]) -> Segment {
    let ir_dim = seg.ir_dim;
    let mut out = Segment::zeros(seg.input_dims.clone(), coeffs.len(), ir_dim);

    let mut combiners: Vec<Vec<(usize, &AlgebraicReal)>> = vec![Vec::new(); seg.mul];
    for (k, coeff) in coeffs.iter().enumerate() {
        for (a, ca) in coeff.iter().enumerate() {
            if !ca.is_zero() {
                combiners[a].push((k, ca));
            }
        }
    }

    for ((flat, a, i), v) in seg.nonzeros() {
        for &(k, ca) in &combiners[a] {
            out.add(flat, k, i, ca * v);
        }
    }
    out
}

fn identity_basis<I: CouplingIrrep>(irreps: &Irreps<I>, index: usize, degree: usize) -> Basis<I> {
    let regrouped = irreps.regroup();
    let mut input_dims = vec![1usize; degree];
    input_dims[index] = irreps.dim();

    let mut segments = Vec::new();
    let mut col_base = 0;
    for (mul, ir) in &regrouped.entries {
        let ir_dim = ir.dimension();
        let mut seg = Segment::zeros(input_dims.clone(), *mul, ir_dim);
        for m in 0..*mul {
            for i in 0..ir_dim {
                let flat = col_base + m * ir_dim + i;
                seg.set(flat, m, i, AlgebraicReal::one());
            }
        }
        col_base += mul * ir_dim;
        segments.push(seg);
    }
    Basis {
        irreps: regrouped,
        segments,
    }
}

fn restrict_keep<I: CouplingIrrep>(
    keep: Option<&BTreeSet<I>>,
    per_index_irreps: &[Irreps<I>],
    lo: usize,
    hi: usize,
) -> Option<BTreeSet<I>> {
    let mut keep = keep?.clone();
    for (index, irreps) in per_index_irreps.iter().enumerate() {
        if (lo..hi).contains(&index) {
            continue;
        }
        let outside: Vec<I> = irreps.iter().map(|(_, ir)| ir.clone()).collect();
        keep = selection_rule_product(&keep, &outside);
    }
    Some(keep)
}

fn multinomial_symmetric_tensor_product<I: CouplingIrrep>(
    irreps: &Irreps<I>,
    degree: usize,
    keep: Option<&BTreeSet<I>>,
) -> Basis<I> {
    let components: Vec<I> = irreps
        .iter()
        .flat_map(|(mul, ir)| std::iter::repeat_n(ir.clone(), *mul))
        .collect();
    let offsets = cumsum_before(components.iter().map(|ir| ir.dimension()));
    let full_dim = irreps.dim();
    let all_irreps: BTreeSet<I> = components.iter().cloned().collect();

    let powers: Vec<Vec<Basis<I>>> = components
        .iter()
        .map(|ir| {
            let single = Irreps::new(vec![(1, ir.clone())]);
            (0..=degree)
                .map(|n| {
                    if n == 0 {
                        return identity_basis(&Irreps::new(vec![(1, ir.trivial())]), 0, 1);
                    }
                    let sub_keep =
                        restrict_keep_power(&ir.trivial(), keep, &all_irreps, degree - n);
                    let keep_vec: Option<Vec<I>> =
                        sub_keep.as_ref().map(|k| k.iter().cloned().collect());
                    reduced_symmetric_tensor_product(&single, n, keep_vec.as_deref())
                })
                .collect()
        })
        .collect();

    let mut terms = Vec::new();
    for term_powers in tuples_with_fixed_sum(components.len(), degree) {
        let nonzero: Vec<(usize, usize)> = term_powers
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n != 0)
            .map(|(k, &n)| (k, n))
            .collect();

        let axis_component: Vec<usize> = nonzero
            .iter()
            .flat_map(|&(k, n)| std::iter::repeat_n(k, n))
            .collect();

        let product = fuse_term(&powers, &nonzero, keep);
        if product.segments.is_empty() {
            continue;
        }
        terms.push(symmetrize_term(
            &product,
            &axis_component,
            &offsets,
            full_dim,
        ));
    }

    sparsify::sparsify_basis(concatenate(terms).regroup())
}

fn fuse_term<I: CouplingIrrep>(
    powers: &[Vec<Basis<I>>],
    nonzero: &[(usize, usize)],
    keep: Option<&BTreeSet<I>>,
) -> Basis<I> {
    let rank: usize = nonzero.iter().map(|&(_, n)| n).sum();
    let mut axis = 0;
    let mut current: Option<Basis<I>> = None;
    for (factor_idx, &(k, n)) in nonzero.iter().enumerate() {
        let embedded = embed_range(&powers[k][n], axis, rank);
        axis += n;
        let last = factor_idx + 1 == nonzero.len();
        let factor_keep = if last { keep } else { None };
        current = Some(match current {
            None if last => filter_basis(embedded, keep),
            None => embedded,
            Some(acc) => reduce_basis_product(&acc, &embedded, factor_keep),
        });
    }
    current.expect("a term summing to degree > 0 has at least one factor")
}

fn filter_basis<I: CouplingIrrep>(basis: Basis<I>, keep: Option<&BTreeSet<I>>) -> Basis<I> {
    let Some(keep) = keep else { return basis };
    let mut entries = Vec::new();
    let mut segments = Vec::new();
    for ((mul, ir), seg) in basis.irreps.entries.into_iter().zip(basis.segments) {
        if keep.contains(&ir) {
            entries.push((mul, ir));
            segments.push(seg);
        }
    }
    Basis {
        irreps: Irreps::new(entries),
        segments,
    }
}

fn symmetrize_term<I: CouplingIrrep>(
    product: &Basis<I>,
    axis_component: &[usize],
    offsets: &[usize],
    full_dim: usize,
) -> Basis<I> {
    let degree = axis_component.len();
    let placements = distinct_permutations(axis_component);
    // 1/√(#placements) = √(1/#placements).
    let weight =
        AlgebraicReal::sqrt_rational(BigRational::new(BigInt::one(), placements.len().into()))
            .expect("placement count is positive");

    let full_dims = vec![full_dim; degree];
    let mut segments: Vec<Segment> = product
        .segments
        .iter()
        .map(|seg| Segment::zeros(full_dims.clone(), seg.mul, seg.ir_dim))
        .collect();

    for perm in &placements {
        for (seg, out) in product.segments.iter().zip(&mut segments) {
            place_permuted(seg, perm, axis_component, offsets, &full_dims, &weight, out);
        }
    }

    Basis {
        irreps: product.irreps.clone(),
        segments,
    }
}

fn place_permuted(
    seg: &Segment,
    perm: &[usize],
    axis_component: &[usize],
    offsets: &[usize],
    full_dims: &[usize],
    weight: &AlgebraicReal,
    out: &mut Segment,
) {
    let src_dims = &seg.input_dims;
    let degree = full_dims.len();
    let mut full_coords = vec![0usize; degree];
    for ((src_flat, m, i), v) in seg.nonzeros() {
        let src_coords = unravel(src_flat, src_dims);
        for (p, &q) in perm.iter().enumerate() {
            full_coords[p] = offsets[axis_component[q]] + src_coords[q];
        }
        out.add(ravel(&full_coords, full_dims), m, i, weight * v);
    }
}

fn concatenate<I: CouplingIrrep>(terms: Vec<Basis<I>>) -> Basis<I> {
    let mut entries = Vec::new();
    let mut segments = Vec::new();
    for term in terms {
        for (entry, seg) in term.irreps.entries.into_iter().zip(term.segments) {
            entries.push(entry);
            segments.push(seg);
        }
    }
    Basis {
        irreps: Irreps::new(entries),
        segments,
    }
}

fn restrict_keep_power<I: CouplingIrrep>(
    trivial: &I,
    keep: Option<&BTreeSet<I>>,
    all_irreps: &BTreeSet<I>,
    extra: usize,
) -> Option<BTreeSet<I>> {
    let keep = keep?;
    let reachable = selection_rule_power(trivial, all_irreps, extra);
    Some(selection_rule_product(
        keep,
        &reachable.iter().cloned().collect::<Vec<_>>(),
    ))
}

fn tuples_with_fixed_sum(length: usize, total: usize) -> Vec<Vec<usize>> {
    if length == 1 {
        return vec![vec![total]];
    }
    let mut out = Vec::new();
    for first in (0..=total).rev() {
        for mut rest in tuples_with_fixed_sum(length - 1, total - first) {
            let mut tuple = Vec::with_capacity(length);
            tuple.push(first);
            tuple.append(&mut rest);
            out.push(tuple);
        }
    }
    out
}

fn distinct_permutations(seq: &[usize]) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..seq.len()).collect();
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    loop {
        let arrangement: Vec<usize> = indices.iter().map(|&i| seq[i]).collect();
        if seen.insert(arrangement) {
            out.push(indices.clone());
        }
        if !next_permutation(&mut indices) {
            break;
        }
    }
    out
}

fn next_permutation(perm: &mut [usize]) -> bool {
    let n = perm.len();
    if n < 2 {
        return false;
    }
    let Some(i) = (0..n - 1).rev().find(|&i| perm[i] < perm[i + 1]) else {
        return false;
    };
    let j = (i + 1..n).rev().find(|&j| perm[j] > perm[i]).unwrap();
    perm.swap(i, j);
    perm[i + 1..].reverse();
    true
}

fn cumsum_before(sizes: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut acc = 0;
    sizes
        .into_iter()
        .map(|s| {
            let before = acc;
            acc += s;
            before
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::O3Irrep;

    fn irreps(s: &str) -> Irreps<O3Irrep> {
        Irreps::parse(s).unwrap()
    }

    #[test]
    fn sparsified_basis_is_all_single_surds() {
        let expected_nnz = [
            ("1x1o", 4, 259),
            ("1x1o", 5, 1065),
            ("1x0e+1x1o", 4, 604),
            ("1x0e+1x1o+1x2e", 3, 3757),
        ];
        for (s, deg, want_nnz) in expected_nnz {
            let b = reduced_symmetric_tensor_product(&irreps(s), deg, None);
            let (mut nnz, mut terms) = (0, 0);
            for seg in &b.segments {
                for (_, v) in seg.nonzeros() {
                    nnz += 1;
                    terms += v.structural_size();
                }
            }
            assert_eq!(nnz, terms, "{s}^{deg}: every entry must be a single surd");
            assert_eq!(nnz, want_nnz, "{s}^{deg} nonzero count");
        }
    }

    #[test]
    fn symmetric_square_of_1o() {
        let b = reduced_symmetric_tensor_product(&irreps("1x1o"), 2, None);
        assert_eq!(b.irreps.to_string(), "1x0e+1x2e");
        assert_eq!(b.input_dims(), &[3, 3]);
    }

    #[test]
    fn antisymmetric_square_of_1o() {
        let b = reduced_antisymmetric_tensor_product(&irreps("1x1o"), 2, None);
        assert_eq!(b.irreps.to_string(), "1x1e");
    }

    #[test]
    fn keep_ir_filters_output() {
        let keep = [O3Irrep::new(2, 1)];
        let b = reduced_symmetric_tensor_product(&irreps("1x1o"), 2, Some(&keep));
        assert_eq!(b.irreps.to_string(), "1x2e");
        assert_eq!(b.segments.len(), 1);
    }

    #[test]
    fn keep_ir_unreachable_is_empty() {
        let keep = [O3Irrep::new(0, 1)];
        let b = reduced_symmetric_tensor_product(&irreps("1x1o"), 5, Some(&keep));
        assert!(b.irreps.entries.is_empty());
        assert!(b.segments.is_empty());
    }
}
