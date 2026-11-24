// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use super::basis::{Basis, Segment, unravel};
use crate::irreps::Irreps;
use crate::rep::{CouplingIrrep, Mat};
use algebraic_real::AlgebraicReal;

pub(crate) fn reduce_basis_product<I: CouplingIrrep>(
    basis1: &Basis<I>,
    basis2: &Basis<I>,
    keep_ir: Option<&BTreeSet<I>>,
) -> Basis<I> {
    let b1 = basis1.regroup();
    let b2 = basis2.regroup();

    if b1.segments.is_empty() || b2.segments.is_empty() {
        return Basis {
            irreps: Irreps::new(Vec::new()),
            segments: Vec::new(),
        };
    }

    let dims1 = b1.input_dims().to_vec();
    let dims2 = b2.input_dims().to_vec();
    assert_eq!(dims1.len(), dims2.len(), "input rank mismatch");

    let out_dims: Vec<usize> = dims1
        .iter()
        .zip(&dims2)
        .map(|(&d1, &d2)| {
            assert!(d1 == 1 || d2 == 1 || d1 == d2, "non-broadcastable axes");
            d1.max(d2)
        })
        .collect();

    let mut tasks = Vec::new();
    for ((mul1, ir1), seg1) in b1.irreps.iter().zip(&b1.segments) {
        for ((mul2, ir2), seg2) in b2.irreps.iter().zip(&b2.segments) {
            for ir in ir1.tensor_product(ir2) {
                if keep_ir.is_some_and(|keep| !keep.contains(&ir)) {
                    continue;
                }
                let paths = I::clebsch_gordan(ir1, ir2, &ir);
                if paths.is_empty() {
                    continue;
                }
                tasks.push((
                    *mul1 * *mul2 * paths.len(),
                    ir,
                    seg1,
                    seg2,
                    *mul1,
                    *mul2,
                    paths,
                ));
            }
        }
    }

    let (entries, segments): (Vec<(usize, I)>, Vec<Segment>) = tasks
        .into_par_iter()
        .map(|(new_mul, ir, seg1, seg2, mul1, mul2, paths)| {
            let ir_dim = ir.dimension();
            let seg = fuse_segment(
                seg1, seg2, &paths, mul1, mul2, &dims1, &dims2, &out_dims, ir_dim,
            );
            ((new_mul, ir), seg)
        })
        .unzip();

    Basis {
        irreps: Irreps::new(entries),
        segments,
    }
    .regroup()
}

#[allow(clippy::too_many_arguments)]
fn fuse_segment(
    seg1: &Segment,
    seg2: &Segment,
    paths: &[Mat<AlgebraicReal>],
    mul1: usize,
    mul2: usize,
    dims1: &[usize],
    dims2: &[usize],
    out_dims: &[usize],
    ir_dim: usize,
) -> Segment {
    let j_dim = seg2.ir_dim;
    let num_paths = paths.len();

    let mut cg_by_ij: BTreeMap<(usize, usize), CgPaths> = BTreeMap::new();
    for (w, path) in paths.iter().enumerate() {
        for i in 0..seg1.ir_dim {
            for j in 0..j_dim {
                for k in 0..ir_dim {
                    let cg = path.get(i * j_dim + j, k);
                    if !cg.is_zero() {
                        cg_by_ij.entry((i, j)).or_default().push((w, k, cg.clone()));
                    }
                }
            }
        }
    }

    let out_partial1: Vec<usize> = (0..seg1.input_size())
        .map(|f| out_partial(f, dims1, out_dims))
        .collect();
    let out_partial2: Vec<usize> = (0..seg2.input_size())
        .map(|f| out_partial(f, dims2, out_dims))
        .collect();

    let mut out = Segment::zeros(out_dims.to_vec(), mul1 * mul2 * num_paths, ir_dim);
    for ((flat1, u, i), x1) in seg1.nonzeros() {
        for ((flat2, v, j), x2) in seg2.nonzeros() {
            let Some(cgs) = cg_by_ij.get(&(i, j)) else {
                continue;
            };
            let out_flat = out_partial1[flat1] + out_partial2[flat2];
            let x12 = x1 * x2;
            for (w, k, cg) in cgs {
                let out_mul = (u * mul2 + v) * num_paths + w;
                out.add(out_flat, out_mul, *k, &x12 * cg);
            }
        }
    }
    out
}

type CgPaths = Vec<(usize, usize, AlgebraicReal)>;

fn out_partial(flat: usize, dims: &[usize], out_dims: &[usize]) -> usize {
    let coords = unravel(flat, dims);
    let mut out_flat = 0;
    for ((&c, &d), &out_d) in coords.iter().zip(dims).zip(out_dims) {
        out_flat = out_flat * out_d + if d == 1 { 0 } else { c };
    }
    out_flat
}
