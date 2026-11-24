// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use crate::exact::SparseTensor;
use crate::irreps::Irreps;
use crate::rep::CouplingIrrep;
use algebraic_real::AlgebraicReal;

pub(crate) fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for (c, &d) in coords.iter_mut().zip(shape).rev() {
        *c = flat % d;
        flat /= d;
    }
    coords
}

pub(crate) fn ravel(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    for (&c, &d) in coords.iter().zip(shape) {
        flat = flat * d + c;
    }
    flat
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Segment {
    pub input_dims: Vec<usize>,
    pub mul: usize,
    pub ir_dim: usize,
    entries: BTreeMap<usize, AlgebraicReal>,
}

impl Segment {
    pub fn zeros(input_dims: Vec<usize>, mul: usize, ir_dim: usize) -> Self {
        Segment {
            input_dims,
            mul,
            ir_dim,
            entries: BTreeMap::new(),
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_dims.iter().product()
    }

    pub fn offset(&self, input_flat: usize, m: usize, i: usize) -> usize {
        (input_flat * self.mul + m) * self.ir_dim + i
    }
    pub fn set(&mut self, input_flat: usize, m: usize, i: usize, value: AlgebraicReal) {
        let off = self.offset(input_flat, m, i);
        if value.is_zero() {
            self.entries.remove(&off);
        } else {
            self.entries.insert(off, value);
        }
    }

    pub fn add(&mut self, input_flat: usize, m: usize, i: usize, value: AlgebraicReal) {
        let off = self.offset(input_flat, m, i);
        match self.entries.entry(off) {
            std::collections::btree_map::Entry::Vacant(e) => {
                if !value.is_zero() {
                    e.insert(value);
                }
            }
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let sum = e.get() + &value;
                if sum.is_zero() {
                    e.remove();
                } else {
                    *e.get_mut() = sum;
                }
            }
        }
    }

    pub fn with_input_dims(&self, input_dims: Vec<usize>) -> Segment {
        assert_eq!(
            input_dims.iter().product::<usize>(),
            self.input_size(),
            "reshape changes the input size"
        );
        Segment {
            input_dims,
            mul: self.mul,
            ir_dim: self.ir_dim,
            entries: self.entries.clone(),
        }
    }

    pub fn into_sparse_tensor(self) -> SparseTensor<AlgebraicReal> {
        let mut shape = self.input_dims;
        shape.push(self.mul);
        shape.push(self.ir_dim);
        SparseTensor::from_flat(shape, self.entries)
    }

    pub fn nonzeros(&self) -> impl Iterator<Item = ((usize, usize, usize), &AlgebraicReal)> {
        let mul = self.mul;
        let ir_dim = self.ir_dim;
        self.entries.iter().map(move |(&off, v)| {
            let i = off % ir_dim;
            let rest = off / ir_dim;
            let m = rest % mul;
            let input_flat = rest / mul;
            ((input_flat, m, i), v)
        })
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Basis<I: CouplingIrrep> {
    pub irreps: Irreps<I>,
    pub segments: Vec<Segment>,
}

impl<I: CouplingIrrep> Basis<I> {
    pub fn input_dims(&self) -> &[usize] {
        self.segments
            .first()
            .map(|s| s.input_dims.as_slice())
            .unwrap_or(&[])
    }

    pub fn regroup(&self) -> Basis<I> {
        let canonical = self.irreps.regroup();
        let mut segments = Vec::with_capacity(canonical.entries.len());
        for (mul, ir) in &canonical.entries {
            let parts: Vec<&Segment> = self
                .irreps
                .iter()
                .zip(&self.segments)
                .filter(|((_, ir_), _)| ir_ == ir)
                .map(|(_, seg)| seg)
                .collect();
            let input_dims = parts[0].input_dims.clone();
            let ir_dim = ir.dimension();
            let mut merged = Segment::zeros(input_dims, *mul, ir_dim);
            let mut base = 0;
            for part in parts {
                for ((flat, m, i), v) in part.nonzeros() {
                    merged.set(flat, base + m, i, v.clone());
                }
                base += part.mul;
            }
            segments.push(merged);
        }
        Basis {
            irreps: canonical,
            segments,
        }
    }

    pub fn to_columns(&self) -> crate::rep::Mat<AlgebraicReal> {
        let rows = self.input_dims().iter().product::<usize>();
        let cols = self.irreps.dim();
        let mut mat = crate::rep::Mat::<AlgebraicReal>::zeros(rows, cols);
        let mut col_base = 0;
        for seg in &self.segments {
            for ((flat, m, i), v) in seg.nonzeros() {
                let col = col_base + m * seg.ir_dim + i;
                mat.set(flat, col, v.clone());
            }
            col_base += seg.mul * seg.ir_dim;
        }
        mat
    }
}
