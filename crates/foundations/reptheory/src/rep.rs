// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::{Add, Mul, Neg, Sub};

use crate::exact::{ComplexAlgebraicReal, Conjugate, Ring};
use algebraic_real::AlgebraicReal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mat<K> {
    pub rows: usize,
    pub cols: usize,
    /// `entries[i*cols + j]` is `(i, j)`.
    pub entries: Vec<K>,
}

impl<K> Mat<K> {
    pub fn get(&self, i: usize, j: usize) -> &K {
        &self.entries[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: K) {
        self.entries[i * self.cols + j] = value;
    }
}

impl<K: Ring> Mat<K>
where
    for<'a> &'a K:
        Add<&'a K, Output = K> + Sub<&'a K, Output = K> + Mul<&'a K, Output = K> + Neg<Output = K>,
{
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            entries: (0..rows * cols).map(|_| K::zero()).collect(),
        }
    }
    pub fn mul(&self, rhs: &Mat<K>) -> Mat<K> {
        assert_eq!(self.cols, rhs.rows, "matmul shape mismatch");
        let mut out = Mat::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a.is_zero() {
                    continue;
                }
                for j in 0..rhs.cols {
                    let term = a * rhs.get(k, j);
                    let acc = out.get(i, j) + &term;
                    out.set(i, j, acc);
                }
            }
        }
        out
    }
}

impl<K: Ring + Conjugate> Mat<K>
where
    for<'a> &'a K:
        Add<&'a K, Output = K> + Sub<&'a K, Output = K> + Mul<&'a K, Output = K> + Neg<Output = K>,
{
    pub fn conjugate(&self) -> Mat<K> {
        Mat {
            rows: self.rows,
            cols: self.cols,
            entries: self.entries.iter().map(Conjugate::conj).collect(),
        }
    }

    pub fn conjugate_transpose(&self) -> Mat<K> {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j).conj());
            }
        }
        out
    }
}

impl Mat<ComplexAlgebraicReal> {}

pub trait Rep {
    fn dim(&self) -> usize;

    fn generators(&self) -> Vec<Mat<ComplexAlgebraicReal>>;
}

/// The data needed to couple irreducible representations, without requiring a
/// generator realization, parser, or highest-weight product.
pub trait CouplingIrrep: Ord + Clone + Send + Sync {
    fn dimension(&self) -> usize;

    fn trivial(&self) -> Self;

    fn tensor_product(&self, other: &Self) -> Vec<Self>;

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>>;
}

impl<I: Irrep> CouplingIrrep for I {
    fn dimension(&self) -> usize {
        Rep::dim(self)
    }

    fn trivial(&self) -> Self {
        Irrep::trivial(self)
    }

    fn tensor_product(&self, other: &Self) -> Vec<Self> {
        Irrep::tensor_product(self, other)
    }

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>> {
        Irrep::clebsch_gordan(a, b, c)
    }
}

pub trait Irrep: Rep + Ord + Clone + Send + Sync {
    fn trivial(&self) -> Self;

    fn tensor_product(&self, other: &Self) -> Vec<Self>;

    fn clebsch_gordan(a: &Self, b: &Self, c: &Self) -> Vec<Mat<AlgebraicReal>>;

    fn cartan_product(a: &Self, b: &Self) -> Self;

    fn parse(s: &str) -> Option<Self>;
}
