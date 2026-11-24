// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::{Add, Mul, Neg, Sub};

use super::Ring;

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

/// A row-major sparse tensor over a `Ring`: it stores only its nonzero entries,
/// and accumulation prunes exact cancellations to zero.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SparseTensor<K> {
    shape: Vec<usize>,
    entries: BTreeMap<usize, K>,
}

impl<K> SparseTensor<K> {
    pub fn from_flat(shape: Vec<usize>, entries: BTreeMap<usize, K>) -> Self {
        SparseTensor { shape, entries }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn nonzeros(&self) -> impl Iterator<Item = (Vec<usize>, &K)> {
        let shape = &self.shape;
        self.entries
            .iter()
            .map(move |(&off, v)| (unravel(off, shape), v))
    }
}

impl<K: Ring> SparseTensor<K>
where
    for<'a> &'a K:
        Add<&'a K, Output = K> + Sub<&'a K, Output = K> + Mul<&'a K, Output = K> + Neg<Output = K>,
{
    pub fn zeros(shape: Vec<usize>) -> Self {
        SparseTensor {
            shape,
            entries: BTreeMap::new(),
        }
    }

    pub fn from_entries(
        shape: Vec<usize>,
        entries: impl IntoIterator<Item = (Vec<usize>, K)>,
    ) -> Self {
        let mut tensor = SparseTensor::zeros(shape);
        for (coords, value) in entries {
            tensor.add(&coords, value);
        }
        tensor
    }

    fn offset(&self, coords: &[usize]) -> usize {
        assert_eq!(coords.len(), self.shape.len(), "coords rank mismatch");
        for (&c, &d) in coords.iter().zip(&self.shape) {
            assert!(c < d, "coordinate {c} out of bounds for axis of size {d}");
        }
        ravel(coords, &self.shape)
    }

    pub fn get(&self, coords: &[usize]) -> K {
        self.entries
            .get(&self.offset(coords))
            .cloned()
            .unwrap_or_else(K::zero)
    }

    pub fn set(&mut self, coords: &[usize], value: K) {
        let off = self.offset(coords);
        if value.is_zero() {
            self.entries.remove(&off);
        } else {
            self.entries.insert(off, value);
        }
    }

    pub fn add(&mut self, coords: &[usize], value: K) {
        let off = self.offset(coords);
        match self.entries.entry(off) {
            Entry::Vacant(e) => {
                if !value.is_zero() {
                    e.insert(value);
                }
            }
            Entry::Occupied(mut e) => {
                let sum = e.get() + &value;
                if sum.is_zero() {
                    e.remove();
                } else {
                    *e.get_mut() = sum;
                }
            }
        }
    }

    pub fn scale(&self, s: &K) -> Self {
        let entries = if s.is_zero() {
            BTreeMap::new()
        } else {
            self.entries.iter().map(|(&off, v)| (off, v * s)).collect()
        };
        SparseTensor {
            shape: self.shape.clone(),
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebraic_real::AlgebraicReal;

    fn s(n: i64) -> AlgebraicReal {
        AlgebraicReal::integer(n)
    }

    #[test]
    fn set_get_and_prune() {
        let mut t = SparseTensor::<AlgebraicReal>::zeros(vec![2, 3]);
        t.set(&[1, 2], s(5));
        assert_eq!(t.get(&[1, 2]), s(5));
        assert_eq!(t.get(&[0, 0]), s(0));
        assert_eq!(t.nnz(), 1);
        t.set(&[1, 2], s(0));
        assert_eq!(t.nnz(), 0);
    }

    #[test]
    fn add_accumulates_and_cancels() {
        let mut t = SparseTensor::<AlgebraicReal>::zeros(vec![2]);
        t.add(&[0], s(3));
        t.add(&[0], s(-3));
        assert_eq!(t.nnz(), 0, "exact cancellation prunes the entry");
    }

    #[test]
    fn nonzeros_round_trip_coords() {
        let t = SparseTensor::from_entries(vec![2, 2, 3], [(vec![1, 0, 2], s(7))]);
        let got: Vec<_> = t.nonzeros().map(|(c, v)| (c, v.clone())).collect();
        assert_eq!(got, vec![(vec![1, 0, 2], s(7))]);
    }

    #[test]
    fn from_flat_matches_row_major() {
        // Offset (1·2 + 0)·3 + 2 = 8 in shape [2,2,3] is coords [1,0,2].
        let mut entries = BTreeMap::new();
        entries.insert(8usize, s(7));
        let t = SparseTensor::from_flat(vec![2, 2, 3], entries);
        assert_eq!(t.get(&[1, 0, 2]), s(7));
    }

    #[test]
    fn scale_is_exact() {
        let t = SparseTensor::from_entries(vec![2], [(vec![0], s(2)), (vec![1], s(3))]);
        let scaled = t.scale(&AlgebraicReal::sqrt(2u32));
        assert_eq!(scaled.get(&[0]), &s(2) * &AlgebraicReal::sqrt(2u32));
        assert_eq!(scaled.scale(&s(0)).nnz(), 0);
    }
}
