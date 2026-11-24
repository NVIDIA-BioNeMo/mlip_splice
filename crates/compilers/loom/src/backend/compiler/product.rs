// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// A lazy Cartesian product ordered by the sum of per-axis ranks. Every axis's
/// second value precedes any axis's third. It retains axis prefixes, never the
/// product. Zero axes yield one tuple; an empty axis yields none.
#[derive(Clone)]
pub(super) struct Product<I: Iterator + Clone>
where
    I::Item: Clone,
{
    axes: Vec<I>,
    values: Vec<Vec<I::Item>>,
    exhausted: Vec<bool>,
    indices: Vec<usize>,
    degree: u128,
    started: bool,
    done: bool,
}

impl<I: Iterator + Clone> Product<I>
where
    I::Item: Clone,
{
    pub(super) fn new(axes: Vec<I>) -> Self {
        let n = axes.len();
        Self {
            axes,
            values: vec![Vec::new(); n],
            exhausted: vec![false; n],
            indices: vec![0; n],
            degree: 0,
            started: false,
            done: false,
        }
    }

    fn fill_degree(&mut self) -> bool {
        for (axis, values) in self.values.iter_mut().enumerate() {
            if !self.exhausted[axis] && values.len() as u128 <= self.degree {
                match self.axes[axis].next() {
                    Some(value) => values.push(value),
                    None => self.exhausted[axis] = true,
                }
            }
            if values.is_empty() {
                return false;
            }
        }
        self.fill_suffix(0, self.degree)
    }

    fn fill_suffix(&mut self, start: usize, mut sum: u128) -> bool {
        for axis in start..self.indices.len() {
            self.indices[axis] = sum.min((self.values[axis].len() - 1) as u128) as usize;
            sum -= self.indices[axis] as u128;
        }
        sum == 0
    }

    fn advance_in_degree(&mut self) -> bool {
        let mut suffix_sum: u128 = 0;
        let mut suffix_capacity: u128 = 0;
        for axis in (0..self.indices.len()).rev() {
            if self.indices[axis] > 0 && suffix_sum < suffix_capacity {
                self.indices[axis] -= 1;
                return self.fill_suffix(axis + 1, suffix_sum + 1);
            }
            suffix_sum += self.indices[axis] as u128;
            suffix_capacity += (self.values[axis].len() - 1) as u128;
        }
        false
    }
}

impl<I: Iterator + Clone> Iterator for Product<I>
where
    I::Item: Clone,
{
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if !self.started {
            self.started = true;
            self.done = !self.fill_degree();
        } else if !self.advance_in_degree() {
            if let Some(degree) = self.degree.checked_add(1) {
                self.degree = degree;
                self.done = !self.fill_degree();
            } else {
                self.done = true;
            }
        }
        if self.done {
            return None;
        }
        Some(
            self.values
                .iter()
                .zip(&self.indices)
                .map(|(values, &index)| values[index].clone())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Product;

    #[test]
    fn independent_and_conditional_choices() {
        let product = Product::new(vec![0..2, 0..3]);
        assert_eq!(
            product.clone().collect::<Vec<_>>(),
            vec![
                vec![0, 0],
                vec![1, 0],
                vec![0, 1],
                vec![1, 1],
                vec![0, 2],
                vec![1, 2]
            ]
        );
        assert_eq!(std::iter::once(vec![]).chain(product).count(), 7);
        assert_eq!(Product::<std::ops::Range<usize>>::new(vec![]).count(), 1);
        assert_eq!(Product::new(vec![0..0, 0..3]).count(), 0);
    }

    #[test]
    fn huge_product_is_lazy_fair_and_restartable() {
        let mut product = Product::new(vec![0..usize::MAX; 20]);
        assert_eq!(product.next(), Some(vec![0; 20]));
        let mut continuation = product.clone();
        let next = product.by_ref().take(20).collect::<Vec<_>>();
        for axis in 0..20 {
            assert!(
                next.iter()
                    .any(|tuple| tuple[axis] == 1 && tuple.iter().sum::<usize>() == 1)
            );
        }
        assert_eq!(next, continuation.by_ref().take(20).collect::<Vec<_>>());
    }

    #[test]
    fn bounded_degrees_visit_every_tuple_once() {
        for a in 0..5 {
            for b in 0..5 {
                for c in 0..5 {
                    let tuples = Product::new(vec![0..a, 0..b, 0..c]).collect::<Vec<_>>();
                    assert_eq!(tuples.len(), a * b * c);
                    assert!(
                        tuples
                            .windows(2)
                            .all(|pair| pair[0].iter().sum::<usize>()
                                <= pair[1].iter().sum::<usize>())
                    );
                    assert_eq!(
                        tuples
                            .iter()
                            .collect::<std::collections::HashSet<_>>()
                            .len(),
                        tuples.len()
                    );
                }
            }
        }
    }
}
