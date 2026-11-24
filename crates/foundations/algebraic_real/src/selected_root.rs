// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::polynomial::{Polynomial, RootIsolator};

/// A real embedding selected by an integer polynomial and root index.
///
/// `IsolatedRoot` caches this shared descriptor, while `NumberFieldElement`
/// uses it as the generator that gives its coefficient vector meaning.
#[derive(Debug)]
pub(crate) struct SelectedRoot {
    root_index: usize,
    isolator: Arc<RootIsolator>,
}

impl SelectedRoot {
    #[cfg(test)]
    pub(crate) fn new(polynomial: Polynomial, root_index: usize) -> Arc<Self> {
        Self::from_isolator(root_index, Arc::new(RootIsolator::new(&polynomial)))
    }

    pub(crate) fn from_isolator(root_index: usize, isolator: Arc<RootIsolator>) -> Arc<Self> {
        assert!(root_index < isolator.root_count(), "root index is in range");
        Arc::new(Self {
            root_index,
            isolator,
        })
    }

    pub(crate) fn same_embedding(&self, other: &Self) -> bool {
        self.polynomial() == other.polynomial() && self.root_index == other.root_index
    }

    pub(crate) fn polynomial(&self) -> &Polynomial {
        self.isolator.polynomial()
    }

    pub(crate) fn root_index(&self) -> usize {
        self.root_index
    }

    pub(crate) fn isolator(&self) -> &Arc<RootIsolator> {
        &self.isolator
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::*;

    #[test]
    #[should_panic(expected = "root index is in range")]
    fn selected_root_rejects_out_of_range_index() {
        let polynomial = vec![BigInt::from(-2), BigInt::from(0), BigInt::from(1)];
        let isolator = Arc::new(RootIsolator::new(&polynomial));
        SelectedRoot::from_isolator(2, isolator);
    }
}
