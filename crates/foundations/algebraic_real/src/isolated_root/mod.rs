// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::polynomial::{Interval, Polynomial, RootIsolator};
use crate::selected_root::SelectedRoot;

#[derive(Clone, Debug)]
pub(crate) struct IsolatedRoot {
    selected: Arc<SelectedRoot>,
    isolation: Interval,
    canonical: Arc<OnceLock<Arc<SelectedRoot>>>,
}

impl IsolatedRoot {
    pub(crate) fn new(selected: Arc<SelectedRoot>, canonical: Option<Arc<SelectedRoot>>) -> Self {
        let isolation = selected
            .isolator()
            .isolate_to_bits(selected.root_index(), 64);
        let canonical_cache = Arc::new(OnceLock::new());
        if let Some(canonical) = canonical {
            canonical_cache
                .set(canonical)
                .expect("a new canonical cache is empty");
        }
        Self {
            selected,
            isolation,
            canonical: canonical_cache,
        }
    }

    pub(crate) fn polynomial(&self) -> &Polynomial {
        self.selected.polynomial()
    }

    pub(crate) fn root_index(&self) -> usize {
        self.selected.root_index()
    }

    pub(crate) fn isolator(&self) -> &Arc<RootIsolator> {
        self.selected.isolator()
    }

    pub(crate) fn cached_canonical(&self) -> Option<&Arc<SelectedRoot>> {
        self.canonical.get()
    }

    pub(crate) fn canonical_with(
        &self,
        compute: impl FnOnce() -> Arc<SelectedRoot>,
    ) -> Arc<SelectedRoot> {
        self.canonical.get_or_init(compute).clone()
    }

    pub(crate) fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.selected, &other.selected)
            || self.selected.same_embedding(&other.selected)
    }

    pub(crate) fn selected(&self) -> &Arc<SelectedRoot> {
        &self.selected
    }

    pub(crate) fn interval(&self, bits: usize) -> Interval {
        let target = BigRational::new(BigInt::one(), BigInt::one() << bits);
        if self.isolation.high() - self.isolation.low() <= target {
            self.isolation.clone()
        } else {
            self.selected
                .isolator()
                .isolate_to_bits(self.selected.root_index(), bits)
        }
    }

    pub(crate) fn signum(&self) -> i32 {
        if self.isolation.high() < &BigRational::zero() {
            return -1;
        }
        if self.isolation.low() > &BigRational::zero() {
            return 1;
        }
        for bits in (128..).step_by(64) {
            let interval = self
                .selected
                .isolator()
                .isolate_to_bits(self.selected.root_index(), bits);
            if interval.high() < &BigRational::zero() {
                return -1;
            }
            if interval.low() > &BigRational::zero() {
                return 1;
            }
        }
        unreachable!()
    }
}
