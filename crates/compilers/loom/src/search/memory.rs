// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// A weighted search-volume limit, deliberately not a live-allocation census.
/// PTX carries a factor of two for compiler overhead. Each complete candidate
/// adds 16 KiB for metadata, including candidates reusing cached source.
/// Repeated source-unit records are charged again; no sharing analysis is needed.
pub(super) struct EstimatedMemoryBudget {
    limit: Option<u64>,
    pub(super) estimated_bytes: u64,
}

impl EstimatedMemoryBudget {
    pub(super) fn new(limit: Option<u64>) -> Self {
        Self {
            limit,
            estimated_bytes: 0,
        }
    }

    pub(super) fn charge(&mut self, ptx_bytes: u64, candidates: u64) {
        self.estimated_bytes = self
            .estimated_bytes
            .saturating_add(ptx_bytes.saturating_mul(2))
            .saturating_add(candidates.saturating_mul(16 * 1024));
    }

    pub(super) fn exhausted(&self) -> bool {
        self.limit
            .is_some_and(|limit| self.estimated_bytes >= limit)
    }

    pub(super) fn checkpoint(&self) -> Result<(), String> {
        if self.exhausted() {
            Err(format!(
                "loom search memory budget exhausted: estimated search volume {} bytes, \
                 budget {} bytes (2 * PTX bytes + 16 KiB per candidate; not process RSS)",
                self.estimated_bytes,
                self.limit.expect("bounded budget")
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_sources_spend_more_than_smaller_sources() {
        let mut small = EstimatedMemoryBudget::new(Some(64 * 1024));
        let mut large = EstimatedMemoryBudget::new(Some(64 * 1024));
        small.charge(1024, 1);
        large.charge(32 * 1024, 1);
        assert_eq!(small.estimated_bytes, 18 * 1024);
        assert!(!small.exhausted());
        assert!(large.exhausted());
    }

    #[test]
    fn cached_source_candidates_and_refused_source_attempts_still_cost() {
        let mut budget = EstimatedMemoryBudget::new(Some(18 * 1024));
        budget.charge(1024, 0);
        assert_eq!(budget.estimated_bytes, 2 * 1024);
        budget.charge(0, 1);
        assert_eq!(budget.estimated_bytes, 18 * 1024);
        assert!(budget.checkpoint().is_err());
    }

    #[test]
    fn charges_are_additive_without_inspecting_process_memory() {
        let mut budget = EstimatedMemoryBudget::new(None);
        budget.charge(100, 1);
        budget.charge(200, 2);
        assert_eq!(budget.estimated_bytes, 600 + 3 * 16 * 1024);
        assert!(!budget.exhausted());
    }

    #[test]
    fn saturation_does_not_reopen_the_budget() {
        let mut budget = EstimatedMemoryBudget::new(Some(u64::MAX));
        budget.charge(u64::MAX, u64::MAX);
        assert_eq!(budget.estimated_bytes, u64::MAX);
        assert!(budget.exhausted());
        budget.charge(1, 1);
        assert!(budget.exhausted());
    }

    #[test]
    fn zero_budget_stops_before_work() {
        assert!(EstimatedMemoryBudget::new(Some(0)).checkpoint().is_err());
    }
}
