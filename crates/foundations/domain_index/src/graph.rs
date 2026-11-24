// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::bridge::AxisBridge;
use std::collections::BTreeSet;

/// An iteration set of `extent` elements.
#[derive(Copy, Clone, Debug)]
pub struct Domain {
    pub extent: i64,
}

/// A many-to-one arrow from a source domain to a target one: one target index
/// per source element.
#[derive(Copy, Clone, Debug)]
pub struct IndexMap {
    pub source_id: i32,
    pub target_id: i32,
    /// Distinct mapped source elements have distinct targets; unmapped
    /// elements do not participate in this contract.
    pub is_injective: bool,
    /// Valid `target_indices` are non-decreasing and every `-1` follows them,
    /// so `preimage_source_indices[i] == i`; preimage walks skip that load.
    pub is_sorted: bool,
    /// `target_indices` may hold `-1` (a source element with no target);
    /// lookups and scatters through this arrow are `>= 0`-guarded.
    pub may_have_unmapped: bool,
}

/// Reach the `other` target of `source` through its `anchor` target.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AlignedView {
    pub source_id: i32,
    pub anchor_target_id: i32,
    pub other_target_id: i32,
}

/// Domains and the index-map / aligned-view arrows between them, with the
/// queries that walk that topology.
#[derive(Clone, Debug)]
pub struct Graph {
    pub domains: Vec<Domain>,
    pub index_maps: Vec<IndexMap>,
    pub aligned_views: Vec<AlignedView>,
}

impl Graph {
    pub fn finite_map_graph(&self) -> Result<crate::FiniteMapGraph, String> {
        let mut graph = crate::FiniteMapGraph::default();
        for domain in &self.domains {
            graph.add_set(domain.extent)?;
        }
        for index_map in &self.index_maps {
            let source = crate::FiniteSetId(
                u32::try_from(index_map.source_id)
                    .map_err(|_| format!("negative source domain {}", index_map.source_id))?,
            );
            let target = crate::FiniteSetId(
                u32::try_from(index_map.target_id)
                    .map_err(|_| format!("negative target domain {}", index_map.target_id))?,
            );
            graph.add_map(source, target)?;
        }
        Ok(graph)
    }

    pub fn domain_extent(&self, id: i32) -> i64 {
        self.domains[id as usize].extent
    }

    pub fn common_source(&self, domains: impl IntoIterator<Item = i32>) -> Result<i32, String> {
        let domains = domains.into_iter().collect::<BTreeSet<_>>();
        if domains.is_empty() {
            return Err("cannot find a common source domain of an empty set".into());
        }
        for &domain in &domains {
            if domain < 0 || domain as usize >= self.domains.len() {
                return Err(format!("domain {domain} is out of range"));
            }
        }
        let reaches = |source, target| self.index_map_reaches(source, target);
        let candidates = (0..self.domains.len() as i32)
            .filter(|&candidate| domains.iter().all(|&domain| reaches(candidate, domain)))
            .collect::<BTreeSet<_>>();
        let present = candidates
            .intersection(&domains)
            .copied()
            .collect::<BTreeSet<_>>();
        let pool = if present.is_empty() {
            &candidates
        } else {
            &present
        };
        let finest = pool
            .iter()
            .copied()
            .filter(|&candidate| {
                !pool
                    .iter()
                    .any(|&other| other != candidate && reaches(other, candidate))
            })
            .collect::<Vec<_>>();
        match finest.as_slice() {
            [domain] => Ok(*domain),
            [] => Err(format!("no common source domain reaches {domains:?}")),
            _ => Err(format!(
                "common source domain of {domains:?} is ambiguous: {finest:?}"
            )),
        }
    }

    fn index_map_reaches(&self, source: i32, target: i32) -> bool {
        let mut pending = vec![source];
        let mut visited = BTreeSet::new();
        while let Some(domain) = pending.pop() {
            if domain == target {
                return true;
            }
            if !visited.insert(domain) {
                continue;
            }
            pending.extend(
                self.index_maps
                    .iter()
                    .filter(|index_map| index_map.source_id == domain)
                    .map(|index_map| index_map.target_id),
            );
        }
        false
    }

    pub fn find_index_map(&self, source: i32, target: i32) -> Option<usize> {
        self.index_maps
            .iter()
            .position(|r| r.source_id == source && r.target_id == target)
    }

    pub fn find_aligned_view(&self, anchor: i32, other: i32) -> Option<usize> {
        self.aligned_views
            .iter()
            .position(|v| v.anchor_target_id == anchor && v.other_target_id == other)
    }

    /// Classify the single-step edge from `iter_dom` to `dom`, or `None` if none
    /// connects them. `Broadcast` is not topology and lives in
    /// [`crate::DomainIndex::axis_bridge`].
    pub fn bridge(&self, iter_dom: i32, dom: i32) -> Option<AxisBridge> {
        if dom == iter_dom {
            return Some(AxisBridge::Direct);
        }
        if let Some(map_idx) = self.find_index_map(iter_dom, dom) {
            return Some(AxisBridge::MapLookup { map_idx });
        }
        if let Some(map_idx) = self.find_index_map(dom, iter_dom) {
            return Some(AxisBridge::PreimageWalk { map_idx });
        }
        if let Some(view_idx) = self.find_aligned_view(iter_dom, dom) {
            return Some(AxisBridge::AlignedLookup { view_idx });
        }
        None
    }

    /// Index of the index map an aligned view walks along its anchor.
    pub fn aligned_view_anchor_map(&self, view_idx: usize) -> usize {
        let v = &self.aligned_views[view_idx];
        self.find_index_map(v.source_id, v.anchor_target_id)
            .expect("validated: view's anchor arrow exists")
    }

    /// The view's *other* leg can carry `-1`, so a gather through it needs a
    /// `>= 0` guard.
    pub fn aligned_view_other_may_have_unmapped(&self, view_idx: usize) -> bool {
        let v = &self.aligned_views[view_idx];
        self.index_maps.iter().any(|r| {
            r.source_id == v.source_id && r.target_id == v.other_target_id && r.may_have_unmapped
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain() -> Domain {
        Domain { extent: 1 }
    }

    fn index_map(source_id: i32, target_id: i32) -> IndexMap {
        IndexMap {
            source_id,
            target_id,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }
    }

    #[test]
    fn common_source_follows_composed_maps() {
        let graph = Graph {
            domains: vec![domain(), domain(), domain()],
            index_maps: vec![index_map(2, 1), index_map(1, 0)],
            aligned_views: Vec::new(),
        };
        assert_eq!(graph.common_source([0, 1, 2]).unwrap(), 2);
        assert_eq!(graph.common_source([0, 1]).unwrap(), 1);
        assert_eq!(graph.common_source([0]).unwrap(), 0);
    }
}
