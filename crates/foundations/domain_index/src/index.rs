// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use crate::bridge::AxisBridge;
use crate::graph::Graph;

/// A placement of domains onto iteration and operand axes over an index map
/// [`Graph`]. Implement the accessors; every other query is derived from them.
pub trait DomainIndex {
    fn graph(&self) -> &Graph;
    fn num_axes(&self) -> usize;
    /// Domain the iteration space walks on `axis`.
    fn iter_domain(&self, axis: usize) -> i32;
    fn num_operands(&self) -> usize;
    /// Domain operand `b` is placed on for `axis`, or `-1` if it broadcasts there.
    fn operand_domain(&self, b: usize, axis: usize) -> i32;

    /// Allocates; use `iter_domain` for a single axis.
    fn iter_domain_per_axis(&self) -> Vec<i32> {
        (0..self.num_axes()).map(|a| self.iter_domain(a)).collect()
    }

    fn operand_carries(&self, b: usize, axis: usize) -> bool {
        self.operand_domain(b, axis) >= 0
    }

    fn operand_batch_extent(&self, b: usize) -> Result<i64, String> {
        let mut extent: i64 = 1;
        for axis in 0..self.num_axes() {
            let dom = self.operand_domain(b, axis);
            if dom >= 0 {
                extent = extent
                    .checked_mul(self.graph().domain_extent(dom))
                    .ok_or_else(|| format!("operand {b} extent overflow"))?;
            }
        }
        Ok(extent)
    }

    /// How far one step along `axis` moves in operand `b`: the extent of every
    /// axis inside it, multiplied out.
    ///
    /// An operand that does not carry the axis does not move along it, so its
    /// stride is `0` — a statement about the addressing, not a count. Every
    /// other answer is a real product, refused rather than wrapped: a stride
    /// that wrapped would address a different element of the same buffer, which
    /// no bounds check downstream can catch.
    fn operand_axis_stride(&self, b: usize, axis: usize) -> Result<i64, String> {
        if !self.operand_carries(b, axis) {
            return Ok(0);
        }
        let inner: Vec<i64> = ((axis + 1)..self.num_axes())
            .filter_map(|ax| {
                let dom = self.operand_domain(b, ax);
                (dom >= 0).then(|| self.graph().domain_extent(dom))
            })
            .collect();
        crate::extent::product(
            &format!("the stride of operand {b} along axis {axis}"),
            &inner,
        )
    }

    fn batch_sizes(&self) -> Vec<i64> {
        (0..self.num_axes())
            .map(|a| self.graph().domain_extent(self.iter_domain(a)))
            .collect()
    }

    /// The iteration elements a launch walks: the product of the batch extents.
    ///
    /// No axes at all is one element — the empty product — and a zero extent is
    /// none of them. Both are real answers, and they are different: a grid sized
    /// from the wrong one launches over a buffer that was never allocated.
    fn batch_total(&self) -> Result<i64, String> {
        crate::extent::product("the iteration space", &self.batch_sizes())
    }

    fn axis_bridge(&self, operand: usize, axis: usize) -> AxisBridge {
        let dom = self.operand_domain(operand, axis);
        if dom < 0 {
            return AxisBridge::Broadcast;
        }
        let iter_dom = self.iter_domain(axis);
        self.graph().bridge(iter_dom, dom).unwrap_or_else(|| {
            panic!(
                "invalid axis bridge for operand {operand} axis {axis}: \
                 no index map / aligned view connects iter={iter_dom} ↔ operand={dom}"
            )
        })
    }

    fn needs_atomic(&self, operand: usize) -> bool {
        (0..self.num_axes()).any(|axis| match self.axis_bridge(operand, axis) {
            AxisBridge::Broadcast => true,
            AxisBridge::MapLookup { map_idx } => !self.graph().index_maps[map_idx].is_injective,
            _ => false,
        })
    }

    /// Whether every address of `operand` receives the same number of values,
    /// so that number follows from the extents alone.
    ///
    /// A broadcast axis sends every one of its positions to every address, which
    /// is arithmetic. A non-injective index map instead decides which address a
    /// position reaches, and its target indices are runtime data: the extents
    /// then give the mean over addresses, while one address may receive as many
    /// values as the whole iteration space has positions.
    fn collisions_are_uniform(&self, operand: usize) -> bool {
        (0..self.num_axes()).all(|axis| match self.axis_bridge(operand, axis) {
            AxisBridge::Direct | AxisBridge::Broadcast => true,
            AxisBridge::MapLookup { map_idx } => self.graph().index_maps[map_idx].is_injective,
            AxisBridge::PreimageWalk { .. } | AxisBridge::AlignedLookup { .. } => false,
        })
    }

    /// The operand sits on the fiber's source domain: reads gather through the
    /// fiber walk, and writes are exact — each source element belongs to exactly
    /// one fiber, so the walk covers it exactly once.
    fn fiber_scatter(&self, operand: usize) -> bool {
        (0..self.num_axes()).any(|axis| {
            matches!(
                self.axis_bridge(operand, axis),
                AxisBridge::PreimageWalk { .. }
            )
        })
    }

    fn used_index_maps(&self) -> Vec<usize> {
        let mut used = BTreeSet::new();
        for b in 0..self.num_operands() {
            for axis in 0..self.num_axes() {
                match self.axis_bridge(b, axis) {
                    AxisBridge::MapLookup { map_idx } | AxisBridge::PreimageWalk { map_idx } => {
                        used.insert(map_idx);
                    }
                    AxisBridge::AlignedLookup { view_idx } => {
                        used.insert(self.graph().aligned_view_anchor_map(view_idx));
                    }
                    _ => {}
                }
            }
        }
        used.into_iter().collect()
    }

    fn used_aligned_views(&self) -> Vec<usize> {
        let mut used = BTreeSet::new();
        for b in 0..self.num_operands() {
            for axis in 0..self.num_axes() {
                if let AxisBridge::AlignedLookup { view_idx } = self.axis_bridge(b, axis) {
                    used.insert(view_idx);
                }
            }
        }
        used.into_iter().collect()
    }

    fn fiber_bridge(&self) -> Option<(usize, usize)> {
        let mut found: Option<(usize, usize)> = None;
        for b in 0..self.num_operands() {
            for axis in 0..self.num_axes() {
                let key = match self.axis_bridge(b, axis) {
                    AxisBridge::PreimageWalk { map_idx } => Some((map_idx, axis)),
                    AxisBridge::AlignedLookup { view_idx } => {
                        Some((self.graph().aligned_view_anchor_map(view_idx), axis))
                    }
                    _ => None,
                };
                if let Some(key) = key {
                    match found {
                        None => found = Some(key),
                        Some(k) if k == key => {}
                        Some(_) => {
                            panic!("multiple preimage maps not yet supported (validated up front)")
                        }
                    }
                }
            }
        }
        found
    }

    fn validate(&self, is_output: &dyn Fn(usize) -> bool) -> Result<(), String> {
        let g = self.graph();
        for axis in 0..self.num_axes() {
            let dom = self.iter_domain(axis);
            if dom < 0 || (dom as usize) >= g.domains.len() {
                return Err(format!(
                    "iter_domain_per_axis[{axis}] = {dom} out of range (num_domains={})",
                    g.domains.len()
                ));
            }
        }
        let mut fiber_seen: Option<(usize, usize)> = None;
        for b in 0..self.num_operands() {
            for axis in 0..self.num_axes() {
                let dom = self.operand_domain(b, axis);
                if dom < -1 || (dom >= 0 && (dom as usize) >= g.domains.len()) {
                    return Err(format!(
                        "operand {b} axis {axis}: domain id {dom} out of range"
                    ));
                }
                let iter_dom = self.iter_domain(axis);
                if dom < 0 || dom == iter_dom {
                    continue;
                }
                let to_coarser = g.find_index_map(iter_dom, dom);
                let to_finer = g.find_index_map(dom, iter_dom);
                let via_view = g.find_aligned_view(iter_dom, dom);
                match (to_coarser, to_finer, via_view) {
                    (None, None, None) => {
                        return Err(format!(
                            "operand {b} axis {axis}: no single-step index map or \
                             aligned view connects iter={iter_dom} ↔ operand={dom}"
                        ));
                    }
                    (Some(_), Some(_), _) => {
                        return Err(format!(
                            "operand {b} axis {axis}: ambiguous — index-map edges \
                             exist in both directions between iter={iter_dom} and operand={dom}"
                        ));
                    }
                    (Some(_), None, _) => {}
                    (None, Some(fiber_ref), _) => {
                        // An output on the fiber's source domain is written
                        // fiber-direct: exact only when the map is
                        // total, so the walk reaches every source element.
                        if is_output(b) && g.index_maps[fiber_ref].may_have_unmapped {
                            return Err(format!(
                                "operand {b}: fiber-scatter output over an index map \
                                 with unmapped elements would leave them unwritten"
                            ));
                        }
                        let key = (fiber_ref, axis);
                        match fiber_seen {
                            None => fiber_seen = Some(key),
                            Some(k) if k == key => {}
                            Some(_) => {
                                return Err(format!(
                                    "operand {b}: multi-axis or multi-map fiber gather \
                                     not yet supported"
                                ));
                            }
                        }
                    }
                    (None, None, Some(view_idx)) => {
                        if is_output(b) {
                            return Err(format!(
                                "operand {b}: outputs cannot bridge through an aligned view"
                            ));
                        }
                        let v = &g.aligned_views[view_idx];
                        let Some(ri) = g.find_index_map(v.source_id, v.anchor_target_id) else {
                            return Err(format!(
                                "operand {b} axis {axis}: aligned view \
                                 ({},{},{}) needs an index map ({},{}) for its anchor walk",
                                v.source_id,
                                v.anchor_target_id,
                                v.other_target_id,
                                v.source_id,
                                v.anchor_target_id,
                            ));
                        };
                        let key = (ri, axis);
                        match fiber_seen {
                            None => fiber_seen = Some(key),
                            Some(k) if k == key => {}
                            Some(_) => {
                                return Err(format!(
                                    "operand {b}: multi-axis or multi-map fiber gather \
                                     not yet supported"
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{AlignedView, AxisBridge, Domain, DomainIndex, Graph, IndexMap};

    struct Index {
        graph: Graph,
        iter_domain_per_axis: Vec<i32>,
        operand_domains: Vec<Vec<i32>>,
    }

    impl DomainIndex for Index {
        fn graph(&self) -> &Graph {
            &self.graph
        }
        fn num_axes(&self) -> usize {
            self.iter_domain_per_axis.len()
        }
        fn iter_domain(&self, axis: usize) -> i32 {
            self.iter_domain_per_axis[axis]
        }
        fn num_operands(&self) -> usize {
            self.operand_domains.len()
        }
        fn operand_domain(&self, b: usize, axis: usize) -> i32 {
            self.operand_domains[b][axis]
        }
    }

    fn refn(source: i32, target: i32, unique: bool, sorted: bool, unmapped: bool) -> IndexMap {
        IndexMap {
            source_id: source,
            target_id: target,
            is_injective: unique,
            is_sorted: sorted,
            may_have_unmapped: unmapped,
        }
    }

    fn kitchen_sink() -> Index {
        Index {
            graph: Graph {
                domains: vec![
                    Domain { extent: 12 },
                    Domain { extent: 4 },
                    Domain { extent: 3 },
                    Domain { extent: 2 },
                    Domain { extent: 8 },
                    Domain { extent: 2 },
                ],
                index_maps: vec![
                    refn(0, 1, false, false, false),
                    refn(0, 2, false, false, true),
                    refn(1, 3, false, false, true),
                    refn(4, 5, true, true, false),
                ],
                aligned_views: vec![AlignedView {
                    source_id: 0,
                    anchor_target_id: 1,
                    other_target_id: 2,
                }],
            },
            iter_domain_per_axis: vec![1, 4],
            operand_domains: vec![
                vec![1, 4],
                vec![0, 4],
                vec![2, 4],
                vec![3, 5],
                vec![-1, -1],
                vec![3, 4],
                vec![-1, -1],
                vec![1, 4],
            ],
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn kitchen_sink_classifies_every_bridge_and_flag() {
        use AxisBridge::*;
        let idx = kitchen_sink();
        idx.validate(&|b| b >= 5).expect("DAG is valid");

        assert!(matches!(idx.axis_bridge(0, 0), Direct));
        assert!(matches!(idx.axis_bridge(1, 0), PreimageWalk { map_idx: 0 }));
        assert!(matches!(
            idx.axis_bridge(2, 0),
            AlignedLookup { view_idx: 0 }
        ));
        assert!(matches!(idx.axis_bridge(3, 0), MapLookup { map_idx: 2 }));
        assert!(matches!(idx.axis_bridge(4, 0), Broadcast));
        assert!(matches!(idx.axis_bridge(5, 0), MapLookup { map_idx: 2 }));
        assert!(matches!(idx.axis_bridge(7, 0), Direct));
        assert!(matches!(idx.axis_bridge(0, 1), Direct));
        assert!(matches!(idx.axis_bridge(3, 1), MapLookup { map_idx: 3 }));
        assert!(matches!(idx.axis_bridge(4, 1), Broadcast));

        assert_eq!(idx.fiber_bridge(), Some((0, 0)));
        assert_eq!(idx.used_index_maps(), vec![0, 2, 3]);
        assert_eq!(idx.used_aligned_views(), vec![0]);

        assert!(idx.needs_atomic(5));
        assert!(idx.needs_atomic(6));
        assert!(!idx.needs_atomic(7));

        // Operand 6 broadcasts both axes, so every position reaches every one of
        // its addresses: the extents count the values an address receives.
        // Operand 5 reaches its addresses through a non-injective map whose
        // targets are runtime data, so the extents give only a mean.
        assert!(idx.collisions_are_uniform(6));
        assert!(!idx.collisions_are_uniform(5));
        assert!(!idx.collisions_are_uniform(1), "a preimage walk");
        assert!(idx.collisions_are_uniform(7), "direct on both axes");

        assert_eq!(idx.num_axes(), 2);
        assert_eq!(idx.batch_sizes(), vec![4, 8]);
    }

    #[test]
    fn fiber_scatter_output_requires_a_total_map() {
        // IndexMap 0 (domain 0 -> 1) is total: an output on the source
        // domain is written fiber-direct, exactly once, without atomics.
        let mut idx = kitchen_sink();
        idx.operand_domains[7] = vec![0, 4];
        idx.validate(&|b| b >= 5).expect("total map scatters");
        assert!(idx.fiber_scatter(7));
        assert!(!idx.needs_atomic(7));

        // IndexMap 1 (domain 0 -> 2) has unmapped elements: the walk would
        // never reach them, so the placement is rejected.
        let mut idx = kitchen_sink();
        idx.iter_domain_per_axis = vec![2, 4];
        idx.operand_domains = vec![vec![0, 4], vec![2, 4], vec![0, 4]];
        assert!(idx.validate(&|b| b >= 2).is_err());
    }

    #[test]
    fn rejects_output_through_an_aligned_view() {
        let mut idx = kitchen_sink();
        idx.operand_domains[7] = vec![2, 4];
        assert!(idx.validate(&|b| b >= 5).is_err());
    }

    #[test]
    fn zero_axes_is_one_batch_cell() {
        let idx = Index {
            graph: Graph {
                domains: Vec::new(),
                index_maps: Vec::new(),
                aligned_views: Vec::new(),
            },
            iter_domain_per_axis: Vec::new(),
            operand_domains: vec![Vec::new(), Vec::new()],
        };

        idx.validate(&|b| b == 1).unwrap();
        assert!(idx.batch_sizes().is_empty());
        assert_eq!(idx.batch_total().unwrap(), 1);
        assert_eq!(idx.operand_batch_extent(0).unwrap(), 1);
        assert_eq!(idx.operand_batch_extent(1).unwrap(), 1);
        assert!(!idx.needs_atomic(1));
    }

    /// A placement over two axes whose extents are each half the counting
    /// range. Its stride and its iteration space are not numbers, and a wrapped
    /// one would address a different element of the same buffer — which no
    /// bounds check downstream can catch, because the address is in range.
    fn uncountable() -> Index {
        Index {
            graph: Graph {
                domains: vec![
                    Domain {
                        extent: i64::MAX / 2,
                    },
                    Domain {
                        extent: i64::MAX / 2,
                    },
                ],
                index_maps: Vec::new(),
                aligned_views: Vec::new(),
            },
            iter_domain_per_axis: vec![0, 1],
            operand_domains: vec![vec![0, 1]],
        }
    }

    #[test]
    fn an_iteration_space_past_the_countable_is_refused() {
        let error = uncountable().batch_total().unwrap_err();
        assert!(error.contains("the iteration space"), "{error}");
        assert!(error.contains("past what an i64"), "{error}");
    }

    #[test]
    fn a_stride_past_the_countable_is_refused() {
        // Two axes inside axis 0 would multiply past the range.
        let mut idx = uncountable();
        idx.iter_domain_per_axis = vec![0, 0, 1];
        idx.operand_domains = vec![vec![0, 0, 1]];
        let error = idx.operand_axis_stride(0, 0).unwrap_err();
        assert!(
            error.contains("the stride of operand 0 along axis 0"),
            "{error}"
        );
    }

    /// The strides a real placement has: the extents inside the axis, and `0`
    /// for an axis the operand does not carry — a statement about the
    /// addressing, not a count that could be confused with an empty one.
    #[test]
    fn a_stride_is_the_extents_inside_the_axis() {
        let idx = Index {
            graph: Graph {
                domains: vec![
                    Domain { extent: 2 },
                    Domain { extent: 3 },
                    Domain { extent: 5 },
                ],
                index_maps: Vec::new(),
                aligned_views: Vec::new(),
            },
            iter_domain_per_axis: vec![0, 1, 2],
            operand_domains: vec![vec![0, 1, 2], vec![0, -1, 2]],
        };
        assert_eq!(idx.operand_axis_stride(0, 0).unwrap(), 15);
        assert_eq!(idx.operand_axis_stride(0, 1).unwrap(), 5);
        assert_eq!(idx.operand_axis_stride(0, 2).unwrap(), 1);
        // Operand 1 broadcasts axis 1: it does not move along it at all, and
        // axis 0 steps over only the axes it carries.
        assert_eq!(idx.operand_axis_stride(1, 1).unwrap(), 0);
        assert_eq!(idx.operand_axis_stride(1, 0).unwrap(), 5);
        assert_eq!(idx.batch_total().unwrap(), 30);
    }

    /// A zero extent makes an empty iteration space — a real answer, and not
    /// the one an axis-less placement gets.
    #[test]
    fn an_empty_axis_is_no_iteration_and_no_axis_is_one() {
        let mut idx = uncountable();
        idx.graph.domains[0].extent = 0;
        idx.graph.domains[1].extent = 4;
        assert_eq!(idx.batch_total().unwrap(), 0);
    }
}
