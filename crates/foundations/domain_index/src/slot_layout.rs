// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which reference arrays a launch must bind for a placement to be
//! addressable, and in what order.
//!
//! A [`Graph`] says how domains map into one another; a placement says where
//! each operand lives. Between them sits a question neither answers alone: to
//! reach an operand element from an iteration index, which target indices,
//! preimage offsets, preimage source indices and aligned views does the device
//! actually need in hand? [`OperandGraph`] answers it, and the answer is an
//! ordered list of [`Slot`]s, each naming one or more arrays in the order a
//! launch takes them.
//!
//! That is a query induced by a placement, which is what this crate is for. It
//! is deliberately not a wire format: nothing here encodes a slot for a reader
//! that did not build it. A slot naming a byte layout two independently-built
//! readers had to agree on would be a different mission with a different
//! obligation, and would belong elsewhere.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{AlignedView, Domain, Graph, IndexMap};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SlotKind {
    /// A declared map: one target index per source element.
    TargetIndices,
    /// The same array for a map no caller declared — the composition along a
    /// path of declared maps. It binds the same port as [`SlotKind::
    /// TargetIndices`]; the kinds differ only in who materializes it.
    ComposedTargetIndices,
    /// The preimage of a map: its offsets, then the source indices the walk
    /// visits.
    Preimage,
    /// The preimage of a sorted map: offsets alone, since the source index a
    /// walk position visits is the position itself.
    PreimageSorted,
    /// One target index per source element in the anchor's walk order — an
    /// [`AlignedView`].
    AlignedTargetIndices,
}

impl SlotKind {
    /// Every kind there is, for a caller that has to be exhaustive over them
    /// and wants that checked rather than assumed.
    pub const ALL: [SlotKind; 5] = [
        SlotKind::TargetIndices,
        SlotKind::ComposedTargetIndices,
        SlotKind::Preimage,
        SlotKind::PreimageSorted,
        SlotKind::AlignedTargetIndices,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SlotKind::TargetIndices => "target_indices",
            SlotKind::ComposedTargetIndices => "composed_target_indices",
            SlotKind::Preimage => "preimage",
            SlotKind::PreimageSorted => "preimage_sorted",
            SlotKind::AlignedTargetIndices => "aligned_target_indices",
        }
    }

    /// How many arrays a launch binds for this kind. Exhaustive, so a new kind
    /// states its own count instead of inheriting a single-array default.
    pub fn array_count(self) -> usize {
        match self {
            SlotKind::TargetIndices
            | SlotKind::ComposedTargetIndices
            | SlotKind::PreimageSorted
            | SlotKind::AlignedTargetIndices => 1,
            SlotKind::Preimage => 2,
        }
    }
}

/// The slot kind of a preimage-walk arrow: `preimage_offsets`, plus
/// `preimage_source_indices` when the map's order needs one.
fn preimage_slot_kind(map: &IndexMap) -> SlotKind {
    if map.is_sorted {
        SlotKind::PreimageSorted
    } else {
        SlotKind::Preimage
    }
}

#[derive(Clone, Debug)]
pub struct Slot {
    pub kind: SlotKind,
    pub source_id: i32,
    pub target_id: i32,
    pub other_target_id: i32,
}

fn reaches(src: i32, dst: i32, adj: &HashMap<i32, Vec<i32>>) -> bool {
    if src == dst {
        return true;
    }
    let mut seen: BTreeSet<i32> = BTreeSet::new();
    let mut stack: Vec<i32> = vec![src];
    while let Some(u) = stack.pop() {
        if !seen.insert(u) {
            continue;
        }
        if u == dst {
            return true;
        }
        if let Some(next) = adj.get(&u) {
            for &v in next {
                if !seen.contains(&v) {
                    stack.push(v);
                }
            }
        }
    }
    false
}

fn find_common_source(doms: &BTreeSet<i32>, adj: &HashMap<i32, Vec<i32>>) -> Result<i32, String> {
    let mut candidates: BTreeSet<i32> = doms.clone();
    for &source in adj.keys() {
        candidates.insert(source);
    }
    let matches: BTreeSet<i32> = candidates
        .iter()
        .copied()
        .filter(|&c| doms.iter().all(|&d| reaches(c, d, adj)))
        .collect();
    if matches.is_empty() {
        return Err(format!(
            "no common source domain among domains {:?}",
            doms.iter().collect::<Vec<_>>()
        ));
    }
    let in_set: BTreeSet<i32> = matches
        .iter()
        .copied()
        .filter(|m| doms.contains(m))
        .collect();
    let pool = if in_set.is_empty() { &matches } else { &in_set };
    let tops: BTreeSet<i32> = pool
        .iter()
        .copied()
        .filter(|m| !pool.iter().any(|o| o != m && reaches(*o, *m, adj)))
        .collect();
    if tops.len() != 1 {
        return Err(format!(
            "common source domain not unique among domains {:?}: candidates {:?}",
            doms.iter().collect::<Vec<_>>(),
            tops.iter().collect::<Vec<_>>()
        ));
    }
    Ok(tops.into_iter().next().unwrap())
}

fn single_element<T: Copy + Ord>(s: &BTreeSet<T>) -> Option<T> {
    if s.len() == 1 {
        s.iter().copied().next()
    } else {
        None
    }
}

/// The operand placement an operation states: which domains exist, how they
/// map into one another, where each operand lives, and which aligned
/// views the caller can put on device. Every iteration-placement question —
/// where to iterate each axis, which promotions the graph admits, which
/// bridge arrays the device needs — is asked of this.
pub struct OperandGraph<'a> {
    domains: &'a [Domain],
    index_maps: &'a [IndexMap],
    operand_domains: &'a [Vec<i32>],
    num_axes: usize,
    num_inputs: usize,
    available_views: &'a [AlignedView],
    /// source -> target, derived from `index maps`.
    adjacency: HashMap<i32, Vec<i32>>,
    default_iter: Vec<i32>,
}

impl<'a> OperandGraph<'a> {
    pub fn new(
        domains: &'a [Domain],
        index_maps: &'a [IndexMap],
        operand_domains: &'a [Vec<i32>],
        num_axes: usize,
        num_inputs: usize,
        available_views: &'a [AlignedView],
    ) -> Result<Self, String> {
        let mut adjacency = HashMap::<i32, Vec<i32>>::new();
        for index_map in index_maps {
            adjacency
                .entry(index_map.source_id)
                .or_default()
                .push(index_map.target_id);
        }
        let mut graph = Self {
            domains,
            index_maps,
            operand_domains,
            num_axes,
            num_inputs,
            available_views,
            adjacency,
            default_iter: Vec::new(),
        };
        graph.default_iter = graph.derive_default_iter()?;
        Ok(graph)
    }

    /// The finest iteration domain per axis, before any promotion.
    pub fn default_iter(&self) -> &[i32] {
        &self.default_iter
    }

    fn derive_default_iter(&self) -> Result<Vec<i32>, String> {
        let index_maps = self.index_maps;
        let adj = &self.adjacency;
        let mut out = Vec::with_capacity(self.num_axes);
        for axis in 0..self.num_axes {
            let doms: BTreeSet<i32> = self
                .operand_domains
                .iter()
                .filter_map(|od| {
                    let d = od[axis];
                    if d < 0 { None } else { Some(d) }
                })
                .collect();
            if doms.is_empty() {
                out.push(-1);
                continue;
            }
            if doms.len() == 1 {
                out.push(*doms.iter().next().unwrap());
                continue;
            }
            // A unique (injective) arrow source->target lets us iterate the
            // *target* end. But that is a legal iteration axis only if every
            // operand on this axis can still be bridged to it the way
            // `build_domain_index` does at default-iter time: a direct
            // source->target csr (`direct(d, c)`), or a target->operand
            // declared-or-composed map (`reaches(c, d)`). Views are NOT available here — they
            // are a later aligned-iter step — so they cannot license the axis.
            // A unique arrow off one operand (e.g. dst->species when the species
            // happen to be one-per-node) must NOT promote its target when a
            // sibling operand (src, edge) has no path to it; keep only coarsers
            // every operand can reach, else fall back to the common source.
            let direct: BTreeSet<(i32, i32)> = index_maps
                .iter()
                .map(|r| (r.source_id, r.target_id))
                .collect();
            let bridgeable = |c: i32| {
                doms.iter()
                    .all(|&d| d == c || direct.contains(&(d, c)) || reaches(c, d, adj))
            };
            let true_coarsers: BTreeSet<i32> = index_maps
                .iter()
                .filter(|r| r.is_injective && doms.contains(&r.source_id))
                .map(|r| r.target_id)
                .filter(|&c| bridgeable(c))
                .collect();
            if !true_coarsers.is_empty() {
                if true_coarsers.len() > 1 {
                    return Err(format!(
                        "axis {axis}: multiple is_injective=true arrows imply different \
                     iter axes (target ends: {:?})",
                        true_coarsers.iter().collect::<Vec<_>>()
                    ));
                }
                out.push(*true_coarsers.iter().next().unwrap());
                continue;
            }
            out.push(find_common_source(&doms, adj)?);
        }
        Ok(out)
    }

    /// The target domains axis `axis` could be promoted onto: those a non-unique
    /// index map reaches from the default iteration domain. `outputs_only`
    /// restricts them to domains an operand *written* by the operation lives on.
    fn axis_anchor_candidates(
        &self,
        axis: usize,
        iter_d: i32,
        outputs_only: bool,
    ) -> BTreeSet<i32> {
        let (index_maps, num_inputs) = (self.index_maps, self.num_inputs);
        self.operand_domains
            .iter()
            .enumerate()
            .filter(|(operand, _)| !outputs_only || *operand >= num_inputs)
            .filter_map(|(_, od)| {
                let d = od[axis];
                if d < 0 || d == iter_d {
                    return None;
                }
                let r = index_maps
                    .iter()
                    .find(|r| r.source_id == iter_d && r.target_id == d)?;
                if r.is_injective {
                    return None;
                }
                Some(d)
            })
            .collect()
    }

    /// The aligned views needed to promote `axis` onto `anchor`, or `None` when
    /// promotion is not feasible. Every operand must reach the anchor through
    /// an index map or an available aligned view. An output mapped from a source
    /// into the anchor needs `fiber_scatter_outputs` so the preimage walk can
    /// write it from inside the loop.
    fn axis_promotion(
        &self,
        axis: usize,
        anchor: i32,
        promote_min_extent: i64,
        fiber_scatter_outputs: bool,
    ) -> Option<Vec<AlignedView>> {
        let (domains, index_maps, operand_domains, num_inputs, available_views) = (
            self.domains,
            self.index_maps,
            self.operand_domains,
            self.num_inputs,
            self.available_views,
        );
        // Promoting onto a small target domain iterates few elements — one
        // thread per target element, underfilling the GPU. Keep the source
        // (default) iteration below the cutoff: the output then scatters into those few
        // cells, which the caller reduces (a whole-grid tree reduce at
        // extent 1, a shared-memory histogram above it) instead of a handful of
        // threads each walking a whole fiber serially. `promote_min_extent`
        // bounds where a fiber gather is worth its coarse-iteration underfill.
        if domains[anchor as usize].extent <= promote_min_extent {
            return None;
        }
        let mut axis_views = Vec::new();
        for (operand, od) in operand_domains.iter().enumerate() {
            let d = od[axis];
            if d < 0 || d == anchor {
                continue;
            }
            let finer_side = index_maps
                .iter()
                .find(|r| r.source_id == d && r.target_id == anchor);
            if let Some(finer_ref) = finer_side {
                let scatters = fiber_scatter_outputs && !finer_ref.may_have_unmapped;
                if operand >= num_inputs && !scatters {
                    return None;
                }
                continue;
            }
            // A view gathers the operand's cell for the anchor's position; it
            // carries no inverse, so an output reaching the anchor only this way
            // has nowhere to accumulate. Outputs must live on the anchor or refine
            // into it.
            if operand >= num_inputs {
                return None;
            }
            let view = available_views.iter().find(|v| {
                v.anchor_target_id == anchor
                    && v.other_target_id == d
                    && index_maps
                        .iter()
                        .any(|r| r.source_id == v.source_id && r.target_id == anchor)
            })?;
            axis_views.push(*view);
        }
        Some(axis_views)
    }

    /// The promoted placement to take when only one can run: it promotes an axis
    /// only onto the single coarse domain the *outputs* name, which is the anchor
    /// that turns a scatter into a write at the iterated position. It is also the
    /// first of [`OperandGraph::promoted_placements`]; callers that can measure
    /// should enumerate the rest.
    ///
    /// `fiber_scatter_outputs` opts outputs into the fiber's source domain: the
    /// promotion then stays feasible when an output refines into the anchor
    /// through a total map — the fiber walk covers each source element
    /// exactly once, so the write is exact (fiber-direct, no atomics). Only
    /// emitters that implement fiber-direct output writes may pass `true`.
    pub fn promoted_iter(
        &self,
        promote_min_extent: i64,
        fiber_scatter_outputs: bool,
    ) -> Result<IterPlacement, String> {
        let iter_default = &self.default_iter;
        let mut out_iter = iter_default.clone();
        let mut used_views: Vec<AlignedView> = Vec::new();
        let mut used_set: BTreeSet<(i32, i32, i32)> = BTreeSet::new();
        for axis in 0..self.num_axes {
            let iter_d = iter_default[axis];
            if iter_d < 0 {
                continue;
            }
            let candidates = self.axis_anchor_candidates(axis, iter_d, true);
            let Some(anchor) = single_element(&candidates) else {
                continue;
            };
            let Some(axis_views) =
                self.axis_promotion(axis, anchor, promote_min_extent, fiber_scatter_outputs)
            else {
                continue;
            };
            out_iter[axis] = anchor;
            for v in axis_views {
                if used_set.insert((v.source_id, v.anchor_target_id, v.other_target_id)) {
                    used_views.push(v);
                }
            }
        }
        Ok((out_iter, used_views))
    }

    /// Every promoted placement the operand graph admits, the one
    /// [`OperandGraph::promoted_iter`] picks first.
    ///
    /// Where to iterate is a traffic question over *all* operands, not only the
    /// written ones: an anchor no output lives on still reads that domain's
    /// operands once per position instead of once per fiber element, and the
    /// outputs elsewhere scatter exactly as they already do under the default
    /// iteration. So the alternatives promote one axis onto any coarse domain the
    /// graph can reach, which is every placement when the operation has a single
    /// batch axis. Callers that can only run one placement take the first;
    /// callers that can measure should try them all.
    pub fn promoted_placements(
        &self,
        promote_min_extent: i64,
        fiber_scatter_outputs: bool,
    ) -> Result<Vec<IterPlacement>, String> {
        let iter_default = &self.default_iter;
        let first = self.promoted_iter(promote_min_extent, fiber_scatter_outputs)?;
        let mut placements = vec![first];
        for axis in 0..self.num_axes {
            let iter_d = iter_default[axis];
            if iter_d < 0 {
                continue;
            }
            for anchor in self.axis_anchor_candidates(axis, iter_d, false) {
                let Some(views) =
                    self.axis_promotion(axis, anchor, promote_min_extent, fiber_scatter_outputs)
                else {
                    continue;
                };
                let mut iter = iter_default.clone();
                iter[axis] = anchor;
                if !placements.iter().any(|(known, _)| *known == iter) {
                    placements.push((iter, views));
                }
            }
        }
        Ok(placements)
    }

    /// The index map graph a problem stores (the operand placements are kept by
    /// the caller, unchanged), the iteration domain per axis, and the recipe slots
    /// that carry the index map/view arrays on device.
    pub fn build_index(
        &self,
        iter_per_axis: &[i32],
        used_views: &[AlignedView],
    ) -> Result<BuiltIndex, String> {
        build_domain_index(
            self.domains,
            self.index_maps,
            self.operand_domains,
            self.num_axes,
            iter_per_axis,
            used_views,
            &self.adjacency,
        )
    }
}

/// An iteration domain per axis, with the aligned views that placement
/// needs on device.
pub type IterPlacement = (Vec<i32>, Vec<AlignedView>);

fn register_anchor_ref(
    source_id: i32,
    anchor_target_id: i32,
    direct_refs: &BTreeMap<(i32, i32), IndexMap>,
    engine_refs: &mut Vec<IndexMap>,
    slots: &mut Vec<Slot>,
    ref_bridge_slot: &mut BTreeMap<(i32, i32), usize>,
) {
    if ref_bridge_slot.contains_key(&(anchor_target_id, source_id)) {
        return;
    }
    let r = direct_refs
        .get(&(source_id, anchor_target_id))
        .copied()
        .unwrap_or(IndexMap {
            source_id,
            target_id: anchor_target_id,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        });
    let slot = engine_refs.len();
    ref_bridge_slot.insert((anchor_target_id, source_id), slot);
    slots.push(Slot {
        kind: preimage_slot_kind(&r),
        source_id,
        target_id: anchor_target_id,
        other_target_id: anchor_target_id,
    });
    engine_refs.push(r);
}

type BuiltIndex = (Graph, Vec<i32>, Vec<Slot>);

fn build_domain_index(
    domains: &[Domain],
    index_maps: &[IndexMap],
    operand_domains: &[Vec<i32>],
    num_axes: usize,
    iter_per_axis: &[i32],
    used_views: &[AlignedView],
    adjacency: &HashMap<i32, Vec<i32>>,
) -> Result<BuiltIndex, String> {
    let mut domains = domains.to_vec();
    let trivial_id: Option<i32> = if iter_per_axis.iter().any(|&d| d < 0) {
        let id = domains.len() as i32;
        domains.push(Domain { extent: 1 });
        Some(id)
    } else {
        None
    };

    let iter_domain_per_axis: Vec<i32> = iter_per_axis
        .iter()
        .map(|&d| if d < 0 { trivial_id.unwrap() } else { d })
        .collect();

    let mut engine_refs: Vec<IndexMap> = Vec::new();
    let mut slots: Vec<Slot> = Vec::new();
    let mut ref_bridge_slot: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut view_bridge_slot: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    let mut engine_views: Vec<AlignedView> = Vec::new();

    let direct_refs: BTreeMap<(i32, i32), IndexMap> = index_maps
        .iter()
        .map(|r| ((r.source_id, r.target_id), *r))
        .collect();

    // A composed chain absorbs `-1` from any arrow on its path, so guard a chain
    // bridge whenever any declared index map carries unmapped elements. The
    // exact composition path is resolved caller-side; a spurious `>= 0` guard on
    // a chain that never holds `-1` just always passes.
    let any_unmapped = direct_refs.values().any(|r| r.may_have_unmapped);

    for od in operand_domains {
        for axis in 0..num_axes {
            let d = od[axis];
            if d < 0 {
                continue;
            }
            let c = iter_per_axis[axis];
            if c < 0 || d == c {
                continue;
            }
            if ref_bridge_slot.contains_key(&(c, d)) || view_bridge_slot.contains_key(&(c, d)) {
                continue;
            }
            let ref_b = direct_refs.get(&(c, d));
            let ref_c = direct_refs.get(&(d, c));
            if ref_b.is_some() && ref_c.is_some() {
                return Err(format!(
                    "index-map ambiguity for iter={c}, operand={d}: \
                     both ({c}, {d}) and ({d}, {c}) declared"
                ));
            }
            let slot = engine_refs.len();
            if let Some(&r) = ref_c {
                ref_bridge_slot.insert((c, d), slot);
                slots.push(Slot {
                    kind: preimage_slot_kind(&r),
                    source_id: d,
                    target_id: c,
                    other_target_id: c,
                });
                engine_refs.push(r);
            } else if let Some(r) = ref_b {
                ref_bridge_slot.insert((c, d), slot);
                engine_refs.push(IndexMap {
                    source_id: c,
                    target_id: d,
                    is_injective: false,
                    // A map-lookup slot ignores sortedness, but carrying it
                    // through lets a consumer (matmul's grouped tiling) rely on
                    // the source side being grouped contiguously by the target.
                    is_sorted: r.is_sorted,
                    may_have_unmapped: r.may_have_unmapped,
                });
                slots.push(Slot {
                    kind: SlotKind::TargetIndices,
                    source_id: c,
                    target_id: d,
                    other_target_id: d,
                });
            } else {
                if let Some(v) = used_views
                    .iter()
                    .find(|v| v.anchor_target_id == c && v.other_target_id == d)
                {
                    register_anchor_ref(
                        v.source_id,
                        c,
                        &direct_refs,
                        &mut engine_refs,
                        &mut slots,
                        &mut ref_bridge_slot,
                    );
                    let vslot = engine_views.len();
                    view_bridge_slot.insert((c, d), vslot);
                    engine_views.push(*v);
                    slots.push(Slot {
                        kind: SlotKind::AlignedTargetIndices,
                        source_id: v.source_id,
                        target_id: c,
                        other_target_id: d,
                    });
                } else if reaches(c, d, adjacency) {
                    ref_bridge_slot.insert((c, d), slot);
                    engine_refs.push(IndexMap {
                        source_id: c,
                        target_id: d,
                        is_injective: false,
                        is_sorted: false,
                        may_have_unmapped: any_unmapped,
                    });
                    slots.push(Slot {
                        kind: SlotKind::ComposedTargetIndices,
                        source_id: c,
                        target_id: d,
                        other_target_id: d,
                    });
                } else {
                    return Err(format!(
                        "no index-map path from iteration domain {c} to operand domain {d}"
                    ));
                }
            }
        }
    }

    Ok((
        Graph {
            domains,
            index_maps: engine_refs,
            aligned_views: engine_views,
        },
        iter_domain_per_axis,
        slots,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kinds are a closed set with distinct names, and the arrays each one
    /// binds are the arrays a launch is built to take.
    #[test]
    fn every_kind_states_how_many_arrays_it_binds() {
        for kind in SlotKind::ALL {
            assert!(
                (1..=2).contains(&kind.array_count()),
                "{} binds {} arrays",
                kind.name(),
                kind.array_count()
            );
        }
        let names: BTreeSet<&str> = SlotKind::ALL.iter().map(|k| k.name()).collect();
        assert_eq!(
            names.len(),
            SlotKind::ALL.len(),
            "two kinds answer to one name"
        );
        assert_eq!(
            SlotKind::ALL.map(|k| k.array_count()),
            [1, 1, 2, 1, 1],
            "the arrays each kind binds are what a launch is built to take"
        );
    }

    #[test]
    fn an_unreachable_explicit_iteration_domain_is_refused() {
        let domains = [Domain { extent: 2 }, Domain { extent: 3 }];
        let operand_domains = [vec![1]];
        let graph = OperandGraph::new(&domains, &[], &operand_domains, 1, 1, &[]).unwrap();
        let error = graph.build_index(&[0], &[]).unwrap_err();
        assert!(error.contains("no index-map path"), "{error}");
    }

    /// A preimage arrow's kind is decided by the map alone: a sorted map needs
    /// no source indices, since the walk position is the source element.
    #[test]
    fn a_preimage_arrows_kind_is_what_its_map_says() {
        let arrow = |sorted| {
            preimage_slot_kind(&IndexMap {
                source_id: 0,
                target_id: 1,
                is_injective: false,
                is_sorted: sorted,
                may_have_unmapped: false,
            })
        };
        assert_eq!(arrow(false), SlotKind::Preimage);
        assert_eq!(arrow(true), SlotKind::PreimageSorted);
    }

    fn index_map(source: i32, target: i32, unique: bool) -> IndexMap {
        IndexMap {
            source_id: source,
            target_id: target,
            is_injective: unique,
            is_sorted: false,
            may_have_unmapped: false,
        }
    }

    // Domain ids for the graph-attention shape: edge -> src, edge -> dst,
    // dst -> species.
    const EDGE: i32 = 0;
    const SRC: i32 = 1;
    const DST: i32 = 2;
    const SPECIES: i32 = 3;

    #[test]
    fn unique_dst_species_does_not_promote_species_over_edge_operands() {
        // dst->species is unique (one atom per species — happens whenever the
        // batch's species are all distinct). The QK op mixes operands on edge,
        // dst and src on one axis; src/edge cannot reach species, so the iter
        // axis must stay on the common source `edge`, not jump to `species`
        // (which used to emit an impossible species->edge chain).
        let refs = [
            index_map(EDGE, SRC, false),
            index_map(EDGE, DST, false),
            index_map(DST, SPECIES, true),
        ];
        let operand_domains = [
            vec![EDGE], // w
            vec![DST],  // q
            vec![SRC],  // k
            vec![EDGE], // sph
            vec![EDGE], // output
        ];
        let graph = OperandGraph::new(&[], &refs, &operand_domains, 1, 4, &[]).unwrap();
        assert_eq!(graph.default_iter(), [EDGE]);
    }

    #[test]
    fn fiber_scatter_outputs_keep_the_promotion_feasible() {
        // Mixed outputs: one on edge (source), one on dst (target, non-unique).
        // The default derive keeps the fine iteration — an edge output blocks
        // the promotion — while the fiber-scatter mode promotes onto dst as
        // long as the edge -> dst map is total.
        let domains = vec![
            Domain { extent: 12 },
            Domain { extent: 4 },
            Domain { extent: 4 },
        ];
        let operand_domains = [vec![EDGE], vec![DST], vec![EDGE], vec![DST]];
        let derive = |refs: &[IndexMap], scatter: bool| {
            OperandGraph::new(&domains, refs, &operand_domains, 1, 2, &[])
                .unwrap()
                .promoted_iter(1, scatter)
                .unwrap()
                .0
        };
        let total = [index_map(EDGE, DST, false)];
        assert_eq!(derive(&total, false), vec![EDGE]);
        assert_eq!(derive(&total, true), vec![DST]);

        let mut unmapped = total;
        unmapped[0].may_have_unmapped = true;
        assert_eq!(derive(&unmapped, true), vec![EDGE]);
    }

    #[test]
    fn unique_arrow_still_promotes_its_coarser_when_all_operands_bridge() {
        // The optimization is preserved where it is sound: operands live only on
        // dst and species, dst->species is unique, and every operand bridges to
        // species (dst via the direct arrow, species natively) — so iterating
        // the target `species` is legal and chosen.
        let refs = [index_map(DST, SPECIES, true)];
        let operand_domains = [vec![DST], vec![SPECIES]];
        let graph = OperandGraph::new(&[], &refs, &operand_domains, 1, 1, &[]).unwrap();
        assert_eq!(graph.default_iter(), [SPECIES]);
    }
}
