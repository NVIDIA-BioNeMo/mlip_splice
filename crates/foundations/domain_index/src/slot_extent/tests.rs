// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The arrays each slot kind binds, and how long each one is.
//!
//! Every extent in the fixture is distinct and no two derived lengths collide,
//! so a rule that read the wrong domain — source for target, target for the
//! view's other end — produces a different number rather than the same one.

use crate::{Domain, Graph, IndexMap};

use super::*;

const SOURCE: i32 = 0;
const TARGET: i32 = 1;
const OTHER: i32 = 2;

/// Distinct extents, so no swap is invisible: source 12, target 5, other 7.
fn graph() -> Graph {
    Graph {
        domains: vec![
            Domain { extent: 12 },
            Domain { extent: 5 },
            Domain { extent: 7 },
        ],
        index_maps: vec![IndexMap {
            source_id: SOURCE,
            target_id: TARGET,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        aligned_views: Vec::new(),
    }
}

fn slot(kind: SlotKind) -> Slot {
    Slot {
        kind,
        source_id: SOURCE,
        target_id: TARGET,
        other_target_id: OTHER,
    }
}

fn lengths(kind: SlotKind) -> Vec<i64> {
    slot_array_lengths(&slot(kind), &graph()).expect("the fixture sizes every kind")
}

/// Every kind, its arrays in launch order. Read against
/// `loom._src.domain_index`: `target_indices` and `preimage_source_indices`
/// are source-length, `preimage_offsets` is target + 1, and
/// `aligned_target_indices` is `other_target_indices[preimage_source_indices]`
/// — source-length.
#[test]
fn every_slot_kind_states_the_arrays_its_launch_binds() {
    // source 12, target 5 -> preimage offsets 6. No two of these numbers are
    // equal.
    for (kind, expected) in [
        (SlotKind::TargetIndices, vec![12]),
        (SlotKind::ComposedTargetIndices, vec![12]),
        (SlotKind::AlignedTargetIndices, vec![12]),
        (SlotKind::PreimageSorted, vec![6]),
        (SlotKind::Preimage, vec![6, 12]),
    ] {
        assert_eq!(lengths(kind), expected, "{}", kind.name());
    }
}

/// Every kind there is is sized above, and every one agrees with the count it
/// separately states.
#[test]
fn every_kind_is_sized_and_agrees_with_its_own_count() {
    for kind in SlotKind::ALL {
        assert_eq!(
            lengths(kind).len(),
            kind.array_count(),
            "{} binds a different number of arrays than it states",
            kind.name()
        );
    }
    assert_eq!(
        SlotKind::ALL.len(),
        5,
        "the kinds sized here are the kinds that exist"
    );
}

/// The closing offset is the whole point of `preimage_offsets`: one preimage
/// per target cell needs one more offset than there are cells, and it is the
/// *target* extent that counts.
#[test]
fn preimage_offsets_carry_one_per_cell_and_the_offset_that_closes_the_last() {
    let target = 5;
    for kind in [SlotKind::PreimageSorted, SlotKind::Preimage] {
        assert_eq!(lengths(kind)[0], target + 1, "{}", kind.name());
    }
    // Not the source extent, and not the target extent alone.
    assert_ne!(lengths(SlotKind::PreimageSorted)[0], 12);
    assert_ne!(lengths(SlotKind::PreimageSorted)[0], 5);
}

/// An aligned view is gathered through the source domain's preimage order, so
/// it is source-length however wide the domain it points into.
#[test]
fn an_aligned_view_is_a_source_length_gather() {
    assert_eq!(lengths(SlotKind::AlignedTargetIndices), vec![12]);
    assert_ne!(
        lengths(SlotKind::AlignedTargetIndices),
        vec![7],
        "not the other end"
    );
    assert_ne!(
        lengths(SlotKind::AlignedTargetIndices),
        vec![5],
        "not the anchor"
    );
}

/// Swapping the domains a slot names changes what it is sized against — which
/// is what makes the fixture's distinct extents load-bearing.
#[test]
fn a_slot_naming_its_domains_the_other_way_round_is_sized_the_other_way_round() {
    let swapped = Slot {
        kind: SlotKind::TargetIndices,
        source_id: TARGET,
        target_id: SOURCE,
        other_target_id: OTHER,
    };
    assert_eq!(
        slot_array_lengths(&swapped, &graph()).unwrap(),
        vec![5],
        "a target-index map is as long as whichever domain it calls source"
    );
}

/// Bytes are indices, four apiece.
#[test]
fn a_byte_extent_is_its_length_in_index_words() {
    assert_eq!(INDEX_NBYTES, 4);
    assert_eq!(
        slot_array_nbytes(&slot(SlotKind::Preimage), &graph()).unwrap(),
        vec![6 * 4, 12 * 4]
    );
}

/// The sequence a launch appends after its data inputs: slot order, then array
/// order within each slot.
#[test]
fn the_reference_arrays_follow_slot_order_then_array_order() {
    let slots = [
        slot(SlotKind::PreimageSorted),
        slot(SlotKind::TargetIndices),
    ];
    assert_eq!(
        reference_array_nbytes(&slots, &graph()).unwrap(),
        vec![6 * 4, 12 * 4]
    );
}

/// A domain the graph does not have is refused rather than sized.
#[test]
fn a_slot_naming_a_domain_the_graph_lacks_is_refused() {
    let missing = Slot {
        kind: SlotKind::TargetIndices,
        source_id: 9,
        target_id: TARGET,
        other_target_id: OTHER,
    };
    assert!(slot_array_lengths(&missing, &graph()).is_err());
}

/// `preimage_offsets` over a domain at the top of the counting range has no
/// length, which would otherwise state a buffer smaller than the walk reads.
#[test]
fn a_length_past_what_an_i64_counts_is_refused() {
    let mut graph = graph();
    graph.domains[TARGET as usize].extent = i64::MAX;
    let error = slot_array_lengths(&slot(SlotKind::PreimageSorted), &graph).unwrap_err();
    assert!(error.contains("preimage offsets"), "{error}");
}

/// A length that fits but whose byte extent does not is refused at the cast.
#[test]
fn a_byte_extent_past_the_address_space_is_refused() {
    let mut graph = graph();
    graph.domains[SOURCE as usize].extent = i64::MAX / 2;
    assert!(slot_array_nbytes(&slot(SlotKind::TargetIndices), &graph).is_err());
}
