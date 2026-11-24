// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How many bytes each DomainIndex reference array occupies.
//!
//! A [`Slot`] says which arrays a launch binds and in what order; this
//! says how long each one is. The two must agree exactly, because a recipe
//! states its input byte extents and a process that did not build it checks the
//! buffers it is handed against them.
//!
//! Every length here is the one the arrays are *built* with, in
//! `loom._src.domain_index`: `target_indices` and `preimage_source_indices`
//! carry one entry per source element, `preimage_offsets` carries one per
//! target cell plus the offset that closes the last preimage, and
//! `aligned_target_indices` is a source-length gather. All of them are
//! `int32`.

use crate::{Graph, Slot, SlotKind};

/// The element every DomainIndex reference array carries.
///
/// One width, not a per-array question: these arrays hold domain indices and
/// preimage offsets, all built as `int32` and read back as `s32`.
pub const INDEX_NBYTES: usize = 4;

/// Which array of a slot a length belongs to, for a caller that has to say
/// which one it could not size.
fn array_name(kind: SlotKind, position: usize) -> &'static str {
    match (kind, position) {
        (SlotKind::TargetIndices | SlotKind::ComposedTargetIndices, _) => "target_indices",
        (SlotKind::AlignedTargetIndices, _) => "aligned_target_indices",
        (SlotKind::Preimage | SlotKind::PreimageSorted, 0) => "preimage_offsets",
        _ => "preimage_source_indices",
    }
}

/// The number of elements in each array `slot` binds, in the order a launch
/// takes them.
///
/// Exhaustive over [`SlotKind`]: a new kind has to state its arrays here, and
/// the count it returns is checked against [`SlotKind::array_count`], so a kind
/// whose two statements disagree is a failure rather than a short signature.
pub fn slot_array_lengths(slot: &Slot, graph: &Graph) -> Result<Vec<i64>, String> {
    let extent = |id: i32| -> Result<i64, String> {
        graph
            .domains
            .get(usize::try_from(id).map_err(|_| format!("domain id {id} is negative"))?)
            .map(|domain| domain.extent)
            .ok_or_else(|| {
                format!(
                    "domain {id} is past the {} the graph has",
                    graph.domains.len()
                )
            })
    };
    let source = || extent(slot.source_id);
    // One per target cell, plus the offset that closes the last preimage.
    let preimage_offsets = || -> Result<i64, String> {
        extent(slot.target_id)?.checked_add(1).ok_or_else(|| {
            format!(
                "slot kind {}: its preimage offsets have no length",
                slot.kind.name()
            )
        })
    };

    let lengths = match slot.kind {
        // One entry per source element: the cell it maps to.
        SlotKind::TargetIndices | SlotKind::ComposedTargetIndices => vec![source()?],
        // `other_target_indices[preimage_source_indices]` — a source-length
        // gather.
        SlotKind::AlignedTargetIndices => vec![source()?],
        SlotKind::PreimageSorted => vec![preimage_offsets()?],
        SlotKind::Preimage => vec![preimage_offsets()?, source()?],
    };

    if lengths.len() != slot.kind.array_count() {
        return Err(format!(
            "slot kind {} binds {} arrays but states {}",
            slot.kind.name(),
            lengths.len(),
            slot.kind.array_count()
        ));
    }
    for (position, &length) in lengths.iter().enumerate() {
        if length <= 0 {
            return Err(format!(
                "slot kind {}: its {} array would hold {length} elements",
                slot.kind.name(),
                array_name(slot.kind, position)
            ));
        }
    }
    Ok(lengths)
}

/// The byte extent of each array `slot` binds, in the order a launch takes
/// them. This is what a recipe signature states for those ports.
pub fn slot_array_nbytes(slot: &Slot, graph: &Graph) -> Result<Vec<usize>, String> {
    slot_array_lengths(slot, graph)?
        .into_iter()
        .map(|length| {
            crate::extent::nbytes(
                &format!("slot kind {}", slot.kind.name()),
                &[length, INDEX_NBYTES as i64],
            )
        })
        .collect()
}

/// Every reference array `slots` bind, in slot order then array order — the
/// sequence a launch appends after its data inputs.
pub fn reference_array_nbytes(slots: &[Slot], graph: &Graph) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for slot in slots {
        out.extend(slot_array_nbytes(slot, graph)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
