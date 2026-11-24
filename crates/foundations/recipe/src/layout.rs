// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How one allocation is carved into the buffers a body declares.

/// Every workspace buffer starts on this boundary, so the region they are
/// carved from has to start on it too.
pub const WORKSPACE_ALIGN: usize = 256;

/// Where each workspace buffer starts and what the whole region spans, or
/// `None` when that region would not fit in an address.
///
/// The only form, because the sizes are whatever a caller has: rounding each up
/// to the alignment and adding them can leave the address space, and there is
/// no allocation that could hold what comes out. A caller holding a validated
/// recipe already knows the answer is `Some` — validation is this returning one
/// — and says so where it unwraps.
pub fn checked_workspace_offsets(
    sizes: impl Iterator<Item = usize>,
) -> Option<(Vec<usize>, usize)> {
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for s in sizes {
        cursor = cursor.checked_next_multiple_of(WORKSPACE_ALIGN)?;
        offsets.push(cursor);
        cursor = cursor.checked_add(s)?;
    }
    Some((offsets, cursor.checked_next_multiple_of(WORKSPACE_ALIGN)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aligned_total(sizes: impl Iterator<Item = usize>) -> usize {
        checked_workspace_offsets(sizes).expect("these fit").1
    }

    #[test]
    fn a_region_past_the_end_of_the_address_space_has_no_layout() {
        assert_eq!(checked_workspace_offsets([usize::MAX, 1].into_iter()), None);
        assert_eq!(checked_workspace_offsets([usize::MAX].into_iter()), None);
    }

    #[test]
    fn aligned_total_packs_to_256() {
        assert_eq!(aligned_total(std::iter::empty()), 0);
        assert_eq!(aligned_total([1usize].into_iter()), 256);
        // 300 then 1: 300 starts at 0, total 301 -> 512; next buffer would
        // start at 512.
        assert_eq!(aligned_total([300usize, 1].into_iter()), 512 + 256);
    }
}
