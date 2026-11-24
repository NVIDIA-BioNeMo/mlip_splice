// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where a partition boundary falls, as the record a later representation is
//! compared against.

use super::super::partition::KernelPart;

/// Where one bounded partition part draws its boundary, and how it got there.
///
/// The slices are the boundary itself: the element/term span this part writes
/// and no other does. `bisections` is how many halvings index map needed to
/// bring an over-cap part inside the cap, and `oversized` marks the part it
/// could not bring inside the cap at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in crate::backend) struct PartBoundary {
    pub bisections: usize,
    pub oversized: bool,
    pub slices: Vec<(usize, usize, Vec<usize>)>,
}

impl PartBoundary {
    pub(in crate::backend) fn of(part: &KernelPart, bisections: usize, oversized: bool) -> Self {
        Self {
            bisections,
            oversized,
            slices: part
                .elements
                .iter()
                .enumerate()
                .flat_map(|(output, selections)| {
                    selections
                        .iter()
                        .map(move |selection| (output, selection.element, selection.terms.clone()))
                })
                .collect(),
        }
    }
}
