// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How much scratch a recipe needs, whichever way its choices go.
//!
//! A body carves one allocation into the buffers it declares, in the order it
//! declares them. A choice adds whatever its worst candidate needs, because the
//! allocation has to be enough before anyone has chosen — and sibling choices
//! add, since a body runs all of them.
//!
//! The arithmetic is fallible all the way down, and validation runs it: a
//! recipe whose declarations do not fit in an address is refused where it is
//! proved. So the bound of a recipe that exists is a number, and asking for it
//! neither wraps nor panics.

use crate::algebra::computation::Computation;
use crate::algebra::data::{Recipe, RecipeBody, Work};
use crate::layout::checked_workspace_offsets;

/// What each workspace buffer of `body` spans, in the index order a node names
/// them by — which is the order one allocation is carved into.
pub fn workspace_sizes<F: Computation>(body: &RecipeBody<F>) -> impl Iterator<Item = usize> + '_ {
    body.workspace().iter().map(|buffer| buffer.nbytes)
}

/// The bound, or `None` when the declarations do not fit in an address.
///
/// This is the form a body has, because a body carries no proof: it is whatever
/// a producer assembled, and the arithmetic below is fallible at every step. The
/// infallible form is a recipe's, where validation has already run this and
/// refused a `None`.
pub fn checked_body_workspace_upper_bound<F: Computation>(body: &RecipeBody<F>) -> Option<usize> {
    let own = checked_workspace_offsets(workspace_sizes(body))?.1;
    let mut total = own;
    for node in body.nodes() {
        let Work::Choice { candidates, .. } = node.work() else {
            continue;
        };
        let mut worst = 0usize;
        for candidate in candidates {
            worst = worst.max(checked_body_workspace_upper_bound(candidate)?);
        }
        total = total.checked_add(worst)?;
    }
    Some(total)
}

pub fn checked_workspace_upper_bound<F: Computation>(recipe: &Recipe<F>) -> Option<usize> {
    checked_body_workspace_upper_bound(recipe.body())
}

/// The bound of a recipe, which exists because the recipe does.
///
/// A recipe is validated before it exists and validation refuses one whose
/// workspace has no layout, so this is the one place the answer is a number
/// rather than a maybe. There is no such form for a body: a body is whatever a
/// producer assembled, and asking it for a number would be asking it to prove
/// something it does not carry.
pub fn workspace_upper_bound<F: Computation>(recipe: &Recipe<F>) -> usize {
    checked_body_workspace_upper_bound(recipe.body())
        .expect("a validated recipe's workspace fits in an address")
}
