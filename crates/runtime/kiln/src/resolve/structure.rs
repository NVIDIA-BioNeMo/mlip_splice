// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};

use recipe::algebra::{Launches, ModuleTable};
use recipe::{ImagePersistence, KernelCode, KernelLaunch, ModuleImage, RecipeBody, Work};

/// The entries each module of `body` is launched at.
///
/// A set: launching one entry from two nodes launches it, and which node came
/// first is not a fact about the module. Every consumer looks a measurement up
/// by name, so nothing downstream depends on an order either.
pub(super) fn entries_by_module<F: Launches>(body: &RecipeBody<F>) -> Vec<BTreeSet<String>> {
    let mut entries = vec![BTreeSet::new(); body.context().modules().len()];
    for node in body.nodes() {
        if let Work::Call {
            call: KernelLaunch { module, entry, .. },
            ..
        } = node.work()
            && let Some(module_entries) = entries.get_mut(*module)
        {
            module_entries.insert(entry.clone());
        }
    }
    entries
}

pub(crate) fn collect_request_entries<F: Launches>(
    body: &RecipeBody<F>,
    entries: &mut [BTreeSet<String>],
) {
    let by_module = entries_by_module(body);
    for (module, module_entries) in body.context().modules().iter().zip(by_module) {
        if let Some(request) = module.code.compile_request()
            && let Some(request_entries) = entries.get_mut(request.0)
        {
            request_entries.extend(module_entries);
        }
    }
    for node in body.nodes() {
        if let Work::Choice { candidates, .. } = node.work() {
            for candidate in candidates {
                collect_request_entries(candidate, entries);
            }
        }
    }
}

/// Every compile request a measurement of `body` must resolve before it can be
/// timed: the modules `body` launches itself, and the same for the first
/// candidate of each nested choice — the default selection the tuner's warm-up
/// run executes before any candidate is measured.
///
/// A nested choice's *other* candidates are deliberately absent. They are
/// reached only when the traversal gets to them, so a subtree the tuning bound
/// prunes is never compiled, and priming this set never becomes eager
/// compilation of every losing branch.
pub(super) fn certain_requests<F: Launches>(body: &RecipeBody<F>, into: &mut Vec<usize>) {
    for module in body.context().modules() {
        if let KernelCode::PtxBacked {
            compile_request,
            images,
        } = &module.code
            && images.is_empty()
            && !into.contains(&compile_request.0)
        {
            into.push(compile_request.0);
        }
    }
    for node in body.nodes() {
        if let Work::Choice { candidates, .. } = node.work()
            && let Some(first) = candidates.first()
        {
            certain_requests(first, into);
        }
    }
}

/// The same, for a caller outside this module that prices a body before it
/// opens it. Naming the private walk keeps one definition of what a body would
/// certainly compile, so a price and a prefetch can never disagree about it.
pub(crate) fn certain_requests_of<F: Launches>(body: &RecipeBody<F>, into: &mut Vec<usize>) {
    certain_requests(body, into);
}

/// The candidates whose requests to start before measuring `order[position]`:
/// the whole window at the first step, then one more at each step after, so at
/// most `window` candidate request sets are ever outstanding and they are
/// started in the order the tuner will measure them.
pub(crate) fn prefetch_at(order: &[usize], position: usize, window: usize) -> &[usize] {
    if window == 0 || position >= order.len() {
        return &[];
    }
    let start = if position == 0 {
        0
    } else {
        (position + window - 1).min(order.len())
    };
    let end = (position + window).min(order.len());
    &order[start.min(end)..end]
}

/// What a resolved body's modules are, apart from their bytes: which kind of
/// image each is and what it is for.
///
/// The bytes are not here. They are payloads, and which payloads a body carries
/// is [`recipe::payloads`]'s to say — see [`crate::tune`], which hashes them from
/// that walk so a payload role added there is measured rather than silently
/// shared between two candidates.
pub(crate) fn hash_resolved_modules<F: Launches>(recipe: &RecipeBody<F>, hasher: &mut impl Hasher) {
    for module in recipe.context().modules() {
        for image in module.code.images() {
            match image {
                ModuleImage::Cubin { target_sm, .. } => (0u8, target_sm).hash(hasher),
                ModuleImage::CudaLoadable { persistence, .. } => {
                    (1u8, *persistence == ImagePersistence::SessionOnly).hash(hasher)
                }
            }
        }
    }
}
