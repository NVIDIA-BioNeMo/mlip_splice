// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which nodes an operation must follow, from the buffers it touches.
//!
//! Effects are what order work: a read follows the last write, and a write
//! follows the last write and every read since it. Nothing else about an
//! operation matters here, so this is stated once over [`Effects`] and holds
//! for every kind of work a body can hold.
//!
//! An input is written by nothing inside the body, so nothing orders against
//! it: the history is over the buffers that can change.

use std::collections::{BTreeSet, HashMap};

use crate::buffer::BufRef;
use crate::effects::Effects;

#[derive(Default)]
struct History {
    last_writer: Option<usize>,
    /// Every node that has read this buffer since it was last written — a set,
    /// because the next writer must follow all of them and reading twice is
    /// reading.
    readers_since_write: BTreeSet<usize>,
}

/// What each buffer has had done to it so far, and by whom.
#[derive(Default)]
pub(crate) struct Hazards {
    history: HashMap<BufRef, History>,
}

fn orders_against(buf: BufRef) -> bool {
    !matches!(buf, BufRef::Input(_))
}

impl Hazards {
    /// The nodes an operation with these effects must follow.
    ///
    /// A set, so an edge discovered twice — once as a read-after-write and
    /// again through an explicit prerequisite — is one edge, and the order they
    /// were discovered in never reaches the recipe.
    pub(crate) fn deps(&self, effects: &Effects, extra: &[usize]) -> BTreeSet<usize> {
        let mut deps: BTreeSet<usize> = extra.iter().copied().collect();
        for (buf, access) in effects.iter() {
            if !orders_against(buf) {
                continue;
            }
            let Some(history) = self.history.get(&buf) else {
                continue;
            };
            if let Some(writer) = history.last_writer {
                deps.insert(writer);
            }
            if access.writes() {
                deps.extend(&history.readers_since_write);
            }
        }
        deps
    }

    /// Record that `node` did these things, so what comes next orders against
    /// it.
    pub(crate) fn commit(&mut self, node: usize, effects: &Effects) {
        for (buf, access) in effects.iter() {
            if !orders_against(buf) {
                continue;
            }
            let history = self.history.entry(buf).or_default();
            if access.reads() {
                history.readers_since_write.insert(node);
            }
            if access.writes() {
                history.last_writer = Some(node);
                history.readers_since_write.clear();
            }
        }
    }
}
