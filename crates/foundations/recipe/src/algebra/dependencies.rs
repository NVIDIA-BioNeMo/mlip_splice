// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a node waits for, and what may run beside it.

use std::collections::BTreeSet;

use crate::algebra::computation::Computation;
use crate::algebra::data::{Recipe, RecipeBody};

/// For each node, every node it transitively waits for. Deps name strictly
/// earlier nodes, so one forward pass is the whole closure.
fn ancestors<F: Computation>(body: &RecipeBody<F>) -> Vec<BTreeSet<usize>> {
    let mut ancestors: Vec<BTreeSet<usize>> = Vec::with_capacity(body.nodes().len());
    for node in body.nodes() {
        let mut waits = BTreeSet::new();
        for &dependency in node.deps() {
            waits.insert(dependency);
            if let Some(reached) = ancestors.get(dependency) {
                waits.extend(reached);
            }
        }
        ancestors.push(waits);
    }
    ancestors
}

impl<F: Computation> RecipeBody<F> {
    /// Every transitive dependency of `node`, in ascending order.
    pub fn waits_for(&self, node: usize) -> Option<Vec<usize>> {
        ancestors(self)
            .get(node)
            .map(|waits| waits.iter().copied().collect())
    }

    /// Every pair of nodes whose dependency paths permit concurrent execution.
    pub fn concurrent(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (later, waits) in ancestors(self).iter().enumerate() {
            for earlier in 0..later {
                if !waits.contains(&earlier) {
                    pairs.push((earlier, later));
                }
            }
        }
        pairs
    }
}

impl<F: Computation> Recipe<F> {
    pub fn waits_for(&self, node: usize) -> Option<Vec<usize>> {
        self.body().waits_for(node)
    }

    pub fn concurrent(&self) -> Vec<(usize, usize)> {
        self.body().concurrent()
    }
}
