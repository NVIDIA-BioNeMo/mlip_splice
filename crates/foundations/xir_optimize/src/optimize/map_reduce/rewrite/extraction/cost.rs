// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical DAG union and accumulation placement are separate cost models.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Score {
    pub(super) placement: FiniteMapWork,
    pub(super) physical: u128,
    pub(super) quality: (usize, usize, usize),
}

impl Score {
    pub(super) fn key(self) -> (u128, (usize, usize, usize), u64, u64) {
        (
            self.physical,
            self.quality,
            self.placement.before_accumulation,
            self.placement.after_accumulation,
        )
    }

    pub(super) fn candidate(self, roots: Vec<ExpressionId>) -> RootExtraction {
        RootExtraction {
            roots,
            work: self.placement,
            physical_work: self.physical,
            quality: self.quality,
        }
    }
}

pub(super) fn arithmetic(node: &ExpressionNode) -> bool {
    matches!(
        node,
        ExpressionNode::Add(..)
            | ExpressionNode::Multiply(..)
            | ExpressionNode::Negate(..)
            | ExpressionNode::Select(..)
    )
}

pub(super) fn evaluate(
    dag: &ExpressionDag,
    roots: impl IntoIterator<Item = (ExpressionId, AccumulationSide)>,
    instances: &[u64],
    mut budget: Option<&mut ExtractionBudget>,
) -> Option<Score> {
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    let mut expressions = HashSet::new();
    let mut work = FiniteMapWork::default();
    let mut physical = 0u128;
    let mut constants = 0usize;
    let mut selects = 0usize;
    while let Some((id, side)) = pending.pop() {
        if let Some(budget) = budget.as_mut() {
            budget.reserve(1).ok()?;
        }
        if !seen.insert((id, side)) {
            continue;
        }
        let node = dag.node(id);
        if expressions.insert(id) {
            if arithmetic(node) {
                physical += u128::from(instances[id.0]);
            }
            match node {
                ExpressionNode::Coefficient { values, .. } => {
                    if let Some(budget) = budget.as_mut() {
                        budget.reserve(values.len()).ok()?;
                    }
                    constants = constants.saturating_add(
                        values
                            .iter()
                            .map(|v| v.structural_size().max(1) as u128)
                            .sum::<u128>()
                            .min(usize::MAX as u128) as usize,
                    );
                }
                ExpressionNode::Select(..) => selects += 1,
                _ => {}
            }
        }
        if matches!(
            node,
            ExpressionNode::Add(..)
                | ExpressionNode::Multiply(..)
                | ExpressionNode::Negate(..)
                | ExpressionNode::Select(..)
        ) {
            let total = match side {
                AccumulationSide::BeforeAccumulation => &mut work.before_accumulation,
                AccumulationSide::AfterAccumulation => &mut work.after_accumulation,
            };
            *total = total.saturating_add(instances[id.0]);
        }
        let child_side = if matches!(node, ExpressionNode::Sum { .. }) {
            AccumulationSide::BeforeAccumulation
        } else {
            side
        };
        pending.extend(node.children().map(|child| (child, child_side)));
    }
    Some(Score {
        placement: work,
        physical,
        quality: (expressions.len(), selects, constants),
    })
}
