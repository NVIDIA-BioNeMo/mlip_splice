// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reversible reference counts for complete root sets in an immutable DAG.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Totals {
    placement: [u128; 2],
    physical: u128,
    nodes: usize,
    selects: usize,
    constants: u128,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Usage {
    counts: Vec<[usize; 2]>,
    totals: Totals,
}

impl Usage {
    pub(super) fn new(
        dag: &ExpressionDag,
        roots: impl IntoIterator<Item = (ExpressionId, AccumulationSide)>,
        instances: &[u64],
        mut budget: Option<&mut ExtractionBudget>,
    ) -> Option<Self> {
        let mut result = Self {
            counts: vec![[0; 2]; dag.len()],
            ..Self::default()
        };
        for root in roots {
            result.change(dag, root, instances, true, &mut budget, &mut None)?;
        }
        Some(result)
    }

    pub(super) fn live(&self, id: ExpressionId) -> bool {
        self.counts
            .get(id.0)
            .is_some_and(|counts| *counts != [0; 2])
    }

    pub(super) fn score(&self) -> Score {
        let totals = self.totals;
        Score {
            placement: FiniteMapWork {
                before_accumulation: totals.placement[0].min(u128::from(u64::MAX)) as u64,
                after_accumulation: totals.placement[1].min(u128::from(u64::MAX)) as u64,
            },
            physical: totals.physical,
            quality: (
                totals.nodes,
                totals.selects,
                totals.constants.min(usize::MAX as u128) as usize,
            ),
        }
    }

    /// Add first to avoid rescanning shared subgraphs. Every mutation reserves
    /// its undo record as well; rollback needs no remaining search allowance.
    /// Returns the complete trial score, including when no improvement commits.
    pub(super) fn improve(
        &mut self,
        dag: &ExpressionDag,
        changes: &[(ExpressionId, ExpressionId)],
        instances: &[u64],
        budget: &mut ExtractionBudget,
    ) -> Option<Score> {
        let saved = self.totals;
        let len = self.counts.len();
        let before = self.score().key();
        self.counts.resize(dag.len(), [0; 2]);
        let mut journal = Vec::new();
        let result = (|| {
            for adding in [true, false] {
                for &(old, new) in changes {
                    if old != new {
                        self.change(
                            dag,
                            (
                                if adding { new } else { old },
                                AccumulationSide::AfterAccumulation,
                            ),
                            instances,
                            adding,
                            &mut Some(&mut *budget),
                            &mut Some(&mut journal),
                        )?;
                    }
                }
            }
            Some(self.score())
        })();
        if !result.is_some_and(|score| score.key() < before) {
            for (id, side, count) in journal.into_iter().rev() {
                self.counts[id.0][side] = count;
            }
            self.counts.truncate(len);
            self.totals = saved;
        }
        result
    }

    fn change(
        &mut self,
        dag: &ExpressionDag,
        root: (ExpressionId, AccumulationSide),
        instances: &[u64],
        adding: bool,
        budget: &mut Option<&mut ExtractionBudget>,
        journal: &mut Option<&mut Vec<(ExpressionId, usize, usize)>>,
    ) -> Option<()> {
        let mut pending = vec![root];
        while let Some((id, side)) = pending.pop() {
            if let Some(budget) = budget.as_mut() {
                budget.reserve(if journal.is_some() { 2 } else { 1 }).ok()?;
            }
            let before_live = self.live(id);
            let count = &mut self.counts[id.0][side.index()];
            if let Some(journal) = journal.as_mut() {
                journal.push((id, side.index(), *count));
            }
            if adding {
                *count += 1;
            } else {
                *count -= 1;
            }
            if *count != usize::from(adding) {
                continue;
            }
            let node = dag.node(id);
            let arithmetic = cost::arithmetic(node);
            if arithmetic {
                adjust(
                    &mut self.totals.placement[side.index()],
                    u128::from(instances[id.0]),
                    adding,
                );
            }
            if before_live != self.live(id) {
                if arithmetic {
                    adjust(
                        &mut self.totals.physical,
                        u128::from(instances[id.0]),
                        adding,
                    );
                }
                if adding {
                    self.totals.nodes += 1;
                } else {
                    self.totals.nodes -= 1;
                }
                match node {
                    ExpressionNode::Select(..) => {
                        if adding {
                            self.totals.selects += 1;
                        } else {
                            self.totals.selects -= 1;
                        }
                    }
                    ExpressionNode::Coefficient { values, .. } => {
                        if let Some(budget) = budget.as_mut() {
                            budget.reserve(values.len()).ok()?;
                        }
                        let size = values
                            .iter()
                            .map(|v| v.structural_size().max(1) as u128)
                            .sum();
                        adjust(&mut self.totals.constants, size, adding);
                    }
                    _ => {}
                }
            }
            let child_side = if matches!(node, ExpressionNode::Sum { .. }) {
                AccumulationSide::BeforeAccumulation
            } else {
                side
            };
            pending.extend(node.children().map(|child| (child, child_side)));
        }
        Some(())
    }
}

fn adjust(total: &mut u128, amount: u128, adding: bool) {
    if adding {
        *total += amount;
    } else {
        *total -= amount;
    }
}

#[cfg(test)]
#[path = "usage/tests.rs"]
mod tests;
