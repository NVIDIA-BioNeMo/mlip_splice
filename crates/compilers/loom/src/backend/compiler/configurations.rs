// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{LaunchEstimates, Render, product::Product};
use crate::domain::reduction::Projection;
use std::{collections::VecDeque, ops::Range, sync::Arc};

/// All choices for one leaf, before any of its PTX is materialized.
#[derive(Clone, Debug)]
pub(super) enum Configuration {
    Launches {
        candidate: usize,
        slots: Vec<usize>,
        buckets: Vec<u32>,
        tile_widths: Vec<Option<u32>>,
    },
    SingleBlockReduction(u32),
}

#[derive(Clone)]
struct LaunchConfigurations {
    candidate: usize,
    slots: Arc<Vec<Vec<usize>>>,
    buckets: Arc<Vec<u32>>,
    tile_widths: Arc<Vec<Vec<Option<u32>>>>,
    selections: Product<Range<usize>>,
}

#[derive(Clone)]
pub(super) struct Configurations {
    launches: VecDeque<LaunchConfigurations>,
    reductions: std::vec::IntoIter<u32>,
    reduction_turn: bool,
}

impl Configurations {
    pub(super) fn new(render: &Render<'_>, estimates: &LaunchEstimates) -> Result<Self, String> {
        let mut launches = VecDeque::new();
        for candidate in estimates
            .baseline_index
            .into_iter()
            .chain(estimates.alternative_indices.iter().copied())
        {
            let slots = Arc::new(
                (0..render.execution.candidates[candidate].launches.len())
                    .map(|launch| render.describe_slots(estimates, candidate, launch))
                    .collect::<Vec<_>>(),
            );
            let bucket_arms = if estimates.baseline_index == Some(candidate) {
                render.execution.domain.folds.bucket_arms()
            } else {
                vec![vec![1; render.loom.outputs().len()]]
            };
            for buckets in bucket_arms {
                let tile_widths = buckets
                    .iter()
                    .enumerate()
                    .map(|(output, &buckets)| {
                        let projection = render
                            .execution
                            .domain
                            .folds
                            .output(output)
                            .projection(buckets);
                        Ok(match projection {
                            Some(projection @ Projection::Fold { .. }) => {
                                let elements = super::reduction::output_elements(
                                    &render.execution.domain.index,
                                    render.loom,
                                    output,
                                )?;
                                projection
                                    .tile_widths(elements)
                                    .into_iter()
                                    .map(Some)
                                    .collect()
                            }
                            _ => vec![None],
                        })
                    })
                    .collect::<Result<Vec<Vec<Option<u32>>>, String>>()?;
                let axes = slots
                    .iter()
                    .map(|slots| 0..slots.len())
                    .chain(tile_widths.iter().map(|widths| 0..widths.len()))
                    .collect();
                launches.push_back(LaunchConfigurations {
                    candidate,
                    slots: slots.clone(),
                    buckets: Arc::new(buckets),
                    tile_widths: Arc::new(tile_widths),
                    selections: Product::new(axes),
                });
            }
        }
        Ok(Self {
            launches,
            reductions: if estimates.baseline_index.is_some() {
                render.settings.compiler.reduction_block_sizes.to_vec()
            } else {
                vec![]
            }
            .into_iter(),
            reduction_turn: false,
        })
    }
}

impl Iterator for Configurations {
    type Item = Configuration;
    fn next(&mut self) -> Option<Self::Item> {
        if self.reduction_turn {
            self.reduction_turn = false;
            if let Some(block) = self.reductions.next() {
                return Some(Configuration::SingleBlockReduction(block));
            }
        }
        while let Some(mut launch) = self.launches.pop_front() {
            if let Some(indices) = launch.selections.next() {
                let (slots, widths) = indices.split_at(launch.slots.len());
                let configuration = Configuration::Launches {
                    candidate: launch.candidate,
                    slots: launch
                        .slots
                        .iter()
                        .zip(slots)
                        .map(|(slots, &index)| slots[index])
                        .collect(),
                    buckets: launch.buckets.as_ref().clone(),
                    tile_widths: launch
                        .tile_widths
                        .iter()
                        .zip(widths)
                        .map(|(widths, &index)| widths[index])
                        .collect(),
                };
                self.launches.push_back(launch);
                self.reduction_turn = true;
                return Some(configuration);
            }
        }
        self.reductions
            .next()
            .map(Configuration::SingleBlockReduction)
    }
}
