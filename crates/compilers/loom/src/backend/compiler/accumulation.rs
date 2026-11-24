// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How deep the float additions reaching one output cell go, for one whole
//! compiled plan.
//!
//! `gpu_ir::longest_addition_chain` reads one kernel's arena and answers for one
//! thread: the per-thread strip loop, the warp shuffles and the shared-memory
//! tree are all inside a single launch, so they are exact there. What no arena
//! can see is how many launches of that kernel land on the same cell, because
//! that is the launch geometry's. This module closes the composition:
//!
//! ```text
//! depth(output j) = max over launches i writing j of
//!                       (within_thread(i) + fan_in(i, j))
//!                   + projection(j)
//!                   + max over producers of within_thread
//! ```
//!
//! `fan_in(i, j)` counts the roundings an atomic fold adds. A cell that `k`
//! blocks atomically add into is summed sequentially in an order the hardware
//! picks, and the first contribution lands on an exact zero, so `k - 1`
//! roundings sit above the deepest contributing thread. A plain store folds
//! nothing, so `k = 1`. What bounds `k` is
//! [`super::super::super::domain::reduction::OutputFold::longest_run`]: the
//! positions one bucket copy of the launch receives, and, where the extents fix
//! it, the values one address receives. Bucket copies shorten it by giving each
//! copy a smaller share of the launch's blocks — which is all that holds without
//! knowing how the index map spreads an address over the copies, and less than
//! the `k / b` the copies are chosen for. That gap is deliberate: the figure is
//! an upper bound, so it may overstate a plan and must never flatter one.
//!
//! The producer term is an upper bound rather than a dataflow result: a
//! consumer reads at most every materialized value, so adding the deepest
//! producer chain can only overstate the composition, never understate it.

use super::{Candidate, DomainIndex, Launch, Plan, WriteMode};

/// One output's figure for one plan, with the quantities it is composed from so
/// a reader can audit it rather than trust it.
///
/// `projections[j]` is the chain of the projection that closes output `j`: the
/// outer `None` when the plan needs no projection there, the inner `None` when
/// the analysis found no bound for the one it needs.
#[allow(clippy::too_many_arguments)]
pub(super) fn describe(
    program: &Plan,
    plan: &Candidate,
    outputs: usize,
    chains: &[(usize, Option<gpu_ir::AdditionChain>)],
    specs: &[super::SlotSpec],
    buckets: &[u32],
    projections: &[Option<Option<gpu_ir::AdditionChain>>],
    candidate: usize,
    placement: super::Placement,
) -> Result<Vec<crate::profile::AdditionDepth>, String> {
    let batch_total = program.index.batch_total()?;
    (0..outputs)
        .map(|output| {
            let fold = program.folds.output(output);
            let collisions = fold.collisions();
            let buckets = *buckets
                .get(output)
                .ok_or("loom bucket assignment is missing an output")?;
            let mut blocks_per_bucket = 0;
            let mut within_thread = Some(0);
            let mut across_blocks = 0;
            let mut producers = Some(0);
            for (launch_index, (slot, chain)) in chains.iter().enumerate() {
                let spec = specs
                    .get(*slot)
                    .ok_or("loom launch names a slot the table does not have")?;
                let chain = chain.map(|chain| chain.within_thread);
                match &plan.launches[launch_index] {
                    Launch::Producer { .. } => {
                        producers = producers.zip(chain).map(|(deepest, own)| deepest.max(own));
                    }
                    Launch::Consumer { part, output_modes } => {
                        if !part.writes_output(output) {
                            continue;
                        }
                        let blocks = i64::from(::recipe::count_covering_blocks(
                            batch_total,
                            spec.positions_per_block() as i32,
                        )?) * i64::from(spec.shards);
                        let per_bucket = (blocks + i64::from(buckets) - 1) / i64::from(buckets);
                        blocks_per_bucket = blocks_per_bucket.max(per_bucket);
                        let folded = match output_modes[output] {
                            WriteMode::Store => 1,
                            _ => fold.longest_run(
                                per_bucket.saturating_mul(i64::from(spec.positions_per_block())),
                            ),
                        };
                        let fan_in = u64::try_from(folded - 1).unwrap_or(0);
                        let reached = chain.map(|own| own + fan_in);
                        within_thread = within_thread
                            .zip(reached)
                            .map(|(deepest, own)| deepest.max(own));
                        across_blocks = across_blocks.max(fan_in);
                    }
                }
            }
            let closing = *projections
                .get(output)
                .ok_or("loom plan is missing an output's projection")?;
            // No projection contributes nothing; one whose chain has no bound
            // leaves the whole figure unbounded.
            // A projection sums this output's `buckets` copies, so it adds at
            // most `buckets - 1` roundings whatever its arena says; the analysis
            // refines that when it can read the fold, and cannot exceed it.
            let ceiling = u64::from(buckets.saturating_sub(1));
            let projection = closing.map(|chain| {
                chain
                    .map(|chain| chain.within_thread.min(ceiling))
                    .unwrap_or(ceiling)
            });
            Ok(crate::profile::AdditionDepth {
                graph: 0,
                placement: placement.name(),
                arm: None,
                candidate,
                output,
                within_thread,
                across_blocks,
                projection,
                total: within_thread
                    .zip(producers)
                    .map(|(deepest, producers)| deepest + producers + projection.unwrap_or(0)),
                blocks_per_bucket,
                collisions,
                buckets,
            })
        })
        .collect()
}
