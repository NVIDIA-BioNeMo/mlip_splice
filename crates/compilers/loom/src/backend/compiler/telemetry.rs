// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    CandidateFlavor, CandidateOrigin, CompileLimits, DomainIndex, Kernel, Launch, LaunchEstimates,
    Placement, Render, consumer_parts, partition_cost, statement_count,
};

pub(super) fn describe_planned_candidates(
    render: &Render<'_>,
    estimates: &LaunchEstimates,
    limits: CompileLimits,
    placement: Placement,
) -> Vec<crate::profile::PlannedCandidate> {
    let incidence = render
        .incidence
        .get()
        .and_then(|incidence| incidence.as_ref().ok());
    let selected = estimates
        .baseline_index
        .map(|candidate| (candidate, None, estimates.baseline.as_slice()))
        .into_iter()
        .chain(estimates.alternative_indices.iter().enumerate().map(
            |(alternative, &candidate)| {
                (
                    candidate,
                    Some(alternative),
                    estimates.alternatives[alternative].as_slice(),
                )
            },
        ));
    selected
        .map(|(candidate_index, alternative, statements)| {
            let candidate = &render.execution.candidates[candidate_index];
            let mut arrangements = std::collections::BTreeSet::new();
            let mut geometry_arrangements = std::collections::BTreeSet::new();
            let mut zero_skip_arrangements = std::collections::BTreeSet::new();
            let mut lane_hoist_arrangements = std::collections::BTreeSet::new();
            let mut register_capped_arrangements = std::collections::BTreeSet::new();
            let mut min_positions_per_block = usize::MAX;
            let mut max_positions_per_block = 0;
            let mut max_io_lanes = 0;
            let mut max_warps = 0;
            let mut max_shards = 0;
            let mut max_passes = 0;
            let selected_slots = (0..candidate.launches.len())
                .flat_map(|launch| render.describe_slots(estimates, candidate_index, launch))
                .collect::<std::collections::BTreeSet<_>>();
            for slot in selected_slots {
                let spec = render.table.specs[slot];
                arrangements.insert(slot);
                if spec.flavor == CandidateFlavor::Geometry {
                    geometry_arrangements.insert(slot);
                }
                if spec.flavor == CandidateFlavor::ZeroSkip {
                    zero_skip_arrangements.insert(slot);
                }
                if spec.flavor == CandidateFlavor::LaneHoist {
                    lane_hoist_arrangements.insert(slot);
                }
                if spec.register_cap.is_some() {
                    register_capped_arrangements.insert(slot);
                }
                let positions = spec.positions_per_block() as usize;
                min_positions_per_block = min_positions_per_block.min(positions);
                max_positions_per_block = max_positions_per_block.max(positions);
                max_io_lanes = max_io_lanes.max(spec.io_lanes as usize);
                max_warps = max_warps.max(spec.warps as usize);
                max_shards = max_shards.max(spec.shards as usize);
                max_passes = max_passes.max(spec.passes as usize);
            }
            let cost = incidence.and_then(|incidence| {
                partition_cost(
                    incidence,
                    &consumer_parts(candidate),
                    &limits.settings.partition,
                )
                .ok()
            });
            let producer_launches = candidate
                .launches
                .iter()
                .filter(|launch| matches!(launch, Launch::Producer { .. }))
                .count();
            crate::profile::PlannedCandidate {
                graph: 0,
                order: limits.element_order.name(),
                statement_cap: limits.statement_cap,
                plan_clusters: limits.plan_clusters,
                placement: placement.name(),
                candidate: candidate_index,
                role: if alternative.is_some() {
                    "alternative"
                } else {
                    "baseline"
                },
                origin: match candidate.origin {
                    CandidateOrigin::Structural => "structural",
                    CandidateOrigin::Traffic => "traffic",
                },
                producer_launches,
                consumer_launches: candidate.launches.len() - producer_launches,
                total_statements: statements.iter().sum(),
                max_statements: statements.iter().copied().max().unwrap_or(0),
                materialized_bytes: candidate
                    .materialized
                    .iter()
                    .map(|value| value.nbytes)
                    .sum(),
                max_closure_statements: cost.map(|cost| cost.max_closure_statements),
                max_weighted_footprint: cost.map(|cost| cost.max_weighted_footprint),
                duplicated_closure_statements: cost.map(|cost| cost.duplicated_closure_statements),
                duplicated_input_loads: cost.map(|cost| cost.duplicated_input_loads),
                distinct_input_loads: cost.map(|cost| cost.distinct_input_loads),
                described_arrangements: arrangements.len(),
                geometry_arrangements: geometry_arrangements.len(),
                zero_skip_arrangements: zero_skip_arrangements.len(),
                lane_hoist_arrangements: lane_hoist_arrangements.len(),
                register_capped_arrangements: register_capped_arrangements.len(),
                min_positions_per_block: if min_positions_per_block != usize::MAX {
                    min_positions_per_block
                } else {
                    0
                },
                max_positions_per_block,
                max_io_lanes,
                max_warps,
                max_shards,
                max_passes,
                fiber_bridge: render.execution.domain.index.fiber_bridge().is_some(),
            }
        })
        .collect()
}

pub(super) fn kernel_unit_record(
    projections: &[gpu_ir::Launch],
    source: &[u8],
    entry: &str,
    dump_ptx: bool,
    placement: Placement,
    flavor: &'static str,
) -> Kernel {
    Kernel {
        graph: 0,
        order: "",
        statement_cap: 0,
        placement: placement.name(),
        entry: entry.to_owned(),
        flavor,
        part: 0,
        slot: 0,
        register_cap: None,
        assembly_mode: "relocatable",
        statements: projections
            .iter()
            .map(|kernel| statement_count(kernel.body()))
            .sum(),
        addition_chain_within_thread: projections
            .iter()
            .map(|kernel| {
                gpu_ir::longest_addition_chain(kernel.body()).map(|chain| chain.within_thread)
            })
            .try_fold(0, |deepest: u64, chain| Some(deepest.max(chain?))),
        addition_chain_folds_across_threads: projections.iter().any(|kernel| {
            gpu_ir::longest_addition_chain(kernel.body())
                .is_some_and(|chain| chain.folds_across_threads)
        }),
        ptx_bytes: source.len(),
        ptx_hash: crate::digest(source),
        ptx: dump_ptx.then(|| String::from_utf8_lossy(source).into_owned()),
        admitted: true,
    }
}
