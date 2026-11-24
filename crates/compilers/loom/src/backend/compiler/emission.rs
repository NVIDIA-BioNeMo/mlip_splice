// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::schedule::MaterializedValue;
#[cfg(test)]
use super::{BLOCK_SIZES, emit_kernel, plan};
use super::{
    Candidate, CandidateFlavor, Flavor, IncidenceProfile, KernelPart, Launch, LiftedProgram, Plan,
    SlotSpec, Split, WriteMode, emit_consumer_kernel, emit_consumer_kernel_flavor,
    emit_producer_kernel, emit_single_block_reduction_kernel, emit_split_kernel, statement_count,
};

#[cfg(test)]
pub(super) fn bounded_parts(
    program: &Plan,
    loom: &LiftedProgram,
    statement_cap: usize,
) -> Result<Vec<KernelPart>, String> {
    fn refine(
        program: &Plan,
        loom: &LiftedProgram,
        part: KernelPart,
        statement_cap: usize,
    ) -> Result<Vec<KernelPart>, String> {
        let probe = emit_kernel(program, loom, BLOCK_SIZES[0], 0, 2, &part)?;
        let statements = statement_count(probe.body());
        if statements <= statement_cap {
            return Ok(vec![part]);
        }
        if part.len() == 1 {
            return Err(format!(
                "loom output element needs {statements} statements, exceeding cap {statement_cap}"
            ));
        }
        let (left, right) = part.split();
        let mut refined = refine(program, loom, left, statement_cap)?;
        refined.extend(refine(program, loom, right, statement_cap)?);
        Ok(refined)
    }

    let mut bounded = Vec::new();
    for part in plan(loom, statement_cap)? {
        bounded.extend(refine(program, loom, part, statement_cap)?);
    }
    Ok(bounded)
}

/// A consumer launch and where it sits among its siblings: the consumer index
/// and count name its kernel, and the producers that ran before it fix the
/// materialized prefix it reads.
struct Consumer<'a> {
    part: &'a KernelPart,
    output_modes: &'a [WriteMode],
    materialized: &'a [MaterializedValue],
    index: usize,
    count: usize,
}

impl<'a> Consumer<'a> {
    /// `producer_error` is what this emitter reports when the launch turns out
    /// to be a producer, which no consumer flavor can express.
    fn at(
        candidate: &'a Candidate,
        launch_index: usize,
        producer_error: &str,
    ) -> Result<Self, String> {
        let Launch::Consumer { part, output_modes } = &candidate.launches[launch_index] else {
            return Err(producer_error.into());
        };
        let produced = candidate.launches[..launch_index]
            .iter()
            .filter(|launch| matches!(launch, Launch::Producer { .. }))
            .count();
        Ok(Self {
            part,
            output_modes,
            materialized: &candidate.materialized[..produced],
            // Every launch is a producer or a consumer, so the consumers before
            // this one are the launches before it that are not producers.
            index: launch_index - produced,
            count: candidate
                .launches
                .iter()
                .filter(|launch| matches!(launch, Launch::Consumer { .. }))
                .count(),
        })
    }
}

pub(in super::super) fn emit_planned_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    block_size: u32,
) -> Result<gpu_ir::Launch, String> {
    if let Launch::Producer { output, inputs } = &candidate.launches[launch_index] {
        return emit_producer_kernel(
            program,
            loom,
            block_size,
            *output,
            inputs,
            &candidate.materialized,
        );
    }
    let site = Consumer::at(candidate, launch_index, "unreachable producer launch")?;
    emit_consumer_kernel(
        program,
        loom,
        block_size,
        site.index,
        site.count,
        site.part,
        site.materialized,
    )
}

pub(super) fn emit_planned_zero_skip_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    spec: &SlotSpec,
) -> Result<gpu_ir::Launch, String> {
    let guard = candidate.guards[launch_index]
        .as_ref()
        .ok_or("loom launch has no guarded region")?;
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch cannot use zero-skip",
    )?;
    emit_consumer_kernel_flavor(
        program,
        loom,
        site.part,
        Flavor {
            materialized: site.materialized,
            guard: Some(guard),
            io_lanes: spec.io_lanes,
            ..Flavor::plain(spec.block_size, site.index, site.count)
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn emit_planned_zero_skip_geometry_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    spec: &SlotSpec,
) -> Result<gpu_ir::Launch, String> {
    let guard = candidate.guards[launch_index]
        .as_ref()
        .ok_or("loom launch has no guarded region")?;
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch cannot use zero-skip",
    )?;
    if !candidate.materialized.is_empty() {
        return Err("loom materialized launch keeps the scalar zero-skip geometry".into());
    }
    emit_consumer_kernel_flavor(
        program,
        loom,
        site.part,
        Flavor {
            guard: Some(guard),
            warps_per_position: spec.warps,
            warp_u: spec.warp_u,
            ..Flavor::plain(spec.block_size, site.index, site.count)
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn emit_planned_single_block_reduction_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    block_size: u32,
) -> Result<gpu_ir::Launch, String> {
    if !candidate.materialized.is_empty() || candidate.launches.len() != 1 {
        return Err("loom single-block reduction requires one consumer launch".into());
    }
    let Launch::Consumer { part, output_modes } = &candidate.launches[0] else {
        return Err("loom single-block reduction cannot use a producer launch".into());
    };
    if output_modes
        .iter()
        .any(|mode| *mode != WriteMode::AddAtomic)
    {
        return Err("loom single-block reduction requires atomic output reductions".into());
    }
    emit_single_block_reduction_kernel(program, loom, block_size, part)
}

// A launch with an optional zero-skip guard still offers unguarded geometry
// variants, exactly like the unguarded baselines it competes with.
pub(super) fn emit_planned_geometry_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    spec: &SlotSpec,
) -> Result<gpu_ir::Launch, String> {
    let fiber_lanes = spec.fiber_lanes;
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch keeps the scalar baseline geometry",
    )?;
    if !candidate.materialized.is_empty() {
        return Err("loom materialized launch keeps the scalar baseline geometry".into());
    }
    if fiber_lanes
        && site
            .part
            .elements
            .iter()
            .enumerate()
            .any(|(output, elements)| {
                !elements.is_empty() && site.output_modes[output] != WriteMode::Store
            })
    {
        return Err("loom fiber-lane reduction requires direct output stores".into());
    }
    let plain = Flavor::plain(spec.block_size, site.index, site.count);
    emit_consumer_kernel_flavor(
        program,
        loom,
        site.part,
        if fiber_lanes {
            Flavor {
                fiber_lanes: true,
                ..plain
            }
        } else {
            Flavor {
                io_lanes: spec.io_lanes,
                warps_per_position: spec.warps,
                warp_u: spec.warp_u,
                ..plain
            }
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn emit_planned_sharded_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    spec: &SlotSpec,
    profile: &IncidenceProfile,
    settings: &super::super::partition::Settings,
) -> Result<gpu_ir::Launch, String> {
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch keeps the scalar baseline geometry",
    )?;
    if !candidate.materialized.is_empty() {
        return Err("loom materialized launch keeps the scalar baseline geometry".into());
    }
    emit_split_kernel(
        program,
        loom,
        site.part,
        Flavor {
            io_lanes: spec.io_lanes,
            warps_per_position: spec.warps,
            ..Flavor::plain(spec.block_size, site.index, site.count)
        },
        Split::Shards(spec.shards),
        profile,
        settings,
    )
}

pub(super) fn emit_planned_passes_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    spec: &SlotSpec,
    profile: &IncidenceProfile,
    settings: &super::super::partition::Settings,
) -> Result<gpu_ir::Launch, String> {
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch keeps the scalar baseline geometry",
    )?;
    if !candidate.materialized.is_empty() {
        return Err("loom materialized launch keeps the scalar baseline geometry".into());
    }
    emit_split_kernel(
        program,
        loom,
        site.part,
        Flavor {
            io_lanes: spec.io_lanes,
            warps_per_position: spec.warps,
            ..Flavor::plain(spec.block_size, site.index, site.count)
        },
        Split::Passes(spec.passes),
        profile,
        settings,
    )
}

pub(super) fn emit_planned_lane_hoisted_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    launch_index: usize,
    block_size: u32,
) -> Result<gpu_ir::Launch, String> {
    if !candidate.hoists[launch_index] {
        return Err("loom launch has no lane-hoist candidate".into());
    }
    let site = Consumer::at(
        candidate,
        launch_index,
        "loom producer launch cannot use lane hoisting",
    )?;
    emit_consumer_kernel_flavor(
        program,
        loom,
        site.part,
        Flavor {
            materialized: site.materialized,
            lane_hoist: true,
            ..Flavor::plain(block_size, site.index, site.count)
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn candidate_statement_counts(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    probe_block_size: u32,
) -> Result<Vec<usize>, String> {
    candidate
        .launches
        .iter()
        .enumerate()
        .map(|(launch, _)| {
            emit_planned_kernel(program, loom, candidate, launch, probe_block_size)
                .map(|kernel| statement_count(kernel.body()))
        })
        .collect()
}

pub(super) fn zero_skip_statement_counts(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    probe_block_size: u32,
) -> Result<Vec<Option<usize>>, String> {
    candidate
        .guards
        .iter()
        .enumerate()
        .map(|(launch, guard)| {
            guard
                .as_ref()
                .map(|_| {
                    emit_planned_zero_skip_kernel(
                        program,
                        loom,
                        candidate,
                        launch,
                        &SlotSpec::plain(CandidateFlavor::ZeroSkip, probe_block_size),
                    )
                    .map(|kernel| statement_count(kernel.body()))
                })
                .transpose()
        })
        .collect()
}

pub(super) fn lane_hoist_statement_counts(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    probe_block_size: u32,
) -> Vec<Option<usize>> {
    candidate
        .hoists
        .iter()
        .enumerate()
        .map(|(launch, &hoist)| {
            hoist
                .then(|| {
                    emit_planned_lane_hoisted_kernel(
                        program,
                        loom,
                        candidate,
                        launch,
                        probe_block_size,
                    )
                    .map(|kernel| statement_count(kernel.body()))
                })
                .transpose()
                .ok()
                .flatten()
        })
        .collect()
}
