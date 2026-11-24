// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::name::{shared_bucket_fold, shared_convert};
use super::super::schedule;
use super::{
    CompileLimits, CompileProfile, LaunchEstimates, LiftedProgram, Placement, Plan, PlanIr,
    PlannedSingle, ProfilePhase, program_signature, reduction,
};
use crate::domain::reduction::Projection;

/// What one compiled arm does to each output's fold. A baseline arm reports
/// the bucket ladder it could take; an alternative plan admits only its direct
/// count. `arm` is stamped when the arm's profile joins the compiled choice.
pub(super) fn describe_fold_candidates(
    program: &Plan,
    loom: &LiftedProgram,
    buckets: &[u32],
    tile_widths: &[Option<u32>],
    uses_bucket_ladder: bool,
    placement: Placement,
) -> Result<Vec<crate::profile::FoldBuckets>, String> {
    (0..loom.outputs().len())
        .map(|output| {
            let fold = program.folds.output(output);
            let buckets = buckets[output];
            let elements = reduction::output_elements(&program.index, loom, output)?;
            let projection = fold.projection(buckets);
            Ok(crate::profile::FoldBuckets {
                graph: 0,
                placement: placement.name(),
                arm: 0,
                output,
                key: fold.key().name(),
                collisions: fold.collisions(),
                buckets,
                admitted: if uses_bucket_ladder {
                    fold.bucket_counts().to_vec()
                } else {
                    vec![buckets]
                },
                cap: fold.bucket_cap().name(),
                cap_bound: fold.bucket_cap().bound(),
                elements,
                workspace_bytes: match projection {
                    Some(_) => fold.accumulation_nbytes(elements, buckets)?,
                    None => 0,
                },
                projection: projection.map_or("none", Projection::name),
                tile_widths: tile_widths[output].into_iter().collect(),
                entry: match projection {
                    Some(Projection::Convert) => shared_convert(&loom.name, output),
                    Some(Projection::Fold { .. }) => shared_bucket_fold(&loom.name, output),
                    None => String::new(),
                },
            })
        })
        .collect()
}

/// An indexed program plans its launches like any other. What it cannot do is
/// carry the cluster regroupings: those price a launch by the operand bytes it
/// gathers, and `IncidenceProfile::input_operand` deliberately has no operand
/// for the base pointer an indexed program reads its inputs through, so a
/// byte-weighted plan over an indexed profile would be a plan over a hole.
fn indexed_limits<'a>(loom: &LiftedProgram, mut limits: CompileLimits<'a>) -> CompileLimits<'a> {
    if loom.has_indexed_ops() {
        limits.plan_clusters = false;
    }
    limits
}

/// Records what each binding of `loom` reads, once per program.
///
/// It sits beside the partition records rather than at a compile entry because
/// the two must cover the same programs: an output-family split is a program of
/// its own, planned here and nowhere else, and a boundary whose dependency sets
/// were never recorded is a boundary a later representation cannot be compared
/// against. A dependency set belongs to the program, not to the arm that
/// planned it, so the arms after the first add nothing — and the membership
/// test comes first because it is far cheaper than the expansion it guards.
fn record_definitions(
    loom: &LiftedProgram,
    recorded: &mut Vec<crate::profile::BindingDependencies>,
) -> Result<(), String> {
    if recorded.iter().any(|record| record.program == loom.name) {
        return Ok(());
    }
    recorded.extend(
        super::super::partition::definition_dependencies(loom)?
            .into_iter()
            .enumerate()
            .map(
                |(definition, dependencies)| crate::profile::BindingDependencies {
                    graph: 0,
                    program: loom.name.clone(),
                    binding: definition,
                    dependencies,
                },
            ),
    );
    Ok(())
}

pub(super) fn plan_single_with_limits_profiled(
    loom: LiftedProgram,
    limits: CompileLimits,
    placement: Placement,
    profile: &mut CompileProfile,
) -> Result<PlannedSingle, String> {
    let limits = indexed_limits(&loom, limits);
    let phase = ProfilePhase::new("plan_ms", &mut profile.timings.plan);
    let program = match placement {
        Placement::FiberDirect(_) => Plan::prepare_for(&loom, placement)?
            .ok_or("fiber-direct iteration does not move the placement off the default")?,
        Placement::Default => Plan::prepare(&loom)?,
    };
    let program = Plan {
        element_order: limits.element_order,
        ..program
    };
    let signature = program_signature(&program, &loom, &program.slots)?;
    let incidence = std::cell::OnceCell::new();
    let execution = PlanIr::build_bounded_with_settings(
        &loom,
        program,
        schedule::Limits {
            statement_cap: limits.statement_cap,
            register_cap: limits.register_cap,
            cut_limit: limits.cut_limit,
            plan_clusters: limits.plan_clusters,
        },
        &incidence,
        limits.settings,
    )?;
    if limits.record_structure {
        record_definitions(&loom, &mut profile.binding_dependencies)?;
        profile
            .partition_parts
            .extend(
                execution
                    .boundaries
                    .iter()
                    .enumerate()
                    .map(|(part, boundary)| crate::profile::PartitionPart {
                        graph: 0,
                        program: loom.name.clone(),
                        order: limits.element_order.name(),
                        statement_cap: limits.statement_cap,
                        plan_clusters: limits.plan_clusters,
                        placement: placement.name(),
                        part,
                        bisections: boundary.bisections,
                        oversized: boundary.oversized,
                        slices: boundary.slices.clone(),
                    }),
            );
    }
    let estimates = LaunchEstimates::measure(
        &loom,
        &execution,
        limits.settings.compiler.block_sizes[0],
        limits.statement_cap,
        &incidence,
        limits.settings,
    )?;
    drop(phase);
    Ok(PlannedSingle {
        loom,
        execution,
        estimates,
        incidence,
        signature,
    })
}
