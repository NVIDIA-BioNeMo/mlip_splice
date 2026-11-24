// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, Candidate, CompileLimits, CompileOptions, CompileProfile, DomainIndex, Launch,
    LiftedProgram, Placement, Plan, num_elements, output_domain_families,
    plan_single_with_limits_profiled, render_target,
};
#[cfg(test)]
use super::{PlanIr, emit_single_block_reduction_kernel, plan};

pub(super) use xir_optimize::PartialContributionRatio;

pub(super) fn planned_partial_contribution_ratios(
    loom: &LiftedProgram,
    options: CompileOptions,
    settings: &BackendSettings,
) -> Option<Vec<PartialContributionRatio>> {
    if !loom.has_output_factors() {
        return Some(Vec::new());
    }
    let passes = super::arms::compile_passes(loom, options, settings).ok()?;
    let mut ratios = Vec::new();
    for (statement_cap, plan_clusters, element_order) in passes {
        let limits = CompileLimits {
            statement_cap,
            register_cap: settings.schedule.default_register_cap,
            cut_limit: settings.schedule.default_cut_limit,
            plan_clusters,
            max_workspace_bytes: options.max_workspace_bytes,
            element_order,
            target_sm: render_target(options).ok()?,
            dump_ptx: options.keep_ptx,
            record_structure: options.record_dependencies_and_partitions,
            settings,
        };
        let baseline = planned_single_ratios(loom, limits, Placement::Default).ok()??;
        merge_ratios(&mut ratios, baseline);
        let partitions = output_domain_families(loom, plan_clusters).ok().flatten();
        for partition in partitions.into_iter().flatten() {
            let mut families = Vec::with_capacity(partition.len());
            let mut complete = true;
            for (family, _, placement) in partition {
                match planned_single_ratios(&family, limits, placement) {
                    Ok(Some(family)) => families.push(Some(family)),
                    Ok(None) => return None,
                    Err(_) => {
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                merge_ratios(&mut ratios, common_ratios(families)?);
            }
        }
    }
    Some(ratios)
}

fn planned_single_ratios(
    loom: &LiftedProgram,
    limits: CompileLimits<'_>,
    placement: Placement,
) -> Result<Option<Vec<PartialContributionRatio>>, String> {
    let planned = plan_single_with_limits_profiled(
        loom.clone(),
        limits,
        placement,
        &mut CompileProfile::default(),
    )?;
    let ratios = (|| {
        let mut ratios = Vec::new();
        let selected = planned
            .estimates
            .baseline_index
            .into_iter()
            .chain(planned.estimates.alternative_indices.iter().copied());
        for index in selected {
            let Some(ratio) = candidate_partial_contribution_ratio(
                &planned.execution.domain,
                loom,
                &planned.execution.candidates[index],
            )?
            else {
                return Ok(None);
            };
            merge_ratios(&mut ratios, [ratio]);
        }
        Ok::<_, String>(Some(ratios))
    })()?;
    Ok(ratios)
}

pub(super) fn candidate_partial_contribution_ratio(
    program: &Plan,
    loom: &LiftedProgram,
    candidate: &Candidate,
) -> Result<Option<PartialContributionRatio>, String> {
    let batch_positions = u64::try_from(program.index.batch_total()?)
        .map_err(|_| "loom iteration has a negative batch extent")?;
    let mut common = None;
    for (output, binding) in loom.outputs().iter().enumerate() {
        let operand = loom.num_inputs + output;
        let output_positions = u64::try_from(program.index.operand_batch_extent(operand)?)
            .map_err(|_| format!("loom output {output} has a negative batch extent"))?;
        let positions_per_launch = if program.index.fiber_scatter(operand) {
            output_positions
        } else {
            batch_positions
        };
        let elements = num_elements(&binding.element_shape)?;
        for element in 0..elements {
            if loom.output_factors(output, element).is_empty() {
                continue;
            }
            let launches = candidate
                .launches
                .iter()
                .filter(|launch| match launch {
                    Launch::Producer { .. } => false,
                    Launch::Consumer { part, .. } => part.elements[output]
                        .iter()
                        .any(|selection| selection.element == element),
                })
                .count();
            if launches == 0 {
                return Err(format!(
                    "loom output factor {output}:{element} has no writing launch"
                ));
            }
            let ratio = PartialContributionRatio::new(
                positions_per_launch
                    .checked_mul(launches as u64)
                    .ok_or("loom partial-contribution count overflow")?,
                output_positions,
            )?;
            match common {
                None => common = Some(ratio),
                Some(existing) if existing == ratio => {}
                Some(_) => return Ok(None),
            }
        }
    }
    Ok(common)
}

pub(super) fn merge_ratios(
    target: &mut Vec<PartialContributionRatio>,
    ratios: impl IntoIterator<Item = PartialContributionRatio>,
) {
    for ratio in ratios {
        if !target.contains(&ratio) {
            target.push(ratio);
        }
    }
}

pub(super) fn common_ratios(
    families: impl IntoIterator<Item = Option<Vec<PartialContributionRatio>>>,
) -> Option<Vec<PartialContributionRatio>> {
    let mut known = Vec::new();
    for ratios in families {
        let ratios = ratios?;
        if !ratios.is_empty() {
            known.push(ratios);
        }
    }
    let mut families = known.into_iter();
    let Some(mut common) = families.next() else {
        return Some(Vec::new());
    };
    for ratios in families {
        common.retain(|ratio| ratios.contains(ratio));
    }
    Some(common)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatOperand;
    use crate::program::{InputElement, OutputFactor};
    use domain_index::{Domain, IndexMap};
    use xir::builder::ModuleBuilder;
    use xir::dtype::DType as Dtype;
    use xir::dtype::DType;
    use xir::shape::Shape;

    fn projection_program(factor_output: usize) -> LiftedProgram {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("partial_contributions");
        let input = builder.param(Shape::scalar(DType::F32));
        let entry = builder.end_func(vec![input, input]);
        let binding = |domain| FlatOperand {
            dtype: Dtype::F32,
            element_shape: Vec::new(),
            lane_extent: 1,
            axis_domains: vec![domain],
        };
        let mut loom = loom_program! {
            name: "partial_contributions".into(),
            module: builder.finish(entry).unwrap(),
            lane_count: 1,
            domains: vec![
                Domain { extent: 3 },
                Domain { extent: 8 },
            ],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            num_axes: 1,
            operands: vec![binding(1), binding(1), binding(0)],
            num_inputs: 1,
            available_views: Vec::new(),
        };
        let factor = vec![OutputFactor::Input(InputElement {
            operand: 0,
            element: 0,
        })];
        let mut factors = vec![vec![Vec::new()], vec![Vec::new()]];
        factors[factor_output][0] = factor;
        loom.set_output_factors(factors);
        loom
    }

    fn fiber_reduction_program() -> LiftedProgram {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("fiber_partial_contributions");
        let _coefficient = builder.param(Shape::scalar(DType::F32));
        let values = builder.param(Shape::scalar(DType::F32));
        let entry = builder.end_func(vec![values]);
        let binding = |domain| FlatOperand {
            dtype: Dtype::F32,
            element_shape: Vec::new(),
            lane_extent: 1,
            axis_domains: vec![domain],
        };
        let mut loom = loom_program! {
            name: "fiber_partial_contributions".into(),
            module: builder.finish(entry).unwrap(),
            lane_count: 1,
            domains: vec![
                Domain { extent: 3 },
                Domain { extent: 8 },
            ],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            num_axes: 1,
            operands: vec![binding(0), binding(1), binding(0)],
            num_inputs: 2,
            available_views: Vec::new(),
        };
        loom.set_output_factors(vec![vec![vec![OutputFactor::Input(InputElement {
            operand: 0,
            element: 0,
        })]]]);
        loom
    }

    fn baseline(loom: &LiftedProgram, placement: Placement) -> (PlanIr, usize) {
        let domain = Plan::prepare_for(loom, placement).unwrap().unwrap();
        let parts = plan(loom, usize::MAX).unwrap();
        let execution = PlanIr::build(loom, domain, parts, usize::MAX, 0).unwrap();
        let candidate = execution
            .candidates
            .iter()
            .position(|candidate| candidate.materialized.is_empty())
            .unwrap();
        (execution, candidate)
    }

    #[test]
    fn direct_output_has_one_contribution_per_output() {
        let loom = projection_program(0);
        let (execution, candidate) = baseline(&loom, Placement::Default);
        assert_eq!(
            candidate_partial_contribution_ratio(
                &execution.domain,
                &loom,
                &execution.candidates[candidate]
            )
            .unwrap(),
            Some(PartialContributionRatio::new(1, 1).unwrap())
        );
    }

    #[test]
    fn balanced_atomic_output_keeps_the_exact_average_ratio() {
        let loom = projection_program(1);
        let (execution, candidate) = baseline(&loom, Placement::Default);
        assert_eq!(
            candidate_partial_contribution_ratio(
                &execution.domain,
                &loom,
                &execution.candidates[candidate]
            )
            .unwrap(),
            Some(PartialContributionRatio::new(8, 3).unwrap())
        );
        let ratio = PartialContributionRatio::new(8, 3).unwrap();
        assert_eq!(
            ratio.scaled_work(xir_optimize::FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 3,
            }),
            ratio.scaled_work(xir_optimize::FiniteMapWork {
                before_accumulation: 8,
                after_accumulation: 0,
            })
        );
    }

    #[test]
    fn planner_reports_ratios_before_emission() {
        let loom = fiber_reduction_program();
        let options = CompileOptions {
            target: Some(crate::backend::compiler::test_target()),
            ..Default::default()
        };
        let ratios =
            planned_partial_contribution_ratios(&loom, options, &BackendSettings::default());
        assert!(ratios.is_some_and(|ratios| !ratios.is_empty()));
    }

    #[test]
    fn fiber_direct_output_has_one_contribution_per_output() {
        let loom = projection_program(1);
        let (execution, candidate) = baseline(&loom, Placement::FiberDirect(0));
        assert_eq!(
            candidate_partial_contribution_ratio(
                &execution.domain,
                &loom,
                &execution.candidates[candidate]
            )
            .unwrap(),
            Some(PartialContributionRatio::new(1, 1).unwrap())
        );
    }

    #[test]
    fn reduction_description_has_one_contribution_per_output() {
        let loom = fiber_reduction_program();
        let (execution, candidate) = baseline(&loom, Placement::Default);
        assert_eq!(
            candidate_partial_contribution_ratio(
                &execution.domain,
                &loom,
                &execution.candidates[candidate]
            )
            .unwrap(),
            Some(PartialContributionRatio::new(1, 1).unwrap())
        );
    }

    #[test]
    fn single_block_reduction_declines_unapplied_output_factors() {
        let loom = projection_program(1);
        let (execution, candidate) = baseline(&loom, Placement::Default);
        let Launch::Consumer { part, .. } = &execution.candidates[candidate].launches[0] else {
            panic!("baseline must be a consumer");
        };
        assert!(
            emit_single_block_reduction_kernel(&execution.domain, &loom, 128, part)
                .unwrap_err()
                .contains("not eligible")
        );
    }
}
