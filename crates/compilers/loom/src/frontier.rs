// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical capability filtering for the exact algebraic graph frontier.

use rayon::prelude::*;

use super::{CompileOptions, FlatProgram, LiftedProgram};
use crate::program::OutputFactor;

struct LowerableAlternative {
    program: LiftedProgram,
    module: String,
    factors: Vec<Vec<Vec<OutputFactor>>>,
}

fn work_dominates(left: xir_optimize::FiniteMapWork, right: xir_optimize::FiniteMapWork) -> bool {
    left != right
        && left.before_accumulation <= right.before_accumulation
        && left.after_accumulation <= right.after_accumulation
}

fn retain_distinct_pareto_lowerings(alternatives: &mut Vec<LowerableAlternative>) {
    let retained = alternatives
        .iter()
        .enumerate()
        .map(|(index, alternative)| {
            !alternatives.iter().enumerate().any(|(other_index, other)| {
                work_dominates(
                    other.program.finite_map_work(),
                    alternative.program.finite_map_work(),
                ) || other_index < index
                    && other.program.finite_map_work() == alternative.program.finite_map_work()
                    && other.module == alternative.module
                    && other.factors == alternative.factors
            })
        })
        .collect::<Vec<_>>();
    let mut index = 0;
    alternatives.retain(|_| {
        let retain = retained[index];
        index += 1;
        retain
    });
}

fn prepare_request(request: FlatProgram) -> Result<LiftedProgram, String> {
    let mut program = LiftedProgram::try_from(request)?;
    super::expand_regions(&mut program.request_mut().module)?;
    let lifted = super::lift::lift_over_domains(&program)?;
    program.request_mut().module = lifted;
    super::scalarize::run(&mut program.request_mut().module)?;
    Ok(program)
}

pub(super) struct ProgramFrontier {
    program: LiftedProgram,
    alternatives: xir_optimize::AlgebraicFrontier,
    options: CompileOptions,
}

impl ProgramFrontier {
    pub(super) fn extract(
        request: FlatProgram,
        options: CompileOptions,
        check: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(Self, xir_optimize::PipelineReport), String> {
        check()?;
        let program = prepare_request(request)?;
        check()?;
        let (alternatives, report) = xir_optimize::describe_algebraic_frontier_with_limits(
            &program.module,
            options.max_exact_work.get(),
            options.max_exact_work_per_expression.get(),
            check,
        )
        .map_err(|error| format!("loom XIR optimization: {error}"))?;
        Ok((
            Self {
                program,
                alternatives,
                options,
            },
            report,
        ))
    }

    pub(super) fn len(&self) -> usize {
        self.alternatives.len()
    }

    pub(super) fn next(
        &mut self,
        check: &mut impl FnMut() -> Result<(), String>,
    ) -> Option<Result<(LiftedProgram, xir_optimize::PipelineReport), String>> {
        if let Err(error) = check() {
            return Some(Err(error));
        }
        let alternative = self.alternatives.next(check)?;
        Some((|| {
            let (alternative, extraction_report) =
                alternative.map_err(|error| error.to_string())?;
            let mut program = self.program.clone();
            program.request_mut().module = alternative.module;
            let factors = super::lift::lower_domain_maps(&mut program.request_mut().module)?;
            program.set_output_factors(factors);
            program.set_finite_map_work(alternative.work);
            check()?;
            let mut report =
                super::optimize_module(&mut program.request_mut().module, self.options)?;
            report.algebraic_work += extraction_report.algebraic_work;
            report.abandoned_algebraic_expressions +=
                extraction_report.abandoned_algebraic_expressions;
            check()?;
            super::recover_map_reduce_regions(&mut program.request_mut().module)?;
            Ok((program, report))
        })())
    }
}

pub(super) fn lower_program_frontier_with_options(
    request: FlatProgram,
    options: CompileOptions,
) -> Result<(Vec<LiftedProgram>, usize, usize), String> {
    let program = prepare_request(request)?;
    let (frontier, initial_report) = xir_optimize::extract_algebraic_frontier_with_limits(
        &program.module,
        options.max_exact_work.get(),
        options.max_exact_work_per_expression.get(),
    )
    .map_err(|error| format!("loom XIR optimization: {error}"))?;
    let lowered = frontier
        .into_par_iter()
        .map(|alternative| -> Result<Result<_, String>, String> {
            let mut candidate = program.clone();
            candidate.request_mut().module = alternative.module;
            if let Err(unsupported) =
                super::lift::validate_domain_map_lowering(&candidate.request_mut().module)
            {
                return Ok(Err(unsupported));
            }
            let factors = super::lift::lower_domain_maps(&mut candidate.request_mut().module)?;
            candidate.set_output_factors(factors.clone());
            candidate.set_finite_map_work(alternative.work);
            Ok(Ok(LowerableAlternative {
                module: candidate.module.to_string(),
                program: candidate,
                factors,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut first_unsupported = None;
    let mut candidates = lowered
        .into_iter()
        .filter_map(|lowered| match lowered {
            Ok(candidate) => Some(candidate),
            Err(unsupported) => {
                first_unsupported.get_or_insert(unsupported);
                None
            }
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(first_unsupported
            .unwrap_or_else(|| "loom algebraic lowering produced no XIR alternative".into()));
    }
    retain_distinct_pareto_lowerings(&mut candidates);
    let candidates = candidates
        .into_par_iter()
        .map(|mut candidate| {
            let report =
                super::optimize_module(&mut candidate.program.request_mut().module, options)?;
            super::recover_map_reduce_regions(&mut candidate.program.request_mut().module)?;
            Ok((candidate.program, report))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let work = initial_report.algebraic_work
        + candidates
            .iter()
            .map(|(_, report)| report.algebraic_work)
            .sum::<usize>();
    let abandoned = initial_report.abandoned_algebraic_expressions
        + candidates
            .iter()
            .map(|(_, report)| report.abandoned_algebraic_expressions)
            .sum::<usize>();
    Ok((
        candidates
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect(),
        work,
        abandoned,
    ))
}

#[cfg(test)]
mod tests {
    use domain_index::Domain;
    use xir::builder::ModuleBuilder;
    use xir::dtype::DType;
    use xir::shape::Shape;

    use super::*;
    use crate::FlatOperand;

    fn alternative(name: &str, before: u64, after: u64) -> LowerableAlternative {
        let mut builder = ModuleBuilder::new();
        builder.begin_func(name);
        let value = builder.param(Shape::scalar(DType::F32));
        let entry = builder.end_func(vec![value]);
        let module = builder.finish(entry).unwrap();
        let binding = FlatOperand {
            dtype: DType::F32,
            element_shape: Vec::new(),
            lane_extent: 1,
            axis_domains: Vec::new(),
        };
        let mut program = LiftedProgram::try_from(FlatProgram {
            module,
            lane_count: 1,
            domains: Vec::<Domain>::new(),
            index_maps: Vec::new(),
            num_axes: 0,
            operands: vec![binding.clone(), binding],
            num_inputs: 1,
            available_views: Vec::new(),
            name: name.into(),
        })
        .unwrap();
        program.set_finite_map_work(xir_optimize::FiniteMapWork {
            before_accumulation: before,
            after_accumulation: after,
        });
        LowerableAlternative {
            module: program.module.to_string(),
            program,
            factors: Vec::new(),
        }
    }

    #[test]
    fn capability_frontier_deduplicates_then_removes_dominated_lowerings() {
        let mut alternatives = vec![
            alternative("after", 0, 3),
            alternative("after", 0, 3),
            alternative("before", 8, 0),
            alternative("dominated", 9, 4),
        ];

        retain_distinct_pareto_lowerings(&mut alternatives);

        assert_eq!(alternatives.len(), 2);
        assert_eq!(
            alternatives[0].program.finite_map_work().after_accumulation,
            3
        );
        assert_eq!(
            alternatives[1]
                .program
                .finite_map_work()
                .before_accumulation,
            8
        );
    }
}
