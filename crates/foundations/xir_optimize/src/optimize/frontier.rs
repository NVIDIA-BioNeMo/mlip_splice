// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::map_reduce::rewrite::{FiniteMapFrontier, describe_finite_map_frontier};
use super::*;

/// Descriptions share bounded per-function IR. Complete modules are reconstructed
/// and cleaned only as requested. Extraction within one function is cooperative
/// only at its boundary; this is not hard preemption of the algebraic optimizer.
pub struct AlgebraicFrontier {
    descriptions: FiniteMapFrontier,
    work: AlgebraicWorkBudget,
    indexed_algebra_abandoned: bool,
}

fn checkpoint(check: &mut impl FnMut() -> Result<(), String>) -> Result<(), OptimizeError> {
    check().map_err(|message| OptimizeError {
        pass: "source-checkpoint",
        message,
    })
}

pub fn describe_algebraic_frontier_with_limits(
    module: &Module,
    max_algebraic_work: usize,
    max_algebraic_expression_work: usize,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<(AlgebraicFrontier, PipelineReport), OptimizeError> {
    checkpoint(check)?;
    let mut prepared = module.clone();
    let mut work = AlgebraicWorkBudget::new(max_algebraic_work, max_algebraic_expression_work);
    let mut report = PassManager::new([
        Pass::AlgebraicCanonicalization,
        Pass::CanonicalizeShapes,
        Pass::ConstantFolding,
        Pass::AlgebraicCanonicalization,
        Pass::ScheduleExactPolynomialRoots,
    ])
    .run_with_budget_checked(&mut prepared, &mut work, check)?;
    let abandonment_before_indexing = work.abandonment_revision();
    let reserve = work.remaining / INDEXED_ALGEBRA_RETRY_BUDGET_DIVISOR;
    work.remaining -= reserve;
    let descriptions =
        describe_finite_map_frontier(&prepared, DEFAULT_MAX_POLYNOMIAL_TERMS, &mut work, check)
            .map_err(|message| OptimizeError {
                pass: "indexed-algebra-frontier",
                message,
            })?;
    work.remaining += reserve;
    let indexed_algebra_abandoned = work.abandonment_revision() != abandonment_before_indexing;
    checkpoint(check)?;
    report.algebraic_work = work.consumed;
    report.abandoned_algebraic_expressions = work.abandoned_count();
    Ok((
        AlgebraicFrontier {
            descriptions,
            work,
            indexed_algebra_abandoned,
        },
        report,
    ))
}

impl AlgebraicFrontier {
    pub fn len(&self) -> usize {
        self.descriptions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn next(
        &mut self,
        check: &mut impl FnMut() -> Result<(), String>,
    ) -> Option<Result<(FiniteMapAlternative, PipelineReport), OptimizeError>> {
        if let Err(error) = checkpoint(check) {
            return Some(Err(error));
        }
        let mut alternative = self.descriptions.next()?;
        Some((|| {
            checkpoint(check)?;
            let consumed = self.work.consumed;
            let abandoned = self.work.abandoned_count();
            schedule_roots_after_abandoned_indexing(
                [&mut alternative.module],
                self.indexed_algebra_abandoned,
                DEFAULT_MAX_POLYNOMIAL_TERMS,
                &mut self.work,
            )
            .map_err(|message| OptimizeError {
                pass: "post-indexed-polynomial-scheduling",
                message,
            })?;
            checkpoint(check)?;
            let manager = PassManager::new([
                Pass::CommonSubexpressionElimination,
                Pass::DeadCodeElimination,
            ]);
            let mut cleanup_work = AlgebraicWorkBudget::new(
                manager.max_algebraic_work,
                manager.max_algebraic_expression_work,
            );
            let mut report = manager.run_with_budget_checked(
                &mut alternative.module,
                &mut cleanup_work,
                check,
            )?;
            report.algebraic_work += self.work.consumed - consumed;
            report.abandoned_algebraic_expressions += self.work.abandoned_count() - abandoned;
            Ok((alternative, report))
        })())
    }
}

#[cfg(test)]
#[path = "frontier_tests.rs"]
mod tests;
