// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic whole-module optimization passes. Constant folding handles
//! byte-exact tensor transforms and boolean/integer scalar operations; floating
//! arithmetic remains in the target backend so host evaluation cannot alter its
//! exceptional-value or rounding behavior.
//!
//! Passes rebuild functions node by node and leave regions they cannot prove
//! unchanged. Indexed algebra additionally searches bounded polynomial and
//! finite-map rewrites. No pass aborts on an unprovable fact — the only fatal
//! conditions are per-pass verification and structural rebuild failures, both
//! of which signal corrupt IR rather than an unproven fact.

mod exact_polynomial;
mod map_reduce {
    pub(super) mod recover;
    pub(super) mod rewrite;
}
mod passes {
    mod algebraic;
    mod cleanup;
    mod fold;
    mod shapes;
    mod unroll;

    pub(super) use algebraic::algebraic_canonicalize;
    pub(super) use cleanup::{cse, dce};
    pub(super) use fold::constant_folding;
    pub(super) use shapes::canonicalize_shapes;
    pub(super) use unroll::unroll_static_regions;
}
mod rebuild;

use std::collections::HashSet;
use std::time::{Duration, Instant};

pub use map_reduce::rewrite::{FiniteMapAlternative, FiniteMapWork, PartialContributionRatio};

use self::exact_polynomial::{
    schedule_balanced_exact_polynomial_roots, schedule_exact_polynomial_roots,
};
use self::map_reduce::recover::recover_map_reduce_regions;
use self::map_reduce::rewrite::indexed_algebra_canonicalize;
mod frontier;
use self::passes::{
    algebraic_canonicalize, canonicalize_shapes, constant_folding, cse, dce, unroll_static_regions,
};
use crate::ir::Module;
use crate::verify;
pub use frontier::{AlgebraicFrontier, describe_algebraic_frontier_with_limits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    UnrollStaticRegions,
    CanonicalizeShapes,
    ConstantFolding,
    AlgebraicCanonicalization,
    ScheduleExactPolynomialRoots,
    RecoverMapReduceRegions,
    /// Rewrite exact arithmetic, gathers and symbolic sums in one typed graph.
    IndexedAlgebraCanonicalization,
    CommonSubexpressionElimination,
    DeadCodeElimination,
}

impl Pass {
    fn name(self) -> &'static str {
        match self {
            Pass::UnrollStaticRegions => "unroll-static-regions",
            Pass::CanonicalizeShapes => "canonicalize-shapes",
            Pass::ConstantFolding => "constant-folding",
            Pass::AlgebraicCanonicalization => "algebraic-canonicalization",
            Pass::ScheduleExactPolynomialRoots => "schedule-exact-polynomial-roots",
            Pass::RecoverMapReduceRegions => "recover-map-reduce-regions",
            Pass::IndexedAlgebraCanonicalization => "indexed-algebra-canonicalization",
            Pass::CommonSubexpressionElimination => "cse",
            Pass::DeadCodeElimination => "dce",
        }
    }

    fn run(
        self,
        module: &mut Module,
        max_folded_literal_bytes: usize,
        max_polynomial_terms: usize,
        algebraic_work: &mut AlgebraicWorkBudget,
    ) -> Result<bool, String> {
        match self {
            Pass::UnrollStaticRegions => unroll_static_regions(module, max_folded_literal_bytes),
            Pass::CanonicalizeShapes => canonicalize_shapes(module),
            Pass::ConstantFolding => constant_folding(module, max_folded_literal_bytes),
            Pass::AlgebraicCanonicalization => {
                algebraic_canonicalize(module, max_folded_literal_bytes, algebraic_work)
            }
            Pass::ScheduleExactPolynomialRoots => {
                schedule_exact_polynomial_roots(module, max_polynomial_terms, algebraic_work)
            }
            Pass::RecoverMapReduceRegions => {
                recover_map_reduce_regions(module, max_polynomial_terms)
            }
            Pass::IndexedAlgebraCanonicalization => {
                indexed_algebra_canonicalize(module, max_polynomial_terms, algebraic_work)
            }
            Pass::CommonSubexpressionElimination => cse(module),
            Pass::DeadCodeElimination => dce(module),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassReport {
    pub name: &'static str,
    pub changed: bool,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub values_before: usize,
    pub values_after: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipelineReport {
    pub passes: Vec<PassReport>,
    pub algebraic_work: usize,
    pub abandoned_algebraic_expressions: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct AlgebraicWorkBudget {
    remaining: usize,
    expression_limit: usize,
    consumed: usize,
    abandoned: HashSet<(usize, u32)>,
    abandoned_attempts: usize,
    abandonment_revision: usize,
}

impl AlgebraicWorkBudget {
    fn new(total: usize, expression_limit: usize) -> Self {
        Self {
            remaining: total,
            expression_limit,
            consumed: 0,
            abandoned: HashSet::new(),
            abandoned_attempts: 0,
            abandonment_revision: 0,
        }
    }

    pub(crate) fn begin_attempt(&mut self) -> AlgebraicProofBudget<'_> {
        let remaining = self.expression_limit;
        AlgebraicProofBudget {
            work: self,
            remaining,
            failed: false,
        }
    }

    fn abandoned_count(&self) -> usize {
        self.abandoned.len().saturating_add(self.abandoned_attempts)
    }

    fn abandonment_revision(&self) -> usize {
        self.abandonment_revision
    }

    fn record_abandonment(&mut self) {
        self.abandonment_revision = self.abandonment_revision.saturating_add(1);
    }

    /// Reserve work shared by several roots against the total allowance.
    /// Exact polynomial values are still built in per-expression attempts.
    pub(crate) fn reserve_batch(&mut self, cost: usize) -> bool {
        if cost > self.remaining {
            self.abandoned_attempts = self.abandoned_attempts.saturating_add(1);
            self.record_abandonment();
            return false;
        }
        self.remaining -= cost;
        self.consumed += cost;
        true
    }

    pub(crate) fn reserve(&mut self, expression: (usize, u32), cost: usize) -> bool {
        if cost > self.expression_limit || cost > self.remaining {
            self.abandoned.insert(expression);
            self.record_abandonment();
            return false;
        }
        self.remaining -= cost;
        self.consumed += cost;
        true
    }
}

/// One proof attempt spends its local allowance and the shared work budget.
/// A failed reservation abandons the attempt once; completed work is not refunded.
pub(crate) struct AlgebraicProofBudget<'a> {
    work: &'a mut AlgebraicWorkBudget,
    remaining: usize,
    failed: bool,
}

impl AlgebraicProofBudget<'_> {
    pub(crate) fn reserve(&mut self, cost: usize) -> bool {
        if self.failed {
            return false;
        }
        if cost > self.remaining || cost > self.work.remaining {
            self.work.abandoned_attempts = self.work.abandoned_attempts.saturating_add(1);
            self.work.record_abandonment();
            self.failed = true;
            return false;
        }
        self.remaining -= cost;
        self.work.remaining -= cost;
        self.work.consumed += cost;
        true
    }

    pub(crate) fn failed(&self) -> bool {
        self.failed
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizeError {
    pub pass: &'static str,
    pub message: String,
}

impl std::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XIR pass {}: {}", self.pass, self.message)
    }
}

impl std::error::Error for OptimizeError {}

// Preserve part of the caller's total allowance for a bounded retry when
// indexed algebra proves that its first extraction was incomplete.
const INDEXED_ALGEBRA_RETRY_BUDGET_DIVISOR: usize = 4;

fn schedule_roots_after_abandoned_indexing<'a>(
    modules: impl IntoIterator<Item = &'a mut Module>,
    indexed_algebra_abandoned: bool,
    max_polynomial_terms: usize,
    work: &mut AlgebraicWorkBudget,
) -> Result<bool, String> {
    if !indexed_algebra_abandoned {
        return Ok(false);
    }
    let mut changed = false;
    for module in modules {
        changed |= schedule_balanced_exact_polynomial_roots(module, max_polynomial_terms, work)?;
    }
    Ok(changed)
}

pub fn extract_algebraic_frontier_with_limits(
    module: &Module,
    max_algebraic_work: usize,
    max_algebraic_expression_work: usize,
) -> Result<(Vec<FiniteMapAlternative>, PipelineReport), OptimizeError> {
    let (mut frontier, mut report) = describe_algebraic_frontier_with_limits(
        module,
        max_algebraic_work,
        max_algebraic_expression_work,
        &mut || Ok(()),
    )?;
    let mut alternatives = Vec::new();
    while let Some(alternative) = frontier.next(&mut || Ok(())) {
        let (alternative, work) = alternative?;
        report.algebraic_work += work.algebraic_work;
        report.abandoned_algebraic_expressions += work.abandoned_algebraic_expressions;
        alternatives.push(alternative);
    }
    Ok((alternatives, report))
}

#[derive(Clone, Debug)]
pub struct PassManager {
    passes: Vec<Pass>,
    max_folded_literal_bytes: usize,
    max_polynomial_terms: usize,
    max_algebraic_work: usize,
    max_algebraic_expression_work: usize,
}

const DEFAULT_MAX_FOLDED_LITERAL_BYTES: usize = 1 << 20;
const DEFAULT_MAX_POLYNOMIAL_TERMS: usize = 1 << 20;
pub const DEFAULT_MAX_ALGEBRAIC_WORK: usize = 1 << 28;
pub const DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK: usize = 1 << 20;

/// The floating-point algebraic pipeline: fold locally, rewrite exact arithmetic
/// and symbolic indexing in one graph, then eliminate duplicate and dead values.
/// Reassociates and cancels floating arithmetic only for callers whose math
/// policy allows it.
pub fn algebraic_pipeline() -> PassManager {
    PassManager::new([
        Pass::CanonicalizeShapes,
        Pass::ConstantFolding,
        Pass::AlgebraicCanonicalization,
        Pass::ScheduleExactPolynomialRoots,
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
}

impl PassManager {
    pub fn new(passes: impl IntoIterator<Item = Pass>) -> Self {
        Self {
            passes: passes.into_iter().collect(),
            max_folded_literal_bytes: DEFAULT_MAX_FOLDED_LITERAL_BYTES,
            max_polynomial_terms: DEFAULT_MAX_POLYNOMIAL_TERMS,
            max_algebraic_work: DEFAULT_MAX_ALGEBRAIC_WORK,
            max_algebraic_expression_work: DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
        }
    }

    pub fn with_max_polynomial_terms(mut self, max_terms: usize) -> Self {
        self.max_polynomial_terms = max_terms;
        self
    }

    pub fn with_max_algebraic_work(mut self, max_work: usize) -> Self {
        self.max_algebraic_work = max_work;
        self
    }

    pub fn with_max_algebraic_expression_work(mut self, max_work: usize) -> Self {
        self.max_algebraic_expression_work = max_work;
        self
    }

    pub fn run(&self, module: &mut Module) -> Result<PipelineReport, OptimizeError> {
        let mut algebraic_work =
            AlgebraicWorkBudget::new(self.max_algebraic_work, self.max_algebraic_expression_work);
        self.run_with_budget(module, &mut algebraic_work)
    }

    fn run_with_budget(
        &self,
        module: &mut Module,
        algebraic_work: &mut AlgebraicWorkBudget,
    ) -> Result<PipelineReport, OptimizeError> {
        self.run_with_budget_checked(module, algebraic_work, &mut || Ok(()))
    }

    fn run_with_budget_checked(
        &self,
        module: &mut Module,
        algebraic_work: &mut AlgebraicWorkBudget,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<PipelineReport, OptimizeError> {
        verify::verify(module).map_err(|error| OptimizeError {
            pass: "input-verification",
            message: error.to_string(),
        })?;
        let mut reports = Vec::with_capacity(self.passes.len());
        for &pass in &self.passes {
            checkpoint().map_err(|message| OptimizeError {
                pass: "source-checkpoint",
                message,
            })?;
            let (nodes_before, values_before) = module_size(module);
            let started = Instant::now();
            let changed = pass
                .run(
                    module,
                    self.max_folded_literal_bytes,
                    self.max_polynomial_terms,
                    algebraic_work,
                )
                .map_err(|message| OptimizeError {
                    pass: pass.name(),
                    message,
                })?;
            let elapsed = started.elapsed();
            verify::verify(module).map_err(|error| OptimizeError {
                pass: pass.name(),
                message: format!("verification failed: {error}"),
            })?;
            let (nodes_after, values_after) = module_size(module);
            reports.push(PassReport {
                name: pass.name(),
                changed,
                nodes_before,
                nodes_after,
                values_before,
                values_after,
                elapsed,
            });
        }
        Ok(PipelineReport {
            passes: reports,
            algebraic_work: algebraic_work.consumed,
            abandoned_algebraic_expressions: algebraic_work.abandoned_count(),
        })
    }
}

fn module_size(module: &Module) -> (usize, usize) {
    module.funcs.iter().fold((0, 0), |(nodes, values), func| {
        (nodes + func.nodes.len(), values + func.values.len())
    })
}

#[cfg(test)]
mod tests {
    use algebraic_real::AlgebraicReal;

    use super::*;
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;
    use crate::ops::{Add, Literal, Multiply};
    use crate::shape::Shape;

    fn shared_polynomial_suffix() -> Module {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("shared_polynomial_suffix");
        let shape = Shape::scalar(DType::F32);
        let values = (0..4)
            .map(|_| builder.param(shape.clone()))
            .collect::<Vec<_>>();
        let one = builder.constant(
            Literal::exact(shape, vec![AlgebraicReal::one()]).expect("scalar one is valid"),
        );
        let terms = values
            .iter()
            .skip(1)
            .map(|&value| {
                let value = builder.multiply(value, one);
                builder.multiply(values[0], value)
            })
            .collect::<Vec<_>>();
        let first_pair = builder.add(terms[0], terms[1]);
        let first = builder.add(first_pair, terms[2]);
        let second = builder.add(terms[1], terms[2]);
        let entry = builder.end_func(vec![first, second]);
        builder.finish(entry).unwrap()
    }

    #[test]
    fn repeated_abandonment_of_one_expression_advances_the_revision() {
        let mut work = AlgebraicWorkBudget::new(0, 0);
        assert!(!work.reserve((0, 0), 1));
        let first = work.abandonment_revision();
        assert!(!work.reserve((0, 0), 1));
        assert!(work.abandonment_revision() > first);
        assert_eq!(work.abandoned_count(), 1);
    }

    #[test]
    fn abandoned_indexing_retries_small_polynomial_roots() {
        let mut module = shared_polynomial_suffix();
        let mut work = AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            schedule_roots_after_abandoned_indexing(
                [&mut module],
                true,
                DEFAULT_MAX_POLYNOMIAL_TERMS,
                &mut work,
            )
            .unwrap()
        );
        assert!(work.consumed <= 1 << 26);
        verify::verify(&module).unwrap();
        let func = module.entry_func();
        assert_eq!(
            func.nodes
                .iter()
                .filter(|node| node.op.as_any().is::<Add>())
                .count(),
            2
        );
        assert_eq!(
            func.nodes
                .iter()
                .filter(|node| node.op.as_any().is::<Multiply>())
                .count(),
            2
        );
    }

    #[test]
    fn completed_indexing_does_not_retry_polynomial_roots() {
        let mut module = shared_polynomial_suffix();
        let before = module.to_string();
        let mut work = AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            !schedule_roots_after_abandoned_indexing(
                [&mut module],
                false,
                DEFAULT_MAX_POLYNOMIAL_TERMS,
                &mut work,
            )
            .unwrap()
        );
        assert_eq!(work.consumed, 0);
        assert_eq!(module.to_string(), before);
    }
}
