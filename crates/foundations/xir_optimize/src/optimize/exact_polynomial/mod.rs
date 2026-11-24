// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod arithmetic;
mod cost;
mod expand;
mod schedule;

use algebraic_real::AlgebraicReal;

pub(super) use self::arithmetic::ExactPolynomial;
use self::cost::{POLYNOMIAL_EXECUTION_WEIGHTS, PolynomialCost, candidate_is_profitable};
#[cfg(test)]
use self::expand::full_expansion;
use self::expand::polynomial_roots;
use super::rebuild::Rebuilder;
use crate::ir::{Func, Module, ValueId};
use crate::ops::{Add, Constant, Literal, Multiply};
use crate::shape::Shape;

fn polynomial_schedule_settings() -> schedule::Settings {
    schedule::Settings {
        operation_weights: POLYNOMIAL_EXECUTION_WEIGHTS,
        balance_associative_trees: false,
        max_island_terms: 128,
        max_greedy_variants: 8,
        max_factors_per_term: 128,
        max_tree_depth: 512,
        extract_step_budget: 20_000_000,
        refine_rounds: 2,
        refine_step_budget: 20_000_000,
    }
}

const ROOT_SCHEDULE_OPERATION_THRESHOLD: u128 = 65_536;

pub(super) fn schedule_exact_polynomial_roots(
    module: &mut Module,
    max_terms: usize,
    budget: &mut super::AlgebraicWorkBudget,
) -> Result<bool, String> {
    schedule_exact_polynomial_roots_above_size(
        module,
        max_terms,
        budget,
        ROOT_SCHEDULE_OPERATION_THRESHOLD,
    )
}

pub(super) fn schedule_balanced_exact_polynomial_roots(
    module: &mut Module,
    max_terms: usize,
    budget: &mut super::AlgebraicWorkBudget,
) -> Result<bool, String> {
    let mut settings = polynomial_schedule_settings();
    settings.balance_associative_trees = true;
    schedule_exact_polynomial_roots_with_settings(module, max_terms, budget, 0, settings)
}

fn schedule_exact_polynomial_roots_above_size(
    module: &mut Module,
    max_terms: usize,
    budget: &mut super::AlgebraicWorkBudget,
    minimum_polynomial_operations: u128,
) -> Result<bool, String> {
    schedule_exact_polynomial_roots_with_settings(
        module,
        max_terms,
        budget,
        minimum_polynomial_operations,
        polynomial_schedule_settings(),
    )
}

fn restrict_associative_tree_balancing(requested: bool, result_count: usize) -> bool {
    requested && result_count == 1
}

fn schedule_exact_polynomial_roots_with_settings(
    module: &mut Module,
    max_terms: usize,
    budget: &mut super::AlgebraicWorkBudget,
    minimum_polynomial_operations: u128,
    settings: schedule::Settings,
) -> Result<bool, String> {
    if max_terms == 0 {
        return Ok(false);
    }
    let reserve_cleanup = |module: &Module, budget: &mut super::AlgebraicWorkBudget| {
        for func in &module.funcs {
            let mut attempt = budget.begin_attempt();
            if !attempt.reserve(func.nodes.len().saturating_mul(2)) {
                return false;
            }
        }
        true
    };
    let mut baseline = module.clone();
    if !reserve_cleanup(&baseline, budget) {
        return Ok(false);
    }
    super::passes::cse(&mut baseline)?;
    super::passes::dce(&mut baseline)?;

    enum Trial {
        Skipped(Func),
        Scheduled(Func),
        Exhausted,
    }
    let trials = baseline
        .funcs
        .iter()
        .map(|old| {
            let old_polynomial_cost = PolynomialCost::count_operations(old)?;
            let below_schedule_size =
                old_polynomial_cost.static_operation_count()? <= minimum_polynomial_operations;
            let within_schedule_depth = PolynomialCost::max_exact_polynomial_operation_depth(old)
                <= settings.max_tree_depth;
            if below_schedule_size && within_schedule_depth {
                return Ok(Trial::Skipped(old.clone()));
            }
            let scan_work = old
                .nodes
                .len()
                .saturating_add(old.values.len().saturating_mul(2));
            if !budget.begin_attempt().reserve(scan_work) {
                return Ok(Trial::Exhausted);
            }
            let roots = polynomial_roots(old);
            let mut expander = expand::BoundedPolynomialExpander::new(old, max_terms);
            let mut islands = Vec::new();
            for (index, node) in old.nodes.iter().enumerate() {
                let root = node.results.len() == 1 && roots[node.results[0].index()];
                let polynomial = if root {
                    match expander.expand(node.results[0], budget) {
                        expand::BoundedExpansion::Available(expansion)
                            if expansion.has_constant() =>
                        {
                            Some(expansion)
                        }
                        expand::BoundedExpansion::Available(_)
                        | expand::BoundedExpansion::Unavailable => None,
                        expand::BoundedExpansion::Exhausted => return Ok(Trial::Exhausted),
                    }
                } else {
                    None
                };
                if let Some(polynomial) = polynomial {
                    if !budget.reserve_batch(1) {
                        return Ok(Trial::Exhausted);
                    }
                    islands.push((index, polynomial));
                }
            }
            if islands.is_empty() {
                return Ok(Trial::Skipped(old.clone()));
            }
            if !budget.reserve_batch(islands.len()) {
                return Ok(Trial::Exhausted);
            }
            let terms = islands
                .iter()
                .map(|(_, expansion)| expansion.polynomial().terms())
                .collect::<Vec<_>>();
            let mut function_settings = settings;
            function_settings.balance_associative_trees = restrict_associative_tree_balancing(
                function_settings.balance_associative_trees,
                old.results.len(),
            );
            let scheduled = {
                let mut reserve = |cost| budget.reserve_batch(cost);
                schedule::schedule_with_shared_budget(&terms, &function_settings, &mut reserve)?
            };
            let Some(batch) = scheduled else {
                return Ok(Trial::Exhausted);
            };
            if !batch.trees.iter().any(Option::is_some) {
                return Ok(Trial::Skipped(old.clone()));
            }
            let mut new = Rebuilder::new(old);
            let mut islands = islands.into_iter().zip(batch.trees).peekable();
            for (index, node) in old.nodes.iter().enumerate() {
                if let Some(((_island, _), Some(tree))) =
                    islands.next_if(|((island, _), _)| *island == index)
                {
                    let shape = old.value_shape(node.results[0]).clone();
                    let value =
                        materialize_term_tree(&mut new, &shape, &tree, &batch.coefficients)?;
                    new.alias(&node.results, &[value])?;
                } else {
                    let inputs = new.mapped_inputs(node)?;
                    new.emit(old, &node.results, node.op.clone(), inputs)?;
                }
            }
            Ok(Trial::Scheduled(new.finish(old)?))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if trials.iter().any(|trial| matches!(trial, Trial::Exhausted)) {
        return Ok(false);
    }
    let mut scheduled = Vec::with_capacity(trials.len());
    let mut funcs = Vec::with_capacity(trials.len());
    for trial in trials {
        match trial {
            Trial::Skipped(func) => {
                scheduled.push(false);
                funcs.push(func);
            }
            Trial::Scheduled(func) => {
                scheduled.push(true);
                funcs.push(func);
            }
            Trial::Exhausted => unreachable!(),
        }
    }
    if !scheduled.iter().any(|&scheduled| scheduled) {
        return Ok(false);
    }
    let mut candidate = baseline.clone();
    candidate.funcs = funcs;
    if !reserve_cleanup(&candidate, budget) {
        return Ok(false);
    }
    super::passes::cse(&mut candidate)?;
    super::passes::dce(&mut candidate)?;

    let graph = &module.finite_map_graph;
    let mut changed = false;
    for (((output, candidate), baseline), scheduled) in module
        .funcs
        .iter_mut()
        .zip(candidate.funcs)
        .zip(baseline.funcs)
        .zip(scheduled)
    {
        if scheduled {
            let baseline_cost = PolynomialCost::count_operations(&baseline)?;
            let candidate_cost = PolynomialCost::count_operations(&candidate)?;
            if candidate_is_profitable(
                graph,
                &baseline_cost,
                &candidate_cost,
                settings.operation_weights,
            )? {
                *output = candidate;
                changed = true;
            }
        }
    }
    Ok(changed)
}

fn materialize_term_tree(
    builder: &mut Rebuilder,
    shape: &Shape,
    tree: &schedule::TermTree,
    coefficients: &[Vec<AlgebraicReal>],
) -> Result<ValueId, String> {
    Ok(match tree {
        schedule::TermTree::Factor(value) => builder.mapped(*value)?,
        schedule::TermTree::Coefficient(id) => {
            let coefficient = coefficients
                .get(*id)
                .ok_or_else(|| format!("island schedule references unknown coefficient {id}"))?;
            builder.emit_fresh(
                shape.clone(),
                Constant {
                    literal: Literal::exact(shape.clone(), coefficient.clone())?,
                },
                [],
            )
        }
        schedule::TermTree::Mul(left, right) => {
            let left = materialize_term_tree(builder, shape, left, coefficients)?;
            let right = materialize_term_tree(builder, shape, right, coefficients)?;
            builder.emit_fresh(shape.clone(), Multiply, [left, right])
        }
        schedule::TermTree::Add(left, right) => {
            let left = materialize_term_tree(builder, shape, left, coefficients)?;
            let right = materialize_term_tree(builder, shape, right, coefficients)?;
            builder.emit_fresh(shape.clone(), Add, [left, right])
        }
    })
}

#[cfg(test)]
mod root_schedule_tests {
    use super::*;
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;

    fn shared_suffix_module() -> Module {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("shared_suffix");
        let shape = Shape::scalar(DType::F32);
        let values = (0..4)
            .map(|_| builder.param(shape.clone()))
            .collect::<Vec<_>>();
        let one = builder.constant(
            Literal::exact(shape, vec![AlgebraicReal::one()]).expect("scalar one is valid"),
        );
        let common = values[0];
        let terms = values
            .into_iter()
            .skip(1)
            .map(|value| {
                let value = builder.multiply(value, one);
                builder.multiply(common, value)
            })
            .collect::<Vec<_>>();
        let first_pair = builder.add(terms[0], terms[1]);
        let first = builder.add(first_pair, terms[2]);
        let second = builder.add(terms[1], terms[2]);
        let entry = builder.end_func(vec![first, second]);
        builder.finish(entry).unwrap()
    }

    fn wide_shared_factor_module(reverse: bool) -> Module {
        const TERMS: usize = 2_800;
        let mut builder = ModuleBuilder::new();
        builder.begin_func("wide_shared_factor");
        let shape = Shape::scalar(DType::F32);
        let common = builder.param(shape.clone());
        let coefficient = builder.constant(
            Literal::exact(shape.clone(), vec![AlgebraicReal::sqrt(2u32)])
                .expect("scalar coefficient is valid"),
        );
        let mut terms = (0..TERMS)
            .map(|_| {
                let value = builder.param(shape.clone());
                let scaled = builder.multiply(value, coefficient);
                builder.multiply(common, scaled)
            })
            .collect::<Vec<_>>();
        if reverse {
            terms.reverse();
        }
        let result = terms
            .into_iter()
            .reduce(|left, right| builder.add(left, right))
            .expect("the polynomial has terms");
        let entry = builder.end_func(vec![result]);
        builder.finish(entry).unwrap()
    }

    fn output_polynomials(module: &Module) -> Vec<(Shape, ExactPolynomial)> {
        let func = module.entry_func();
        func.results
            .iter()
            .map(|&result| {
                (
                    func.value_shape(result).clone(),
                    full_expansion(func, result, 1024)
                        .expect("test output is polynomial")
                        .polynomial()
                        .clone(),
                )
            })
            .collect()
    }

    fn no_constant_module() -> Module {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("no_constant");
        let shape = Shape::scalar(DType::F32);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let result = builder.add(left, right);
        let entry = builder.end_func(vec![result]);
        builder.finish(entry).unwrap()
    }

    fn dead_float_and_live_integer_module() -> Module {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("non_float");
        let float = Shape::scalar(DType::F32);
        let dead_left = builder.param(float.clone());
        let dead_right = builder.param(float);
        let _dead = builder.add(dead_left, dead_right);
        let integer = Shape::scalar(DType::I32);
        let left = builder.param(integer.clone());
        let right = builder.param(integer);
        let result = builder.add(left, right);
        let entry = builder.end_func(vec![result]);
        builder.finish(entry).unwrap()
    }

    #[test]
    fn bounded_root_scheduling_exposes_a_shared_suffix() {
        let mut module = shared_suffix_module();
        let expected = output_polynomials(&module);
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            schedule_exact_polynomial_roots_above_size(&mut module, 1024, &mut budget, 0,).unwrap()
        );
        crate::verify::verify(&module).unwrap();
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
        assert!(budget.consumed > 0);
        assert_eq!(output_polynomials(&module), expected);
    }

    #[test]
    fn default_root_scheduling_is_invariant_to_wide_sum_order() {
        let mut forward = wide_shared_factor_module(false);
        let mut reverse = wide_shared_factor_module(true);
        for module in [&mut forward, &mut reverse] {
            let mut budget = super::super::AlgebraicWorkBudget::new(1 << 28, 1 << 24);
            assert!(
                schedule_exact_polynomial_roots(module, 1 << 20, &mut budget).unwrap(),
                "the default threshold must schedule this profitable polynomial"
            );
            crate::verify::verify(module).unwrap();
        }
        assert_eq!(format!("{forward:#?}"), format!("{reverse:#?}"));
    }

    #[test]
    fn root_scheduling_skips_functions_below_size_threshold() {
        let mut module = shared_suffix_module();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(!schedule_exact_polynomial_roots(&mut module, 1024, &mut budget).unwrap());
        assert_eq!(format!("{module:#?}"), original);
    }

    #[test]
    fn exhausted_cleanup_budget_keeps_the_input() {
        let mut module = shared_suffix_module();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(0, 0);
        assert!(!schedule_exact_polynomial_roots(&mut module, 1024, &mut budget).unwrap());
        assert_eq!(format!("{module:#?}"), original);
        crate::verify::verify(&module).unwrap();
    }

    #[test]
    fn partial_work_exhaustion_keeps_the_input() {
        let mut module = shared_suffix_module();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(64, 1 << 20);
        assert!(
            !schedule_exact_polynomial_roots_above_size(&mut module, 1024, &mut budget, 0).unwrap()
        );
        assert!(budget.consumed > 0);
        assert_eq!(format!("{module:#?}"), original);
    }

    #[test]
    fn size_threshold_is_inclusive() {
        let original = shared_suffix_module();
        let original_snapshot = format!("{original:#?}");
        let mut cleaned = original.clone();
        super::super::passes::cse(&mut cleaned).unwrap();
        super::super::passes::dce(&mut cleaned).unwrap();
        let threshold = PolynomialCost::count_operations(cleaned.entry_func())
            .unwrap()
            .static_operation_count()
            .unwrap();

        let mut at_threshold = original.clone();
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            !schedule_exact_polynomial_roots_above_size(
                &mut at_threshold,
                1024,
                &mut budget,
                threshold,
            )
            .unwrap()
        );
        assert_eq!(format!("{at_threshold:#?}"), original_snapshot);

        let mut above_threshold = original;
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            schedule_exact_polynomial_roots_above_size(
                &mut above_threshold,
                1024,
                &mut budget,
                threshold - 1,
            )
            .unwrap()
        );
    }

    #[test]
    fn a_polynomial_without_constants_has_no_schedulable_island() {
        let mut module = no_constant_module();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            !schedule_exact_polynomial_roots_above_size(&mut module, 1024, &mut budget, 0).unwrap()
        );
        assert_eq!(format!("{module:#?}"), original);
    }

    #[test]
    fn dead_float_and_live_integer_work_does_not_trigger_scheduling() {
        let mut module = dead_float_and_live_integer_module();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 1 << 20);
        assert!(
            !schedule_exact_polynomial_roots_above_size(&mut module, 1024, &mut budget, 0).unwrap()
        );
        assert_eq!(format!("{module:#?}"), original);
    }

    #[test]
    fn recovered_tree_balancing_is_only_requested_for_one_result() {
        assert!(restrict_associative_tree_balancing(true, 1));
        assert!(!restrict_associative_tree_balancing(true, 2));
        assert!(!restrict_associative_tree_balancing(false, 1));
    }

    #[test]
    fn tensor_coefficients_are_checked_before_expansion() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("tensor_coefficient");
        let shape = Shape::new(DType::F32, [64]);
        let input = builder.param(shape.clone());
        let coefficient =
            builder.constant(Literal::exact(shape, vec![AlgebraicReal::sqrt(2u32); 64]).unwrap());
        let result = builder.multiply(input, coefficient);
        let entry = builder.end_func(vec![result]);
        let mut module = builder.finish(entry).unwrap();
        let original = format!("{module:#?}");
        let mut budget = super::super::AlgebraicWorkBudget::new(1 << 26, 64);
        assert!(
            !schedule_exact_polynomial_roots_above_size(&mut module, 1024, &mut budget, 0).unwrap()
        );
        assert_eq!(format!("{module:#?}"), original);
        assert!(budget.abandoned_count() > 0);
    }
}
