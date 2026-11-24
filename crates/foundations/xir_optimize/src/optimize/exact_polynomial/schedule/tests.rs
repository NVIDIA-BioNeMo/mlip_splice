// SPDX-License-Identifier: Apache-2.0

use super::super::cost::{POLYNOMIAL_EXECUTION_WEIGHTS, PolynomialOperation};
use super::*;

fn settings() -> Settings {
    Settings {
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

fn ones(elements: usize) -> Vec<AlgebraicReal> {
    vec![AlgebraicReal::one(); elements]
}

fn term(factors: &[u32], coefficient: Vec<AlgebraicReal>) -> ExactMonomial {
    ExactMonomial::new(factors.iter().copied().map(ValueId).collect(), coefficient)
}

fn tree_cost(tree: &TermTree, settings: &Settings) -> u64 {
    match tree {
        TermTree::Factor(_) | TermTree::Coefficient(_) => 0,
        TermTree::Mul(left, right) => {
            settings
                .operation_weights
                .cost(PolynomialOperation::Multiply)
                + tree_cost(left, settings)
                + tree_cost(right, settings)
        }
        TermTree::Add(left, right) => {
            settings.operation_weights.cost(PolynomialOperation::Add)
                + tree_cost(left, settings)
                + tree_cost(right, settings)
        }
    }
}

fn contains(tree: &TermTree, target: &TermTree) -> bool {
    if tree == target {
        return true;
    }
    match tree {
        TermTree::Mul(left, right) | TermTree::Add(left, right) => {
            contains(left, target) || contains(right, target)
        }
        _ => false,
    }
}

/// Expand a scheduled tree back into the canonical factor-multiset →
/// coefficient map, mirroring `exact_polynomial_apply`.
fn expand(
    tree: &TermTree,
    coefficients: &[Vec<AlgebraicReal>],
    elements: usize,
) -> HashMap<Vec<ValueId>, Vec<AlgebraicReal>> {
    match tree {
        TermTree::Factor(value) => HashMap::from([(vec![*value], ones(elements))]),
        TermTree::Coefficient(id) => HashMap::from([(Vec::new(), coefficients[*id].clone())]),
        TermTree::Mul(left, right) => {
            let left = expand(left, coefficients, elements);
            let right = expand(right, coefficients, elements);
            let mut product: HashMap<Vec<ValueId>, Vec<AlgebraicReal>> = HashMap::new();
            for (left_factors, left_coefficient) in &left {
                for (right_factors, right_coefficient) in &right {
                    let mut factors = left_factors.clone();
                    factors.extend_from_slice(right_factors);
                    factors.sort_unstable_by_key(|value| value.index());
                    let coefficient = left_coefficient
                        .iter()
                        .zip(right_coefficient)
                        .map(|(left, right)| left * right)
                        .collect::<Vec<_>>();
                    merge_term(&mut product, factors, coefficient);
                }
            }
            product
        }
        TermTree::Add(left, right) => {
            let mut sum = expand(left, coefficients, elements);
            for (factors, coefficient) in expand(right, coefficients, elements) {
                merge_term(&mut sum, factors, coefficient);
            }
            sum
        }
    }
}

fn merge_term(
    into: &mut HashMap<Vec<ValueId>, Vec<AlgebraicReal>>,
    factors: Vec<ValueId>,
    coefficient: Vec<AlgebraicReal>,
) {
    into.entry(factors)
        .and_modify(|accumulator| {
            for (accumulator, value) in accumulator.iter_mut().zip(&coefficient) {
                *accumulator = &*accumulator + value;
            }
        })
        .or_insert(coefficient);
}

fn expansion_of(terms: &[ExactMonomial]) -> HashMap<Vec<ValueId>, Vec<AlgebraicReal>> {
    let mut expansion = HashMap::new();
    for term in terms {
        merge_term(
            &mut expansion,
            term.factors().to_vec(),
            term.coefficient().to_vec(),
        );
    }
    expansion
}

fn schedule_all(islands: &[&[ExactMonomial]]) -> BatchSchedule {
    schedule_with_work_limit(islands, &settings(), usize::MAX)
        .unwrap()
        .unwrap()
}

fn schedule_one(terms: &[ExactMonomial]) -> (Option<TermTree>, Vec<Vec<AlgebraicReal>>) {
    let mut batch = schedule_all(&[terms]);
    (batch.trees.remove(0), batch.coefficients)
}

#[test]
fn factors_shared_head_and_beats_flat() {
    let terms = [term(&[1, 2], ones(1)), term(&[1, 3], ones(1))];
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn keeps_flat_when_no_factor_is_shared() {
    let terms = [term(&[1, 2], ones(1)), term(&[3, 4], ones(1))];
    assert!(schedule_one(&terms).0.is_none());
}

#[test]
fn preserves_exact_coefficients_through_factoring() {
    let two = &AlgebraicReal::one() + &AlgebraicReal::one();
    let terms = [
        term(&[1, 2], vec![two.clone()]),
        term(&[1, 3], vec![two.clone()]),
    ];
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        3 * settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn shares_the_residual_across_groups() {
    // ac + ad + bc + bd: both greedy groups leave a shared residual,
    // which the shared DAG retains; extraction keeps a(c+d) + b(c+d).
    let terms = [
        term(&[1, 3], ones(1)),
        term(&[1, 4], ones(1)),
        term(&[2, 3], ones(1)),
        term(&[2, 4], ones(1)),
    ];
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        2 * settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + 3 * settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn shares_residuals_across_islands() {
    // Alone, ac + ad + bc + bd extracts with the (c+d) residual; batched
    // with ea + eb — whose only factoring is e·(a+b) — the marginal pass
    // flips it to c(a+b) + d(a+b) so both islands share one (a+b).
    let product = [
        term(&[1, 3], ones(1)),
        term(&[1, 4], ones(1)),
        term(&[2, 3], ones(1)),
        term(&[2, 4], ones(1)),
    ];
    let scaled = [term(&[1, 5], ones(1)), term(&[2, 5], ones(1))];
    let batch = schedule_all(&[&product, &scaled]);
    let shared = TermTree::Add(
        Box::new(TermTree::Factor(ValueId(1))),
        Box::new(TermTree::Factor(ValueId(2))),
    );
    for (tree, terms) in batch.trees.iter().zip([&product[..], &scaled[..]]) {
        let tree = tree.as_ref().unwrap();
        assert!(contains(tree, &shared), "missing shared residual: {tree:?}");
        assert_eq!(expand(tree, &batch.coefficients, 1), expansion_of(terms));
    }
}

#[test]
fn splits_oversized_islands_into_factor_groups() {
    // 200 terms, two disjoint 100-term groups: 1·xₖ and 2·yₖ. Each group
    // factors to one Mul; the quotients and the combine are pure Adds.
    let mut terms = Vec::new();
    for k in 0..100 {
        terms.push(term(&[1, 10 + k], ones(1)));
        terms.push(term(&[2, 200 + k], ones(1)));
    }
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        2 * settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + 199 * settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn factors_nested_shared_heads_in_oversized_islands() {
    // 130 terms 1·2·xₖ: the partition peels factor 1, then factor 2
    // inside the still-oversized quotient — 1·(2·Σxₖ).
    let terms: Vec<ExactMonomial> = (0..130).map(|k| term(&[1, 2, 10 + k], ones(1))).collect();
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        2 * settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + 129 * settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn keeps_flat_when_an_oversized_island_shares_nothing() {
    let terms: Vec<ExactMonomial> = (0..130)
        .map(|k| term(&[2 * k, 2 * k + 1], ones(1)))
        .collect();
    assert!(schedule_one(&terms).0.is_none());
}

#[test]
fn factors_single_factor_terms_into_the_group() {
    // ab + a + ac = a·(b + 1 + c): the pulled single-factor term becomes
    // an explicit ones coefficient.
    let terms = [
        term(&[1, 2], ones(1)),
        term(&[1], ones(1)),
        term(&[1, 3], ones(1)),
    ];
    let (tree, coefficients) = schedule_one(&terms);
    let tree = tree.unwrap();
    let settings = settings();
    assert_eq!(
        tree_cost(&tree, &settings),
        settings
            .operation_weights
            .cost(PolynomialOperation::Multiply)
            + 2 * settings.operation_weights.cost(PolynomialOperation::Add)
    );
    assert_eq!(expand(&tree, &coefficients, 1), expansion_of(&terms));
}

#[test]
fn interns_identical_coefficients_once() {
    let two = &AlgebraicReal::one() + &AlgebraicReal::one();
    let left = [
        term(&[1, 2], vec![two.clone()]),
        term(&[1, 3], vec![two.clone()]),
    ];
    let right = [
        term(&[4, 5], vec![two.clone()]),
        term(&[4, 6], vec![two.clone()]),
    ];
    let batch = schedule_all(&[&left, &right]);
    assert_eq!(batch.coefficients, vec![vec![two]]);
    for (tree, terms) in batch.trees.iter().zip([&left[..], &right[..]]) {
        let tree = tree.as_ref().unwrap();
        assert_eq!(expand(tree, &batch.coefficients, 1), expansion_of(terms));
    }
}

#[test]
fn work_limit_exhaustion_returns_no_partial_schedule() {
    let product = [
        term(&[1, 3], ones(1)),
        term(&[1, 4], ones(1)),
        term(&[2, 3], ones(1)),
        term(&[2, 4], ones(1)),
    ];
    let scaled = [term(&[1, 5], ones(1)), term(&[2, 5], ones(1))];
    assert!(
        schedule_with_work_limit(&[&product, &scaled], &settings(), 1)
            .unwrap()
            .is_none()
    );
}

#[test]
fn extraction_limit_returns_no_partial_split_schedule() {
    let split = (0..130)
        .map(|k| term(&[1, 2, 10 + k], ones(1)))
        .collect::<Vec<_>>();
    let request = [term(&[3, 5], ones(1)), term(&[4, 5], ones(1))];
    let mut limited = settings();
    limited.extract_step_budget = 0;
    assert!(
        schedule_with_work_limit(&[&split, &request], &limited, usize::MAX)
            .unwrap()
            .is_none()
    );
}

#[test]
fn refinement_limit_returns_no_partial_split_schedule() {
    let split = (0..130)
        .map(|k| term(&[1, 2, 10 + k], ones(1)))
        .collect::<Vec<_>>();
    let request = [term(&[3, 5], ones(1)), term(&[4, 5], ones(1))];
    let mut limited = settings();
    limited.refine_step_budget = 0;
    assert!(
        schedule_with_work_limit(&[&split, &request], &limited, usize::MAX)
            .unwrap()
            .is_none()
    );
}

#[test]
fn terms_beyond_the_factor_limit_are_not_scheduled() {
    let left = (0..129).collect::<Vec<_>>();
    let mut right = left.clone();
    right.push(200);
    let terms = [term(&left, ones(1)), term(&right, ones(1))];
    let batch = schedule_all(&[&terms]);
    assert_eq!(batch.trees, [None]);
    assert!(batch.coefficients.is_empty());
}
