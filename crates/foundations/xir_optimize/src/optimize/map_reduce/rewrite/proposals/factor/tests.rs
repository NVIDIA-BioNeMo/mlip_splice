// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use super::{ENode, MAX_POLYNOMIAL_SIZE, Snapshot};
use crate::dtype::DType;
use crate::ir::ValueId;
use crate::shape::Shape;

fn term(factors: &[u32], coefficient: Vec<AlgebraicReal>) -> Polynomial {
    let elements = coefficient.len();
    factors.iter().fold(
        Polynomial::constant(coefficient, MAX_POLYNOMIAL_SIZE).unwrap(),
        |polynomial, &factor| {
            polynomial
                .multiply(
                    &Polynomial::variable(factor, elements, MAX_POLYNOMIAL_SIZE).unwrap(),
                    MAX_POLYNOMIAL_SIZE,
                    &mut |_| true,
                )
                .unwrap()
        },
    )
}

fn sum(terms: impl IntoIterator<Item = Polynomial>) -> Polynomial {
    terms
        .into_iter()
        .reduce(|left, right| {
            left.add(&right, MAX_POLYNOMIAL_SIZE, &mut |_| true)
                .unwrap()
        })
        .unwrap()
}

fn expand(form: &Form, ty: usize, elements: usize) -> Polynomial {
    match form {
        Form::Class(class) => Polynomial::variable(*class, elements, MAX_POLYNOMIAL_SIZE).unwrap(),
        Form::Coefficient(actual, values) => {
            assert_eq!(*actual, ty);
            Polynomial::constant(values.clone(), MAX_POLYNOMIAL_SIZE).unwrap()
        }
        Form::Binary(name, actual, left, right) => {
            assert_eq!(*actual, ty);
            let left = expand(left, ty, elements);
            let right = expand(right, ty, elements);
            match *name {
                "Add" => left
                    .add(&right, MAX_POLYNOMIAL_SIZE, &mut |_| true)
                    .unwrap(),
                "Multiply" => left
                    .multiply(&right, MAX_POLYNOMIAL_SIZE, &mut |_| true)
                    .unwrap(),
                _ => panic!("not a polynomial operation"),
            }
        }
        _ => panic!("not a polynomial witness"),
    }
}

fn unlimited() -> AlgebraicWorkBudget {
    AlgebraicWorkBudget::new(1 << 26, 1 << 20)
}

#[test]
fn factor_subset_keeps_the_unshared_remainder_and_repeated_factors() {
    let polynomial = sum([
        term(&[0, 0, 1], vec![AlgebraicReal::sqrt(2u32)]),
        term(&[0, 0, 2], vec![AlgebraicReal::sqrt(3u32)]),
        term(&[3], vec![AlgebraicReal::integer(-7)]),
        term(&[], vec![AlgebraicReal::integer(11)]),
    ]);
    let witnesses = witnesses(&polynomial, 7, 1, &mut unlimited());
    assert!(!witnesses.is_empty());
    for witness in witnesses {
        assert_eq!(expand(&witness.form, 7, 1), polynomial);
        assert!(witness.work < materialization_work(&polynomial, 1));
        assert!(matches!(witness.form, Form::Binary("Add", 7, ..)));
    }
}

#[test]
fn dense_exact_coefficients_are_not_replaced_with_one_scalar() {
    let polynomial = sum([
        term(
            &[0, 1],
            vec![AlgebraicReal::sqrt(2u32), AlgebraicReal::integer(-3)],
        ),
        term(
            &[0, 2],
            vec![AlgebraicReal::sqrt(3u32), AlgebraicReal::integer(5)],
        ),
    ]);
    let witnesses = witnesses(&polynomial, 4, 2, &mut unlimited());
    assert!(!witnesses.is_empty());
    for witness in witnesses {
        assert_eq!(expand(&witness.form, 4, 2), polynomial);
    }
}

#[test]
fn quotients_above_the_old_structural_limit_are_planned_recursively() {
    let polynomial =
        sum((0..120).map(|i| term(&[0, 1 + i % 3, 4 + i, 124], vec![AlgebraicReal::one()])));
    assert_eq!(polynomial.len(), 120);
    let structural_size = polynomial
        .terms()
        .iter()
        .map(|term| {
            term.factors().len()
                + term
                    .coefficient()
                    .iter()
                    .map(AlgebraicReal::structural_size)
                    .sum::<usize>()
        })
        .sum::<usize>();
    assert_eq!(structural_size, 600);
    let witnesses = witnesses(&polynomial, 0, 1, &mut unlimited());
    assert!(!witnesses.is_empty());
    for witness in witnesses {
        assert!(witness.work <= MAX_FORM_WORK);
        assert_eq!(expand(&witness.form, 0, 1), polynomial);
    }
}

#[test]
fn alternate_variable_orders_remain_equivalent() {
    let polynomial = sum([
        term(&[0, 1, 2], vec![AlgebraicReal::one()]),
        term(&[0, 1, 3], vec![AlgebraicReal::one()]),
        term(&[1, 4], vec![AlgebraicReal::one()]),
    ]);
    let witnesses = witnesses(&polynomial, 0, 1, &mut unlimited());
    assert!(witnesses.len() >= 2);
    for witness in &witnesses {
        assert_eq!(expand(&witness.form, 0, 1), polynomial);
    }
    let mut work = unlimited();
    assert!(witnesses.iter().any(|witness| {
        super::recognize::equal(&witnesses[0].form, &witness.form, &mut work.begin_attempt())
            == Some(false)
    }));
}

#[test]
fn lack_of_shared_variables_produces_no_factor_witness() {
    let polynomial = sum([
        term(&[0, 1], vec![AlgebraicReal::sqrt(2u32)]),
        term(&[2, 3], vec![AlgebraicReal::sqrt(3u32)]),
    ]);
    assert!(witnesses(&polynomial, 0, 1, &mut unlimited()).is_empty());
}

#[test]
fn global_and_local_limits_bound_failed_planning() {
    let polynomial = sum((0..120).map(|i| term(&[0, i + 1], vec![AlgebraicReal::one()])));
    for (total, local) in [(0, 100), (100, 0), (500, 20), (20, 500)] {
        let mut budget = AlgebraicWorkBudget::new(total, local);
        assert!(witnesses(&polynomial, 0, 1, &mut budget).is_empty());
        assert!(budget.consumed <= total);
        assert_eq!(budget.remaining + budget.consumed, total);
    }
}

struct Graph {
    snapshot: Snapshot,
}

impl Graph {
    fn new() -> Self {
        Self {
            snapshot: Snapshot {
                classes: Vec::new(),
                types: Vec::new(),
                values: Vec::new(),
            },
        }
    }

    fn node(&mut self, ty: usize, node: ENode) -> u32 {
        let class = self.snapshot.classes.len() as u32;
        self.snapshot.classes.push(vec![node]);
        self.snapshot.types.push(ty);
        class
    }

    fn variable(&mut self, ty: usize) -> u32 {
        let id = self.snapshot.classes.len() as u32;
        self.node(ty, ENode::Value(ValueId(id)))
    }

    fn form(
        &mut self,
        form: &Form,
        shapes: &[Shape],
        coefficients: &mut Vec<super::ExprTree>,
    ) -> u32 {
        match form {
            Form::Class(class) => *class,
            Form::Coefficient(ty, values) => {
                let id = coefficients.len();
                coefficients.push(super::ExprTree::Coefficient {
                    shape: shapes[*ty].clone(),
                    values: values.clone().into(),
                });
                self.node(*ty, ENode::Coefficient(id))
            }
            Form::Binary(name, ty, left, right) => {
                let left = self.form(left, shapes, coefficients);
                let right = self.form(right, shapes, coefficients);
                let instances = shapes[*ty].num_elements().unwrap().try_into().unwrap();
                let node = match *name {
                    "Add" => ENode::Add {
                        instances,
                        left,
                        right,
                    },
                    "Multiply" => ENode::Multiply {
                        instances,
                        left,
                        right,
                    },
                    _ => panic!("not a polynomial operation"),
                };
                self.node(*ty, node)
            }
            _ => panic!("not a polynomial witness"),
        }
    }

    fn product(&mut self, ty: usize, factors: &[u32]) -> u32 {
        factors
            .iter()
            .copied()
            .reduce(|left, right| {
                self.node(
                    ty,
                    ENode::Multiply {
                        instances: 1,
                        left,
                        right,
                    },
                )
            })
            .unwrap()
    }

    fn sum(&mut self, ty: usize, mut terms: Vec<u32>) -> u32 {
        while terms.len() > 1 {
            terms = terms
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 1 {
                        pair[0]
                    } else {
                        self.node(
                            ty,
                            ENode::Add {
                                instances: 1,
                                left: pair[0],
                                right: pair[1],
                            },
                        )
                    }
                })
                .collect();
        }
        terms[0]
    }
}

#[test]
fn proposal_analysis_covers_a_120_term_root_above_256_structural_units() {
    let mut graph = Graph::new();
    let variables = (0..125).map(|_| graph.variable(0)).collect::<Vec<_>>();
    let terms = (0..120)
        .map(|i| {
            graph.product(
                0,
                &[
                    variables[0],
                    variables[1 + i % 3],
                    variables[4 + i],
                    variables[124],
                ],
            )
        })
        .collect();
    let root = graph.sum(0, terms);
    let polynomial =
        sum((0..120u32).map(|i| term(&[0, 1 + i % 3, 4 + i, 124], vec![AlgebraicReal::one()])));
    let proposals = super::proposals(
        &graph.snapshot,
        &[Shape::scalar(DType::F32)],
        &[],
        MAX_POLYNOMIAL_SIZE,
        &mut unlimited(),
    );
    let proposals = proposals
        .iter()
        .filter(|proposal| proposal.class == root)
        .collect::<Vec<_>>();
    assert!(!proposals.is_empty());
    assert!(matches!(proposals[0].form, Form::Binary("Multiply", ..)));
    for proposal in proposals {
        assert_eq!(expand(&proposal.form, 0, 1), polynomial);
    }
}

#[test]
fn typed_classes_are_not_merged_and_each_root_gets_a_first_factor_witness() {
    let mut graph = Graph::new();
    let mut roots = Vec::new();
    let mut polynomials = Vec::new();
    for ty in 0..3 {
        let v = (0..4).map(|_| graph.variable(ty)).collect::<Vec<_>>();
        let left = graph.product(ty, &[v[0], v[1], v[2]]);
        let right = graph.product(ty, &[v[0], v[1], v[3]]);
        roots.push(graph.sum(ty, vec![left, right]));
        polynomials.push(sum([
            term(&[v[0], v[1], v[2]], vec![AlgebraicReal::one()]),
            term(&[v[0], v[1], v[3]], vec![AlgebraicReal::one()]),
        ]));
    }
    let shapes = [
        Shape::scalar(DType::F32).with_domain_axes([domain_index::FiniteSetId(0)]),
        Shape::scalar(DType::F32).with_domain_axes([domain_index::FiniteSetId(1)]),
        Shape::scalar(DType::F64).with_domain_axes([domain_index::FiniteSetId(0)]),
    ];
    let proposals = super::proposals(&graph.snapshot, &shapes, &[], 4096, &mut unlimited());
    let mut positions = Vec::new();
    for (ty, root) in roots.iter().enumerate() {
        let own = proposals
            .iter()
            .enumerate()
            .filter(|(_, proposal)| proposal.class == *root)
            .collect::<Vec<_>>();
        assert!(own.len() >= 2);
        for (_, proposal) in &own {
            assert_eq!(expand(&proposal.form, ty, 1), polynomials[ty]);
        }
        positions.push((own[0].0, own[1].0));
    }
    for first in &positions {
        for second in &positions {
            assert!(first.0 < second.1);
        }
    }
}

#[test]
fn an_existing_flat_normal_form_is_not_proposed_again() {
    let mut graph = Graph::new();
    let x = graph.variable(0);
    let y = graph.variable(0);
    graph.sum(0, vec![x, y]);
    let proposals = super::proposals(
        &graph.snapshot,
        &[Shape::scalar(DType::F32)],
        &[],
        4096,
        &mut unlimited(),
    );
    assert!(proposals.is_empty());
}

#[test]
fn non_multiquadratic_coefficients_keep_their_exact_roots() {
    let cube_root = AlgebraicReal::integer(2)
        .pow_rational(&1.into(), &3u32.into())
        .unwrap();
    let polynomial = sum([
        term(&[0, 1], vec![cube_root.clone()]),
        term(&[0, 2], vec![AlgebraicReal::sqrt(3u32)]),
    ]);
    let witnesses = witnesses(&polynomial, 0, 1, &mut unlimited());
    assert!(!witnesses.is_empty());
    for witness in witnesses {
        assert_eq!(expand(&witness.form, 0, 1), polynomial);
    }
    assert_eq!(cube_root.degree(), 3);
}

#[test]
fn a_denied_membership_check_cannot_claim_a_witness_is_absent() {
    let mut graph = Graph::new();
    let x = graph.variable(0);
    let y = graph.variable(0);
    let root = graph.sum(0, vec![x, y]);
    let form = Form::binary("Add", 0, Form::Class(x), Form::Class(y));
    let mut budget = AlgebraicWorkBudget::new(100, 1);
    assert_eq!(
        super::recognize::contains(
            &graph.snapshot,
            &[],
            root,
            &form,
            &mut budget.begin_attempt(),
        ),
        None
    );
}

#[test]
fn an_existing_factor_witness_is_not_proposed_again() {
    let mut graph = Graph::new();
    let x = graph.variable(0);
    let y = graph.variable(0);
    let z = graph.variable(0);
    let sum = graph.sum(0, vec![y, z]);
    let root = graph.product(0, &[x, sum]);
    let proposals = super::proposals(
        &graph.snapshot,
        &[Shape::scalar(DType::F32)],
        &[],
        4096,
        &mut unlimited(),
    );
    let mut work = unlimited();
    for proposal in proposals.iter().filter(|proposal| proposal.class == root) {
        assert_eq!(
            super::recognize::contains(
                &graph.snapshot,
                &[],
                root,
                &proposal.form,
                &mut work.begin_attempt(),
            ),
            Some(false)
        );
    }
}

#[test]
fn depth_bounded_quotient_planning_keeps_all_remaining_terms() {
    let polynomial = sum((0..140u32).flat_map(|group| {
        [
            term(&[group * 3, group * 3 + 1], vec![AlgebraicReal::one()]),
            term(&[group * 3, group * 3 + 2], vec![AlgebraicReal::one()]),
        ]
    }));
    let witnesses = witnesses(&polynomial, 0, 1, &mut unlimited());
    assert!(!witnesses.is_empty());
    for witness in witnesses {
        assert_eq!(expand(&witness.form, 0, 1), polynomial);
        assert!(witness.work <= MAX_FORM_WORK);
    }
}

#[test]
fn the_form_reservoir_is_separate_from_polynomial_analysis_size() {
    let left = (0..1101).collect::<Vec<_>>();
    let right = std::iter::once(0).chain(1101..2201).collect::<Vec<_>>();
    let polynomial = sum([
        term(&left, vec![AlgebraicReal::one()]),
        term(&right, vec![AlgebraicReal::one()]),
    ]);
    assert!(materialization_work(&polynomial, 1) > MAX_FORM_WORK);
    let mut work = unlimited();
    assert!(witnesses(&polynomial, 0, 1, &mut work).is_empty());
    assert!(work.consumed < 1 << 26);
}

#[test]
fn later_budget_denials_do_not_revoke_a_completed_witness() {
    let polynomial = sum([
        term(&[0, 1, 2], vec![AlgebraicReal::one()]),
        term(&[0, 1, 3], vec![AlgebraicReal::one()]),
    ]);
    let mut found_partial = false;
    for total in 1..500 {
        let mut budget = AlgebraicWorkBudget::new(total, 1000);
        let forms = witnesses(&polynomial, 0, 1, &mut budget);
        if !forms.is_empty() && budget.abandoned_count() != 0 {
            for witness in forms {
                assert_eq!(expand(&witness.form, 0, 1), polynomial);
            }
            found_partial = true;
            break;
        }
    }
    assert!(found_partial);
}

#[test]
fn more_work_keeps_a_completed_factor_witness() {
    let mut graph = Graph::new();
    let v = (0..4).map(|_| graph.variable(0)).collect::<Vec<_>>();
    let xy = graph.product(0, &[v[0], v[1]]);
    let xyz = graph.product(0, &[xy, v[2]]);
    let xyw = graph.product(0, &[xy, v[3]]);
    let root = graph.sum(0, vec![xyz, xyw]);
    let shapes = [Shape::scalar(DType::F32)];
    let mut completed = false;
    for total in 1..=512 {
        let mut work = AlgebraicWorkBudget::new(total, 10_000);
        let forms = super::proposals(&graph.snapshot, &shapes, &[], 4096, &mut work);
        let found = forms
            .iter()
            .any(|p| p.class == root && matches!(p.form, Form::Binary("Multiply", ..)));
        assert!(
            !completed || found,
            "lost the completed factor witness at total budget {total}"
        );
        completed |= found;
    }
    assert!(completed);
}

#[test]
fn first_witness_slots_precede_second_variant_slots() {
    let mut graph = Graph::new();
    let mut roots = Vec::new();
    for _ in 0..1200 {
        let v = (0..4).map(|_| graph.variable(0)).collect::<Vec<_>>();
        let xy = graph.product(0, &[v[0], v[1]]);
        let xyz = graph.product(0, &[xy, v[2]]);
        let xyw = graph.product(0, &[xy, v[3]]);
        roots.push(graph.sum(0, vec![xyz, xyw]));
    }
    let shapes = [Shape::scalar(DType::F32)];
    let mut work = AlgebraicWorkBudget::new(1 << 28, 1 << 20);
    let forms = super::proposals(&graph.snapshot, &shapes, &[], 4096, &mut work);
    let seen = forms
        .iter()
        .map(|p| p.class)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(forms.len(), 2048);
    assert_eq!(roots.iter().filter(|r| seen.contains(r)).count(), 1200);
    let first_slots = forms[..roots.len()]
        .iter()
        .map(|p| p.class)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(first_slots.len(), roots.len());
    assert!(roots.iter().all(|root| first_slots.contains(root)));
}

#[test]
fn an_overfull_first_round_does_not_spend_slots_on_extra_variants() {
    let mut graph = Graph::new();
    let mut roots = Vec::new();
    for _ in 0..2050 {
        let v = (0..4).map(|_| graph.variable(0)).collect::<Vec<_>>();
        let xy = graph.product(0, &[v[0], v[1]]);
        let xyz = graph.product(0, &[xy, v[2]]);
        let xyw = graph.product(0, &[xy, v[3]]);
        roots.push(graph.sum(0, vec![xyz, xyw]));
    }
    let shapes = [Shape::scalar(DType::F32)];
    let mut work = AlgebraicWorkBudget::new(1 << 28, 1 << 20);
    let forms = super::proposals(&graph.snapshot, &shapes, &[], 4096, &mut work);
    let seen = forms
        .iter()
        .map(|p| p.class)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(forms.len(), 2048);
    assert_eq!(seen.len(), 2048);
    assert!(seen.iter().all(|root| roots.contains(root)));
}

#[test]
fn completed_terminal_witnesses_survive_unrelated_analysis_exhaustion() {
    for depth in [0, 500, 1000] {
        let mut graph = Graph::new();
        let x = graph.variable(0);
        let mut shared = graph.variable(0);
        let zero = graph.node(0, ENode::Coefficient(0));
        let alias = graph.sum(0, vec![x, zero]);
        let annihilated = graph.product(0, &[x, zero]);
        for _ in 0..depth {
            shared = graph.product(0, &[shared, shared]);
        }
        let shape = Shape::scalar(DType::F32);
        let coefficients = [super::ExprTree::Coefficient {
            shape: shape.clone(),
            values: vec![AlgebraicReal::zero()].into(),
        }];
        let mut budget = AlgebraicWorkBudget::new(1 << 16, 256);
        let proposals =
            super::proposals(&graph.snapshot, &[shape], &coefficients, 256, &mut budget);
        assert!(
            proposals.iter().any(|proposal| {
                proposal.class == alias && matches!(proposal.form, Form::Class(id) if id == x)
            }),
            "lost the completed alias with {depth} unrelated levels"
        );
        assert!(
            proposals.iter().any(|proposal| {
                proposal.class == annihilated
                    && matches!(&proposal.form,
                Form::Coefficient(0, values) if values == &[AlgebraicReal::zero()])
            }),
            "lost the completed constant with {depth} unrelated levels"
        );
        assert_eq!(budget.consumed + budget.remaining, 1 << 16);
        if depth == 1000 {
            assert_eq!(budget.remaining, 0);
        }
    }
}

#[test]
fn completed_nonterminal_witnesses_survive_unrelated_analysis_exhaustion() {
    for case in ["canonical_sum", "grouped_factor"] {
        for depth in [0, 500, 1000] {
            let mut graph = Graph::new();
            let v = (0..5).map(|_| graph.variable(0)).collect::<Vec<_>>();
            let mut shared = v[4];
            let shape = Shape::scalar(DType::F32);
            let mut coefficients = Vec::new();
            let (root, expected, operation) = if case == "canonical_sum" {
                coefficients.push(super::ExprTree::Coefficient {
                    shape: shape.clone(),
                    values: vec![AlgebraicReal::integer(2)].into(),
                });
                let two = graph.node(0, ENode::Coefficient(0));
                let pair = graph.sum(0, vec![v[0], v[1]]);
                let twice = graph.product(0, &[two, pair]);
                let negative = graph.node(
                    0,
                    ENode::Negate {
                        instances: 1,
                        value: pair,
                    },
                );
                let expected = sum([
                    term(&[v[0]], vec![AlgebraicReal::one()]),
                    term(&[v[1]], vec![AlgebraicReal::one()]),
                ]);
                (graph.sum(0, vec![twice, negative]), expected, "Add")
            } else {
                let xy = graph.product(0, &[v[0], v[1]]);
                let xyz = graph.product(0, &[xy, v[2]]);
                let xyw = graph.product(0, &[xy, v[3]]);
                let expected = sum([
                    term(&[v[0], v[1], v[2]], vec![AlgebraicReal::one()]),
                    term(&[v[0], v[1], v[3]], vec![AlgebraicReal::one()]),
                ]);
                (graph.sum(0, vec![xyz, xyw]), expected, "Multiply")
            };
            for _ in 0..depth {
                shared = graph.product(0, &[shared, shared]);
            }
            let mut budget = AlgebraicWorkBudget::new(1 << 16, 256);
            let proposals =
                super::proposals(&graph.snapshot, &[shape], &coefficients, 256, &mut budget);
            let own = proposals
                .iter()
                .filter(|proposal| proposal.class == root)
                .collect::<Vec<_>>();
            assert!(!own.is_empty(), "lost {case} with {depth} unrelated levels");
            assert!(matches!(own[0].form, Form::Binary(name, ..) if name == operation));
            for proposal in own {
                assert_eq!(expand(&proposal.form, 0, 1), expected);
            }
            assert_eq!(budget.consumed + budget.remaining, 1 << 16);
            if depth == 1000 {
                assert_eq!(budget.remaining, 0);
            }
        }
    }
}

#[test]
fn flat_first_witness_slots_are_reserved_before_extra_factors() {
    let mut graph = Graph::new();
    let x = graph.variable(0);
    let zero = graph.node(0, ENode::Coefficient(0));
    let alias = graph.sum(0, vec![x, zero]);
    let mut roots = vec![alias];
    for _ in 0..2046 {
        let v = (0..4).map(|_| graph.variable(0)).collect::<Vec<_>>();
        let xy = graph.product(0, &[v[0], v[1]]);
        let xyz = graph.product(0, &[xy, v[2]]);
        let xyw = graph.product(0, &[xy, v[3]]);
        roots.push(graph.sum(0, vec![xyz, xyw]));
    }
    let shape = Shape::scalar(DType::F32);
    let coefficients = [super::ExprTree::Coefficient {
        shape: shape.clone(),
        values: vec![AlgebraicReal::zero()].into(),
    }];
    let mut budget = AlgebraicWorkBudget::new(1 << 28, 1 << 20);
    let proposals = super::proposals(&graph.snapshot, &[shape], &coefficients, 4096, &mut budget);
    let seen = proposals
        .iter()
        .map(|proposal| proposal.class)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(proposals.len(), 2048);
    assert_eq!(seen.len(), roots.len());
    assert!(roots.iter().all(|root| seen.contains(root)));
    assert!(
        proposals[..2047]
            .iter()
            .all(|proposal| matches!(proposal.form, Form::Binary("Multiply", ..)))
    );
    assert_eq!(proposals[2047].class, alias);
    assert!(matches!(proposals[2047].form, Form::Class(id) if id == x));
}

#[test]
fn later_larger_scopes_keep_priority_in_a_full_first_witness_reservoir() {
    let mut graph = Graph::new();
    for _ in 0..2050 {
        let v = (0..4).map(|_| graph.variable(0)).collect::<Vec<_>>();
        let xy = graph.product(0, &[v[0], v[1]]);
        let xyz = graph.product(0, &[xy, v[2]]);
        let xyw = graph.product(0, &[xy, v[3]]);
        graph.sum(0, vec![xyz, xyw]);
    }
    let v = (0..5).map(|_| graph.variable(0)).collect::<Vec<_>>();
    let xyz = graph.product(0, &[v[0], v[1], v[2]]);
    let left = graph.product(0, &[xyz, v[3]]);
    let right = graph.product(0, &[xyz, v[4]]);
    let root = graph.sum(0, vec![left, right]);
    let mut budget = AlgebraicWorkBudget::new(1 << 28, 1 << 20);
    let proposals = super::proposals(
        &graph.snapshot,
        &[Shape::scalar(DType::F32)],
        &[],
        4096,
        &mut budget,
    );
    assert_eq!(proposals.len(), 2048);
    assert_eq!(proposals[0].class, root);
    assert!(matches!(proposals[0].form, Form::Binary("Multiply", ..)));
    let seen = proposals
        .iter()
        .map(|proposal| proposal.class)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(seen.len(), 2048);
}

#[test]
fn virtual_canonical_membership_matches_materialized_balanced_forms() {
    let shapes = [
        Shape::scalar(DType::F32),
        Shape::new(DType::F32, [2]),
        Shape::scalar(DType::F64).with_domain_axes([domain_index::FiniteSetId(3)]),
    ];
    let cube_root = AlgebraicReal::integer(2)
        .pow_rational(&1.into(), &3u32.into())
        .unwrap();
    for (ty, shape) in shapes.iter().enumerate() {
        let elements = shape.num_elements().unwrap();
        for terms in 0..10 {
            for degree in 0..10 {
                for coefficient in [
                    AlgebraicReal::one(),
                    cube_root.clone(),
                    AlgebraicReal::integer(-3),
                ] {
                    let mut graph = Graph::new();
                    let variables = (0..100).map(|_| graph.variable(ty)).collect::<Vec<_>>();
                    let absent = graph.variable(ty);
                    let polynomial = (0..terms).fold(
                        Polynomial::constant(vec![AlgebraicReal::zero(); elements], 4096).unwrap(),
                        |sum, index| {
                            let values = (0..elements)
                                .map(|cell| {
                                    if cell == 0 {
                                        coefficient.clone()
                                    } else {
                                        AlgebraicReal::sqrt(3u32)
                                    }
                                })
                                .collect();
                            sum.add(
                                &term(&variables[index * 10..index * 10 + degree], values),
                                4096,
                                &mut |_| true,
                            )
                            .unwrap()
                        },
                    );
                    let form = super::materialize(&polynomial, ty, elements);
                    let mut coefficients = Vec::new();
                    let root = graph.form(&form, &shapes, &mut coefficients);
                    for (class, expected) in [(root, true), (absent, false)] {
                        let mut work = unlimited();
                        assert_eq!(
                            super::recognize::contains(
                                &graph.snapshot,
                                &coefficients,
                                class,
                                &form,
                                &mut work.begin_attempt()
                            ),
                            Some(expected)
                        );
                        assert_eq!(
                            super::recognize::contains_normal_form(
                                &graph.snapshot,
                                &coefficients,
                                class,
                                &polynomial,
                                &mut work.begin_attempt()
                            ),
                            Some(expected),
                            "ty={ty}, terms={terms}, degree={degree}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn virtual_canonical_membership_charges_constructors_and_coefficient_cells() {
    let mut graph = Graph::new();
    let x = graph.variable(0);
    let y = graph.variable(0);
    let root = graph.sum(0, vec![x, y]);
    let polynomial = sum([
        term(&[x], vec![AlgebraicReal::one()]),
        term(&[y], vec![AlgebraicReal::one()]),
    ]);
    for total in 0..3 {
        let mut budget = AlgebraicWorkBudget::new(total, 100);
        assert_eq!(
            super::recognize::contains_normal_form(
                &graph.snapshot,
                &[],
                root,
                &polynomial,
                &mut budget.begin_attempt()
            ),
            None
        );
        assert!(budget.consumed <= total);
    }
    let values = vec![AlgebraicReal::integer(2), AlgebraicReal::sqrt(3u32)];
    let constant = Polynomial::constant(values.clone(), 4096).unwrap();
    let coefficients = [super::ExprTree::Coefficient {
        shape: Shape::new(DType::F32, [2]),
        values: values.into(),
    }];
    let root = graph.node(1, ENode::Coefficient(0));
    for total in 0..7 {
        let mut budget = AlgebraicWorkBudget::new(total, 100);
        assert_eq!(
            super::recognize::contains_normal_form(
                &graph.snapshot,
                &coefficients,
                root,
                &constant,
                &mut budget.begin_attempt()
            ),
            None
        );
        assert!(budget.consumed <= total);
    }
    let mut work = unlimited();
    assert_eq!(
        super::recognize::contains_normal_form(
            &graph.snapshot,
            &coefficients,
            root,
            &constant,
            &mut work.begin_attempt()
        ),
        Some(true)
    );
}
