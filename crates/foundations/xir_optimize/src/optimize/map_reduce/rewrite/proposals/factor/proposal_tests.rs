// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dtype::DType;
use crate::ir::ValueId;

#[test]
fn attempt_reservations_are_cumulative_atomic_and_count_failure_once() {
    let mut work = AlgebraicWorkBudget::new(20, 5);
    {
        let mut attempt = work.begin_attempt();
        assert!(attempt.reserve(3));
        assert!(!attempt.reserve(3));
        assert!(!attempt.reserve(1));
        assert!(attempt.failed());
    }
    assert_eq!(work.consumed, 3);
    assert_eq!(work.remaining, 17);
    assert_eq!(work.abandoned_count(), 1);
    {
        let mut attempt = work.begin_attempt();
        assert!(attempt.reserve(5));
    }
    assert_eq!(work.consumed, 8);
    assert_eq!(work.abandoned_count(), 1);
}

#[test]
fn a_global_denial_does_not_spend_either_allowance() {
    let mut work = AlgebraicWorkBudget::new(4, 10);
    {
        let mut attempt = work.begin_attempt();
        assert!(attempt.reserve(3));
        assert!(!attempt.reserve(2));
        assert_eq!(attempt.remaining, 7);
    }
    assert_eq!(work.remaining, 1);
    assert_eq!(work.consumed, 3);
    assert_eq!(work.abandoned_count(), 1);
    {
        let mut attempt = work.begin_attempt();
        assert!(attempt.reserve(1));
        assert!(!attempt.reserve(1));
    }
    assert_eq!(work.consumed, 4);
    assert_eq!(work.abandoned_count(), 2);
}

fn snapshot() -> Snapshot {
    Snapshot {
        classes: vec![
            vec![ENode::Value(ValueId(0))],
            vec![ENode::Value(ValueId(1))],
            vec![ENode::Add {
                instances: 1,
                left: 0,
                right: 1,
            }],
        ],
        types: vec![0; 3],
        values: Vec::new(),
    }
}

fn expansion<'a>(snapshot: &'a Snapshot, shapes: &'a [Shape]) -> Expansion<'a> {
    Expansion {
        snapshot,
        shapes,
        coefficients: &[],
        max_terms: 256,
        active: vec![false; 3],
        cache: vec![None; 3],
    }
}

#[test]
fn expansion_and_materialization_share_one_local_allowance() {
    let snapshot = snapshot();
    let shapes = [Shape::scalar(DType::F32)];
    let mut measurement = AlgebraicWorkBudget::new(1000, 1000);
    expansion(&snapshot, &shapes)
        .expand(2, 0, &mut measurement.begin_attempt())
        .unwrap();
    let expansion_work = measurement.consumed;
    let mut expansion = expansion(&snapshot, &shapes);
    let mut work = AlgebraicWorkBudget::new(1000, expansion_work);
    {
        let mut attempt = work.begin_attempt();
        let polynomial = expansion.expand(2, 0, &mut attempt).unwrap();
        assert!(!attempt.reserve(materialization_work(&polynomial, 1)));
    }
    assert_eq!(work.consumed, expansion_work);
    assert_eq!(work.abandoned_count(), 1);
    {
        let mut attempt = work.begin_attempt();
        let polynomial = expansion.expand(2, 0, &mut attempt).unwrap();
        assert_eq!(
            attempt.remaining, expansion_work,
            "a cached polynomial is free"
        );
        assert!(attempt.reserve(materialization_work(&polynomial, 1)));
        assert!(matches!(
            materialize(&polynomial, 0, 1),
            Form::Binary("Add", ..)
        ));
    }
    assert_eq!(work.consumed, expansion_work + 3);
    assert_eq!(work.abandoned_count(), 1);
}

#[test]
fn a_failed_expansion_keeps_only_completed_cache_entries() {
    let snapshot = snapshot();
    let shapes = [Shape::scalar(DType::F32)];
    let mut expansion = expansion(&snapshot, &shapes);
    let mut work = AlgebraicWorkBudget::new(100, 5);
    {
        let mut attempt = work.begin_attempt();
        assert!(expansion.expand(2, 0, &mut attempt).is_none());
        assert!(attempt.failed());
    }
    assert!(expansion.cache[0].is_some());
    assert!(expansion.cache[1].is_none());
    assert!(expansion.cache[2].is_none());
    assert!(expansion.active.iter().all(|active| !active));
    assert_eq!(work.consumed, 5);
    assert_eq!(work.abandoned_count(), 1);
}

fn polynomial() -> Polynomial {
    let root = AlgebraicReal::sqrt(2u32);
    let left = Polynomial::constant(vec![root.clone()], 256)
        .unwrap()
        .multiply(&Polynomial::variable(0, 1, 256).unwrap(), 256, &mut |_| {
            true
        })
        .unwrap();
    let right = Polynomial::constant(vec![&root * &AlgebraicReal::integer(3)], 256)
        .unwrap()
        .multiply(&Polynomial::variable(1, 1, 256).unwrap(), 256, &mut |_| {
            true
        })
        .unwrap();
    left.add(&right, 256, &mut |_| true).unwrap()
}

#[test]
fn a_failed_factoring_attempt_keeps_the_normal_form() {
    let polynomial = polynomial();
    let materialization = materialization_work(&polynomial, 1);
    let mut work = AlgebraicWorkBudget::new(100, materialization);
    let normal = {
        let mut attempt = work.begin_attempt();
        assert!(attempt.reserve(materialization));
        materialize(&polynomial, 0, 1)
    };
    {
        let mut attempt = work.begin_attempt();
        assert!(factored_form(&polynomial, 0, 1, 256, &mut attempt).is_none());
        assert!(attempt.failed());
    }
    assert!(matches!(normal, Form::Binary("Add", ..)));
    assert_eq!(work.abandoned_count(), 1);
    assert!(work.consumed <= materialization * 2);
}

#[test]
fn coefficient_arithmetic_charges_before_each_operation() {
    let polynomial = polynomial();
    let mut charges = 0;
    assert!(
        polynomial
            .add(&polynomial, 256, &mut |cost| {
                assert!(cost > 0);
                charges += 1;
                false
            })
            .is_none()
    );
    assert_eq!(charges, 1);
    charges = 0;
    assert!(
        polynomial
            .multiply(&polynomial, 256, &mut |cost| {
                assert!(cost > 0);
                charges += 1;
                charges < 2
            })
            .is_none()
    );
    assert_eq!(charges, 2);
}

#[test]
fn both_public_reports_include_abandoned_proof_attempts() {
    let mut builder = crate::builder::ModuleBuilder::new();
    builder.begin_func("proof_budget_report");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let sum = builder.add(x, y);
    let entry = builder.end_func(vec![sum]);
    let module = builder.finish(entry).unwrap();
    let (_, report) =
        crate::optimize::extract_algebraic_frontier_with_limits(&module, 100, 5).unwrap();
    assert_eq!(report.abandoned_algebraic_expressions, 2);
    let report =
        crate::optimize::PassManager::new([crate::optimize::Pass::IndexedAlgebraCanonicalization])
            .with_max_algebraic_work(100)
            .with_max_algebraic_expression_work(5)
            .run(&mut module.clone())
            .unwrap();
    assert_eq!(report.abandoned_algebraic_expressions, 1);
}
