// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn a_rejected_or_exhausted_trial_restores_exact_counts_and_totals() {
    let (dag, _, original, proof) = super::super::tests::factored_dag();
    let instances = vec![u64::MAX; dag.len()];
    let after = AccumulationSide::AfterAccumulation;
    let original_roots = [original, original];
    let changes = [(original, proof), (original, proof)];
    let mut completed = false;
    for limit in 0..100 {
        let mut usage =
            Usage::new(&dag, original_roots.map(|id| (id, after)), &instances, None).unwrap();
        let expected =
            Usage::new(&dag, original_roots.map(|id| (id, after)), &instances, None).unwrap();
        let result = usage.improve(
            &dag,
            &changes,
            &instances,
            &mut ExtractionBudget { remaining: limit },
        );
        if result.is_some_and(|score| score.key() < expected.score().key()) {
            completed = true;
            assert_eq!(
                usage.score(),
                evaluate(&dag, [(proof, after)], &instances, None).unwrap()
            );
            assert_eq!(usage.score().physical, 2 * u128::from(u64::MAX));
            assert_eq!(usage.score().placement.after_accumulation, u64::MAX);
            let saved =
                Usage::new(&dag, [(proof, after), (proof, after)], &instances, None).unwrap();
            assert_eq!(
                usage.improve(
                    &dag,
                    &[(proof, original), (proof, original)],
                    &instances,
                    &mut ExtractionBudget::new()
                ),
                Some(expected.score())
            );
            assert_eq!(usage, saved);
        } else {
            assert_eq!(usage, expected);
        }
    }
    assert!(completed);
}

#[test]
fn incremental_scores_match_full_union_with_mixed_sides_and_duplicate_edges() {
    let mut maps = domain_index::FiniteMapGraph::default();
    let set = maps.add_set(1).unwrap();
    let map = maps.add_map(set, set).unwrap();
    let mut dag = ExpressionDag::default();
    let x = dag.intern(ExpressionNode::Value(ValueId(0)));
    let mut nodes = vec![x];
    for i in 0..32 {
        let a = nodes[i / 2];
        let b = nodes[i / 3];
        let node = match i % 4 {
            0 => ExpressionNode::Multiply(a, a),
            1 => ExpressionNode::Sum {
                map,
                axis: 0,
                value: a,
            },
            2 => ExpressionNode::Add(a, b),
            _ => ExpressionNode::Select(x, a, b),
        };
        nodes.push(dag.intern(node));
    }
    let instances = vec![u64::MAX; dag.len()];
    let after = AccumulationSide::AfterAccumulation;
    for i in 0..nodes.len() {
        let originals = [nodes[i], nodes[i / 2], nodes[i]];
        for j in 0..nodes.len() {
            let chosen = [nodes[j], nodes[i / 2], nodes[j]];
            let mut usage =
                Usage::new(&dag, originals.map(|id| (id, after)), &instances, None).unwrap();
            let before = evaluate(&dag, originals.map(|id| (id, after)), &instances, None).unwrap();
            let next = evaluate(&dag, chosen.map(|id| (id, after)), &instances, None).unwrap();
            assert_eq!(usage.score(), before);
            let changes = originals.into_iter().zip(chosen).collect::<Vec<_>>();
            let trial = usage
                .improve(&dag, &changes, &instances, &mut ExtractionBudget::new())
                .unwrap();
            assert_eq!(trial, next);
            assert_eq!(
                usage.score(),
                if next.key() < before.key() {
                    next
                } else {
                    before
                }
            );
        }
    }
}
