// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn factored_dag() -> (ExpressionDag, Vec<u32>, ExpressionId, ExpressionId) {
    let mut dag = ExpressionDag::default();
    let x = dag.intern(ExpressionNode::Value(ValueId(0)));
    let y = dag.intern(ExpressionNode::Value(ValueId(1)));
    let z = dag.intern(ExpressionNode::Value(ValueId(2)));
    let xy = dag.intern(ExpressionNode::Multiply(x, y));
    let xz = dag.intern(ExpressionNode::Multiply(x, z));
    let original = dag.intern(ExpressionNode::Add(xy, xz));
    let yz = dag.intern(ExpressionNode::Add(y, z));
    let proof = dag.intern(ExpressionNode::Multiply(x, yz));
    (dag, vec![0, 1, 2, 3, 4, 5, 6, 5], original, proof)
}

fn original_candidate(
    dag: &ExpressionDag,
    originals: &[ExpressionId],
    instances: &[u64],
) -> RootExtraction {
    let score = evaluate(
        dag,
        originals
            .iter()
            .map(|&id| (id, AccumulationSide::AfterAccumulation)),
        instances,
        None,
    )
    .unwrap();
    score.candidate(originals.to_vec())
}

#[test]
fn a_completed_shared_root_improvement_survives_tight_work_exhaustion() {
    let (mut dag, classes, original, proof) = factored_dag();
    let immutable = dag
        .nodes()
        .map(|(_, node)| node.clone())
        .collect::<Vec<_>>();
    let originals = [original, original];
    let mut instances = vec![1; dag.len()];
    let mut candidates = vec![original_candidate(&dag, &originals, &instances)];
    // Every budget prefix retains complete roots. Locate the first prefix that
    // finishes an improvement, before the remaining source scan can finish.
    let limit = (0..256)
        .find(|&limit| {
            let mut trial_dag = dag.clone();
            let mut trial_instances = instances.clone();
            let mut trial_candidates =
                vec![original_candidate(&trial_dag, &originals, &trial_instances)];
            Witnesses {
                source_len: 6,
                classes: &classes,
                proofs: &[proof],
            }
            .seed_for_test(
                &originals,
                &mut trial_dag,
                &mut trial_instances,
                &mut trial_candidates,
                &mut ExtractionBudget { remaining: limit },
            );
            assert!(trial_candidates.iter().any(|c| c.roots == originals));
            assert!(trial_candidates.iter().all(|c| c.roots.len() == 2));
            trial_candidates.iter().any(|c| c.roots == [proof, proof])
        })
        .unwrap();
    let mut budget = ExtractionBudget { remaining: limit };
    Witnesses {
        source_len: 6,
        classes: &classes,
        proofs: &[proof],
    }
    .seed_for_test(
        &originals,
        &mut dag,
        &mut instances,
        &mut candidates,
        &mut budget,
    );
    assert_eq!(budget.remaining, 0);
    assert!(
        candidates
            .iter()
            .any(|c| c.roots == originals && c.work.after_accumulation == 3)
    );
    assert!(
        candidates
            .iter()
            .any(|c| c.roots == [proof, proof] && c.work.after_accumulation == 2)
    );
    assert!(candidates.iter().all(|c| c.roots.len() == 2));
    assert_eq!(
        dag.nodes()
            .take(immutable.len())
            .map(|(_, node)| node.clone())
            .collect::<Vec<_>>(),
        immutable
    );
    for mut budget in [
        ExtractionBudget { remaining: 0 },
        ExtractionBudget { remaining: 1 },
    ] {
        Witnesses {
            source_len: 6,
            classes: &classes,
            proofs: &[proof],
        }
        .seed_for_test(
            &originals,
            &mut dag,
            &mut instances,
            &mut candidates,
            &mut budget,
        );
        assert!(candidates.iter().any(|c| c.roots == [proof, proof]));
        assert!(candidates.iter().any(|c| c.roots == originals));
    }
}

#[test]
fn global_shared_work_rejects_a_locally_cheaper_factorization() {
    let (mut dag, classes, original, proof) = factored_dag();
    let originals = [original, ExpressionId(3), ExpressionId(4), original];
    let mut instances = vec![1; dag.len()];
    let after = AccumulationSide::AfterAccumulation;
    assert_eq!(
        evaluate(&dag, [(original, after)], &instances, None)
            .unwrap()
            .placement
            .after_accumulation,
        3
    );
    assert_eq!(
        evaluate(&dag, [(proof, after)], &instances, None)
            .unwrap()
            .placement
            .after_accumulation,
        2
    );
    assert_eq!(
        evaluate(
            &dag,
            [proof, ExpressionId(3), ExpressionId(4)].map(|id| (id, after)),
            &instances,
            None
        )
        .unwrap()
        .placement
        .after_accumulation,
        4
    );
    let mut candidates = vec![original_candidate(&dag, &originals, &instances)];
    Witnesses {
        source_len: 6,
        classes: &classes,
        proofs: &[proof],
    }
    .seed_for_test(
        &originals,
        &mut dag,
        &mut instances,
        &mut candidates,
        &mut ExtractionBudget::new(),
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].roots, originals);
    assert_eq!(candidates[0].work.after_accumulation, 3);
}

#[test]
fn cyclic_classes_and_label_permutations_do_not_change_concrete_witnesses() {
    for labels in [[0, 1], [1, 0]] {
        let mut dag = ExpressionDag::default();
        let x = dag.intern(ExpressionNode::Value(ValueId(0)));
        let zero = dag.intern(ExpressionNode::Coefficient {
            shape: crate::shape::Shape::scalar(crate::dtype::DType::F32),
            values: vec![algebraic_real::AlgebraicReal::zero()].into(),
        });
        let sum = dag.intern(ExpressionNode::Add(x, zero));
        let mut classes = vec![Vec::new(); 2];
        classes[labels[0] as usize] = vec![
            ENode::Value(ValueId(0)),
            ENode::Add {
                instances: 1,
                left: labels[0],
                right: labels[1],
            },
        ];
        classes[labels[1] as usize] = vec![ENode::Coefficient(0)];
        let source_classes = [labels[0], labels[1], labels[0]];
        let candidates = extract(
            &classes,
            &[labels[0], labels[0]],
            &[sum, sum],
            Witnesses {
                source_len: 3,
                classes: &source_classes,
                proofs: &[x],
            },
            &[1, 1],
            &[zero],
            &mut dag,
        );
        assert!(candidates.iter().any(|c| c.roots == [sum, sum]));
        assert!(
            candidates
                .iter()
                .any(|c| c.roots == [x, x] && c.work == FiniteMapWork::default())
        );
        for (id, node) in dag.nodes() {
            assert!(node.children().all(|child| child.0 < id.0));
        }
    }
}

#[test]
fn proof_selection_is_invariant_under_class_label_permutations() {
    for labels in [
        [0, 1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1, 0],
        [3, 0, 6, 1, 4, 2, 5],
    ] {
        let (mut dag, classes, original, proof) = factored_dag();
        let classes = classes
            .into_iter()
            .map(|class| labels[class as usize])
            .collect::<Vec<_>>();
        let mut instances = vec![1; dag.len()];
        let mut candidates = vec![original_candidate(&dag, &[original], &instances)];
        Witnesses {
            source_len: 6,
            classes: &classes,
            proofs: &[proof],
        }
        .seed_for_test(
            &[original],
            &mut dag,
            &mut instances,
            &mut candidates,
            &mut ExtractionBudget::new(),
        );
        assert!(
            candidates
                .iter()
                .any(|c| c.roots == [proof] && c.work.after_accumulation == 2)
        );
    }
}

impl Witnesses<'_> {
    fn seed_for_test(
        &self,
        originals: &[ExpressionId],
        dag: &mut ExpressionDag,
        instances: &mut Vec<u64>,
        candidates: &mut Vec<RootExtraction>,
        budget: &mut ExtractionBudget,
    ) {
        let usage = Usage::new(
            dag,
            originals
                .iter()
                .map(|&id| (id, AccumulationSide::AfterAccumulation)),
            instances,
            None,
        )
        .unwrap();
        self.seed(originals, dag, instances, candidates, usage, budget);
    }
}

#[test]
fn mixed_context_roots_separate_placement_from_the_emitted_arithmetic_union() {
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;
    use crate::ops::Multiply;
    use crate::optimize::map_reduce::rewrite::{original_work, physical_work};
    use crate::optimize::rebuild::Rebuilder;
    use crate::shape::Shape;

    let mut maps = domain_index::FiniteMapGraph::default();
    let fine = maps.add_set(8).unwrap();
    let coarse = maps.add_set(3).unwrap();
    let singleton = maps.add_set(1).unwrap();
    let map = maps.add_map(fine, coarse).unwrap();
    let outer = maps.add_map(coarse, singleton).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(maps.clone());
    builder.begin_func("mixed_contexts");
    let x = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let u = builder.multiply(x, x);
    let sum = builder.sum_along_map(u, map, 0);
    let nested = builder.sum_along_map(sum, outer, 0);
    let entry = builder.end_func(vec![u, sum, nested, u]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let source = expression_dag::import(func).unwrap().unwrap();
    let roots = [u, sum, nested, u].map(|v| source.values[v.index()].unwrap());
    let mut instances = vec![0; source.dag.len()];
    instances[roots[0].0] = 8;
    let after = AccumulationSide::AfterAccumulation;
    let placement = FiniteMapWork {
        before_accumulation: 8,
        after_accumulation: 8,
    };
    for roots in [
        roots.to_vec(),
        roots.into_iter().rev().collect(),
        roots[..2].to_vec(),
    ] {
        let score = evaluate(
            &source.dag,
            roots.iter().map(|&id| (id, after)),
            &instances,
            None,
        )
        .unwrap();
        assert_eq!(score.placement, placement);
        assert_eq!(score.placement.instantiate(1), 16);
        assert_eq!(score.placement.instantiate(3), 32);
        assert_eq!(score.physical, 8);
        let usage = Usage::new(
            &source.dag,
            roots.iter().map(|&id| (id, after)),
            &instances,
            None,
        )
        .unwrap();
        assert_eq!(score, usage.score());
    }
    assert_eq!(original_work(func, &maps).unwrap(), placement);
    assert_eq!(physical_work(func, &maps).unwrap(), 8);
    let mut rebuilt = Rebuilder::new(func);
    let mut emitted = HashMap::new();
    for (&value, id) in func.results.iter().zip(roots) {
        let result =
            expression_dag::emit(&mut rebuilt, &maps, &source.dag, id, &mut emitted).unwrap();
        rebuilt.alias(&[value], &[result]).unwrap();
    }
    let emitted = rebuilt.finish(func).unwrap();
    assert_eq!(
        emitted
            .nodes
            .iter()
            .filter(|node| node.op.as_any().is::<Multiply>())
            .count(),
        1
    );
    assert_eq!(physical_work(&emitted, &maps).unwrap(), 8);
    assert_eq!(original_work(&emitted, &maps).unwrap(), placement);
    let mut result = module.clone();
    result.funcs[entry.index()] = emitted;
    xir::verify::verify(&result).unwrap();
}

#[test]
fn cost_directed_selection_keeps_only_the_globally_useful_subset_of_proofs() {
    let mut dag = ExpressionDag::default();
    let values = (0..6)
        .map(|i| dag.intern(ExpressionNode::Value(ValueId(i))))
        .collect::<Vec<_>>();
    let ab = dag.intern(ExpressionNode::Multiply(values[0], values[1]));
    let ac = dag.intern(ExpressionNode::Multiply(values[0], values[2]));
    let first = dag.intern(ExpressionNode::Add(ab, ac));
    let de = dag.intern(ExpressionNode::Multiply(values[3], values[4]));
    let df = dag.intern(ExpressionNode::Multiply(values[3], values[5]));
    let second = dag.intern(ExpressionNode::Add(de, df));
    let source_len = dag.len();
    let bc = dag.intern(ExpressionNode::Add(values[1], values[2]));
    let first_proof = dag.intern(ExpressionNode::Multiply(values[0], bc));
    let ef = dag.intern(ExpressionNode::Add(values[4], values[5]));
    let second_proof = dag.intern(ExpressionNode::Multiply(values[3], ef));
    let mut classes = (0..dag.len() as u32).collect::<Vec<_>>();
    classes[first_proof.0] = classes[first.0];
    classes[second_proof.0] = classes[second.0];
    let originals = [first, second, de, df];
    let mut instances = vec![1; dag.len()];
    let mut candidates = vec![original_candidate(&dag, &originals, &instances)];
    Witnesses {
        source_len,
        classes: &classes,
        proofs: &[first_proof, second_proof],
    }
    .seed_for_test(
        &originals,
        &mut dag,
        &mut instances,
        &mut candidates,
        &mut ExtractionBudget::new(),
    );
    assert!(
        candidates
            .iter()
            .any(|c| c.roots == [first_proof, second, de, df] && c.physical_work == 5)
    );
    assert!(!candidates.iter().any(|c| c.roots[1] == second_proof));
}

#[test]
fn cost_directed_selection_finds_the_best_of_three_proofs_in_every_order() {
    let mut dag = ExpressionDag::default();
    let values = (0..4)
        .map(|i| dag.intern(ExpressionNode::Value(ValueId(i))))
        .collect::<Vec<_>>();
    let ab = dag.intern(ExpressionNode::Multiply(values[0], values[1]));
    let ac = dag.intern(ExpressionNode::Multiply(values[0], values[2]));
    let ad = dag.intern(ExpressionNode::Multiply(values[0], values[3]));
    let abac = dag.intern(ExpressionNode::Add(ab, ac));
    let original = dag.intern(ExpressionNode::Add(abac, ad));
    let source_len = dag.len();
    let bc = dag.intern(ExpressionNode::Add(values[1], values[2]));
    let abc = dag.intern(ExpressionNode::Multiply(values[0], bc));
    let first = dag.intern(ExpressionNode::Add(abc, ad));
    let bcd = dag.intern(ExpressionNode::Add(bc, values[3]));
    let middle = dag.intern(ExpressionNode::Multiply(values[0], bcd));
    let cd = dag.intern(ExpressionNode::Add(values[2], values[3]));
    let acd = dag.intern(ExpressionNode::Multiply(values[0], cd));
    let last = dag.intern(ExpressionNode::Add(ab, acd));
    let mut classes = (0..dag.len() as u32).collect::<Vec<_>>();
    for proof in [first, middle, last] {
        classes[proof.0] = classes[original.0];
    }
    for proofs in [
        [first, middle, last],
        [first, last, middle],
        [middle, first, last],
        [middle, last, first],
        [last, first, middle],
        [last, middle, first],
    ] {
        let mut dag = dag.clone();
        let mut instances = vec![1; dag.len()];
        let mut candidates = vec![original_candidate(&dag, &[original], &instances)];
        Witnesses {
            source_len,
            classes: &classes,
            proofs: &proofs,
        }
        .seed_for_test(
            &[original],
            &mut dag,
            &mut instances,
            &mut candidates,
            &mut ExtractionBudget::new(),
        );
        let best = candidates
            .iter()
            .min_by_key(|candidate| candidate.physical_work)
            .unwrap();
        assert_eq!(best.roots, [middle]);
        assert_eq!(best.physical_work, 3);
    }
}

#[test]
fn complete_root_pruning_preserves_physical_and_placement_tradeoffs() {
    let mut dag = ExpressionDag::default();
    let xs = (0..9)
        .map(|i| dag.intern(ExpressionNode::Value(ValueId(i))))
        .collect::<Vec<_>>();
    fn add_chain(dag: &mut ExpressionDag, values: &[ExpressionId]) -> ExpressionId {
        values
            .iter()
            .copied()
            .reduce(|a, b| dag.intern(ExpressionNode::Add(a, b)))
            .unwrap()
    }
    let q = add_chain(&mut dag, &xs);
    let nx0 = dag.intern(ExpressionNode::Negate(xs[0]));
    let nx1 = dag.intern(ExpressionNode::Negate(xs[1]));
    let f = dag.intern(ExpressionNode::Add(q, nx0));
    let g = dag.intern(ExpressionNode::Add(q, nx1));
    let sg = dag.intern(ExpressionNode::Sum {
        map: domain_index::FiniteMapId(0),
        axis: 0,
        value: g,
    });
    let f_short = add_chain(&mut dag, &xs[1..]);
    let mut ys = vec![xs[0]];
    ys.extend_from_slice(&xs[2..]);
    let g_short = add_chain(&mut dag, &ys);
    let sg_short = dag.intern(ExpressionNode::Sum {
        map: domain_index::FiniteMapId(0),
        axis: 0,
        value: g_short,
    });
    let shared = vec![f, sg];
    let unshared = vec![f_short, sg_short];
    let instances = vec![1; dag.len()];
    let score_for = |roots: &[ExpressionId]| {
        evaluate(
            &dag,
            roots
                .iter()
                .map(|&id| (id, AccumulationSide::AfterAccumulation)),
            &instances,
            None,
        )
        .unwrap()
    };
    let a = score_for(&shared);
    let b = score_for(&unshared);
    assert_eq!((a.physical, b.physical), (12, 14));
    assert_eq!(
        (
            a.placement.before_accumulation,
            a.placement.after_accumulation
        ),
        (10, 10)
    );
    assert_eq!(
        (
            b.placement.before_accumulation,
            b.placement.after_accumulation
        ),
        (7, 7)
    );
    for choices in [
        [(shared.clone(), a), (unshared.clone(), b)],
        [(unshared.clone(), b), (shared.clone(), a)],
    ] {
        let mut frontier = Vec::new();
        for (roots, score) in choices {
            super::insert_combined_roots(
                &mut frontier,
                CombinedRoots {
                    extraction: score.candidate(roots),
                    operations: vec![],
                },
                true,
                &dag,
            );
        }
        assert_eq!(frontier.len(), 2);
        assert!(frontier.iter().any(|c| c.extraction.roots == shared));
        assert!(frontier.iter().any(|c| c.extraction.roots == unshared));
    }
}

#[test]
fn root_combination_matches_physical_and_context_unions_across_bitset_words() {
    let map = domain_index::FiniteMapId(0);
    for count in [1, u64::MAX] {
        let mut classes = vec![vec![ENode::Value(ValueId(0))]];
        for i in 1..=70 {
            classes.push(vec![ENode::Multiply {
                instances: count,
                left: i - 1,
                right: i - 1,
            }]);
        }
        classes.push(vec![ENode::Sum {
            map,
            axis: 0,
            value: 70,
        }]);
        for roots in [vec![70], vec![71], vec![70, 71, 70], vec![71, 70, 71]] {
            let mut dag = ExpressionDag::default();
            let frontier = refine_snapshot_roots(classes.clone(), roots, &[], &mut dag).unwrap();
            assert_eq!(frontier.len(), 1);
            let instances = vec![count; dag.len()];
            let candidate = &frontier[0];
            let score = evaluate(
                &dag,
                candidate
                    .roots
                    .iter()
                    .map(|&id| (id, AccumulationSide::AfterAccumulation)),
                &instances,
                None,
            )
            .unwrap();
            assert_eq!(candidate.physical_work, 70 * u128::from(count));
            assert_eq!(candidate.physical_work, score.physical);
            assert_eq!(candidate.work, score.placement);
            assert_eq!(candidate.quality, score.quality);
        }
    }
}

#[test]
fn completed_placement_tradeoff_is_retained_when_trial_uses_last_budget_unit() {
    let map = domain_index::FiniteMapId(0);
    let mut dag = ExpressionDag::default();
    let c = dag.intern(ExpressionNode::Value(ValueId(0)));
    let x = dag.intern(ExpressionNode::Value(ValueId(1)));
    let sum = dag.intern(ExpressionNode::Sum {
        map,
        axis: 0,
        value: x,
    });
    let original = dag.intern(ExpressionNode::Multiply(c, sum));
    let source_len = dag.len();
    let gathered = dag.intern(ExpressionNode::Gather {
        map,
        axis: 0,
        value: c,
    });
    let product = dag.intern(ExpressionNode::Multiply(gathered, x));
    let proof = dag.intern(ExpressionNode::Sum {
        map,
        axis: 0,
        value: product,
    });
    let mut classes = (0..dag.len() as u32).collect::<Vec<_>>();
    classes[proof.0] = classes[original.0];
    let mut found = false;
    for limit in 0..256 {
        let mut dag = dag.clone();
        let mut instances = vec![0; dag.len()];
        instances[original.0] = 3;
        instances[product.0] = 8;
        let mut frontier = vec![original_candidate(&dag, &[original], &instances)];
        let mut budget = ExtractionBudget { remaining: limit };
        Witnesses {
            source_len,
            classes: &classes,
            proofs: &[proof],
        }
        .seed_for_test(
            &[original],
            &mut dag,
            &mut instances,
            &mut frontier,
            &mut budget,
        );
        assert!(frontier.iter().any(|c| c.roots == [original]));
        if let Some(candidate) = frontier.iter().find(|c| c.roots == [proof]) {
            assert_eq!(candidate.physical_work, 8);
            assert_eq!(
                candidate.work,
                FiniteMapWork {
                    before_accumulation: 8,
                    after_accumulation: 0
                }
            );
            assert_eq!(
                frontier
                    .iter()
                    .min_by_key(|c| c.physical_work)
                    .unwrap()
                    .roots,
                [original]
            );
            assert_eq!(
                frontier
                    .iter()
                    .min_by_key(|c| c.work.instantiate(3))
                    .unwrap()
                    .roots,
                [proof]
            );
            if budget.remaining == 0 {
                found = true;
                break;
            }
        }
    }
    assert!(found);
}
