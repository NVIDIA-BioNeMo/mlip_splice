// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::FiniteMapGraph;

use super::*;
use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::shape::Shape;

fn value(id: u32) -> ENode {
    ENode::Value(ValueId(id))
}

#[test]
fn projection_keeps_work_on_both_sides_of_accumulation() {
    let map = domain_index::FiniteMapId(0);
    let classes = vec![
        vec![value(0)],
        vec![value(1)],
        vec![ENode::Gather {
            map,
            axis: 0,
            value: 0,
        }],
        vec![ENode::Multiply {
            instances: 8,
            left: 2,
            right: 1,
        }],
        vec![
            ENode::Sum {
                map,
                axis: 0,
                value: 3,
            },
            ENode::Multiply {
                instances: 3,
                left: 0,
                right: 5,
            },
        ],
        vec![ENode::Sum {
            map,
            axis: 0,
            value: 1,
        }],
    ];

    let frontiers = select_candidates::pareto_frontiers(&classes).unwrap();
    let mut work = frontiers[4][AccumulationSide::AfterAccumulation.index()]
        .iter()
        .map(|candidate| candidate.work)
        .collect::<Vec<_>>();
    work.sort_by_key(|work| (work.before_accumulation, work.after_accumulation));

    assert_eq!(
        work,
        [
            FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 3,
            },
            FiniteMapWork {
                before_accumulation: 8,
                after_accumulation: 0,
            },
        ]
    );
    let selected = |partials| {
        work.iter()
            .min_by_key(|candidate| candidate.instantiate(partials))
            .copied()
            .unwrap()
    };
    assert_eq!(selected(1).after_accumulation, 3);
    assert_eq!(selected(3).before_accumulation, 8);
}

#[test]
fn multiple_roots_count_shared_arithmetic_once() {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph.clone());
    builder.begin_func("shared_projection_frontier");
    let coefficient = builder.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let values = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let coefficient = builder.gather_along_map(coefficient, map, 0);
    let product = builder.multiply(coefficient, values);
    let result = builder.sum_along_map(product, map, 0);
    let entry = builder.end_func(vec![result, result]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();
    let original = crate::optimize::map_reduce::rewrite::expression(func, result, &mut 0).unwrap();

    let mut work = extract_roots(func, &graph, &[original.clone(), original])
        .unwrap()
        .into_iter()
        .map(|candidate| candidate.work)
        .collect::<Vec<_>>();
    work.sort_by_key(|work| (work.before_accumulation, work.after_accumulation));

    assert_eq!(
        work,
        [
            FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 3,
            },
            FiniteMapWork {
                before_accumulation: 8,
                after_accumulation: 0,
            },
        ]
    );
}

#[test]
fn work_counts_a_shared_expression_once() {
    let classes = vec![
        vec![value(0)],
        vec![ENode::Add {
            instances: 8,
            left: 0,
            right: 0,
        }],
        vec![ENode::Multiply {
            instances: 8,
            left: 1,
            right: 1,
        }],
    ];

    let frontiers = select_candidates::pareto_frontiers(&classes).unwrap();
    assert_eq!(
        frontiers[2][AccumulationSide::AfterAccumulation.index()][0].work,
        FiniteMapWork {
            before_accumulation: 0,
            after_accumulation: 16,
        }
    );
}

#[test]
fn dependency_worklist_reaches_a_reverse_ordered_dag() {
    const CLASSES: usize = 256;
    let mut classes = (0..CLASSES - 1)
        .map(|class| {
            vec![ENode::Add {
                instances: 1,
                left: (class + 1) as u32,
                right: (CLASSES - 1) as u32,
            }]
        })
        .collect::<Vec<_>>();
    classes.push(vec![value(0)]);

    let frontiers = select_candidates::pareto_frontiers(&classes).unwrap();

    assert_eq!(
        frontiers[0][AccumulationSide::AfterAccumulation.index()][0]
            .work
            .after_accumulation,
        (CLASSES - 1) as u64
    );
}

#[test]
fn costlier_gathered_arithmetic_remains_available_for_sharing() {
    let map = domain_index::FiniteMapId(0);
    let classes = vec![
        vec![value(0)],
        vec![value(1)],
        vec![ENode::Gather {
            map,
            axis: 0,
            value: 0,
        }],
        vec![ENode::Gather {
            map,
            axis: 0,
            value: 1,
        }],
        vec![
            ENode::Add {
                instances: 8,
                left: 2,
                right: 3,
            },
            ENode::Gather {
                map,
                axis: 0,
                value: 5,
            },
        ],
        vec![ENode::Add {
            instances: 3,
            left: 0,
            right: 1,
        }],
    ];

    let frontiers = select_candidates::pareto_frontiers(&classes).unwrap();
    let frontier = &frontiers[4][AccumulationSide::AfterAccumulation.index()];
    assert_eq!(frontier.len(), 2);
    assert_eq!(frontier[0].work.after_accumulation, 3);
    assert_eq!(frontier[1].work.after_accumulation, 8);
}

#[test]
fn bounding_a_frontier_keeps_both_extremes() {
    let mut dag = ExpressionDag::default();
    let mut frontier = Vec::new();
    for before in 0..32 {
        select_candidates::insert_candidate(
            &mut frontier,
            ExtractionCandidate {
                work: FiniteMapWork {
                    before_accumulation: before,
                    after_accumulation: 31 - before,
                },
                tree: dag.intern(ExpressionNode::Value(ValueId(before as u32))),
                operations: HashSet::new(),
                quality: (1, 0, 0),
            },
            &dag,
        );
    }

    assert_eq!(frontier.len(), MAX_FRONTIER);
    assert_eq!(frontier.first().unwrap().work.before_accumulation, 0);
    assert_eq!(frontier.last().unwrap().work.before_accumulation, 31);
}

#[test]
fn cartesian_construction_is_charged_before_expanding_choices() {
    let mut dag = ExpressionDag::default();
    let left = (0..16)
        .map(|id| ExtractionCandidate::value(ValueId(id), &mut dag))
        .collect::<Vec<_>>();
    let right = left.clone();
    let work = select_candidates::cartesian_tree_nodes(&[&left, &right, &left]);
    assert_eq!(work, 4 * 16usize.pow(3));
    let mut budget = ExtractionBudget {
        remaining: work - 1,
    };
    assert_eq!(
        budget.reserve(work),
        Err(ExtractionLimit::ConstructionBudget)
    );
    assert_eq!(budget.remaining, work - 1);
}

#[test]
fn exhausted_construction_budget_refuses_extraction() {
    let classes = vec![
        vec![value(0)],
        vec![ENode::Add {
            instances: 1,
            left: 0,
            right: 0,
        }],
    ];
    let result = select_candidates::pareto_frontiers_with_coefficients(
        &classes,
        &[],
        &mut ExtractionBudget { remaining: 2 },
        &mut ExpressionDag::default(),
    );
    assert!(matches!(result, Err(ExtractionLimit::ConstructionBudget)));
}
