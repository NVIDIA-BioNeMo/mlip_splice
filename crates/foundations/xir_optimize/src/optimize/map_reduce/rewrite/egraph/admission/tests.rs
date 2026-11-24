// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::builder::ModuleBuilder;
use crate::dtype::DType;

fn scalar_module() -> crate::ir::Module {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("admission");
    let x = builder.param(Shape::scalar(DType::F32));
    let sum = builder.add(x, x);
    let entry = builder.end_func(vec![sum]);
    builder.finish(entry).unwrap()
}

fn doubled(class: u32, ty: usize) -> Form {
    Form::binary(
        "Multiply",
        ty,
        Form::Coefficient(ty, vec![AlgebraicReal::integer(2)]),
        Form::Class(class),
    )
}

#[test]
fn whole_plan_failure_leaves_every_pool_unchanged() {
    let module = scalar_module();
    let mut dag = ExpressionDag::default();
    let x = dag.intern(ExpressionNode::Value(module.entry_func().params[0]));
    let sum = dag.intern(ExpressionNode::Add(x, x));
    let mut graph = RewriteGraph::new(module.entry_func(), &module.finite_map_graph).unwrap();
    graph.import(&dag).unwrap();
    let (snapshot, classes) = graph.snapshot_dag(&dag).unwrap();
    let coefficients = graph.coefficients.clone();
    #[expect(
        clippy::mutable_key_type,
        reason = "AlgebraicReal caches preserve exact equality and hash"
    )]
    let coefficient_ids = graph.coefficient_ids.clone();
    let shapes = graph.shapes.clone();
    let shape_ids = graph.shape_ids.clone();
    let original_nodes = dag
        .nodes()
        .map(|(_, node)| node.clone())
        .collect::<Vec<_>>();
    for (capacity, work) in [(1, 100), (10, 2)] {
        let mut admission = Admission::new(
            &snapshot,
            &graph.shapes,
            &graph.coefficients,
            graph.maps,
            capacity,
        );
        assert!(
            !admission
                .plan_with_work(
                    Proposal {
                        class: classes[sum.0],
                        form: doubled(classes[x.0], snapshot.types[classes[x.0] as usize])
                    },
                    &mut TraversalBudget { remaining: work }
                )
                .unwrap()
        );
        assert!(admission.is_empty());
        assert_eq!(admission.reserved_keys(), 0);
        let plan = admission.finish();
        assert!(
            plan.commit(&mut graph, &snapshot, &classes, &mut dag)
                .unwrap()
                .is_empty()
        );
        assert!(graph.snapshot().unwrap() == snapshot);
        assert_eq!(graph.coefficients, coefficients);
        assert_eq!(graph.coefficient_ids, coefficient_ids);
        assert_eq!(graph.shapes, shapes);
        assert_eq!(graph.shape_ids, shape_ids);
        assert_eq!(
            dag.nodes()
                .map(|(_, node)| node.clone())
                .collect::<Vec<_>>(),
            original_nodes
        );
    }
}

#[test]
fn unfolded_visits_and_unique_constructor_residency_have_distinct_units() {
    let module = scalar_module();
    let mut dag = ExpressionDag::default();
    let x = dag.intern(ExpressionNode::Value(module.entry_func().params[0]));
    let sum = dag.intern(ExpressionNode::Add(x, x));
    let mut graph = RewriteGraph::new(module.entry_func(), &module.finite_map_graph).unwrap();
    graph.import(&dag).unwrap();
    let (snapshot, classes) = graph.snapshot_dag(&dag).unwrap();
    let ty = snapshot.types[classes[x.0] as usize];
    let form = || {
        let double_negation = || {
            Form::Unary(
                "Negate",
                ty,
                Box::new(Form::Unary(
                    "Negate",
                    ty,
                    Box::new(Form::Class(classes[x.0])),
                )),
            )
        };
        Form::binary("Add", ty, double_negation(), double_negation())
    };
    assert_eq!(form_work(&form()), 7);
    let mut admission =
        Admission::new(&snapshot, &graph.shapes, &graph.coefficients, graph.maps, 3);
    let mut work = TraversalBudget { remaining: 6 };
    assert!(
        !admission
            .plan_with_work(
                Proposal {
                    class: classes[sum.0],
                    form: form()
                },
                &mut work
            )
            .unwrap()
    );
    assert_eq!(admission.reserved_keys(), 0);
    for _ in 0..2 {
        let mut work = TraversalBudget { remaining: 7 };
        assert!(
            admission
                .plan_with_work(
                    Proposal {
                        class: classes[sum.0],
                        form: form()
                    },
                    &mut work
                )
                .unwrap()
        );
        assert_eq!(work.remaining, 0);
        assert_eq!(admission.reserved_keys(), 3);
    }
    let plan = admission.finish();
    let proofs = plan
        .commit(&mut graph, &snapshot, &classes, &mut dag)
        .unwrap();
    assert_eq!(proofs[0], proofs[1]);
    assert_eq!(dag.len(), 5);
    assert_eq!(
        graph
            .snapshot()
            .unwrap()
            .classes
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        5
    );
}

#[test]
fn completed_polynomial_proofs_survive_later_search_exhaustion() {
    for terminal in [false, true] {
        let module = scalar_module();
        let mut dag = ExpressionDag::default();
        let x = dag.intern(ExpressionNode::Value(module.entry_func().params[0]));
        let sum = dag.intern(ExpressionNode::Add(x, x));
        let mut graph = RewriteGraph::new(module.entry_func(), &module.finite_map_graph).unwrap();
        graph.import(&dag).unwrap();
        let (snapshot, classes) = graph.snapshot_dag(&dag).unwrap();
        let mut budget = AlgebraicWorkBudget::new(1 << 16, 4096);
        let proposals = crate::optimize::map_reduce::rewrite::proposals::proposals(
            &snapshot,
            &graph.shapes,
            &graph.coefficients,
            256,
            &mut budget,
        );
        assert!(!proposals.is_empty());
        if terminal {
            let remaining = budget.remaining;
            assert!(budget.reserve((99, 99), remaining.min(4096)));
            while budget.remaining > 0 {
                let remaining = budget.remaining.min(4096);
                assert!(budget.reserve((99, 99), remaining));
            }
            assert_eq!(budget.remaining, 0);
        } else {
            assert!(!budget.begin_attempt().reserve(4097));
            assert!(budget.remaining > 0);
        }
        let mut admission = Admission::new(
            &snapshot,
            &graph.shapes,
            &graph.coefficients,
            graph.maps,
            MAX_ADDED_NODES,
        );
        for proposal in proposals {
            assert!(admission.plan(proposal).unwrap());
        }
        let plan = admission.finish();
        let proofs = plan
            .commit(&mut graph, &snapshot, &classes, &mut dag)
            .unwrap();
        assert!(!proofs.is_empty());
        let (_, final_classes) = graph.snapshot_dag(&dag).unwrap();
        assert!(
            proofs
                .iter()
                .any(|id| final_classes[id.0] == final_classes[sum.0])
        );
        assert!(
            proofs
                .iter()
                .any(|&id| matches!(dag.node(id), ExpressionNode::Multiply(..)))
        );
    }
}

#[test]
fn an_oversized_later_plan_cannot_revoke_or_pollute_a_complete_plan() {
    let module = scalar_module();
    let mut dag = ExpressionDag::default();
    let x = dag.intern(ExpressionNode::Value(module.entry_func().params[0]));
    let sum = dag.intern(ExpressionNode::Add(x, x));
    let mut graph = RewriteGraph::new(module.entry_func(), &module.finite_map_graph).unwrap();
    graph.import(&dag).unwrap();
    let (snapshot, classes) = graph.snapshot_dag(&dag).unwrap();
    let ty = snapshot.types[classes[x.0] as usize];
    let mut admission =
        Admission::new(&snapshot, &graph.shapes, &graph.coefficients, graph.maps, 3);
    assert!(
        admission
            .plan(Proposal {
                class: classes[sum.0],
                form: doubled(classes[x.0], ty)
            })
            .unwrap()
    );
    assert!(
        !admission
            .plan(Proposal {
                class: classes[x.0],
                form: Form::Unary(
                    "Negate",
                    ty,
                    Box::new(Form::Unary(
                        "Negate",
                        ty,
                        Box::new(Form::Class(classes[x.0]))
                    ))
                )
            })
            .unwrap()
    );
    assert_eq!(admission.reserved_keys(), 2);
    let plan = admission.finish();
    let proofs = plan
        .commit(&mut graph, &snapshot, &classes, &mut dag)
        .unwrap();
    assert_eq!(proofs.len(), 1);
    assert_eq!(dag.len(), 4);
    assert_eq!(graph.coefficients.len(), 1);
    assert!(
        !dag.nodes()
            .any(|(_, node)| matches!(node, ExpressionNode::Negate(_)))
    );
}

#[test]
fn select_admission_uses_the_operation_type_rules_without_partial_mutation() {
    let mut maps = domain_index::FiniteMapGraph::default();
    let a = maps.add_set(3).unwrap();
    let b = maps.add_set(3).unwrap();
    for dims in [vec![], vec![2], vec![2, 3]] {
        let branch = Shape::new(DType::F32, dims.clone()).with_domain_axes([a, b]);
        let predicates = [
            (
                Shape::new(DType::F32, dims.clone()).with_domain_axes([a, b]),
                false,
            ),
            (Shape::scalar(DType::Bool).with_domain_axes([a, b]), true),
            (
                Shape::new(DType::Bool, dims.clone()).with_domain_axes([a, b]),
                true,
            ),
            (Shape::scalar(DType::Bool).with_domain_axes([b, a]), false),
            (Shape::new(DType::Bool, [6]).with_domain_axes([a, b]), false),
            (Shape::scalar(DType::Bool), false),
        ];
        for (predicate, valid) in predicates {
            let mut builder = ModuleBuilder::with_finite_map_graph(maps.clone());
            builder.begin_func("select_types");
            let p = builder.param(predicate);
            let x = builder.param(branch.clone());
            let entry = builder.end_func(vec![x]);
            let module = builder.finish(entry).unwrap();
            let mut dag = ExpressionDag::default();
            let p = dag.intern(ExpressionNode::Value(p));
            let x = dag.intern(ExpressionNode::Value(x));
            let mut graph = RewriteGraph::new(module.entry_func(), &maps).unwrap();
            graph.import(&dag).unwrap();
            let (snapshot, classes) = graph.snapshot_dag(&dag).unwrap();
            let shapes = graph.shapes.clone();
            let shape_ids = graph.shape_ids.clone();
            let coefficients = graph.coefficients.clone();
            #[expect(
                clippy::mutable_key_type,
                reason = "AlgebraicReal caches preserve exact equality and hash"
            )]
            let coefficient_ids = graph.coefficient_ids.clone();
            let before = dag
                .nodes()
                .map(|(_, node)| node.clone())
                .collect::<Vec<_>>();
            let ty = snapshot.types[classes[x.0] as usize];
            let mut admission = Admission::new(
                &snapshot,
                &graph.shapes,
                &graph.coefficients,
                graph.maps,
                16,
            );
            let select = Form::Select(
                ty,
                Box::new(Form::Class(classes[p.0])),
                Box::new(Form::Class(classes[x.0])),
                Box::new(Form::Class(classes[x.0])),
            );
            let select = if valid {
                select
            } else {
                Form::binary(
                    "Add",
                    ty,
                    Form::Coefficient(
                        ty,
                        vec![AlgebraicReal::one(); branch.element_count().unwrap()],
                    ),
                    select,
                )
            };
            let accepted = admission.plan(Proposal {
                class: classes[x.0],
                form: select,
            });
            if valid {
                assert_eq!(accepted, Ok(true));
                let proofs = admission
                    .finish()
                    .commit(&mut graph, &snapshot, &classes, &mut dag)
                    .unwrap();
                let mut rebuilt = crate::optimize::rebuild::Rebuilder::new(module.entry_func());
                let value = crate::optimize::map_reduce::rewrite::extraction::expression_dag::emit(
                    &mut rebuilt,
                    &maps,
                    &dag,
                    proofs[0],
                    &mut HashMap::new(),
                )
                .unwrap();
                rebuilt
                    .alias(&module.entry_func().results, &[value])
                    .unwrap();
                let mut output = module.clone();
                output.funcs[entry.index()] = rebuilt.finish(module.entry_func()).unwrap();
                crate::verify::verify(&output).unwrap();
            } else {
                assert!(accepted.is_err());
                assert!(admission.is_empty());
                assert_eq!(admission.reserved_keys(), 0);
                assert!(
                    admission
                        .finish()
                        .commit(&mut graph, &snapshot, &classes, &mut dag)
                        .unwrap()
                        .is_empty()
                );
                assert!(graph.snapshot().unwrap() == snapshot);
                assert_eq!(graph.shapes, shapes);
                assert_eq!(graph.shape_ids, shape_ids);
                assert_eq!(graph.coefficients, coefficients);
                assert_eq!(graph.coefficient_ids, coefficient_ids);
                assert_eq!(
                    dag.nodes()
                        .map(|(_, node)| node.clone())
                        .collect::<Vec<_>>(),
                    before
                );
            }
        }
    }
}
