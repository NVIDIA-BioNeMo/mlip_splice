// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::FiniteMapGraph;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::Module;
use xir::shape::Shape;
use xir_optimize::{Pass, PassManager, extract_algebraic_frontier_with_limits};

fn product_of_sums(extent: i64, factors: usize) -> Module {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(extent).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("product_of_symbolic_sums");
    let mut product = None;
    for _ in 0..factors {
        let x = b.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
        let y = b.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
        let sum = b.add(x, y);
        let sum = b.sum_along_map(sum, map, 0);
        product = Some(match product {
            None => sum,
            Some(previous) => b.multiply(previous, sum),
        });
    }
    let entry = b.end_func(vec![product.unwrap()]);
    b.finish(entry).unwrap()
}

#[test]
fn zero_rewrite_budget_keeps_an_executable_original() {
    let original = product_of_sums(8, 3);
    for (total, expression) in [(0, 1024), (1024, 0), (0, 0)] {
        let mut module = original.clone();
        let before = module.to_string();
        let report = PassManager::new([Pass::IndexedAlgebraCanonicalization])
            .with_max_algebraic_work(total)
            .with_max_algebraic_expression_work(expression)
            .run(&mut module)
            .unwrap();
        assert_eq!(report.algebraic_work, 0);
        assert_eq!(module.to_string(), before);
        assert!(!report.passes[0].changed);
    }
}

#[test]
fn products_of_symbolic_sums_obey_the_work_budget() {
    let original = product_of_sums(8, 20);
    for budget in [1, 32, 512, 4096] {
        let (frontier, report) =
            extract_algebraic_frontier_with_limits(&original, budget, budget).unwrap();
        assert!(!frontier.is_empty());
        assert!(frontier.len() <= 16);
        assert!(report.algebraic_work <= budget);
        for alternative in frontier {
            xir::verify::verify(&alternative.module).unwrap();
            let func = alternative.module.entry_func();
            let expected = original.entry_func();
            assert_eq!(
                func.results
                    .iter()
                    .map(|&value| func.value_shape(value))
                    .collect::<Vec<_>>(),
                expected
                    .results
                    .iter()
                    .map(|&value| expected.value_shape(value))
                    .collect::<Vec<_>>()
            );
            assert!(alternative.module.entry_func().nodes.len() <= 1024);
        }
    }
}

#[test]
fn domain_cardinality_does_not_materialize_symbolic_sums() {
    let mut small = product_of_sums(8, 2);
    let mut large = product_of_sums(1_000_000_000, 2);
    let optimizer = PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .with_max_algebraic_work(1 << 16)
    .with_max_algebraic_expression_work(1 << 14);
    let small_report = optimizer.run(&mut small).unwrap();
    let large_report = optimizer.run(&mut large).unwrap();
    assert!(small_report.algebraic_work <= 1 << 16);
    assert!(large_report.algebraic_work <= 1 << 16);
    assert!(small.entry_func().nodes.len() <= 32);
    assert!(large.entry_func().nodes.len() <= 32);
    assert!(
        large
            .entry_func()
            .nodes
            .iter()
            .any(|node| node.op.mnemonic() == "sum_along_map")
    );
}

#[test]
fn deeply_shared_arithmetic_keeps_a_bounded_executable_dag() {
    let mut b = ModuleBuilder::new();
    b.begin_func("shared_squares");
    let mut value = b.param(Shape::scalar(DType::F32));
    for _ in 0..24 {
        value = b.multiply(value, value);
    }
    let entry = b.end_func(vec![value, value]);
    let mut module = b.finish(entry).unwrap();
    let report = PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .with_max_algebraic_work(256)
    .with_max_algebraic_expression_work(64)
    .run(&mut module)
    .unwrap();
    assert!(report.algebraic_work <= 256);
    xir::verify::verify(&module).unwrap();
    let func = module.entry_func();
    assert_eq!(func.results[0], func.results[1]);
    assert!(func.nodes.len() <= 64);
}

#[test]
fn frontier_keeps_the_original_beside_a_cheaper_symbolic_placement() {
    let original = product_of_sums(8, 1);
    let (frontier, _) =
        extract_algebraic_frontier_with_limits(&original, 1 << 16, 1 << 14).unwrap();
    assert!(frontier.iter().any(|alternative| {
        alternative.module.to_string() == original.to_string()
            && alternative.work.before_accumulation == 8
            && alternative.work.after_accumulation == 0
    }));
    assert!(frontier.iter().any(|alternative| {
        alternative.work.before_accumulation == 0 && alternative.work.after_accumulation == 3
    }));
}
