// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use domain_index::FiniteMapGraph;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ops::{Add, Constant, GatherAlongMap, Literal, SumAlongMap};
use xir::shape::Shape;
use xir_optimize::{
    DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK, Pass, PassManager,
    algebraic_pipeline, extract_algebraic_frontier_with_limits,
};

fn refinement() -> (
    FiniteMapGraph,
    domain_index::FiniteSetId,
    domain_index::FiniteSetId,
    domain_index::FiniteMapId,
) {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    (graph, fine, coarse, map)
}

#[test]
fn gather_and_sum_change_only_the_named_domain_axis() {
    let (graph, fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("projection");
    let coefficient = builder.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let values = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let gathered = builder.gather_along_map(coefficient, map, 0);
    let product = builder.multiply(gathered, values);
    let sum = builder.sum_along_map(product, map, 0);
    let entry = builder.end_func(vec![sum]);
    let module = builder.finish(entry).unwrap();
    let func = module.entry_func();

    assert!(func.value_shape(gathered).dims.is_empty());
    assert_eq!(func.value_shape(gathered).domain_axes.as_slice(), &[fine]);
    assert_eq!(func.value_shape(sum).domain_axes.as_slice(), &[coarse]);
    assert_eq!(func.value_shape(sum).num_elements(), Some(1));
}

#[test]
fn map_direction_and_axis_are_verified() {
    let (graph, fine, _, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("invalid");
    let value = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    assert!(
        builder
            .try_op1(GatherAlongMap { map, axis: 0 }, &[value])
            .is_err()
    );
    assert!(
        builder
            .try_op1(SumAlongMap { map, axis: 1 }, &[value])
            .is_err()
    );
}

#[test]
fn projection_formula_moves_a_gathered_factor_out_of_a_sum() {
    let (graph, fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("projection");
    let coefficient = builder.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let values = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let gathered = builder.gather_along_map(coefficient, map, 0);
    let product = builder.multiply(values, gathered);
    let sum = builder.sum_along_map(product, map, 0);
    let entry = builder.end_func(vec![sum]);
    let mut module = builder.finish(entry).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let mnemonics = module
        .entry_func()
        .nodes
        .iter()
        .map(|node| node.op.mnemonic())
        .collect::<Vec<_>>();
    assert_eq!(mnemonics, ["sum_along_map", "multiply"]);
}

#[test]
fn multi_output_frontier_shares_work_and_crosses_at_partial_count() {
    let (graph, fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("shared_projection_frontier");
    let coefficient = builder.param(Shape::scalar(DType::F32).with_domain_axes([coarse]));
    let values = builder.param(Shape::scalar(DType::F32).with_domain_axes([fine]));
    let gathered = builder.gather_along_map(coefficient, map, 0);
    let product = builder.multiply(gathered, values);
    let result = builder.sum_along_map(product, map, 0);
    let entry = builder.end_func(vec![result, result]);
    let module = builder.finish(entry).unwrap();

    let frontier = extract_algebraic_frontier_with_limits(
        &module,
        DEFAULT_MAX_ALGEBRAIC_WORK,
        DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
    )
    .unwrap()
    .0;
    let mut work = frontier
        .iter()
        .map(|alternative| alternative.work)
        .collect::<Vec<_>>();
    work.sort_by_key(|work| (work.before_accumulation, work.after_accumulation));

    assert_eq!(work.len(), 2);
    assert_eq!(work[0].before_accumulation, 0);
    assert_eq!(work[0].after_accumulation, 3);
    assert_eq!(work[1].before_accumulation, 8);
    assert_eq!(work[1].after_accumulation, 0);
    assert!(work[0].instantiate(1) < work[1].instantiate(1));
    assert!(work[1].instantiate(3) < work[0].instantiate(3));
}

#[test]
fn projection_formula_saturates_across_every_domain_axis() {
    let (graph, fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("multi_axis_projection");
    let coefficient =
        builder.param(Shape::scalar(DType::F32).with_domain_axes(std::iter::repeat_n(coarse, 4)));
    let values =
        builder.param(Shape::scalar(DType::F32).with_domain_axes(std::iter::repeat_n(fine, 4)));
    let mut gathered = coefficient;
    for axis in (0..4).rev() {
        gathered = builder.gather_along_map(gathered, map, axis);
    }
    let mut sum = builder.multiply(gathered, values);
    for axis in 0..4 {
        sum = builder.sum_along_map(sum, map, axis);
    }
    let entry = builder.end_func(vec![sum]);
    let mut module = builder.finish(entry).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let mnemonics = module
        .entry_func()
        .nodes
        .iter()
        .map(|node| node.op.mnemonic())
        .collect::<Vec<_>>();
    assert_eq!(
        mnemonics,
        [
            "sum_along_map",
            "sum_along_map",
            "sum_along_map",
            "sum_along_map",
            "multiply"
        ]
    );
}

#[test]
fn arithmetic_moves_to_the_smaller_side_of_a_finite_map() {
    let (graph, _fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("smaller_scope");
    let shape = Shape::scalar(DType::F32).with_domain_axes([coarse]);
    let left = builder.param(shape.clone());
    let right = builder.param(shape);
    let left = builder.gather_along_map(left, map, 0);
    let right = builder.gather_along_map(right, map, 0);
    let sum = builder.add(left, right);
    let result = builder.sum_along_map(sum, map, 0);
    let entry = builder.end_func(vec![result]);
    let mut module = builder.finish(entry).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    let add = func
        .nodes
        .iter()
        .find(|node| node.op.as_any().is::<Add>())
        .unwrap();
    assert_eq!(
        func.value_shape(add.results[0]).domain_axes.as_slice(),
        &[coarse]
    );
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.as_any().is::<GatherAlongMap>())
            .count(),
        1
    );
    assert_eq!(
        func.value_shape(func.results[0]).domain_axes.as_slice(),
        &[coarse]
    );
}

#[test]
fn gathered_arithmetic_moves_without_an_output_sum() {
    let (graph, _fine, coarse, map) = refinement();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("wide_output");
    let shape = Shape::scalar(DType::F32).with_domain_axes([coarse]);
    let left = builder.param(shape.clone());
    let right = builder.param(shape);
    let left = builder.gather_along_map(left, map, 0);
    let right = builder.gather_along_map(right, map, 0);
    let result = builder.add(left, right);
    let entry = builder.end_func(vec![result]);
    let mut module = builder.finish(entry).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    let add = func
        .nodes
        .iter()
        .find(|node| node.op.as_any().is::<Add>())
        .unwrap();
    assert_eq!(
        func.value_shape(add.results[0]).domain_axes.as_slice(),
        &[coarse]
    );
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.as_any().is::<GatherAlongMap>())
            .count(),
        1
    );
}

#[test]
fn arithmetic_stays_on_the_smaller_source_of_a_finite_map() {
    let mut graph = FiniteMapGraph::default();
    let source = graph.add_set(3).unwrap();
    let target = graph.add_set(8).unwrap();
    let map = graph.add_map(source, target).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("smaller_source");
    let shape = Shape::scalar(DType::F32).with_domain_axes([target]);
    let left = builder.param(shape.clone());
    let right = builder.param(shape);
    let left = builder.gather_along_map(left, map, 0);
    let right = builder.gather_along_map(right, map, 0);
    let sum = builder.add(left, right);
    let result = builder.sum_along_map(sum, map, 0);
    let entry = builder.end_func(vec![result]);
    let mut module = builder.finish(entry).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    let add = func
        .nodes
        .iter()
        .find(|node| node.op.as_any().is::<Add>())
        .unwrap();
    assert_eq!(
        func.value_shape(add.results[0]).domain_axes.as_slice(),
        &[source]
    );
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.as_any().is::<GatherAlongMap>())
            .count(),
        2
    );
    assert_eq!(
        func.value_shape(func.results[0]).domain_axes.as_slice(),
        &[target]
    );
}

#[test]
fn exact_coefficients_move_out_of_nested_finite_sums() {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let unit = graph.add_set(1).unwrap();
    let refinement = graph.add_map(fine, coarse).unwrap();
    graph.add_map(fine, unit).unwrap();
    graph.add_map(coarse, unit).unwrap();

    let shape = Shape::scalar(DType::F32).with_domain_axes([fine, fine]);
    let root = AlgebraicReal::sqrt(2u32);
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("exact_projection");
    let x = builder.param(shape.clone());
    let y = builder.param(shape.clone());
    let a = builder.constant(Literal::exact(shape.clone(), vec![root.clone()]).unwrap());
    let b =
        builder.constant(Literal::exact(shape, vec![&root * &AlgebraicReal::integer(3)]).unwrap());
    let left = builder.multiply(a, x);
    let right = builder.multiply(b, y);
    let mut value = builder.add(left, right);
    value = builder.sum_along_map(value, refinement, 0);
    value = builder.sum_along_map(value, refinement, 1);
    let entry = builder.end_func(vec![value]);
    let mut module = builder.finish(entry).unwrap();

    algebraic_pipeline().run(&mut module).unwrap();

    let constants = module
        .entry_func()
        .nodes
        .iter()
        .filter_map(|node| node.op.as_any().downcast_ref::<Constant>())
        .filter(|constant| {
            constant
                .literal
                .exact
                .as_ref()
                .is_some_and(|values| values.iter().any(|value| value.as_fraction().is_none()))
        })
        .collect::<Vec<_>>();
    assert!(!constants.is_empty());
    for constant in constants {
        assert_eq!(
            constant.literal.shape.domain_axes.as_slice(),
            &[coarse, coarse]
        );
    }
}
