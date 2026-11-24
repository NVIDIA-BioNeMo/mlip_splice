// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use domain_index::{FiniteMapGraph, FiniteMapId, FiniteSetId};
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Def, Module};
use xir::ops::{Constant, Literal};
use xir::shape::Shape;
use xir_optimize::{
    DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK,
    extract_algebraic_frontier_with_limits,
};

fn refinement() -> (FiniteMapGraph, FiniteSetId, FiniteSetId, FiniteMapId) {
    let mut graph = FiniteMapGraph::default();
    let fine = graph.add_set(8).unwrap();
    let coarse = graph.add_set(3).unwrap();
    let map = graph.add_map(fine, coarse).unwrap();
    (graph, fine, coarse, map)
}

fn assert_zero_frontier(module: &Module) {
    let frontier = extract_algebraic_frontier_with_limits(
        module,
        DEFAULT_MAX_ALGEBRAIC_WORK,
        DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
    )
    .unwrap()
    .0;
    let zeros = frontier.iter().filter(|alternative| {
        let func = alternative.module.entry_func();
        func.results.iter().all(|&result| {
            xir::analysis::constant_literal(func, result)
                .and_then(Literal::algebraic_values)
                .is_some_and(|values| values.iter().all(AlgebraicReal::is_zero))
        })
    });
    let mut zero_count = 0;
    for alternative in zeros {
        zero_count += 1;
        let func = alternative.module.entry_func();
        for &result in &func.results {
            let Def::NodeResult { node, slot: 0 } = func.values[result.index()].def else {
                panic!("proof did not extract a constant");
            };
            let literal = &func.nodes[node]
                .op
                .as_any()
                .downcast_ref::<Constant>()
                .expect("proof did not extract a constant")
                .literal;
            assert!(
                literal
                    .algebraic_values()
                    .unwrap()
                    .iter()
                    .all(AlgebraicReal::is_zero)
            );
        }
        assert_eq!(alternative.work.before_accumulation, 0);
        assert_eq!(alternative.work.after_accumulation, 0);
        assert!(
            func.nodes
                .iter()
                .all(|node| node.op.as_any().is::<Constant>())
        );
    }
    assert!(zero_count > 0, "the common graph must prove exact zero");
}

#[test]
fn sum_linearity_composes_with_additive_cancellation() {
    let (graph, fine, _, map) = refinement();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("sum_linearity");
    let shape = Shape::scalar(DType::F32).with_domain_axes([fine]);
    let x = b.param(shape.clone());
    let y = b.param(shape);
    let sum = b.add(x, y);
    let sum = b.sum_along_map(sum, map, 0);
    let sx = b.sum_along_map(x, map, 0);
    let sy = b.sum_along_map(y, map, 0);
    let remainder = b.binary("subtract", sum, sx).unwrap();
    let zero = b.binary("subtract", remainder, sy).unwrap();
    let entry = b.end_func(vec![zero]);
    assert_zero_frontier(&b.finish(entry).unwrap());
}

#[test]
fn exact_projection_and_sum_linearity_share_one_proof() {
    let (graph, fine, coarse, map) = refinement();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("exact_projection");
    let fine_shape = Shape::scalar(DType::F32).with_domain_axes([fine]);
    let coarse_shape = Shape::scalar(DType::F32).with_domain_axes([coarse]);
    let a = b.param(coarse_shape.clone());
    let x = b.param(fine_shape.clone());
    let y = b.param(fine_shape.clone());
    let root = b.constant(Literal::exact(fine_shape, vec![AlgebraicReal::sqrt(2u32)]).unwrap());
    let square = b.multiply(root, root);
    let pulled = b.gather_along_map(a, map, 0);
    let xy = b.add(x, y);
    let product = b.multiply(pulled, xy);
    let product = b.multiply(square, product);
    let left = b.sum_along_map(product, map, 0);
    let sx = b.sum_along_map(x, map, 0);
    let sy = b.sum_along_map(y, map, 0);
    let sums = b.add(sx, sy);
    let weighted = b.multiply(a, sums);
    let two = b.constant(Literal::new(coarse_shape, 2.0f32.to_le_bytes().to_vec()));
    let right = b.multiply(two, weighted);
    let zero = b.binary("subtract", left, right).unwrap();
    let entry = b.end_func(vec![zero, zero]);
    assert_zero_frontier(&b.finish(entry).unwrap());
}

#[test]
fn gather_congruence_exposes_exact_polynomial_identity() {
    let (graph, _, coarse, map) = refinement();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("gather_polynomial");
    let shape = Shape::scalar(DType::F32).with_domain_axes([coarse]);
    let x = b.param(shape.clone());
    let y = b.param(shape.clone());
    let two = b.constant(Literal::new(shape, 2.0f32.to_le_bytes().to_vec()));
    let gx = b.gather_along_map(x, map, 0);
    let gy = b.gather_along_map(y, map, 0);
    let sum = b.add(gx, gy);
    let left = b.multiply(sum, sum);
    let xx = b.multiply(x, x);
    let yy = b.multiply(y, y);
    let xy = b.multiply(x, y);
    let twice_xy = b.multiply(two, xy);
    let right = b.add(xx, yy);
    let right = b.add(right, twice_xy);
    let right = b.gather_along_map(right, map, 0);
    let zero = b.binary("subtract", left, right).unwrap();
    let entry = b.end_func(vec![zero]);
    assert_zero_frontier(&b.finish(entry).unwrap());
}

#[test]
fn nested_sums_keep_exact_coefficients_and_axis_identity() {
    let (graph, fine, coarse, map) = refinement();
    let mut b = ModuleBuilder::with_finite_map_graph(graph);
    b.begin_func("nested_sum_linearity");
    let shape = Shape::new(DType::F64, [2]).with_domain_axes([fine, fine]);
    let x = b.param(shape.clone());
    let y = b.param(shape.clone());
    let half = b.constant(Literal::new(
        shape,
        [0.5f64, 0.25]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect(),
    ));
    let xy = b.add(x, y);
    let mut left = b.multiply(half, xy);
    let mut sx = x;
    let mut sy = y;
    for axis in [1, 0] {
        left = b.sum_along_map(left, map, axis);
        sx = b.sum_along_map(sx, map, axis);
        sy = b.sum_along_map(sy, map, axis);
    }
    let sums = b.add(sx, sy);
    let out_shape = Shape::new(DType::F64, [2]).with_domain_axes([coarse, coarse]);
    let half = b.constant(Literal::new(
        out_shape,
        [0.5f64, 0.25]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect(),
    ));
    let right = b.multiply(half, sums);
    let zero = b.binary("subtract", left, right).unwrap();
    let entry = b.end_func(vec![zero]);
    assert_zero_frontier(&b.finish(entry).unwrap());
}

#[test]
fn the_typed_space_also_proves_scalar_arithmetic_without_maps() {
    let mut b = ModuleBuilder::new();
    b.begin_func("scalar_commutative_cancellation");
    let x = b.param(Shape::scalar(DType::F32));
    let y = b.param(Shape::scalar(DType::F32));
    let xy = b.multiply(x, y);
    let yx = b.multiply(y, x);
    let zero = b.binary("subtract", xy, yx).unwrap();
    let entry = b.end_func(vec![zero]);
    let mut module = b.finish(entry).unwrap();
    xir_optimize::PassManager::new([
        xir_optimize::Pass::IndexedAlgebraCanonicalization,
        xir_optimize::Pass::CommonSubexpressionElimination,
        xir_optimize::Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();
    let func = module.entry_func();
    let literal =
        xir::analysis::constant_literal(func, func.results[0]).expect("exact scalar zero");
    assert_eq!(
        literal.algebraic_values(),
        Some(vec![AlgebraicReal::zero()])
    );
}
