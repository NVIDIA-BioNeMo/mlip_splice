// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use smallvec::smallvec;

use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Def, Func};
use xir::ops::{BroadcastInDim, ComparisonDirection, Constant, Literal, Select};
use xir::shape::Shape;
use xir_optimize::{Pass, PassManager};

#[test]
fn algebraic_pass_folds_exact_surd_constants() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let root =
            func.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::sqrt(2u32)]).unwrap());
        let negative =
            func.constant(Literal::exact(shape, vec![-AlgebraicReal::sqrt(2u32)]).unwrap());
        vec![func.add(root, negative), func.multiply(root, root)]
    });
    let mut strict = builder.finish(main).unwrap();
    let mut algebraic = strict.clone();

    PassManager::new([Pass::ConstantFolding])
        .run(&mut strict)
        .unwrap();
    assert_eq!(strict.entry_func().nodes.len(), 4);

    PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut algebraic)
        .unwrap();
    let zero = literal_for_result(algebraic.entry_func(), 0);
    let two = literal_for_result(algebraic.entry_func(), 1);
    assert!(zero.exact.as_ref().unwrap()[0].is_zero());
    assert_eq!(two.exact.as_ref().unwrap()[0], AlgebraicReal::integer(2));
    assert_eq!(f32_values(zero), vec![0.0]);
    assert_eq!(f32_values(two), vec![2.0]);
}

#[test]
fn algebraic_work_is_charged_before_folding() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let root =
            func.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::sqrt(2u32)]).unwrap());
        let one = func.constant(Literal::exact(shape, vec![AlgebraicReal::one()]).unwrap());
        vec![func.add(root, one)]
    });
    let mut module = builder.finish(main).unwrap();
    let report = PassManager::new([Pass::AlgebraicCanonicalization])
        .with_max_algebraic_work(1)
        .with_max_algebraic_expression_work(1)
        .run(&mut module)
        .unwrap();
    assert_eq!(report.algebraic_work, 0);
    assert_eq!(report.abandoned_algebraic_expressions, 1);
    assert_eq!(
        module.entry_func().nodes.last().unwrap().op.mnemonic(),
        "add"
    );
}

#[test]
fn shape_folding_preserves_exact_values() {
    let root = AlgebraicReal::sqrt(2u32);
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let scalar =
            func.constant(Literal::exact(Shape::scalar(DType::F32), vec![root.clone()]).unwrap());
        vec![func.op1(
            BroadcastInDim {
                out_dims: smallvec![3],
                broadcast_dimensions: smallvec![],
            },
            &[scalar],
        )]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([Pass::ConstantFolding])
        .run(&mut module)
        .unwrap();

    assert_eq!(
        result_literal(module.entry_func()).exact.as_ref().unwrap(),
        &vec![root; 3]
    );
}

#[test]
fn cse_includes_exact_provenance() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let a = func.constant(
            Literal::exact(shape.clone(), vec![AlgebraicReal::integer(16_777_216)]).unwrap(),
        );
        let b =
            func.constant(Literal::exact(shape, vec![AlgebraicReal::integer(16_777_217)]).unwrap());
        vec![a, b]
    });
    let mut module = builder.finish(main).unwrap();
    assert_eq!(
        f32_values(literal_for_result(module.entry_func(), 0)),
        f32_values(literal_for_result(module.entry_func(), 1))
    );

    PassManager::new([Pass::CommonSubexpressionElimination])
        .run(&mut module)
        .unwrap();

    assert_ne!(
        module.entry_func().results[0],
        module.entry_func().results[1]
    );
}

#[test]
fn indexed_algebra_pass_merges_commutative_monomials() {
    let root = AlgebraicReal::sqrt(2u32);
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let y = func.param(shape.clone());
        let a = func.constant(Literal::exact(shape.clone(), vec![root.clone()]).unwrap());
        let b = func.constant(Literal::exact(shape, vec![root.clone()]).unwrap());
        let ax = func.multiply(a, x);
        let by = func.multiply(b, y);
        let xy = func.multiply(ax, y);
        let yx = func.multiply(by, x);
        vec![func.add(xy, yx)]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 3);
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "add")
            .count(),
        0
    );
    let exact = func
        .nodes
        .iter()
        .find_map(|node| {
            node.op
                .as_any()
                .downcast_ref::<Constant>()
                .and_then(|constant| constant.literal.exact.as_ref())
        })
        .unwrap();
    assert_eq!(exact, &vec![&root * &AlgebraicReal::integer(2)]);
}

#[test]
fn indexed_algebra_pass_merges_selects_with_a_shared_comparison() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let a = func.param(shape.clone());
        let b = func.param(shape.clone());
        let c = func.param(shape.clone());
        let d = func.param(shape);
        let predicate = func.compare(ComparisonDirection::Lt, None, a, b);
        let left = func.op1(Select, &[predicate, a, b]);
        let right = func.op1(Select, &[predicate, c, d]);
        vec![func.add(left, right)]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let nodes = &module.entry_func().nodes;
    assert_eq!(
        nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "select")
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "add")
            .count(),
        2
    );
}

#[test]
fn indexed_algebra_pass_keeps_a_scalar_comparison_for_tensor_selects() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let scalar = Shape::scalar(DType::F32);
        let tensor = Shape::new(DType::F32, [2]);
        let x = func.param(scalar.clone());
        let y = func.param(scalar);
        let a = func.param(tensor.clone());
        let b = func.param(tensor.clone());
        let c = func.param(tensor.clone());
        let d = func.param(tensor);
        let predicate = func.compare(ComparisonDirection::Le, None, x, y);
        let left = func.op1(Select, &[predicate, a, b]);
        let right = func.op1(Select, &[predicate, c, d]);
        vec![func.add(left, right)]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "select")
            .count(),
        1
    );
    let compare = func
        .nodes
        .iter()
        .find_map(|node| node.op.as_any().downcast_ref::<xir::ops::Compare>())
        .unwrap();
    assert_eq!(compare.comparison_direction, ComparisonDirection::Le);
}

#[test]
fn indexed_algebra_pass_reuses_the_guard_factor_for_gated_islands() {
    // w_gated·(1−8·r·inv)²: the gate envelope divides the island exactly, so
    // the render must reuse the predicate's materialized envelope squared —
    // never the expanded monomials, which cancel catastrophically at the
    // guard boundary.
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let r = func.param(shape.clone());
        let inv = func.param(shape.clone());
        let w = func.param(shape.clone());
        let one = func.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::one()]).unwrap());
        let eight =
            func.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::integer(8)]).unwrap());
        let zero = func.constant(Literal::zeros(shape).unwrap());
        let scaled = func.multiply(eight, r);
        let v = func.multiply(scaled, inv);
        let envelope = func.binary("subtract", one, v).unwrap();
        let predicate = func.compare(ComparisonDirection::Ge, None, envelope, zero);
        let gated = func.op1(Select, &[predicate, w, zero]);
        let envelope2 = func.multiply(envelope, envelope);
        vec![func.multiply(gated, envelope2)]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    let expanded = [AlgebraicReal::integer(64), -&AlgebraicReal::integer(16)];
    for node in &func.nodes {
        if let Some(constant) = node.op.as_any().downcast_ref::<Constant>()
            && let Some(exact) = constant.literal.exact.as_ref()
        {
            assert!(
                !exact.iter().any(|value| expanded.contains(value)),
                "expanded envelope coefficient materialized: {node:?}"
            );
        }
    }
    let compare = func
        .nodes
        .iter()
        .find(|node| node.op.mnemonic() == "compare")
        .unwrap();
    let envelope = compare.inputs[0];
    let select = func
        .nodes
        .iter()
        .find(|node| node.op.mnemonic() == "select")
        .unwrap();
    fn product_leaves(func: &Func, value: xir::ir::ValueId, leaves: &mut Vec<xir::ir::ValueId>) {
        if let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def
            && func.nodes[node].op.mnemonic() == "multiply"
        {
            for &input in &func.nodes[node].inputs {
                product_leaves(func, input, leaves);
            }
            return;
        }
        leaves.push(value);
    }
    let mut leaves = Vec::new();
    product_leaves(func, func.results[0], &mut leaves);
    leaves.sort_unstable_by_key(|value| value.index());
    let mut expected = vec![envelope, envelope, select.results[0]];
    expected.sort_unstable_by_key(|value| value.index());
    assert_eq!(leaves, expected, "result is not envelope²·gated");
}

#[test]
fn indexed_algebra_pass_factors_work_shared_by_select_branches() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let predicate = func.param(Shape::scalar(DType::Bool));
        let shape = Shape::scalar(DType::F32);
        let a = func.param(shape.clone());
        let b = func.param(shape.clone());
        let shared = func.param(shape);
        let on_true = func.add(a, shared);
        let on_false = func.add(b, shared);
        vec![func.op1(Select, &[predicate, on_true, on_false])]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let nodes = &module.entry_func().nodes;
    assert_eq!(
        nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "select")
            .count(),
        1
    );
    assert_eq!(
        nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "add")
            .count(),
        1
    );
}

#[test]
fn indexed_algebra_pass_drops_exact_cancellation() {
    let root = AlgebraicReal::sqrt(2u32);
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let y = func.param(shape.clone());
        let positive = func.constant(Literal::exact(shape.clone(), vec![root.clone()]).unwrap());
        let negative = func.constant(Literal::exact(shape, vec![-root]).unwrap());
        let positive_x = func.multiply(positive, x);
        let negative_y = func.multiply(negative, y);
        let xy = func.multiply(positive_x, y);
        let yx = func.multiply(negative_y, x);
        vec![func.add(xy, yx)]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let literal = result_literal(module.entry_func());
    assert!(literal.exact.as_ref().unwrap()[0].is_zero());
    assert_eq!(f32_values(literal), vec![0.0]);
}

#[test]
fn indexed_algebra_pass_obeys_its_expansion_budget() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let y = func.param(shape.clone());
        let z = func.param(shape.clone());
        let coefficient =
            func.constant(Literal::exact(shape, vec![AlgebraicReal::sqrt(2u32)]).unwrap());
        let sum = func.add(x, y);
        let scaled = func.multiply(coefficient, sum);
        vec![func.multiply(scaled, z)]
    });
    let mut module = builder.finish(main).unwrap();
    let before = module.entry_func().nodes.len();

    let report = PassManager::new([Pass::IndexedAlgebraCanonicalization])
        .with_max_polynomial_terms(1)
        .run(&mut module)
        .unwrap();

    assert!(!report.passes[0].changed);
    assert_eq!(module.entry_func().nodes.len(), before);
}

#[test]
fn indexed_algebra_pass_bounds_deep_sums_without_recursion() {
    const TERMS: i64 = 20_000;
    let root = AlgebraicReal::sqrt(2u32);
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let coefficient = func.constant(Literal::exact(shape, vec![root.clone()]).unwrap());
        let term = func.multiply(coefficient, x);
        let mut sum = term;
        for _ in 1..TERMS {
            sum = func.add(sum, term);
        }
        vec![sum]
    });
    let mut module = builder.finish(main).unwrap();

    let report = PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::DeadCodeElimination,
    ])
    .with_max_polynomial_terms(100_000)
    .run(&mut module)
    .unwrap();

    assert!(report.passes[0].changed);
    assert!(module.entry_func().nodes.len() < TERMS as usize);
    xir::verify::verify(&module).unwrap();
}

fn f32_values(literal: &Literal) -> Vec<f32> {
    literal
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn result_literal(func: &Func) -> &Literal {
    literal_for_result(func, 0)
}

fn literal_for_result(func: &Func, result: usize) -> &Literal {
    let Def::NodeResult { node, slot: 0 } = func.values[func.results[result].index()].def else {
        panic!("result is not a node result");
    };
    &func.nodes[node]
        .op
        .as_any()
        .downcast_ref::<xir::ops::Constant>()
        .expect("result is not constant")
        .literal
}
