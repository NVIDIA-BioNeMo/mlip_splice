// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;

use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Def, Func};
use xir::ops::{Constant, Literal};
use xir::shape::Shape;
use xir_optimize::{Pass, PassManager};

#[test]
fn mixed_exact_and_ordinary_coefficients_prove_surd_cancellation() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let root =
            func.constant(Literal::exact(shape.clone(), vec![AlgebraicReal::sqrt(2u32)]).unwrap());
        let two = func.constant(f32_scalar(2.0));
        let root_squared = func.multiply(root, root);
        let left = func.multiply(root_squared, x);
        let right = func.multiply(two, x);
        vec![func.binary("subtract", left, right).unwrap()]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let zero = result_literal(module.entry_func());
    assert_eq!(zero.algebraic_values(), Some(vec![AlgebraicReal::zero()]));
    assert_eq!(f32_values(zero), vec![0.0]);
}

#[test]
fn ordinary_float_coefficients_seed_polynomial_islands() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape);
        let two = func.constant(f32_scalar(2.0));
        let twice = func.multiply(two, x);
        let once = func.binary("subtract", twice, x).unwrap();
        vec![func.binary("subtract", once, x).unwrap()]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let zero = result_literal(module.entry_func());
    assert_eq!(zero.algebraic_values(), Some(vec![AlgebraicReal::zero()]));
    assert_eq!(module.entry_func().nodes.len(), 1);
}

#[test]
fn point_one_coefficient_is_not_treated_as_decimal_one_tenth() {
    let tenth = AlgebraicReal::one()
        .checked_div(&AlgebraicReal::integer(10))
        .unwrap();
    let dyadic_point_one = AlgebraicReal::integer(13_421_773)
        .checked_div(&AlgebraicReal::integer(134_217_728))
        .unwrap();
    let expected = &dyadic_point_one - &tenth;
    assert!(!expected.is_zero());

    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let ordinary = func.constant(f32_scalar(0.1));
        let exact = func.constant(Literal::exact(shape, vec![tenth]).unwrap());
        vec![func.binary("subtract", ordinary, exact).unwrap()]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();

    assert_eq!(
        result_literal(module.entry_func()).algebraic_values(),
        Some(vec![expected])
    );
}

#[test]
fn algebraic_folding_of_ordinary_floats_is_local_to_finite_inputs() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let one_and_half = func.constant(f32_scalar(1.5));
        let two_and_quarter = func.constant(f32_scalar(2.25));
        let nan = func.constant(f32_scalar(f32::NAN));
        let finite = func.add(one_and_half, two_and_quarter);
        let nonfinite = func.add(one_and_half, nan);
        vec![finite, nonfinite]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();

    let func = module.entry_func();
    assert_eq!(
        literal_for_result(func, 0).algebraic_values(),
        Some(vec![
            AlgebraicReal::integer(15)
                .checked_div(&AlgebraicReal::integer(4))
                .unwrap()
        ])
    );
    assert_eq!(producing_mnemonic(func, 1), "add");
}

#[test]
fn strict_constant_folding_leaves_ordinary_float_arithmetic_unchanged() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let left = func.constant(f32_scalar(0.1));
        let right = func.constant(f32_scalar(0.2));
        vec![func.add(left, right)]
    });
    let mut module = builder.finish(main).unwrap();
    let before = module.entry_func().nodes.len();

    let report = PassManager::new([Pass::ConstantFolding])
        .run(&mut module)
        .unwrap();

    let func = module.entry_func();
    assert!(!report.passes[0].changed);
    assert_eq!(func.nodes.len(), before);
    assert_eq!(producing_mnemonic(func, 0), "add");
    assert!(
        func.nodes
            .iter()
            .filter_map(|node| node.op.as_any().downcast_ref::<Constant>())
            .all(|constant| constant.literal.exact.is_none())
    );
}

#[test]
fn ordinary_coefficient_islands_obey_the_expansion_budget() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let x = func.param(shape.clone());
        let y = func.param(shape);
        let two = func.constant(f32_scalar(2.0));
        let sum = func.add(x, y);
        vec![func.multiply(two, sum)]
    });
    let mut module = builder.finish(main).unwrap();
    let before = module.entry_func().nodes.len();

    let report = PassManager::new([Pass::IndexedAlgebraCanonicalization])
        .with_max_polynomial_terms(1)
        .run(&mut module)
        .unwrap();

    assert!(!report.passes[0].changed);
    assert_eq!(module.entry_func().nodes.len(), before);
    assert_eq!(producing_mnemonic(module.entry_func(), 0), "multiply");
}

fn f32_scalar(value: f32) -> Literal {
    Literal::new(Shape::scalar(DType::F32), value.to_le_bytes().to_vec())
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
        .downcast_ref::<Constant>()
        .expect("result is not constant")
        .literal
}

fn producing_mnemonic(func: &Func, result: usize) -> &'static str {
    let Def::NodeResult { node, slot: 0 } = func.values[func.results[result].index()].def else {
        panic!("result is not a node result");
    };
    func.nodes[node].op.mnemonic()
}
