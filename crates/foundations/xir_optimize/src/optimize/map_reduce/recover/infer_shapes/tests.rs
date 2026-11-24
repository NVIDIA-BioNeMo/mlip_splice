// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::ir::Module;
use crate::ops::{Add, Multiply, Tanh};
use crate::shape::Shape;

use super::*;

/// Builds a function over four scalar parameters and hands back the module plus
/// the values the test asked for.
fn scalars(
    build: impl FnOnce(&mut ModuleBuilder, &[ValueId]) -> Vec<ValueId>,
) -> (Module, Vec<ValueId>) {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let scalar = Shape::scalar(DType::F32);
    let params = (0..4)
        .map(|_| builder.param(scalar.clone()))
        .collect::<Vec<_>>();
    let wanted = build(&mut builder, &params);
    let main = builder.end_func(wanted.clone());
    (builder.finish(main).unwrap(), wanted)
}

fn assign_params_as_leaves(func: &Func) -> (Shapes, Vec<Option<ShapeId>>) {
    let params = func.params.clone();
    assign(func, 1 << 12, move |value| params.contains(&value))
}

/// The point of abstracting leaves: `a*b` and `c*d` are the same program on
/// different inputs, so they must share a shape and therefore a batch.
#[test]
fn products_of_different_operands_share_a_shape() {
    let (module, wanted) = scalars(|builder, p| {
        let left = builder.op(Multiply, &[p[0], p[1]])[0];
        let right = builder.op(Multiply, &[p[2], p[3]])[0];
        vec![left, right]
    });
    let (_, assigned) = assign_params_as_leaves(module.entry_func());
    assert_eq!(assigned[wanted[0].index()], assigned[wanted[1].index()]);
    assert!(assigned[wanted[0].index()].is_some_and(|shape| shape != LEAF));
}

/// Different operations never share a batch, however similar their operands:
/// one batch emits one operation sequence.
#[test]
fn different_operations_do_not_share_a_shape() {
    let (module, wanted) = scalars(|builder, p| {
        let product = builder.op(Multiply, &[p[0], p[1]])[0];
        let sum = builder.op(Add, &[p[0], p[1]])[0];
        vec![product, sum]
    });
    let (_, assigned) = assign_params_as_leaves(module.entry_func());
    assert_ne!(assigned[wanted[0].index()], assigned[wanted[1].index()]);
}

/// Nothing here is arithmetic-specific: a transcendental composes into a shape
/// exactly as a multiply does, so batching is not tied to polynomials.
#[test]
fn transcendental_expressions_share_a_shape() {
    let (module, wanted) = scalars(|builder, p| {
        let left = builder.op(Tanh, &[p[0]])[0];
        let left = builder.op(Multiply, &[left, p[1]])[0];
        let right = builder.op(Tanh, &[p[2]])[0];
        let right = builder.op(Multiply, &[right, p[3]])[0];
        vec![left, right]
    });
    let func = module.entry_func();
    let (shapes, assigned) = assign_params_as_leaves(func);
    assert_eq!(assigned[wanted[0].index()], assigned[wanted[1].index()]);
    let shape = assigned[wanted[0].index()].unwrap();
    assert_eq!(shapes.size(shape), 2, "the shape is tanh then multiply");
    assert_eq!(shapes.leaves(shape), 2);
}

/// Depth matters: `a*(b*c)` is not `(a*b)*c`, so they take different batches.
#[test]
fn association_changes_the_shape() {
    let (module, wanted) = scalars(|builder, p| {
        let inner = builder.op(Multiply, &[p[1], p[2]])[0];
        let right_leaning = builder.op(Multiply, &[p[0], inner])[0];
        let inner = builder.op(Multiply, &[p[0], p[1]])[0];
        let left_leaning = builder.op(Multiply, &[inner, p[2]])[0];
        vec![right_leaning, left_leaning]
    });
    let (_, assigned) = assign_params_as_leaves(module.entry_func());
    assert_ne!(assigned[wanted[0].index()], assigned[wanted[1].index()]);
}

/// Leaves are read in the unfolding's order, and a value used twice occupies
/// both positions — the gather simply reads that pool entry twice.
#[test]
fn leaf_order_repeats_a_shared_operand() {
    let (module, wanted) = scalars(|builder, p| vec![builder.op(Multiply, &[p[0], p[0]])[0]]);
    let func = module.entry_func();
    let (shapes, assigned) = assign_params_as_leaves(func);
    let mut leaves = Vec::new();
    leaf_order(func, wanted[0], &shapes, &assigned, &mut leaves).unwrap();
    assert_eq!(leaves, vec![func.params[0], func.params[0]]);
    assert_eq!(shapes.leaves(assigned[wanted[0].index()].unwrap()), 2);
}

/// A shape whose unfolding would explode is refused, so a batch can never
/// replay an astronomically larger program than the DAG it replaces.
#[test]
fn exponentially_unfolding_shapes_are_refused() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let scalar = Shape::scalar(DType::F32);
    let param = builder.param(scalar);
    // Each level squares the previous one, so the unfolding doubles every step.
    let mut value = param;
    for _ in 0..20 {
        value = builder.op(Multiply, &[value, value])[0];
    }
    let main = builder.end_func(vec![value]);
    let module = builder.finish(main).unwrap();
    let func = module.entry_func();

    let params = func.params.clone();
    let (_, assigned) = assign(func, 1 << 12, move |candidate| params.contains(&candidate));
    assert!(
        assigned[value.index()].is_none(),
        "an exponentially unfolding shape was accepted"
    );
}
