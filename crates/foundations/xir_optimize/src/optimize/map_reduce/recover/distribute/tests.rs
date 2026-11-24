// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::builder::ModuleBuilder;
use crate::dtype::DType;
use crate::ops::{Add, Maximum, Multiply, Tanh};
use crate::shape::Shape;

use super::super::scalar_region;
use super::*;

fn build(
    params: usize,
    body: impl FnOnce(&mut ModuleBuilder, &[ValueId]) -> ValueId,
) -> (crate::ir::Module, ValueId) {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let scalar = Shape::scalar(DType::F32);
    let values = (0..params)
        .map(|_| builder.param(scalar.clone()))
        .collect::<Vec<_>>();
    let root = body(&mut builder, &values);
    let main = builder.end_func(vec![root]);
    (builder.finish(main).unwrap(), root)
}

fn expand(module: &crate::ir::Module, root: ValueId, limit: usize) -> Option<Vec<Term>> {
    let func = module.entry_func();
    let region = scalar_region(func);
    Expansion::new(ScalarOp::Add, limit).terms(func, root, &region)
}

/// The whole reason this module exists: the extractor writes `a*(b + c)`, and
/// that form contains no wide reduction. Distributing recovers the two terms the
/// e-class also denotes, without touching the program.
#[test]
fn a_product_over_a_sum_expands_into_terms() {
    let (module, root) = build(3, |builder, p| {
        let sum = builder.op(Add, &[p[1], p[2]])[0];
        builder.op(Multiply, &[p[0], sum])[0]
    });
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 2, "a*(b + c) is two terms");
    for term in &terms {
        assert_eq!(term.factors.len(), 2, "each term is a product of two atoms");
        assert!(term.factors.contains(&module.entry_func().params[0]));
    }
}

/// Nested factoring is what the extractor actually produces. The expansion has
/// to reach through every level, not just the outermost product.
#[test]
fn nested_factoring_expands_to_the_full_product() {
    let (module, root) = build(4, |builder, p| {
        let inner = builder.op(Add, &[p[2], p[3]])[0];
        let scaled = builder.op(Multiply, &[p[1], inner])[0];
        let outer = builder.op(Add, &[p[1], scaled])[0];
        builder.op(Multiply, &[p[0], outer])[0]
    });
    // a*(b + b*(c + d)) = a*b + a*b*c + a*b*d
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 3);
    let mut arities = terms.iter().map(|t| t.factors.len()).collect::<Vec<_>>();
    arities.sort_unstable();
    assert_eq!(arities, vec![2, 3, 3]);
}

/// An operation with no declared law is an atom. The expansion stops there
/// rather than guessing, which is what keeps this generic instead of arithmetic.
#[test]
fn an_operation_without_a_law_is_an_atom() {
    let (module, root) = build(3, |builder, p| {
        let sum = builder.op(Add, &[p[1], p[2]])[0];
        // tanh does not distribute over add, so the sum stays inside it.
        let activated = builder.op(Tanh, &[sum])[0];
        builder.op(Multiply, &[p[0], activated])[0]
    });
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 1, "tanh sealed the sum inside one atom");
    assert_eq!(terms[0].factors.len(), 2);
}

/// The declared table is the only algebra. `max` is associative and commutative
/// but multiply does not distribute over it, so nothing expands.
#[test]
fn a_reducer_without_the_law_does_not_expand() {
    assert!(!distributes_over(ScalarOp::Multiply, ScalarOp::Maximum));
    assert!(distributes_over(ScalarOp::Add, ScalarOp::Maximum));
    let (module, root) = build(3, |builder, p| {
        let peak = builder.op(Maximum, &[p[1], p[2]])[0];
        builder.op(Multiply, &[p[0], peak])[0]
    });
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 1, "multiply does not distribute over add here");
}

/// Distribution is exponential in a shared DAG. The bound has to turn that into
/// a refusal, not an allocation the size of the machine.
#[test]
fn an_exponential_expansion_is_refused_rather_than_built() {
    let (module, root) = build(2, |builder, p| {
        // Each level squares the term count: 2^20 terms if fully expanded.
        let mut value = builder.op(Add, &[p[0], p[1]])[0];
        for _ in 0..20 {
            value = builder.op(Multiply, &[value, value])[0];
        }
        value
    });
    assert!(
        expand(&module, root, 1 << 12).is_none(),
        "an exponential expansion was built instead of refused"
    );
    // The same program is fine once the bound admits it.
    let (small, small_root) = build(2, |builder, p| {
        let mut value = builder.op(Add, &[p[0], p[1]])[0];
        for _ in 0..3 {
            value = builder.op(Multiply, &[value, value])[0];
        }
        value
    });
    assert_eq!(
        expand(&small, small_root, 1 << 12).map(|terms| terms.len()),
        Some(256)
    );
}

/// A shared subexpression must expand once and be copied, or the walk costs
/// exponential time even when its result fits.
#[test]
fn a_shared_subexpression_expands_once() {
    let (module, root) = build(3, |builder, p| {
        let shared = builder.op(Add, &[p[0], p[1]])[0];
        let left = builder.op(Multiply, &[shared, p[2]])[0];
        let right = builder.op(Multiply, &[shared, p[2]])[0];
        builder.op(Add, &[left, right])[0]
    });
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 4, "(a+b)*c twice is four terms");
}

/// A gradient program is full of subtractions and negations. Neither is the
/// reducer nor distributes over it, so without the commuting law each one seals
/// everything below it into a single atom and truncates the expansion.
#[test]
fn subtraction_and_negation_do_not_stop_the_expansion() {
    let (module, root) = build(4, |builder, p| {
        let sum = builder.op(Add, &[p[2], p[3]])[0];
        let scaled = builder.op(Multiply, &[p[1], sum])[0];
        let negated = builder.op(crate::ops::Negate, &[scaled])[0];
        // a - (-(b*(c + d))) reaches all the way down to c and d.
        let difference = builder.op(crate::ops::Subtract, &[p[0], negated])[0];
        builder.op(Multiply, &[p[0], difference])[0]
    });
    let terms = expand(&module, root, 64).expect("the expansion fits");
    assert_eq!(terms.len(), 3, "a*a + a*b*c + a*b*d, up to sign");
    let negated = terms.iter().filter(|t| !t.wrappers.is_empty()).count();
    assert_eq!(negated, 2, "the two terms under the negation carry it");
    assert!(
        terms.iter().all(|t| t.factors.len() >= 2),
        "every term reached its atoms"
    );
}
