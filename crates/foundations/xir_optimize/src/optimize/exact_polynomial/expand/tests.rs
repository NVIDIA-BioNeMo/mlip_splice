// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::optimize::AlgebraicWorkBudget;
use crate::{builder::ModuleBuilder, dtype::DType, ir::Module, ops::Literal, shape::Shape};

fn shared_square() -> Module {
    let mut b = ModuleBuilder::new();
    b.begin_func("shared_square");
    let shape = Shape::scalar(DType::F64);
    let x = b.param(shape.clone());
    let y = b.param(shape.clone());
    let sum = b.add(x, y);
    let square = b.multiply(sum, sum);
    let two = b.constant(Literal::new(shape, 2.0f64.to_le_bytes().to_vec()));
    let doubled = b.multiply(two, square);
    let zero = b.binary("subtract", square, square).unwrap();
    let zero_product = b.multiply(zero, square);
    let entry = b.end_func(vec![square, doubled, zero_product]);
    b.finish(entry).unwrap()
}

#[test]
fn shared_cache_entries_remain_identical_after_borrowed_operations() {
    let module = shared_square();
    let func = module.entry_func();
    let mut expander = BoundedPolynomialExpander::new(func, 1024);
    let mut work = AlgebraicWorkBudget::new(1 << 16, 1 << 12);
    let BoundedExpansion::Available(shared) = expander.expand(func.results[0], &mut work) else {
        panic!("the shared square did not expand");
    };
    for &result in &func.results {
        let BoundedExpansion::Available(expansion) = expander.expand(result, &mut work) else {
            panic!("a polynomial result did not expand");
        };
        let expected = full_expansion(func, result, 1024).unwrap();
        assert_eq!(expansion.polynomial, expected.polynomial);
        assert_eq!(expansion.expanded_terms, expected.expanded_terms);
        assert_eq!(expansion.has_constant, expected.has_constant);
    }
    let BoundedExpansion::Available(cached) = expander.expand(func.results[0], &mut work) else {
        panic!("the shared square was lost");
    };
    assert!(Arc::ptr_eq(&shared, &cached));
    assert_eq!(work.abandoned_count(), 0);
}

#[test]
fn exhausted_expansion_can_resume_without_poisoning_completed_inputs() {
    let module = shared_square();
    let func = module.entry_func();
    let root = func.results[1];
    let expected = full_expansion(func, root, 1024).unwrap();
    for total in 0..100 {
        let mut expander = BoundedPolynomialExpander::new(func, 1024);
        let mut limited = AlgebraicWorkBudget::new(total, 1024);
        let _ = expander.expand(root, &mut limited);
        assert!(limited.consumed <= total);
        let mut work = AlgebraicWorkBudget::new(1 << 16, 1 << 12);
        let BoundedExpansion::Available(result) = expander.expand(root, &mut work) else {
            panic!("resuming after {total} units failed");
        };
        assert_eq!(result.polynomial, expected.polynomial);
    }
}
