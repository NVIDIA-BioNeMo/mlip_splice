// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unrolling a region into the circuit it denotes. Each program here holds
//! constants alone, so folding the unrolled circuit answers with the sorted or
//! accumulated values themselves: the network is checked by arithmetic rather
//! than by counting nodes.

use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{FuncId, Module};
use xir::ops::{Add, Compare, ComparisonDirection, Constant, Literal, Sort, While};
use xir::shape::Shape;
use xir_optimize::{Pass, PassManager};

fn i32_literal(values: &[i32]) -> Literal {
    Literal::new(
        Shape::new(DType::I32, [values.len() as i64]),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn folded_i32s(module: &Module, result: usize) -> Vec<i32> {
    let func = module.entry_func();
    let value = func.results[result];
    let literal = xir::analysis::constant_literal(func, value)
        .unwrap_or_else(|| panic!("result {result} did not fold to a constant"));
    literal
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect()
}

fn unroll_and_fold(module: &mut Module) -> bool {
    let report = PassManager::new([Pass::UnrollStaticRegions, Pass::ConstantFolding])
        .run(module)
        .expect("pipeline");
    report.passes[0].changed
}

/// `comparator(lhs_key, rhs_key, …) = lhs_key < rhs_key`, over `operands` keys
/// interleaved as `sort` passes them, comparing the first operand alone.
fn ascending_by_first_key(builder: &mut ModuleBuilder, operands: usize) -> FuncId {
    builder.func("ascending", |func| {
        let scalars = (0..2 * operands)
            .map(|_| func.param(Shape::scalar(DType::I32)))
            .collect::<Vec<_>>();
        vec![func.op1(
            Compare {
                comparison_direction: ComparisonDirection::Lt,
                compare_type: None,
            },
            &[scalars[0], scalars[1]],
        )]
    })
}

#[test]
fn a_sort_becomes_a_network_that_orders_its_elements() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 1);
    let main = builder.func("main", |func| {
        let values = func.constant(i32_literal(&[5, -3, 9, 0, 9, 1, -7]));
        func.op(
            Sort {
                dimension: 0,
                is_stable: true,
                comparator,
            },
            &[values],
        )
        .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert_eq!(folded_i32s(&module, 0), [-7, -3, 0, 1, 5, 9, 9]);
}

#[test]
fn a_sort_of_equal_keys_keeps_the_original_order() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 2);
    let main = builder.func("main", |func| {
        let keys = func.constant(i32_literal(&[1, 0, 1, 0, 1]));
        let payload = func.constant(i32_literal(&[10, 11, 12, 13, 14]));
        func.op(
            Sort {
                dimension: 0,
                is_stable: true,
                comparator,
            },
            &[keys, payload],
        )
        .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert_eq!(folded_i32s(&module, 0), [0, 0, 1, 1, 1]);
    assert_eq!(folded_i32s(&module, 1), [11, 13, 10, 12, 14]);
}

#[test]
fn a_sort_orders_each_lane_of_a_chosen_axis() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 1);
    let main = builder.func("main", |func| {
        // Two rows of three, sorted along the rows.
        let values = func.constant(Literal::new(
            Shape::new(DType::I32, [2, 3]),
            [3, 1, 2, 9, 8, 7]
                .iter()
                .flat_map(|v: &i32| v.to_le_bytes())
                .collect(),
        ));
        func.op(
            Sort {
                dimension: 1,
                is_stable: false,
                comparator,
            },
            &[values],
        )
        .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert_eq!(folded_i32s(&module, 0), [1, 2, 3, 7, 8, 9]);
}

#[test]
fn a_sort_longer_than_the_bound_is_left_alone() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 1);
    let main = builder.func("main", |func| {
        let values = func.param(Shape::new(DType::I32, [4096]));
        func.op(
            Sort {
                dimension: 0,
                is_stable: true,
                comparator,
            },
            &[values],
        )
        .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(!unroll_and_fold(&mut module));

    assert!(
        module
            .entry_func()
            .nodes
            .iter()
            .any(|node| node.op.mnemonic() == "sort")
    );
}

/// `while (i < limit) { (i, total) = (i + 1, total + i) }`, carrying two scalars.
fn counting_loop(builder: &mut ModuleBuilder, limit: i32) -> (FuncId, FuncId) {
    let scalar = Shape::scalar(DType::I32);
    let condition = builder.func("below_limit", |func| {
        let index = func.param(scalar.clone());
        let _total = func.param(scalar.clone());
        let limit = func.constant(Literal::new(scalar.clone(), limit.to_le_bytes().to_vec()));
        vec![func.op1(
            Compare {
                comparison_direction: ComparisonDirection::Lt,
                compare_type: None,
            },
            &[index, limit],
        )]
    });
    let body = builder.func("step", |func| {
        let index = func.param(scalar.clone());
        let total = func.param(scalar.clone());
        let one = func.constant(Literal::new(scalar.clone(), 1i32.to_le_bytes().to_vec()));
        vec![func.op1(Add, &[index, one]), func.op1(Add, &[total, index])]
    });
    (condition, body)
}

#[test]
fn a_loop_with_a_readable_count_becomes_its_iterations() {
    let mut builder = ModuleBuilder::new();
    let (condition, body) = counting_loop(&mut builder, 5);
    let scalar = Shape::scalar(DType::I32);
    let main = builder.func("main", |func| {
        let index = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        let total = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        func.op(While::new(condition, body), &[index, total])
            .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    // Five iterations: the index reaches the limit, the total reaches 0+1+2+3+4.
    assert_eq!(folded_i32s(&module, 0), [5]);
    assert_eq!(folded_i32s(&module, 1), [10]);
    assert!(
        !module
            .entry_func()
            .nodes
            .iter()
            .any(|node| node.op.mnemonic() == "while")
    );
}

#[test]
fn a_loop_whose_predicate_is_not_decided_is_left_alone() {
    let mut builder = ModuleBuilder::new();
    let (condition, body) = counting_loop(&mut builder, 5);
    let scalar = Shape::scalar(DType::I32);
    let main = builder.func("main", |func| {
        // The starting index is read at run time, so no iteration count exists.
        let index = func.param(scalar.clone());
        let total = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        func.op(While::new(condition, body), &[index, total])
            .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(!unroll_and_fold(&mut module));

    let nodes = &module.entry_func().nodes;
    assert!(nodes.iter().any(|node| node.op.mnemonic() == "while"));
    // Nothing speculatively emitted survives: one constant and one loop.
    assert_eq!(
        nodes
            .iter()
            .filter(|node| node.op.as_any().is::<Constant>())
            .count(),
        1,
        "speculative unrolling left nodes behind"
    );
}

#[test]
fn a_loop_that_never_ends_is_left_alone() {
    let mut builder = ModuleBuilder::new();
    let scalar = Shape::scalar(DType::I32);
    let condition = builder.func("always", |func| {
        let _index = func.param(scalar.clone());
        vec![func.constant(Literal::new(Shape::scalar(DType::Bool), vec![1]))]
    });
    let body = builder.func("identity", |func| {
        let index = func.param(scalar.clone());
        vec![index]
    });
    let main = builder.func("main", |func| {
        let index = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        func.op(While::new(condition, body), &[index]).to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(!unroll_and_fold(&mut module));

    assert!(
        module
            .entry_func()
            .nodes
            .iter()
            .any(|node| node.op.mnemonic() == "while")
    );
}

#[test]
fn a_program_without_regions_is_untouched() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::I32, [4]));
        vec![func.op1(Add, &[x, x])]
    });
    let mut module = builder.finish(main).unwrap();
    let before = module.entry_func().nodes.len();

    let report = PassManager::new([Pass::UnrollStaticRegions])
        .run(&mut module)
        .unwrap();

    assert!(!report.passes[0].changed);
    assert_eq!(module.entry_func().nodes.len(), before);
}

#[test]
fn an_unrolled_loop_carries_tensors() {
    let mut builder = ModuleBuilder::new();
    let scalar = Shape::scalar(DType::I32);
    let vector = Shape::new(DType::I32, [3]);
    let condition = builder.func("below_two", |func| {
        let index = func.param(scalar.clone());
        let _values = func.param(vector.clone());
        let two = func.constant(Literal::new(scalar.clone(), 2i32.to_le_bytes().to_vec()));
        vec![func.op1(
            Compare {
                comparison_direction: ComparisonDirection::Lt,
                compare_type: None,
            },
            &[index, two],
        )]
    });
    let body = builder.func("double", |func| {
        let index = func.param(scalar.clone());
        let values = func.param(vector.clone());
        let one = func.constant(Literal::new(scalar.clone(), 1i32.to_le_bytes().to_vec()));
        vec![
            func.op1(Add, &[index, one]),
            func.op1(Add, &[values, values]),
        ]
    });
    let main = builder.func("main", |func| {
        let index = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        let values = func.constant(i32_literal(&[1, 2, 3]));
        func.op(While::new(condition, body), &[index, values])
            .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert_eq!(folded_i32s(&module, 1), [4, 8, 12]);
}

#[test]
fn a_nested_region_is_unrolled_too() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 1);
    let scalar = Shape::scalar(DType::I32);
    let vector = Shape::new(DType::I32, [3]);
    let condition = builder.func("once", |func| {
        let index = func.param(scalar.clone());
        let _values = func.param(vector.clone());
        let one = func.constant(Literal::new(scalar.clone(), 1i32.to_le_bytes().to_vec()));
        vec![func.op1(
            Compare {
                comparison_direction: ComparisonDirection::Lt,
                compare_type: None,
            },
            &[index, one],
        )]
    });
    let body = builder.func("sort_the_carry", |func| {
        let index = func.param(scalar.clone());
        let values = func.param(vector.clone());
        let one = func.constant(Literal::new(scalar.clone(), 1i32.to_le_bytes().to_vec()));
        let sorted = func.op(
            Sort {
                dimension: 0,
                is_stable: true,
                comparator,
            },
            &[values],
        )[0];
        vec![func.op1(Add, &[index, one]), sorted]
    });
    let main = builder.func("main", |func| {
        let index = func.constant(Literal::new(scalar.clone(), 0i32.to_le_bytes().to_vec()));
        let values = func.constant(i32_literal(&[3, 1, 2]));
        func.op(While::new(condition, body), &[index, values])
            .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert_eq!(folded_i32s(&module, 1), [1, 2, 3]);
    let mnemonics = module
        .entry_func()
        .nodes
        .iter()
        .map(|node| node.op.mnemonic())
        .collect::<Vec<_>>();
    assert!(!mnemonics.contains(&"while"), "{mnemonics:?}");
    assert!(!mnemonics.contains(&"sort"), "{mnemonics:?}");
}

#[test]
fn a_sort_of_one_element_needs_no_gate() {
    let mut builder = ModuleBuilder::new();
    let comparator = ascending_by_first_key(&mut builder, 1);
    let main = builder.func("main", |func| {
        let values = func.param(Shape::new(DType::I32, [1]));
        func.op(
            Sort {
                dimension: 0,
                is_stable: true,
                comparator,
            },
            &[values],
        )
        .to_vec()
    });
    let mut module = builder.finish(main).unwrap();

    assert!(unroll_and_fold(&mut module));

    assert!(module.entry_func().nodes.is_empty());
}
