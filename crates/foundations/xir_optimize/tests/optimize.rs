// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use smallvec::{SmallVec, smallvec};

use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Def, Func, FuncId};
use xir::op::{InferCtx, Op};
use xir::ops::{
    BroadcastInDim, ComparisonDirection, ComparisonType, Concatenate, Convert, Gather, Literal,
    Pad, Reshape, Select, Slice, Transpose,
};
use xir::shape::{Shape, ShapeError};
use xir_optimize::{Pass, PassManager, PipelineReport};

fn cleanup_pipeline() -> PassManager {
    PassManager::new([
        Pass::CanonicalizeShapes,
        Pass::ConstantFolding,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
}

#[test]
fn cleanup_pipeline_canonicalizes_cses_and_eliminates_dead_code() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [2, 3]));
        let converted = func.op1(Convert { dtype: DType::F32 }, &[x]);
        let reshaped = func.op1(
            Reshape {
                out_dims: smallvec![2, 3],
            },
            &[converted],
        );
        let transposed = func.op1(
            Transpose {
                permutation: smallvec![0, 1],
            },
            &[reshaped],
        );
        let broadcast = func.op1(
            BroadcastInDim {
                out_dims: smallvec![2, 3],
                broadcast_dimensions: smallvec![0, 1],
            },
            &[transposed],
        );
        let sliced = func.op1(
            Slice {
                start_indices: smallvec![0, 0],
                limit_indices: smallvec![2, 3],
                strides: smallvec![1, 1],
            },
            &[broadcast],
        );
        let concatenated = func.op1(Concatenate { dimension: 0 }, &[sliced]);
        let zero = func.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        let padded = func.op1(
            Pad {
                edge_padding_low: smallvec![0, 0],
                edge_padding_high: smallvec![0, 0],
                interior_padding: smallvec![0, 0],
            },
            &[concatenated, zero],
        );
        let sum_a = func.add(padded, padded);
        let sum_b = func.add(padded, padded);
        let _dead = func.multiply(sum_a, padded);
        vec![sum_a, sum_b]
    });
    let mut module = builder.finish(main).unwrap();

    let report = cleanup_pipeline().run(&mut module).unwrap();

    assert_report(
        &report,
        &["canonicalize-shapes", "constant-folding", "cse", "dce"],
    );
    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 1);
    assert_eq!(func.nodes[0].op.mnemonic(), "add");
    assert_eq!(
        func.nodes[0].inputs.as_slice(),
        &[func.params[0], func.params[0]]
    );
    assert_eq!(func.results[0], func.results[1]);
    assert!(report.passes[0].changed);
    assert!(!report.passes[1].changed);
    assert!(report.passes[2].changed);
    assert!(report.passes[3].changed);
}

#[test]
fn isomorphic_scalar_operations_are_packed_after_algebraic_optimization() {
    // Each term reads its own adjacent pair, so the terms are genuinely distinct
    // rather than one value written twice: nothing here is recovered by replaying
    // work the program already shares.
    const TERMS: i64 = 20;
    const OUTPUTS: i64 = 2;
    let mut builder = ModuleBuilder::new();
    let main = builder.func("scalar_polynomials", |func| {
        let values = func.param(Shape::new(DType::F32, [2 * TERMS + OUTPUTS]));
        let weights = func.param(Shape::new(DType::F32, [TERMS]));
        let scalar = |func: &mut ModuleBuilder, value, element| {
            let slice = func.op1(
                Slice {
                    start_indices: smallvec![element],
                    limit_indices: smallvec![element + 1],
                    strides: smallvec![1],
                },
                &[value],
            );
            func.op1(
                Reshape {
                    out_dims: SmallVec::new(),
                },
                &[slice],
            )
        };
        let coefficient = func.constant(
            Literal::exact(Shape::scalar(DType::F32), vec![AlgebraicReal::sqrt(2u32)]).unwrap(),
        );
        let mut outputs = Vec::new();
        for output in 0..OUTPUTS {
            let mut terms = Vec::new();
            for term in 0..TERMS {
                let left = scalar(func, values, 2 * term + output);
                let right = scalar(func, values, 2 * term + output + 1);
                let weight = scalar(func, weights, term);
                let product = func.multiply(left, right);
                let product = func.multiply(product, weight);
                terms.push(func.multiply(coefficient, product));
            }
            let value = terms[1..]
                .iter()
                .fold(terms[0], |sum, &term| func.add(sum, term));
            outputs.push(func.op1(
                Reshape {
                    out_dims: smallvec![1],
                },
                &[value],
            ));
        }
        vec![func.op1(Concatenate { dimension: 0 }, &outputs)]
    });
    let mut module = builder.finish(main).unwrap();

    let report = PassManager::new([
        Pass::IndexedAlgebraCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
        Pass::RecoverMapReduceRegions,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    assert!(report.passes[3].changed);
    let func = module.entry_func();
    // What was recovered, not merely that something was: each output's twenty
    // terms become one batch, so every leaf position is a gather that wide and
    // the output is one reduction over it rather than twenty scalar additions.
    let gathers = func
        .nodes
        .iter()
        .filter(|node| node.op.as_any().is::<Gather>())
        .map(|node| func.value_shape(node.results[0]).dims.as_slice().to_vec())
        .collect::<Vec<_>>();
    assert!(!gathers.is_empty(), "no operand was gathered");
    assert!(
        gathers.iter().all(|dims| dims.as_slice() == [TERMS]),
        "a batch narrower than the term list: {gathers:?}"
    );
    let scatters = func
        .nodes
        .iter()
        .filter(|node| node.op.as_any().is::<xir::ops::Scatter>())
        .collect::<Vec<_>>();
    assert_eq!(scatters.len(), OUTPUTS as usize, "one reduction per output");
    for scatter in scatters {
        assert_eq!(
            func.value_shape(scatter.inputs[2]).dims.as_slice(),
            &[TERMS],
            "a reduction over fewer terms than the output has"
        );
    }
    assert_eq!(
        func.value_shape(func.results[0]).dims.as_slice(),
        &[OUTPUTS]
    );
}

#[test]
fn shape_composition_collapses_nested_transposes_reshapes_and_broadcasts() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [1, 2, 3]));
        let y = func.param(Shape::new(DType::F32, [1, 2]));
        let transpose_a = func.op1(
            Transpose {
                permutation: smallvec![1, 2, 0],
            },
            &[x],
        );
        let transpose_b = func.op1(
            Transpose {
                permutation: smallvec![2, 0, 1],
            },
            &[transpose_a],
        );
        let reshape_a = func.op1(
            Reshape {
                out_dims: smallvec![6],
            },
            &[x],
        );
        let reshape_b = func.op1(
            Reshape {
                out_dims: smallvec![2, 3],
            },
            &[reshape_a],
        );
        let broadcast_a = func.op1(
            BroadcastInDim {
                out_dims: smallvec![3, 1, 2],
                broadcast_dimensions: smallvec![1, 2],
            },
            &[y],
        );
        let broadcast_b = func.op1(
            BroadcastInDim {
                out_dims: smallvec![4, 3, 1, 2],
                broadcast_dimensions: smallvec![1, 2, 3],
            },
            &[broadcast_a],
        );
        vec![transpose_b, reshape_b, broadcast_b]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 2);
    assert_eq!(func.results[0], func.params[0]);
    assert_eq!(func.nodes[0].op.mnemonic(), "reshape");
    assert_eq!(func.nodes[0].inputs.as_slice(), &[func.params[0]]);
    let broadcast = func.nodes[1]
        .op
        .as_any()
        .downcast_ref::<BroadcastInDim>()
        .unwrap();
    assert_eq!(broadcast.broadcast_dimensions.as_slice(), &[2, 3]);
    assert_eq!(func.nodes[1].inputs.as_slice(), &[func.params[1]]);
}

#[test]
fn unknown_operations_are_not_csed_or_eliminated() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let _first = func.op1(Opaque, &[x]);
        let _second = func.op1(Opaque, &[x]);
        vec![x]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    assert_eq!(module.entry_func().nodes.len(), 2);
}

#[test]
fn cse_keys_constants_by_shape_and_payload() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::scalar(DType::F32);
        let one_a = func.constant(Literal::new(shape.clone(), 1.0f32.to_le_bytes().to_vec()));
        let one_b = func.constant(Literal::new(shape.clone(), 1.0f32.to_le_bytes().to_vec()));
        let two = func.constant(Literal::new(shape, 2.0f32.to_le_bytes().to_vec()));
        vec![one_a, one_b, two]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 2);
    assert_eq!(func.results[0], func.results[1]);
    assert_ne!(func.results[0], func.results[2]);
}

#[test]
fn constant_folding_evaluates_shape_operation_chains() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let input = func.constant(i32_literal(&[4], &[1, 2, 3, 4]));
        let reshaped = func.op1(
            Reshape {
                out_dims: smallvec![2, 2],
            },
            &[input],
        );
        let transposed = func.op1(
            Transpose {
                permutation: smallvec![1, 0],
            },
            &[reshaped],
        );
        let sliced = func.op1(
            Slice {
                start_indices: smallvec![0, 0],
                limit_indices: smallvec![2, 1],
                strides: smallvec![1, 1],
            },
            &[transposed],
        );
        let broadcast = func.op1(
            BroadcastInDim {
                out_dims: smallvec![2, 3],
                broadcast_dimensions: smallvec![0, 1],
            },
            &[sliced],
        );
        let concatenated = func.op1(Concatenate { dimension: 1 }, &[broadcast, broadcast]);
        vec![concatenated]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 1);
    let literal = result_literal(func);
    assert_eq!(literal.shape, Shape::new(DType::I32, [2, 6]));
    assert_eq!(
        i32_values(literal),
        vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    );
}

#[test]
fn constant_folding_handles_padding_and_selection() {
    let mut builder = ModuleBuilder::new();
    let padded = builder.func("padded", |func| {
        let input = func.constant(i32_literal(&[3], &[10, 20, 30]));
        let padding = func.constant(i32_literal(&[], &[-1]));
        vec![func.op1(
            Pad {
                edge_padding_low: smallvec![-1],
                edge_padding_high: smallvec![1],
                interior_padding: smallvec![1],
            },
            &[input, padding],
        )]
    });
    let mut module = builder.finish(padded).unwrap();
    cleanup_pipeline().run(&mut module).unwrap();
    assert_eq!(
        i32_values(result_literal(module.entry_func())),
        vec![-1, 20, -1, 30, -1]
    );

    let mut builder = ModuleBuilder::new();
    let selected = builder.func("selected", |func| {
        let dynamic = func.param(Shape::new(DType::I32, [2]));
        let predicate = func.constant(Literal::new(Shape::scalar(DType::Bool), vec![1]));
        let yes = func.constant(i32_literal(&[2], &[7, 8]));
        vec![func.op1(Select, &[predicate, yes, dynamic])]
    });
    let mut module = builder.finish(selected).unwrap();
    cleanup_pipeline().run(&mut module).unwrap();
    assert_eq!(i32_values(result_literal(module.entry_func())), vec![7, 8]);
}

#[test]
fn constant_folding_evaluates_integer_and_boolean_elementwise_ops() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let a = func.constant(i8_literal(&[2], &[120, -7]));
        let b = func.constant(i8_literal(&[2], &[10, 3]));
        let shifts = func.constant(i8_literal(&[2], &[2, 1]));
        let bases = func.constant(i8_literal(&[2], &[2, 3]));
        let exponents = func.constant(i8_literal(&[2], &[7, 3]));
        let add = func.binary("add", a, b).unwrap();
        let sub = func.binary("subtract", a, b).unwrap();
        let mul = func.binary("multiply", a, b).unwrap();
        let div = func.binary("divide", a, b).unwrap();
        let rem = func.binary("remainder", a, b).unwrap();
        let neg = func.unary("negate", a).unwrap();
        let bit_not = func.unary("not", a).unwrap();
        let left = func.binary("shift_left", a, shifts).unwrap();
        let arithmetic = func.binary("shift_right_arithmetic", a, shifts).unwrap();
        let logical = func.binary("shift_right_logical", a, shifts).unwrap();
        let power = func.binary("power", bases, exponents).unwrap();
        let compare = func.compare(ComparisonDirection::Lt, None, a, b);
        let compare_unsigned = func.compare(
            ComparisonDirection::Lt,
            Some(ComparisonType::Unsigned),
            a,
            b,
        );
        vec![
            add,
            sub,
            mul,
            div,
            rem,
            neg,
            bit_not,
            left,
            arithmetic,
            logical,
            power,
            compare,
            compare_unsigned,
        ]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    let expected = [
        vec![-126, -4],
        vec![110, -10],
        vec![-80, -21],
        vec![12, -2],
        vec![0, -1],
        vec![-120, 7],
        vec![-121, 6],
        vec![-32, -14],
        vec![30, -4],
        vec![30, 124],
        vec![-128, 27],
    ];
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(i8_values(literal_for_result(func, index)), *expected);
    }
    assert_eq!(literal_for_result(func, 11).data, vec![0, 1]);
    assert_eq!(literal_for_result(func, 12).data, vec![0, 0]);
}

#[test]
fn constant_folding_respects_literal_size_budget() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let input = func.constant(i32_literal(&[], &[3]));
        vec![func.op1(
            BroadcastInDim {
                out_dims: smallvec![262_145],
                broadcast_dimensions: SmallVec::new(),
            },
            &[input],
        )]
    });
    let mut module = builder.finish(main).unwrap();

    let report = cleanup_pipeline().run(&mut module).unwrap();

    assert_eq!(module.entry_func().nodes.len(), 2);
    assert!(!report.passes[1].changed);
}

#[test]
fn algebraic_pass_applies_floating_identities() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let y = func.param(Shape::new(DType::F32, [4]));
        let zero = func.constant(f32_literal(&[4], &[0.0; 4]));
        let one = func.constant(f32_literal(&[4], &[1.0; 4]));
        let add_zero = func.add(x, zero);
        let mul_one = func.multiply(x, one);
        let mul_zero = func.multiply(x, zero);
        let div_one = func.binary("divide", x, one).unwrap();
        let inner_neg = func.unary("negate", x).unwrap();
        let double_neg = func.unary("negate", inner_neg).unwrap();
        let zero_minus = func.binary("subtract", zero, x).unwrap();
        let neg_y = func.unary("negate", y).unwrap();
        let add_neg = func.add(x, neg_y);
        vec![
            add_zero, mul_one, mul_zero, div_one, double_neg, zero_minus, add_neg,
        ]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();

    let func = module.entry_func();
    let x = func.params[0];
    let y = func.params[1];
    assert_eq!(func.results[0], x, "x + 0 -> x");
    assert_eq!(func.results[1], x, "x * 1 -> x");
    assert!(
        f32_values(literal_for_result(func, 2))
            .iter()
            .all(|&v| v == 0.0),
        "x * 0 -> 0"
    );
    assert_eq!(func.results[3], x, "x / 1 -> x");
    assert_eq!(func.results[4], x, "-(-x) -> x");
    let (negate, negate_inputs) = producing_node(func, 5);
    assert_eq!(negate, "negate", "0 - x -> -x");
    assert_eq!(negate_inputs, vec![x]);
    let (subtract, subtract_inputs) = producing_node(func, 6);
    assert_eq!(subtract, "subtract", "x + (-y) -> x - y");
    assert_eq!(subtract_inputs, vec![x, y]);
}

#[test]
fn algebraic_pass_orders_commutative_operands_and_exposes_cse() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let a = func.param(Shape::new(DType::F32, [4]));
        let b = func.param(Shape::new(DType::F32, [4]));
        let forward = func.multiply(a, b);
        let reversed = func.multiply(b, a);
        vec![forward, reversed]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([
        Pass::AlgebraicCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
    .run(&mut module)
    .unwrap();

    let func = module.entry_func();
    assert_eq!(func.nodes.len(), 1, "a*b and b*a collapse to one product");
    assert_eq!(func.nodes[0].op.mnemonic(), "multiply");
    assert_eq!(func.results[0], func.results[1]);
    assert_eq!(
        func.nodes[0].inputs.as_slice(),
        &[func.params[0], func.params[1]]
    );
}

#[test]
fn algebraic_pass_reaches_a_stable_fixed_point() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let y = func.param(Shape::new(DType::F32, [4]));
        let reversed = func.multiply(y, x);
        let zero = func.constant(f32_literal(&[4], &[0.0; 4]));
        vec![func.add(reversed, zero)]
    });
    let mut module = builder.finish(main).unwrap();

    let first = PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();
    assert!(first.passes[0].changed);

    let second = PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();
    assert!(!second.passes[0].changed, "rewrites are idempotent");
}

#[test]
fn algebraic_pass_respects_the_literal_budget() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let two = func.constant(f32_literal(&[262_145], &vec![2.0; 262_145]));
        vec![func.binary("divide", two, two).unwrap()]
    });
    let mut module = builder.finish(main).unwrap();

    let report = PassManager::new([Pass::AlgebraicCanonicalization])
        .run(&mut module)
        .unwrap();

    assert!(
        !report.passes[0].changed,
        "constant quotient stays over-budget"
    );
    assert!(report.passes[0].nodes_after <= report.passes[0].nodes_before);
    assert_eq!(producing_node(module.entry_func(), 0).0, "divide");
}

#[test]
fn algebraic_pipeline_preserves_self_division() {
    for dtype in [DType::F16, DType::Bf16, DType::F32, DType::F64] {
        for dims in [vec![], vec![4], vec![262_145]] {
            let mut builder = ModuleBuilder::new();
            let main = builder.func("main", |func| {
                let x = func.param(Shape::new(dtype, dims));
                vec![func.binary("divide", x, x).unwrap()]
            });
            let mut module = builder.finish(main).unwrap();

            xir_optimize::algebraic_pipeline().run(&mut module).unwrap();

            let func = module.entry_func();
            assert_eq!(func.nodes.len(), 1);
            let (op, inputs) = producing_node(func, 0);
            assert_eq!(op, "divide");
            assert_eq!(inputs, vec![func.params[0]; 2]);
        }
    }
}

#[test]
fn algebraic_pass_folds_only_defined_constant_quotients() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let zero = func.constant(f32_literal(&[], &[0.0]));
        let two = func.constant(f32_literal(&[], &[2.0]));
        vec![
            func.binary("divide", zero, zero).unwrap(),
            func.binary("divide", two, two).unwrap(),
        ]
    });
    let mut module = builder.finish(main).unwrap();

    PassManager::new([Pass::AlgebraicCanonicalization, Pass::DeadCodeElimination])
        .run(&mut module)
        .unwrap();

    let func = module.entry_func();
    let (op, inputs) = producing_node(func, 0);
    assert_eq!(op, "divide");
    assert_eq!(inputs[0], inputs[1]);
    assert_eq!(f32_values(literal_for_result(func, 1)), vec![1.0]);
}

fn algebraic_pipeline() -> PassManager {
    PassManager::new([
        Pass::CanonicalizeShapes,
        Pass::ConstantFolding,
        Pass::AlgebraicCanonicalization,
        Pass::CommonSubexpressionElimination,
        Pass::DeadCodeElimination,
    ])
}

// A provable region (a constant reshape that folds) sits beside an opaque one
// (an unknown op the analyses cannot see through). One pass run folds the
// provable region and preserves the opaque one — no exception, no fallback.
#[test]
fn mixed_provability_optimizes_the_provable_region_and_preserves_the_opaque_one() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let opaque = func.op1(Opaque, &[x]);
        let constant = func.constant(i32_literal(&[2, 2], &[1, 2, 3, 4]));
        let folded = func.op1(
            Reshape {
                out_dims: smallvec![4],
            },
            &[constant],
        );
        vec![opaque, folded]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(producing_node(func, 0).0, "opaque");
    assert_eq!(i32_values(literal_for_result(func, 1)), [1, 2, 3, 4]);
}

// Poison one node (an opaque op the folder cannot analyze) between two foldable
// regions; every other rewrite still fires. Partial provability yields partial
// benefit, never an all-or-nothing bail.
#[test]
fn poisoning_one_node_still_folds_every_independent_region() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let a = func.constant(i32_literal(&[2, 2], &[1, 2, 3, 4]));
        let first = func.op1(
            Reshape {
                out_dims: smallvec![4],
            },
            &[a],
        );
        let poison = func.op1(Opaque, &[x]);
        let b = func.constant(i32_literal(&[4, 1], &[5, 6, 7, 8]));
        let second = func.op1(
            Reshape {
                out_dims: smallvec![4],
            },
            &[b],
        );
        vec![first, poison, second]
    });
    let mut module = builder.finish(main).unwrap();

    cleanup_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(i32_values(literal_for_result(func, 0)), [1, 2, 3, 4]);
    assert_eq!(producing_node(func, 1).0, "opaque");
    assert_eq!(i32_values(literal_for_result(func, 2)), [5, 6, 7, 8]);
}

// A program in which nothing is provable runs the whole pipeline without error
// and returns every node unchanged. Guards against a pass that aborts, mangles,
// or reports spurious change when no enabling fact holds.
#[test]
fn a_fully_opaque_program_runs_the_pipeline_without_error() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let a = func.op1(Opaque, &[x]);
        let b = func.op1(Opaque, &[a]);
        vec![b]
    });
    let mut module = builder.finish(main).unwrap();

    let report = cleanup_pipeline().run(&mut module).unwrap();

    assert!(report.passes.iter().all(|pass| !pass.changed));
    assert_eq!(module.entry_func().nodes.len(), 2);
}

// The exact-zero fact proven on `x * 0` collapses that subgraph under the
// algebraic policy, while an adjacent opaque op keeps its node: locality holds
// across an algebraic rewrite too.
#[test]
fn an_algebraic_zero_rewrite_stays_local_beside_an_opaque_op() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let x = func.param(Shape::new(DType::F32, [4]));
        let zero = func.constant(Literal::zeros(Shape::new(DType::F32, [4])).unwrap());
        let killed = func.multiply(x, zero);
        let opaque = func.op1(Opaque, &[x]);
        vec![killed, opaque]
    });
    let mut module = builder.finish(main).unwrap();

    algebraic_pipeline().run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(f32_values(literal_for_result(func, 0)), vec![0.0; 4]);
    assert_eq!(producing_node(func, 1).0, "opaque");
}

fn f32_literal(dims: &[i64], values: &[f32]) -> Literal {
    Literal::new(
        Shape::new(DType::F32, dims.iter().copied()),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
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

fn producing_node(func: &Func, result: usize) -> (&'static str, Vec<xir::ir::ValueId>) {
    let Def::NodeResult { node, slot: 0 } = func.values[func.results[result].index()].def else {
        panic!("result is not a node result");
    };
    let node = &func.nodes[node];
    (node.op.mnemonic(), node.inputs.to_vec())
}

fn i32_literal(dims: &[i64], values: &[i32]) -> Literal {
    Literal::new(
        Shape::new(DType::I32, dims.iter().copied()),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

fn i8_literal(dims: &[i64], values: &[i8]) -> Literal {
    Literal::new(
        Shape::new(DType::I8, dims.iter().copied()),
        values.iter().map(|value| *value as u8).collect(),
    )
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

fn i8_values(literal: &Literal) -> Vec<i8> {
    literal.data.iter().map(|value| *value as i8).collect()
}

fn i32_values(literal: &Literal) -> Vec<i32> {
    literal
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect()
}

fn assert_report(report: &PipelineReport, names: &[&str]) {
    assert_eq!(
        report
            .passes
            .iter()
            .map(|pass| pass.name)
            .collect::<Vec<_>>(),
        names
    );
    for pass in &report.passes {
        assert!(pass.nodes_after <= pass.nodes_before);
        assert!(pass.values_after <= pass.values_before);
    }
}

#[derive(Clone, Debug)]
struct Opaque;

impl Op for Opaque {
    fn mnemonic(&self) -> &'static str {
        "opaque"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        Ok(smallvec![ctx.inputs()[0].clone()])
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn called_funcs(&self) -> &[FuncId] {
        &[]
    }
}
