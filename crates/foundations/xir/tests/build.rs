// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::{SmallVec, smallvec};

use xir::builder::ModuleBuilder;
use xir::dim_numbers::{DotDimensionNumbers, GatherDimensionNumbers};
use xir::dtype::DType;
use xir::ir::{Func, FuncId, Module, ValueId};
use xir::op::{InferCtx, Op};
use xir::ops::{
    Add, Clamp, Compare, ComparisonDirection, ComparisonType, Convolution, DotAlgorithm, Gather,
    Literal, ReduceWindow, Select, While,
};
use xir::print::module_to_string;
use xir::shape::Shape;
use xir::verify::verify;

/// A small dense layer: `relu(x @ w + bias)`.
#[test]
fn matmul_bias_relu() {
    let mut m = ModuleBuilder::new();
    let main = m.func("main", |f| {
        let x = f.param(Shape::new(DType::F32, [128, 256]));
        let w = f.param(Shape::new(DType::F32, [256, 64]));
        let bias = f.param(Shape::new(DType::F32, [64]));

        let xw = f.dot_general(
            x,
            w,
            DotDimensionNumbers::matmul(),
            DotAlgorithm::tf32_tf32_f32(),
        ); // f32[128,64]
        let bb = f.broadcast_in_dim(bias, &[128, 64], &[1]); // f32[128,64]
        let pre = f.add(xw, bb);

        let zero_s = f.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        let zero = f.broadcast_in_dim(zero_s, &[128, 64], &[]);
        let out = f.maximum(pre, zero);
        vec![out]
    });
    let module = m.finish(main).unwrap();

    let ef = module.entry_func();
    assert_eq!(ef.value_shape(ef.results[0]).dims.as_slice(), &[128, 64]);

    let expected = "\
func @main(%0: f32[128,256], %1: f32[256,64], %2: f32[64]) -> f32[128,64] {
  %3 = dot_general %0, %1 {lhs_contracting_dimensions=[1], rhs_contracting_dimensions=[0], lhs_batching_dimensions=[], rhs_batching_dimensions=[], algorithm=tf32_tf32_f32} : f32[128,64]
  %4 = broadcast_in_dim %2 {out_dims=[128, 64], broadcast_dimensions=[1]} : f32[128,64]
  %5 = add %3, %4 : f32[128,64]
  %6 = constant : f32[]
  %7 = broadcast_in_dim %6 {out_dims=[128, 64], broadcast_dimensions=[]} : f32[128,64]
  %8 = maximum %5, %7 : f32[128,64]
  return %8
}
";
    assert_eq!(module_to_string(&module), expected);
}

/// A reducer sub-graph referenced by a higher-order `reduce`.
#[test]
fn reduce_sum_over_axis() {
    let mut m = ModuleBuilder::new();
    let add = m.func("add", |f| {
        let a = f.param(Shape::scalar(DType::F32));
        let b = f.param(Shape::scalar(DType::F32));
        let c = f.add(a, b);
        vec![c]
    });
    let main = m.func("main", |f| {
        let x = f.param(Shape::new(DType::F32, [128, 64]));
        let init = f.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        let s = f.reduce(x, init, &[1], add); // f32[128]
        vec![s]
    });
    let module = m.finish(main).unwrap();
    let ef = module.entry_func();
    assert_eq!(ef.value_shape(ef.results[0]).dims.as_slice(), &[128]);
}

/// A `while` whose cond/body sub-graphs carry the loop state.
#[test]
fn while_loop_carries_shape() {
    let mut m = ModuleBuilder::new();
    let cond = m.func("cond", |f| {
        let _v = f.param(Shape::new(DType::F32, [4]));
        let c = f.constant(Literal::zeros(Shape::scalar(DType::Bool)).unwrap());
        vec![c]
    });
    let body = m.func("body", |f| {
        let v = f.param(Shape::new(DType::F32, [4]));
        let z = f.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        let one = f.broadcast_in_dim(z, &[4], &[]);
        let v2 = f.add(v, one);
        vec![v2]
    });
    let main = m.func("main", |f| {
        let init = f.param(Shape::new(DType::F32, [4]));
        let r = f.op(While::new(cond, body), &[init]);
        vec![r[0]]
    });
    let module = m.finish(main).unwrap();
    let ef = module.entry_func();
    assert_eq!(ef.value_shape(ef.results[0]).dims.as_slice(), &[4]);
}

/// No implicit broadcast: a binary op rejects unequal operand shapes.
#[test]
fn add_rejects_mismatched_shapes() {
    let inputs = [Shape::new(DType::F32, [4]), Shape::new(DType::F32, [5])];
    let op = Add;
    let graph = domain_index::FiniteMapGraph::default();
    let ctx = InferCtx::new(&inputs, &[], &graph);
    assert!(op.infer(&ctx).is_err());
}

/// `select` takes a bool predicate over f32 branches and infers f32 — the
/// predicate dtype differs from the branches', so it must NOT use the shared
/// same-dtype elementwise rule.
#[test]
fn select_bool_pred_over_f32_branches_infers_f32() {
    let inputs = [
        Shape::new(DType::Bool, [4]),
        Shape::new(DType::F32, [4]),
        Shape::new(DType::F32, [4]),
    ];
    let graph = domain_index::FiniteMapGraph::default();
    let ctx = InferCtx::new(&inputs, &[], &graph);
    let out = Select.infer(&ctx).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].dtype, DType::F32);
    assert_eq!(out[0].dims.as_slice(), &[4]);
    // A scalar predicate is allowed too.
    let scalar_pred = [
        Shape::scalar(DType::Bool),
        Shape::new(DType::F32, [4]),
        Shape::new(DType::F32, [4]),
    ];
    assert!(
        Select
            .infer(&InferCtx::new(
                &scalar_pred,
                &[],
                &domain_index::FiniteMapGraph::default(),
            ))
            .is_ok()
    );
}

/// `clamp` accepts scalar bounds broadcast against the operand; result is the
/// operand's shape+dtype.
#[test]
fn clamp_allows_scalar_bounds() {
    let inputs = [
        Shape::scalar(DType::F32),
        Shape::new(DType::F32, [4]),
        Shape::scalar(DType::F32),
    ];
    let out = Clamp
        .infer(&InferCtx::new(
            &inputs,
            &[],
            &domain_index::FiniteMapGraph::default(),
        ))
        .unwrap();
    assert_eq!(out[0].dims.as_slice(), &[4]);
    assert_eq!(out[0].dtype, DType::F32);
}

/// Build a sub-graph (the reducer) on the imperative stack *while* `main` is
/// still open — the path the StableHLO walker takes for inline regions.
#[test]
fn nested_region_reducer() {
    let mut m = ModuleBuilder::new();
    m.begin_func("main");
    let x = m.param(Shape::new(DType::F32, [8, 4]));
    let init = m.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());

    m.begin_func("add"); // nested: opened while `main` is still on the stack
    let a = m.param(Shape::scalar(DType::F32));
    let b = m.param(Shape::scalar(DType::F32));
    let c = m.add(a, b);
    let reducer = m.end_func(vec![c]);

    let s = m.reduce(x, init, &[1], reducer); // back in `main`; f32[8]
    let main = m.end_func(vec![s]);

    let module = m.finish(main).unwrap();
    let ef = module.entry_func();
    assert_eq!(ef.value_shape(ef.results[0]).dims.as_slice(), &[8]);
}

/// A `compare` carrying an explicit `compare_type` prints it as an attr; the
/// default `None` omits it.
#[test]
fn compare_type_prints_when_present() {
    let inputs = [Shape::new(DType::I32, [4]), Shape::new(DType::I32, [4])];
    let with = Compare {
        comparison_direction: ComparisonDirection::Lt,
        compare_type: Some(ComparisonType::Signed),
    };
    assert_eq!(
        with.attrs(),
        vec![
            ("comparison_direction", "lt".to_string()),
            ("compare_type", "signed".to_string()),
        ]
    );
    assert!(
        with.infer(&InferCtx::new(
            &inputs,
            &[],
            &domain_index::FiniteMapGraph::default(),
        ))
        .is_ok()
    );

    let without = Compare {
        comparison_direction: ComparisonDirection::Lt,
        compare_type: None,
    };
    assert_eq!(
        without.attrs(),
        vec![("comparison_direction", "lt".to_string())]
    );
}

/// A `gather` carries `indices_are_sorted` and the StableHLO batching dims.
#[test]
fn gather_indices_are_sorted_and_batching_dims_print() {
    let dnums = GatherDimensionNumbers {
        offset_dims: smallvec![1],
        collapsed_slice_dims: SmallVec::new(),
        operand_batching_dims: smallvec![0],
        start_indices_batching_dims: smallvec![0],
        start_index_map: smallvec![1],
        index_vector_dim: 1,
    };
    let g = Gather {
        dnums,
        slice_sizes: smallvec![1, 4],
        indices_are_sorted: true,
    };
    let a = g.attrs();
    assert!(a.contains(&("indices_are_sorted", "true".to_string())));
    assert!(a.contains(&("operand_batching_dims", "[0]".to_string())));
    assert!(a.contains(&("start_indices_batching_dims", "[0]".to_string())));
}

/// `convolution`/`reduce_window` carry StableHLO's single Nx2 `padding`; it
/// prints as `[[lo, hi], ...]`.
#[test]
fn padding_prints_as_pairs() {
    let conv = Convolution {
        dnums: Default::default(),
        window_strides: smallvec![1],
        padding: smallvec![[1, 2]],
        lhs_dilation: smallvec![1],
        rhs_dilation: smallvec![1],
        window_reversal: SmallVec::new(),
        feature_group_count: 1,
        batch_group_count: 1,
    };
    assert!(conv.attrs().contains(&("padding", "[[1, 2]]".to_string())));

    let rw = ReduceWindow {
        window_dimensions: smallvec![2, 2],
        window_strides: smallvec![1, 1],
        base_dilations: smallvec![1, 1],
        window_dilations: smallvec![1, 1],
        padding: smallvec![[0, 0], [1, 1]],
        reducer: FuncId(0),
    };
    assert!(
        rw.attrs()
            .contains(&("padding", "[[0, 0], [1, 1]]".to_string()))
    );
}

/// The verifier rejects a module that references an out-of-range value.
#[test]
fn verify_rejects_dangling_value() {
    let bad = Func {
        name: "bad".to_string(),
        params: Vec::new(),
        nodes: Vec::new(),
        results: vec![ValueId(0)],
        values: Vec::new(),
    };
    let module = Module {
        finite_map_graph: domain_index::FiniteMapGraph::default(),
        funcs: vec![bad],
        entry: FuncId(0),
    };
    assert!(verify(&module).is_err());
}
