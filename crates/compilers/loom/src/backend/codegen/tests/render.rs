// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn csr_fiber_kernel_renders_as_ptx() {
    let mut program = program(
        &[Dtype::F32, Dtype::F32],
        vec![Domain { extent: 12 }, Domain { extent: 3 }],
        vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
    );
    program.operands[0].axis_domains[0] = 0;
    program.operands[1].axis_domains[0] = 1;
    let kernel = emit_test_kernel(&program);
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("ld.global.b32"));
    assert!(ptx.contains("st.global"));
}

#[test]
fn elementwise_visible_and_hidden_reductions_render_as_ptx() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [3]));
    let two = builder.constant(Literal::new(
        Shape::scalar(DType::F32),
        2f32.to_le_bytes().to_vec(),
    ));
    let two = builder.broadcast_in_dim(two, &[3], &[]);
    let y = builder.multiply(x, two);
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let reducer = builder.func("add", |function| {
        let lhs = function.param(Shape::scalar(DType::F32));
        let rhs = function.param(Shape::scalar(DType::F32));
        vec![function.add(lhs, rhs)]
    });
    let sum = builder.op1(
        Reduce {
            dimensions: [0].into_iter().collect(),
            reducer,
        },
        &[y, zero],
    );
    let main = builder.end_func(vec![y, sum]);
    let module = builder.finish(main).unwrap();
    let binding = |shape: &[i64], lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: shape.to_vec(),
        lane_extent,
        axis_domains: vec![0],
    };
    let program = loom_program! {
        name: "typed_eval".into(),
        module,
        lane_count: 8,
        domains: vec![Domain {
            extent: 4,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(&[3], 8), binding(&[3], 8), binding(&[], 1)],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let kernel = emit_test_kernel(&program);
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("mul.rn.f32"));
    assert!(ptx.contains("add.rn.f32"));
}
