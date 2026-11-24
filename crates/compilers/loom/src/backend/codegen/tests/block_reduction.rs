// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn atomic_reduce_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let main = builder.end_func(vec![x]);
    let binding = |domain| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent: 1,
        axis_domains: vec![domain],
    };
    loom_program! {
        name: "atomic_reduce".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 1,
        domains: vec![
            Domain {
                extent: 65,
            },
            Domain {
                extent: 1,
            },
        ],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![binding(0), binding(1)],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

#[test]
fn atomic_reduction_masks_unmapped_output_positions() {
    let mut loom = atomic_reduce_program();
    loom.index_maps[0].may_have_unmapped = true;
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let direct = emit_single_block_reduction_kernel(&prepared, &loom, 256, &parts[0]).unwrap();
    gpu_ir::verify_launch(&direct).unwrap();
}

#[test]
fn a_block_reduction_places_its_evaluation_once() {
    let loom = atomic_reduce_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let direct = emit_single_block_reduction_kernel(&prepared, &loom, 256, &parts[0]).unwrap();
    let f = direct.body();
    let mut steps = Vec::new();
    collect_steps(f, f.entry(), &mut steps);
    let mut once = std::collections::BTreeSet::new();
    for step in &steps {
        assert!(once.insert(step.index()), "a step stands in a block twice");
    }
    let ptx = gpu_ir::ptx::render_ptx_module(&[direct], 120);
    let entry = ptx
        .split(".visible .entry loom_atomic_reduce_br_bs256")
        .nth(1)
        .expect("the block reduction is entered");
    assert_eq!(entry.matches("mul.lo.s64").count(), 1, "{entry}");
    assert_eq!(entry.matches("ld.global.nc.f32").count(), 1, "{entry}");
}

/// Every step of a body, its regions included.
fn collect_steps(f: &gpu_ir::Function, block: gpu_ir::BlockId, into: &mut Vec<gpu_ir::OpId>) {
    for &op in &f.block(block).ops {
        into.push(op);
        for &region in &f.op(op).regions {
            for &inner in &f.region(region).blocks {
                collect_steps(f, inner, into);
            }
        }
    }
}
