// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn emit_test_kernel(program: &LiftedProgram) -> Launch {
    let prepared = Plan::prepare(program).unwrap();
    let parts = plan(program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert_eq!(parts.len(), 1);
    emit_kernel(&prepared, program, BLOCK_SIZES[0], 0, 1, &parts[0]).unwrap()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_vectorized_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part_index: usize,
    num_parts: usize,
    part: &KernelPart,
    materialized: &[MaterializedValue],
    io_lanes: u8,
) -> Result<Launch, String> {
    emit_consumer_kernel_flavor(
        program,
        loom,
        part,
        Flavor {
            materialized,
            io_lanes,
            warps_per_position: 1,
            warp_u: false,
            ..Flavor::plain(block_size, part_index, num_parts)
        },
    )
    .map(|consumer| consumer.kernel)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_multi_warp_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part_index: usize,
    num_parts: usize,
    part: &KernelPart,
    materialized: &[MaterializedValue],
    warps_per_position: u8,
) -> Result<Launch, String> {
    emit_consumer_kernel_flavor(
        program,
        loom,
        part,
        Flavor {
            materialized,
            io_lanes: 1,
            warps_per_position,
            warp_u: false,
            ..Flavor::plain(block_size, part_index, num_parts)
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn emit_warp_u_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part: &KernelPart,
    warps_per_position: u8,
) -> Result<Launch, String> {
    emit_consumer_kernel_flavor(
        program,
        loom,
        part,
        Flavor {
            materialized: &[],
            io_lanes: 1,
            warps_per_position,
            warp_u: true,
            ..Flavor::plain(block_size, 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
}

pub(super) fn program(
    dtypes: &[Dtype],
    domains: Vec<Domain>,
    index_maps: Vec<IndexMap>,
) -> LiftedProgram {
    program_of_dtype(dtypes, DType::F32, domains, index_maps)
}

/// `program`, with the XIR dtype the module computes and returns stated: an
/// output stores what the program computed, so a fixture whose bindings are
/// narrower than `xir_dtype` accumulates in the wider one and converts after.
pub(super) fn program_of_dtype(
    dtypes: &[Dtype],
    xir_dtype: DType,
    domains: Vec<Domain>,
    index_maps: Vec<IndexMap>,
) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(xir_dtype, []));
    let main = builder.end_func(vec![x]);
    let module = builder.finish(main).unwrap();
    let operands = dtypes
        .iter()
        .map(|&dtype| FlatOperand {
            dtype,
            element_shape: Vec::new(),
            lane_extent: 8,
            axis_domains: vec![0],
        })
        .collect::<Vec<_>>();
    loom_program! {
        name: "backend_contract".into(),
        module,
        lane_count: 8,
        domains,
        index_maps,
        num_axes: 1,
        operands,
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

/// Edges refine into senders (total) and receivers; the edge output is
/// written fiber-direct while iterating the sender domain, the receiver
/// input bridges through the aligned view.
pub(super) fn sender_scatter_program(u: i64, edge_lane: i64) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let edge = builder.param(Shape::new(DType::F32, []));
    let receiver = builder.param(Shape::new(DType::F32, []));
    let product = builder.multiply(edge, receiver);
    let main = builder.end_func(vec![product, receiver]);
    let module = builder.finish(main).unwrap();
    let binding = |domain, lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![domain],
    };
    let index_map = |target| IndexMap {
        source_id: 0,
        target_id: target,
        is_injective: false,
        is_sorted: false,
        may_have_unmapped: false,
    };
    loom_program! {
        name: "sender_scatter".into(),
        module,
        lane_count: u,
        domains: vec![
            Domain {
                extent: 12,
            },
            Domain {
                extent: 3,
            },
            Domain {
                extent: 3,
            },
        ],
        index_maps: vec![index_map(1), index_map(2)],
        num_axes: 1,
        operands: vec![
            binding(0, u),
            binding(2, u),
            binding(0, edge_lane),
            binding(1, u),
        ],
        num_inputs: 2,
        available_views: vec![AlignedView {
            source_id: 0,
            anchor_target_id: 1,
            other_target_id: 2,
        }],
    }
}

pub(super) fn multi_warp_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let main = builder.end_func(vec![x, x]);
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "multi_warp".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 256,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(256), binding(256), binding(1)],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

/// The input sits on the target domain and is reached by gathering through a
/// index map that leaves cells unmapped, so addressing it needs a liveness
/// predicate: this is the shape that makes an emitter bind `loom_input_alive_*`.
pub(super) fn gathered_input_program() -> LiftedProgram {
    let mut program = multi_warp_program();
    program.domains = vec![Domain { extent: 12 }, Domain { extent: 3 }];
    program.index_maps = vec![IndexMap {
        source_id: 0,
        target_id: 1,
        is_injective: false,
        is_sorted: false,
        may_have_unmapped: true,
    }];
    program.operands[0].axis_domains[0] = 1;
    program
}

pub(super) fn fiber_multi_warp_program() -> LiftedProgram {
    let mut fiber = multi_warp_program();
    fiber.domains = vec![Domain { extent: 12 }, Domain { extent: 3 }];
    fiber.index_maps = vec![IndexMap {
        source_id: 0,
        target_id: 1,
        is_injective: false,
        is_sorted: false,
        may_have_unmapped: false,
    }];
    fiber.operands[1].axis_domains[0] = 1;
    fiber.operands[2].axis_domains[0] = 1;
    fiber
}

/// Whether any loop the body holds counts by `step`.
pub(super) fn has_for_with_step(f: &gpu_ir::Function, step: i64) -> bool {
    fn walk(f: &gpu_ir::Function, ops: &[gpu_ir::OpId], step: i64) -> bool {
        ops.iter().any(|&op| {
            let counts_by = matches!(f.op(op).kind, gpu_ir::OpKind::Loop { .. })
                && f.operand_op(f.op(op).operands[2]).is_some_and(|held| {
                    matches!(
                        f.op(held).kind,
                        gpu_ir::OpKind::IntConst { value, .. } if value == step
                    )
                });
            counts_by
                || f.op(op).regions.iter().any(|&region| {
                    f.region(region)
                        .blocks
                        .iter()
                        .any(|&block| walk(f, &f.block(block).ops, step))
                })
        })
    }
    walk(f, &f.block(f.entry()).ops, step)
}

/// The steps a kernel body runs, for a test that asks about its opening or its
/// length.
pub(super) fn steps_of(f: &gpu_ir::Function) -> Vec<gpu_ir::OpId> {
    f.block_steps(f.entry()).collect()
}

/// What one step is, for a test that asserts the shape of a body rather than
/// the identity of its values.
pub(super) fn shape_of(f: &gpu_ir::Function, op: gpu_ir::OpId) -> String {
    match &f.op(op).kind {
        gpu_ir::OpKind::Bind => match f.name(f.op(op).results[0]) {
            Some(name) => format!("bind {name}"),
            None => "bind".to_string(),
        },
        kind => format!("{kind:?}"),
    }
}

/// The body as a sequence of step shapes, which is what a prologue comparison
/// is about: two groups open the same way when they run the same steps in the
/// same order, whatever values they went on to define.
pub(super) fn shapes_of(f: &gpu_ir::Function, ops: &[gpu_ir::OpId]) -> Vec<String> {
    ops.iter().map(|&op| shape_of(f, op)).collect()
}

/// Every step a list of them holds, however deeply.
pub(super) fn nested(f: &gpu_ir::Function, ops: &[gpu_ir::OpId]) -> Vec<gpu_ir::OpId> {
    let mut out = Vec::new();
    for &op in ops {
        out.push(op);
        for &region in &f.op(op).regions {
            for &block in &f.region(region).blocks {
                out.extend(nested(f, &f.block(block).ops));
            }
        }
    }
    out
}
