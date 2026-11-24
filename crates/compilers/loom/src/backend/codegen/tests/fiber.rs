// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn fiber_lanes_split_small_hidden_axes_and_the_fiber() {
    let mut fiber = fiber_multi_warp_program();
    fiber.lane_count = 3;
    fiber.operands[0].lane_extent = 3;
    fiber.operands[1].lane_extent = 3;
    let prepared = Plan::prepare(&fiber).unwrap();
    let parts = plan(&fiber, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_consumer_kernel_flavor(
        &prepared,
        &fiber,
        &parts[0],
        Flavor {
            materialized: &[],
            fiber_lanes: true,
            ..Flavor::plain(128, 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    assert!(has_for_with_step(kernel.body(), 4));
    assert!(has_for_with_step(kernel.body(), 8));
    assert!(kernel.function.shared_arrays().is_empty());
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains(".entry loom_multi_warp_c_bs128_fl8"), "{ptx}");
    assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 8, "{ptx}");
    assert!(!ptx.contains("bar.sync"), "{ptx}");
}

#[test]
fn unit_hidden_axis_uses_only_width_one_accumulators() {
    let mut fiber = fiber_multi_warp_program();
    fiber.lane_count = 1;
    for binding in &mut fiber.operands {
        binding.lane_extent = 1;
    }
    let prepared = Plan::prepare(&fiber).unwrap();
    let parts = plan(&fiber, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_vectorized_kernel(&prepared, &fiber, 32, 0, 1, &parts[0], &[], 1).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(!ptx.contains("loom_fiber_acc"), "{ptx}");
    assert_eq!(
        ptx.matches("st.global.L1::evict_first.f32").count(),
        2,
        "{ptx}"
    );
}

#[test]
fn four_warps_share_one_position_and_fold_width_one_outputs() {
    let program = multi_warp_program();
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &program, 128, 0, 1, &parts[0], &[], 4).unwrap();
    assert_eq!(kernel.max_threads, 128);
    assert_eq!(kernel.function.shared_arrays().len(), 1);
    assert_eq!(kernel.function.shared_arrays()[0].elem, Scalar::F32);
    assert_eq!(kernel.function.shared_arrays()[0].len, 4);
    let body = kernel.body();
    assert!(steps_of(body).iter().any(|&op| {
        matches!(body.op(op).kind, gpu_ir::OpKind::Bind)
            && body
                .operand_op(body.op(op).operands[0])
                .is_some_and(|cast| {
                    matches!(body.op(cast).kind, gpu_ir::OpKind::Cast(Scalar::S64))
                        && body
                            .operand_op(body.op(cast).operands[0])
                            .is_some_and(|held| {
                                matches!(
                                    body.op(held).kind,
                                    gpu_ir::OpKind::Sreg(gpu_ir::Sreg::CtaidX)
                                )
                            })
                })
    }));
    // The position sweep starts at a value the prologue bound and strides the
    // block.
    assert!(steps_of(body).iter().any(|&op| {
        matches!(body.op(op).kind, gpu_ir::OpKind::Loop { .. })
            && body.is_materialized(body.op(op).operands[0])
            && body
                .operand_op(body.op(op).operands[2])
                .is_some_and(|step| {
                    matches!(
                        body.op(step).kind,
                        gpu_ir::OpKind::IntConst {
                            scalar: Scalar::S32,
                            value: 128
                        }
                    )
                })
    }));
    let narrow_output = &prepared
        .layout
        .param_name(prepared.layout.operand_params[2]);
    let narrow_output_base = "loom_output_base_1";
    let f = kernel.body();
    // The narrow output is reached through a base this arm binds off the
    // parameter, and written through that base.
    let stores_to = |ops: &[gpu_ir::OpId]| {
        let base = ops.iter().find(|&&op| {
            f.op(op).results.first().and_then(|&result| f.name(result)) == Some(narrow_output_base)
                && f.operand_op(f.op(op).operands[0])
                    .is_some_and(|held| matches!(f.op(held).kind, gpu_ir::OpKind::Offset))
                && f.reads(f.op(op).operands[0])
                    .into_iter()
                    .any(|read| f.name(read) == Some(&**narrow_output))
        });
        let Some(&base) = base else {
            return false;
        };
        let based = f.op(base).results[0];
        ops.iter().any(|&op| {
            matches!(f.op(op).kind, gpu_ir::OpKind::Write { .. })
                && f.reads(f.op(op).operands[0]).contains(&based)
        })
    };
    assert!(steps_of(f).iter().any(|&op| {
        if !matches!(f.op(op).kind, gpu_ir::OpKind::If) {
            return false;
        }
        let Some(compare) = f.operand_op(f.op(op).operands[0]) else {
            return false;
        };
        if !matches!(f.op(compare).kind, gpu_ir::OpKind::Cmp(gpu_ir::CmpOp::Eq)) {
            return false;
        }
        let against = f.op(compare).operands[1];
        let zero = f.operand_op(against).is_some_and(|held| {
            matches!(
                f.op(held).kind,
                gpu_ir::OpKind::IntConst {
                    scalar: Scalar::S32,
                    value: 0
                }
            )
        });
        zero && stores_to(&f.block(f.region_block(f.op(op).regions[0])).ops)
    }));
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains(".entry loom_multi_warp_c_bs128_wp4"), "{ptx}");
    assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 5, "{ptx}");
    assert_eq!(ptx.matches("st.shared.f32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("bar.sync 0;").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("ld.shared.f32").count(), 1, "{ptx}");
    assert_eq!(
        ptx.matches("st.global.L1::evict_first.f32").count(),
        2,
        "{ptx}"
    );
}

#[test]
fn four_warp_positions_reject_other_shapes_and_plan_features() {
    let program = multi_warp_program();
    let prepared = Plan::prepare(&program).unwrap();
    let mut parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_multi_warp_kernel(&prepared, &program, 32, 0, 1, &parts[0], &[], 4).is_err());
    assert!(emit_multi_warp_kernel(&prepared, &program, 32, 0, 1, &parts[0], &[], 2).is_err());
    assert!(emit_multi_warp_kernel(&prepared, &program, 128, 0, 1, &parts[0], &[], 3).is_err());

    let mut too_narrow = multi_warp_program();
    too_narrow.lane_count = 64;
    too_narrow.operands[0].lane_extent = 64;
    too_narrow.operands[1].lane_extent = 64;
    let narrow_plan = Plan::prepare(&too_narrow).unwrap();
    let narrow_parts = plan(&too_narrow, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_multi_warp_kernel(
            &narrow_plan,
            &too_narrow,
            128,
            0,
            1,
            &narrow_parts[0],
            &[],
            4,
        )
        .is_err()
    );

    let mut no_wide = multi_warp_program();
    for binding in &mut no_wide.operands {
        binding.lane_extent = 1;
    }
    let no_wide_plan = Plan::prepare(&no_wide).unwrap();
    let no_wide_parts = plan(&no_wide, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_multi_warp_kernel(
            &no_wide_plan,
            &no_wide,
            128,
            0,
            1,
            &no_wide_parts[0],
            &[],
            4,
        )
        .is_err()
    );

    let mut bf16_wide = multi_warp_program();
    bf16_wide.operands[0].dtype = Dtype::Bf16;
    let bf16_plan = Plan::prepare(&bf16_wide).unwrap();
    let bf16_parts = plan(&bf16_wide, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_multi_warp_kernel(&bf16_plan, &bf16_wide, 128, 0, 1, &bf16_parts[0], &[], 4,).is_err()
    );

    let materialized = MaterializedValue {
        name: "x".into(),
        scalar: Scalar::F32,
        nbytes: 4,
        closure_statements: 1,
        consumer_count: 1,
    };
    assert!(
        emit_multi_warp_kernel(
            &prepared,
            &program,
            128,
            0,
            1,
            &parts[0],
            &[materialized],
            4,
        )
        .is_err()
    );

    parts[0].elements[0][0].continuation = true;
    assert!(emit_multi_warp_kernel(&prepared, &program, 128, 0, 1, &parts[0], &[], 4).is_err());

    // A fiber walk splits across warps only when every warp runs the same
    // u passes: the cross-warp combine barriers inside the u loop.
    let mut ragged = fiber_multi_warp_program();
    ragged.lane_count = 48;
    for binding in &mut ragged.operands {
        if binding.lane_extent != 1 {
            binding.lane_extent = 48;
        }
    }
    let ragged_plan = Plan::prepare(&ragged).unwrap();
    assert!(ragged_plan.index.fiber_bridge().is_some());
    let ragged_parts = plan(&ragged, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_multi_warp_kernel(&ragged_plan, &ragged, 128, 0, 1, &ragged_parts[0], &[], 4,)
            .is_err()
    );
}

#[test]
fn fiber_split_strides_the_walk_and_folds_wide_partials() {
    let fiber = fiber_multi_warp_program();
    let prepared = Plan::prepare(&fiber).unwrap();
    assert!(prepared.index.fiber_bridge().is_some());
    let parts = plan(&fiber, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &fiber, 128, 0, 1, &parts[0], &[], 4).unwrap();
    // Each warp covers the full hidden axis (step 32, not 128) and walks
    // fiber elements k = lo + warp, stepping by the warp count.
    assert!(has_for_with_step(kernel.body(), 32));
    assert!(!has_for_with_step(kernel.body(), 128));
    assert!(has_for_with_step(kernel.body(), 4));
    // One width-1 partial per warp plus one wide partial per non-warp-0
    // warp lane.
    assert_eq!(kernel.function.shared_arrays().len(), 2);
    assert_eq!(kernel.function.shared_arrays()[0].len, 4);
    assert_eq!(kernel.function.shared_arrays()[1].len, 3 * 32);
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains(".entry loom_multi_warp_c_bs128_wp4"), "{ptx}");
    // Deposit barrier + buffer-reuse barrier inside the u loop, one more
    // before the width-1 combine.
    assert_eq!(ptx.matches("bar.sync 0;").count(), 3, "{ptx}");
    // Wide deposit + width-1 deposit; three unrolled wide folds and the
    // width-1 combine loop.
    assert_eq!(ptx.matches("st.shared.f32").count(), 2, "{ptx}");
    assert_eq!(ptx.matches("ld.shared.f32").count(), 4, "{ptx}");
    assert_eq!(
        ptx.matches("st.global.L1::evict_first.f32").count(),
        2,
        "{ptx}"
    );
}

/// The u-split arrangement: each warp walks the whole fiber (step 1)
/// over its own hidden-axis slice (u step = threads per position), so
/// wide outputs need no cross-warp fold at all and only the width-1
/// scalar partials touch shared memory.
#[test]
fn warp_u_split_walks_the_whole_fiber_per_u_slice() {
    let fiber = fiber_multi_warp_program();
    let prepared = Plan::prepare(&fiber).unwrap();
    assert!(prepared.index.fiber_bridge().is_some());
    let parts = plan(&fiber, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_warp_u_kernel(&prepared, &fiber, 128, &parts[0], 4).unwrap();
    assert!(has_for_with_step(kernel.body(), 128));
    assert!(!has_for_with_step(kernel.body(), 32));
    assert!(!has_for_with_step(kernel.body(), 4));
    assert_eq!(kernel.function.shared_arrays().len(), 1);
    assert_eq!(kernel.function.shared_arrays()[0].len, 4);
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains(".entry loom_multi_warp_c_bs128_wu4"), "{ptx}");

    // Without a fiber bridge the plain cross already splits the hidden
    // axis: the arrangement would be a duplicate.
    let plainer = multi_warp_program();
    let prepared = Plan::prepare(&plainer).unwrap();
    let parts = plan(&plainer, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_warp_u_kernel(&prepared, &plainer, 128, &parts[0], 4).is_err());
}

/// Width-1 fiber-direct outputs decline the u-split: every warp's
/// per-edge tree would yield a partial of the same finer-element scalar.
#[test]
fn warp_u_split_declines_width_one_scatter_writes() {
    let program = sender_scatter_program(64, 64);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_warp_u_kernel(&prepared, &program, 64, &parts[0], 2).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(
        ptx.contains(".entry loom_sender_scatter_c_bs64_wu2"),
        "{ptx}"
    );
    assert!(!ptx.contains("red.global"), "{ptx}");

    let program = sender_scatter_program(64, 1);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_warp_u_kernel(&prepared, &program, 64, &parts[0], 2).is_err());
}

#[test]
fn fiber_split_keeps_scatter_writes_exact() {
    // Wide fiber-scatter outputs: each edge is visited by exactly one
    // warp, so the in-loop stores stay plain.
    let program = sender_scatter_program(64, 64);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    assert!(prepared.index.fiber_scatter(2));
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &program, 64, 0, 1, &parts[0], &[], 2).unwrap();
    assert!(has_for_with_step(kernel.body(), 2));
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(
        ptx.contains(".entry loom_sender_scatter_c_bs64_wp2"),
        "{ptx}"
    );
    assert!(!ptx.contains("red.global"), "{ptx}");

    // Width-1 fiber-scatter outputs keep the per-edge warp tree: the
    // owning warp runs every u pass, so lane 0's first-pass store and
    // later-pass accumulate are untouched by the split.
    let program = sender_scatter_program(64, 1);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &program, 64, 0, 1, &parts[0], &[], 2).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 5, "{ptx}");
    assert!(!ptx.contains("red.global"), "{ptx}");

    // The walk is not splittable when u passes diverge across lanes.
    let program = sender_scatter_program(48, 48);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_kernel(&prepared, &program, 64, 0, 1, &parts[0]).is_ok());
    assert!(emit_multi_warp_kernel(&prepared, &program, 64, 0, 1, &parts[0], &[], 2).is_err());
}
