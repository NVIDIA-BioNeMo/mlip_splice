// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn vectorized_half_io_renders_vec4_and_vec8_ptx() {
    for (lanes, load, store) in [
        (4, "ld.global.nc.v2.b32", "st.global.L1::evict_first.v2.b32"),
        (8, "ld.global.nc.v4.b32", "st.global.L1::evict_first.v4.b32"),
    ] {
        let mut program = program_of_dtype(
            &[Dtype::Bf16, Dtype::Bf16],
            DType::Bf16,
            vec![Domain { extent: 4 }],
            Vec::new(),
        );
        program.lane_count = 256;
        for binding in &mut program.operands {
            binding.lane_extent = 256;
        }
        let prepared = Plan::prepare(&program).unwrap();
        let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
        let kernel =
            emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], lanes)
                .unwrap();
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
        assert!(ptx.contains(&format!("_bs32_v{lanes}")), "{ptx}");
        assert!(ptx.contains(load), "{ptx}");
        assert!(ptx.contains(store), "{ptx}");
        assert!(
            emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], 2,)
                .is_err()
        );
    }
}

#[test]
fn vectorized_f32_io_spans_lane_multiples() {
    let make = |u| {
        let mut program = program(
            &[Dtype::F32, Dtype::F32],
            vec![Domain { extent: 4 }],
            Vec::new(),
        );
        program.lane_count = u;
        for binding in &mut program.operands {
            binding.lane_extent = u;
        }
        program
    };

    for u in [64, 128, 256] {
        let aligned = make(u);
        let prepared = Plan::prepare(&aligned).unwrap();
        let parts = plan(&aligned, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
        let kernel =
            emit_vectorized_kernel(&prepared, &aligned, WARP_SIZE, 0, 1, &parts[0], &[], 2)
                .unwrap();
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
        assert!(ptx.contains("_bs32_v2"), "{ptx}");
        assert!(ptx.contains("ld.global.nc.v2.f32"), "{ptx}");
        assert!(ptx.contains("st.global.L1::evict_first.v2.f32"), "{ptx}");
    }

    let wide = make(256);
    let prepared = Plan::prepare(&wide).unwrap();
    let parts = plan(&wide, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel =
        emit_vectorized_kernel(&prepared, &wide, WARP_SIZE, 0, 1, &parts[0], &[], 4).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains("_bs32_v4"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.v4.f32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.v4.f32"), "{ptx}");

    let unaligned = make(96);
    let prepared = Plan::prepare(&unaligned).unwrap();
    let parts = plan(&unaligned, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_vectorized_kernel(&prepared, &unaligned, WARP_SIZE, 0, 1, &parts[0], &[], 2,).is_err()
    );
    let quarter = make(192);
    let prepared = Plan::prepare(&quarter).unwrap();
    let parts = plan(&quarter, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_vectorized_kernel(&prepared, &quarter, WARP_SIZE, 0, 1, &parts[0], &[], 4,).is_err()
    );
}
#[test]
fn multi_warp_blocks_tile_positions_and_cross_vector_lanes() {
    let program = multi_warp_program();
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();

    let tiled = emit_multi_warp_kernel(&prepared, &program, 128, 0, 1, &parts[0], &[], 2).unwrap();
    assert_eq!(tiled.max_threads, 128);
    assert_eq!(tiled.function.shared_arrays().len(), 1);
    assert_eq!(tiled.function.shared_arrays()[0].len, 4);
    let ptx = gpu_ir::ptx::render_ptx_module(&[tiled], 120);
    assert!(ptx.contains(".entry loom_multi_warp_c_bs128_wp2"), "{ptx}");
    assert_eq!(ptx.matches("bar.sync 0;").count(), 1, "{ptx}");

    let crossed = emit_consumer_kernel_flavor(
        &prepared,
        &program,
        &parts[0],
        Flavor {
            materialized: &[],
            io_lanes: 2,
            warps_per_position: 4,
            warp_u: false,
            ..Flavor::plain(128, 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    assert!(has_for_with_step(crossed.body(), 256));
    let ptx = gpu_ir::ptx::render_ptx_module(&[crossed], 120);
    assert!(
        ptx.contains(".entry loom_multi_warp_c_bs128_v2_wp4"),
        "{ptx}"
    );
    assert!(ptx.contains("ld.global.nc.v2.f32"), "{ptx}");
}
#[test]
fn vectorized_io_rides_fiber_iteration() {
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
    let prepared = Plan::prepare(&fiber).unwrap();
    assert!(prepared.index.fiber_bridge().is_some());
    let parts = plan(&fiber, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel =
        emit_vectorized_kernel(&prepared, &fiber, WARP_SIZE, 0, 1, &parts[0], &[], 2).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains("_bs32_v2"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.v2.f32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.v2.f32"), "{ptx}");
}

/// A width-1 output folds each thread's lanes to one scalar, which the vector
/// path still reaches: what a vec2 needs is a wide f32 operand, not a wide
/// output.
#[test]
fn f32_vec2_admits_a_width_one_output() {
    let mut wide = program(
        &[Dtype::F32, Dtype::F32],
        vec![Domain { extent: 4 }],
        Vec::new(),
    );
    wide.lane_count = 128;
    for binding in &mut wide.operands {
        binding.lane_extent = 128;
    }
    let prepared = Plan::prepare(&wide).unwrap();
    let parts = plan(&wide, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_vectorized_kernel(&prepared, &wide, WARP_SIZE, 0, 1, &parts[0], &[], 2,).is_ok());

    wide.operands[1].lane_extent = 1;
    let prepared = Plan::prepare(&wide).unwrap();
    let parts = plan(&wide, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_vectorized_kernel(&prepared, &wide, WARP_SIZE, 0, 1, &parts[0], &[], 2,).is_ok());
}

#[test]
fn vectorized_half_input_folds_hidden_reduction_in_f32() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::Bf16));
    let x = builder.op1(Convert { dtype: DType::F32 }, &[x]);
    let two = builder.constant(Literal::new(
        Shape::scalar(DType::F32),
        2f32.to_le_bytes().to_vec(),
    ));
    let y = builder.multiply(x, two);
    let y = builder.op1(Convert { dtype: DType::Bf16 }, &[y]);
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let program = loom_program! {
        name: "vector_reduction".into(),
        module,
        lane_count: 256,
        domains: vec![Domain {
            extent: 4,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            FlatOperand {
                dtype: Dtype::Bf16,
                element_shape: Vec::new(),
                lane_extent: 256,
                axis_domains: vec![0],
            },
            FlatOperand {
                dtype: Dtype::Bf16,
                element_shape: Vec::new(),
                lane_extent: 1,
                axis_domains: vec![0],
            },
        ],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel =
        emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], 8).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains("ld.global.nc.v4.b32"), "{ptx}");
    assert_eq!(ptx.matches("cvt.f32.bf16").count(), 16, "{ptx}");
    assert!(ptx.contains("mul.rn.f32x2"), "{ptx}");
    assert!(ptx.contains("shfl.sync.down.b32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.b16"), "{ptx}");
}

#[test]
fn vectorized_self_division_is_preserved_for_wide_and_folded_outputs() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let input = builder.param(Shape::scalar(DType::Bf16));
    let ratio = builder.binary("divide", input, input).unwrap();
    let main = builder.end_func(vec![ratio, ratio]);
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::Bf16,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![0],
    };
    let program = loom_program! {
        name: "vector_self_division".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 256,
        domains: vec![Domain {
            extent: 4,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(256), binding(256), binding(1)],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel =
        emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], 8).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains("st.global.L1::evict_first.v4.b32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.b16"), "{ptx}");
    assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 5, "{ptx}");
    assert!(ptx.contains("ld.global.nc.v4.b32"), "{ptx}");
    assert!(ptx.contains("div.rn.f32"), "{ptx}");
}

#[test]
fn vectorized_f16_io_handles_visible_segments_and_width_one_input() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let narrow = builder.param(Shape::new(DType::F16, [3]));
    let wide = builder.param(Shape::new(DType::F16, [3]));
    let narrow = builder.op1(Convert { dtype: DType::F32 }, &[narrow]);
    let wide = builder.op1(Convert { dtype: DType::F32 }, &[wide]);
    let output = builder.add(narrow, wide);
    let output = builder.op1(Convert { dtype: DType::F16 }, &[output]);
    let main = builder.end_func(vec![output]);
    let module = builder.finish(main).unwrap();
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::F16,
        element_shape: vec![3],
        lane_extent,
        axis_domains: vec![0],
    };
    let program = loom_program! {
        name: "vector_visible_segments".into(),
        module,
        lane_count: 256,
        domains: vec![Domain {
            extent: 4,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(1), binding(256), binding(256)],
        num_inputs: 2,
        available_views: Vec::new(),
    };
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    for (lanes, load, store) in [
        (4, "ld.global.nc.v2.b32", "st.global.L1::evict_first.v2.b32"),
        (8, "ld.global.nc.v4.b32", "st.global.L1::evict_first.v4.b32"),
    ] {
        let kernel =
            emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], lanes)
                .unwrap();
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
        assert!(ptx.contains(load), "{ptx}");
        assert!(ptx.contains(store), "{ptx}");
        assert!(ptx.contains("ld.global.nc.b16"), "{ptx}");
        assert!(ptx.contains("cvt.f32.f16"), "{ptx}");
        assert!(ptx.contains("cvt.rn.f16.f32"), "{ptx}");
        assert_eq!(ptx.matches(store).count(), 3, "{ptx}");
    }
}

/// A continuation reads its partial back to add to it, so the read-back is as
/// wide as the lanes the thread carries; a scalar read-back would not even
/// type-check against the vector it accumulates.
#[test]
fn a_vectorized_continuation_reads_its_partial_back_at_the_lane_width() {
    let mut direct = program_of_dtype(
        &[Dtype::Bf16, Dtype::Bf16],
        DType::Bf16,
        vec![Domain { extent: 4 }],
        Vec::new(),
    );
    direct.lane_count = 256;
    for binding in &mut direct.operands {
        binding.lane_extent = 256;
    }
    let prepared = Plan::prepare(&direct).unwrap();
    let mut parts = plan(&direct, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    parts[0].elements[0][0].continuation = true;
    let kernel =
        emit_vectorized_kernel(&prepared, &direct, WARP_SIZE, 0, 1, &parts[0], &[], 8).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("ld.global.v4.b32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.v4.b32"), "{ptx}");
}

#[test]
fn vectorized_io_rejects_shift_ops() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let value = builder.param(Shape::scalar(DType::U32));
    let amount = builder.param(Shape::scalar(DType::U32));
    let shifted = builder.binary("shift_left", value, amount).unwrap();
    let main = builder.end_func(vec![shifted]);
    let binding = FlatOperand {
        dtype: Dtype::Bf16,
        element_shape: Vec::new(),
        lane_extent: 256,
        axis_domains: vec![0],
    };
    let shifted = loom_program! {
        name: "vector_shift".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 256,
        domains: vec![Domain {
            extent: 4,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding.clone(), binding.clone(), binding],
        num_inputs: 2,
        available_views: Vec::new(),
    };
    let prepared = Plan::prepare(&shifted).unwrap();
    let parts = plan(&shifted, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(
        emit_vectorized_kernel(&prepared, &shifted, WARP_SIZE, 0, 1, &parts[0], &[], 8,)
            .unwrap_err()
            .contains("not eligible")
    );
}

#[test]
fn vectorized_map_lookup_uses_vector_numeric_zero_for_unmapped_inputs() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let input = builder.param(Shape::scalar(DType::Bf16));
    let main = builder.end_func(vec![input]);
    let binding = |domain| FlatOperand {
        dtype: Dtype::Bf16,
        element_shape: Vec::new(),
        lane_extent: 256,
        axis_domains: vec![domain],
    };
    let program = loom_program! {
        name: "vector_map_lookup".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 256,
        domains: vec![
            Domain {
                extent: 12,
            },
            Domain {
                extent: 3,
            },
        ],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: true,
        }],
        num_axes: 1,
        operands: vec![binding(1), binding(0)],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel =
        emit_vectorized_kernel(&prepared, &program, WARP_SIZE, 0, 1, &parts[0], &[], 8).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    assert!(ptx.contains("ld.global.nc.v4.b32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.v4.b32"), "{ptx}");
    assert!(ptx.contains("selp"), "{ptx}");
}
