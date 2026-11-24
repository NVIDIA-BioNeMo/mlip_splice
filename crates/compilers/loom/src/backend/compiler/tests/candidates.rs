// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(super) fn guarded_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let predicate = builder.param(Shape::scalar(DType::Bool));
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let mut expensive = builder.multiply(x, y);
    for _ in 0..8 {
        expensive = builder.multiply(expensive, y);
    }
    let selected = builder.op1(Select, &[predicate, expensive, zero]);
    let main = builder.end_func(vec![selected]);
    let binding = |dtype| FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "guarded_recipe".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::Bool),
            binding(Dtype::F32),
            binding(Dtype::F32),
            binding(Dtype::F32),
        ],
        num_inputs: 3,
        available_views: Vec::new(),
    }
}

fn indexed_program(guarded: bool) -> LiftedProgram {
    const SLOTS: i64 = 3;
    const UPDATES: i64 = 4;
    let mut builder = ModuleBuilder::new();
    builder.begin_func("add");
    let left = builder.param(Shape::scalar(DType::F32));
    let right = builder.param(Shape::scalar(DType::F32));
    let sum = builder.add(left, right);
    let reducer = builder.end_func(vec![sum]);
    builder.begin_func("main");
    let operand = builder.param(Shape::new(DType::F32, [SLOTS]));
    let indices = builder.constant(Literal::new(
        Shape::new(DType::I32, [UPDATES, 1]),
        [0i32, 1, 2, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
    ));
    let updates = builder.param(Shape::new(DType::F32, [UPDATES]));
    let predicate = guarded.then(|| builder.param(Shape::scalar(DType::Bool)));
    let scattered = builder.op1(
        xir::ops::Scatter {
            dnums: xir::dim_numbers::ScatterDimensionNumbers {
                update_window_dims: Default::default(),
                inserted_window_dims: [0].into_iter().collect(),
                input_batching_dims: Default::default(),
                scatter_indices_batching_dims: Default::default(),
                scatter_dims_to_operand_dims: [0].into_iter().collect(),
                index_vector_dim: 1,
            },
            indices_are_sorted: false,
            unique_indices: false,
            update: reducer,
        },
        &[operand, indices, updates],
    );
    let output = predicate.map_or(scattered, |predicate| {
        let zero = builder.constant(Literal::zeros(Shape::new(DType::F32, [SLOTS])).unwrap());
        builder.op1(Select, &[predicate, scattered, zero])
    });
    let main = builder.end_func(vec![output]);
    let binding = |dtype, visible_shape| FlatOperand {
        dtype,
        element_shape: visible_shape,
        lane_extent: 1,
        axis_domains: vec![0],
    };
    let mut operands = vec![
        binding(Dtype::F32, vec![SLOTS]),
        binding(Dtype::F32, vec![UPDATES]),
    ];
    if guarded {
        operands.push(binding(Dtype::Bool, Vec::new()));
    }
    operands.push(binding(Dtype::F32, vec![SLOTS]));
    let mut module = builder.finish(main).unwrap();
    crate::scalarize::run(&mut module).unwrap();
    loom_program! {
        name: "indexed".into(),
        module,
        lane_count: 1,
        domains: vec![Domain {
            extent: 5,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands,
        num_inputs: 2 + usize::from(guarded),
        available_views: Vec::new(),
    }
}

#[test]
fn unguarded_indexed_program_omits_its_zero_skip_slot() {
    let loom = indexed_program(false);
    let domain = Plan::prepare(&loom).unwrap();
    let incidence = std::cell::OnceCell::new();
    let execution = PlanIr::build_bounded(
        &loom,
        domain,
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        &incidence,
        false,
    )
    .unwrap();
    assert!(execution.candidates[0].guards.iter().all(Option::is_none));
    let table = SlotTable::indexed(&[32, 64, 128]);
    assert_eq!(table.specs[1].flavor, CandidateFlavor::ZeroSkip);

    let (_, profile) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    assert!(profile.kernels.iter().any(|unit| unit.flavor == "baseline"));
    assert!(profile.kernels.iter().all(|unit| unit.flavor != "zero_skip"));
}

#[test]
fn guarded_indexed_program_renders_its_plain_zero_skip_slot() {
    let loom = indexed_program(true);
    let domain = Plan::prepare(&loom).unwrap();
    let incidence = std::cell::OnceCell::new();
    let execution = PlanIr::build_bounded(
        &loom,
        domain,
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        &incidence,
        false,
    )
    .unwrap();
    assert!(execution.candidates[0].guards.iter().any(Option::is_some));

    let (_, profile) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    assert!(
        profile
            .kernels
            .iter()
            .any(|unit| unit.flavor == "zero_skip" && unit.entry.ends_with("_bs32_zs"))
    );
}

fn lane_hoist_program(mixed_outputs: bool) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let narrow = builder.param(Shape::scalar(DType::F32));
    let wide = builder.param(Shape::scalar(DType::F32));
    let narrow_stage = builder.multiply(narrow, narrow);
    let hoistable = builder.multiply(narrow_stage, wide);
    let mut outputs = vec![hoistable];
    if mixed_outputs {
        outputs.push(builder.multiply(wide, wide));
    }
    let main = builder.end_func(outputs);
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![0],
    };
    let mut operands = vec![binding(1), binding(8), binding(8)];
    if mixed_outputs {
        operands.push(binding(8));
    }
    loom_program! {
        name: "lane_hoist_recipe".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands,
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

fn vector_io_program(u: i64) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let input = builder.param(Shape::scalar(DType::Bf16));
    let main = builder.end_func(vec![input]);
    let binding = FlatOperand {
        dtype: Dtype::Bf16,
        element_shape: Vec::new(),
        lane_extent: u,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "vector_io_recipe".into(),
        module: builder.finish(main).unwrap(),
        lane_count: u,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding.clone(), binding],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

fn vector2_io_program(u: i64, output_width_one: bool) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let input = builder.param(Shape::scalar(DType::F32));
    let main = builder.end_func(vec![input]);
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "vector2_io_recipe".into(),
        module: builder.finish(main).unwrap(),
        lane_count: u,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(u), binding(if output_width_one { 1 } else { u })],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

fn multi_warp_recipe_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.multiply(x, x);
    let main = builder.end_func(vec![y, y]);
    let binding = |lane_extent| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "multi_warp_recipe".into(),
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

fn shared_diamond_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let mut shared = builder.multiply(x, y);
    for _ in 0..12 {
        shared = builder.multiply(shared, y);
    }
    let left = builder.add(shared, x);
    let right = builder.add(shared, y);
    let main = builder.end_func(vec![left, right]);
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "shared_diamond_recipe".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(), binding(), binding(), binding()],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

#[test]
fn compile_offers_only_aligned_vector_io_candidates() {
    let entries = |u| {
        let (compiled, _) = compile_slot_variations_profiled(
            vector_io_program(u),
            test_options(None),
            DEFAULT_MAX_KERNEL_STATEMENTS,
            usize::MAX,
            0,
            true,
        )
        .unwrap();
        let recipe = compiled.recipe;
        let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
            panic!("vector I/O launch is not a choice");
        };
        candidates
            .iter()
            .map(|candidate| {
                let Work::Call {
                    call: KernelLaunch { entry, .. },
                    geometry,
                    ..
                } = single_launch_work(candidate)
                else {
                    panic!("vector I/O candidate is not a kernel launch");
                };
                let [Geometry { grid, block, .. }] = geometry[..] else {
                    panic!("vector I/O candidate is not a kernel launch");
                };
                (entry.clone(), block[0], grid[0])
            })
            .collect::<Vec<_>>()
    };

    let aligned = entries(256);
    // Scalar, v4, and v8 at every block size, plus the scalar whole-program twin.
    assert_eq!(aligned.len(), BLOCK_SIZES.len() * 4);
    for &block in &BLOCK_SIZES {
        assert!(
            aligned
                .iter()
                .any(|(entry, b, _)| entry.contains("_v4") && *b == block)
        );
        assert!(
            aligned
                .iter()
                .any(|(entry, b, _)| entry.contains("_v8") && *b == block)
        );
    }
    assert!(aligned.iter().any(|(entry, block, grid)| {
        entry.ends_with("_bs128")
            && *block == 128
            && *grid == ::recipe::count_covering_blocks(7, 4).unwrap()
    }));
    assert!(!aligned.iter().any(|(entry, _, _)| entry.contains("_wp")));

    let vec4_only = entries(128);
    // Scalar and v4 at every block size, plus the scalar whole-program twin.
    assert_eq!(vec4_only.len(), BLOCK_SIZES.len() * 3);
    assert!(
        vec4_only
            .iter()
            .any(|(entry, block, _)| entry.contains("_v4") && *block == 32)
    );
    assert!(!vec4_only.iter().any(|(entry, _, _)| entry.contains("_v8")));
    assert!(!vec4_only.iter().any(|(entry, _, _)| entry.contains("_wp")));

    let tail = entries(260);
    // No lane width divides 260, so only the scalar render and its
    // whole-program twin stand.
    assert_eq!(tail.len(), BLOCK_SIZES.len() * 2);
    assert!(
        tail.iter()
            .all(|(entry, _, _)| !entry.contains("_v4") && !entry.contains("_v8"))
    );
    assert!(!tail.iter().any(|(entry, _, _)| entry.contains("_wp")));
}

#[test]
fn compile_offers_f32_vector_and_multi_warp_geometries() {
    let entries = |loom| {
        let (compiled, _) = compile_slot_variations_profiled(
            loom,
            test_options(None),
            DEFAULT_MAX_KERNEL_STATEMENTS,
            usize::MAX,
            0,
            true,
        )
        .unwrap();
        let recipe = compiled.recipe;
        let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
            panic!("f32 vector I/O launch is not a choice");
        };
        candidates
            .iter()
            .map(|candidate| {
                let Work::Call {
                    call: KernelLaunch { entry, .. },
                    ..
                } = single_launch_work(candidate)
                else {
                    panic!("f32 vector I/O candidate is not a kernel launch");
                };
                entry.clone()
            })
            .collect::<Vec<_>>()
    };

    let u128 = entries(vector2_io_program(128, false));
    assert!(u128.iter().any(|entry| entry.ends_with("_bs32_v2")));
    assert!(u128.iter().any(|entry| entry.ends_with("_bs128_v2")));
    assert!(u128.iter().any(|entry| entry.ends_with("_bs32_v4")));
    assert!(u128.iter().any(|entry| entry.ends_with("_bs128_wp4")));
    assert!(u128.iter().any(|entry| entry.ends_with("_bs128_v2_wp4")));
    assert!(u128.iter().all(|entry| !entry.contains("_v8")));

    let u256 = entries(vector2_io_program(256, false));
    assert!(u256.iter().any(|entry| entry.ends_with("_bs32_v2")));

    let u96 = entries(vector2_io_program(96, false));
    assert!(u96.iter().all(|entry| !entry.contains("_v2")));
    assert!(u96.iter().any(|entry| entry.ends_with("_bs128_wp2")));
    assert!(u96.iter().all(|entry| !entry.contains("_wp4")));

    // A width-1 output folds each thread's lanes into one scalar before the warp
    // tree. That fold is exact, so the wide loads still pay: the same vector and
    // warp geometries are offered as for a lane-wide output.
    let folded = entries(vector2_io_program(128, true));
    assert!(folded.iter().any(|entry| entry.ends_with("_bs32_v2")));
    assert!(folded.iter().any(|entry| entry.ends_with("_bs32_v4")));
    assert!(folded.iter().any(|entry| entry.ends_with("_bs128_v2_wp4")));
    assert!(folded.iter().all(|entry| !entry.contains("_v8")));
}

#[test]
fn compile_offers_sharded_geometries() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [4]));
    let y = builder.multiply(x, x);
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![4],
        lane_extent: 128,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "sharded_recipe".into(),
        module,
        lane_count: 128,
        domains: vec![Domain {
            extent: 7,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(), binding()],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let (compiled, _) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("sharded launch is not a choice");
    };
    let launches = candidates
        .iter()
        .map(|candidate| {
            let Work::Call {
                call: KernelLaunch { entry, .. },
                geometry,
                ..
            } = single_launch_work(candidate)
            else {
                panic!("candidate is not a kernel launch");
            };
            let [Geometry { grid, .. }] = geometry[..] else {
                panic!("candidate is not a kernel launch");
            };
            (entry.clone(), grid)
        })
        .collect::<Vec<_>>();
    // The shard count rides grid.y; unsharded variants keep grid.y == 1.
    assert!(
        launches
            .iter()
            .any(|(entry, grid)| entry.ends_with("_sh2") && grid[1] == 2)
    );
    assert!(
        launches
            .iter()
            .any(|(entry, grid)| entry.ends_with("_bs128_v2_wp4_sh3") && grid[1] == 3)
    );
    assert!(
        launches
            .iter()
            .filter(|(entry, _)| !entry.contains("_sh"))
            .all(|(_, grid)| grid[1] == 1)
    );
    // Pass splits only target register-fat bodies; this small program
    // stays below the statement floor.
    assert!(launches.iter().all(|(entry, _)| !entry.contains("_pp")));
}

#[test]
fn compile_offers_pass_splits_for_large_kernels() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [64]));
    let mut y = x;
    for _ in 0..20 {
        y = builder.multiply(y, x);
    }
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![64],
        lane_extent: 8,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "pass_split".into(),
        module,
        lane_count: 8,
        domains: vec![Domain {
            extent: 17,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(), binding()],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let (compiled, _) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("pass split launch is not a choice");
    };
    let launches = candidates
        .iter()
        .map(|candidate| {
            let Work::Call {
                call: KernelLaunch { entry, .. },
                geometry,
                ..
            } = single_launch_work(candidate)
            else {
                panic!("candidate is not a kernel launch");
            };
            let [Geometry { grid, .. }] = geometry[..] else {
                panic!("candidate is not a kernel launch");
            };
            (entry.clone(), grid)
        })
        .collect::<Vec<_>>();
    // Passes run sequentially inside one block: grid.y stays 1.
    for count in Settings::default().split_counts {
        assert!(
            launches
                .iter()
                .any(|(entry, grid)| entry.ends_with(&format!("_pp{count}")) && grid[1] == 1),
            "missing _pp{count} in {launches:?}"
        );
    }
}

#[test]
fn compile_adds_register_capped_vector_geometries_for_large_kernels() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [64]));
    let mut y = x;
    for _ in 0..20 {
        y = builder.multiply(y, x);
    }
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![64],
        lane_extent: 128,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "vec_reg_capped".into(),
        module,
        lane_count: 128,
        domains: vec![Domain {
            extent: 17,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![binding(), binding()],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let (compiled, _) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("vector reg-cap launch is not a choice");
    };
    let launches = candidates
        .iter()
        .map(|candidate| candidate_launch(&recipe, candidate).expect("candidate launches a kernel"))
        .collect::<Vec<_>>();
    let cap = Settings::default().register_caps[0];
    let capped = |suffix: &str| {
        launches
            .iter()
            .any(|(entry, _, registers)| *registers == Some(cap) && entry.ends_with(suffix))
    };
    assert!(capped("_bs32_v2"));
    assert!(capped("_bs128_v2_wp4"));
    // Scalar crosses get no cap: warps split the u range and shrink
    // per-thread liveness instead of widening it.
    assert!(launches.iter().all(|(entry, _, registers)| {
        registers.is_none() || !entry.contains("_wp4") || entry.contains("_v")
    }));
}

#[test]
fn guarded_launch_offers_multi_warp_zero_skip_positions() {
    let mut loom = guarded_program();
    loom.lane_count = 256;
    for binding in &mut loom.operands {
        binding.lane_extent = if binding.dtype == Dtype::Bool { 1 } else { 256 };
    }
    let program = Plan::prepare(&loom).unwrap();
    let execution = PlanIr::build_bounded(
        &loom,
        program,
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        &std::cell::OnceCell::new(),
        true,
    )
    .unwrap();
    let candidate = execution
        .candidates
        .iter()
        .find(|candidate| candidate.materialized.is_empty())
        .unwrap();
    emit_planned_zero_skip_geometry_kernel(
        &execution.domain,
        &loom,
        candidate,
        0,
        &SlotSpec {
            warps: 4,
            ..SlotSpec::plain(CandidateFlavor::ZeroSkip, 128)
        },
    )
    .unwrap();
    let (compiled, _) = compile_slot_variations_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("guarded launch is not a choice");
    };
    let entries = candidates
        .iter()
        .filter_map(|candidate| match single_launch_work(candidate) {
            Work::Call {
                call: KernelLaunch { entry, .. },
                ..
            } => Some(entry.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        entries.iter().any(|entry| entry.contains("_zs_wp4")),
        "{entries:?}"
    );
}

#[test]
fn descriptions_rank_multi_warp_before_optional_scalar_blocks() {
    let loom = multi_warp_recipe_program();
    let settings = BackendSettings::default();
    let limits = CompileLimits { statement_cap: DEFAULT_MAX_KERNEL_STATEMENTS, register_cap: usize::MAX,
        cut_limit: 0, plan_clusters: true, max_workspace_bytes: None,
        element_order: ElementOrder::Trace, target_sm: TEST_TARGET_SM, dump_ptx: false,
        record_structure: false, settings: &settings };
    let planned = super::pipeline::plan_single_with_limits_profiled(
        loom, limits, Placement::Default, &mut CompileProfile::default()).unwrap();
    let table = SlotTable::new(&BLOCK_SIZES, &settings.compiler);
    let render = Render { loom: &planned.loom, execution: &planned.execution,
        incidence: &planned.incidence, settings: &settings, table: &table,
        sm: TEST_TARGET_SM, statement_cap: DEFAULT_MAX_KERNEL_STATEMENTS, required: true };
    let slots = render.describe_slots(&planned.estimates, planned.estimates.baseline_index.unwrap(), 0);
    assert_eq!(slots[0], 0);
    let multi = slots.iter().position(|&slot| {
        let spec = table.specs[slot];
        spec.warps == 4 && spec.io_lanes == 1 && spec.block_size == 128
    }).unwrap();
    let scalar = slots.iter().position(|&slot| {
        let spec = table.specs[slot];
        spec.flavor == CandidateFlavor::Baseline && spec.block_size == 64
    }).unwrap();
    assert!(multi < scalar);
}

#[test]
fn compile_puts_baseline_and_lane_hoist_in_one_choice() {
    let (compiled, _) = compile_slot_variations_profiled(
        lane_hoist_program(false),
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("lane-hoist launch is not a choice");
    };
    // The baseline, its whole-program twin, and the lane hoist at every block
    // size.
    assert_eq!(candidates.len(), BLOCK_SIZES.len() * 3);
    assert_eq!(
        candidates
            .iter()
            .filter(|candidate| matches!(
                single_launch_work(candidate),
                Work::Call { call: KernelLaunch { entry, .. }, .. } if entry.contains("_lh")
            ))
            .count(),
        BLOCK_SIZES.len()
    );
    let mut blocks = candidates
        .iter()
        .map(|candidate| {
            let Work::Call {
                call: KernelLaunch { entry, .. },
                geometry,
                ..
            } = single_launch_work(candidate)
            else {
                panic!("lane-hoist candidate is not a kernel launch");
            };
            let [Geometry { block, .. }] = geometry[..] else {
                panic!("lane-hoist candidate is not a kernel launch");
            };
            assert!(entry.starts_with("loom_lane_hoist_recipe_c_bs"));
            block[0]
        })
        .collect::<Vec<_>>();
    blocks.sort_unstable();
    let mut expected = [BLOCK_SIZES, BLOCK_SIZES, BLOCK_SIZES].concat();
    expected.sort_unstable();
    assert_eq!(blocks, expected);
}

/// The flavors whose slots sit past the baseline block sizes: a lane hoist, a
/// multi-warp arrangement, and its vector-2 sibling.
#[test]
fn launch_without_a_hoistable_stage_does_not_reject_its_sibling() {
    let loom = lane_hoist_program(true);
    let prepared = Plan::prepare(&loom).unwrap();
    let cap = (1..DEFAULT_MAX_KERNEL_STATEMENTS).find(|&cap| {
        bounded_parts(&prepared, &loom, cap).is_ok_and(|parts| parts.len() == 2)
    }).expect("mixed outputs admit a two-launch bounded plan");
    let (compiled, _) = compile_slot_variations_profiled(
        loom, test_options(None), cap, usize::MAX, 0, false).unwrap();
    let arms = pass_arms(&compiled.recipe);
    let mut menus = vec![BTreeSet::new(), BTreeSet::new()];
    for arm in arms {
        assert_complete(arm);
        assert_eq!(arm.nodes().len(), 2);
        for (index, node) in arm.nodes().iter().enumerate() {
            let Work::Choice { candidates, .. } = node.work() else { panic!("selected launch"); };
            let (entry, block, cap) = candidate_launch(&compiled.recipe, &candidates[0]).unwrap();
            menus[index].insert((entry, block, cap));
        }
    }
    let mut hoisted_blocks = menus.iter().map(|menu| {
        menu.iter().filter(|(entry, _, _)| entry.contains("_lh"))
            .map(|(_, block, _)| *block).collect::<BTreeSet<_>>()
    }).collect::<Vec<_>>();
    hoisted_blocks.sort();
    assert_eq!(hoisted_blocks, [BTreeSet::new(), BLOCK_SIZES.into_iter().collect()]);
    for menu in menus {
        assert_eq!(menu.iter().filter(|(entry, _, _)| !entry.contains("_lh"))
            .map(|(_, block, _)| *block).collect::<BTreeSet<_>>(), BLOCK_SIZES.into_iter().collect());
    }
}
