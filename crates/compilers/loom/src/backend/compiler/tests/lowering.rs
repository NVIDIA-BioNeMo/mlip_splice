// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

#[test]
fn atomic_reduction_candidates_use_no_workspace() {
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
    let loom = loom_program! {
        name: "whole_reduce".into(),
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
    let [node] = recipe.body().nodes() else {
        panic!("reduction recipe should contain one whole-recipe choice");
    };
    let Work::Choice { candidates, .. } = node.work() else {
        panic!("reduction recipe is not a whole-recipe choice");
    };
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.workspace().is_empty()),
        "a reduction candidate must not reserve shape-dependent workspace"
    );
    let direct = candidates
        .iter()
        .filter(|candidate| {
            candidate.nodes().iter().any(|node| {
                matches!(node.work(), Work::Call { call: KernelLaunch { entry, .. }, .. } if entry.contains("_br_bs"))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        direct.len(),
        Settings::default().reduction_block_sizes.len()
    );
    for candidate in direct {
        assert_eq!(candidate.nodes().len(), 1);
        assert!(
            !candidate
                .nodes()
                .iter()
                .any(|node| matches!(node.work(), Work::Memset { .. }))
        );
    }
}

#[test]
fn workspace_bound_removes_oversized_candidates() {
    let loom = shared_diamond_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let statement_cap = (73..512)
        .find(|&cap| bounded_parts(&prepared, &loom, cap).is_ok_and(|parts| parts.len() == 2))
        .unwrap();
    let options = CompileOptions {
        max_workspace_bytes: Some(0),
        ..test_options(None)
    };
    let (compiled, _) =
        compile_slot_variations_profiled(loom, options, statement_cap, 0, 1, true).unwrap();
    assert_eq!(
        ::recipe::algebra::workspace_upper_bound(&compiled.recipe),
        0
    );
}

#[test]
fn compile_puts_baseline_and_zero_skip_in_one_choice() {
    let (compiled, _) = compile_slot_variations_profiled(
        guarded_program(),
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let [node] = recipe.body().nodes() else {
        panic!("guarded recipe should contain one launch choice");
    };
    let Work::Choice { candidates, .. } = node.work() else {
        panic!("guarded recipe launch is not a choice");
    };
    // Every block size renders three ways: the baseline, its whole-program
    // twin, and the zero-skip guard.
    assert_eq!(candidates.len(), BLOCK_SIZES.len() * 3);
    assert_eq!(
        candidates
            .iter()
            .filter(|candidate| matches!(
                single_launch_work(candidate),
                Work::Call { call: KernelLaunch { entry, .. }, .. } if entry.contains("_zs")
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
                panic!("guarded candidate is not a kernel launch");
            };
            let [Geometry { block, .. }] = geometry[..] else {
                panic!("guarded candidate is not a kernel launch");
            };
            assert!(entry.starts_with("loom_guarded_recipe_c_bs"));
            block[0]
        })
        .collect::<Vec<_>>();
    blocks.sort_unstable();
    let mut expected = [BLOCK_SIZES, BLOCK_SIZES, BLOCK_SIZES].concat();
    expected.sort_unstable();
    assert_eq!(blocks, expected);
}

#[test]
fn compile_adds_register_capped_variants_for_large_kernels() {
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
        name: "reg_capped".into(),
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
    let choice = recipe
        .body()
        .nodes()
        .iter()
        .find_map(|node| match node.work() {
            Work::Choice { candidates, .. } => Some(candidates),
            _ => None,
        })
        .expect("recipe contains a launch choice");
    for cap in Settings::default().register_caps {
        // A cap is not another kernel: the capped candidate launches the same
        // entry, under a request that states the cap.
        let capped = choice
            .iter()
            .filter_map(|candidate| candidate_launch(&recipe, candidate))
            .filter(|(_, _, registers)| *registers == Some(cap))
            .collect::<Vec<_>>();
        assert_eq!(capped.len(), BLOCK_SIZES.len());
        for (entry, block, _) in capped {
            // A capped candidate keeps its baseline launch geometry.
            assert!(entry.contains(&format!("_bs{block}")));
        }
    }
}

#[test]
fn compile_candidate_order_is_deterministic() {
    fn launch_order(recipe: &::recipe::Recipe) -> Vec<(String, u32)> {
        let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
            panic!("launch is not a choice");
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
                    panic!("candidate is not a kernel launch");
                };
                let [Geometry { block, .. }] = geometry[..] else {
                    panic!("candidate is not a kernel launch");
                };
                (entry.clone(), block[0])
            })
            .collect()
    }

    let compile = || {
        compile_slot_variations_profiled(
            guarded_program(),
            test_options(None),
            DEFAULT_MAX_KERNEL_STATEMENTS,
            usize::MAX,
            0,
            true,
        )
        .unwrap()
        .0
        .recipe
    };
    assert_eq!(launch_order(&compile()), launch_order(&compile()));
}

#[test]
fn oversized_zero_skip_is_omitted_without_rejecting_baseline() {
    let loom = guarded_program();
    let domain = Plan::prepare(&loom).unwrap();
    let execution = PlanIr::build_bounded(
        &loom,
        domain,
        usize::MAX,
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
    let baseline =
        candidate_statement_counts(&execution.domain, &loom, candidate, BLOCK_SIZES[0]).unwrap()[0];
    let zero_skip = zero_skip_statement_counts(&execution.domain, &loom, candidate, BLOCK_SIZES[0])
        .unwrap()[0]
        .unwrap();
    assert!(zero_skip > baseline);

    let (compiled, _) =
        compile_slot_variations_profiled(loom, test_options(None), zero_skip - 1, usize::MAX, 0, true)
            .unwrap();
    let recipe = compiled.recipe;
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("baseline recipe launch is not a choice");
    };
    // The baseline and its whole-program twin at every block size, and no
    // guard: the guarded render does not fit the cap.
    assert_eq!(candidates.len(), BLOCK_SIZES.len() * 2);
    assert!(candidates.iter().all(|candidate| {
        matches!(
            single_launch_work(candidate),
            Work::Call { call: KernelLaunch { entry, .. }, .. } if !entry.contains("_zs")
        )
    }));
}

#[test]
fn materialized_cut_lowers_to_producer_consumer_recipe_choice() {
    let loom = shared_diamond_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let statement_cap = (73..512)
        .find(|&cap| bounded_parts(&prepared, &loom, cap).is_ok_and(|parts| parts.len() == 2))
        .unwrap();
    let (compiled, _) =
        compile_slot_variations_profiled(loom, test_options(None), statement_cap, 0, 1, true).unwrap();
    let recipe = compiled.recipe;
    let num_inputs = compiled.input_count;
    let num_outputs = compiled.output_count;

    assert_eq!((num_inputs, num_outputs), (2, 2));
    let [root] = recipe.body().nodes() else {
        panic!("materialized plan is not a whole-recipe choice");
    };
    let Work::Choice { candidates, .. } = root.work() else {
        panic!("materialized plan is not a choice");
    };
    let cuts = candidates.iter().filter(|candidate| !candidate.workspace().is_empty()).collect::<Vec<_>>();
    assert!(!cuts.is_empty());
    let mut blocks = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
    for cut in cuts {
        assert_complete(cut);
        assert_eq!(cut.workspace().len(), 1);
        assert_eq!(cut.workspace()[0].nbytes, 7 * 8 * 4);
        assert_eq!(cut.nodes().len(), 3);
        assert!(cut.nodes()[0].deps().is_empty());
        assert_eq!(*cut.nodes()[1].deps(), BTreeSet::from([0]));
        assert_eq!(*cut.nodes()[2].deps(), BTreeSet::from([0]));
        for (index, node) in cut.nodes().iter().enumerate() {
            let Work::Choice { candidates, .. } = node.work() else { panic!("selected launch"); };
            let Work::Call { geometry, .. } = single_launch_work(&candidates[0]) else { unreachable!(); };
            let [Geometry { block, .. }] = geometry[..] else { panic!("one geometry"); };
            assert_eq!(&block[1..], &[1, 1]);
            blocks[index].insert(block[0]);
        }
    }
    assert_eq!(blocks[0], BTreeSet::from([BLOCK_SIZES[0]]));
    for blocks in &blocks[1..] { assert_eq!(*blocks, BLOCK_SIZES.into_iter().collect()); }

}

#[test]
fn continuation_launches_chain_through_their_shared_output() {
    let program = || {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let x = builder.param(Shape::scalar(DType::F32));
        let y = builder.param(Shape::scalar(DType::F32));
        let mut sum = builder.multiply(x, y);
        for _ in 1..32 {
            let term = builder.multiply(x, y);
            sum = builder.add(sum, term);
        }
        let main = builder.end_func(vec![sum]);
        let binding = || FlatOperand {
            dtype: Dtype::F32,
            element_shape: Vec::new(),
            lane_extent: 8,
            axis_domains: vec![0],
        };
        loom_program! {
            name: "continuation_chain".into(),
            module: builder.finish(main).unwrap(),
            lane_count: 8,
            domains: vec![Domain {
                extent: 3,
            }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding(), binding()],
            num_inputs: 2,
            available_views: Vec::new(),
        }
    };
    // Every launch writes the single output, so ordered continuations must
    // chain through inferred writer-after-writer edges on that buffer.
    let mut saw_split = false;
    for cap in 73..512 {
        let Ok(CompiledProgram { recipe, .. }) = compile_with_cap(program(), cap) else {
            continue;
        };
        let lowered: Vec<&::recipe::RecipeBody> = match recipe.body().nodes() {
            [node] => match node.work() {
                Work::Choice { candidates, .. } => candidates.iter().collect(),
                _ => vec![&recipe.body()],
            },
            _ => vec![&recipe.body()],
        };
        for candidate in lowered {
            let choices = candidate
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, node)| matches!(node.work(), Work::Choice { .. }))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            saw_split |= choices.len() >= 2;
            assert!(
                choices
                    .windows(2)
                    .all(|pair| candidate.nodes()[pair[1]].deps().contains(&pair[0])),
                "cap {cap} leaves ordered continuation launches unchained"
            );
        }
        if saw_split {
            break;
        }
    }
    assert!(saw_split, "the additive chain never split into launches");
}

#[test]
fn shared_work_grouping_lowers_as_a_whole_recipe_choice() {
    let loom = shared_diamond_program();
    let statement_cap = (73..256)
        .find_map(|cap| {
            let baseline = plan(&loom, cap).ok()?;
            plan_shared_work(
                &IncidenceProfile::prepare(&loom).unwrap(),
                cap,
                &baseline,
                &PartitionSettings::default(),
            )
            .ok()
            .flatten()
            .map(|_| cap)
        })
        .unwrap();
    let (compiled, _) =
        compile_slot_variations_profiled(loom, test_options(None), statement_cap, usize::MAX, 0, true)
            .unwrap();
    let recipe = compiled.recipe;
    let num_inputs = compiled.input_count;
    let num_outputs = compiled.output_count;

    assert_eq!((num_inputs, num_outputs), (2, 2));
    let [root] = recipe.body().nodes() else {
        panic!("grouped plan is not a whole-recipe choice");
    };
    let Work::Choice { candidates, .. } = root.work() else {
        panic!("grouped plan is not a choice");
    };
    assert!(candidates.len() >= 2);
    for candidate in candidates { assert_complete(candidate); }
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.workspace().is_empty())
    );
}

#[test]
fn narrow_scatter_accumulates_wide_then_finalizes() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let value = builder.param(Shape::scalar(DType::I8));
    let main = builder.end_func(vec![value]);
    let module = builder.finish(main).unwrap();
    let loom = loom_program! {
        name: "narrow_scatter".into(),
        module,
        lane_count: 8,
        domains: vec![
            Domain {
                extent: 17,
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
        operands: vec![
            FlatOperand {
                dtype: Dtype::I8,
                element_shape: Vec::new(),
                lane_extent: 8,
                axis_domains: vec![0],
            },
            FlatOperand {
                dtype: Dtype::I8,
                element_shape: Vec::new(),
                lane_extent: 8,
                axis_domains: vec![1],
            },
        ],
        num_inputs: 1,
        available_views: Vec::new(),
    };
    let (compiled, _) = compile_slot_variations_profiled(
        loom, test_options(None), DEFAULT_MAX_KERNEL_STATEMENTS, usize::MAX, 0, true).unwrap();
    let recipe = compiled.recipe;
    assert_eq!((compiled.input_count, compiled.output_count), (2, 1));
    let mut blocks = BTreeSet::new();
    for arm in pass_arms(&recipe) {
        assert_complete(arm);
        assert_eq!(arm.workspace().len(), 1);
        assert_eq!(arm.workspace()[0].nbytes, 8 * 4);
        assert_eq!(arm.nodes().len(), 3);
        assert!(matches!(arm.nodes()[0].work(), Work::Memset {
            destination: Address { buf: WritableBuf::Workspace(0), byte_offset: 0 }, nbytes: 32, .. }));
        let Work::Choice { candidates, output_binding, .. } = arm.nodes()[1].work() else { panic!("selected partition"); };
        assert_eq!(output_binding, &vec![BufRef::Workspace(0)]);
        let candidate = &candidates[0];
        assert!(candidate.workspace().is_empty());
        let Work::Call { call: KernelLaunch { args, .. }, geometry, .. } = single_launch_work(candidate) else { unreachable!(); };
        let [Geometry { grid, block, .. }] = geometry[..] else { panic!("one geometry"); };
        blocks.insert(block[0]);
        assert_eq!(grid, [::recipe::count_covering_blocks(17, (block[0] / WARP_SIZE) as i32).unwrap(), 1, 1]);
        assert_eq!(&block[1..], &[1, 1]);
        assert!(matches!(args.last().and_then(|arg| arg.segments().first()),
            Some(Segment::Pointer(recipe::Address { buf: BufRef::Output(0), .. }))));
        assert!(matches!(arm.nodes()[2].work(), Work::Call {
            call: KernelLaunch { entry, .. }, .. } if entry == "loom_narrow_scatter_cvt0"));
        assert_eq!(*arm.nodes()[2].deps(), BTreeSet::from([1]));
    }
    assert_eq!(blocks, BLOCK_SIZES.into_iter().collect());

}

#[test]
fn oversized_visible_program_splits_into_bounded_launches() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [8]));
    let mut y = x;
    for _ in 0..20 {
        y = builder.multiply(y, x);
    }
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![8],
        lane_extent: 8,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "split_visible".into(),
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
    let statement_cap = 128;
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = bounded_parts(&prepared, &loom, statement_cap).unwrap();
    assert_eq!(parts.len(), 4);
    assert_eq!(parts.iter().map(KernelPart::len).sum::<usize>(), 8);
    for (part_index, part) in parts.iter().enumerate() {
        let kernel = emit_kernel(
            &prepared,
            &loom,
            BLOCK_SIZES[0],
            part_index,
            parts.len(),
            part,
        )
        .unwrap();
        assert!(statement_count(kernel.body()) <= statement_cap);
    }

    let options = CompileOptions {
        search_budget: Some(std::time::Duration::from_secs_f64(0.1 / 1000.0)),
        max_kernel_statements: std::num::NonZeroUsize::new(statement_cap),
        ..test_options(None)
    };
    let budgeted_recipe = compile_profiled(loom.clone(), options).unwrap().0.recipe;
    for arm in pass_arms(&budgeted_recipe) {
        for choice in choice_nodes(arm) {
            let Work::Choice { candidates, .. } = arm.nodes()[choice].work() else {
                unreachable!()
            };
            assert_eq!(candidates.len(), 1);
            let Work::Call { geometry, .. } = candidates[0].nodes()[0].work() else {
                panic!("candidate node is not a kernel launch");
            };
            let [Geometry { block, .. }] = geometry[..] else {
                panic!("candidate node is not a kernel launch");
            };
            assert_eq!(block[0], BLOCK_SIZES[0]);
        }
    }

    let CompiledProgram { recipe, .. } = compile_with_cap(loom, statement_cap).unwrap();
    // Each complete arm selects one geometry per partition. The partitions
    // run in order, write-after-write on the shared output.
    let arm = pass_arms(&recipe)[0];
    let choices = choice_nodes(arm);
    assert_eq!(choices.len(), parts.len());
    for &choice in &choices {
        let Work::Choice { candidates, .. } = arm.nodes()[choice].work() else {
            unreachable!()
        };
        assert_eq!(candidates.len(), 1);
    }
    for pair in choices.windows(2) {
        assert!(arm.nodes()[pair[1]].deps().contains(&pair[0]));
    }
}

#[test]
fn oversized_additive_element_splits_into_ordered_continuations() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let mut sum = builder.multiply(x, x);
    for _ in 1..80 {
        let term = builder.multiply(x, x);
        sum = builder.add(sum, term);
    }
    let main = builder.end_func(vec![sum]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "split_additive".into(),
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
    let statement_cap = 128;
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = bounded_parts(&prepared, &loom, statement_cap).unwrap();
    assert!(parts.len() > 1);
    assert!(!parts[0].output_continues(0));
    assert!(parts[1..].iter().all(|part| part.output_continues(0)));
    assert_eq!(parts.iter().map(KernelPart::len).sum::<usize>(), 80);

    let CompiledProgram { recipe, .. } = compile_with_cap(loom, statement_cap).unwrap();
    // One block-size Choice per additive continuation, ordered so each
    // continuation reads what its predecessor accumulated.
    for arm in pass_arms(&recipe) {
        assert_complete(arm);
        let choices = choice_nodes(arm);
        assert_eq!(choices.len(), parts.len());
        for pair in choices.windows(2) {
            assert!(arm.nodes()[pair[1]].deps().contains(&pair[0]));
        }
    }
}

/// Complete root selections, or the recipe itself when only one survived.
fn pass_arms(recipe: &::recipe::Recipe) -> Vec<&::recipe::RecipeBody> {
    if let [node] = recipe.body().nodes()
        && let Work::Choice { candidates, .. } = node.work()
        && candidates.iter().any(|arm| {
            arm.nodes()
                .iter()
                .any(|n| matches!(n.work(), Work::Choice { .. }))
        })
    {
        return candidates.iter().collect();
    }
    vec![&recipe.body()]
}

fn choice_nodes(body: &::recipe::RecipeBody) -> Vec<usize> {
    body.nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.work(), Work::Choice { .. }))
        .map(|(index, _)| index)
        .collect()
}

#[test]
fn partitions_choose_block_size_independently() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, [8]));
    let mut y = x;
    for _ in 0..20 {
        y = builder.multiply(y, x);
    }
    let main = builder.end_func(vec![y]);
    let module = builder.finish(main).unwrap();
    let binding = || FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![8],
        lane_extent: 8,
        axis_domains: vec![0],
    };
    let loom = loom_program! {
        name: "independent_blocks".into(),
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
    let statement_cap = 128;
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = bounded_parts(&prepared, &loom, statement_cap).unwrap();
    assert!(
        parts.len() >= 2,
        "need at least two partitions to test independent block sizes"
    );

    let (CompiledProgram { recipe, .. }, _) = compile_slot_variations_profiled(
        loom, test_options(None), statement_cap, usize::MAX, 0, false).unwrap();
    let tuples = pass_arms(&recipe).into_iter().map(|arm| {
        assert_complete(arm);
        let choices = choice_nodes(arm);
        assert_eq!(choices.len(), parts.len());
        choices.iter().map(|&choice| {
            let Work::Choice { candidates, .. } = arm.nodes()[choice].work() else { unreachable!(); };
            let Work::Call { geometry, .. } = single_launch_work(&candidates[0]) else { unreachable!(); };
            let [Geometry { block, .. }] = geometry[..] else { panic!("one geometry"); };
            block[0]
        }).collect::<Vec<_>>()
    }).collect::<BTreeSet<_>>();
    for launch in 0..parts.len() {
        for block in BLOCK_SIZES {
            let mut tuple = vec![BLOCK_SIZES[0]; parts.len()];
            tuple[launch] = block;
            assert!(tuples.contains(&tuple), "missing independent launch geometry {tuple:?}");
        }
    }
}
