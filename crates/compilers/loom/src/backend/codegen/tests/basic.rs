// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn names(params: &[Param]) -> Vec<&str> {
    params.iter().map(|param| param.name.as_str()).collect()
}

fn distinct(names: &[&str]) -> bool {
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len() == names.len()
}

/// A consumer takes what the layout says, in that order, and nothing twice.
#[test]
fn a_consumer_declares_the_layout_it_was_planned_against() {
    let loom = multi_warp_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let consumer = emit_consumer_kernel_flavor(
        &prepared,
        &loom,
        &parts[0],
        Flavor::plain(BLOCK_SIZES[0], 0, 1),
    )
    .unwrap();

    let taken = consumer.kernel.function.parameters();
    let declared = names(&taken);
    assert_eq!(declared, names(&prepared.layout.params));
    assert!(distinct(&declared), "{declared:?} repeats a name");
    assert!(
        consumer.kernel.function.shared_arrays().is_empty(),
        "one warp shares nothing"
    );
}

/// Several warps to a position share one slot per warp per accumulator, and
/// they are declared in accumulator order.
#[test]
fn a_multi_warp_consumer_declares_one_partial_per_accumulator() {
    let loom = multi_warp_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &loom, 128, 0, 1, &parts[0], &[], 2).unwrap();

    let arrays = kernel.function.shared_arrays();
    let shared = arrays
        .iter()
        .map(|array| array.name.as_str())
        .collect::<Vec<_>>();
    assert!(!shared.is_empty(), "several warps share their partials");
    assert!(distinct(&shared), "{shared:?} repeats a name");
    let mut ordered = shared.clone();
    ordered.sort_unstable();
    assert_eq!(shared, ordered, "declared in accumulator order");
    assert!(
        shared.iter().all(|name| name.contains("partial")),
        "{shared:?}"
    );
}

use super::super::super::partition::{IncidenceProfile, Settings as PartitionSettings};

/// The merge splits where the emitter said the prologue ended: the shared
/// opening is exactly what one group reported, and what follows it is one
/// statement per group rather than more of the opening.
#[test]
fn the_merge_splits_where_the_emitter_reported() {
    let loom = multi_warp_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let flavor = Flavor::plain(BLOCK_SIZES[0], 0, 1);
    let whole = emit_consumer_kernel_flavor(&prepared, &loom, &parts[0], flavor).unwrap();
    let merged = emit_split_kernel(
        &prepared,
        &loom,
        &parts[0],
        flavor,
        Split::Passes(2),
        &IncidenceProfile::prepare(&loom).unwrap(),
        &PartitionSettings::default(),
    )
    .unwrap();

    let reported = whole.prologue_end;
    assert_eq!(
        shapes_of(merged.body(), &steps_of(merged.body())[..reported]),
        shapes_of(
            whole.kernel.body(),
            &steps_of(whole.kernel.body())[..reported]
        ),
        "the shared opening is the prologue the emitter reported"
    );
    assert_eq!(
        steps_of(merged.body()).len(),
        reported + 2,
        "and after it, one statement per group"
    );
    assert!(matches!(
        merged.body().op(steps_of(merged.body())[reported]).kind,
        gpu_ir::OpKind::Scope
    ));
}

/// The emitter reports where the position prologue ends, and what it laid
/// down is the lane, the flat position, the bounds guard, and one index per
/// axis the iteration actually walks — an axis of extent 1 is the constant
/// zero and binds nothing.
#[test]
fn the_reported_prologue_is_the_position_opening() {
    for program in [
        multi_warp_program(),
        sender_scatter_program(8, 8),
        sender_scatter_program(32, 1),
    ] {
        let prepared = Plan::prepare(&program).unwrap();
        let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
        let consumer = emit_consumer_kernel_flavor(
            &prepared,
            &program,
            &parts[0],
            Flavor::plain(BLOCK_SIZES[0], 0, 1),
        )
        .unwrap();
        let sizes = prepared.index.batch_sizes();
        let walked = sizes.iter().filter(|&&size| size != 1).count();
        assert_eq!(
            consumer.prologue_end,
            3 + walked,
            "sizes {sizes:?} open with lane, position, guard and {walked} indices"
        );
        assert!(
            steps_of(consumer.kernel.body()).len() > consumer.prologue_end,
            "the prologue is an opening, not the whole body"
        );
    }
}

#[test]
fn fiber_scatter_wide_output_stores_inside_the_fiber_loop() {
    let program = sender_scatter_program(8, 8);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    assert!(prepared.index.fiber_bridge().is_some());
    assert!(prepared.index.fiber_scatter(2));
    assert!(!prepared.index.fiber_scatter(3));
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert_eq!(parts.len(), 1);
    let kernel = emit_kernel(&prepared, &program, 32, 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(!ptx.contains("red.global"), "{ptx}");
    // One fiber-direct store per edge, one direct store per sender.
    assert_eq!(
        ptx.matches("st.global.L1::evict_first.f32").count(),
        2,
        "{ptx}"
    );
}

#[test]
fn fiber_scatter_width1_output_reduces_across_the_warp_per_element() {
    let program = sender_scatter_program(32, 1);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert_eq!(parts.len(), 1);
    let kernel = emit_kernel(&prepared, &program, 32, 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(!ptx.contains("red.global"), "{ptx}");
    // The per-edge warp tree inside the fiber loop; u=32 is a single pass,
    // so lane 0 stores without reading the output back.
    assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 5, "{ptx}");
    assert!(!ptx.contains("ld.global.f32 "), "{ptx}");

    // Two u passes at u=64: the second pass accumulates onto the first.
    let program = sender_scatter_program(64, 1);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_kernel(&prepared, &program, 32, 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("ld.global.f32"), "{ptx}");

    // Divergent u passes cannot host the warp tree.
    let program = sender_scatter_program(8, 1);
    let prepared = Plan::prepare_fiber_direct(&program).unwrap().unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert!(emit_kernel(&prepared, &program, 32, 0, 1, &parts[0]).is_err());
}

#[test]
fn sender_placement_requires_a_moved_iteration() {
    // All outputs on the edge domain: nothing promotes, no sender plan.
    let mut program = sender_scatter_program(8, 8);
    program.operands[3].axis_domains[0] = 0;
    assert!(Plan::prepare_fiber_direct(&program).unwrap().is_none());
}

#[test]
fn a_typed_layout_uses_physical_input_scalars_and_output_accumulators() {
    let mut program = program(
        &[Dtype::F32, Dtype::F32],
        vec![Domain { extent: 4 }],
        Vec::new(),
    );
    program.operands[0].dtype = Dtype::Bool;
    program.operands[1].dtype = Dtype::U64;
    let prepared = Plan::prepare(&program).unwrap();
    assert!(matches!(
        prepared.layout.params[0].ty,
        gpu_ir::ParamTy::Ptr(Scalar::U8)
    ));
    // An output whose storage differs from the dtype the program computes in
    // accumulates in the latter and converts once, in its projection kernel, so
    // the main kernel's parameter carries the accumulator rather than storage.
    assert!(matches!(
        prepared.layout.params[1].ty,
        gpu_ir::ParamTy::Ptr(Scalar::F32)
    ));
}

/// An f16 output accumulates in f32 and narrows in its projection kernel: the
/// main kernel stores the math dtype, and the conversion happens once, after.
#[test]
fn a_narrow_output_narrows_in_its_projection_kernel() {
    let mut program = program(
        &[Dtype::F32, Dtype::F32],
        vec![Domain { extent: 4 }],
        Vec::new(),
    );
    program.operands[1].dtype = Dtype::F16;
    let prepared = Plan::prepare(&program).unwrap();
    let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_kernel(&prepared, &program, 32, 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(!ptx.contains("cvt.rn.f16.f32"), "{ptx}");
    assert!(ptx.contains("st.global.L1::evict_first.f32"), "{ptx}");

    let projection = super::super::super::reduction::projection_kernel(
        &prepared,
        &program,
        0,
        crate::domain::reduction::Projection::Convert,
    )
    .unwrap();
    let projected = gpu_ir::ptx::render_ptx_module(&[projection], 80);
    assert!(projected.contains("cvt.rn.f16.f32"), "{projected}");
    assert!(projected.contains("st.global.b16"), "{projected}");
}

#[test]
fn preimage_recipe_arrays_match_ptx_parameter_layout() {
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
    let prepared = Plan::prepare(&program).unwrap();
    assert_eq!(prepared.slots.len(), 1);
    // An unsorted preimage walk binds its offsets and its source indices, and
    // the kernel takes them between the input and the output.
    assert_eq!(prepared.slots[0].kind.array_count(), 2);
    assert_eq!(
        prepared
            .layout
            .params
            .iter()
            .map(|param| param.name.as_str())
            .collect::<Vec<_>>(),
        [
            "in_0",
            "preimage_offsets_0",
            "preimage_source_indices_0",
            "out_0"
        ]
    );
}

#[test]
fn direct_and_hidden_reduction_kernels_render_as_ptx() {
    for output_width in [8, 1] {
        let mut program = program(
            &[Dtype::F32, Dtype::F32],
            vec![Domain { extent: 4 }],
            Vec::new(),
        );
        program.operands[1].lane_extent = output_width;
        let kernel = emit_test_kernel(&program);
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
        assert!(ptx.contains(".entry loom_backend_contract_c_bs32"));
        assert!(ptx.contains("st.global"));
        if output_width == 1 {
            assert_eq!(ptx.matches("shfl.sync.down.b32").count(), 5);
        } else {
            assert!(!ptx.contains("shfl.sync.down"));
        }
    }
}

/// A guard that folds into accumulators produces the accumulators.
///
/// The branch is a definition: it hands back one value per accumulator, and
/// what follows reads those. Naming the storage they are produced into is what
/// lets the renderer give them the registers the accumulators already run in,
/// so the join costs nothing — but the values are new ones either way, and the
/// steps after the branch must be reading them and not what came in.
#[test]
fn the_guards_that_fold_into_accumulators_produce_them() {
    let loom = multi_warp_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &loom, 128, 0, 1, &parts[0], &[], 2).unwrap();

    let f = kernel.body();
    let mut folds = 0;
    for op in nested(f, &steps_of(f)) {
        if !matches!(f.op(op).kind, gpu_ir::OpKind::If) {
            continue;
        }
        // The combine is the branch holding the walk over the warps' slots.
        let holds_a_walk = f.op(op).regions.iter().any(|&region| {
            f.region(region).blocks.iter().any(|&block| {
                f.block(block)
                    .ops
                    .iter()
                    .any(|&inner| matches!(f.op(inner).kind, gpu_ir::OpKind::Loop { .. }))
            })
        });
        if !holds_a_walk {
            continue;
        }
        folds += 1;
        assert!(
            !f.op(op).results.is_empty(),
            "the block combine writes in place where it should produce a value"
        );
        assert_eq!(
            f.op(op).into.len(),
            f.op(op).results.len(),
            "every result names the storage it is produced into"
        );
        assert!(
            f.op(op).into.iter().all(Option::is_some),
            "a result of the combine names no storage"
        );
        // The arm that does not run hands back exactly those places, so the
        // renderer can see the branch writes them whichever way it goes.
        let untaken = f.region_block(f.op(op).regions[1]);
        assert!(f.block(untaken).ops.is_empty(), "the other arm does work");
        assert_eq!(
            f.block(untaken).terminator.yields,
            f.op(op).into.iter().flatten().copied().collect::<Vec<_>>(),
            "the other arm hands back something other than the accumulators"
        );
    }
    assert!(folds > 0, "the fixture reaches no block combine");

    // And what follows a combine reads what it produced: once an accumulator
    // has been folded into, the value that reached the branch names storage
    // that has moved on, and no step after it in the same list may name it.
    fn nothing_reads_what_was_replaced(f: &gpu_ir::Function, block: gpu_ir::BlockId) {
        let ops = &f.block(block).ops;
        for (index, &op) in ops.iter().enumerate() {
            for &region in &f.op(op).regions {
                nothing_reads_what_was_replaced(f, f.region_block(region));
            }
            if !matches!(f.op(op).kind, gpu_ir::OpKind::If) || f.op(op).results.is_empty() {
                continue;
            }
            let replaced = f.op(op).into.iter().flatten().copied();
            for stale in replaced {
                let mut later = std::collections::BTreeSet::new();
                for &after in &ops[index + 1..] {
                    if f.op(after).destination(0) == Some(stale) {
                        break;
                    }
                    f.add_op_reads(after, &mut later);
                }
                for &yielded in &f.block(block).terminator.yields {
                    f.add_reads(yielded, &mut later);
                }
                assert!(
                    !later.contains(&stale),
                    "a step after the fold still names the accumulator it replaced"
                );
            }
        }
    }
    nothing_reads_what_was_replaced(f, f.entry());
}

/// And the branch costs nothing: the accumulators keep their registers across
/// it, so neither arm copies and no default is written before it.
#[test]
fn a_fold_into_an_accumulator_joins_without_a_copy() {
    let loom = multi_warp_program();
    let prepared = Plan::prepare(&loom).unwrap();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_multi_warp_kernel(&prepared, &loom, 128, 0, 1, &parts[0], &[], 2).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);

    // A join would show as a float move: one before the branch for the default
    // and one at the end of the arm. The accumulators are folded in place, so
    // the only float moves are the ones that zero them.
    let moved = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("mov.f32"))
        .collect::<Vec<_>>();
    assert!(
        moved.iter().all(|line| line.contains("0f00000000")),
        "the fold copies rather than folding in place: {moved:?}"
    );

    // And the accumulator the walk folds into is the one the store reads.
    let combined = ptx
        .lines()
        .filter(|line| line.trim_start().starts_with("add.rn.f32"))
        .map(|line| {
            line.split_whitespace()
                .nth(1)
                .unwrap()
                .trim_end_matches(',')
                .to_owned()
        })
        .collect::<std::collections::BTreeSet<_>>();
    let stored = ptx
        .lines()
        .filter(|line| line.contains("st.global"))
        .filter_map(|line| line.rsplit(", ").next())
        .map(|held| held.trim_end_matches(';').to_owned())
        .collect::<Vec<_>>();
    assert!(
        stored.iter().any(|held| combined.contains(held)),
        "the store reads a register nothing folded into: {ptx}"
    );
}
