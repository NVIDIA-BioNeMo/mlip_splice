// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn lane_hoist_moves_only_uniform_stages_before_u() {
    for u in [1, 5, 32, 64] {
        for output_width_one in [false, true] {
            let loom = lane_hoist_program(u, output_width_one, false, true);
            let plan = baseline_plan(&loom);
            let candidate = &plan.candidates[0];
            assert_eq!(candidate.hoists, vec![true]);
            let Launch::Consumer { part, .. } = &candidate.launches[0] else {
                panic!("lane-hoist launch is not a consumer");
            };
            let kernel = emit_consumer_kernel_flavor(
                &plan.domain,
                &loom,
                part,
                Flavor {
                    materialized: &candidate.materialized,
                    lane_hoist: true,
                    ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
                },
            )
            .map(|consumer| consumer.kernel)
            .unwrap();
            let loop_body = u_loop_body(kernel.body());
            assert!(loom_value_bindings(kernel.body()) > 0);
            assert!(body_load_count_in(kernel.body(), &u_loop_prefix(kernel.body())) > 0);
            assert!(u == 1 || body_load_count_in(kernel.body(), &loop_body) > 0);
            let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
            assert!(ptx.contains("_lh"));
            assert!(ptx.contains("ld.global"));
            assert_eq!(ptx.matches("mul.rn.f32").count(), 2);
        }
    }
}
/// The kernel is assembled as prologue, then what the hoist lifted, then the
/// `u` loop holding the trip. Swapping the two, or putting both in one place,
/// leaves a kernel that still builds and still renders — only the shape says
/// which statements the loop repeats.
#[test]
fn what_the_hoist_lifted_precedes_the_u_loop_that_holds_the_trip() {
    let loom = lane_hoist_program(64, false, false, true);
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("lane-hoist launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            lane_hoist: true,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();

    let loops = hidden_axis_loops(kernel.body());
    assert_eq!(loops, 1, "the trip is one loop, not none and not two");
    let prefix = u_loop_prefix(kernel.body());
    let trip = u_loop_body(kernel.body());
    assert!(!trip.is_empty(), "the trip is what the loop repeats");
    assert!(
        loom_value_bindings_in(kernel.body(), &prefix) > 0,
        "the lifted stage is before the loop"
    );
    assert!(
        body_load_count_in(kernel.body(), &prefix) > 0,
        "the lifted stage is the one whose loads were lifted"
    );
    // What the loop binds cannot be read before it, so the trip statements
    // reach for the lane index and the lifted ones cannot.
    let f = kernel.body();
    let index = f
        .block(f.region_block(f.op(u_loop(f, &body(f))).regions[0]))
        .params[0];
    let reaches = |ops: &[gpu_ir::OpId]| {
        let mut read = std::collections::BTreeSet::new();
        for &op in ops {
            f.add_op_reads(op, &mut read);
        }
        read.contains(&index)
    };
    assert!(
        !reaches(&prefix),
        "nothing before the loop reads the index the loop binds"
    );
    assert!(reaches(&trip), "the trip is what varies with that index");
}

#[test]
fn lane_hoist_moves_atomic_output_bases_before_u() {
    let mut loom = lane_hoist_program(128, false, true, true);
    loom.domains[1].extent = 1;
    loom.index_maps[0].may_have_unmapped = true;
    let plan = baseline_plan(&loom);
    assert!(plan.domain.index.fiber_bridge().is_none());
    assert!(plan.domain.folds.output(0).combines_atomically());
    let candidate = &plan.candidates[0];
    assert_eq!(candidate.hoists, vec![true]);
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("lane-hoist launch is not a consumer");
    };

    let baseline = emit_planned_kernel(&plan.domain, &loom, candidate, 0, BLOCK_SIZES[0]).unwrap();
    assert!(!has_let_named_in(
        baseline.body(),
        &u_loop_prefix(baseline.body()),
        "loom_output_base_"
    ));
    assert!(has_let_named_in(
        baseline.body(),
        &u_loop_body(baseline.body()),
        "loom_output_base_"
    ));
    assert!(has_let_named_in(
        baseline.body(),
        &u_loop_body(baseline.body()),
        "loom_output_alive_"
    ));

    let hoisted = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            lane_hoist: true,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    let prefix = u_loop_prefix(hoisted.body());
    let loop_body = u_loop_body(hoisted.body());
    assert!(has_let_named_in(
        hoisted.body(),
        &prefix,
        "loom_output_base_"
    ));
    assert!(has_let_named_in(
        hoisted.body(),
        &prefix,
        "loom_output_alive_"
    ));
    assert!(!has_let_named_in(
        hoisted.body(),
        &loop_body,
        "loom_output_base_"
    ));
    assert!(!has_let_named_in(
        hoisted.body(),
        &loop_body,
        "loom_output_alive_"
    ));
    assert!(has_red_add_in(hoisted.body(), &loop_body));

    let baseline_ptx = gpu_ir::ptx::render_ptx_module(std::slice::from_ref(&baseline), 80);
    let ptx = gpu_ir::ptx::render_ptx_module(std::slice::from_ref(&hoisted), 80);
    assert_target_indices_load_is_outside_ptx_u_loop(&hoisted, &ptx);
    assert!(ptx_u_loop(&baseline_ptx).contains("mul.lo.s64"));
    assert!(!ptx_u_loop(&ptx).contains("mul.lo.s64"));
}
#[test]
fn mixed_lane_hoist_keeps_scalar_updates_ordered_inside_u() {
    let loom = mixed_lane_hoist_program();
    let plan = baseline_plan(&loom);
    assert!(plan.domain.index.fiber_bridge().is_none());
    assert!(plan.domain.folds.output(1).combines_atomically());
    let candidate = &plan.candidates[0];
    assert_eq!(candidate.hoists, vec![true]);
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("lane-hoist launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            lane_hoist: true,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    let prefix = u_loop_prefix(kernel.body());
    let loop_body = u_loop_body(kernel.body());
    assert_eq!(
        let_count_named_in(kernel.body(), &prefix, "loom_input_base_"),
        2
    );
    assert_eq!(
        let_count_named_in(kernel.body(), &prefix, "loom_output_base_"),
        1
    );
    assert!(has_let_named_in(
        kernel.body(),
        &prefix,
        "loom_output_base_1"
    ));
    assert!(has_let_named_in(
        kernel.body(),
        &prefix,
        "loom_output_alive_1"
    ));
    assert!(!has_let_named_in(
        kernel.body(),
        &loop_body,
        "loom_input_base_"
    ));
    assert!(!has_let_named_in(
        kernel.body(),
        &loop_body,
        "loom_output_base_"
    ));
    assert!(!has_let_named_in(
        kernel.body(),
        &loop_body,
        "loom_output_alive_"
    ));
    assert!(has_red_add_in(kernel.body(), &loop_body));

    // The accumulation is a binding produced into the accumulator it
    // updates, so it is found by the place it takes over rather than by a
    // name it writes.
    let f = kernel.body();
    let (fold_index, folded) = loop_body
        .iter()
        .enumerate()
        .find_map(|(index, &op)| {
            let place = f.op(op).destination(0)?;
            f.name(place)
                .is_some_and(|name| name.starts_with("loom_out_acc_0_"))
                .then(|| (index, f.op(op).operands[0]))
        })
        .expect("scalar output has no U-loop accumulation");
    let varying = f
        .reads(folded)
        .into_iter()
        .filter(|&value| f.name(value).is_some_and(|name| name.starts_with("loom_v")))
        .collect::<Vec<_>>();
    assert!(!varying.is_empty());
    for value in varying {
        let binding_index = loop_body
            .iter()
            .position(|&op| matches!(f.op(op).kind, OpKind::Bind) && f.op(op).results[0] == value)
            .expect("scalar accumulation depends on a missing varying binding");
        assert!(binding_index < fold_index);
    }

    let ptx = gpu_ir::ptx::render_ptx_module(std::slice::from_ref(&kernel), 80);
    assert_target_indices_load_is_outside_ptx_u_loop(&kernel, &ptx);
    assert!(!ptx_u_loop(&ptx).contains("mul.lo.s64"));
}
#[test]
fn lane_hoist_is_not_planned_for_fiber_iteration() {
    let loom = lane_hoist_program(8, false, true, true);
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    assert!(candidate.hoists.iter().all(|&hoist| !hoist));
}
#[test]
fn direct_width_one_input_has_no_lane_hoist_candidate() {
    let loom = lane_hoist_program(8, false, false, false);
    let plan = baseline_plan(&loom);
    assert!(plan.candidates[0].hoists.iter().all(|&hoist| !hoist));
}
