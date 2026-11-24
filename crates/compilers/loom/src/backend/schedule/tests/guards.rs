// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn guarded_consumer_contracts_multiply_add_inside_the_guard() {
    let loom = pointwise_multiply_add(true);
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let guard = candidate.guards[0].as_ref().unwrap();
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("guarded launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: Some(guard),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("redux.sync.or.b32"));
}
#[test]
fn exact_zero_select_guards_both_predicate_polarities() {
    for (zero_on_false, negated) in [(true, false), (false, true)] {
        let loom = guarded_select_program(8, zero_on_false, 1, false, false, false);
        let plan = baseline_plan(&loom);
        let guard = plan.candidates[0].guards[0].as_ref().unwrap();
        assert_eq!(guard.regions[0].outputs, vec![(0, 0)]);
        // A part travels between plans, so what it carries about a guard is the
        // key — which value, which element, and whether it is the inverse.
        assert_eq!(
            guard.regions[0].key.inverted, negated,
            "wrong predicate polarity"
        );
    }
}
#[test]
fn one_predicate_groups_multiple_guarded_outputs() {
    let loom = guarded_select_program(8, true, 2, false, false, false);
    let plan = baseline_plan(&loom);
    let guards = &plan.candidates[0].guards;
    assert_eq!(guards.len(), 1);
    assert_eq!(
        guards[0].as_ref().unwrap().regions[0].outputs,
        vec![(0, 0), (1, 0)]
    );
}
#[test]
fn fiber_zero_skip_groups_additive_terms_by_guard_key() {
    let loom = multi_guard_fiber_program();
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let guarded = candidate.guards[0].as_ref().unwrap();
    assert_eq!(guarded.regions.len(), 2);
    assert_eq!(guarded.remainder.len(), 0);
    assert!(guarded.regions.iter().all(|region| region.part.len() == 1));
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("multi-guard launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: Some(guarded),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    let baseline = emit_planned_kernel(&plan.domain, &loom, candidate, 0, BLOCK_SIZES[0]).unwrap();
    assert_eq!(warp_guard_count(kernel.body()), 2);
    assert_eq!(warp_guard_count(baseline.body()), 0);
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    let baseline_ptx = gpu_ir::ptx::render_ptx_module(&[baseline], 80);
    assert_eq!(ptx.matches("redux.sync.or.b32").count(), 2);
    assert!(
        ptx.matches(" bra ").count() >= baseline_ptx.matches(" bra ").count() + 2,
        "each regional vote should control its own PTX branch"
    );
    assert_eq!(ptx.matches("st.global").count(), 1);
    assert!(!ptx.contains("loom_profile_input"));
}
#[test]
fn a_fiber_zero_skip_guard_folds_in_the_accumulator_register() {
    let loom = multi_guard_fiber_program();
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("multi-guard launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: candidate.guards[0].as_ref(),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    // Each guard produces its accumulator into the storage it came from, and
    // the accumulator is what the trip carries, so the fold runs in that
    // register and the join copies nothing. The one float move the kernel holds
    // is the accumulator's zero.
    let moved = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("mov.f32"))
        .collect::<Vec<_>>();
    assert_eq!(moved, vec!["mov.f32 %f1, 0f00000000;"], "{ptx}");
    assert_eq!(ptx.matches("redux.sync.or.b32").count(), 2, "{ptx}");
}

#[test]
fn multi_guard_atomic_scatter_keeps_the_baseline_plan() {
    let mut loom = multi_guard_fiber_program();
    loom.domains[1].extent = 1;
    let plan = baseline_plan(&loom);
    assert!(plan.domain.index.fiber_bridge().is_none());
    assert!(plan.domain.folds.output(0).combines_atomically());
    assert!(plan.candidates[0].guards[0].is_none());
}
/// A zero-skip guard replaces the untaken arm with nothing, which is exact over
/// the algebraic numbers: zero times anything is zero, so the zero propagates
/// through a downstream multiply and the region stays a skip. IEEE-754 disagrees
/// only where the other factor is an infinity or a NaN, which is the reassociation
/// Loom performs and documents rather than an accident of this plan.
#[test]
fn a_zero_arm_stays_a_skip_through_a_downstream_multiply() {
    let loom = guarded_select_program(8, true, 1, false, true, false);
    let plan = baseline_plan(&loom);
    assert!(plan.candidates[0].guards[0].is_some());
}
#[test]
fn algebraic_accepts_multiply_by_zero_as_an_exact_zero_arm() {
    let loom = guarded_select_program(8, true, 1, true, false, false);
    let plan = baseline_plan(&loom);
    assert!(plan.candidates[0].guards[0].is_some());
}
#[test]
fn exact_zero_guards_cover_all_supported_u_and_fiber_plans() {
    for u in [1, 5, 32, 64] {
        for fiber in [false, true] {
            let loom = guarded_select_program(u, true, 1, false, false, fiber);
            let plan = baseline_plan(&loom);
            assert!(plan.candidates[0].guards.iter().any(Option::is_some));
        }
    }
}
#[test]
fn exact_zero_guard_handles_width_one_reduction_outputs() {
    for u in [1, 5, 32, 64] {
        let mut loom = guarded_select_program(u, true, 1, false, false, false);
        loom.operands.last_mut().unwrap().lane_extent = 1;
        let plan = baseline_plan(&loom);
        let candidate = &plan.candidates[0];
        let guard = candidate.guards[0].as_ref().unwrap();
        let Launch::Consumer { part, .. } = &candidate.launches[0] else {
            panic!("guarded width-one launch is not a consumer");
        };
        let kernel = emit_consumer_kernel_flavor(
            &plan.domain,
            &loom,
            part,
            Flavor {
                materialized: &candidate.materialized,
                guard: Some(guard),
                io_lanes: 1,
                ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
            },
        )
        .map(|consumer| consumer.kernel)
        .unwrap();
        assert!(has_zero_initialized_output_accumulator(kernel.body()));
        assert!(guarded_else_is_empty(kernel.body()));
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
        assert!(ptx.contains("redux.sync.or.b32"));
        assert!(ptx.contains("st.global"));
    }
}
#[test]
fn zero_skip_ptx_votes_branches_and_directly_stores_all_zero() {
    let loom = guarded_select_program(8, true, 1, false, false, false);
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let guard = candidate.guards[0].as_ref().unwrap();
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("guarded baseline launch is not a consumer");
    };
    let baseline = emit_planned_kernel(&plan.domain, &loom, candidate, 0, BLOCK_SIZES[0]).unwrap();
    let skip = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: Some(guard),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    assert!(has_inactive_direct_zero_store(skip.body()));
    let baseline_ptx = gpu_ir::ptx::render_ptx_module(&[baseline], 80);
    let skip_ptx = gpu_ir::ptx::render_ptx_module(&[skip], 80);
    assert!(skip_ptx.contains("_zs"));
    assert!(!baseline_ptx.contains("activemask.b32"));
    assert!(!baseline_ptx.contains("redux.sync.or.b32"));
    assert!(skip_ptx.contains("activemask.b32"));
    assert!(skip_ptx.contains("redux.sync.or.b32"));
    assert!(skip_ptx.contains(" bra ") || skip_ptx.contains("bra.uni"));
}
#[test]
fn zero_skip_keeps_expensive_operand_loads_behind_the_warp_guard() {
    let loom = guarded_select_program(8, true, 1, false, false, false);
    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let guard = candidate.guards[0].as_ref().unwrap();
    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("guarded baseline launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: Some(guard),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();

    assert_eq!(unprotected_load_count(kernel.body()), 1);
    assert!(guarded_then_load_count(kernel.body()).is_some_and(|loads| loads >= 2));
}
#[test]
fn guarded_additive_continuations_skip_rmw_but_initial_store_writes_zero() {
    let loom = guarded_additive_program(80);
    let (_, plan) = (64..512)
        .find_map(|statement_cap| {
            let domain = DomainPlan::prepare(&loom).ok()?;
            let plan = PlanIr::build_bounded(
                &loom,
                domain,
                statement_cap,
                usize::MAX,
                0,
                &OnceCell::new(),
                true,
            )
            .ok()?;
            let candidate = &plan.candidates[0];
            (candidate.launches.len() > 1 && candidate.guards.iter().all(Option::is_some))
                .then_some((statement_cap, plan))
        })
        .expect("guarded additive output admits bounded continuations");
    let candidate = &plan.candidates[0];

    for (launch_index, launch) in candidate.launches.iter().enumerate() {
        let Launch::Consumer { part, output_modes } = launch else {
            panic!("guarded additive launch is not a consumer");
        };
        let kernel = emit_consumer_kernel_flavor(
            &plan.domain,
            &loom,
            part,
            Flavor {
                materialized: &candidate.materialized,
                guard: Some(candidate.guards[launch_index].as_ref().unwrap()),
                io_lanes: 1,
                ..Flavor::plain(BLOCK_SIZES[0], launch_index, candidate.launches.len())
            },
        )
        .map(|consumer| consumer.kernel)
        .unwrap();
        if launch_index == 0 {
            assert_eq!(output_modes, &[WriteMode::Store]);
            assert!(has_inactive_direct_zero_store(kernel.body()));
        } else {
            assert_eq!(output_modes, &[WriteMode::AddOrdered]);
            assert!(guarded_else_is_empty(kernel.body()));
        }
    }
}
#[test]
fn zero_skip_guard_is_inside_the_u_and_fiber_loops() {
    for u in [1, 5, 32, 64] {
        for (fiber, expected_depth) in [(false, 1), (true, 2)] {
            let loom = guarded_select_program(u, true, 1, false, false, fiber);
            let plan = baseline_plan(&loom);
            let candidate = &plan.candidates[0];
            let guard = candidate.guards[0].as_ref().unwrap();
            let Launch::Consumer { part, .. } = &candidate.launches[0] else {
                panic!("guarded baseline launch is not a consumer");
            };
            let kernel = emit_consumer_kernel_flavor(
                &plan.domain,
                &loom,
                part,
                Flavor {
                    materialized: &candidate.materialized,
                    guard: Some(guard),
                    io_lanes: 1,
                    ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
                },
            )
            .map(|consumer| consumer.kernel)
            .unwrap();
            assert_eq!(
                warp_guard_loop_depth(kernel.body(), 0),
                Some(expected_depth)
            );
        }
    }
}
