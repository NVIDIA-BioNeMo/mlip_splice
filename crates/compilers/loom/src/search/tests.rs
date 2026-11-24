// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{Domain, IndexMap};
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::shape::Shape;

use crate::{CompileOptions, FlatOperand, FlatProgram};

fn request() -> FlatProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("first_plan");
    let x = builder.param(Shape::new(DType::F32, vec![4]));
    let w = builder.param(Shape::new(DType::F32, vec![4]));
    let product = builder.multiply(x, w);
    let squared = builder.multiply(product, product);
    let entry = builder.end_func(vec![product, squared]);
    let binding = |domain| FlatOperand {
        dtype: DType::F32,
        element_shape: vec![4],
        lane_extent: 32,
        axis_domains: vec![domain],
    };
    FlatProgram {
        name: "first_plan".into(),
        module: builder.finish(entry).unwrap(),
        lane_count: 32,
        domains: vec![Domain { extent: 16 }, Domain { extent: 4 }],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![binding(0), binding(1), binding(0), binding(1)],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

#[test]
fn tiny_budget_publishes_the_first_optimized_frontier_configuration() {
    let (compiled, profile) = super::compile_flat(
        request(),
        CompileOptions {
            search_budget: Some(std::time::Duration::from_nanos(1)),
            shortlist_budget: None,
            target: Some(crate::GpuTarget {
                sm: 100,
                ..crate::backend::compiler::test_target()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(compiled.output_count, 2);
    assert!(!profile.kernels.is_empty());
    assert_eq!(profile.candidate_elapsed.len(), 1);
    assert_eq!(profile.arms.len(), 1);
    assert!(!profile.planned_candidates.is_empty());
    assert_eq!(profile.frontier_graphs.len(), 1);
    assert_eq!(profile.frontier_graphs[0].graph, 0);
    assert_eq!(profile.frontier_graphs[0].recipe_arm, Some(0));
    assert!(!profile.element_orders.is_empty());
    let options = first_options();
    let (mut frontier, _) =
        crate::frontier::ProgramFrontier::extract(request(), options, &mut || Ok(())).unwrap();
    let (program, _) = frontier.next(&mut || Ok(())).unwrap().unwrap();
    let mut passes = crate::backend::compiler::GraphPasses::new(program, options).unwrap();
    assert_eq!(profile.frontier_graphs[0].frontier_size, frontier.len() + 1);
    let (candidate, expected) = passes.next(&mut || Ok(())).unwrap();
    assert_complete(candidate.unwrap().recipe.body());
    assert_eq!(
        profile
            .kernels
            .iter()
            .map(|unit| &unit.ptx_hash)
            .collect::<Vec<_>>(),
        expected
            .kernels
            .iter()
            .map(|unit| &unit.ptx_hash)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        profile.arms[0].plan_clusters,
        expected.arms[0].plan_clusters
    );
    assert_eq!(
        compiled.input_count,
        2 + compiled
            .index_arrays
            .iter()
            .map(|slot| slot.kind.array_count())
            .sum::<usize>()
    );
}

#[test]
fn direct_f32_scatter_needs_no_workspace() {
    let (compiled, _) = super::compile_flat(
        request(),
        CompileOptions {
            search_budget: Some(std::time::Duration::from_nanos(1)),
            shortlist_budget: None,
            max_workspace_bytes: Some(0),
            target: Some(crate::GpuTarget {
                sm: 100,
                ..crate::backend::compiler::test_target()
            }),
            ..Default::default()
        },
    )
    .expect("direct F32 scatter needs no scratch");
    assert_eq!(
        ::recipe::algebra::workspace_upper_bound(&compiled.recipe),
        0
    );
}

fn projection_request(dtype: DType) -> FlatProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("projection");
    let coefficient = builder.param(Shape::scalar(dtype));
    let values = builder.param(Shape::scalar(dtype));
    let doubled = builder.add(coefficient, coefficient);
    let product = builder.multiply(doubled, values);
    let negated = builder.op(xir::ops::Negate, &[product])[0];
    let entry = builder.end_func(vec![negated]);
    let binding = |domain, lanes| FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: lanes,
        axis_domains: vec![domain],
    };
    FlatProgram {
        module: builder.finish(entry).unwrap(),
        lane_count: 4,
        domains: vec![Domain { extent: 3 }, Domain { extent: 8 }],
        index_maps: vec![IndexMap {
            source_id: 1,
            target_id: 0,
            is_sorted: false,
            is_injective: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![binding(0, 1), binding(1, 4), binding(0, 1)],
        num_inputs: 2,
        available_views: Vec::new(),
        name: "projection".into(),
    }
}

fn first_options() -> CompileOptions {
    CompileOptions {
        search_budget: Some(std::time::Duration::from_nanos(1)),
        shortlist_budget: None,
        target: Some(crate::GpuTarget {
            sm: 100,
            ..crate::backend::compiler::test_target()
        }),
        ..Default::default()
    }
}

#[test]
fn first_plan_lowers_add_negate_and_nested_domain_lane_sums() {
    let (compiled, profile) =
        super::compile_flat(projection_request(DType::F32), first_options()).unwrap();
    assert_eq!(compiled.output_count, 1);
    assert_eq!(profile.arms.len(), 1);
    assert!(profile.fold_buckets.iter().all(|fold| fold.graph == 0));
}

#[test]
fn first_plan_enforces_conversion_workspace() {
    let mut request = projection_request(DType::F32);
    request.operands.last_mut().unwrap().dtype = DType::F16;
    let (compiled, _) = super::compile_flat(request.clone(), first_options()).unwrap();
    let workspace = ::recipe::algebra::workspace_upper_bound(&compiled.recipe);
    assert!(workspace > 0);
    let error = super::compile_flat(
        request.clone(),
        CompileOptions {
            max_workspace_bytes: Some(workspace - 1),
            ..first_options()
        },
    )
    .err()
    .expect("conversion needs workspace");
    assert!(error.contains("workspace"), "{error}");
    super::compile_flat(
        request,
        CompileOptions {
            max_workspace_bytes: Some(workspace),
            ..first_options()
        },
    )
    .unwrap();
}

fn assert_complete(body: &::recipe::RecipeBody) {
    for node in body.nodes() {
        if let ::recipe::Work::Choice { candidates, .. } = node.work() {
            assert_eq!(
                candidates.len(),
                1,
                "an emitted candidate contains an unresolved choice"
            );
            assert_complete(&candidates[0]);
        }
    }
}

#[test]
fn optional_search_keeps_complete_candidates_and_unique_graph_records() {
    let (_, first) = super::compile_flat(request(), first_options()).unwrap();
    let mut saved = Vec::new();
    let mut observed = Vec::new();
    let profile = super::stream_flat_candidates(
        request(),
        CompileOptions {
            search_budget: None,
            ..first_options()
        },
        |candidate| {
            assert_complete(candidate.recipe.body());
            saved.push(candidate.clone());
            Ok(())
        },
        |(_, elapsed)| {
            observed.push(elapsed);
            if observed.len() == 12 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        },
    )
    .unwrap();
    assert_eq!(saved.len(), 12);
    assert_eq!(profile.candidate_elapsed, observed);
    assert!(observed.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(profile.kernels[0].ptx_hash, first.kernels[0].ptx_hash);
    assert_eq!(profile.frontier_graphs[0].graph, 0);
    assert!(profile.kernels.iter().all(|unit| {
        profile
            .frontier_graphs
            .iter()
            .any(|graph| graph.graph == unit.graph)
    }));
    let distinct = profile
        .frontier_graphs
        .iter()
        .map(|record| record.graph)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(distinct.len(), profile.frontier_graphs.len());
    assert!(saved.iter().all(|candidate| candidate.output_count == 2));
}

#[test]
fn observer_stops_immediately_after_first_successful_save() {
    let saved = std::cell::Cell::new(false);
    let profile = super::stream_flat_candidates(
        request(),
        CompileOptions {
            search_budget: None,
            ..first_options()
        },
        |candidate| {
            assert_complete(candidate.recipe.body());
            saved.set(true);
            Ok(())
        },
        |(_, _)| {
            assert!(saved.get());
            std::ops::ControlFlow::Break(())
        },
    )
    .unwrap();
    assert_eq!(profile.candidate_elapsed.len(), 1);
    assert_eq!(profile.frontier_graphs.len(), 1);
    assert_eq!(profile.frontier_graphs[0].recipe_arm, Some(0));
}

#[test]
fn a_failed_save_is_never_observed() {
    let result = super::stream_flat_candidates(
        request(),
        first_options(),
        |_| Err("save refused".into()),
        |_| panic!("failed save was observed"),
    );
    assert_eq!(result.unwrap_err(), "save refused");
}

#[test]
fn cumulative_clock_includes_preparation_refusals_and_save() {
    use std::{cell::Cell, time::Duration};
    let clock = Cell::new(0u64);
    let mut ledger = super::Publications::default();
    clock.set(7); // Preparation and the mandatory source attempt.
    let mut save = |_: &()| {
        clock.set(clock.get() + 3);
        Ok(())
    };
    let first = ledger
        .commit(&(), &mut save, || Duration::from_millis(clock.get()))
        .unwrap();
    clock.set(clock.get() + 11); // A refused optional attempt.
    clock.set(clock.get() + 5); // The next viable attempt.
    let next = ledger
        .commit(&(), &mut save, || Duration::from_millis(clock.get()))
        .unwrap();
    assert_eq!(
        (first.as_millis(), next.as_millis(), ledger.count),
        (10, 29, 2)
    );
    assert!(
        ledger
            .commit(&(), &mut |_| Err("save refused".into()), || panic!(
                "failed commit clock"
            ))
            .is_err()
    );
    assert_eq!(ledger.count, 2);
}

#[test]
fn expired_horizon_does_not_open_the_frontier() {
    let budget = super::SourceBudget {
        started: std::time::Instant::now() - std::time::Duration::from_secs(1),
        limit_ms: Some(1.0),
    };
    assert!(budget.expired());
    assert_eq!(budget.remaining_ms(), Some(0.0));
    assert!(
        crate::frontier::ProgramFrontier::extract(request(), first_options(), &mut || budget
            .checkpoint())
        .is_err()
    );
}

#[test]
fn cooperative_stop_preserves_the_saved_prefix() {
    let mut saved = Vec::new();
    let (first, _) = super::compile_flat(request(), first_options()).unwrap();
    saved.push(first.clone());
    let options = CompileOptions {
        search_budget: None,
        ..first_options()
    };
    let (mut frontier, _) =
        crate::frontier::ProgramFrontier::extract(request(), options, &mut || Ok(())).unwrap();
    let (program, _) = frontier.next(&mut || Ok(())).unwrap().unwrap();
    let mut passes = crate::backend::compiler::GraphPasses::new(program, options).unwrap();
    let mut checkpoints = 0;
    let (result, _) = passes
        .next(&mut || {
            checkpoints += 1;
            if checkpoints >= 5 {
                Err("stopped".into())
            } else {
                Ok(())
            }
        })
        .unwrap();
    assert_eq!(result.err().as_deref(), Some("stopped"));
    assert_eq!(saved.len(), 1);
    assert_complete(saved[0].recipe.body());
    assert_eq!(saved[0].input_count, first.input_count);
}

#[test]
fn expiry_is_ignored_through_refusals_until_a_successful_publication() {
    let budget = super::SourceBudget {
        started: std::time::Instant::now(),
        limit_ms: Some(f64::MIN_POSITIVE),
    };
    assert!(budget.expired());
    let mut ledger = super::Publications::default();
    let mut saved = Vec::new();
    for attempt in [Err("first plan refused"), Err("geometry refused"), Ok(3)] {
        assert!(ledger.checkpoint(budget).is_ok());
        if let Ok(candidate) = attempt {
            ledger
                .commit(
                    &candidate,
                    &mut |candidate| {
                        saved.push(*candidate);
                        Ok(())
                    },
                    || budget.started.elapsed(),
                )
                .unwrap();
        }
    }
    assert_eq!(saved, [3]);
    assert_eq!(ledger.count, 1);
    assert!(ledger.checkpoint(budget).is_err());
}

#[test]
fn failed_saving_does_not_enable_optional_expiry() {
    let budget = super::SourceBudget {
        started: std::time::Instant::now(),
        limit_ms: Some(f64::MIN_POSITIVE),
    };
    let mut ledger = super::Publications::default();
    let error = ledger
        .commit(&(), &mut |_| Err("save failed".into()), || {
            panic!("no timestamp before successful save")
        })
        .unwrap_err();
    assert_eq!(error, "save failed");
    assert_eq!(ledger.count, 0);
    assert!(ledger.checkpoint(budget).is_ok());
}

#[test]
fn explicit_graph_selection_uses_the_optimized_retained_frontier() {
    let options = CompileOptions {
        only_frontier_graph: Some(0),
        ..first_options()
    };
    let (programs, _, _) =
        crate::frontier::lower_program_frontier_with_options(request(), options).unwrap();
    let (_, count) = crate::backend::compiler::select_graph(programs.clone(), options).unwrap();
    for graph in 0..count {
        let options = CompileOptions {
            only_frontier_graph: Some(graph),
            ..options
        };
        let (program, _) =
            crate::backend::compiler::select_graph(programs.clone(), options).unwrap();
        let mut passes = crate::backend::compiler::GraphPasses::new(program, options).unwrap();
        let mut record = passes.graph_record(graph, count);
        record.recipe_arm = Some(0);
        let (expected, expected_profile) = passes.next(&mut || Ok(())).unwrap();
        assert_complete(expected.unwrap().recipe.body());
        let (_, profile) = super::compile_flat(request(), options).unwrap();
        assert_eq!(profile.frontier_graphs.len(), 1);
        assert_eq!(profile.frontier_graphs[0].graph, graph);
        assert_eq!(profile.frontier_graphs[0].frontier_size, count);
        assert_eq!(profile.frontier_graphs[0].recipe_arm, record.recipe_arm);
        assert_eq!(
            profile.frontier_graphs[0].register_pressure,
            record.register_pressure
        );
        assert_eq!(profile.candidate_elapsed.len(), 1);
        assert!(!profile.element_orders.is_empty());
        assert!(
            profile
                .element_orders
                .iter()
                .all(|order| order.graph == graph)
        );
        assert!(profile.kernels.iter().all(|unit| unit.graph == graph));
        assert_eq!(
            profile
                .kernels
                .iter()
                .map(|unit| &unit.ptx_hash)
                .collect::<Vec<_>>(),
            expected_profile
                .kernels
                .iter()
                .map(|unit| &unit.ptx_hash)
                .collect::<Vec<_>>()
        );
    }
    let error = super::stream_flat_candidates(
        request(),
        CompileOptions {
            only_frontier_graph: Some(count),
            ..options
        },
        |_| panic!("invalid graph was saved"),
        |_| panic!("invalid graph was observed"),
    )
    .unwrap_err();
    assert!(error.contains("frontier_graph"), "{error}");
}

#[test]
fn exhausted_optimized_configurations_return_an_error_without_publication() {
    let mut request = projection_request(DType::F32);
    request.operands.last_mut().unwrap().dtype = DType::F16;
    let error = super::stream_flat_candidates(
        request,
        CompileOptions {
            max_workspace_bytes: Some(0),
            ..first_options()
        },
        |_| panic!("refused optimized source was saved"),
        |_| panic!("refused optimized source was observed"),
    )
    .unwrap_err();
    assert!(
        error.contains("no viable optimized source configuration"),
        "{error}"
    );
    assert!(error.contains("workspace"), "{error}");
}

#[test]
fn frontier_preparation_failure_returns_without_publication() {
    let mut request = request();
    request.num_inputs = request.operands.len() + 1;
    let error = super::stream_flat_candidates(
        request,
        first_options(),
        |_| panic!("failed frontier produced a source"),
        |_| panic!("failed frontier was observed"),
    )
    .unwrap_err();
    assert!(error.contains("input"), "{error}");
}

#[test]
fn shortlist_budget_does_not_prune_source_capture() {
    let options = CompileOptions {
        search_budget: None,
        shortlist_budget: Some(std::time::Duration::ZERO),
        target: Some(crate::GpuTarget {
            sm: 100,
            sm_count: 148,
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
        }),
        ..first_options()
    };
    let mut sources = Vec::new();
    let mut emitted = 0;
    let mut profile = super::stream_flat_candidates(
        request(),
        options,
        |source| {
            sources.push(source.clone());
            Ok(())
        },
        |_| {
            emitted += 1;
            if emitted == 12 {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        },
    )
    .unwrap();
    assert_eq!(sources.len(), 12);
    assert_eq!(profile.arms.len(), 12);
    assert_eq!(profile.candidate_elapsed.len(), 12);
    assert!(profile.shortlisted_candidates.is_empty());
    let original = sources
        .iter()
        .map(|source| ::recipe::serialize(&source.recipe))
        .collect::<Vec<_>>();
    let selected = crate::shortlist::select(&sources, &mut profile, options).unwrap();
    assert!(!selected.is_empty());
    assert!(selected.len() < sources.len());
    assert_eq!(profile.arms.len(), 12);
    assert!(profile.shortlist_estimated_ms.unwrap() > 0.0);
    let shortlist = selected
        .iter()
        .map(|&index| sources[index].clone())
        .collect::<Vec<_>>();
    let combined = crate::backend::compiler::combine_compiled(shortlist, 2).unwrap();
    combined.recipe.validate().unwrap();
    if selected.len() > 1 {
        let ::recipe::Work::Choice { candidates, .. } = combined.recipe.body().nodes()[0].work()
        else {
            panic!("missing shortlist choice")
        };
        assert_eq!(candidates.len(), selected.len());
    }
    assert_eq!(
        original,
        sources
            .iter()
            .map(|source| ::recipe::serialize(&source.recipe))
            .collect::<Vec<_>>()
    );
}

#[test]
fn production_handoff_records_a_frozen_shortlist_without_assembly() {
    let (compiled, profile) = super::compile_flat(
        request(),
        CompileOptions {
            shortlist_budget: Some(std::time::Duration::ZERO),
            target: Some(crate::GpuTarget {
                sm: 100,
                sm_count: 148,
                warp_size: 32,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
            }),
            ..first_options()
        },
    )
    .unwrap();
    assert_eq!(profile.shortlisted_candidates, [0]);
    assert_eq!(profile.candidate_elapsed.len(), 1);
    assert!(profile.shortlist_compile_units.unwrap() > 0);
    assert!(profile.shortlist_estimated_ms.unwrap() > 0.0);
    assert_complete(compiled.recipe.body());
    assert!(!compiled.recipe.context().is_empty());
}

#[test]
fn memory_budget_stops_before_the_first_plan() {
    for frontier_graph in [None, Some(0)] {
        let error = super::stream_flat_candidates(
            request(),
            CompileOptions {
                search_size_estimate_limit: Some(0),
                only_frontier_graph: frontier_graph,
                ..first_options()
            },
            |_| panic!("no publication with a zero memory budget"),
            |_| panic!("no emission with a zero memory budget"),
        )
        .unwrap_err();
        assert!(error.contains("search memory budget exhausted"), "{error}");
    }
}

#[test]
fn disabled_memory_budget_preserves_the_first_plan_after_time_expiry() {
    let (compiled, profile) = super::compile_flat(
        request(),
        CompileOptions {
            search_size_estimate_limit: None,
            ..first_options()
        },
    )
    .unwrap();
    assert_complete(compiled.recipe.body());
    assert_eq!(profile.candidate_elapsed.len(), 1);
    assert_eq!(
        profile.stop_reason,
        Some(crate::profile::StopReason::SearchBudget)
    );
    assert!(profile.search_size_estimate >= 16 * 1024);
    assert_eq!(
        CompileOptions::default().search_size_estimate_limit,
        Some(16 << 30)
    );
}

#[test]
fn estimated_memory_limit_keeps_the_complete_plan_that_reaches_it() {
    let options = CompileOptions {
        search_budget: None,
        search_size_estimate_limit: None,
        ..first_options()
    };
    let mut reference = Vec::new();
    let first = super::stream_flat_candidates(
        request(),
        options,
        |candidate| {
            reference.push(::recipe::serialize(&candidate.recipe).unwrap());
            Ok(())
        },
        |_| std::ops::ControlFlow::Break(()),
    )
    .unwrap();
    let limit = first.search_size_estimate;
    assert!(limit > 16 * 1024);
    assert_eq!(
        first.stop_reason,
        Some(crate::profile::StopReason::CallerStopped)
    );
    for _ in 0..2 {
        let mut saved = Vec::new();
        let profile = super::stream_flat_candidates(
            request(),
            CompileOptions {
                search_size_estimate_limit: Some(limit),
                ..options
            },
            |candidate| {
                candidate.recipe.validate().unwrap();
                saved.push(::recipe::serialize(&candidate.recipe).unwrap());
                Ok(())
            },
            |_| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
        assert_eq!(saved, reference);
        assert_eq!(
            profile.stop_reason,
            Some(crate::profile::StopReason::SearchSizeEstimateLimit)
        );
        assert_eq!(profile.search_size_estimate, limit);
        assert_eq!(profile.candidate_elapsed.len(), 1);
    }
}

#[test]
fn estimated_memory_limit_finalizes_the_saved_prefix() {
    let (compiled, profile) = super::compile_flat(
        request(),
        CompileOptions {
            search_budget: None,
            search_size_estimate_limit: Some(1),
            ..first_options()
        },
    )
    .unwrap();
    compiled.recipe.validate().unwrap();
    assert_eq!(profile.candidate_elapsed.len(), 1);
    assert_eq!(
        profile.stop_reason,
        Some(crate::profile::StopReason::SearchSizeEstimateLimit)
    );
    assert_eq!(profile.shortlisted_candidates, [0]);
}
