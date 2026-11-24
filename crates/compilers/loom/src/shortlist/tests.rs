// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::profile::{Arm, FrontierGraph};
use compilations::Compilations;
use recipe::{
    Blob, CompileRequest, CompileRequestId, PtxCompileOptions, PtxTarget, RecipeBuilder,
    RecipeSignature, ResourceBounds, SourceImage, SourceKind,
};

fn request(source: &str) -> CompileRequest {
    CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 7],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(source.as_bytes().to_vec()),
        }],
        PtxCompileOptions::default(),
    )
}

fn candidate(requests: Vec<CompileRequest>, launches: &[(usize, &str)]) -> CompiledProgram {
    let mut builder = RecipeBuilder::new(0, 0);
    let ids = requests
        .into_iter()
        .map(|request| builder.compile_request(request))
        .collect::<Vec<_>>();
    for &(request, entry) in launches {
        let module = builder.requested_module(ids[request], Vec::new(), ResourceBounds::default());
        builder.kernel(module, entry).add();
    }
    CompiledProgram {
        recipe: builder
            .finish(RecipeSignature::empty())
            .unwrap()
            .into_recipe(),
        input_count: 0,
        output_count: 0,
        index_arrays: Vec::new(),
    }
}

fn guarded(source: &str, prerequisite: usize) -> CompileRequest {
    let mut request = request(source);
    request.require(
        CompileRequestId(prerequisite),
        ResourceBounds {
            min_registers: Some(32),
            max_local_bytes: Some(0),
        },
    );
    request
}

/// The assembly lane alone, so a case can state exact unit arithmetic: one
/// worker, and no per-candidate tuning cost. Both are set explicitly, since
/// either calibrated default would fold a second lane into the arithmetic.
/// The budget rounds up to whole nanoseconds, the resolution of a `Duration`,
/// so a budget stated as exactly a candidate's cost still affords it.
fn options(budget: Option<f64>) -> CompileOptions {
    CompileOptions {
        shortlist_budget: budget
            .map(|ms| std::time::Duration::from_nanos((ms * 1_000_000.0).ceil() as u64)),
        tuning_cost_per_candidate: std::time::Duration::ZERO,
        assumed_assembly_workers: std::num::NonZeroUsize::new(1).unwrap(),
        target: Some(GpuTarget {
            sm: 100,
            sm_count: 148,
            warp_size: 32,
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
        }),
        ..Default::default()
    }
}

fn profile(work: &[u64]) -> CompileProfile {
    CompileProfile {
        arms: work
            .iter()
            .enumerate()
            .map(|(graph, _)| Arm {
                graph,
                order: "test",
                statement_cap: 32,
                plan_clusters: false,
            })
            .collect(),
        frontier_graphs: work
            .iter()
            .enumerate()
            .map(|(graph, &cost)| FrontierGraph {
                graph,
                frontier_size: work.len(),
                recipe_arm: Some(graph),
                before_accumulation: cost,
                after_accumulation: 0,
                estimated_statements: 0,
                register_pressure: 0,
                hoistable_computed_values: 0,
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn compilation_cost_deduplicates_shared_units_and_transitive_guards() {
    let candidates = vec![
        candidate(vec![request("base")], &[(0, "base")]),
        candidate(
            vec![request("base"), guarded("middle", 0)],
            &[(1, "middle")],
        ),
        candidate(
            vec![request("base"), guarded("middle", 0), guarded("top", 1)],
            &[(2, "top")],
        ),
    ];
    let table = Compilations::new(&candidates);
    let (closure, units) = table.closure(2, &[false; 3], &BTreeSet::new());
    assert_eq!(closure, [2, 0, 1]);
    assert_eq!(units.len(), 3);
    let expected = ["base", "middle", "top"]
        .iter()
        .map(|source| recipe::CompilationCostModel::default().request_cost_ms(&request(source)))
        .sum::<f64>();
    assert_eq!(table.additional_cost_ms(&units, &BTreeSet::new()), expected);
    assert_eq!(table.additional_cost_ms(&units, &units), 0.0);
}

#[test]
fn entry_witnesses_preserve_full_requests_without_duplicate_candidates() {
    let candidates = vec![
        candidate(vec![request("a and b")], &[(0, "a")]),
        candidate(vec![request("a and b")], &[(0, "b")]),
        candidate(vec![request("a and b")], &[(0, "b")]),
    ];
    let table = Compilations::new(&candidates);
    let (closure, units) = table.closure(2, &[false; 3], &BTreeSet::new());
    assert_eq!(closure, [2, 0]);
    assert_eq!(units.len(), 1);
    let mut covered = BTreeSet::new();
    table.cover_entries(0, &mut covered);
    table.cover_entries(2, &mut covered);
    let (closure, _) = table.closure(1, &[true, false, true], &covered);
    assert_eq!(closure, [1]);
}

#[test]
fn native_units_ignore_guard_identity_but_not_assembly_options() {
    let a = candidate(vec![request("base"), guarded("unit", 0)], &[(1, "entry")]);
    let b = candidate(vec![request("unit")], &[(0, "entry")]);
    let mut capped = request("unit");
    capped.options.max_registers = Some(64);
    let c = candidate(vec![capped], &[(0, "entry")]);
    let table = Compilations::new(&[a, b, c]);
    let (_, first) = table.closure(0, &[false; 3], &BTreeSet::new());
    let (_, second) = table.closure(1, &[false; 3], &BTreeSet::new());
    let (_, third) = table.closure(2, &[false; 3], &BTreeSet::new());
    assert_eq!(table.additional_cost_ms(&second, &first), 0.0);
    assert!(table.additional_cost_ms(&third, &first) > 0.0);
}

#[test]
fn first_ranked_candidate_may_exceed_limit_but_paid_units_are_free() {
    let candidates = (0..4)
        .map(|_| candidate(vec![request("same")], &[(0, "entry")]))
        .collect::<Vec<_>>();
    let mut report = profile(&[1, 2, 3, 4]);
    let selected = select(&candidates, &mut report, options(Some(0.0))).unwrap();
    assert_eq!(selected.len(), 4);
    assert_eq!(report.shortlist_compile_units, Some(1));
    assert_eq!(report.shortlisted_candidates, selected);
    assert!(report.shortlist_estimated_ms.unwrap() > 0.0);
}

#[test]
fn exact_budget_boundary_excludes_only_unaffordable_candidates() {
    let candidates = ["a", "b", "c"]
        .iter()
        .map(|source| candidate(vec![request(source)], &[(0, "entry")]))
        .collect::<Vec<_>>();
    let cost = recipe::CompilationCostModel::default().request_cost_ms(&request("a"));
    for (budget, count) in [
        (0.0, 1),
        (cost * 2.0 - 0.001, 1),
        (cost * 2.0, 2),
        (cost * 3.0, 3),
    ] {
        let mut report = profile(&[1, 2, 3]);
        assert_eq!(
            select(&candidates, &mut report, options(Some(budget)))
                .unwrap()
                .len(),
            count
        );
        assert_eq!(report.shortlist_compile_units, Some(count));
        assert_eq!(report.arms.len(), 3);
    }
}

#[test]
fn unlimited_selection_preserves_generation_order_without_device_queries() {
    let candidates = vec![
        candidate(vec![request("a")], &[(0, "entry")]),
        candidate(vec![request("b")], &[(0, "entry")]),
    ];
    let mut report = profile(&[10, 1]);
    assert_eq!(
        select(
            &candidates,
            &mut report,
            CompileOptions {
                shortlist_budget: None,
                ..Default::default()
            }
        )
        .unwrap(),
        [0, 1]
    );
}

#[test]
fn the_slower_lane_sets_the_predicted_wall_time() {
    assert_eq!(predicted_wall_ms(320.0, 12.0, 16), 20.0);
    assert_eq!(predicted_wall_ms(320.0, 100.0, 16), 100.0);
    assert_eq!(predicted_wall_ms(0.0, 0.0, 1), 0.0);
    assert_eq!(predicted_wall_ms(64.0, 0.0, 64), 1.0);
}

#[test]
fn the_tuning_cost_bounds_the_count_even_when_assembly_is_shared() {
    let candidates = (0..4)
        .map(|_| candidate(vec![request("same")], &[(0, "entry")]))
        .collect::<Vec<_>>();
    let unit = recipe::CompilationCostModel::default().request_cost_ms(&request("same"));
    for (budget, count) in [(0.0, 1), (unit, 1), (unit + 24.0, 3), (unit + 240.0, 4)] {
        let mut report = profile(&[1, 2, 3, 4]);
        let selected = select(
            &candidates,
            &mut report,
            CompileOptions {
                tuning_cost_per_candidate: std::time::Duration::from_millis(12),
                assumed_assembly_workers: std::num::NonZeroUsize::new(1).unwrap(),
                ..options(Some(budget))
            },
        )
        .unwrap();
        assert_eq!(selected.len(), count, "budget {budget}");
        assert_eq!(report.shortlist_compile_units, Some(1));
        assert_eq!(
            report.shortlist_estimated_ms,
            Some((12.0 * count as f64).max(unit))
        );
    }
}

#[test]
fn parallel_workers_make_a_shared_shortlist_affordable() {
    let candidates = ["a", "b", "c"]
        .iter()
        .map(|source| candidate(vec![request(source)], &[(0, "entry")]))
        .collect::<Vec<_>>();
    let unit = recipe::CompilationCostModel::default().request_cost_ms(&request("a"));
    for (workers, count) in [(1, 2), (2, 3), (4, 3)] {
        let mut report = profile(&[1, 2, 3]);
        let selected = select(
            &candidates,
            &mut report,
            CompileOptions {
                assumed_assembly_workers: std::num::NonZeroUsize::new(workers).unwrap(),
                ..options(Some(unit * 2.0))
            },
        )
        .unwrap();
        assert_eq!(selected.len(), count, "{workers} workers");
    }
}

/// A shortlist a recipe carries must be the same wherever it was selected, so
/// the default worker count is the calibrated constant and never this host's
/// cores.
#[test]
fn the_default_worker_count_does_not_read_the_host() {
    let candidates = ["a", "b", "c"]
        .iter()
        .map(|source| candidate(vec![request(source)], &[(0, "entry")]))
        .collect::<Vec<_>>();
    let unit = recipe::CompilationCostModel::default().request_cost_ms(&request("a"));
    let defaults = CompileOptions::default();
    assert_eq!(defaults.assumed_assembly_workers.get(), 3);
    assert_eq!(
        defaults.tuning_cost_per_candidate,
        std::time::Duration::from_millis(4)
    );
    let mut report = profile(&[1, 2, 3]);
    let selected = select(
        &candidates,
        &mut report,
        CompileOptions {
            assumed_assembly_workers: defaults.assumed_assembly_workers,
            ..options(Some(unit * 2.0))
        },
    )
    .unwrap();
    assert_eq!(selected.len(), 3);
}

#[test]
fn an_unaffordable_candidate_does_not_block_a_later_cheaper_one() {
    let scores = [Score {
        work: 1,
        parallelism: 1,
    }; 3];
    let order = boltzmann_order(&scores);
    let mut candidates = (0..3)
        .map(|index| candidate(vec![request(&index.to_string())], &[(0, "entry")]))
        .collect::<Vec<_>>();
    candidates[order[1]] = candidate(vec![request(&"large".repeat(1024))], &[(0, "entry")]);
    let cost = recipe::CompilationCostModel::default().request_cost_ms(&request("0"));
    let mut report = profile(&[1, 1, 1]);
    assert_eq!(
        select(&candidates, &mut report, options(Some(2.0 * cost))).unwrap(),
        [order[0], order[2]]
    );
    assert_eq!(report.shortlist_estimated_ms, Some(2.0 * cost));
}

#[test]
fn root_work_is_priced_alongside_each_candidates_own_work() {
    let candidates = (0..2)
        .map(|index| {
            candidate(
                vec![request("root"), request(&index.to_string())],
                &[(0, "root"), (1, "entry")],
            )
        })
        .collect::<Vec<_>>();
    let table = Compilations::new(&candidates);
    let (_, first) = table.closure(0, &[false; 2], &BTreeSet::new());
    let (_, second) = table.closure(1, &[false; 2], &BTreeSet::new());
    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 2);
    let own_cost = recipe::CompilationCostModel::default().request_cost_ms(&request("1"));
    assert_eq!(table.additional_cost_ms(&second, &first), own_cost);
}
