// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn order_options(element_order: Option<&str>) -> CompileOptions {
    CompileOptions {
        only_element_order: element_order.map(|name| name.parse().unwrap()),
        ..test_options(None)
    }
}

fn orders_of(profile: &CompileProfile) -> Vec<&str> {
    let mut orders = profile
        .kernels
        .iter()
        .map(|unit| unit.order)
        .collect::<Vec<_>>();
    orders.sort_unstable();
    orders.dedup();
    orders
}

#[test]
fn both_orders_are_described_before_either_is_compiled() {
    let loom = mixed_output_domains_program_with_depth(4);
    let (_, profile) = compile_profiled(loom, order_options(None)).unwrap();

    let described = profile
        .element_orders
        .iter()
        .map(|record| record.order)
        .collect::<Vec<_>>();
    assert_eq!(described, ["trace", "liveness"]);
    for record in &profile.element_orders {
        let summary = record.summary.iter().copied().collect::<BTreeMap<_, _>>();
        assert!(summary["elements"] >= 2, "{summary:?}");
        assert!(summary["live_peak"] > 0, "{summary:?}");
        assert!(summary["live_area"] >= summary["live_peak"], "{summary:?}");
    }
    // The two summaries agree on the program and differ only where the
    // sequence can move them.
    let [trace, liveness] = &profile.element_orders[..] else {
        panic!("two orders");
    };
    let field = |record: &ElementOrderSummary, name: &str| {
        record
            .summary
            .iter()
            .find(|(key, _)| *key == name)
            .expect("field")
            .1
    };
    for invariant in ["elements", "bindings", "shared_bindings", "closure_edges"] {
        assert_eq!(
            field(trace, invariant),
            field(liveness, invariant),
            "{invariant}"
        );
    }
}

#[test]
fn a_chosen_order_is_the_only_one_compiled() {
    for order in ["trace", "liveness"] {
        let loom = mixed_output_domains_program();
        let (_, profile) = compile_profiled(loom, order_options(Some(order))).unwrap();

        assert_eq!(orders_of(&profile), [order], "{order}");
        assert!(!profile.kernels.is_empty(), "{order}");
    }
}

#[test]
fn crossing_both_orders_produces_both_arms_worth_of_units() {
    let loom = mixed_output_domains_program();
    let (_, profile) = compile_profiled(loom, order_options(None)).unwrap();

    assert_eq!(orders_of(&profile), ["liveness", "trace"]);
    let single = compile_profiled(mixed_output_domains_program(), order_options(Some("trace")))
        .unwrap()
        .1;
    assert!(
        profile.kernels.len() > single.kernels.len(),
        "crossing both orders should build more units than one order: {} vs {}",
        profile.kernels.len(),
        single.kernels.len()
    );
}

#[test]
fn every_unit_record_names_the_arm_that_asked_for_it() {
    let loom = mixed_output_domains_program();
    let (_, profile) = compile_profiled(loom, order_options(None)).unwrap();

    for unit in &profile.kernels {
        assert!(!unit.order.is_empty(), "{unit:?}");
        assert!(unit.statement_cap > 0, "{unit:?}");
        assert!(!unit.entry.is_empty(), "{unit:?}");
        assert_eq!(unit.ptx_hash.len(), 16, "{unit:?}");
    }
}

#[test]
fn the_arm_list_names_what_the_recipe_choice_holds() {
    let loom = mixed_output_domains_program();
    let (CompiledProgram { recipe, .. }, profile) = compile_profiled(loom, order_options(None)).unwrap();

    // Entry names repeat across orders, so an arm is identified by its
    // position in this list and by nothing else.
    let arms = profile.arms.iter().map(|arm| arm.order).collect::<Vec<_>>();
    assert!(arms.contains(&"trace") && arms.contains(&"liveness"));
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("two arms should produce a whole-recipe choice");
    };
    assert_eq!(candidates.len(), profile.arms.len());
    for arm in &profile.arms {
        assert!(arm.statement_cap > 0, "{arm:?}");
    }
}

#[test]
fn every_complete_arm_uses_the_chosen_order() {
    let loom = mixed_output_domains_program();
    let (_, profile) = compile_profiled(loom, order_options(Some("liveness"))).unwrap();

    assert!(!profile.arms.is_empty());
    assert!(profile.arms.iter().all(|arm| arm.order == "liveness"));
}

#[test]
fn an_unknown_order_name_is_rejected() {
    let error = "shuffle".parse::<ElementOrder>().unwrap_err();
    assert!(error.contains("element order"), "{error}");
}

#[test]
fn a_chosen_frontier_graph_is_compiled_alone_and_identified() {
    let programs = vec![
        mixed_output_domains_program(),
        mixed_output_domains_program(),
    ];
    let options = CompileOptions {
        only_frontier_graph: Some(1),
        ..test_options(None)
    };

    let (program, count) = select_graph(programs, options).unwrap();
    let mut record = GraphPasses::new(program.clone(), options).unwrap().graph_record(1, count);
    let (_, mut profile) = compile_profiled(program, CompileOptions { search_budget: Some(std::time::Duration::from_nanos(1)), ..options }).unwrap();
    profile.set_graph(1);
    record.recipe_arm = Some(0);
    profile.frontier_graphs.push(record);

    assert_eq!(profile.frontier_graphs.len(), 1);
    assert_eq!(profile.frontier_graphs[0].graph, 1);
    assert_eq!(profile.frontier_graphs[0].frontier_size, 2);
    assert_eq!(profile.frontier_graphs[0].recipe_arm, Some(0));
    assert!(profile.frontier_graphs[0].register_pressure > 0);
    assert!(profile.arms.iter().all(|arm| arm.graph == 1));
    assert!(profile.element_orders.iter().all(|order| order.graph == 1));
    assert!(!profile.planned_candidates.is_empty());
    assert!(
        profile
            .planned_candidates
            .iter()
            .all(|candidate| candidate.graph == 1
                && candidate.producer_launches + candidate.consumer_launches > 0
                && candidate.described_arrangements > 0
                && candidate.max_positions_per_block >= candidate.min_positions_per_block)
    );
    assert!(profile.kernels.iter().all(|unit| unit.graph == 1));
}

#[test]
fn a_frontier_graph_outside_the_retained_frontier_is_rejected() {
    let error = select_graph(
        vec![mixed_output_domains_program()],
        CompileOptions {
            only_frontier_graph: Some(1),
            ..test_options(None)
        },
    )
    .expect_err("out-of-range graph must fail");

    assert!(error.contains("frontier_graph"), "{error}");
    assert!(error.contains("[0, 1)"), "{error}");
}
