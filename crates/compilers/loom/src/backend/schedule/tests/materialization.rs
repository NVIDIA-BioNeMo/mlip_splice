// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn shared_diamond_materializes_once_for_two_consumers() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 3 }],
        Vec::new(),
        0,
    );
    let plan = cut_plan(&loom, 0);

    assert_eq!(plan.candidates.len(), 2);
    let cut = &plan.candidates[1];
    assert_eq!(cut.materialized.len(), 1);
    assert_eq!(cut.materialized[0].consumer_count, 2);
    assert!(cut.materialized[0].closure_statements >= 8);
    assert!(matches!(
        cut.launches[0],
        Launch::Producer {
            output: 0,
            ref inputs
        } if inputs.is_empty()
    ));
    assert_eq!(
        cut.launches
            .iter()
            .filter(|launch| matches!(launch, Launch::Consumer { .. }))
            .count(),
        2
    );
}
#[test]
fn materialized_workspace_preserves_f16_and_bf16_storage() {
    for (dtype, xir_dtype, scalar) in [
        (Dtype::F16, DType::F16, Scalar::F16),
        (Dtype::Bf16, DType::Bf16, Scalar::BF16),
    ] {
        let loom = diamond(dtype, xir_dtype, vec![Domain { extent: 3 }], Vec::new(), 0);
        let plan = cut_plan(&loom, 0);
        let materialized = &plan.candidates[1].materialized[0];
        assert_eq!(materialized.scalar, scalar);
        assert_eq!(materialized.nbytes, 3 * 8 * 2);
    }
}
#[test]
fn exact_materialization_saves_an_indivisible_dag() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 3 }],
        Vec::new(),
        0,
    );
    let (statement_cap, ir) = bounded_materialized_plan(&loom);
    assert!(plan(&loom, statement_cap).is_err());
    assert_eq!(ir.candidates.len(), 1);
    let candidate = &ir.candidates[0];
    assert!(!candidate.materialized.is_empty());

    let num_consumers = candidate
        .launches
        .iter()
        .filter(|launch| matches!(launch, Launch::Consumer { .. }))
        .count();
    let mut consumer_index = 0;
    let mut saw_reload = false;
    for launch in &candidate.launches {
        let kernel = match launch {
            Launch::Producer { output, inputs } => {
                assert!(inputs.iter().all(|input| input < output));
                let kernel = emit_producer_kernel(
                    &ir.domain,
                    &loom,
                    BLOCK_SIZES[0],
                    *output,
                    inputs,
                    &candidate.materialized,
                )
                .unwrap();
                let f = kernel.body();
                let position = f
                    .op(body(f)[1])
                    .results
                    .first()
                    .copied()
                    .expect("producer defines its iteration position first");
                assert!(has_position_store(f, position, loom.lane_count));
                kernel
            }
            Launch::Consumer { part, output_modes } => {
                for (output, mode) in output_modes.iter().enumerate() {
                    let expected = if part.output_continues(output) {
                        WriteMode::AddOrdered
                    } else {
                        WriteMode::Store
                    };
                    assert_eq!(*mode, expected);
                }
                let kernel = emit_consumer_kernel(
                    &ir.domain,
                    &loom,
                    BLOCK_SIZES[0],
                    consumer_index,
                    num_consumers,
                    part,
                    &candidate.materialized,
                )
                .unwrap();
                consumer_index += 1;
                // The same relation the producer keeps, from the other
                // side: a consumer declares the whole layout, and the
                // workspaces it reads are appended past it.
                let params = kernel.function.parameters();
                let declared = params
                    .iter()
                    .map(|param| param.name.as_str())
                    .collect::<Vec<_>>();
                let layout = ir
                    .domain
                    .layout
                    .params
                    .iter()
                    .map(|param| param.name.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(
                    declared[..layout.len()],
                    layout[..],
                    "the declared prefix is the layout itself"
                );
                assert!(
                    declared[layout.len()..]
                        .iter()
                        .all(|name| name.starts_with("loom_material_")),
                    "past the prefix, only workspaces: {declared:?}"
                );
                saw_reload |= body_loads_materialized(kernel.body());
                kernel
            }
        };
        assert!(
            statement_count(kernel.body()) <= statement_cap,
            "launch has {} statements above cap {statement_cap}",
            statement_count(kernel.body())
        );
    }
    assert!(saw_reload);
}
#[test]
fn materialized_left_term_stays_before_its_bounded_continuation() {
    let loom = asymmetric_additive_terms();
    let (statement_cap, ir) = (1..1024)
        .find_map(|statement_cap| {
            let domain = DomainPlan::prepare(&loom).ok()?;
            let ir =
                PlanIr::build_bounded(&loom, domain, statement_cap, 0, 16, &OnceCell::new(), true)
                    .ok()?;
            let candidate = &ir.candidates[0];
            let modes = candidate
                .launches
                .iter()
                .filter_map(|launch| match launch {
                    Launch::Consumer { output_modes, .. } => Some(output_modes[0]),
                    Launch::Producer { .. } => None,
                })
                .collect::<Vec<_>>();
            let first_consumer = candidate
                .launches
                .iter()
                .position(|launch| matches!(launch, Launch::Consumer { .. }))?;
            (!candidate.materialized.is_empty()
                && modes == [WriteMode::Store, WriteMode::AddOrdered]
                && candidate.launches[first_consumer..]
                    .iter()
                    .all(|launch| matches!(launch, Launch::Consumer { .. })))
            .then_some((statement_cap, ir))
        })
        .expect("the oversized left term can be materialized without moving its continuation");
    assert!(plan(&loom, statement_cap).is_err());

    let candidate = &ir.candidates[0];
    let first_consumer = candidate
        .launches
        .iter()
        .position(|launch| matches!(launch, Launch::Consumer { .. }))
        .unwrap();
    assert!(
        candidate.launches[..first_consumer]
            .iter()
            .all(|launch| matches!(launch, Launch::Producer { .. }))
    );
    assert!(
        candidate.launches[first_consumer..]
            .iter()
            .all(|launch| matches!(launch, Launch::Consumer { .. }))
    );

    let consumers = candidate
        .launches
        .iter()
        .filter_map(|launch| match launch {
            Launch::Consumer { part, output_modes } => Some((part, output_modes[0])),
            Launch::Producer { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(consumers.len(), 2);
    assert!(!consumers[0].0.output_continues(0));
    assert_eq!(consumers[0].1, WriteMode::Store);
    assert!(consumers[1].0.output_continues(0));
    assert_eq!(consumers[1].1, WriteMode::AddOrdered);

    let left = emit_kernel(&ir.domain, &loom, BLOCK_SIZES[0], 0, 2, consumers[0].0).unwrap();
    let right = emit_kernel(&ir.domain, &loom, BLOCK_SIZES[0], 1, 2, consumers[1].0).unwrap();
    assert!(statement_count(left.body()) > statement_cap);
    assert!(statement_count(right.body()) <= statement_cap);
}
/// A producer kernel opens with its position, runs one hidden-axis loop, and
/// takes the inputs it reads plus the workspace it fills. The gallery reaches
/// no producer, so nothing else pins this shape.
#[test]
fn a_producer_runs_one_hidden_axis_loop_after_its_position_prologue() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 7 }],
        Vec::new(),
        0,
    );
    let plan = cut_plan(&loom, 0);
    let producer = emit_producer_kernel(
        &plan.domain,
        &loom,
        BLOCK_SIZES[0],
        0,
        &[],
        &plan.candidates[1].materialized,
    )
    .unwrap();

    // The opening, in order: the lane, the flat position, the bounds guard.
    let f = producer.body();
    let opening = body(f);
    let named = |index: usize| {
        f.name(f.op(opening[index]).results[0])
            .expect("the opening binds named values")
            .to_string()
    };
    assert!(named(0).contains("lane"));
    assert!(named(1).contains("position"));
    assert!(matches!(f.op(opening[2]).kind, OpKind::If));

    assert_eq!(
        hidden_axis_loops(producer.body()),
        1,
        "one hidden-axis loop, not none and not nested pairs"
    );
    let prefix = u_loop_prefix(producer.body());
    let trip = u_loop_body(producer.body());
    assert!(!trip.is_empty(), "the loop repeats something");
    assert!(
        body_load_count_in(f, &trip) > 0,
        "the work that depends on the lane is inside the loop"
    );
    let index = f
        .block(f.region_block(f.op(u_loop(f, &opening)).regions[0]))
        .params[0];
    assert!(
        prefix.iter().all(|&op| !step_operands(f, op)
            .iter()
            .any(|&value| f.reads(value).contains(&index))),
        "nothing before the loop reads the index the loop binds"
    );

    // What it takes: the program's inputs, then the workspace it fills.
    let declared = producer.function.parameters();
    let params = declared
        .iter()
        .map(|param| param.name.as_str())
        .collect::<Vec<_>>();
    let mut distinct = params.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), params.len(), "{params:?} repeats a name");
    // The relation V4b's ParameterValues rests on: what it declares opens with
    // the layout's own prefix — every layout param but the outputs — and the
    // workspaces are appended past it, so they are not layout params.
    let outputs = loom.outputs().len();
    let prefix = plan.domain.layout.params.len() - outputs;
    assert_eq!(
        params[..prefix],
        plan.domain.layout.params[..prefix]
            .iter()
            .map(|param| param.name.as_str())
            .collect::<Vec<_>>()[..],
        "the declared prefix is the layout's own"
    );
    assert!(
        params[prefix..]
            .iter()
            .all(|name| name.starts_with("loom_material_")),
        "past the prefix, only workspaces: {params:?}"
    );
    assert_eq!(params, ["in_0", "in_1", "loom_material_0"]);
}

#[test]
fn materialized_workspace_is_indexed_by_iteration_position() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 7 }],
        Vec::new(),
        0,
    );
    let plan = cut_plan(&loom, 0);
    let materialized = &plan.candidates[1].materialized[0];
    let producer = emit_producer_kernel(
        &plan.domain,
        &loom,
        BLOCK_SIZES[0],
        0,
        &[],
        &plan.candidates[1].materialized,
    )
    .unwrap();
    let f = producer.body();
    let position = f.op(body(f)[1]).results[0];

    assert_eq!(plan.domain.index.batch_total().unwrap(), 7);
    assert_eq!(materialized.nbytes, 7 * 8 * 4);
    assert!(has_position_store(f, position, 8));
}
#[test]
fn fiber_bridge_keeps_only_the_baseline_candidate() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 7 }, Domain { extent: 3 }],
        vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        1,
    );

    let plan = cut_plan(&loom, 0);
    assert_eq!(plan.candidates.len(), 1);
}
#[test]
fn cut_candidate_exists_only_above_the_register_cap() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 3 }],
        Vec::new(),
        0,
    );

    assert_eq!(cut_plan(&loom, usize::MAX).candidates.len(), 1);
    assert_eq!(cut_plan(&loom, 0).candidates.len(), 2);
}

/// A cut is counted per value and carried as a name, so within one evaluation a
/// name has to identify a binding.
///
/// The plan counts how many launches recompute each binding by its `ValueId` in
/// the profile's arena, and hands the winner to every kernel as the name to look
/// for in its own evaluation. Two bindings sharing a name would make those two
/// steps disagree about which value was chosen.
#[test]
fn one_name_is_one_binding_of_an_evaluation() {
    let programs = [
        diamond(
            Dtype::F32,
            DType::F32,
            vec![Domain { extent: 3 }],
            Vec::new(),
            0,
        ),
        asymmetric_additive_terms(),
        pointwise_add_chain(),
        reduce_pointwise_sum_program(true),
        dot_program(),
        guarded_additive_program(3),
    ];
    for loom in &programs {
        let evaluation = evaluation_profile_with_provenance(loom).unwrap();
        let f = evaluation.body.function();
        let mut named = BTreeSet::new();
        for op in evaluation.body.steps() {
            let Some(name) = f.name(f.op(op).results[0]) else {
                continue;
            };
            assert!(
                named.insert(name.to_owned()),
                "{} binds {name} twice",
                loom.name
            );
            assert_eq!(
                f.value_named(name),
                Some(f.op(op).results[0]),
                "{} resolves {name} to another binding",
                loom.name
            );
        }
        assert!(!named.is_empty(), "{} binds nothing", loom.name);
    }
}

#[test]
fn bounded_plan_retains_required_materialization() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 3 }],
        Vec::new(),
        0,
    );
    let (statement_cap, _) = bounded_materialized_plan(&loom);
    let domain = DomainPlan::prepare(&loom).unwrap();
    let ir = PlanIr::build_bounded_with_settings(
        &loom,
        domain,
        super::super::Limits {
            statement_cap,
            register_cap: 0,
            cut_limit: 16,
            plan_clusters: false,
        },
        &OnceCell::new(),
        &crate::backend::BackendSettings::default(),
    )
    .unwrap();
    assert_eq!(ir.candidates.len(), 1);
    let candidate = &ir.candidates[0];
    assert!(!candidate.materialized.is_empty());
    assert!(
        candidate
            .launches
            .iter()
            .any(|launch| matches!(launch, Launch::Producer { .. }))
    );
    assert!(
        candidate
            .launches
            .iter()
            .any(|launch| matches!(launch, Launch::Consumer { .. }))
    );
}
