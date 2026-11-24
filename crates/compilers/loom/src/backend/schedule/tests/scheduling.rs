// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn shared_work_grouping_reduces_duplicated_closures() {
    let loom = diamond(
        Dtype::F32,
        DType::F32,
        vec![Domain { extent: 3 }],
        Vec::new(),
        0,
    );
    let (cap, grouped) = (73..256)
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
            .map(|parts| (cap, parts))
        })
        .expect("diamond admits shared-work grouping");
    let baseline = plan(&loom, cap).unwrap();
    let profile = IncidenceProfile::prepare(&loom).unwrap();
    let settings = PartitionSettings::default();
    let baseline_cost = partition_cost(&profile, &baseline, &settings).unwrap();
    let grouped_cost = partition_cost(&profile, &grouped, &settings).unwrap();

    assert!(
        grouped_cost.duplicated_closure_statements < baseline_cost.duplicated_closure_statements
    );
    let domain = DomainPlan::prepare(&loom).unwrap();
    let ir =
        PlanIr::build_bounded(&loom, domain, cap, usize::MAX, 0, &OnceCell::new(), true).unwrap();
    assert_eq!(
        ir.candidates
            .iter()
            .filter(|candidate| candidate.materialized.is_empty())
            .count(),
        2
    );
}
#[test]
fn algebraic_multiply_add_emits_fma_ptx() {
    let loom = pointwise_multiply_add(false);
    let domain = DomainPlan::prepare(&loom).unwrap();
    let parts = plan(&loom, usize::MAX).unwrap();
    assert_eq!(parts.len(), 1);
    let kernel = emit_kernel(&domain, &loom, BLOCK_SIZES[0], 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
    assert!(ptx.contains("fma.rn.f32"));
    assert!(!ptx.contains("mul.rn.f32"));
}
#[test]
fn algebraic_splits_additive_terms_into_ordered_continuations() {
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
    let module = builder.finish(main).unwrap();
    let loom = loom_program! {
        name: "plan_continuations".into(),
        module,
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    };
    let parts = (73..256)
        .find_map(|cap| plan(&loom, cap).ok().filter(|parts| parts.len() > 1))
        .unwrap();
    let domain = DomainPlan::prepare(&loom).unwrap();
    let plan = PlanIr::build(&loom, domain, parts, usize::MAX, 0).unwrap();
    let modes = plan.candidates[0]
        .launches
        .iter()
        .map(|launch| match launch {
            Launch::Consumer { output_modes, .. } => output_modes[0],
            Launch::Producer { .. } => panic!("baseline has no producer"),
        })
        .collect::<Vec<_>>();

    assert_eq!(modes[0], WriteMode::Store);
    assert!(modes[1..].iter().all(|mode| *mode == WriteMode::AddOrdered));

    // Every launch but the last is read back by its continuation: only the
    // final writer's store may evict-first.
    let launches = &plan.candidates[0].launches;
    for (index, launch) in launches.iter().enumerate() {
        let Launch::Consumer { part, .. } = launch else {
            panic!("baseline has no producer");
        };
        let last = index + 1 == launches.len();
        assert_eq!(part.elements[0][0].read_later, !last);
        let kernel = emit_planned_kernel(
            &plan.domain,
            &loom,
            &plan.candidates[0],
            index,
            BLOCK_SIZES[0],
        )
        .unwrap();
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
        assert_eq!(ptx.contains("st.global.L1::evict_first"), last, "{ptx}");
    }
}
