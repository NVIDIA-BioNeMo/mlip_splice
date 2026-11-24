// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// `x + y` then 24 more `+ x` is one sum of 26 monomials, and term enumeration
/// names every one: an unbounded cap keeps them in a single part, and the count
/// is what says the sum was distributed rather than held whole.
/// Every named value the selected terms reach, following each binding they read
/// through to what it was bound to: `add_reads` stops at the nearest binding, so
/// the leaves of a fused element are one step further in.
fn reached_names(f: &Function, steps: &[OpId], roots: &[Term]) -> BTreeSet<String> {
    let mut read = BTreeSet::new();
    for root in roots {
        root.add_reads(f, &mut read);
    }
    loop {
        let of = read.iter().copied().collect::<Vec<_>>();
        let before = read.len();
        for op in slice_bindings(f, steps, &of).unwrap() {
            f.add_reads(f.op(op).operands[0], &mut read);
        }
        if read.len() == before {
            break;
        }
    }
    read.iter()
        .filter_map(|&value| f.name(value).map(str::to_owned))
        .collect()
}

#[test]
fn a_pointwise_add_chain_enumerates_every_monomial() {
    let loom = pointwise_add_chain();
    let parts = plan(&loom, usize::MAX).unwrap();
    assert_eq!(parts.len(), 1);
    let selection = &parts[0].elements[0][0];
    assert_eq!(selection.terms.len(), 26);
    assert_eq!(parts[0].len(), 26);
    assert!(selection.reduction.is_none());
}
#[test]
fn reduce_and_dot_regions_split() {
    let reduce = reduce_sum_program(
        Dtype::F32,
        DType::F32,
        Literal::zeros(Shape::scalar(DType::F32)).unwrap(),
    );
    let dot = dot_program();

    for loom in [&reduce, &dot] {
        let parts = (73..512)
            .find_map(|cap| plan(loom, cap).ok().filter(|parts| parts.len() > 1))
            .expect("the reduction region admits a bounded additive split");
        assert!(!parts[0].elements[0][0].continuation);
        assert!(
            parts[1..]
                .iter()
                .all(|part| part.elements[0][0].continuation)
        );
        assert!(parts.iter().all(|part| {
            part.elements[0]
                .iter()
                .all(|selection| selection.reduction.is_some())
        }));
    }
}
#[test]
fn reduction_regions_record_ordered_launches_and_one_init_owner() {
    let reduce = reduce_sum_program(
        Dtype::F32,
        DType::F32,
        Literal::zeros(Shape::scalar(DType::F32)).unwrap(),
    );
    let dot = dot_program();

    for (loom, provenance) in [
        (&reduce, ReductionProvenance::Reduce),
        (&dot, ReductionProvenance::Dot),
    ] {
        let (_, plan) = split_reduction_plan(loom, provenance);
        let candidate = &plan.candidates[0];
        let region = candidate
            .reductions
            .iter()
            .find(|region| region.provenance == provenance)
            .unwrap();
        assert_eq!(region.init_launch, region.launches[0]);
        assert!(region.launches.windows(2).all(|pair| pair[0] < pair[1]));
        for (fragment, &launch) in region.launches.iter().enumerate() {
            let Launch::Consumer { output_modes, .. } = &candidate.launches[launch] else {
                panic!("reduction fragment is not a consumer");
            };
            assert_eq!(
                output_modes[region.output],
                if fragment == 0 {
                    WriteMode::Store
                } else {
                    WriteMode::AddOrdered
                }
            );
        }
    }
}
/// A reduction split across launches applies its init exactly once, in the
/// fragment that opens the accumulator; every later fragment adds its own term
/// into what is already there, and none of them reads a profile value — the
/// terms are re-derived in each kernel's own arena.
#[test]
fn split_reduction_fragments_apply_their_init_once() {
    const INIT: f64 = 3.25;
    let loom = reduce_sum_program(
        Dtype::F32,
        DType::F32,
        Literal::new(
            Shape::scalar(DType::F32),
            (INIT as f32).to_le_bytes().to_vec(),
        ),
    );
    let (_, plan) = split_reduction_plan(&loom, ReductionProvenance::Reduce);
    let candidate = &plan.candidates[0];
    let region = candidate
        .reductions
        .iter()
        .find(|region| region.provenance == ReductionProvenance::Reduce)
        .unwrap();
    let mut init_occurrences = 0;
    for (fragment, &launch) in region.launches.iter().enumerate() {
        let kernel =
            emit_planned_kernel(&plan.domain, &loom, candidate, launch, BLOCK_SIZES[0]).unwrap();
        init_occurrences += body_count_float_constant(kernel.body(), INIT);
        let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 80);
        // The opening fragment stores the init it holds; every later one adds
        // its own term into the accumulator.
        assert_eq!(ptx.contains("add.rn.f32"), fragment > 0, "{ptx}");
        assert!(!ptx.contains("loom_profile_input"));
    }
    assert_eq!(init_occurrences, 1);
}
/// A reduction that fits one kernel stays one region with one store: every
/// term of the sum is selected together, so nothing continues into a later
/// launch and nothing has to be read back.
#[test]
fn an_unsplit_reduction_stays_one_store_region() {
    let loom = reduce_sum_program(
        Dtype::F32,
        DType::F32,
        Literal::new(Shape::scalar(DType::F32), 3.25f32.to_le_bytes().to_vec()),
    );
    let domain = DomainPlan::prepare(&loom).unwrap();
    let plan = PlanIr::build(
        &loom,
        domain,
        plan(&loom, usize::MAX).unwrap(),
        usize::MAX,
        0,
    )
    .unwrap();
    let candidate = &plan.candidates[0];
    let region = candidate
        .reductions
        .iter()
        .find(|region| region.provenance == ReductionProvenance::Reduce)
        .unwrap();
    assert_eq!(region.launches, vec![region.init_launch]);
    let launch = region.launches[0];
    let Launch::Consumer { part, output_modes } = &candidate.launches[launch] else {
        panic!("a reduction region is not a consumer");
    };
    assert_eq!(output_modes[region.output], WriteMode::Store);
    let selection = &part.elements[region.output][0];
    assert!(!selection.continuation);
    // The XIR reduce folds 32 inputs into its init, and enumeration names the
    // init and each input: 33 terms, all in this one launch.
    assert_eq!(selection.terms, (0..33).collect::<Vec<_>>());
    assert!(selection.reduction.is_some());
    assert_eq!(candidate.launches.len(), 1);
}
#[test]
fn reduction_region_workspace_scalar_matches_xir_math_dtype() {
    for (dtype, xir_dtype, scalar) in [
        (Dtype::F16, DType::F16, Scalar::F16),
        (Dtype::Bf16, DType::Bf16, Scalar::BF16),
        (Dtype::F32, DType::F32, Scalar::F32),
        (Dtype::F64, DType::F64, Scalar::F64),
    ] {
        let loom = reduce_sum_program(
            dtype,
            xir_dtype,
            Literal::zeros(Shape::scalar(xir_dtype)).unwrap(),
        );
        let domain = DomainPlan::prepare(&loom).unwrap();
        let plan = PlanIr::build(
            &loom,
            domain,
            plan(&loom, usize::MAX).unwrap(),
            usize::MAX,
            0,
        )
        .unwrap();
        let region = plan.candidates[0]
            .reductions
            .iter()
            .find(|region| region.provenance == ReductionProvenance::Reduce)
            .unwrap();
        assert_eq!(region.scalar, scalar);
        assert_eq!(region.scalar.size(), scalar.size());
    }
}
#[test]
fn xir_hidden_lane_and_fiber_reduction_regions_do_not_merge() {
    let base = || {
        reduce_sum_program(
            Dtype::F32,
            DType::F32,
            Literal::zeros(Shape::scalar(DType::F32)).unwrap(),
        )
    };
    let mut hidden_lane = base();
    hidden_lane.operands.last_mut().unwrap().lane_extent = 1;

    let mut fiber = base();
    fiber.domains = vec![Domain { extent: 7 }, Domain { extent: 3 }];
    fiber.index_maps = vec![IndexMap {
        source_id: 0,
        target_id: 1,
        is_injective: false,
        is_sorted: false,
        may_have_unmapped: false,
    }];
    fiber.operands.last_mut().unwrap().axis_domains[0] = 1;

    for (loom, structural) in [
        (&hidden_lane, ReductionProvenance::HiddenLane),
        (&fiber, ReductionProvenance::Fiber),
    ] {
        let domain = DomainPlan::prepare(loom).unwrap();
        let plan =
            PlanIr::build(loom, domain, plan(loom, usize::MAX).unwrap(), usize::MAX, 0).unwrap();
        let regions = &plan.candidates[0].reductions;
        assert_eq!(regions.len(), 2);
        assert_eq!(
            regions
                .iter()
                .map(|region| region.provenance)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([ReductionProvenance::Reduce, structural])
        );
        assert!(
            regions
                .iter()
                .all(|region| region.output == 0 && region.element == 0)
        );
    }
}
/// A reduce over a pointwise add distributes through both: the sum of 8 cells,
/// each itself `a + b`, enumerates as the 16 inputs plus the init. Every term is
/// one input element, so no term straddles the pointwise boundary.
#[test]
fn a_reduce_over_a_pointwise_add_enumerates_both_operands() {
    for computed_init in [false, true] {
        let loom = reduce_pointwise_sum_program(computed_init);
        let parts = plan(&loom, usize::MAX).unwrap();
        assert_eq!(parts.len(), 1);
        // 8 cells of `a + b` is 16 monomials, plus the init: one term when it is
        // a constant, two when the program computes it from its own operands.
        let expected = if computed_init { 18 } else { 17 };
        assert_eq!(parts[0].elements[0][0].terms.len(), expected);

        // A selection holding every term keeps the element's own value, which is
        // the whole sum: one root reading every input of both operands.
        let profile = evaluation_profile_with_provenance(&loom).unwrap();
        let held = &profile.body;
        let f = held.function();
        let (_, roots) =
            selected_term_values(f, &held.steps(), &profile.outputs, &parts[0]).unwrap();
        assert_eq!(roots.len(), 1);
        let named = reached_names(f, &held.steps(), &roots);
        for operand in 0..2 {
            for element in 0..8 {
                assert!(
                    named.contains(&format!("loom_profile_input_{operand}_{element}")),
                    "{named:?}"
                );
            }
        }
    }
}
/// A computed init is not privileged: its operands are terms of the sum like any
/// others, each landing in one fragment, and the fragments together cover every
/// term of the element exactly once.
#[test]
fn a_fragmented_reduce_spreads_its_computed_init_over_terms() {
    let loom = reduce_pointwise_sum_program(true);
    let parts = (73..512)
        .find_map(|cap| plan(&loom, cap).ok().filter(|parts| parts.len() > 1))
        .expect("computed-init reduction admits multiple fragments");
    let profile = evaluation_profile_with_provenance(&loom).unwrap();
    let held = &profile.body;
    let f = held.function();
    let mut selected_terms = BTreeSet::new();
    let mut fragments_reading_init = 0;
    for part in &parts {
        let selection = &part.elements[0][0];
        selected_terms.extend(selection.terms.iter().copied());
        let (_, roots) = selected_term_values(f, &held.steps(), &profile.outputs, part).unwrap();
        let names = reached_names(f, &held.steps(), &roots);
        if names.contains("loom_profile_input_2_0") || names.contains("loom_profile_input_3_0") {
            fragments_reading_init += 1;
        }
    }
    // The init is computed from two operands, and each is a term of its own.
    assert_eq!(fragments_reading_init, 2);
    // 8 cells of `a + b`, plus those two.
    assert_eq!(selected_terms, (0..18).collect::<BTreeSet<_>>());
}
#[test]
fn dot_region_terms_are_exact_products() {
    let loom = dot_program();
    let parts = plan(&loom, usize::MAX).unwrap();
    let profile = evaluation_profile_with_provenance(&loom).unwrap();
    let held = &profile.body;
    let f = held.function();
    // Every term selected together keeps the contraction's own value, which
    // reads both operands at every contracted element and nothing else.
    let (_, roots) = selected_term_values(f, &held.steps(), &profile.outputs, &parts[0]).unwrap();
    assert_eq!(roots.len(), 1);
    let named = reached_names(f, &held.steps(), &roots);
    let expected = (0..2)
        .flat_map(|operand| {
            (0..32).map(move |element| format!("loom_profile_input_{operand}_{element}"))
        })
        .collect::<BTreeSet<_>>();
    assert!(expected.is_subset(&named), "{named:?}");
}
