// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn candidate_ptx(entry: &str, max_threads: u32, parameter: &str, body: &str) -> Vec<u8> {
    format!(
        ".version 8.0\n.target sm_80\n.address_size 64\n\n\
             .visible .entry {entry}(\n    .param {parameter} arg\n)\n\
             .maxntid {max_threads}, 1, 1\n{{\n    {body}\n    ret;\n}}\n"
    )
    .into_bytes()
}

/// The requests and modules the recipe carries once every one of these modules
/// is installed into it.
fn installed(modules: &[ModuleSource]) -> ::recipe::Recipe {
    let mut builder = ::recipe::RecipeBuilder::new(0, 0);
    for module in modules {
        let installed = module.install(&mut builder);
        builder.kernel(installed, &module.entry).add();
    }
    builder
        .finish(::recipe::RecipeSignature::empty())
        .unwrap()
        .into_recipe()
}

fn test_target() -> ::recipe::PtxTarget {
    ::recipe::PtxTarget {
        isa: [8, 0],
        minimum_sm: 80,
    }
}

fn mixed_output_domains_program_with_depth(depth: usize) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let edge = builder.param(Shape::scalar(DType::F32));
    let receiver = builder.param(Shape::scalar(DType::F32));
    let mut receiver_output = receiver;
    for _ in 0..depth {
        receiver_output = builder.multiply(receiver_output, receiver);
    }
    let main = builder.end_func(vec![edge, receiver_output]);
    let module = builder.finish(main).unwrap();
    let binding = |domain| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![domain],
    };
    loom_program! {
        name: "mixed_domains".into(),
        module,
        lane_count: 8,
        domains: vec![
            Domain {
                extent: 50,
            },
            Domain {
                extent: 10,
            },
            Domain {
                extent: 10,
            },
        ],
        index_maps: vec![
            IndexMap {
                source_id: 0,
                target_id: 1,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            },
            IndexMap {
                source_id: 0,
                target_id: 2,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            },
        ],
        num_axes: 1,
        operands: vec![binding(0), binding(2), binding(0), binding(1)],
        num_inputs: 2,
        available_views: vec![AlignedView {
            source_id: 0,
            anchor_target_id: 1,
            other_target_id: 2,
        }],
    }
}

fn mixed_output_domains_program() -> LiftedProgram {
    mixed_output_domains_program_with_depth(0)
}

#[test]
fn mixed_output_domains_offer_fiber_recipe_without_sender_memset() {
    let loom = mixed_output_domains_program();
    let partitions = output_domain_families(&loom, true).unwrap().unwrap();
    assert_eq!(partitions[0].len(), 2);
    assert!(partitions[0].iter().any(|(family, outputs, _)| {
        outputs == &[1]
            && Plan::prepare(family)
                .unwrap()
                .index
                .fiber_bridge()
                .is_some()
    }));

    let (compiled, _) = compile_with_limits_profiled(
        loom,
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let inputs = compiled.input_count;
    let outputs = compiled.output_count;
    let slots = compiled.index_arrays;
    assert_eq!(outputs, 2);
    assert_eq!(
        inputs,
        2 + slots
            .iter()
            .map(|slot| slot.kind.array_count())
            .sum::<usize>()
    );
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("mixed domains should produce a whole-recipe choice");
    };
    assert_eq!(candidates.len(), 3);
    fn writes_sender_memset(recipe: &::recipe::RecipeBody) -> bool {
        recipe.nodes().iter().any(|node| match node.work() {
            Work::Memset {
                destination:
                    Address {
                        buf: WritableBuf::Output(1),
                        byte_offset: 0,
                    },
                ..
            } => true,
            Work::Choice { candidates, .. } => candidates.iter().any(writes_sender_memset),
            _ => false,
        })
    }
    assert!(writes_sender_memset(&candidates[0]));
    assert!(!writes_sender_memset(&candidates[1]));
    assert_eq!(
        candidates[1]
            .nodes()
            .iter()
            .filter(|node| matches!(node.work(), Work::Choice { .. }))
            .count(),
        2
    );
    // The sender-iteration arm writes every output fiber-direct or at the
    // iterated position: no atomics anywhere, so no memset at all.
    fn writes_any_memset(recipe: &::recipe::RecipeBody) -> bool {
        recipe.nodes().iter().any(|node| match node.work() {
            Work::Memset { .. } => true,
            Work::Choice { candidates, .. } => candidates.iter().any(writes_any_memset),
            _ => false,
        })
    }
    assert!(!writes_any_memset(&candidates[2]));
    assert_eq!(
        candidates[2]
            .nodes()
            .iter()
            .filter(|node| matches!(node.work(), Work::Choice { .. }))
            .count(),
        1
    );
}

#[test]
fn failed_output_domain_preflight_falls_back_to_the_baseline() {
    let plan_fails = |loom: &LiftedProgram, cap: usize| {
        Plan::prepare(loom)
            .and_then(|program| {
                PlanIr::build_bounded(
                    loom,
                    program,
                    cap,
                    usize::MAX,
                    usize::MAX,
                    &std::cell::OnceCell::new(),
                    true,
                )
            })
            .is_err()
    };
    let loom = mixed_output_domains_program_with_depth(24);
    let partitions = output_domain_families(&loom, true).unwrap().unwrap();
    let statement_cap = (1..DEFAULT_MAX_KERNEL_STATEMENTS)
        .find(|&cap| {
            !plan_fails(&loom, cap)
                && partitions
                    .iter()
                    .flatten()
                    .any(|(family, _, _)| plan_fails(family, cap))
        })
        .expect("need a cap where only the fiber family is oversized");

    let (compiled, _) = compile_with_limits_profiled(
        loom,
        test_options(None),
        statement_cap,
        usize::MAX,
        usize::MAX,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let slots = compiled.index_arrays;
    assert!(slots.iter().all(|slot| slot.kind != SlotKind::AlignedTargetIndices));
    assert!(recipe.body().nodes().iter().any(|node| matches!(
        node.work(),
        Work::Memset {
            destination: Address {
                buf: WritableBuf::Output(1),
                byte_offset: 0,
            },
            ..
        }
    )));
}

#[test]
fn output_domain_candidate_unions_and_remaps_every_slot_kind() {
    let (compiled, _) = compile_with_limits_profiled(
        mixed_output_domains_program(),
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let inputs = compiled.input_count;
    let slots = compiled.index_arrays;
    let descriptions = slots
        .iter()
        .map(|slot| (slot.kind, slot.source_id, slot.target_id, slot.other_target_id))
        .collect::<Vec<_>>();
    assert_eq!(
        descriptions,
        vec![
            (SlotKind::TargetIndices, 0, 2, 2),
            (SlotKind::TargetIndices, 0, 1, 1),
            (SlotKind::Preimage, 0, 1, 1),
            (SlotKind::AlignedTargetIndices, 0, 1, 2),
        ]
    );
    assert_eq!(inputs, 7);

    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("mixed domains should produce a whole-recipe choice");
    };
    let family_bindings = candidates[1]
        .nodes()
        .iter()
        .filter_map(|node| {
            let Work::Choice { input_binding, .. } = node.work() else {
                return None;
            };
            Some(input_binding.clone())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        family_bindings,
        vec![
            vec![BufRef::Input(0), BufRef::Input(1), BufRef::Input(2)],
            vec![
                BufRef::Input(0),
                BufRef::Input(1),
                BufRef::Input(4),
                BufRef::Input(5),
                BufRef::Input(6),
            ],
        ]
    );
}

fn requested_alias(entry: &str) -> SourceAlias {
    SourceAlias { source: candidate_ptx(entry, 128, ".u64", "mov.u32 %r1, 1;").into(), entry: entry.into(),
        cap: None, assembly_mode: ::recipe::PtxAssemblyMode::Relocatable, statements: 20,
        addition_chain: None,
        rank: CandidateRank { flavor: CandidateFlavor::Baseline, part: 0, block: 0 } }
}

#[test]
fn one_source_can_back_many_installed_modules() {
    let alias = requested_alias("loom_a");
    let (modules, side) = request_modules(&[alias], &[], 0, test_target()).unwrap();
    assert!(side.is_empty());
    let recipe = installed(&[modules[0].clone(), modules[0].clone()]);
    assert_eq!(recipe.context().as_slice().len(), 1);
    assert!(recipe.body().context().iter().all(|module|
        module.code.compile_request() == Some(::recipe::CompileRequestId(0))));
}

#[test]
fn a_capped_selection_keeps_its_uncapped_prerequisite_not_an_extra_launch() {
    let mut alias = requested_alias("loom_a");
    alias.cap = Some(168);
    let (modules, _) = request_modules(&[alias], &[], 1024, test_target()).unwrap();
    let recipe = installed(&modules);
    assert_eq!(recipe.body().nodes().len(), 1);
    assert_eq!(recipe.body().context().len(), 1);
    let requests = recipe.context().as_slice();
    assert_eq!(requests.len(), 2);
    let (base, cap) = (&requests[0], &requests[1]);
    assert_eq!(base.options.max_registers, None);
    assert_eq!(cap.options.max_registers, Some(168));
    assert_eq!(cap.requires.iter().collect::<Vec<_>>(), vec![(&::recipe::CompileRequestId(0),
        &::recipe::ResourceBounds { min_registers: Some(169), max_local_bytes: Some(0) })]);
    assert_eq!(base.sources[0].blob.content_digest().unwrap(), cap.sources[0].blob.content_digest().unwrap());
    assert_eq!(recipe.body().context()[0].resource_bounds,
        ::recipe::ResourceBounds { min_registers: None, max_local_bytes: Some(1024) });
}

#[test]
fn assembly_modes_share_a_source_but_not_a_request() {
    let relocatable = requested_alias("loom_a");
    let whole = SourceAlias { assembly_mode: ::recipe::PtxAssemblyMode::WholeProgram, ..relocatable.clone() };
    let (modules, _) = request_modules(&[relocatable, whole], &[], 0, test_target()).unwrap();
    let recipe = installed(&modules);
    let requests = recipe.context().as_slice();
    assert_eq!(requests.len(), 2);
    assert_ne!(requests[0].options.assembly_mode, requests[1].options.assembly_mode);
    assert_eq!(requests[0].sources[0].blob.content_digest().unwrap(), requests[1].sources[0].blob.content_digest().unwrap());
}

/// Every module of a body and of every candidate below it.
fn all_modules(body: &::recipe::RecipeBody) -> Vec<&::recipe::KernelModule> {
    let mut modules = body.context().iter().collect::<Vec<_>>();
    for node in body.nodes() {
        if let Work::Choice { candidates, .. } = node.work() {
            for candidate in candidates {
                modules.extend(all_modules(candidate));
            }
        }
    }
    modules
}

#[test]
fn every_candidate_module_resolves_from_the_recipe_wide_request_table() {
    let (compiled, _) = compile_with_limits_profiled(
        guarded_program(),
        test_options(None),
        DEFAULT_MAX_KERNEL_STATEMENTS,
        usize::MAX,
        0,
        true,
    )
    .unwrap();
    let recipe = compiled.recipe;
    let requests = recipe.context().as_slice();
    assert!(!requests.is_empty());

    // No module carries an image: the executor assembles, and every module says
    // which compile it wants and what that image must fit.
    let modules = all_modules(recipe.body());
    assert!(!modules.is_empty());
    for module in &modules {
        assert_eq!(module.code.images().count(), 0);
        let request = module
            .code
            .compile_request()
            .expect("a module resolves from a request");
        assert!(request.0 < requests.len());
    }

    // The table is interned: no two entries state the same compile, however
    // many modules resolve from each.
    assert!(modules.len() >= requests.len());
    let keys = requests
        .iter()
        .map(|request| {
            (
                request.sources[0].blob.content_digest().unwrap(),
                request.options,
                request
                    .requires
                    .iter()
                    .map(|(prerequisite, bounds)| (prerequisite.0, *bounds))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(keys.len(), requests.len());

    let target = ptx_target(TEST_TARGET_SM);
    for request in requests {
        assert_eq!(request.sources[0].kind, ::recipe::SourceKind::Ptx);
        assert_eq!(request.sources[0].target, target);
        assert_ne!(
            request.options.assembly_mode,
            ::recipe::PtxAssemblyMode::EnvironmentDefault,
            "loom states the mode it assembles under"
        );
        // A cap presupposes its uncapped twin over the very same source, and
        // nothing else ever states a prerequisite.
        match request.options.max_registers {
            None => assert!(request.requires.is_empty()),
            Some(_) => {
                let [(prerequisite, bounds)] = &request.requires.iter().collect::<Vec<_>>()[..]
                else {
                    panic!("a capped request presupposes exactly its uncapped twin");
                };
                assert_eq!(
                    **bounds,
                    ::recipe::ResourceBounds {
                        min_registers: request.options.max_registers.map(|cap| cap + 1),
                        max_local_bytes: Some(0),
                    }
                );
                let base = &requests[prerequisite.0];
                assert_eq!(base.options.max_registers, None);
                assert_eq!(base.options.assembly_mode, request.options.assembly_mode);
                assert_eq!(
                    base.sources[0].blob.content_digest().unwrap(),
                    request.sources[0].blob.content_digest().unwrap()
                );
            }
        }
    }
}
