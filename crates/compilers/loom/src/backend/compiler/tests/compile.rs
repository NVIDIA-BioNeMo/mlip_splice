// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


fn test_options(search_budget_ms: Option<u64>) -> CompileOptions {
    CompileOptions {
        search_budget: search_budget_ms.map(std::time::Duration::from_millis),
        target: Some(crate::backend::compiler::test_target()),
        ..Default::default()
    }
}

fn compile(loom: LiftedProgram, options: CompileOptions) -> Result<CompiledProgram, String> {
    Ok(compile_profiled(loom, options)?.0)
}

fn compile_profiled(loom: LiftedProgram, options: CompileOptions) -> Result<(CompiledProgram, CompileProfile), String> {
    let num_inputs = loom.num_inputs;
    let started = Instant::now();
    let mut profile = CompileProfile::default();
    let mut candidates = Vec::new();
    let mut passes = GraphPasses::new(loom, options)?;
    let mut first_error = None;
    loop {
        let mut checkpoint = || {
            if !candidates.is_empty() && options.search_budget.is_some_and(|limit| started.elapsed() >= limit) {
                Err("test source budget expired".into())
            } else { Ok(()) }
        };
        if checkpoint().is_err() { break; }
        let Some((result, next_profile)) = passes.next(&mut checkpoint) else { break; };
        profile.merge(&next_profile);
        match result {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => { first_error.get_or_insert(error); }
        }
    }
    if candidates.is_empty() { return Err(first_error.unwrap_or_else(|| "no viable source configuration".into())); }
    Ok((combine_compiled(candidates, num_inputs)?, profile))
}

fn compile_with_cap(loom: LiftedProgram, cap: usize) -> Result<CompiledProgram, String> {
    compile(loom, CompileOptions { max_kernel_statements: std::num::NonZeroUsize::new(cap), ..test_options(Some(100)) })
}

pub(in crate::backend) fn compile_with_limits_profiled(loom: LiftedProgram, options: CompileOptions,
    statement_cap: usize, register_cap: usize, cut_limit: usize, plan_clusters: bool,
) -> Result<(CompiledProgram, CompileProfile), String> {
    let settings = BackendSettings::default();
    let limits = CompileLimits { statement_cap, register_cap, cut_limit, plan_clusters,
        max_workspace_bytes: options.max_workspace_bytes, element_order: ElementOrder::Trace,
        target_sm: TEST_TARGET_SM, dump_ptx: options.keep_ptx, record_structure: options.record_dependencies_and_partitions,
        settings: &settings };
    let mut profile = CompileProfile::default();
    let mut partitions = vec![vec![(loom.clone(), (0..loom.outputs().len()).collect(), Placement::Default)]];
    partitions.extend(output_domain_families(&loom, plan_clusters)?.unwrap_or_default());
    let mut candidates = Vec::new();
    for partition in partitions {
        let producer = super::partitions::PartitionCandidates::new(partition, limits,
            loom.num_inputs, loom.outputs().len(), &mut profile, &mut || Ok(()));
        match producer {
            Ok(mut producer) => {
                if let Some(Ok(candidate)) = producer.next(&mut profile, &mut || Ok(())) { candidates.push(candidate); }
            }
            Err(error) if candidates.is_empty() => return Err(error),
            Err(_) => {}
        }
    }
    Ok((combine_compiled(candidates, loom.num_inputs)?, profile))
}

/// Cover every described launch slot while fixing its siblings at their first
/// slot. The Cartesian-product tests cover combinations of these axes.
fn compile_slot_variations_profiled(
    loom: LiftedProgram, options: CompileOptions, statement_cap: usize,
    register_cap: usize, cut_limit: usize, plan_clusters: bool,
) -> Result<(CompiledProgram, CompileProfile), String> {
    use super::configurations::Configuration;
    let settings = BackendSettings::default();
    let limits = CompileLimits { statement_cap, register_cap, cut_limit, plan_clusters,
        max_workspace_bytes: options.max_workspace_bytes, element_order: ElementOrder::Trace,
        target_sm: TEST_TARGET_SM, dump_ptx: options.keep_ptx, record_structure: options.record_dependencies_and_partitions,
        settings: &settings };
    let mut profile = CompileProfile::default();
    let planned = super::pipeline::plan_single_with_limits_profiled(
        loom.clone(), limits, Placement::Default, &mut profile)?;
    let table = if loom.has_indexed_ops() { SlotTable::indexed(&settings.compiler.block_sizes) }
        else { SlotTable::new(&settings.compiler.block_sizes, &settings.compiler) };
    let render = Render { loom: &planned.loom, execution: &planned.execution,
        incidence: &planned.incidence, settings: &settings, table: &table,
        sm: TEST_TARGET_SM, statement_cap, required: true };
    let mut configurations = Vec::new();
    for candidate in planned.estimates.baseline_index.into_iter()
        .chain(planned.estimates.alternative_indices.iter().copied()) {
        let menus = (0..planned.execution.candidates[candidate].launches.len())
            .map(|launch| render.describe_slots(&planned.estimates, candidate, launch))
            .collect::<Vec<_>>();
        let Some(first) = menus.iter().map(|slots| slots.first().copied()).collect::<Option<Vec<_>>>() else { continue; };
        let configuration = |slots| Configuration::Launches { candidate, slots,
            buckets: vec![1; loom.outputs().len()], tile_widths: vec![None; loom.outputs().len()] };
        configurations.push(configuration(first.clone()));
        for (launch, menu) in menus.iter().enumerate() {
            for &slot in &menu[1..] {
                let mut slots = first.clone();
                slots[launch] = slot;
                configurations.push(configuration(slots));
            }
        }
    }
    configurations.extend(settings.compiler.reduction_block_sizes.into_iter().map(Configuration::SingleBlockReduction));
    let mut producer = super::materialize::SingleCandidates::new(planned, limits, Placement::Default, true, &mut profile)?;
    let mut candidates = Vec::new();
    let mut refusal = None;
    for configuration in configurations {
        match producer.materialize(configuration, &mut profile, &mut || Ok(())) {
            Ok(candidate) => {
                assert_complete(candidate.recipe.body());
                candidates.push(candidate);
            }
            Err(error) => { refusal.get_or_insert(error); }
        }
    }
    if candidates.is_empty() { return Err(refusal.unwrap_or_else(|| "no described slots".into())); }
    Ok((combine_compiled(candidates, loom.num_inputs)?, profile))
}

fn assert_complete(body: &::recipe::RecipeBody) {
    for node in body.nodes() {
        if let Work::Choice { candidates, .. } = node.work() {
            assert_eq!(candidates.len(), 1, "a complete leaf cannot have an open choice");
            assert_complete(&candidates[0]);
        }
    }
}

fn single_launch_work(body: &::recipe::RecipeBody) -> &Work {
    let [node] = body.nodes() else { panic!("expected a single launch"); };
    match node.work() {
        Work::Choice { candidates, .. } => {
            let [candidate] = candidates.as_slice() else { panic!("expected a complete selection"); };
            single_launch_work(candidate)
        }
        work @ Work::Call { .. } => work,
        _ => panic!("expected a kernel call"),
    }
}

