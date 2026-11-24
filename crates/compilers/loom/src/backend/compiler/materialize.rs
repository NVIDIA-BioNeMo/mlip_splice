// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, CompileLimits, CompileProfile, CompiledProgram, HashMap, ModuleSource,
    Placement, PlannedSingle, ProfilePhase, Render, SlotTable,
    configurations::{Configuration, Configurations},
    emit_planned_single_block_reduction_kernel, lower_candidate,
    lower_single_block_reduction_candidate, ptx_target, reduction,
    render::RenderCache,
    request_modules, unit_records,
};
use crate::domain::reduction::Projection;

/// An immutable plan owns the materialization cache for its target and limits.
pub(super) struct SingleCandidates {
    planned: PlannedSingle,
    table: SlotTable,
    configurations: Configurations,
    cache: RenderCache,
    modules: HashMap<(usize, usize, usize), ModuleSource>,
    /// The accuracy figure each rendered slot carries, kept beside its module so
    /// a cache hit reports the same chain the miss measured.
    chains: HashMap<(usize, usize, usize), Option<gpu_ir::AdditionChain>>,
    projections: HashMap<(usize, bool), ModuleSource>,
    projection_chains: HashMap<(usize, bool), Option<gpu_ir::AdditionChain>>,
    reductions: HashMap<u32, Result<ModuleSource, String>>,
    settings: BackendSettings,
    sm: u16,
    statement_cap: usize,
    max_workspace_bytes: Option<usize>,
    dump_ptx: bool,
    placement: Placement,
    required: bool,
}

impl SingleCandidates {
    pub(super) fn new(
        planned: PlannedSingle,
        limits: CompileLimits,
        placement: Placement,
        required: bool,
        profile: &mut CompileProfile,
    ) -> Result<Self, String> {
        let table = if planned.loom.has_indexed_ops() {
            SlotTable::indexed(&limits.settings.compiler.block_sizes)
        } else {
            SlotTable::new(
                &limits.settings.compiler.block_sizes,
                &limits.settings.compiler,
            )
        };
        let render = Render {
            loom: &planned.loom,
            execution: &planned.execution,
            incidence: &planned.incidence,
            settings: limits.settings,
            table: &table,
            sm: limits.target_sm,
            statement_cap: limits.statement_cap,
            required,
        };
        profile
            .planned_candidates
            .extend(super::telemetry::describe_planned_candidates(
                &render,
                &planned.estimates,
                limits,
                placement,
            ));
        let configurations = Configurations::new(&render, &planned.estimates)?;
        Ok(Self {
            planned,
            table,
            configurations,
            cache: RenderCache::default(),
            modules: HashMap::new(),
            chains: HashMap::new(),
            projections: HashMap::new(),
            projection_chains: HashMap::new(),
            reductions: HashMap::new(),
            settings: *limits.settings,
            sm: limits.target_sm,
            statement_cap: limits.statement_cap,
            max_workspace_bytes: limits.max_workspace_bytes,
            dump_ptx: limits.dump_ptx,
            placement,
            required,
        })
    }

    pub(super) fn configurations(&self) -> Configurations {
        self.configurations.clone()
    }

    pub(super) fn materialize(
        &mut self,
        configuration: Configuration,
        profile: &mut CompileProfile,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<CompiledProgram, String> {
        checkpoint()?;
        let recipe = match configuration {
            Configuration::SingleBlockReduction(block) => {
                self.request_reduction(block, profile, checkpoint)?;
                let module = self.reductions[&block].as_ref().map_err(Clone::clone)?;
                checkpoint()?;
                let phase = ProfilePhase::new("build_recipe_ms", &mut profile.timings.build_recipe);
                let recipe = lower_single_block_reduction_candidate(
                    &self.planned.loom,
                    module,
                    block,
                    &self.planned.signature,
                )?;
                drop(phase);
                recipe
            }
            Configuration::Launches {
                candidate,
                slots,
                buckets,
                tile_widths,
            } => {
                let mut modules = Vec::with_capacity(slots.len());
                let mut chains = Vec::with_capacity(slots.len());
                for (launch, slot) in slots.into_iter().enumerate() {
                    checkpoint()?;
                    let key = (candidate, launch, slot);
                    if !self.modules.contains_key(&key) {
                        let planned = &self.planned;
                        let render = Render {
                            loom: &planned.loom,
                            execution: &planned.execution,
                            incidence: &planned.incidence,
                            settings: &self.settings,
                            table: &self.table,
                            sm: self.sm,
                            statement_cap: self.statement_cap,
                            required: self.required,
                        };
                        let phase =
                            ProfilePhase::new("render_ptx_ms", &mut profile.timings.render_ptx);
                        let alias = render.slot_alias(
                            &mut self.cache,
                            candidate,
                            launch,
                            slot,
                            checkpoint,
                        )?;
                        drop(phase);
                        checkpoint()?;
                        self.chains.insert(key, alias.addition_chain);
                        let phase = ProfilePhase::new(
                            "state_requests_ms",
                            &mut profile.timings.state_requests,
                        );
                        let (requested, _) = request_modules(
                            std::slice::from_ref(&alias),
                            &[],
                            self.settings.compiler.register_cap_spill_band_bytes,
                            ptx_target(self.sm),
                        )?;
                        profile.kernels.extend(unit_records(
                            std::slice::from_ref(&alias),
                            self.dump_ptx,
                            self.placement,
                        ));
                        self.modules.insert(
                            key,
                            requested.into_iter().next().expect("one requested slot"),
                        );
                        drop(phase);
                    }
                    modules.push(vec![(slot, self.modules[&key].clone())]);
                    chains.push((slot, self.chains[&key]));
                }
                let mut projections = Vec::with_capacity(buckets.len());
                let mut projection_chains = Vec::with_capacity(buckets.len());
                for (output, (&buckets, tile_width)) in
                    buckets.iter().zip(tile_widths.iter().copied()).enumerate()
                {
                    checkpoint()?;
                    let projection = self
                        .planned
                        .execution
                        .domain
                        .folds
                        .output(output)
                        .projection(buckets);
                    let selected = match projection {
                        None => None,
                        Some(projection) => {
                            let key = (output, matches!(projection, Projection::Fold { .. }));
                            if !self.projections.contains_key(&key) {
                                let phase = ProfilePhase::new(
                                    "render_ptx_ms",
                                    &mut profile.timings.render_ptx,
                                );
                                let kernel = reduction::projection_kernel(
                                    &self.planned.execution.domain,
                                    &self.planned.loom,
                                    output,
                                    projection,
                                )?;
                                checkpoint()?;
                                let entry = kernel.function.name.clone();
                                self.projection_chains
                                    .insert(key, gpu_ir::longest_addition_chain(kernel.body()));
                                let source = gpu_ir::ptx::render_module(
                                    std::slice::from_ref(&kernel),
                                    i32::from(self.sm),
                                )
                                .map_err(|error| error.to_string())?;
                                drop(phase);
                                let phase = ProfilePhase::new(
                                    "state_requests_ms",
                                    &mut profile.timings.state_requests,
                                );
                                let (_, requested) = request_modules(
                                    &[],
                                    &[(source.as_bytes(), &entry)],
                                    self.settings.compiler.register_cap_spill_band_bytes,
                                    ptx_target(self.sm),
                                )?;
                                self.projections.insert(
                                    key,
                                    requested.into_iter().next().expect("one projection"),
                                );
                                profile.kernels.push(super::telemetry::kernel_unit_record(
                                    &[kernel],
                                    source.as_bytes(),
                                    &entry,
                                    self.dump_ptx,
                                    self.placement,
                                    "projection",
                                ));
                                drop(phase);
                            }
                            projection_chains.push(Some(self.projection_chains[&key]));
                            Some(reduction::ProjectionModule {
                                source: self.projections[&key].clone(),
                                tile_width,
                            })
                        }
                    };
                    if projection.is_none() {
                        projection_chains.push(None);
                    }
                    projections.push(selected);
                }
                checkpoint()?;
                let phase = ProfilePhase::new("build_recipe_ms", &mut profile.timings.build_recipe);
                let planned = &self.planned;
                let recipe = lower_candidate(
                    &planned.execution.domain,
                    &planned.loom,
                    &planned.execution.candidates[candidate],
                    &self.table.specs,
                    &planned.signature,
                    &vec![],
                    Some(modules),
                    &projections,
                    &buckets,
                )?;
                drop(phase);
                profile
                    .addition_depths
                    .extend(super::accumulation::describe(
                        &planned.execution.domain,
                        &planned.execution.candidates[candidate],
                        planned.loom.outputs().len(),
                        &chains,
                        &self.table.specs,
                        &buckets,
                        &projection_chains,
                        candidate,
                        self.placement,
                    )?);
                profile
                    .fold_buckets
                    .extend(super::pipeline::describe_fold_candidates(
                        &planned.execution.domain,
                        &planned.loom,
                        &buckets,
                        &tile_widths,
                        planned.estimates.baseline_index == Some(candidate),
                        self.placement,
                    )?);
                recipe
            }
        };
        if !super::within_workspace_bound(&recipe, self.max_workspace_bytes) {
            return Err(super::workspace_limit_error(self.max_workspace_bytes));
        }
        Ok(CompiledProgram {
            recipe: recipe.into_recipe(),
            input_count: self.planned.signature.num_inputs(),
            output_count: self.planned.loom.outputs().len(),
            index_arrays: self.planned.execution.domain.slots.clone(),
        })
    }

    fn request_reduction(
        &mut self,
        block: u32,
        profile: &mut CompileProfile,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<(), String> {
        if self.reductions.contains_key(&block) {
            return Ok(());
        }
        let phase = ProfilePhase::new("render_ptx_ms", &mut profile.timings.render_ptx);
        let candidate = self
            .planned
            .estimates
            .baseline_index
            .ok_or("no baseline reduction")?;
        let kernel = match emit_planned_single_block_reduction_kernel(
            &self.planned.execution.domain,
            &self.planned.loom,
            &self.planned.execution.candidates[candidate],
            block,
        ) {
            Ok(kernel) => kernel,
            Err(error) => {
                self.reductions.insert(block, Err(error));
                return Ok(());
            }
        };
        checkpoint()?;
        let entry = kernel.function.name.clone();
        let source = gpu_ir::ptx::render_module(std::slice::from_ref(&kernel), i32::from(self.sm))
            .map_err(|error| error.to_string())?;
        drop(phase);
        let phase = ProfilePhase::new("state_requests_ms", &mut profile.timings.state_requests);
        let (_, modules) = request_modules(
            &[],
            &[(source.as_bytes(), &entry)],
            self.settings.compiler.register_cap_spill_band_bytes,
            ptx_target(self.sm),
        )?;
        self.reductions.insert(
            block,
            Ok(modules.into_iter().next().expect("one reduction")),
        );
        profile.kernels.push(super::telemetry::kernel_unit_record(
            &[kernel],
            source.as_bytes(),
            &entry,
            self.dump_ptx,
            self.placement,
            "single-block-reduction",
        ));
        drop(phase);
        Ok(())
    }
}

#[cfg(test)]
#[path = "materialize_tests.rs"]
mod tests;
