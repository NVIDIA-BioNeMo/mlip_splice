// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, CandidateFlavor, CandidateRank, DomainIndex, HashMap, IncidenceProfile,
    Launch, LaunchEstimates, LiftedProgram, PlanIr, SlotSpec, SlotTable, SourceAlias,
    emit_planned_geometry_kernel, emit_planned_kernel, emit_planned_lane_hoisted_kernel,
    emit_planned_passes_kernel, emit_planned_sharded_kernel,
    emit_planned_zero_skip_geometry_kernel, emit_planned_zero_skip_kernel, slot_weight,
    statement_count,
};

#[derive(Clone)]
struct Rendered {
    source: std::sync::Arc<Vec<u8>>,
    entry: String,
    statements: usize,
    addition_chain: Option<gpu_ir::AdditionChain>,
}

/// This cache belongs to exactly one immutable plan and target. Its key adds
/// structural candidate, launch, and every source-affecting geometry axis.
#[derive(Default)]
pub(super) struct RenderCache {
    sources: HashMap<(usize, usize, SlotSpec), Result<Rendered, String>>,
}

/// Everything the slot renders read: the planned candidates, the table of
/// arrangements they may take, and the machine they are rendered for.
pub(super) struct Render<'a> {
    pub(super) loom: &'a LiftedProgram,
    pub(super) execution: &'a PlanIr,
    pub(super) incidence: &'a std::cell::OnceCell<Result<IncidenceProfile, String>>,
    pub(super) settings: &'a BackendSettings,
    pub(super) table: &'a SlotTable,
    pub(super) sm: u16,
    pub(super) statement_cap: usize,
    pub(super) required: bool,
}

impl Render<'_> {
    /// Describe possible slots without emitting kernel IR or PTX. Emission may
    /// still refuse a described slot; that refuses its entire configuration.
    pub(super) fn describe_slots(
        &self,
        estimates: &LaunchEstimates,
        candidate: usize,
        launch: usize,
    ) -> Vec<usize> {
        let alternative = estimates.baseline_index != Some(candidate);
        let statements = self.launch_statements(estimates, candidate, launch);
        let producer = matches!(
            self.execution.candidates[candidate].launches[launch],
            Launch::Producer { .. }
        );
        let mut slots = self
            .table
            .specs
            .iter()
            .enumerate()
            .filter(|(slot, spec)| {
                if producer {
                    return *slot == 0
                        || (Some(*slot) == self.table.first_regcap
                            && statements > self.settings.partition.register_cap_statement_floor);
                }
                if alternative
                    && (!matches!(
                        spec.flavor,
                        CandidateFlavor::Baseline
                            | CandidateFlavor::RegCap
                            | CandidateFlavor::Geometry
                    ) || spec.shards != 1
                        || spec.passes != 1)
                {
                    return false;
                }
                match spec.flavor {
                    CandidateFlavor::RegCap => {
                        statements > self.settings.partition.register_cap_statement_floor
                    }
                    CandidateFlavor::ZeroSkip => {
                        estimates.zero_skip.get(launch).is_some_and(Option::is_some)
                    }
                    CandidateFlavor::LaneHoist => {
                        self.required && estimates.hoist.get(launch).is_some_and(Option::is_some)
                    }
                    CandidateFlavor::Geometry if spec.passes > 1 => {
                        statements > self.settings.partition.register_cap_statement_floor
                    }
                    _ => true,
                }
            })
            .map(|(slot, spec)| {
                let weight =
                    slot_weight(self.execution.domain.index.fiber_bridge().is_some(), spec);
                let count = statements * usize::from(spec.io_lanes);
                (
                    slot,
                    weight / self.settings.compiler.candidate_cost_ms(count),
                )
            })
            .collect::<Vec<_>>();
        slots.sort_by(|a, b| {
            (b.0 == 0)
                .cmp(&(a.0 == 0))
                .then(b.1.total_cmp(&a.1))
                .then(a.0.cmp(&b.0))
        });
        slots.into_iter().map(|(slot, _)| slot).collect()
    }

    fn launch_statements(
        &self,
        estimates: &LaunchEstimates,
        candidate: usize,
        launch: usize,
    ) -> usize {
        if estimates.baseline_index == Some(candidate) {
            estimates.baseline[launch]
        } else {
            let ordinal = estimates
                .alternative_indices
                .iter()
                .position(|&index| index == candidate)
                .expect("described candidate");
            estimates.alternatives[ordinal][launch]
        }
    }

    pub(super) fn slot_alias(
        &self,
        cache: &mut RenderCache,
        candidate: usize,
        launch: usize,
        slot: usize,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Result<SourceAlias, String> {
        checkpoint()?;
        let spec = self.table.specs[slot];
        let mut source_spec = spec;
        source_spec.register_cap = None;
        if matches!(
            spec.flavor,
            CandidateFlavor::RegCap | CandidateFlavor::WholeProgram
        ) {
            source_spec.flavor = CandidateFlavor::Baseline;
        }
        let key = (candidate, launch, source_spec);
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.sources.entry(key) {
            let result = match self.emit(candidate, launch, &source_spec) {
                Err(error) => Err(error),
                Ok(kernel) => {
                    checkpoint()?;
                    let statements = statement_count(kernel.body());
                    if statements > self.statement_cap {
                        Err("loom selected slot exceeds statement cap".into())
                    } else {
                        let entry = kernel.function.name.clone();
                        let addition_chain = gpu_ir::longest_addition_chain(kernel.body());
                        gpu_ir::ptx::render_module(&[kernel], i32::from(self.sm))
                            .map(|source| Rendered {
                                source: std::sync::Arc::new(source.into_bytes()),
                                entry,
                                statements,
                                addition_chain,
                            })
                            .map_err(|error| error.to_string())
                    }
                }
            };
            entry.insert(result);
        }
        checkpoint()?;
        let rendered = cache.sources[&key].clone()?;
        if spec.register_cap.is_some()
            && matches!(
                spec.flavor,
                CandidateFlavor::Geometry | CandidateFlavor::ZeroSkip
            )
            && rendered.statements <= self.settings.partition.register_cap_statement_floor / 4
        {
            return Err("loom selected register cap has insufficient pressure".into());
        }
        Ok(SourceAlias {
            source: rendered.source,
            entry: rendered.entry,
            cap: spec.register_cap,
            assembly_mode: if spec.flavor == CandidateFlavor::WholeProgram {
                ::recipe::PtxAssemblyMode::WholeProgram
            } else {
                ::recipe::PtxAssemblyMode::Relocatable
            },
            statements: rendered.statements * usize::from(spec.io_lanes),
            addition_chain: rendered.addition_chain,
            rank: CandidateRank {
                flavor: spec.flavor,
                part: launch,
                block: slot,
            },
        })
    }

    fn emit(
        &self,
        candidate: usize,
        launch: usize,
        spec: &SlotSpec,
    ) -> Result<gpu_ir::Launch, String> {
        let program = &self.execution.domain;
        let candidate = &self.execution.candidates[candidate];
        match spec.flavor {
            CandidateFlavor::Baseline => {
                emit_planned_kernel(program, self.loom, candidate, launch, spec.block_size)
            }
            CandidateFlavor::ZeroSkip if spec.warps == 1 => {
                emit_planned_zero_skip_kernel(program, self.loom, candidate, launch, spec)
            }
            CandidateFlavor::ZeroSkip => {
                emit_planned_zero_skip_geometry_kernel(program, self.loom, candidate, launch, spec)
            }
            CandidateFlavor::LaneHoist => emit_planned_lane_hoisted_kernel(
                program,
                self.loom,
                candidate,
                launch,
                spec.block_size,
            ),
            CandidateFlavor::Geometry if spec.shards > 1 || spec.passes > 1 => {
                let incidence = self
                    .incidence
                    .get_or_init(|| IncidenceProfile::prepare(self.loom))
                    .as_ref()
                    .map_err(Clone::clone)?;
                if spec.shards > 1 {
                    emit_planned_sharded_kernel(
                        program,
                        self.loom,
                        candidate,
                        launch,
                        spec,
                        incidence,
                        &self.settings.partition,
                    )
                } else {
                    emit_planned_passes_kernel(
                        program,
                        self.loom,
                        candidate,
                        launch,
                        spec,
                        incidence,
                        &self.settings.partition,
                    )
                }
            }
            CandidateFlavor::Geometry => {
                emit_planned_geometry_kernel(program, self.loom, candidate, launch, spec)
            }
            _ => unreachable!("assembly options have a source geometry"),
        }
    }
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
