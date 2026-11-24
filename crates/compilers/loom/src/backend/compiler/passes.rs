// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    BackendSettings, CompileLimits, CompileOptions, CompileProfile, CompiledProgram, ElementOrder,
    FamilyPartition, LiftedProgram, Placement, partitions::PartitionCandidates,
};
use std::collections::VecDeque;

type Pass = (usize, bool, ElementOrder);

pub(crate) struct GraphPasses {
    program: LiftedProgram,
    settings: BackendSettings,
    options: CompileOptions,
    passes: std::vec::IntoIter<Pass>,
    partitions: VecDeque<(FamilyPartition, Pass)>,
    active: VecDeque<(PartitionCandidates, Pass)>,
    described_orders: bool,
}

impl GraphPasses {
    pub(crate) fn new(program: LiftedProgram, options: CompileOptions) -> Result<Self, String> {
        let settings = super::target_limits::configure_backend_settings(&program, options)?;
        let passes = super::arms::compile_passes(&program, options, &settings)?.into_iter();
        Ok(Self {
            program,
            settings,
            options,
            passes,
            partitions: VecDeque::new(),
            active: VecDeque::new(),
            described_orders: false,
        })
    }

    pub(crate) fn graph_record(
        &self,
        graph: usize,
        frontier_size: usize,
    ) -> crate::profile::FrontierGraph {
        super::graph_record(&self.program, graph, frontier_size)
    }

    pub(crate) fn next(
        &mut self,
        checkpoint: &mut impl FnMut() -> Result<(), String>,
    ) -> Option<(Result<CompiledProgram, String>, CompileProfile)> {
        let mut profile = CompileProfile::default();
        loop {
            let opened = (|| {
                checkpoint()?;
                if !self.described_orders {
                    profile.element_orders = super::arms::describe_orders(&self.program);
                    self.described_orders = true;
                    checkpoint()?;
                }
                let pending = if let Some(pass) = self.passes.next() {
                    let families = super::output_domain_families(&self.program, pass.1)
                        .ok()
                        .flatten();
                    if let Some(families) = families {
                        self.partitions
                            .extend(families.into_iter().map(|partition| (partition, pass)));
                    }
                    Some((
                        vec![(
                            self.program.clone(),
                            (0..self.program.outputs().len()).collect(),
                            Placement::Default,
                        )],
                        pass,
                    ))
                } else {
                    self.partitions.pop_front()
                };
                if let Some((partition, pass)) = pending {
                    let (statement_cap, plan_clusters, element_order) = pass;
                    let limits = CompileLimits {
                        statement_cap,
                        plan_clusters,
                        element_order,
                        register_cap: self.settings.schedule.default_register_cap,
                        cut_limit: self.settings.schedule.default_cut_limit,
                        max_workspace_bytes: self.options.max_workspace_bytes,
                        target_sm: super::render_target(self.options)?,
                        dump_ptx: self.options.keep_ptx,
                        record_structure: self.options.record_dependencies_and_partitions,
                        settings: &self.settings,
                    };
                    let candidates = PartitionCandidates::new(
                        partition,
                        limits,
                        self.program.num_inputs,
                        self.program.outputs().len(),
                        &mut profile,
                        checkpoint,
                    )?;
                    self.active.push_back((candidates, pass));
                }
                Ok::<_, String>(())
            })();
            if let Err(error) = opened {
                return Some((Err(error), profile));
            }
            while let Some((mut candidates, pass)) = self.active.pop_front() {
                if let Some(result) = candidates.next(&mut profile, checkpoint) {
                    self.active.push_back((candidates, pass));
                    if result.is_ok() {
                        profile.arms.push(crate::profile::Arm {
                            graph: 0,
                            order: pass.2.name(),
                            statement_cap: pass.0,
                            plan_clusters: pass.1,
                        });
                    }
                    for unit in &mut profile.kernels {
                        unit.order = pass.2.name();
                        unit.statement_cap = pass.0;
                    }
                    return Some((result, profile));
                }
            }
            if self.passes.len() == 0 && self.partitions.is_empty() {
                return None;
            }
        }
    }
}
