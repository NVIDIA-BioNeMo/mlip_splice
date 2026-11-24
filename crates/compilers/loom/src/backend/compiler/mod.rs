// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{DomainIndex, Slot};
use std::collections::HashMap;
use std::time::Instant;

use super::super::eval::num_elements;
use super::super::progress::{OpenPhase, open_phase};
use super::super::watchdog_event;
use super::BackendSettings;
#[cfg(test)]
use super::codegen::{BLOCK_SIZES, emit_kernel};
use super::codegen::{
    Flavor, Split, WARP_SIZE, emit_consumer_kernel, emit_consumer_kernel_flavor,
    emit_producer_kernel, emit_single_block_reduction_kernel, emit_split_kernel,
};
use super::facts::LoomFacts;
use super::iteration::{Placement, Plan};
#[cfg(test)]
use super::partition::{DEFAULT_MAX_KERNEL_STATEMENTS, plan, plan_shared_work};
use super::partition::{
    ElementOrder, IncidenceProfile, KernelPart, partition_cost, program_order_summaries,
    statement_count,
};
use super::reduction;
use super::schedule::{Candidate, CandidateOrigin, Launch, PlanIr, WriteMode};
use crate::profile::Kernel;
use crate::{CompileOptions, CompileProfile, LiftedProgram};

mod accumulation;
mod arms;
mod bindings;
mod combine;
mod configurations;
mod contributions;
mod emission;
mod estimates;
mod graph_frontier;
mod materialize;
mod product;
use graph_frontier::graph_record;
pub(crate) use graph_frontier::select_graph;
mod lowering;
mod partitions;
mod passes;
mod pipeline;
mod render;
mod signature;
mod slots;
mod sources;
mod target_limits;
mod telemetry;
#[cfg(test)]
mod tests;

pub(crate) use combine::combine_compiled;
pub(crate) use passes::GraphPasses;

use bindings::{
    bind_recipe, consumer_parts, input_bindings, output_domain_families, same_slot,
    select_alternatives,
};
use contributions::{PartialContributionRatio, planned_partial_contribution_ratios};
#[cfg(test)]
use emission::bounded_parts;
pub(super) use emission::emit_planned_kernel;
use emission::{
    candidate_statement_counts, emit_planned_geometry_kernel, emit_planned_lane_hoisted_kernel,
    emit_planned_passes_kernel, emit_planned_sharded_kernel,
    emit_planned_single_block_reduction_kernel, emit_planned_zero_skip_geometry_kernel,
    emit_planned_zero_skip_kernel, lane_hoist_statement_counts, zero_skip_statement_counts,
};
use estimates::LaunchEstimates;
use lowering::{lower_candidate, lower_single_block_reduction_candidate};
use pipeline::plan_single_with_limits_profiled;
use render::Render;
use signature::{SplicedArm, program_signature, shared_signature};
use slots::{CandidateFlavor, SlotSpec, SlotTable, slot_weight};
pub(super) use sources::ModuleSource;
use sources::{FamilyPartition, request_modules, unit_records};

fn within_workspace_bound<F: ::recipe::algebra::Computation>(
    recipe: &::recipe::ProvedRecipe<F>,
    max_workspace_bytes: Option<usize>,
) -> bool {
    max_workspace_bytes
        .is_none_or(|limit| ::recipe::algebra::workspace_upper_bound(recipe.recipe()) <= limit)
}

fn workspace_limit_error(limit: Option<usize>) -> String {
    format!(
        "loom has no candidate within max_workspace_bytes={}",
        limit.expect("a workspace limit removed every candidate")
    )
}

/// What compiling one Loom program yields: the recipe, how many buffers it
/// binds (data inputs plus the arrays its domain-index slots contribute), how
/// many it writes, and the slots themselves.
#[derive(Clone)]
pub struct CompiledProgram {
    pub recipe: ::recipe::Recipe,
    pub input_count: usize,
    pub output_count: usize,
    pub index_arrays: Vec<Slot>,
}

struct PlannedSingle {
    loom: LiftedProgram,
    execution: PlanIr,
    estimates: LaunchEstimates,
    incidence: std::cell::OnceCell<Result<IncidenceProfile, String>>,
    signature: ::recipe::RecipeSignature,
}

/// The machine this compile renders for: what the caller stated, or the device
/// it is running on. A compile that can read neither has no target to state.
fn render_target(options: CompileOptions) -> Result<u16, String> {
    if let Some(target) = options.target {
        return Ok(target.sm);
    }
    let sm = cuda_driver::Device::current()
        .sm_arch()
        .map_err(|error| format!("loom has no device to read its render target from: {error}"))?;
    u16::try_from(sm).map_err(|_| format!("loom render target sm_{sm} is not a real machine"))
}

/// What a rendered module targets: the ISA the renderer writes for this machine,
/// and the machine itself.
fn ptx_target(sm: u16) -> ::recipe::PtxTarget {
    ::recipe::PtxTarget {
        isa: gpu_ir::ptx::isa_version(i32::from(sm)),
        minimum_sm: sm,
    }
}

/// The machine every test compiles for, so no test needs a device to render.
#[cfg(test)]
pub(super) const TEST_TARGET_SM: u16 = 90;

/// A whole sm_90 target, so a test can render and shortlist with no device.
#[cfg(test)]
pub(crate) fn test_target() -> crate::GpuTarget {
    crate::GpuTarget {
        sm: TEST_TARGET_SM,
        sm_count: 132,
        warp_size: 32,
        max_threads_per_sm: 2048,
        max_blocks_per_sm: 32,
    }
}

/// Planner limits shared by every candidate compile of one program.
#[derive(Clone, Copy)]
struct CompileLimits<'a> {
    statement_cap: usize,
    register_cap: usize,
    cut_limit: usize,
    plan_clusters: bool,
    max_workspace_bytes: Option<usize>,
    element_order: ElementOrder,
    /// The machine every kernel is rendered for and every compile request
    /// declares.
    target_sm: u16,
    dump_ptx: bool,
    /// Record the structural baseline for every partition this arm plans.
    record_structure: bool,
    settings: &'a BackendSettings,
}

type CompiledParts = Vec<Vec<Option<ModuleSource>>>;
#[derive(Clone, Copy, Debug)]
pub(super) struct Settings {
    pub compile_ms_per_unit: f64,
    pub compile_ms_per_statement: f64,
    pub compile_ms_per_statement_squared: f64,
    pub register_caps: [u32; 3],
    pub geometry_crosses: [(u8, u8); 9],
    pub split_crosses: [(u8, u8); 6],
    pub split_counts: [u8; 2],
    pub reduction_block_sizes: [u32; 4],
    pub fine_cap_divisor: usize,
    pub fine_cap_floor: usize,
    pub block_sizes: [u32; 4],
    /// A register-capped variant joins its launch choice only while its spill
    /// frame stays within this many bytes: on this workload class spilled
    /// caps lose on the extra DRAM traffic before occupancy can pay it back.
    pub register_cap_spill_band_bytes: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            compile_ms_per_unit: 8.0,
            compile_ms_per_statement: 0.030,
            compile_ms_per_statement_squared: 2.4e-6,
            register_caps: [168, 130, 128],
            geometry_crosses: [
                (4, 1),
                (2, 1),
                (1, 2),
                (1, 4),
                (1, 8),
                (4, 2),
                (2, 2),
                (4, 4),
                (2, 4),
            ],
            split_crosses: [(4, 1), (4, 4), (4, 2), (2, 2), (1, 2), (1, 1)],
            split_counts: [2, 3],
            reduction_block_sizes: [128, 256, 512, 1024],
            fine_cap_divisor: 4,
            fine_cap_floor: 256,
            block_sizes: [32, 64, 128, 256],
            register_cap_spill_band_bytes: 1024,
        }
    }
}

impl Settings {
    fn candidate_cost_ms(&self, statements: usize) -> f64 {
        let n = statements as f64;
        self.compile_ms_per_unit
            + n * self.compile_ms_per_statement
            + n * n * self.compile_ms_per_statement_squared
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateRank {
    flavor: CandidateFlavor,
    part: usize,
    block: usize,
}

#[derive(Clone, Debug)]
struct SourceAlias {
    source: std::sync::Arc<Vec<u8>>,
    entry: String,
    /// A register-capped clone shares its base kernel's render byte for byte —
    /// the cap is a compile option over that source, not another render, so it
    /// launches the base entry.
    cap: Option<u32>,
    /// How the assembler must consume this render. A whole-program clone shares
    /// the base render and differs only here.
    assembly_mode: ::recipe::PtxAssemblyMode,
    statements: usize,
    /// The longest addition chain this render's arena builds, as the accuracy
    /// analysis reads it; `None` when that analysis finds no bound.
    addition_chain: Option<gpu_ir::AdditionChain>,
    rank: CandidateRank,
}

struct ProfilePhase<'a> {
    name: &'static str,
    started: Instant,
    total: &'a mut std::time::Duration,
    /// Held for the phase's life, so a phase still open after a minute is
    /// announced; dropped with the guard.
    _open: OpenPhase,
}

impl<'a> ProfilePhase<'a> {
    fn new(name: &'static str, total: &'a mut std::time::Duration) -> Self {
        watchdog_event("start", name);
        Self {
            name,
            started: Instant::now(),
            total,
            _open: open_phase(name),
        }
    }
}

impl Drop for ProfilePhase<'_> {
    fn drop(&mut self) {
        *self.total += self.started.elapsed();
        watchdog_event("end", self.name);
    }
}
