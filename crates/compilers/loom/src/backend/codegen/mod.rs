// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{AxisBridge, DomainIndex, operand_has_alive};
use gpu_ir::{
    Builder, Declaration, DeviceFunction, Evict, Launch, OpId, Param, ParamTy, PtrTy, Scalar,
    SharedArr, Space, Ty, Value, ValueId,
};
use xir::dtype::DType;
use xir::ops::ScalarOp;

use super::super::LiftedProgram;
use super::super::program::{InputElement, OutputFactor};
use super::name::{KernelName, Tuning};
use crate::domain::{BucketMaskValues, EmitCtx, GpuIndex, ParameterValues};

use super::super::eval::{
    Evaluator, convert_output_factor, math_scalar, num_elements, output_factor_elementwise,
    output_scalar, scalar_constant, zero,
};
use super::iteration::{Index, Plan, storage_scalar};
use super::partition::{
    IncidenceProfile, KernelPart, SelectedOutputs, Settings as PartitionSettings,
    contract_fma_bindings, group_bindings_by_element, selected_bindings, selected_term_values,
    slice_bindings,
};
use super::reduction;
use super::schedule::{GuardedPlan, GuardedRegion, MaterializedValue};

mod accumulate;
mod block_reduction;
mod consumer;
mod evaluation;
mod fiber_walk;
mod flavor;
mod lane_sweep;
mod producer;
mod writes;

use accumulate::vector_reduction_scalar;
pub(super) use block_reduction::emit_single_block_reduction_kernel;
pub(super) use consumer::emit_consumer_kernel_flavor;
use evaluation::{
    At, Folded, access_substitutions, build_input_access, evaluate_at, expand_program_for_lanes,
    guarded_body, inactive_direct_writes, materialized_name, materialized_param,
    materialized_params,
};
pub(super) use flavor::Flavor;
use writes::{Emit, Written, fiber_scatter_writes, output_writes};

pub(super) use producer::emit_producer_kernel;

pub(super) const WARP_SIZE: u32 = 32;
#[cfg(test)]
pub(super) const BLOCK_SIZES: [u32; 4] = [32, 64, 128, 256];

/// Where operand `operand`'s element lives, as a value of the body being built.
fn operand_flat_index<I: DomainIndex>(
    ctx: &EmitCtx<'_, I>,
    b: &mut gpu_ir::Builder,
    operand: usize,
    fiber_edge: Option<gpu_ir::Value>,
) -> gpu_ir::Value {
    domain_index::operand_flat_index(ctx, &mut GpuIndex(b), operand, fiber_edge)
}

/// The predicate operand `operand` is mapped under, when a bridge leaves cells
/// unmapped and it therefore has one.
fn operand_alive<I: DomainIndex>(
    ctx: &EmitCtx<'_, I>,
    b: &mut gpu_ir::Builder,
    operand: usize,
) -> Option<gpu_ir::Value> {
    domain_index::operand_alive(ctx, &mut GpuIndex(b), operand)
}

fn push_axis_indices(
    body: &mut gpu_ir::Builder,
    flat: &gpu_ir::Value,
    sizes: &[i64],
) -> Vec<gpu_ir::Value> {
    let mut axis_indices = Vec::with_capacity(sizes.len());
    for axis in 0..sizes.len() {
        if sizes[axis] == 1 {
            axis_indices.push(body.i64(0));
            continue;
        }
        let stride = sizes[axis + 1..].iter().product::<i64>();
        let mut value = flat.clone();
        if stride != 1 {
            let stride = body.i64(stride);
            value = body.div(value, stride);
        }
        if axis != 0 {
            let extent = body.i64(sizes[axis]);
            value = body.rem(value, extent);
        }
        let name = format!("loom_axis_{axis}");
        axis_indices.push(body.bind_value_as(name, Ty::scalar(Scalar::S64), value));
    }
    axis_indices
}

pub(super) fn emit_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part_index: usize,
    num_parts: usize,
    part: &KernelPart,
) -> Result<Launch, String> {
    emit_consumer_kernel(program, loom, block_size, part_index, num_parts, part, &[])
}

pub(super) fn emit_consumer_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part_index: usize,
    num_parts: usize,
    part: &KernelPart,
    materialized: &[MaterializedValue],
) -> Result<Launch, String> {
    let plain = Flavor::plain(block_size, part_index, num_parts);
    Ok(emit_consumer_kernel_flavor(
        program,
        loom,
        part,
        Flavor {
            materialized,
            ..plain
        },
    )?
    .kernel)
}

/// The launch structure a whole-element split takes.
///
/// `Shards` gives the grid a y-axis of sibling branches, one per group: each
/// block executes only its group's closure — the loom analog of the descriptor
/// backend's `spp`. Narrower per-block read sets and `n`x the resident blocks
/// trade against re-reading inputs the groups share.
///
/// `Passes` is its sequential dual: the same groups run one after another in
/// sibling scopes of one block, so only one group's loads and accumulators are
/// live at a time — the analog of `pp`. Lower peak liveness trades against
/// recomputing whatever the groups' closures share.
pub(super) enum Split {
    Shards(u8),
    Passes(u8),
}

impl Split {
    fn groups(&self) -> u8 {
        match self {
            Split::Shards(groups) | Split::Passes(groups) => *groups,
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            Split::Shards(_) => "shard",
            Split::Passes(_) => "pass split pp",
        }
    }

    /// The name tuning this split carries on top of the group flavor.
    fn tuning(&self, flavor: Flavor<'_>) -> Tuning {
        let (element_shards, passes) = match self {
            Split::Shards(shards) => (*shards, 1),
            Split::Passes(passes) => (1, *passes),
        };
        Tuning {
            element_shards,
            passes,
            ..flavor.tuning(None)
        }
    }

    /// Fold one group's body into the merged one, after the shared position
    /// prologue.
    fn merge(&self, b: &mut Builder, group: usize, body: Vec<OpId>) {
        match self {
            Split::Shards(_) => {
                let branch = b.ctaid_y();
                let branch = b.cast(Scalar::S32, branch);
                let wanted = b.i32(group as i64);
                let taken = b.eq(branch, wanted);
                b.if_(taken, |arm| arm.place_all(body));
            }
            Split::Passes(_) => b.scoped(|arm| arm.place_all(body)),
        }
    }
}

/// One kernel per whole-element group of the part, merged into a single entry
/// the way `split` prescribes.
pub(super) fn emit_split_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    part: &KernelPart,
    flavor: Flavor<'_>,
    split: Split,
    profile: &IncidenceProfile,
    settings: &PartitionSettings,
) -> Result<Launch, String> {
    let groups = part
        .shard(usize::from(split.groups()), profile, settings)
        .ok_or_else(|| {
            format!(
                "loom {}{} finds no narrowing element split",
                split.tag(),
                split.groups()
            )
        })?;
    let emitted = groups
        .iter()
        .map(|group| emit_consumer_kernel_flavor(program, loom, group, flavor))
        .collect::<Result<Vec<_>, String>>()?;
    // Every group body opens with the same position prologue (lane, flat
    // position, bounds guard, axis indices); it is hoisted from the first
    // group and the per-group remainders become the split's siblings. Each
    // group reports where its own prologue ends, and they must agree — one
    // group prefixed differently would put a sibling's statements in the
    // shared opening.
    let prologue_end = emitted[0].prologue_end;
    assert!(
        emitted
            .iter()
            .all(|group| group.prologue_end == prologue_end),
        "the groups of one split opened with prologues of different lengths"
    );
    let mut kernels = emitted.into_iter().map(|group| group.kernel);
    let first = kernels.next().expect("at least one group");
    let max_threads = first.max_threads;
    let max_registers = first.max_registers;
    let mut merged = Builder::resuming(gpu_ir::Body::of(first.into_body()));
    let mut body = merged.steps();
    assert!(
        body.len() >= prologue_end,
        "a split group's body is shorter than the prologue it reported"
    );
    let own = body.split_off(prologue_end);
    split.merge(&mut merged, 0, own);
    for (group, kernel) in kernels.enumerate() {
        let source = kernel.into_body();
        // The groups open on the same prologue, so the merged one's values
        // stand for the group's: that is the whole substitution, and it is what
        // makes a group's body readable where it lands.
        let substitutions = prologue_substitutions(&mut merged, &source, prologue_end)?;
        let steps = source.block_steps(source.entry()).collect::<Vec<_>>()[prologue_end..].to_vec();
        let imported = merged.import_ops(&source, &steps, &substitutions)?;
        let own = imported.into_ops();
        split.merge(&mut merged, group + 1, own);
    }
    Ok(Launch::new(
        DeviceFunction::new(
            KernelName::consumer(&loom.name, flavor.block_size, split.tuning(flavor)).symbol(),
            merged.into_function(),
        ),
        max_threads,
        max_registers,
    ))
}

/// What a group's prologue values are, in the merged body: the declarations it
/// takes, and every value the shared opening defines, matched by position.
///
/// The groups are emitted from the same plan, so their prologues are the same
/// steps in the same order; that is what the length check above proves, and what
/// lets one body's values stand for another's.
fn prologue_substitutions(
    merged: &mut Builder,
    source: &gpu_ir::Function,
    prologue_end: usize,
) -> Result<Vec<(ValueId, Value)>, String> {
    let mut substitutions = Vec::new();
    for (index, declared) in source.declarations().iter().enumerate() {
        let held = source.declared(index);
        let name = match declared {
            Declaration::Param(param) => param.name.clone(),
            Declaration::Shared(array) => array.name.clone(),
        };
        let existing = merged
            .function()
            .declarations()
            .iter()
            .position(|held| match held {
                Declaration::Param(param) => param.name == name,
                Declaration::Shared(array) => array.name == name,
            });
        let value = match existing {
            Some(position) => {
                let declared = merged.function().declared(position);
                merged.value_of(declared)
            }
            None => match declared {
                Declaration::Param(param) => merged.parameter(param.clone()),
                Declaration::Shared(array) => merged.shared_array(array.clone()),
            },
        };
        substitutions.push((held, value));
    }
    let entry = source.entry();
    let merged_ops = merged.steps();
    for index in 0..prologue_end {
        let op = source.block_steps(entry).collect::<Vec<_>>()[index];
        let mirror = *merged_ops
            .get(index)
            .ok_or("a split group's prologue is longer than the merged body's")?;
        for (slot, &result) in source.op(op).results.iter().enumerate() {
            let held = merged.function().op(mirror).results[slot];
            let value = merged.value_of(held);
            substitutions.push((result, value));
        }
    }
    Ok(substitutions)
}

#[cfg(test)]
mod tests;
