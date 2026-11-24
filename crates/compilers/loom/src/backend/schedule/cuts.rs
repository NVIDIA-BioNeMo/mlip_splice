// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::codegen::{emit_consumer_kernel, emit_kernel, emit_producer_kernel};
use super::super::partition::{
    KernelPart, evaluation_profile_with_provenance, selected_term_values, slice_bindings,
    statement_count,
};
use super::{Candidate, DomainPlan, Launch, MaterializedValue, Settings};
use crate::LiftedProgram;
use domain_index::DomainIndex;
use gpu_ir::{Body, Scalar, ValTy, ValueId};
use std::collections::BTreeMap;

/// What every cap probe of one plan needs: the program, where it iterates, the
/// cap a part must fit, whether a part may be halved at all, and the block size
/// the probe kernel is rendered at.
pub(super) struct CapProbe<'a> {
    pub(super) domain: &'a DomainPlan,
    pub(super) loom: &'a LiftedProgram,
    pub(super) statement_cap: usize,
    pub(super) allow_partial_splits: bool,
    pub(super) block_size: u32,
}

/// Brings `part` inside the statement cap by halving it, recording how many
/// halvings each surviving part cost. `bisections` is the depth reached so far.
pub(super) fn refine_parts(
    probe: &CapProbe<'_>,
    part: KernelPart,
    parts: &mut Vec<(KernelPart, bool, usize)>,
    bisections: usize,
) -> Result<(), String> {
    let kernel = emit_kernel(probe.domain, probe.loom, probe.block_size, 0, 1, &part)?;
    if statement_count(kernel.body()) <= probe.statement_cap {
        parts.push((part, false, bisections));
    } else if probe.allow_partial_splits && part.len() > 1 {
        let (left, right) = part.split();
        for half in [left, right] {
            refine_parts(probe, half, parts, bisections + 1)?;
        }
    } else {
        parts.push((part, true, bisections));
    }
    Ok(())
}

pub(super) fn consumer_statement_count(
    domain: &DomainPlan,
    loom: &LiftedProgram,
    part: &KernelPart,
    materialized: &[MaterializedValue],
    probe_block_size: u32,
) -> Result<usize, String> {
    emit_consumer_kernel(domain, loom, probe_block_size, 0, 1, part, materialized)
        .map(|kernel| statement_count(kernel.body()))
}

pub(super) fn candidate_fits(
    domain: &DomainPlan,
    loom: &LiftedProgram,
    candidate: &Candidate,
    statement_cap: usize,
    probe_block_size: u32,
) -> Result<bool, String> {
    let num_consumers = candidate
        .launches
        .iter()
        .filter(|launch| matches!(launch, Launch::Consumer { .. }))
        .count();
    let mut consumer = 0;
    let mut produced = 0;
    for launch in &candidate.launches {
        let kernel = match launch {
            Launch::Producer { output, inputs } => {
                if *output != produced {
                    return Err("loom materialized outputs are not produced in order".into());
                }
                let kernel = emit_producer_kernel(
                    domain,
                    loom,
                    probe_block_size,
                    *output,
                    inputs,
                    &candidate.materialized,
                )?;
                produced += 1;
                kernel
            }
            Launch::Consumer { part, .. } => {
                let kernel = emit_consumer_kernel(
                    domain,
                    loom,
                    probe_block_size,
                    consumer,
                    num_consumers,
                    part,
                    &candidate.materialized[..produced],
                )?;
                consumer += 1;
                kernel
            }
        };
        if statement_count(kernel.body()) > statement_cap {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn choose_cut(
    loom: &LiftedProgram,
    parts: &[KernelPart],
    domain: &DomainPlan,
    settings: &Settings,
) -> Result<Option<MaterializedValue>, String> {
    let profile = evaluation_profile_with_provenance(loom)?;
    let (outputs, body) = (&profile.outputs, &profile.body);
    let f = body.function();
    // How many of this plan's launches each binding is inside the closure of.
    // Every part slices the one profile arena, so the count is per value; a
    // binding two launches both recompute is what a cut is worth.
    let mut uses = BTreeMap::<ValueId, usize>::new();
    for part in parts {
        let (_, roots) = selected_term_values(f, &body.steps(), outputs, part)?;
        let mut reads = std::collections::BTreeSet::new();
        for root in &roots {
            root.add_reads(f, &mut reads);
        }
        let reads = reads.into_iter().collect::<Vec<_>>();
        for op in slice_bindings(f, &body.steps(), &reads)? {
            *uses.entry(f.op(op).results[0]).or_default() += 1;
        }
    }

    let selected = uses
        .into_iter()
        .filter(|&(_, consumer_count)| consumer_count >= 2)
        .filter_map(|(root, consumer_count)| {
            let scalar = cut_scalar(f, root)?;
            let name = f.name(root)?.to_owned();
            let closure_statements = slice_bindings(f, &body.steps(), &[root]).ok()?.len();
            (closure_statements >= settings.min_cut_closure).then_some((
                closure_statements * (consumer_count - 1),
                name,
                root,
                consumer_count,
                scalar,
            ))
        })
        // Most work saved, and among cuts worth the same, the first name: the
        // choice is then the program's rather than the order this compile
        // happened to number its values in.
        .max_by(|left, right| left.0.cmp(&right.0).then_with(|| right.1.cmp(&left.1)));
    let Some((_, _, root, consumer_count, scalar)) = selected else {
        return Ok(None);
    };
    Ok(Some(materialized_value(
        loom,
        domain,
        body,
        root,
        scalar,
        consumer_count,
    )?))
}

/// The scalar a binding would be materialized as, or `None` where it cannot be
/// one: a pointer has no element type to store, and a predicate is not a value
/// this writes to memory and reads back.
pub(super) fn cut_scalar(f: &gpu_ir::Function, value: ValueId) -> Option<Scalar> {
    match f.ty(value) {
        ValTy::Data(ty) if ty.scalar != Scalar::Pred => Some(ty.scalar),
        _ => None,
    }
}

pub(super) fn materialized_value(
    loom: &LiftedProgram,
    domain: &DomainPlan,
    body: &Body,
    root: ValueId,
    scalar: Scalar,
    consumer_count: usize,
) -> Result<MaterializedValue, String> {
    let f = body.function();
    let name = f
        .name(root)
        .ok_or("loom materialization cuts at a binding the profile does not name")?
        .to_owned();
    let closure_statements = slice_bindings(f, &body.steps(), &[root])?.len();
    let elements = usize::try_from(domain.index.batch_total()?)
        .map_err(|_| "loom materialization batch size exceeds usize")?
        .checked_mul(
            usize::try_from(loom.lane_count).map_err(|_| "loom materialization U exceeds usize")?,
        )
        .ok_or("loom materialization element count overflows usize")?;
    let nbytes = elements
        .checked_mul(scalar.size() as usize)
        .ok_or("loom materialization size overflows usize")?;
    Ok(MaterializedValue {
        name,
        scalar,
        nbytes,
        closure_statements,
        consumer_count,
    })
}
