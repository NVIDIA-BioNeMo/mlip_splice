// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hidden-axis trip of a launch whose iteration stays on one position: each
//! trip evaluates the position's program at one hidden lane, writing lane-wide
//! outputs straight out and folding width-1 outputs into scalar accumulators.

use std::collections::BTreeSet;

use gpu_ir::{OpId, OpKind, Value, ValueId};

use super::accumulate::scalar_accumulation;
use super::consumer::{Accumulators, Body, Consumer, guarded_parts};
use super::{
    At, Emit, Flavor, Folded, GuardedRegion, KernelPart, Written, evaluate_at, guarded_body,
    inactive_direct_writes, output_scalar, output_writes, vector_reduction_scalar,
};

pub(super) fn push(
    consumer: &Consumer<'_>,
    body: &mut Body,
    builder: &mut gpu_ir::Builder,
    u: &Value,
    accumulators: &mut Accumulators,
) -> Result<(), String> {
    let Consumer { part, flavor, .. } = *consumer;
    if flavor.guard.is_some() {
        for (selected, region) in guarded_parts(part, flavor.guard) {
            let steps = evaluate(builder, u, consumer, selected, region, accumulators)?.0;
            body.trip.extend(steps);
        }
        return Ok(());
    }
    let (mut active, scalar_updates) = evaluate(builder, u, consumer, part, None, accumulators)?;
    if flavor.lane_hoist {
        let (hoisted, varying) = hoist_lane_uniform(builder, active, u)?;
        body.hoisted = hoisted;
        active = varying;
    }
    active.extend(scalar_updates);
    body.trip.extend(active);
    Ok(())
}

/// The trip body for one selected sub-part, plus the width-1 accumulations a
/// lane hoist has to keep after the cut.
fn evaluate(
    builder: &mut gpu_ir::Builder,
    u: &Value,
    consumer: &Consumer<'_>,
    selected: &KernelPart,
    region: Option<&GuardedRegion>,
    accumulators: &mut Accumulators,
) -> Result<(Vec<OpId>, Vec<OpId>), String> {
    let Consumer {
        program,
        loom,
        part,
        flavor,
        contract_accumulation,
        ref flat,
        ..
    } = *consumer;
    let Flavor {
        materialized,
        lane_hoist,
        io_lanes,
        ..
    } = flavor;
    let emit = Emit {
        program,
        loom,
        ctx: &consumer.ctx,
    };
    let evaluation = evaluate_at(
        builder,
        emit,
        At {
            flat: flat.clone(),
            u: u.clone(),
            fiber_edge: None,
            io_lanes,
        },
        selected,
        materialized,
        region,
    )?;
    let mut active = evaluation.ops;
    // Lane hoisting cuts the body at its u-invariant stage, so the width-1
    // accumulations come back separately and land after the cut.
    let mut scalar_updates = Vec::new();
    // The folds that land inside the guarded region: what they reached is the
    // region's value, so it leaves through what the region yields.
    let mut folded = Vec::new();
    let mut folded_at = Vec::new();
    for (output, values) in evaluation.outputs.into_iter().enumerate() {
        if loom.outputs()[output].lane_extent != 1 {
            active.extend(output_writes(
                builder,
                emit,
                Written {
                    output,
                    values: &values,
                    elements: &selected.elements[output],
                },
                u,
                io_lanes,
            )?);
            continue;
        }
        let scalar = vector_reduction_scalar(output_scalar(loom, output)?, io_lanes);
        for (element, value) in values.into_iter().enumerate() {
            let (Some(place), Some(value)) = (accumulators[output][element].clone(), value) else {
                continue;
            };
            let before = place.clone();
            let (fragment, reached) = builder.build_fragment(|f| {
                scalar_accumulation(
                    f,
                    place,
                    value,
                    io_lanes,
                    scalar,
                    contract_accumulation,
                    false,
                )
            });
            accumulators[output][element] = Some(reached.clone());
            let accumulation = builder.place_fragment(fragment);
            if lane_hoist {
                scalar_updates.extend(accumulation);
            } else {
                active.extend(accumulation);
                folded.push(Folded {
                    before,
                    after: reached,
                });
                folded_at.push((output, element));
            }
        }
    }
    let mut inactive = Vec::new();
    if evaluation.guard.is_some() && region.is_some_and(|region| region.part.len() == part.len()) {
        inactive.extend(inactive_direct_writes(
            builder, emit, selected, u, io_lanes,
        )?);
    }
    let (steps, produced) = guarded_body(
        builder,
        evaluation.guard_prefix,
        evaluation.guard,
        active,
        inactive,
        folded,
    );
    for ((output, element), value) in folded_at.into_iter().zip(produced) {
        accumulators[output][element] = Some(value);
    }
    Ok((steps, scalar_updates))
}

/// Split a flat trip body at its hidden-lane dependence: steps whose closure
/// never reaches the trip's index are loop-invariant and hoist above it.
fn hoist_lane_uniform(
    builder: &gpu_ir::Builder,
    steps: Vec<OpId>,
    u: &Value,
) -> Result<(Vec<OpId>, Vec<OpId>), String> {
    let f = builder.function();
    let index = u.id_in(f);
    let mut varying = BTreeSet::<ValueId>::new();
    let mut hoisted = Vec::new();
    let mut inner = Vec::new();
    let mut moved_stage = false;
    for op in steps {
        let init = match f.op(op).kind {
            OpKind::Bind => f.op(op).operands[0],
            OpKind::Write { .. } | OpKind::If => {
                inner.push(op);
                continue;
            }
            _ => {
                return Err("loom lane hoisting expects flat bindings and output writes".into());
            }
        };
        let defined = f.op(op).results[0];
        let read = f.reads(init);
        if read.contains(&index) || read.iter().any(|value| varying.contains(value)) {
            varying.insert(defined);
            inner.push(op);
        } else {
            moved_stage |= f
                .name(defined)
                .is_some_and(|name| name.starts_with("loom_v"));
            hoisted.push(op);
        }
    }
    if !moved_stage {
        return Err("loom launch has no lane-uniform stage to hoist".into());
    }
    Ok((hoisted, inner))
}
