// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::Trip;

use gpu_ir::{Carry, Value};

use super::super::name::KernelName;

use super::accumulate::{Warps, push_block_sum, warp_sum};
use super::{
    At, BucketMaskValues, DeviceFunction, DomainIndex, Emit, EmitCtx, KernelPart, Launch,
    LiftedProgram, ParameterValues, Plan, Scalar, Ty, WARP_SIZE, evaluate_at, num_elements,
    operand_alive, output_scalar, push_axis_indices, reduction, zero,
};

pub(in super::super) fn emit_single_block_reduction_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part: &KernelPart,
) -> Result<Launch, String> {
    emit_block_reduction(program, loom, block_size, part)
}

fn emit_block_reduction(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    part: &KernelPart,
) -> Result<Launch, String> {
    if loom.has_output_factors()
        || program.index.fiber_bridge().is_some()
        || block_size < WARP_SIZE
        || !block_size.is_multiple_of(WARP_SIZE)
    {
        return Err("loom single-block reduction is not eligible for this kernel".into());
    }
    reduction::validate(program, loom)?;
    let num_outputs = loom.outputs().len();
    let mut output_scalars = Vec::with_capacity(num_outputs);
    let mut output_elements = Vec::with_capacity(num_outputs);
    for (output, binding) in loom.outputs().iter().enumerate() {
        let operand = loom.num_inputs + output;
        let elements = num_elements(&binding.element_shape)?;
        let selected = &part.elements[output];
        let complete = selected.len() == elements
            && selected.iter().enumerate().all(|(element, selection)| {
                selection.element == element && !selection.continuation
            });
        let scalar = output_scalar(loom, output)?;
        let fold = program.folds.output(output);
        // This kernel writes the output buffer itself, with no projection after
        // it, so the dtype it accumulates in must already be the output's own:
        // a fold that accumulates wider owes a conversion this path never runs.
        if scalar == Scalar::Pred
            || binding.lane_extent != 1
            || program.index.operand_batch_extent(operand)? != 1
            || !fold.combines_atomically()
            || fold.parameter_scalar() != scalar
            || !fold.stores_its_accumulator()
            || !complete
        {
            return Err(
                "loom single-block reduction requires complete scalar atomic outputs".into(),
            );
        }
        output_scalars.push(scalar);
        output_elements.push(elements);
    }

    let entry = KernelName::reduction(&loom.name, block_size).symbol();
    let mut body = gpu_ir::Builder::new();
    // The block-direct kernel writes its outputs directly, so no bucket mask reaches it.
    let declared =
        ParameterValues::declare(&mut body, &program.layout, program.layout.params.len());
    let lane = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let width = body.i32(i64::from(WARP_SIZE));
        let lane = body.rem(tid, width);
        body.bind_value_as("loom_lane", Ty::scalar(Scalar::S32), lane)
    };
    let warp = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let width = body.i32(i64::from(WARP_SIZE));
        let warp = body.div(tid, width);
        body.bind_value_as("loom_warp", Ty::scalar(Scalar::S32), warp)
    };
    let mut accumulators: Vec<Vec<Value>> = Vec::with_capacity(num_outputs);
    for (output, (&scalar, &elements)) in output_scalars.iter().zip(&output_elements).enumerate() {
        accumulators.push(
            (0..elements)
                .map(|element| {
                    let empty = zero(&mut body, scalar);
                    body.bind_value_as(
                        format!("loom_br_acc_{output}_{element}"),
                        Ty::scalar(scalar),
                        empty,
                    )
                })
                .collect::<Vec<_>>(),
        );
    }

    // It writes its outputs directly, so it takes no bucket mask.
    let no_masks = BucketMaskValues::none();
    let at = Position {
        program,
        loom,
        part,
        declared: &declared,
        bucket_masks: &no_masks,
    };
    let carried = carried_over(&body, &accumulators);
    let start = {
        let tid = body.tid_x();
        body.cast(Scalar::S64, tid)
    };
    let total = body.i64(program.index.batch_total()?);
    let stride = body.i64(i64::from(block_size));
    let open = body.open_loop_body(Scalar::S64, carried);
    let flat = open.index();
    body.name_value(&flat, "loom_position");
    let mut inner = rebound(&accumulators, open.given());
    let built = push_position_body(&mut body, &at, flat, &mut inner);
    let yields = flattened(&inner);
    let position = body.close_loop_body(open, yields);
    built?;
    let swept = body.place_counted_loop(Trip::PerThread, start, total, stride, position);
    accumulators = rebound(&accumulators, &swept);

    let num_warps = block_size / WARP_SIZE;
    for (output, &scalar) in output_scalars.iter().enumerate() {
        for (element, place) in accumulators[output][..output_elements[output]]
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>()
        {
            let summed = warp_sum(&mut body, place);
            // The buffer is the handle the kernel took for it, and the store is
            // built where its address is: the two are one value, so a store
            // cannot be written against a buffer this kernel never declared.
            let buffer = declared.get(program.layout.operand_params[loom.num_inputs + output]);
            let index = body.i64(element as i64);
            let cell = body.offset(buffer, index);
            let lane = lane.clone();
            let combined = push_block_sum(
                &mut body,
                summed,
                format!("loom_br_block_{output}_{element}"),
                scalar,
                Warps {
                    lane: lane.clone(),
                    warp: warp.clone(),
                    count: num_warps,
                },
                |f, total| {
                    let zero = f.i32(0);
                    let leader = f.eq(lane, zero);
                    let (store, ()) = f.build_fragment(|g| g.store(cell, total));
                    let store = store.into_ops();
                    f.if_(leader, |arm| arm.place_all(store));
                    Ok(())
                },
            )?;
            accumulators[output][element] = combined;
            body.barrier();
        }
    }

    Ok(Launch::new(
        DeviceFunction::new(entry, body.into_function()),
        block_size,
        None,
    ))
}

/// The accumulators as one list, in output-then-element order.
fn flattened(accumulators: &[Vec<Value>]) -> Vec<Value> {
    accumulators.iter().flatten().cloned().collect()
}

/// `accumulators` with each place replaced by the matching entry of `values`.
fn rebound(accumulators: &[Vec<Value>], values: &[Value]) -> Vec<Vec<Value>> {
    let mut values = values.iter();
    accumulators
        .iter()
        .map(|places| {
            places
                .iter()
                .map(|_| values.next().expect("one value per accumulator").clone())
                .collect()
        })
        .collect()
}

/// Every accumulator carried across the sweep, each updating the place it
/// started from.
fn carried_over(b: &gpu_ir::Builder, accumulators: &[Vec<Value>]) -> Vec<Carry> {
    flattened(accumulators)
        .into_iter()
        .map(|place| Carry {
            ty: b.value_ty(&place).data(),
            init: place.clone(),
            into: Some(place),
        })
        .collect()
}

/// What one position's body is emitted against: the program it reduces, the part
/// it covers, and what the kernel prologue declared and bound.
struct Position<'a> {
    program: &'a Plan,
    loom: &'a LiftedProgram,
    part: &'a KernelPart,
    declared: &'a ParameterValues,
    bucket_masks: &'a BucketMaskValues,
}

/// One position's whole hidden axis, folded into the block's accumulators.
///
/// `flat` is the position this body runs for, whichever construct bound it — the
/// sweep's index or the prologue's binding — so the two placements differ only
/// in what they hand in.
fn push_position_body(
    outer: &mut gpu_ir::Builder,
    at: &Position<'_>,
    flat: Value,
    accumulators: &mut Vec<Vec<Value>>,
) -> Result<(), String> {
    let Position {
        program,
        loom,
        part,
        ..
    } = *at;
    let axis_indices = push_axis_indices(outer, &flat, &program.index.batch_sizes());
    let ctx = EmitCtx {
        index: &program.index,
        layout: &program.layout,
        declared: at.declared,
        bucket_masks: at.bucket_masks,
        axis_indices: &axis_indices,
        view_elements: &[],
        fiber_index: None,
    };
    let carried = carried_over(outer, accumulators);
    let (start, stop, step) = (outer.i32(0), outer.i32(loom.lane_count), outer.i32(1));
    let open = outer.open_loop_body(Scalar::S32, carried);
    let u = open.index();
    outer.name_value(&u, "loom_u");
    let mut inner = rebound(accumulators, open.given());
    let built = (|f: &mut gpu_ir::Builder| -> Result<(), String> {
        let laid = f.steps();
        let evaluation = evaluate_at(
            f,
            Emit {
                program,
                loom,
                ctx: &ctx,
            },
            At {
                flat: flat.clone(),
                u: u.clone(),
                fiber_edge: None,
                io_lanes: 1,
            },
            part,
            &[],
            None,
        )?;
        let mut placed = laid;
        placed.extend(evaluation.ops);
        let before_folds = f.steps();
        for (output, values) in evaluation.outputs.into_iter().enumerate() {
            let operand = loom.num_inputs + output;
            let alive = operand_alive(&ctx, f, operand);
            let places = inner[output].clone();
            // Each fold reads the accumulator and is produced back into it, so
            // what the fold hands back is the accumulator from here on. Under a
            // guard the arm hands it back and the other arm hands back what the
            // accumulator already holds; with nothing to guard it, the folds run
            // where they stand and hand themselves back directly.
            let fold = |f: &mut gpu_ir::Builder| {
                values
                    .into_iter()
                    .zip(&places)
                    .map(|(value, place)| {
                        let value = value.expect("complete block-reduction output has a value");
                        let sum = f.add(place.clone(), value);
                        f.update(place.clone(), sum)
                    })
                    .collect::<Vec<_>>()
            };
            inner[output] = match alive {
                Some(alive) => {
                    let results = places
                        .iter()
                        .map(|place| (f.value_ty(place).data(), Some(place.clone())))
                        .collect::<Vec<_>>();
                    f.open_arm();
                    let folded = fold(f);
                    let taken = f.close_arm(folded);
                    f.open_arm();
                    let untaken = f.close_arm(places);
                    f.if_arms(alive, &results, taken, untaken)
                }
                None => fold(f),
            };
        }
        let folds = f
            .steps()
            .into_iter()
            .filter(|step| !before_folds.contains(step))
            .collect::<Vec<_>>();
        placed.extend(folds);
        f.replace_placed(placed);
        Ok(())
    })(outer);
    let yields = flattened(&inner);
    let trip = outer.close_loop_body(open, yields);
    built?;
    let swept = outer.place_counted_loop(Trip::PerThread, start, stop, step, trip);
    *accumulators = rebound(accumulators, &swept);
    Ok(())
}
