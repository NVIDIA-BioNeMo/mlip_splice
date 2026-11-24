// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hidden-axis trip of a launch whose iteration crosses a fiber bridge: each
//! trip walks the coarse position's finer elements, gathering through the walk
//! and folding the fiber into per-element accumulators.

use gpu_ir::{Carry, OpId, Trip, Value};

use super::accumulate::{accumulator_scalar, interleave_accumulations};
use super::consumer::{
    Accumulators, Body, Consumer, accumulator_places, declare_accumulators, guarded_parts,
    rebind_accumulators, shared_array,
};
use super::{
    At, Emit, EmitCtx, Flavor, Folded, LiftedProgram, Scalar, SharedArr, Ty, WARP_SIZE, Written,
    evaluate_at, fiber_scatter_writes, guarded_body, output_scalar, output_writes,
};

/// The outputs a fiber walk folds across the fiber and writes once the walk is
/// over: wide outputs spanning the whole hidden axis that the walk does not
/// scatter. Deciding what the prologue claims and deciding what the trip writes
/// are the same question, so they ask it here.
fn folded(loom: &LiftedProgram, fiber_scatter: &[bool], output: usize) -> bool {
    let binding = &loom.outputs()[output];
    binding.lane_extent != 1 && binding.lane_extent == loom.lane_count && !fiber_scatter[output]
}

/// What a fiber walk settles in the position prologue, before the trip that
/// reads it: the shared stripe its warps combine through, and the warp index a
/// split walk strides by.
pub(super) struct Prepared {
    /// The stripe claimed for the warps to combine through, when they do. Only a
    /// walk that bound `warp` claims one, so a combine always has the index it
    /// strides and deposits by.
    stripe: Option<Vec<StagedAccumulator>>,
    /// The warp index the prologue bound when the warps divide the walk: the
    /// stride reads it and the shared combine deposits by it, and one condition
    /// decides whether either exists.
    warp: Option<Value>,
    /// Where in the shared stripe this thread's warp deposits.
    partial_base: Option<Value>,
}

/// Claim everything a fiber walk needs from the position prologue.
///
/// Nothing here is the trip's, and the trip touches no prologue of its own: what
/// a launch takes and where its shared memory goes is settled before the
/// hidden-axis loop opens, so the loop is free to be a region with a scope.
pub(super) fn prepare(
    consumer: &Consumer<'_>,
    prologue: &mut gpu_ir::Builder,
) -> Result<Prepared, String> {
    let Consumer {
        loom,
        part,
        flavor,
        fiber_scatter,
        threads_per_position,
        fiber_split,
        ..
    } = *consumer;
    let Flavor {
        io_lanes,
        fiber_lanes,
        ..
    } = flavor;
    // The warp index a walk the warps divide strides by: one condition decides
    // the binding and everything that reads it.
    let warp = (!fiber_lanes && fiber_split).then(|| {
        let tid = prologue.tid_x();
        let tid = prologue.cast(Scalar::S32, tid);
        let per_position = prologue.i32(threads_per_position as i64);
        let within = prologue.rem(tid, per_position);
        let lanes = prologue.i32(WARP_SIZE as i64);
        let index = prologue.div(within, lanes);
        prologue.bind_value_as("loom_fiber_warp", Ty::scalar(Scalar::S32), index)
    });
    // The warps combine through shared memory exactly when the walk divides
    // across them and leaves something to write: an output it folds and this
    // part selects. Anything less writes nothing after the walk, and claims
    // nothing here. The first half is the condition that bound `warp`, read off
    // the binding rather than restated, because the combine deposits by it.
    let combines = warp.is_some()
        && (0..loom.outputs().len())
            .any(|output| folded(loom, fiber_scatter, output) && part.writes_output(output));
    let (stripe, partial_base) = if combines {
        let (staged, base) = claim_warp_stripe(consumer, prologue)?;
        (Some(staged), Some(base))
    } else {
        (None, None)
    };
    let _ = io_lanes;
    Ok(Prepared {
        stripe,
        warp,
        partial_base,
    })
}

/// What the walk itself is emitted against, for the two wrappers that place it:
/// one runs it on every lane, the other only on the lanes a lane split gave a
/// fiber to.
struct Walk<'a> {
    consumer: &'a Consumer<'a>,
    map_idx: usize,
    u: &'a Value,
    accumulator_scalars: &'a [Scalar],
}

/// One trip per finer element of this position's fiber, placed under `into`.
///
/// The body is built by whichever builder places it, because a loop goes where
/// its bounds are read: directly in the trip, or inside the guard a lane split
/// puts around it.
#[allow(clippy::too_many_arguments)]
fn place_walk(
    into: &mut gpu_ir::Builder,
    walk: &Walk<'_>,
    k_init: Value,
    hi: Value,
    k_step: Value,
    wide: &mut Accumulators,
    scalar: &mut Accumulators,
) -> Result<(), String> {
    let Walk {
        consumer,
        map_idx,
        u,
        accumulator_scalars,
    } = *walk;
    let Consumer {
        program,
        loom,
        part,
        flavor,
        fiber_scatter,
        contract_accumulation,
        ref lane,
        ref flat,
        ..
    } = *consumer;
    let Flavor {
        materialized,
        guard,
        io_lanes,
        ..
    } = flavor;
    let ctx = &consumer.ctx;
    let preimage_source_indices = program.layout.ref_params(map_idx).preimage_source_indices;

    let mut carried = accumulator_places(wide);
    carried.extend(accumulator_places(scalar));
    let carried = carried
        .into_iter()
        .map(|place| {
            let ty = into.value_ty(&place).data();
            Carry {
                ty,
                init: place.clone(),
                into: Some(place),
            }
        })
        .collect::<Vec<_>>();
    let open = into.open_loop_body(Scalar::S32, carried);
    let k = open.index();
    into.name_value(&k, "_k");
    let given = open.given().to_vec();
    let (wide_given, scalar_given) = given.split_at(accumulator_places(wide).len());
    let mut inner_wide = wide.clone();
    let mut inner_scalar = scalar.clone();
    rebind_accumulators(&mut inner_wide, wide_given.to_vec());
    rebind_accumulators(&mut inner_scalar, scalar_given.to_vec());

    let built = (|g: &mut gpu_ir::Builder| -> Result<(), String> {
        // The fiber iteration's gathered indices — the edge and each aligned
        // view's element — bind once at the top of the loop body: cloning the
        // gather into every operand would reload the same cell per operand, and
        // the loop's stores keep ptxas from merging the loads.
        let edge_init = match preimage_source_indices {
            Some(source_indices) => {
                let perm = ctx.declared.get(source_indices);
                let cell = g.offset(perm, k.clone());
                let loaded = g.load(cell, 1, false);
                g.cast(Scalar::S64, loaded)
            }
            None => g.cast(Scalar::S64, k.clone()),
        };
        let edge = g.bind_value_as("loom_edge", Ty::scalar(Scalar::S64), edge_init);
        let mut view_elements = Vec::with_capacity(program.layout.view_params.len());
        for (slot, &view) in program.layout.view_params.iter().enumerate() {
            let array = ctx.declared.get(view);
            let cell = g.offset(array, k.clone());
            let loaded = g.load(cell, 1, false);
            let element = g.cast(Scalar::S64, loaded);
            view_elements.push(g.bind_value_as(
                format!("loom_view_elem_{slot}"),
                Ty::scalar(Scalar::S64),
                element,
            ));
        }
        let fiber_ctx = EmitCtx {
            index: &program.index,
            layout: &program.layout,
            declared: ctx.declared,
            bucket_masks: ctx.bucket_masks,
            axis_indices: ctx.axis_indices,
            view_elements: &view_elements,
            fiber_index: Some(k.clone()),
        };
        for (selected, region) in guarded_parts(part, guard) {
            // What the walk has laid down so far. Everything this region builds
            // is placed as it is created, and the steps it hands back are the
            // ones that run — so the frame is restored to this and the region's
            // own list appended, rather than keeping both.
            let laid = g.steps();
            let emit = Emit {
                program,
                loom,
                ctx: &fiber_ctx,
            };
            let evaluation = evaluate_at(
                g,
                emit,
                At {
                    flat: flat.clone(),
                    u: u.clone(),
                    fiber_edge: Some(edge.clone()),
                    io_lanes,
                },
                selected,
                materialized,
                region,
            )?;
            let mut active = evaluation.ops;
            for (output, scatter) in fiber_scatter.iter().enumerate() {
                if !scatter {
                    continue;
                }
                active.extend(fiber_scatter_writes(
                    g,
                    emit,
                    Written {
                        output,
                        values: &evaluation.outputs[output],
                        elements: &selected.elements[output],
                    },
                    u,
                    edge.clone(),
                    lane,
                    io_lanes,
                )?);
            }
            let before = accumulator_snapshot(g, &inner_wide, &inner_scalar);
            let active = interleave_accumulations(
                g,
                active,
                evaluation.outputs,
                &mut inner_scalar,
                &mut inner_wide,
                io_lanes,
                accumulator_scalars,
                contract_accumulation,
            );
            // What the accumulations reached is the guarded region's value, so
            // it leaves through the results the region yields.
            let (folded, at) = accumulator_folds(g, &before, &inner_wide, &inner_scalar);
            let (placed, produced) = guarded_body(
                g,
                evaluation.guard_prefix,
                evaluation.guard,
                active,
                Vec::new(),
                folded,
            );
            rebind_folded(&mut inner_wide, &mut inner_scalar, &at, produced);
            let mut body = laid;
            body.extend(placed);
            g.replace_placed(body);
        }
        Ok(())
    })(into);
    let mut yields = accumulator_places(&inner_wide);
    yields.extend(accumulator_places(&inner_scalar));
    let body = into.close_loop_body(open, yields);
    built?;
    let results = into.place_counted_loop(Trip::PerThread, k_init, hi, k_step, body);
    let (wide_results, scalar_results) = results.split_at(accumulator_places(wide).len());
    rebind_accumulators(wide, wide_results.to_vec());
    rebind_accumulators(scalar, scalar_results.to_vec());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push(
    consumer: &Consumer<'_>,
    map_idx: usize,
    axis: usize,
    body: &mut Body,
    trip: &mut gpu_ir::Builder,
    u: &Value,
    prepared: &Prepared,
    scalar: &mut Accumulators,
) -> Result<(), String> {
    let Consumer {
        program,
        loom,
        part,
        flavor,
        fiber_scatter,
        u_span,
        u: lane_count,
        ..
    } = *consumer;
    let axis_indices = consumer.ctx.axis_indices;
    let Flavor {
        io_lanes,
        warps_per_position,
        fiber_lanes,
        ..
    } = flavor;
    let accumulator_scalars = (0..loom.outputs().len())
        .map(|output| accumulator_scalar(loom, output, io_lanes))
        .collect::<Result<Vec<_>, String>>()?;
    // Everything the walk lays down is the trip's, so it is built aside and
    // handed back in one piece rather than appended where the trip is placed.
    let (fragment, built) = trip.build_fragment(|builder| {
        // The trip opens on the accumulators at their zero.
        let mut wide = declare_accumulators(
            builder,
            loom,
            part,
            "loom_fiber_acc",
            |output| Ok(Ty::vec(output_scalar(loom, output)?, io_lanes)),
            |output| folded(loom, fiber_scatter, output),
        )?;
        let offsets = program
            .layout
            .ref_params(map_idx)
            .preimage_offsets
            .ok_or_else(|| {
                format!("fiber index map {map_idx} has no preimage-offsets parameter")
            })?;
        let offsets = consumer.ctx.declared.get(offsets);
        let coarse = axis_indices[axis].clone();
        let lo = {
            let cell = builder.offset(offsets.clone(), coarse.clone());
            builder.load(cell, 1, false)
        };
        let hi = {
            let one = builder.i64(1);
            let next = builder.add(coarse, one);
            let cell = builder.offset(offsets, next);
            builder.load(cell, 1, false)
        };
        let (k_init, k_step) = if fiber_lanes {
            let fiber_lane = {
                let tid = builder.tid_x();
                let tid = builder.cast(Scalar::S32, tid);
                let lanes = builder.i32(WARP_SIZE as i64);
                let in_warp = builder.rem(tid, lanes);
                let span = builder.i32(i64::from(u_span));
                builder.div(in_warp, span)
            };
            (
                builder.add(lo, fiber_lane),
                builder.i32(i64::from(WARP_SIZE / u_span)),
            )
        } else if let Some(warp) = &prepared.warp {
            (
                builder.add(lo, warp.clone()),
                builder.i32(i64::from(warps_per_position)),
            )
        } else {
            (lo, builder.i32(1))
        };
        let walk = Walk {
            consumer,
            map_idx,
            u,
            accumulator_scalars: &accumulator_scalars,
        };
        if fiber_lanes {
            let limit = builder.i32(i64::from(lane_count));
            let within = builder.lt(consumer.lane.clone(), limit);
            // A lane split gives the walk only to the lanes that hold a fiber,
            // so the accumulators it folds are the ones that arm produces.
            let places = {
                let mut places = accumulator_places(&wide);
                places.extend(accumulator_places(scalar));
                places
            };
            let results = places
                .iter()
                .map(|place| (builder.value_ty(place).data(), Some(place.clone())))
                .collect::<Vec<_>>();
            builder.open_arm();
            let mut taken_wide = wide.clone();
            let mut taken_scalar = scalar.clone();
            let placed = place_walk(
                builder,
                &walk,
                k_init,
                hi,
                k_step,
                &mut taken_wide,
                &mut taken_scalar,
            );
            let mut yields = accumulator_places(&taken_wide);
            yields.extend(accumulator_places(&taken_scalar));
            let taken = builder.close_arm(yields);
            placed?;
            builder.open_arm();
            let untaken = builder.close_arm(places);
            let produced = builder.if_arms(within, &results, taken, untaken);
            let (wide_produced, scalar_produced) =
                produced.split_at(accumulator_places(&wide).len());
            rebind_accumulators(&mut wide, wide_produced.to_vec());
            rebind_accumulators(scalar, scalar_produced.to_vec());
            let mut offset = WARP_SIZE / 2;
            while offset >= u_span {
                for place in wide.iter_mut().flatten() {
                    if let Some(held) = place.clone() {
                        let delta = builder.i32(i64::from(offset));
                        let other = builder.shfl_down(held.clone(), delta);
                        let sum = builder.add(held.clone(), other);
                        *place = Some(builder.update(held, sum));
                    }
                }
                offset /= 2;
            }
        } else {
            place_walk(builder, &walk, k_init, hi, k_step, &mut wide, scalar)?;
        }
        let (writes, written) = builder.build_fragment(|f| {
            let mut out = Vec::new();
            for (output, places) in wide.iter().enumerate() {
                if !folded(loom, fiber_scatter, output) {
                    continue;
                }
                let values = places.clone();
                let emit = Emit {
                    program,
                    loom,
                    ctx: &consumer.ctx,
                };
                let written = Written {
                    output,
                    values: &values,
                    elements: &part.elements[output],
                };
                out.extend(output_writes(f, emit, written, u, io_lanes)?);
            }
            Ok::<Vec<OpId>, String>(out)
        });
        builder.append_fragment(writes);
        let write_body_ops = written?;
        // Each warp holds only its stripe of the fiber sum when the warps
        // divided the walk, so what it wrote goes through the combine instead of
        // out. The prologue claimed the stripe and the warp index together
        // exactly then, so they are what says so — and a combine of nothing is
        // nothing, which is the one thing they cannot tell.
        let combine = (!write_body_ops.is_empty())
            .then(|| {
                prepared
                    .stripe
                    .as_deref()
                    .zip(prepared.warp.as_ref())
                    .zip(prepared.partial_base.as_ref())
            })
            .flatten();
        if fiber_lanes && !write_body_ops.is_empty() {
            let first_fiber_lane = {
                let tid = builder.tid_x();
                let tid = builder.cast(Scalar::S32, tid);
                let lanes = builder.i32(WARP_SIZE as i64);
                let in_warp = builder.rem(tid, lanes);
                let span = builder.i32(i64::from(u_span));
                builder.lt(in_warp, span)
            };
            let within = {
                let limit = builder.i32(i64::from(lane_count));
                builder.lt(consumer.lane.clone(), limit)
            };
            let pred = builder.and(first_fiber_lane, within);
            builder.if_(pred, |taken| taken.place_all(write_body_ops));
        } else if let Some(((staged, warp), partial_base)) = combine {
            push_warp_stripe_combine(
                consumer,
                builder,
                staged,
                warp,
                partial_base,
                &mut wide,
                write_body_ops,
            )?;
        } else {
            builder.place_all(write_body_ops);
        }
        Ok::<(), String>(())
    });
    built?;
    body.trip.extend(trip.place_fragment(fragment));
    Ok(())
}

/// Where in the shared stripe a thread's warp deposits, bound once in the
/// position prologue.
const FIBER_PARTIAL_BASE: &str = "loom_fiber_partial_base";

/// One accumulator a split walk stages through shared memory: the output and
/// element it belongs to, the array the warps deposit it in, and what that array
/// holds.
struct StagedAccumulator {
    output: usize,
    element: usize,
    stripe: String,
    elem: Scalar,
}

/// Claim the shared memory a split walk's warps combine through: one stripe per
/// live accumulator, and the slot each thread deposits at. Both belong to the
/// position prologue, so both are bound before the trip that reads them.
fn claim_warp_stripe(
    consumer: &Consumer<'_>,
    prologue: &mut gpu_ir::Builder,
) -> Result<(Vec<StagedAccumulator>, Value), String> {
    let Consumer {
        loom,
        part,
        flavor,
        fiber_scatter,
        threads_per_position,
        positions_per_block,
        ..
    } = *consumer;
    let Flavor {
        io_lanes,
        warps_per_position,
        ..
    } = flavor;
    let mut staged = Vec::new();
    for (output, binding) in loom.outputs().iter().enumerate() {
        if !folded(loom, fiber_scatter, output) {
            continue;
        }
        let elem = output_scalar(loom, output)?;
        let count = super::num_elements(&binding.element_shape)?;
        for element in 0..count {
            if !part.contains(output, element) {
                continue;
            }
            staged.push(StagedAccumulator {
                output,
                element,
                stripe: fiber_partial_name(output, element),
                elem,
            });
        }
    }
    let partial_lanes = i64::from(positions_per_block)
        * i64::from(warps_per_position - 1)
        * i64::from(WARP_SIZE)
        * i64::from(io_lanes);
    let occupied = prologue
        .shared_arrays()
        .iter()
        .map(|arr| arr.elem.size() * arr.len)
        .sum::<i64>();
    let needed = staged
        .iter()
        .map(|partial| partial_lanes * partial.elem.size())
        .sum::<i64>();
    const SHARED_BUDGET: i64 = 48 * 1024;
    if occupied + needed > SHARED_BUDGET {
        return Err(format!(
            "loom {warps_per_position}-warp fiber combine needs {} B of shared memory, over the {SHARED_BUDGET} B budget",
            occupied + needed
        ));
    }
    let base = {
        let tid = prologue.tid_x();
        let tid = prologue.cast(Scalar::S32, tid);
        let per_position = prologue.i32(threads_per_position as i64);
        let position = prologue.div(tid, per_position);
        let stride = prologue.i32(i64::from(warps_per_position - 1) * i64::from(WARP_SIZE));
        let block = prologue.mul(position, stride);
        let tid = prologue.tid_x();
        let tid = prologue.cast(Scalar::S32, tid);
        let lanes = prologue.i32(WARP_SIZE as i64);
        let in_warp = prologue.rem(tid, lanes);
        let slot = prologue.add(block, in_warp);
        let width = prologue.i32(io_lanes as i64);
        let base = prologue.mul(slot, width);
        prologue.bind_value_as(FIBER_PARTIAL_BASE, Ty::scalar(Scalar::S32), base)
    };
    for partial in &staged {
        prologue.shared_array(SharedArr {
            name: partial.stripe.clone(),
            elem: partial.elem,
            len: partial_lanes,
        });
    }
    Ok((staged, base))
}

/// A split walk leaves each warp with only its stripe of the fiber sum: warps
/// 1..W deposit their partial vectors in shared memory, and warp 0 folds them
/// into its accumulators and alone runs the writes — one writer per (element,
/// u), same bytes as one warp.
fn push_warp_stripe_combine(
    consumer: &Consumer<'_>,
    trip: &mut gpu_ir::Builder,
    staged: &[StagedAccumulator],
    warp: &Value,
    partial_base: &Value,
    wide: &mut Accumulators,
    write_body: Vec<OpId>,
) -> Result<(), String> {
    let Consumer { loom, flavor, .. } = *consumer;
    let Flavor {
        io_lanes,
        warps_per_position,
        ..
    } = flavor;
    let warp_stride = i64::from(WARP_SIZE) * i64::from(io_lanes);
    let (deposits, deposited) = trip.build_fragment(|f| {
        for partial in staged {
            let Some(place) = wide[partial.output][partial.element].clone() else {
                continue;
            };
            let stripe = shared_array(f, &partial.stripe)
                .ok_or("loom fiber combine has no shared stripe")?;
            let one = f.i32(1);
            let index = f.sub(warp.clone(), one);
            let stride = f.i32(warp_stride);
            let offset = f.mul(index, stride);
            let slot = f.add(partial_base.clone(), offset);
            let cell = f.offset(stripe, slot);
            f.store(cell, place);
        }
        Ok::<(), String>(())
    });
    deposited?;
    {
        let one = trip.i32(1);
        let depositing = trip.ge(warp.clone(), one);
        let deposits = deposits.into_ops();
        trip.if_(depositing, |guarded| guarded.place_all(deposits));
    }
    trip.barrier();
    // Warp 0 folds the stripe into its own accumulators and writes; nothing
    // after this reads them, so the fold and the writes are one guarded region
    // producing nothing.
    let (fold, folded) = trip.build_fragment(|f| {
        for partial in staged {
            let Some(place) = wide[partial.output][partial.element].clone() else {
                continue;
            };
            let stripe = shared_array(f, &partial.stripe)
                .ok_or("loom fiber combine has no shared stripe")?;
            let mut held = place;
            for warp in 1..warps_per_position {
                let offset = f.i32(i64::from(warp - 1) * warp_stride);
                let slot = f.add(partial_base.clone(), offset);
                let cell = f.offset(stripe.clone(), slot);
                let partial = f.load(cell, io_lanes, false);
                let sum = f.add(held.clone(), partial);
                held = f.update(held, sum);
            }
            wide[partial.output][partial.element] = Some(held);
        }
        Ok::<(), String>(())
    });
    folded?;
    let mut arm = fold.into_ops();
    arm.extend(write_body);
    {
        let zero = trip.i32(0);
        let first = trip.eq(warp.clone(), zero);
        trip.if_(first, |guarded| guarded.place_all(arm));
    }
    // Later u passes reuse the buffers: the fold must land first.
    if loom.lane_count > i64::from(WARP_SIZE * u32::from(io_lanes)) {
        trip.barrier();
    }
    Ok(())
}

fn fiber_partial_name(output: usize, element: usize) -> String {
    format!("loom_fiber_partial_{output}_{element}")
}

/// Every accumulator place the walk holds, wide then scalar, by value id.
fn accumulator_snapshot(
    b: &gpu_ir::Builder,
    wide: &Accumulators,
    scalar: &Accumulators,
) -> Vec<gpu_ir::ValueId> {
    let f = b.function();
    accumulator_places(wide)
        .into_iter()
        .chain(accumulator_places(scalar))
        .map(|place| place.id_in(f))
        .collect()
}

/// The accumulators an evaluation folded into, and where they sit in the wide
/// and scalar lists. One the evaluation left alone needs no result: the region
/// would hand back what it was already holding.
fn accumulator_folds(
    b: &gpu_ir::Builder,
    before: &[gpu_ir::ValueId],
    wide: &Accumulators,
    scalar: &Accumulators,
) -> (Vec<Folded>, Vec<usize>) {
    let f = b.function();
    let after = accumulator_places(wide)
        .into_iter()
        .chain(accumulator_places(scalar))
        .collect::<Vec<_>>();
    let mut folded = Vec::new();
    let mut at = Vec::new();
    for (slot, value) in after.iter().enumerate() {
        if value.id_in(f) == before[slot] {
            continue;
        }
        folded.push(Folded {
            before: b.value_of(before[slot]),
            after: value.clone(),
        });
        at.push(slot);
    }
    (folded, at)
}

/// Puts what the guarded region produced back where the accumulator it folded
/// stood.
fn rebind_folded(
    wide: &mut Accumulators,
    scalar: &mut Accumulators,
    at: &[usize],
    produced: Vec<Value>,
) {
    let split = accumulator_places(wide).len();
    let mut held = accumulator_places(wide);
    held.extend(accumulator_places(scalar));
    for (&slot, value) in at.iter().zip(produced) {
        held[slot] = value;
    }
    let (wide_held, scalar_held) = held.split_at(split);
    rebind_accumulators(wide, wide_held.to_vec());
    rebind_accumulators(scalar, scalar_held.to_vec());
}
