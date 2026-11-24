// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Carry, OpId, Value};

use super::super::name::KernelName;
use super::accumulate::{accumulator_scalar, vector_reduction_scalar, warp_sum};
use super::flavor::{self, Flavor};
use super::{
    BucketMaskValues, DeviceFunction, DomainIndex, Emit, EmitCtx, GuardedPlan, GuardedRegion,
    Index, KernelPart, Launch, LiftedProgram, ParameterValues, Plan, Scalar, SharedArr, Ty,
    WARP_SIZE, Written, materialized_params, num_elements, output_scalar, output_writes,
    push_axis_indices, reduction, zero,
};
use super::{fiber_walk, lane_sweep};
use gpu_ir::Counted;
use gpu_ir::Trip;

/// The places a launch folds its width-1 outputs into: one per selected element
/// of each accumulated output, and `None` where the launch accumulates nothing.
///
/// An accumulator is a place, not a name: each fold defines a new value produced
/// into the storage the last one held, so what the list holds is always the
/// value the fold has reached.
pub(super) type Accumulators = Vec<Vec<Option<Value>>>;

/// Everything a hidden-axis trip is emitted against: the program it walks, the
/// arrangement it walks it in, and the values the position prologue bound.
pub(super) struct Consumer<'a> {
    pub program: &'a Plan,
    pub loom: &'a LiftedProgram,
    pub part: &'a KernelPart,
    pub flavor: Flavor<'a>,
    pub ctx: EmitCtx<'a, Index>,
    pub fiber_scatter: &'a [bool],
    pub contract_accumulation: bool,
    /// Threads jointly covering one position.
    pub threads_per_position: u32,
    pub positions_per_block: u32,
    /// The power-of-two hidden extent a lane-split fiber walk packs into a warp;
    /// 0 when the walk does not split lanes.
    pub u_span: u32,
    pub fiber_split: bool,
    pub u: i32,
    pub flat: Value,
    /// The hidden lane this thread opens on.
    pub lane: Value,
}

/// The hidden-axis trip under construction: the steps a lane hoist lifted out of
/// it, and its body.
pub(super) struct Body {
    pub hoisted: Vec<OpId>,
    pub trip: Vec<OpId>,
}

/// One accumulator per selected element of each output, bound at the
/// accumulation's zero. An output the launch does not accumulate in this form,
/// and an element the part does not select, both get `None`.
pub(super) fn declare_accumulators(
    b: &mut gpu_ir::Builder,
    loom: &LiftedProgram,
    part: &KernelPart,
    prefix: &str,
    ty: impl Fn(usize) -> Result<Ty, String>,
    accumulated: impl Fn(usize) -> bool,
) -> Result<Accumulators, String> {
    let mut accumulators = Vec::with_capacity(loom.outputs().len());
    for (output, binding) in loom.outputs().iter().enumerate() {
        let count = num_elements(&binding.element_shape)?;
        if !accumulated(output) {
            accumulators.push(vec![None; count]);
            continue;
        }
        let ty = ty(output)?;
        let mut places = Vec::with_capacity(count);
        for element in 0..count {
            if !part.contains(output, element) {
                places.push(None);
                continue;
            }
            let zero = zero(b, ty.scalar);
            let zero = if ty.lanes == 1 {
                zero
            } else {
                b.splat(ty.lanes, zero)
            };
            places.push(Some(b.bind_value_as(
                format!("{prefix}_{output}_{element}"),
                ty,
                zero,
            )));
        }
        accumulators.push(places);
    }
    Ok(accumulators)
}

/// The places an accumulator set holds, in output-then-element order.
pub(super) fn accumulator_places(accumulators: &Accumulators) -> Vec<Value> {
    accumulators.iter().flatten().flatten().cloned().collect()
}

/// `accumulators` with each place replaced by the next of `values`, which must
/// come in the order [`accumulator_places`] gives them.
pub(super) fn rebind_accumulators(accumulators: &mut Accumulators, values: Vec<Value>) {
    let mut values = values.into_iter();
    for place in accumulators.iter_mut().flatten() {
        if place.is_some() {
            *place = Some(values.next().expect("one value per accumulator"));
        }
    }
    debug_assert!(values.next().is_none());
}

/// The parts a launch evaluates: the guard's remainder, then each of its skip
/// regions; or the whole part when the launch carries no guard.
pub(super) fn guarded_parts<'a>(
    part: &'a KernelPart,
    guard: Option<&'a GuardedPlan>,
) -> Vec<(&'a KernelPart, Option<&'a GuardedRegion>)> {
    let Some(guard) = guard else {
        return vec![(part, None)];
    };
    std::iter::once((&guard.remainder, None))
        .filter(|(remainder, _)| remainder.len() > 0)
        .chain(
            guard
                .regions
                .iter()
                .map(|region| (&region.part, Some(region))),
        )
        .collect()
}

/// A consumer kernel, and how much of its body is the position prologue every
/// group of an element split shares.
///
/// The boundary is reported by the emitter that laid the prologue down, so a
/// merge splits where the steps actually end rather than where a formula says
/// they should.
pub(in crate::backend) struct ConsumerKernel {
    pub kernel: Launch,
    pub prologue_end: usize,
}

pub(in crate::backend) fn emit_consumer_kernel_flavor(
    program: &Plan,
    loom: &LiftedProgram,
    part: &KernelPart,
    flavor: Flavor,
) -> Result<ConsumerKernel, String> {
    let Flavor {
        block_size,
        io_lanes,
        warps_per_position,
        warp_u,
        fiber_lanes,
        materialized,
        ..
    } = flavor;
    let func = loom.module.entry_func();
    let fiber_scatter = (0..func.results.len())
        .map(|output| program.index.fiber_scatter(loom.num_inputs + output))
        .collect::<Vec<_>>();
    flavor::check(flavor, program, loom, part, &fiber_scatter)?;
    let threads_per_position = WARP_SIZE * u32::from(warps_per_position);
    let fiber_split = flavor::fiber_split(program, warps_per_position, warp_u);
    let u = i32::try_from(loom.lane_count)
        .map_err(|_| format!("loom u={} exceeds i32", loom.lane_count))?;
    reduction::validate(program, loom)?;
    flavor::check_fiber_direct_writes(flavor, loom, part, &fiber_scatter)?;
    let u_span = if fiber_lanes {
        u32::try_from(loom.lane_count)
            .map_err(|_| format!("loom u={} exceeds u32", loom.lane_count))?
            .next_power_of_two()
    } else {
        0
    };
    let fiber_lane_count = fiber_lanes
        .then(|| {
            u8::try_from(WARP_SIZE / u_span)
                .map_err(|_| "loom fiber lane count exceeds u8".to_owned())
        })
        .transpose()?;
    let entry =
        KernelName::consumer(&loom.name, block_size, flavor.tuning(fiber_lane_count)).symbol();
    let positions_per_block = block_size / threads_per_position;
    // Threads jointly covering the hidden axis: a fiber split keeps the full
    // axis per warp (the warps divide the fiber walk instead).
    let u_threads = if fiber_lanes {
        u_span
    } else if fiber_split {
        WARP_SIZE
    } else {
        threads_per_position
    };
    let mut prologue = gpu_ir::Builder::new();
    // What the kernel takes is declared before it binds anything, in the order a
    // launch binds it.
    let declared =
        ParameterValues::declare(&mut prologue, &program.layout, program.layout.params.len());
    for param in materialized_params(materialized) {
        prologue.parameter(param);
    }
    let bucket_masks = BucketMaskValues::declare(&mut prologue, &program.layout);
    let lane = {
        let tid = prologue.tid_x();
        let tid = prologue.cast(Scalar::S32, tid);
        let threads = prologue.i32(u_threads as i64);
        let within = prologue.rem(tid, threads);
        let width = prologue.i32(io_lanes as i64);
        prologue.mul(within, width)
    };
    let lane = prologue.bind_value_as("loom_lane", Ty::scalar(Scalar::S32), lane);
    let flat = {
        let block = prologue.ctaid_x();
        let block = prologue.cast(Scalar::S64, block);
        if positions_per_block == 1 {
            block
        } else {
            let per_block = prologue.i64(positions_per_block as i64);
            let blocks = prologue.mul(block, per_block);
            let tid = prologue.tid_x();
            let tid = prologue.cast(Scalar::S32, tid);
            let per_position = prologue.i32(threads_per_position as i64);
            let within = prologue.div(tid, per_position);
            let within = prologue.cast(Scalar::S64, within);
            prologue.add(blocks, within)
        }
    };
    let flat = prologue.bind_value_as("loom_position", Ty::scalar(Scalar::S64), flat);
    {
        let total = prologue.i64(program.index.batch_total()?);
        let past_end = prologue.ge(flat.clone(), total);
        prologue.if_(past_end, |guarded| guarded.ret());
    }

    let axis_indices = push_axis_indices(&mut prologue, &flat, &program.index.batch_sizes());
    // Everything laid down so far is the position prologue: the lane, the flat
    // position, the bounds guard and the axis indices. What follows is this
    // group's own.
    let prologue_end = prologue.steps().len();
    let ctx = EmitCtx {
        index: &program.index,
        layout: &program.layout,
        declared: &declared,
        bucket_masks: &bucket_masks,
        axis_indices: &axis_indices,
        view_elements: &[],
        fiber_index: None,
    };

    let mut accumulators = declare_accumulators(
        &mut prologue,
        loom,
        part,
        "loom_out_acc",
        |output| Ok(Ty::scalar(accumulator_scalar(loom, output, io_lanes)?)),
        |output| loom.outputs()[output].lane_extent == 1 && !fiber_scatter[output],
    )?;
    // Multi-warp positions combine their per-warp scalar sums through shared
    // memory, one slot per warp per live accumulator.
    if warps_per_position > 1 {
        for (output, places) in accumulators.iter().enumerate() {
            for element in places
                .iter()
                .enumerate()
                .filter_map(|(element, place)| place.as_ref().map(|_| element))
            {
                prologue.shared_array(SharedArr {
                    name: scalar_partial_name(output, element),
                    elem: accumulator_scalar(loom, output, io_lanes)?,
                    len: i64::from(block_size / WARP_SIZE),
                });
            }
        }
    }

    let consumer = Consumer {
        program,
        loom,
        part,
        flavor,
        ctx,
        fiber_scatter: &fiber_scatter,
        contract_accumulation: true,
        threads_per_position,
        positions_per_block,
        u_span,
        fiber_split,
        u,
        flat: flat.clone(),
        lane: lane.clone(),
    };
    // A fiber walk claims what it needs from the position prologue before the
    // trip opens, because the trip is a region and the prologue is not. The
    // bridge and what it claimed are one value, so the trip has both or neither.
    let walk = match program.index.fiber_bridge() {
        Some((map_idx, axis)) => Some((
            map_idx,
            axis,
            fiber_walk::prepare(&consumer, &mut prologue)?,
        )),
        None => None,
    };

    // The trip carries the width-1 accumulators: the fold happens in the body
    // and the value it reaches is what the next iteration is given, which is the
    // same register throughout.
    let carried = accumulator_places(&accumulators)
        .into_iter()
        .map(|place| {
            let ty = prologue.value_ty(&place).data();
            Carry {
                ty,
                init: place.clone(),
                into: Some(place),
            }
        })
        .collect::<Vec<_>>();
    let open = prologue.open_loop_body(Scalar::S32, carried);
    let trip_u = open.index();
    prologue.name_value(&trip_u, "loom_u");
    let mut inner = accumulators.clone();
    rebind_accumulators(&mut inner, open.given().to_vec());
    let mut body = Body {
        hoisted: Vec::new(),
        trip: Vec::new(),
    };
    let built = match &walk {
        Some((map_idx, axis, prepared)) => fiber_walk::push(
            &consumer,
            *map_idx,
            *axis,
            &mut body,
            &mut prologue,
            &trip_u,
            prepared,
            &mut inner,
        ),
        None => lane_sweep::push(&consumer, &mut body, &mut prologue, &trip_u, &mut inner),
    };
    built?;
    let Body { hoisted, trip } = body;
    prologue.replace_placed(trip);
    let yields = accumulator_places(&inner);
    let trip = prologue.close_loop_body(open, yields);
    prologue.place_all(hoisted);
    let span = if fiber_lanes {
        i64::from(u_span)
    } else {
        i64::from(u)
    };
    let stride = u_threads as i64 * io_lanes as i64;
    // Every thread starts within one stride of zero — `lane` is `(tid mod
    // u_threads) · io_lanes` — so the trip counts agree exactly when the stride
    // divides the span. The block combines through shared memory inside this
    // body, and a barrier the threads reach a different number of times is one
    // they never all reach.
    let counts_agree = stride != 0 && span % stride == 0;
    let stop = prologue.i32(span);
    let step = prologue.i32(stride);
    let folded = prologue.place_counted_loop(
        if counts_agree {
            Trip::BlockUniform
        } else {
            Trip::PerThread
        },
        lane.clone(),
        stop,
        step,
        trip,
    );
    rebind_accumulators(&mut accumulators, folded);

    let mut body = prologue;
    for place in accumulators.iter_mut().flatten() {
        if let Some(held) = place.clone() {
            *place = Some(warp_sum(&mut body, held));
        }
    }
    // Under a fiber split every warp's lane 0 sees `lane == 0`; the combine and
    // the final stores belong to the position's first thread alone.
    let position_lane_zero = |b: &mut gpu_ir::Builder| {
        if fiber_split || fiber_lanes {
            let tid = b.tid_x();
            let tid = b.cast(Scalar::S32, tid);
            let per_position = b.i32(threads_per_position as i64);
            let within = b.rem(tid, per_position);
            let zero = b.i32(0);
            b.eq(within, zero)
        } else {
            let zero = b.i32(0);
            b.eq(lane.clone(), zero)
        }
    };
    if !body.shared_arrays().is_empty() {
        let pred = position_lane_zero(&mut body);
        push_block_scalar_combine(
            &mut body,
            loom,
            &mut accumulators,
            io_lanes,
            threads_per_position,
            warps_per_position,
            pred,
        )?;
    }
    let mut scalar_writes = Vec::new();
    for (output, binding) in loom.outputs().iter().enumerate() {
        if binding.lane_extent != 1 || fiber_scatter[output] {
            continue;
        }
        let result_scalar = output_scalar(loom, output)?;
        let accumulator_scalar = vector_reduction_scalar(result_scalar, io_lanes);
        let values = accumulators[output]
            .clone()
            .into_iter()
            .map(|place| {
                place.map(|place| {
                    if accumulator_scalar == result_scalar {
                        place
                    } else {
                        body.cast(result_scalar, place)
                    }
                })
            })
            .collect::<Vec<_>>();
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
        let at_lane_zero = body.i32(0);
        scalar_writes.extend(output_writes(&mut body, emit, written, &at_lane_zero, 1)?);
    }
    if !scalar_writes.is_empty() {
        let pred = position_lane_zero(&mut body);
        body.if_(pred, |guarded| guarded.place_all(scalar_writes));
    }

    Ok(ConsumerKernel {
        prologue_end,
        kernel: Launch::new(
            DeviceFunction::new(entry, body.into_function()),
            block_size,
            None,
        ),
    })
}

/// Multi-warp positions finish their width-1 sums across the block: each warp's
/// lane 0 deposits its accumulator, then the position's first thread folds the
/// warps' slots back into the accumulators.
#[allow(clippy::too_many_arguments)]
fn push_block_scalar_combine(
    body: &mut gpu_ir::Builder,
    loom: &LiftedProgram,
    accumulators: &mut Accumulators,
    io_lanes: u8,
    threads_per_position: u32,
    warps_per_position: u8,
    position_lane_zero: Value,
) -> Result<(), String> {
    let warp = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let lanes = body.i32(WARP_SIZE as i64);
        let index = body.div(tid, lanes);
        body.bind_value_as("loom_warp", Ty::scalar(Scalar::S32), index)
    };
    let lane_in_warp = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let lanes = body.i32(WARP_SIZE as i64);
        let index = body.rem(tid, lanes);
        body.bind_value_as("loom_lane_in_warp", Ty::scalar(Scalar::S32), index)
    };
    let (deposits, ()) = body.build_fragment(|f| {
        for (output, places) in accumulators.iter().enumerate() {
            for (element, place) in places.iter().enumerate() {
                let Some(place) = place else { continue };
                let Some(stripe) = shared_array(f, &scalar_partial_name(output, element)) else {
                    continue;
                };
                let slot = f.offset(stripe, warp.clone());
                f.store(slot, place.clone());
            }
        }
    });
    {
        let zero = body.i32(0);
        let first = body.eq(lane_in_warp, zero);
        let deposits = deposits.into_ops();
        body.if_(first, |guarded| guarded.place_all(deposits));
    }
    body.barrier();
    // The combine produces into the accumulators: the arm folds the warps'
    // slots, and the path that does not run yields what they already hold.
    let places = accumulator_places(accumulators);
    let results = places
        .iter()
        .map(|place| (body.value_ty(place).data(), Some(place.clone())))
        .collect::<Vec<_>>();
    let scalars = accumulators
        .iter()
        .enumerate()
        .flat_map(|(output, places)| {
            places
                .iter()
                .enumerate()
                .filter(|(_, place)| place.is_some())
                .map(move |(element, _)| (output, element))
        })
        .collect::<Vec<_>>();
    body.open_arm();
    // Which of the block's warp slots this position owns. Only the arm walks
    // them, so it is bound where the walk is.
    let warp_base = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let per_position = body.i32(threads_per_position as i64);
        let position = body.div(tid, per_position);
        let stride = body.i32(i64::from(warps_per_position));
        let base = body.mul(position, stride);
        body.bind_value_as("loom_warp_base", Ty::scalar(Scalar::S32), base)
    };
    let mut folded = Vec::with_capacity(places.len());
    for (place, &(output, element)) in places.iter().zip(&scalars) {
        let scalar = vector_reduction_scalar(output_scalar(loom, output)?, io_lanes);
        let zero = zero(body, scalar);
        let opened = body.update(place.clone(), zero);
        let stripe = shared_array(body, &scalar_partial_name(output, element))
            .ok_or("loom block combine has no shared stripe for an accumulator")?;
        let start = body.i32(0);
        let stop = body.i32(i64::from(warps_per_position));
        let step = body.i32(1);
        let carried = vec![Carry {
            ty: Ty::scalar(scalar),
            init: opened.clone(),
            into: Some(opened),
        }];
        let summed = body.counted_loop_producing(
            Counted {
                index: Scalar::S32,
                start,
                stop,
                step,
                trip: Trip::PerThread,
            },
            carried,
            |f, index, given| {
                let at = f.add(warp_base.clone(), index);
                let cell = f.offset(stripe.clone(), at);
                let partial = f.load(cell, 1, false);
                let sum = f.add(given[0].clone(), partial);
                vec![f.update(given[0].clone(), sum)]
            },
        );
        folded.push(summed.into_iter().next().expect("one carried value"));
    }
    let taken = body.close_arm(folded);
    body.open_arm();
    let untaken = body.close_arm(places.clone());
    let combined = body.if_arms(position_lane_zero, &results, taken, untaken);
    rebind_accumulators(accumulators, combined);
    Ok(())
}

/// The value standing for a shared array this kernel declared, by name.
pub(super) fn shared_array(b: &gpu_ir::Builder, name: &str) -> Option<Value> {
    let position = b.function().declarations().iter().position(
        |declared| matches!(declared, gpu_ir::Declaration::Shared(array) if array.name == name),
    )?;
    Some(b.value_of(b.function().declared(position)))
}

fn scalar_partial_name(output: usize, element: usize) -> String {
    format!("loom_out_partial_{output}_{element}")
}
