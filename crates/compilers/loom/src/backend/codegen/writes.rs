// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::accumulate::warp_sum;
use gpu_ir::{OpId, Value};

use super::{
    DType, EmitCtx, Evict, Index, InputElement, LiftedProgram, OutputFactor, Plan, PtrTy, Scalar,
    Space, Ty, WARP_SIZE, convert_output_factor, math_scalar, num_elements, operand_alive,
    operand_flat_index, output_factor_elementwise, output_scalar, reduction, scalar_constant,
    storage_scalar, vector_reduction_scalar, zero,
};

/// What every emitted write needs beyond the values: the plan it writes into,
/// the program whose operands it names, and the DomainIndex context the
/// position body established.
#[derive(Clone, Copy)]
pub(super) struct Emit<'a> {
    pub program: &'a Plan,
    pub loom: &'a LiftedProgram,
    pub ctx: &'a EmitCtx<'a, Index>,
}

/// Which elements of which output a write covers, and what they hold. An
/// element the part does not select has no value.
pub(super) struct Written<'a> {
    pub output: usize,
    pub values: &'a [Option<Value>],
    pub elements: &'a [super::super::partition::ElementPart],
}

/// The accumulation buffer's base for `output`, offset onto the bucket this
/// block folds into when the output's fold picks its bucket count at launch.
fn accumulator_base(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    output: usize,
    cell_offset: gpu_ir::Value,
) -> Result<gpu_ir::Value, String> {
    let Emit { program, loom, ctx } = emit;
    let operand = loom.num_inputs + output;
    let elements = super::super::reduction::output_elements(&program.index, loom, output)?;
    let base = ctx.declared.get(program.layout.operand_params[operand]);
    // A mask is only ever read here, and only a kernel writing through the fold
    // declares one: reaching this with a mask the kernel never took is what
    // `BucketMaskValues::get` refuses.
    let bucket = program.layout.bucket_mask_slot(output).and_then(|slot| {
        let mask = ctx.bucket_masks.get(slot);
        program
            .folds
            .output(output)
            .bucket_offset(b, mask, elements as i64)
    });
    let offset = match bucket {
        Some(bucket) => b.add(cell_offset, bucket),
        None => cell_offset,
    };
    Ok(b.offset(base, offset))
}

/// Binds the operand's global base pointer to a local: the element loop then
/// indexes it instead of re-deriving the cell address once per element.
fn bind_base_pointer(
    b: &mut gpu_ir::Builder,
    program: &Plan,
    output: usize,
    name: &str,
    init: gpu_ir::Value,
) -> gpu_ir::Value {
    b.bind_pointer_as(
        name.to_string(),
        PtrTy {
            elem: program.folds.output(output).parameter_scalar(),
            space: Space::Global,
        },
        init,
    )
}

/// This launch's accumulator cell for `output`, scaled to the cell's width.
fn cell_base(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    output: usize,
    fiber_edge: Option<gpu_ir::Value>,
) -> Result<gpu_ir::Value, String> {
    let loom = emit.loom;
    let operand = loom.num_inputs + output;
    let binding = &loom.operands[operand];
    let visible = num_elements(&binding.element_shape)? as i64;
    let cell = operand_flat_index(emit.ctx, b, operand, fiber_edge);
    let width = b.i64(visible * binding.lane_extent);
    let scaled = b.mul(cell, width);
    accumulator_base(b, emit, output, scaled)
}

fn load_input_element(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    input: InputElement,
    u: &gpu_ir::Value,
    fiber_edge: Option<gpu_ir::Value>,
    lanes: u8,
) -> Result<Value, String> {
    let loom = emit.loom;
    let binding = &loom.operands[input.operand];
    let visible = num_elements(&binding.element_shape)? as i64;
    let cell = operand_flat_index(emit.ctx, b, input.operand, fiber_edge);
    let lane = lane_offset(b, binding.lane_extent, u);
    let width = b.i64(visible * binding.lane_extent);
    let scaled = b.mul(cell, width);
    let element = b.i64(input.element as i64 * binding.lane_extent);
    let offset = b.add(scaled, element);
    let offset = b.add(offset, lane);
    let array = emit
        .ctx
        .declared
        .get(emit.program.layout.operand_params[input.operand]);
    let address = b.offset(array, offset);
    let mut loaded = b.load(
        address,
        if binding.lane_extent == 1 { 1 } else { lanes },
        true,
    );
    if binding.lane_extent == 1 && lanes > 1 {
        loaded = b.splat(lanes, loaded);
    }
    if let Some(alive) = operand_alive(emit.ctx, b, input.operand) {
        let dtype = loom
            .module
            .entry_func()
            .value_shape(loom.module.entry_func().params[input.operand])
            .dtype;
        let alive = b.splat(lanes, alive);
        let zero = zero(b, math_scalar(dtype)?);
        let zero = b.splat(lanes, zero);
        loaded = b.select(alive, loaded, zero);
    }
    Ok(loaded)
}

fn output_factor_value(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    factor: &OutputFactor,
    u: &gpu_ir::Value,
    fiber_edge: Option<gpu_ir::Value>,
    lanes: u8,
) -> Result<Value, String> {
    Ok(match factor {
        OutputFactor::Input(input) => load_input_element(b, emit, *input, u, fiber_edge, lanes)?,
        OutputFactor::Constant(literal) => scalar_constant(b, literal, lanes)?,
        OutputFactor::Convert { from, to, input } => {
            let value = output_factor_value(b, emit, input, u, fiber_edge, lanes)?;
            convert_output_factor(b, value, *from, *to, lanes)?
        }
        OutputFactor::Elementwise { op, dtype, inputs } => {
            let values = inputs
                .iter()
                .map(|input| output_factor_value(b, emit, input, u, fiber_edge.clone(), lanes))
                .collect::<Result<Vec<_>, String>>()?;
            output_factor_elementwise(b, *op, *dtype, values, lanes)?
        }
    })
}

/// Apply the factors that are constant over this output's reduction fiber.
#[allow(clippy::too_many_arguments)]
fn owed(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    output: usize,
    element: usize,
    value: Value,
    u: &gpu_ir::Value,
    fiber_edge: Option<gpu_ir::Value>,
    lanes: u8,
) -> Result<Value, String> {
    let loom = emit.loom;
    let mut value = value;
    for factor in loom.output_factors(output, element) {
        let factor = output_factor_value(b, emit, factor, u, fiber_edge.clone(), lanes)?;
        value = b.mul(factor, value);
    }
    Ok(value)
}

/// Which hidden channel of an element's cell this thread writes: a width-1
/// output holds one slot per element, a wide one a slot per channel.
fn lane_offset(b: &mut gpu_ir::Builder, lane_extent: i64, u: &gpu_ir::Value) -> gpu_ir::Value {
    if lane_extent == 1 {
        b.i64(0)
    } else {
        b.cast(Scalar::S64, u.clone())
    }
}

/// Write fiber-scattered values at the exact finer-domain cell visited by the walk.
#[allow(clippy::too_many_arguments)]
pub(super) fn fiber_scatter_writes(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    written: Written<'_>,
    u: &gpu_ir::Value,
    edge: gpu_ir::Value,
    lane: &gpu_ir::Value,
    io_lanes: u8,
) -> Result<Vec<OpId>, String> {
    let (fragment, result) = b.build_fragment(|f| {
        let Emit { program, loom, .. } = emit;
        let Written {
            output,
            values,
            elements,
        } = written;
        if elements.is_empty() {
            return Ok(());
        }
        let operand = loom.num_inputs + output;
        let binding = &loom.operands[operand];
        let result_scalar = output_scalar(loom, output)?;
        let base = cell_base(f, emit, output, Some(edge.clone()))?;
        let base = bind_base_pointer(
            f,
            program,
            output,
            &format!("loom_scatter_base_{output}"),
            base,
        );
        let u_stride = i64::from(WARP_SIZE) * i64::from(io_lanes);
        for selection in elements {
            let element = selection.element;
            let value = owed(
                f,
                emit,
                output,
                element,
                values[element]
                    .as_ref()
                    .expect("selected Loom output element has a value")
                    .clone(),
                u,
                Some(edge.clone()),
                io_lanes,
            )?;
            if binding.lane_extent != 1 {
                let at = f.i64(element as i64 * binding.lane_extent);
                let lane = lane_offset(f, binding.lane_extent, u);
                let offset = f.add(at, lane);
                let ptr = f.offset(base.clone(), offset);
                let evict = if selection.read_later {
                    Evict::Normal
                } else {
                    Evict::First
                };
                reduction::write(
                    f,
                    program,
                    operand,
                    ptr,
                    value,
                    io_lanes,
                    selection.continuation,
                    evict,
                );
                continue;
            }
            let accumulator_scalar = vector_reduction_scalar(result_scalar, io_lanes);
            let mut folded: Option<Value> = None;
            for lane in 0..io_lanes {
                let lane_value = if io_lanes == 1 {
                    value.clone()
                } else {
                    f.extract(lane, value.clone())
                };
                let lane_value = f.cast(accumulator_scalar, lane_value);
                folded = Some(match folded {
                    None => lane_value,
                    Some(held) => f.add(held, lane_value),
                });
            }
            let folded = folded.expect("io_lanes is nonzero");
            let bound = f.bind_value_as(
                format!("loom_scatter_val_{output}_{element}"),
                Ty::scalar(accumulator_scalar),
                folded,
            );
            let summed = warp_sum(f, bound);
            let stored = if accumulator_scalar == result_scalar {
                summed
            } else {
                f.cast(result_scalar, summed)
            };
            let at = f.i64(element as i64);
            let ptr = f.offset(base.clone(), at);
            let evict = if !selection.read_later && loom.lane_count <= u_stride {
                Evict::First
            } else {
                Evict::Normal
            };
            let (guarded, written) = f.build_fragment(|g| {
                if selection.continuation {
                    reduction::write(g, program, operand, ptr, stored, 1, true, evict);
                } else if loom.lane_count <= u_stride {
                    reduction::write(g, program, operand, ptr, stored, 1, false, evict);
                } else {
                    let stride = g.i32(u_stride);
                    let early = g.lt(u.clone(), stride.clone());
                    let (store_ptr, store_value) = (ptr.clone(), stored.clone());
                    g.if_(early, |h| {
                        reduction::write(
                            h,
                            program,
                            operand,
                            store_ptr,
                            store_value,
                            1,
                            false,
                            evict,
                        );
                    });
                    let late = g.ge(u.clone(), stride);
                    g.if_(late, |h| {
                        reduction::write(h, program, operand, ptr, stored, 1, true, evict);
                    });
                }
                Ok::<(), String>(())
            });
            written?;
            let zero = f.i32(0);
            let at_lane_zero = f.eq(lane.clone(), zero);
            let guarded = guarded.into_ops();
            f.if_(at_lane_zero, |h| h.place_all(guarded));
        }
        Ok::<(), String>(())
    });
    result?;
    Ok(b.place_fragment(fragment))
}

pub(super) fn output_writes(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    written: Written<'_>,
    u: &gpu_ir::Value,
    io_lanes: u8,
) -> Result<Vec<OpId>, String> {
    let (fragment, result) = b.build_fragment(|f| {
        let Emit { program, loom, ctx } = emit;
        let Written {
            output,
            values,
            elements,
        } = written;
        if elements.is_empty() {
            return Ok(());
        }
        let operand = loom.num_inputs + output;
        let binding = &loom.operands[operand];
        let func = loom.module.entry_func();
        let dtype = func.value_shape(func.results[output]).dtype;
        let storage = storage_scalar(binding.dtype)?;
        let base = cell_base(f, emit, output, None)?;
        let base = bind_base_pointer(
            f,
            program,
            output,
            &format!("loom_output_base_{output}"),
            base,
        );
        let alive = operand_alive(ctx, f, operand).map(|predicate| {
            f.bind_value_as(
                format!("loom_output_alive_{output}"),
                Ty::scalar(Scalar::Pred),
                predicate,
            )
        });
        for selection in elements {
            let element = selection.element;
            let lanes = if binding.lane_extent == 1 {
                1
            } else {
                io_lanes
            };
            let value = &owed(
                f,
                emit,
                output,
                element,
                values[element]
                    .as_ref()
                    .expect("selected Loom output element has a value")
                    .clone(),
                u,
                None,
                lanes,
            )?;
            let lane = lane_offset(f, binding.lane_extent, u);
            let at = f.i64(element as i64 * binding.lane_extent);
            let offset = f.add(at, lane);
            let value = if dtype == DType::Bool {
                let one = f.int(Scalar::U32, 1);
                let zero = f.int(Scalar::U32, 0);
                let selected = f.select(value.clone(), one, zero);
                f.cast(storage, selected)
            } else {
                value.clone()
            };
            let ptr = f.offset(base.clone(), offset);
            let evict = if selection.read_later {
                Evict::Normal
            } else {
                Evict::First
            };
            match &alive {
                // The write sits under the mask exactly as it did: one guarded
                // region holding one write.
                Some(pred) => f.if_(pred.clone(), |guarded| {
                    reduction::write(
                        guarded,
                        program,
                        operand,
                        ptr,
                        value,
                        lanes,
                        selection.continuation,
                        evict,
                    );
                }),
                None => reduction::write(
                    f,
                    program,
                    operand,
                    ptr,
                    value,
                    lanes,
                    selection.continuation,
                    evict,
                ),
            }
        }
        Ok::<(), String>(())
    });
    result?;
    Ok(b.place_fragment(fragment))
}
