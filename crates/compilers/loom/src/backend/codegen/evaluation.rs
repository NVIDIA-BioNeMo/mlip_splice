// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{OpId, OpKind, Value};

use super::{
    AxisBridge, DType, DomainIndex, Emit, EmitCtx, Evaluator, Evict, GuardedRegion, Index,
    KernelPart, LiftedProgram, MaterializedValue, Param, ParamTy, Plan, PtrTy, Scalar,
    SelectedOutputs, Space, Ty, Written, contract_fma_bindings, group_bindings_by_element,
    math_scalar, num_elements, operand_alive, operand_flat_index, operand_has_alive, output_scalar,
    output_writes, selected_bindings, selected_term_values, slice_bindings, storage_scalar, zero,
};

/// What one launch evaluates at one position: the values its outputs reach, the
/// steps that produce them, and the guard those steps run under.
pub(super) struct Evaluation {
    pub(super) outputs: SelectedOutputs,
    pub(super) ops: Vec<OpId>,
    pub(super) guard_prefix: Vec<OpId>,
    pub(super) guard: Option<Value>,
}

/// Where in the iteration an evaluation sits: the position it runs for, the
/// hidden-lane index, the fiber edge a walk gathers through, and the lane width
/// the loads carry.
#[derive(Clone)]
pub(super) struct At {
    pub flat: Value,
    pub u: Value,
    pub fiber_edge: Option<Value>,
    pub io_lanes: u8,
}

/// What the program reads, as this kernel reaches it: the cell pointer of each
/// input, the lane fold where it carries lanes, and the liveness predicate where
/// its bridges leave cells unmapped.
///
/// The cached evaluation names these as parameters of its own arena, so what a
/// kernel builds here is what those parameters stand for when it imports it.
type InputAccess = Vec<(String, Value)>;

/// What each parameter the cached evaluation declares stands for in the kernel
/// importing it.
///
/// The name is the contract between the two sides, because the access and the
/// evaluation are built by different code against the same program and their
/// values belong to different arenas. Which value of the evaluation carries a
/// parameter is that arena's own question, so it is answered by the declaration
/// rather than looked up by the name again.
pub(super) fn access_substitutions(
    source: &gpu_ir::Function,
    access: &InputAccess,
) -> Result<Vec<(gpu_ir::ValueId, Value)>, String> {
    source
        .declarations()
        .iter()
        .enumerate()
        .filter_map(|(index, declared)| match declared {
            gpu_ir::Declaration::Param(param) => Some((index, param)),
            gpu_ir::Declaration::Shared(_) => None,
        })
        .map(|(index, param)| {
            let value = access
                .iter()
                .find(|(name, _)| *name == param.name)
                .map(|(_, value)| value.clone())
                .ok_or_else(|| {
                    format!(
                        "loom evaluation reads {}, which this kernel does not bind",
                        param.name
                    )
                })?;
            Ok((source.declared(index), value))
        })
        .collect()
}

pub(super) fn evaluate_at(
    builder: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    at: At,
    part: &KernelPart,
    materialized: &[MaterializedValue],
    guard: Option<&GuardedRegion>,
) -> Result<Evaluation, String> {
    let Emit { program, loom, ctx } = emit;
    let At {
        flat,
        u,
        fiber_edge,
        io_lanes,
    } = at;
    let evaluation = expand_program_for_lanes(program, loom, io_lanes)?;
    // The access this kernel binds, then the arithmetic that reads it: one
    // stream, sliced as one, exactly as when the two were built together.
    let (access_ops, access) =
        build_input_access(builder, program, loom, ctx, &u, fiber_edge.as_ref())?;

    let source = evaluation.body.function();
    let mut substitutions = access_substitutions(source, &access)?;
    substitutions.reserve(materialized.len());
    // A materialized value is written out by one kernel and read back by
    // another: the load is what its consumers see, and what it used to read is
    // not imported.
    for (index, value) in materialized.iter().enumerate() {
        let held = source.value_named(&value.name).ok_or_else(|| {
            format!(
                "loom materialization names {}, which the evaluation does not bind",
                value.name
            )
        })?;
        let base = builder
            .parameters()
            .iter()
            .position(|param| param.name == materialized_name(index))
            .map(|position| builder.function().declared(position))
            .map(|declared| builder.value_of(declared))
            .ok_or_else(|| "loom materialized value has no kernel parameter".to_string())?;
        let width = builder.i64(loom.lane_count);
        let scaled = builder.mul(flat.clone(), width);
        let lane = builder.cast(Scalar::S64, u.clone());
        let offset = builder.add(scaled, lane);
        let cell = builder.offset(base, offset);
        let loaded = builder.load(cell, 1, true);
        substitutions.push((held, loaded));
    }
    let mut import = builder.import(&evaluation.body, &substitutions)?;

    let mut statements = access_ops.clone();
    statements.extend(import.steps(builder.function()));
    let mut outputs = Vec::with_capacity(evaluation.outputs.len());
    for values in &evaluation.outputs {
        let mut carried = Vec::with_capacity(values.len());
        for &value in values {
            let held = builder.imported(&mut import, source, value)?;
            carried.push(held.id_in(builder.function()));
        }
        outputs.push(carried);
    }

    let runtime_part = part.clone();
    let (terms, roots) =
        selected_term_values(builder.function(), &statements, &outputs, &runtime_part)?;
    let mut reads = std::collections::BTreeSet::new();
    for root in &roots {
        root.add_reads(builder.function(), &mut reads);
    }
    let reads = reads.into_iter().collect::<Vec<_>>();
    let mut bindings = slice_bindings(builder.function(), &statements, &reads)?;
    let (guard_prefix, guard) = match guard {
        Some(region) => {
            let held = evaluation
                .statement_guards
                .values()
                .find(|guard| guard.key == region.key)
                .map(|guard| guard.predicate)
                .ok_or("loom guarded region lost its runtime predicate")?;
            let predicate = builder.imported(&mut import, source, held)?;
            let held = predicate.id_in(builder.function());
            if builder
                .function()
                .operand_op(held)
                .is_some_and(|op| matches!(builder.function().op(op).kind, OpKind::WarpAny))
            {
                return Err("loom guarded region votes on a value already voted on".into());
            }
            let needed = slice_bindings(builder.function(), &statements, &[held])?;
            // The input access is the kernel's, not the arm's: a base pointer is
            // the same address whether the guard skips the work or not, and it is
            // read by both arms and by the writes after them. Keeping it in the
            // guard's body would place it inside the taken arm, where a value it
            // defines dies at the arm's end while the rest of the kernel still
            // reads it — which is the dominance rule the verifier states, and
            // exactly what refused every zero-skip candidate.
            let hoisted = access_ops
                .iter()
                .copied()
                .chain(needed)
                .collect::<std::collections::BTreeSet<_>>();
            // In the order the statements hold them, so a binding still follows
            // whatever it reads.
            let prefix = statements
                .iter()
                .copied()
                .filter(|op| hoisted.contains(op))
                .collect::<Vec<_>>();
            bindings.retain(|op| !prefix.contains(op));
            (prefix, Some(predicate))
        }
        None => (Vec::new(), None),
    };
    let func = loom.module.entry_func();
    let element_tys = func
        .results
        .iter()
        .map(|&result| {
            Ok(Ty::vec(
                math_scalar(func.value_shape(result).dtype)?,
                io_lanes,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (mut bindings, mut selected) =
        group_bindings_by_element(builder, bindings, &terms, program.element_order, |output| {
            element_tys[output]
        })?;
    let mut roots = selected_bindings(&selected);
    bindings = contract_fma_bindings(builder, bindings, &mut roots)?;
    let mut roots = roots.into_iter();
    for value in selected.iter_mut().flatten().filter_map(Option::as_mut) {
        *value = roots
            .next()
            .expect("selected Loom output has a sliced binding");
    }
    debug_assert!(roots.next().is_none());
    Ok(Evaluation {
        outputs: selected,
        ops: bindings,
        guard_prefix,
        guard,
    })
}

/// One accumulator a guarded region folds into: what it held coming in, and
/// what the fold inside the region reached.
pub(super) struct Folded {
    pub before: Value,
    pub after: Value,
}

/// The steps of a guarded region, and what each accumulator it folds into holds
/// once it has run.
///
/// A fold inside the region is the region's own value, and it leaves through
/// what the region yields — so the guard produces one result per accumulator,
/// into the place it came from: the arm that runs hands back what it reached,
/// and the arm that does not hands back what the accumulator already held.
pub(super) fn guarded_body(
    b: &mut gpu_ir::Builder,
    mut prefix: Vec<OpId>,
    guard: Option<Value>,
    active: Vec<OpId>,
    inactive: Vec<OpId>,
    folded: Vec<Folded>,
) -> (Vec<OpId>, Vec<Value>) {
    let Some(guard) = guard else {
        prefix.extend(active);
        return (prefix, folded.into_iter().map(|held| held.after).collect());
    };
    let results = folded
        .iter()
        .map(|held| (b.value_ty(&held.before).data(), Some(held.before.clone())))
        .collect::<Vec<_>>();
    let reached = folded
        .iter()
        .map(|held| held.after.clone())
        .collect::<Vec<_>>();
    let held_already = folded
        .into_iter()
        .map(|held| held.before)
        .collect::<Vec<_>>();
    let (fragment, produced) = b.build_fragment(|f| {
        let voted = f.warp_any(guard);
        f.open_arm();
        f.place_all(active);
        let taken = f.close_arm(reached);
        f.open_arm();
        f.place_all(inactive);
        let untaken = f.close_arm(held_already);
        f.if_arms(voted, &results, taken, untaken)
    });
    prefix.extend(b.place_fragment(fragment));
    (prefix, produced)
}

pub(super) fn inactive_direct_writes(
    b: &mut gpu_ir::Builder,
    emit: Emit<'_>,
    part: &KernelPart,
    u: &Value,
    io_lanes: u8,
) -> Result<Vec<OpId>, String> {
    let loom = emit.loom;
    let mut out = Vec::new();
    for (output, binding) in loom.outputs().iter().enumerate() {
        if binding.lane_extent == 1 {
            continue;
        }
        let scalar = output_scalar(loom, output)?;
        let selections = part.elements[output]
            .iter()
            .filter(|selection| !selection.continuation)
            .cloned()
            .collect::<Vec<_>>();
        if selections.is_empty() {
            continue;
        }
        let zero = zero(b, scalar);
        let zero = if io_lanes == 1 {
            zero
        } else {
            b.splat(io_lanes, zero)
        };
        let mut values = vec![None; num_elements(&binding.element_shape)?];
        for selection in &selections {
            values[selection.element] = Some(zero.clone());
        }
        out.extend(output_writes(
            b,
            emit,
            Written {
                output,
                values: &values,
                elements: &selections,
            },
            u,
            io_lanes,
        )?);
    }
    Ok(out)
}

/// Binds what each input is reached through in this kernel: its cell pointer,
/// the lane fold when it carries lanes, and the liveness predicate when its
/// bridges leave cells unmapped.
///
/// Built by the builder emitting the kernel, because where an operand sits is
/// that kernel's question. The arithmetic that reads these is not, and is
/// cached: see [`expand_program_for_lanes`].
pub(super) fn build_input_access(
    builder: &mut gpu_ir::Builder,
    program: &Plan,
    loom: &LiftedProgram,
    ctx: &EmitCtx<'_, Index>,
    u: &Value,
    fiber_edge: Option<&Value>,
) -> Result<(Vec<OpId>, InputAccess), String> {
    let (fragment, access) = builder.build_fragment(|f| {
        let mut access = InputAccess::new();
        for (operand, binding) in loom.operands[..loom.num_inputs].iter().enumerate() {
            let storage = storage_scalar(binding.dtype)?;
            let visible = num_elements(&binding.element_shape)? as i64;
            let cell = operand_flat_index(ctx, f, operand, fiber_edge.cloned());
            let ptr_ty = PtrTy {
                elem: storage,
                space: Space::Global,
            };
            let width = f.i64(visible * binding.lane_extent);
            let scaled = f.mul(cell, width);
            let array = ctx.declared.get(program.layout.operand_params[operand]);
            let address = f.offset(array, scaled);
            let base = f.bind_pointer_as(format!("loom_input_base_{operand}"), ptr_ty, address);
            access.push((format!("loom_input_base_{operand}"), base.clone()));
            // The lane folds into a per-operand pointer so every element load
            // is a constant byte offset off one register — otherwise each load
            // costs its own mad/add address chain, and on unified INT/FP32
            // cores that integer work steals FP throughput in the fiber loop.
            if binding.lane_extent != 1 {
                let lane = f.cast(Scalar::S64, u.clone());
                let address = f.offset(base, lane);
                let lane_base =
                    f.bind_pointer_as(format!("loom_input_lane_base_{operand}"), ptr_ty, address);
                access.push((format!("loom_input_lane_base_{operand}"), lane_base));
            }
            if let Some(predicate) = operand_alive(ctx, f, operand) {
                let alive = f.bind_value_as(
                    format!("loom_input_alive_{operand}"),
                    Ty::scalar(Scalar::Pred),
                    predicate,
                );
                access.push((format!("loom_input_alive_{operand}"), alive));
            }
        }
        Ok::<InputAccess, String>(access)
    });
    Ok((builder.place_fragment(fragment), access?))
}

/// The arithmetic an input feeds, over the access it is reached through.
///
/// It reads the operands as parameters of its own arena and never says where
/// they come from, so it depends on the program and the lane width and on
/// nothing about the kernel emitting it — which is why the lane width alone keys
/// the cache, and why a kernel imports it over an access of its own.
pub(super) fn expand_program_for_lanes(
    program: &Plan,
    loom: &LiftedProgram,
    io_lanes: u8,
) -> Result<std::sync::Arc<super::super::super::eval::Evaluation>, String> {
    let key = io_lanes;
    if let Some((_, hit)) = program
        .eval_cache
        .lock()
        .map_err(|_| "loom evaluation cache poisoned")?
        .iter()
        .find(|(cached, _)| *cached == key)
    {
        return Ok(hit.clone());
    }
    let func = loom.module.entry_func();
    let indexed_program = loom.has_indexed_ops();
    let mut evaluator = Evaluator::new(&loom.module, io_lanes);
    let mut params = Vec::with_capacity(loom.num_inputs);
    let mut indexed_params = Vec::with_capacity(loom.num_inputs);
    for (operand, (&param, binding)) in func
        .params
        .iter()
        .zip(&loom.operands[..loom.num_inputs])
        .enumerate()
    {
        let dtype = func.value_shape(param).dtype;
        let scalar = math_scalar(dtype)?;
        let storage = storage_scalar(binding.dtype)?;
        let visible = num_elements(&binding.element_shape)? as i64;
        let base_name = if binding.lane_extent == 1 {
            format!("loom_input_base_{operand}")
        } else {
            format!("loom_input_lane_base_{operand}")
        };
        let base = evaluator.builder().parameter(Param {
            name: base_name,
            ty: ParamTy::Ptr(storage),
        });
        let alive = operand_has_alive(&program.index, operand).then(|| {
            evaluator.builder().parameter(Param {
                name: format!("loom_input_alive_{operand}"),
                ty: ParamTy::Scalar(Scalar::Pred),
            })
        });
        let evict = gather_evict(&program.index, loom, operand);
        let load_lanes = if binding.lane_extent == 1 {
            1
        } else {
            io_lanes
        };
        let elems = (0..visible)
            .map(|element| {
                let b = evaluator.builder();
                let offset = b.i64(element * binding.lane_extent);
                let cell = b.offset(base.clone(), offset);
                let loaded = b.load_evict(cell, load_lanes, true, evict);
                let loaded = if binding.lane_extent == 1 && io_lanes > 1 {
                    b.splat(io_lanes, loaded)
                } else {
                    loaded
                };
                let value = if dtype == DType::Bool {
                    let bits = b.cast(Scalar::U32, loaded);
                    let zero = b.int(Scalar::U32, 0);
                    let zero = b.splat(io_lanes, zero);
                    b.cmp(gpu_ir::CmpOp::Ne, bits, zero)
                } else {
                    loaded
                };
                Ok(match &alive {
                    Some(pred) if scalar == Scalar::Pred => {
                        let pred = b.splat(io_lanes, pred.clone());
                        b.and(pred, value)
                    }
                    Some(pred) => {
                        let zero = zero(b, scalar);
                        let zero = b.splat(io_lanes, zero);
                        b.select(pred.clone(), value, zero)
                    }
                    None => value,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if indexed_program {
            indexed_params.push(super::super::super::eval::IndexedTensor::pointer(
                base,
                scalar,
                binding.lane_extent,
                load_lanes,
                io_lanes,
                dtype == DType::Bool,
                alive,
                evict,
                visible as usize,
            ));
        }
        params.push(elems);
    }
    let indexed_params = indexed_program.then_some(indexed_params);
    evaluator.bind_params(params, indexed_params)?;
    let result = std::sync::Arc::new(evaluator.run_with_provenance()?);
    program
        .eval_cache
        .lock()
        .map_err(|_| "loom evaluation cache poisoned")?
        .push((key, result.clone()));
    Ok(result)
}

/// `evict_last` for a fiber-walk gather of a coarser-domain operand: the walk
/// revisits each of its elements ~fanout times (reuse cutoff as in
/// `plan_traffic_clusters`: 2 x operand positions <= walk positions). Width-1
/// operands and broadcast weights stay unhinted.
fn gather_evict(index: &Index, loom: &LiftedProgram, operand: usize) -> Evict {
    if loom.operands[operand].lane_extent == 1 {
        return Evict::Normal;
    }
    let through_view = (0..index.num_axes()).any(|axis| {
        matches!(
            index.axis_bridge(operand, axis),
            AxisBridge::AlignedLookup { .. }
        )
    });
    if !through_view {
        return Evict::Normal;
    }
    let Some((map_idx, _)) = index.fiber_bridge() else {
        return Evict::Normal;
    };
    let finer = index.graph().index_maps[map_idx].source_id;
    let walk_positions = index.graph().domain_extent(finer);
    match index.operand_batch_extent(operand) {
        Ok(extent) if 2 * extent <= walk_positions => Evict::Last,
        _ => Evict::Normal,
    }
}

pub(super) fn materialized_name(index: usize) -> String {
    format!("loom_material_{index}")
}

pub(super) fn materialized_params(
    values: &[MaterializedValue],
) -> impl Iterator<Item = Param> + use<'_> {
    values.iter().enumerate().map(|(index, value)| Param {
        name: materialized_name(index),
        ty: ParamTy::Ptr(value.scalar),
    })
}

pub(super) fn materialized_param(index: usize, value: &MaterializedValue) -> Param {
    Param {
        name: materialized_name(index),
        ty: ParamTy::Ptr(value.scalar),
    }
}
