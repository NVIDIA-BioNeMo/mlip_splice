// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::Trip;

use super::super::name::KernelName;
use super::{
    BucketMaskValues, DeviceFunction, DomainIndex, EmitCtx, Launch, LiftedProgram,
    MaterializedValue, ParameterValues, Plan, Scalar, Ty, WARP_SIZE, access_substitutions,
    build_input_access, contract_fma_bindings, expand_program_for_lanes, materialized_name,
    materialized_param, push_axis_indices, slice_bindings,
};

pub(in super::super) fn emit_producer_kernel(
    program: &Plan,
    loom: &LiftedProgram,
    block_size: u32,
    output: usize,
    inputs: &[usize],
    materialized: &[MaterializedValue],
) -> Result<Launch, String> {
    if program.index.fiber_bridge().is_some() {
        return Err("loom materialization does not support fiber iteration".into());
    }
    let u = i32::try_from(loom.lane_count)
        .map_err(|_| format!("loom u={} exceeds i32", loom.lane_count))?;
    let target = materialized
        .get(output)
        .ok_or("loom producer materialized output is out of range")?;
    let entry = KernelName::producer(&loom.name, output, block_size).symbol();
    let positions_per_block = block_size / WARP_SIZE;
    let flat_name = "loom_position";
    let lane_name = "loom_lane";
    let num_outputs = loom.outputs().len();
    let input_params = program.layout.params.len() - num_outputs;
    let mut body = gpu_ir::Builder::new();
    // The program's inputs, then the workspaces this producer reads, then the
    // one it fills. Every input it reads was produced earlier, so no slot it
    // reads is the slot it writes.
    assert!(
        inputs.iter().all(|&input| input < output),
        "a producer reads only workspaces produced before its own"
    );
    // The layout's own prefix — everything but the outputs, which a producer
    // never writes — then the workspaces, which are not layout parameters and
    // take no slot.
    let declared = ParameterValues::declare(&mut body, &program.layout, input_params);
    for param in inputs
        .iter()
        .map(|&input| materialized_param(input, &materialized[input]))
        .chain(std::iter::once(materialized_param(output, target)))
    {
        body.parameter(param);
    }
    let lane = {
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let warp = body.i32(WARP_SIZE as i64);
        body.rem(tid, warp)
    };
    let lane = body.bind_value_as(lane_name, Ty::scalar(Scalar::S32), lane);
    let flat = {
        let block = body.ctaid_x();
        let block = body.cast(Scalar::S64, block);
        let per_block = body.i64(positions_per_block as i64);
        let blocks = body.mul(block, per_block);
        let tid = body.tid_x();
        let tid = body.cast(Scalar::S32, tid);
        let warp = body.i32(WARP_SIZE as i64);
        let within = body.div(tid, warp);
        let within = body.cast(Scalar::S64, within);
        body.add(blocks, within)
    };
    let flat = body.bind_value_as(flat_name, Ty::scalar(Scalar::S64), flat);
    let total = body.i64(program.index.batch_total()?);
    let past_end = body.ge(flat.clone(), total);
    body.if_(past_end, |inner| inner.ret());
    let axis_indices = push_axis_indices(&mut body, &flat, &program.index.batch_sizes());
    // A producer fills a workspace, never through the fold, so it takes none.
    let no_masks = BucketMaskValues::none();
    let ctx = EmitCtx {
        index: &program.index,
        layout: &program.layout,
        declared: &declared,
        bucket_masks: &no_masks,
        axis_indices: &axis_indices,
        view_elements: &[],
        fiber_index: None,
    };
    let evaluation = expand_program_for_lanes(program, loom, 1)?;
    let start = lane;
    let stop = body.i32(u as i64);
    let step = body.i32(WARP_SIZE as i64);
    let open = body.open_loop_body(Scalar::S32, Vec::new());
    let trip_u = open.index();
    body.name_value(&trip_u, "loom_u");
    let built = (|f: &mut gpu_ir::Builder| -> Result<(), String> {
        // The access this kernel binds, then the arithmetic that reads it, then
        // the slice: one stream, so the slice keeps exactly the bindings it used
        // to.
        let (access_ops, access) = build_input_access(f, program, loom, &ctx, &trip_u, None)?;
        let source = evaluation.body.function();
        let mut substitutions = access_substitutions(source, &access)?;
        substitutions.reserve(inputs.len());
        let position = |f: &mut gpu_ir::Builder| {
            let width = f.i64(loom.lane_count);
            let scaled = f.mul(flat.clone(), width);
            let lane = f.cast(Scalar::S64, trip_u.clone());
            f.add(scaled, lane)
        };
        for &input in inputs {
            let value = materialized
                .get(input)
                .ok_or("loom producer materialized input is out of range")?;
            let held = source.value_named(&value.name).ok_or_else(|| {
                format!(
                    "loom materialization names {}, which the evaluation does not bind",
                    value.name
                )
            })?;
            let base = kernel_parameter(f, &materialized_name(input))
                .ok_or("loom producer materialized input has no parameter")?;
            let offset = position(f);
            let cell = f.offset(base, offset);
            let loaded = f.load(cell, 1, true);
            substitutions.push((held, loaded));
        }
        let import = f.import(&evaluation.body, &substitutions)?;
        let mut statements = access_ops;
        statements.extend(import.steps(f.function()));
        let held = source
            .value_named(&target.name)
            .ok_or("loom producer target is not a binding of the evaluation")?;
        let root = import.value(held);
        let mut roots = [root];
        let root_ids = [roots[0].id_in(f.function())];
        let sliced = slice_bindings(f.function(), &statements, &root_ids)?;
        let sliced = contract_fma_bindings(f, sliced, &mut roots)?;
        f.replace_placed(sliced);
        let base = kernel_parameter(f, &materialized_name(output))
            .ok_or("loom producer output has no parameter")?;
        let offset = position(f);
        let cell = f.offset(base, offset);
        let [root] = roots;
        f.store(cell, root);
        Ok(())
    })(&mut body);
    let trip = body.close_loop_body(open, Vec::new());
    built?;
    body.place_counted_loop(Trip::PerThread, start, stop, step, trip);
    Ok(Launch::new(
        DeviceFunction::new(entry, body.into_function()),
        block_size,
        None,
    ))
}

/// The value standing for a parameter this kernel declared, by name.
fn kernel_parameter(b: &gpu_ir::Builder, name: &str) -> Option<gpu_ir::Value> {
    let position = b.function().declarations().iter().position(
        |declared| matches!(declared, gpu_ir::Declaration::Param(param) if param.name == name),
    )?;
    Some(b.value_of(b.function().declared(position)))
}
