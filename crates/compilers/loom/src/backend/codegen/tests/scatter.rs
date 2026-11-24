// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A scatter whose indices are only known at run time: one packed binding,
//! walked with a loop, branching once per destination slot.

use xir::dim_numbers::ScatterDimensionNumbers;
use xir::ops::{Add, Scatter};

use super::*;
use crate::backend::partition::{Bindings, evaluation_profile_with_provenance};

const SLOTS: usize = 3;
const UPDATES: usize = 4;

/// `operand[indices[k]] += updates[k]`, with the indices a run-time operand
/// rather than a literal — the shape `xir`'s exact reduction emits, and the
/// only one that reaches the walked lowering.
fn runtime_scatter_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("add");
    let left = builder.param(Shape::scalar(DType::F32));
    let right = builder.param(Shape::scalar(DType::F32));
    let sum = builder.op(Add, &[left, right])[0];
    builder.end_func(vec![sum]);

    builder.begin_func("main");
    let operand = builder.param(Shape::new(DType::F32, [SLOTS as i64]));
    let indices = builder.param(Shape::new(DType::I32, [UPDATES as i64, 1]));
    let updates = builder.param(Shape::new(DType::F32, [UPDATES as i64]));
    let scattered = builder.op(
        Scatter {
            dnums: ScatterDimensionNumbers {
                update_window_dims: Default::default(),
                inserted_window_dims: [0].into_iter().collect(),
                input_batching_dims: Default::default(),
                scatter_indices_batching_dims: Default::default(),
                scatter_dims_to_operand_dims: [0].into_iter().collect(),
                index_vector_dim: 1,
            },
            indices_are_sorted: false,
            unique_indices: false,
            update: xir::ir::FuncId(0),
        },
        &[operand, indices, updates],
    )[0];
    let main = builder.end_func(vec![scattered]);

    let binding = |dtype, visible_shape: Vec<i64>| FlatOperand {
        dtype,
        element_shape: visible_shape,
        lane_extent: 1,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "runtime_scatter".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 1,
        domains: vec![Domain {
            extent: 5,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::F32, vec![SLOTS as i64]),
            binding(Dtype::I32, vec![UPDATES as i64, 1]),
            binding(Dtype::F32, vec![UPDATES as i64]),
            binding(Dtype::F32, vec![SLOTS as i64]),
        ],
        num_inputs: 3,
        available_views: Vec::new(),
    }
}

/// The one binding a walked scatter is: its init reaches the loop that stands
/// for the value it reduces to.
fn walked(f: &gpu_ir::Function, ops: &[gpu_ir::OpId]) -> Vec<gpu_ir::OpId> {
    fn reaches_a_loop(f: &gpu_ir::Function, value: gpu_ir::ValueId) -> bool {
        let Some(op) = f.operand_op(value) else {
            return false;
        };
        matches!(f.op(op).kind, gpu_ir::OpKind::Loop { .. })
            || f.op(op)
                .operands
                .iter()
                .any(|&operand| reaches_a_loop(f, operand))
    }
    ops.iter()
        .copied()
        .filter(|&op| reaches_a_loop(f, f.op(op).operands[0]))
        .collect()
}

/// The loop a walked scatter's binding holds.
fn walk_of(f: &gpu_ir::Function, packed: gpu_ir::OpId) -> gpu_ir::OpId {
    f.operand_op(f.op(packed).operands[0])
        .expect("the binding holds the walk")
}

/// The profile is bindings and nothing else, and the whole walked scatter is
/// one of them: one packed value, one definition, so the partition metric sees
/// exactly what the folded expression cost before.
#[test]
fn a_runtime_scatter_is_one_packed_binding_of_one_operand_loop() {
    let loom = runtime_scatter_program();
    let profile = evaluation_profile_with_provenance(&loom).unwrap();
    let f = profile.body.function();
    let ops = profile.body.steps();

    // A body that is not flat bindings is one no profile consumer can read.
    Bindings::of(f, &ops).expect("the profile is flat bindings");
    assert!(
        ops.iter()
            .all(|&op| matches!(f.op(op).kind, gpu_ir::OpKind::Bind)),
        "the profile body holds a step that is not a binding"
    );

    let walked = walked(f, &ops);
    assert_eq!(walked.len(), 1, "one binding holds the whole walk");
    let packed = walked[0];
    let held = f.op(packed).results[0];
    assert_eq!(
        f.data_ty(held),
        gpu_ir::Ty::vec(Scalar::F32, SLOTS as u8),
        "the accumulators stay packed in one value"
    );
    let walk = walk_of(f, packed);
    assert!(
        !f.op(walk).placed,
        "a loop standing for the value it reduces to is an operand"
    );
    assert_eq!(
        f.op(walk).results.len(),
        1,
        "the walk carries one packed value and reduces to one"
    );
    assert_eq!(
        f.data_ty(f.op(walk).results[0]),
        gpu_ir::Ty::vec(Scalar::F32, SLOTS as u8),
        "and that value is the packed accumulator"
    );

    // Each slot is read back out of that one binding, as it was when the fold
    // was one expression.
    let reads = profile.outputs[0]
        .iter()
        .filter(|&&value| {
            f.defining_op(value).is_some_and(|bound| {
                f.operand_op(f.op(bound).operands[0]).is_some_and(|read| {
                    matches!(f.op(read).kind, gpu_ir::OpKind::Extract(_))
                        && f.op(read).operands[0] == held
                })
            })
        })
        .count();
    assert_eq!(
        reads, SLOTS,
        "every slot is extracted from the packed value"
    );
}

/// The walk's body reads its destination and its update once per iteration and
/// branches once per slot, which is what makes widening the slot count cost
/// branches rather than loads.
#[test]
fn a_runtime_scatter_reads_each_update_once_per_iteration() {
    let loom = runtime_scatter_program();
    let profile = evaluation_profile_with_provenance(&loom).unwrap();
    let f = profile.body.function();
    let packed = walked(f, &profile.body.steps())[0];
    let body = f.region_block(f.op(walk_of(f, packed)).regions[0]);

    let top = f.block(body).ops.clone();
    let bound = top
        .iter()
        .filter(|&&op| matches!(f.op(op).kind, gpu_ir::OpKind::Bind))
        .count();
    let guards = top
        .iter()
        .filter(|&&op| matches!(f.op(op).kind, gpu_ir::OpKind::If))
        .count();
    assert_eq!(
        bound, 2,
        "the destination and the update are each bound once per iteration"
    );
    assert_eq!(guards, SLOTS, "one branch per destination slot");
    assert_eq!(
        f.block(body).params.len(),
        2,
        "the walk binds its index and the one packed accumulator"
    );
    assert_eq!(
        f.data_ty(f.block(body).params[1]),
        gpu_ir::Ty::vec(Scalar::F32, SLOTS as u8),
    );
    assert_eq!(
        f.block(body).terminator.yields.len(),
        1,
        "and yields that one value"
    );
    // Every branch produces into the accumulator the loop carries, so a slot
    // updates its own chunk of it in place.
    for guard in top
        .iter()
        .filter(|&&op| matches!(f.op(op).kind, gpu_ir::OpKind::If))
    {
        assert_eq!(f.op(*guard).results.len(), 1);
        assert_eq!(
            f.op(*guard).destination(0),
            Some(f.block(body).params[1]),
            "a slot's branch produces into the packed accumulator"
        );
    }
}

/// The lines of the innermost loop's body: those after the last loop label and
/// before the end label that follows it.
fn innermost_loop_body(ptx: &str) -> String {
    let lines = ptx.lines().collect::<Vec<_>>();
    let top = lines
        .iter()
        .rposition(|line| line.starts_with("$Lloop_") && line.ends_with(':'))
        .expect("a loop label");
    let end = lines[top..]
        .iter()
        .position(|line| line.starts_with("$Ldone_") && line.ends_with(':'))
        .expect("a loop end")
        + top;
    lines[top + 1..end].join("\n")
}

/// The whole path, to PTX: the profile partitions, a kernel emits, and the
/// rendered text walks the updates once, reading each update once and
/// combining it into the slot its destination names.
#[test]
fn a_runtime_scatter_partitions_and_renders() {
    let loom = runtime_scatter_program();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert_eq!(parts.len(), 1);
    let prepared = Plan::prepare(&loom).unwrap();
    let kernel = emit_kernel(&prepared, &loom, BLOCK_SIZES[0], 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);

    assert!(ptx.contains(".entry loom_runtime_scatter"), "{ptx}");
    // Two counted loops and no more: the hidden axis, and the walk over the
    // updates inside it. Were the walk rebuilt per slot there would be four.
    assert_eq!(
        ptx.lines()
            .filter(|line| line.starts_with("$Lloop_") && line.ends_with(':'))
            .count(),
        2,
        "the scatter is walked once, not once per slot: {ptx}"
    );
    let walk = innermost_loop_body(&ptx);
    let walk = walk.as_str();
    // One update read per iteration, however many slots it may land in.
    assert_eq!(
        walk.matches("ld.global.nc.f32").count(),
        1,
        "the update is read once per iteration, not once per slot: {ptx}"
    );
    assert_eq!(
        walk.matches("setp.eq.s32").count(),
        SLOTS,
        "one destination comparison per slot: {ptx}"
    );
    // Each combine writes the accumulator it reads, under its own branch, so a
    // slot costs a comparison and a branch and nothing is copied to join them.
    assert_eq!(
        walk.matches("add.rn.f32").count(),
        SLOTS,
        "one combine per slot: {ptx}"
    );
    assert!(
        !walk.contains("mov.f32"),
        "a slot update is produced into the accumulator it carries: {ptx}"
    );
    for slot in 0..SLOTS {
        assert!(
            walk.contains(&format!(", {slot};")),
            "slot {slot} is compared against its own destination: {ptx}"
        );
    }
}

/// The walk's registers, seen whole: each slot's combine writes the chunk of
/// the packed accumulator it owns, under its own branch, and no chunk it does
/// not own is copied.
#[test]
fn a_runtime_scatter_updates_each_chunk_in_place() {
    let loom = runtime_scatter_program();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let prepared = Plan::prepare(&loom).unwrap();
    let kernel = emit_kernel(&prepared, &loom, BLOCK_SIZES[0], 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);
    let walk = innermost_loop_body(&ptx);

    // One `add` per slot, each writing the register it reads first, and the
    // three of them writing three different registers.
    let combines = walk
        .lines()
        .filter(|line| line.trim_start().starts_with("add.rn.f32"))
        .map(|line| {
            let operands = line
                .trim()
                .trim_end_matches(';')
                .split_once(' ')
                .unwrap()
                .1
                .split(", ")
                .map(str::to_owned)
                .collect::<Vec<_>>();
            assert_eq!(operands[0], operands[1], "a slot combines in place: {ptx}");
            operands[0].clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(combines.len(), SLOTS, "{ptx}");
    let mut distinct = combines.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        SLOTS,
        "each slot owns its own chunk of the packed accumulator: {ptx}"
    );
}

/// The same walk under the policy that contracts. Contraction rebuilds every
/// binding, and a structured operation is rebuilt through its region — once,
/// with every result it produces standing for the one opposite it. Rebuilding
/// it per read would put one walk in the text per slot.
#[test]
fn an_algebraic_runtime_scatter_still_walks_its_updates_once() {
    let loom = runtime_scatter_program();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    assert_eq!(parts.len(), 1);
    let prepared = Plan::prepare(&loom).unwrap();
    let kernel = emit_kernel(&prepared, &loom, BLOCK_SIZES[0], 0, 1, &parts[0]).unwrap();
    let ptx = gpu_ir::ptx::render_ptx_module(&[kernel], 120);

    assert_eq!(
        ptx.lines()
            .filter(|line| line.starts_with("$Lloop_") && line.ends_with(':'))
            .count(),
        2,
        "the contracted scatter is walked once, not once per slot: {ptx}"
    );
    let walk = innermost_loop_body(&ptx);
    assert_eq!(
        walk.matches("ld.global.nc.f32").count(),
        1,
        "the update is still read once per iteration: {ptx}"
    );
    assert_eq!(
        walk.matches("setp.eq.s32").count(),
        SLOTS,
        "one destination comparison per slot: {ptx}"
    );
    assert_eq!(
        walk.matches("add.rn.f32").count(),
        SLOTS,
        "one combine per slot: {ptx}"
    );
}
