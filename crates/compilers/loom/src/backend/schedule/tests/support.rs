// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a rendered body is asked about, over the arena it is.
//!
//! Every question here is one of two walks: over the steps a block runs, or over
//! the operand cone a step reads. Both are written once, and each helper is the
//! predicate it counts or looks for.

use gpu_ir::{BinOp, BlockId, Function, OpId, OpKind, Scalar, ValueId, WriteMode};

/// The steps of a body, each with the region depth it sits at, outermost first
/// and in order.
pub(super) fn steps(f: &Function, ops: &[OpId]) -> Vec<(usize, OpId)> {
    let mut out = Vec::new();
    fn walk(f: &Function, ops: &[OpId], depth: usize, out: &mut Vec<(usize, OpId)>) {
        for &op in ops {
            out.push((depth, op));
            for &region in &f.op(op).regions {
                for &block in &f.region(region).blocks {
                    walk(f, &f.block(block).ops, depth + 1, out);
                }
            }
        }
    }
    walk(f, ops, 0, &mut out);
    out
}

/// The steps of a body, without their depth.
pub(super) fn flat(f: &Function, ops: &[OpId]) -> Vec<OpId> {
    steps(f, ops).into_iter().map(|(_, op)| op).collect()
}

/// The operand operations one value reads, itself first, each once.
fn cone(f: &Function, value: ValueId, out: &mut Vec<OpId>) {
    let Some(op) = f.operand_op(value) else {
        return;
    };
    if out.contains(&op) {
        return;
    }
    out.push(op);
    for index in 0..f.op(op).operands.len() {
        cone(f, f.op(op).operands[index], out);
    }
    for index in 0..f.op(op).regions.len() {
        let region = f.op(op).regions[index];
        for slot in 0..f.region(region).blocks.len() {
            let block = f.region(region).blocks[slot];
            for step in flat(f, &f.block(block).ops) {
                for operand in 0..f.op(step).operands.len() {
                    cone(f, f.op(step).operands[operand], out);
                }
            }
            for yielded in 0..f.block(block).terminator.yields.len() {
                cone(f, f.block(block).terminator.yields[yielded], out);
            }
        }
    }
}

/// Every operation a value's operand cone holds, in the order it is read.
pub(super) fn reads(f: &Function, value: ValueId) -> Vec<OpId> {
    let mut out = Vec::new();
    cone(f, value, &mut out);
    out
}

/// Every value one step reads directly: its operands, the values its regions
/// yield, and the operands of the steps inside them.
pub(super) fn step_operands(f: &Function, op: OpId) -> Vec<ValueId> {
    let mut out = f.op(op).operands.clone();
    for &region in &f.op(op).regions {
        for &block in &f.region(region).blocks {
            for inner in flat(f, &f.block(block).ops) {
                out.extend(f.op(inner).operands.iter().copied());
            }
            out.extend(f.block(block).terminator.yields.iter().copied());
        }
    }
    out
}

/// Every value the whole body reads, in order.
fn body_operands(f: &Function, ops: &[OpId]) -> Vec<ValueId> {
    ops.iter().flat_map(|&op| step_operands(f, op)).collect()
}

pub(super) fn is_kind(f: &Function, value: ValueId, wanted: impl Fn(&OpKind) -> bool) -> bool {
    f.operand_op(value).is_some_and(|op| wanted(&f.op(op).kind))
}

/// Whether a value reads a materialized workspace this kernel takes as a
/// parameter.
pub(super) fn value_loads_materialized(f: &Function, value: ValueId) -> bool {
    reads(f, value).into_iter().any(|op| {
        matches!(f.op(op).kind, OpKind::Load { .. })
            && f.reads(f.op(op).operands[0]).into_iter().any(|read| {
                f.name(read)
                    .is_some_and(|name| name.starts_with("loom_material_"))
            })
    })
}

pub(super) fn body_loads_materialized_in(f: &Function, ops: &[OpId]) -> bool {
    body_operands(f, ops)
        .into_iter()
        .any(|value| value_loads_materialized(f, value))
}

/// How many distinct constants of this value the body holds.
///
/// A constant is one operation however many steps read it, so this counts
/// distinct operations rather than reads or structural equality.
pub(super) fn body_count_float_constant_in(f: &Function, ops: &[OpId], expected: f64) -> usize {
    let mut seen = Vec::new();
    for value in body_operands(f, ops) {
        for op in reads(f, value) {
            if !seen.contains(&op)
                && matches!(
                    f.op(op).kind,
                    OpKind::FloatConst { value, .. } if value.to_bits() == expected.to_bits()
                )
            {
                seen.push(op);
            }
        }
    }
    seen.len()
}

fn is_warp_guard(f: &Function, op: OpId) -> bool {
    matches!(f.op(op).kind, OpKind::If)
        && is_kind(f, f.op(op).operands[0], |kind| {
            matches!(kind, OpKind::WarpAny)
        })
}

pub(super) fn warp_guard_loop_depth_in(f: &Function, ops: &[OpId], depth: usize) -> Option<usize> {
    fn walk(f: &Function, ops: &[OpId], depth: usize) -> Option<usize> {
        for &op in ops {
            if is_warp_guard(f, op) {
                return Some(depth);
            }
            let inner = if matches!(f.op(op).kind, OpKind::Loop { .. }) {
                depth + 1
            } else {
                depth
            };
            for &region in &f.op(op).regions {
                for &block in &f.region(region).blocks {
                    if let Some(found) = walk(f, &f.block(block).ops, inner) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }
    walk(f, ops, depth)
}

pub(super) fn warp_guard_count_in(f: &Function, ops: &[OpId]) -> usize {
    flat(f, ops)
        .into_iter()
        .filter(|&op| is_warp_guard(f, op))
        .count()
}

pub(super) fn has_inactive_direct_zero_store_in(f: &Function, ops: &[OpId]) -> bool {
    flat(f, ops).into_iter().any(|op| {
        if !is_warp_guard(f, op) {
            return false;
        }
        let else_ = f.region_block(f.op(op).regions[1]);
        f.block(else_).ops.iter().any(|&inner| {
            matches!(f.op(inner).kind, OpKind::Write { .. })
                && is_kind(
                    f,
                    f.op(inner).operands[1],
                    |kind| matches!(kind, OpKind::FloatConst { value, .. } if *value == 0.0),
                )
        })
    })
}

fn load_count_over(f: &Function, values: impl IntoIterator<Item = ValueId>) -> usize {
    let mut seen = Vec::new();
    for value in values {
        for op in reads(f, value) {
            if !seen.contains(&op)
                && matches!(f.op(op).kind, OpKind::Load { .. } | OpKind::ConstLoad(_))
            {
                seen.push(op);
            }
        }
    }
    seen.len()
}

pub(super) fn body_load_count_in(f: &Function, ops: &[OpId]) -> usize {
    load_count_over(f, body_operands(f, ops))
}

fn region_load_count(f: &Function, block: BlockId) -> usize {
    let mut values = body_operands(f, &f.block(block).ops);
    values.extend(f.block(block).terminator.yields.iter().copied());
    load_count_over(f, values)
}

pub(super) fn guarded_then_load_count_in(f: &Function, ops: &[OpId]) -> Option<usize> {
    flat(f, ops)
        .into_iter()
        .find(|&op| is_warp_guard(f, op))
        .map(|op| region_load_count(f, f.region_block(f.op(op).regions[0])))
}

/// The reads a body performs outside any warp guard: a guarded region protects
/// the reads inside it, so only the guard's own reads are unprotected.
pub(super) fn unprotected_load_count_in(f: &Function, ops: &[OpId]) -> usize {
    let mut values = Vec::new();
    fn walk(f: &Function, ops: &[OpId], values: &mut Vec<ValueId>) {
        for &op in ops {
            values.extend(f.op(op).operands.iter().copied());
            if is_warp_guard(f, op) {
                continue;
            }
            for &region in &f.op(op).regions {
                for &block in &f.region(region).blocks {
                    walk(f, &f.block(block).ops, values);
                    values.extend(f.block(block).terminator.yields.iter().copied());
                }
            }
        }
    }
    walk(f, ops, &mut values);
    load_count_over(f, values)
}

/// Whether the skipped path of a warp guard runs nothing and hands every
/// accumulator back exactly as it found it — which is what makes the guard cost
/// nothing where it skips.
pub(super) fn guarded_else_is_empty_in(f: &Function, ops: &[OpId]) -> bool {
    flat(f, ops).into_iter().any(|op| {
        if !is_warp_guard(f, op) {
            return false;
        }
        let else_ = f.region_block(f.op(op).regions[1]);
        let handed_back = &f.block(else_).terminator.yields;
        f.block(else_).ops.is_empty()
            && handed_back.len() == f.op(op).into.len()
            && handed_back
                .iter()
                .zip(&f.op(op).into)
                .all(|(&yielded, place)| *place == Some(yielded))
    })
}

pub(super) fn has_zero_initialized_output_accumulator_in(f: &Function, ops: &[OpId]) -> bool {
    ops.iter().any(|&op| {
        matches!(f.op(op).kind, OpKind::Bind)
            && f.name(f.op(op).results[0])
                .is_some_and(|name| name.starts_with("loom_out_acc_"))
            && is_kind(
                f,
                f.op(op).operands[0],
                |kind| matches!(kind, OpKind::FloatConst { value, .. } if *value == 0.0),
            )
    })
}

/// Whether a loop is the hidden-axis trip: it carries the launch's accumulators
/// and steps over the hidden extent, which is what makes its index `loom_u`.
fn is_hidden_axis_loop(f: &Function, op: OpId) -> bool {
    matches!(f.op(op).kind, OpKind::Loop { .. })
        && f.name(f.block(f.region_block(f.op(op).regions[0])).params[0])
            .is_some_and(|name| name.contains("loom_u"))
}

/// How many hidden-axis loops a body holds, however deeply.
pub(super) fn hidden_axis_loops_in(f: &Function, ops: &[OpId]) -> usize {
    flat(f, ops)
        .into_iter()
        .filter(|&op| is_hidden_axis_loop(f, op))
        .count()
}

pub(super) fn u_loop(f: &Function, ops: &[OpId]) -> OpId {
    ops.iter()
        .copied()
        .find(|&op| is_hidden_axis_loop(f, op))
        .expect("kernel has no U loop")
}

pub(super) fn u_loop_body_in(f: &Function, ops: &[OpId]) -> Vec<OpId> {
    let loop_ = u_loop(f, ops);
    f.block_steps(f.region_block(f.op(loop_).regions[0]))
        .collect()
}

pub(super) fn u_loop_prefix_in(f: &Function, ops: &[OpId]) -> Vec<OpId> {
    let index = ops
        .iter()
        .position(|&op| is_hidden_axis_loop(f, op))
        .expect("kernel has no U loop");
    ops[..index].to_vec()
}

fn binding_named(f: &Function, op: OpId, prefix: &str) -> bool {
    matches!(f.op(op).kind, OpKind::Bind)
        && f.name(f.op(op).results[0])
            .is_some_and(|name| name.starts_with(prefix))
}

pub(super) fn loom_value_bindings_in(f: &Function, ops: &[OpId]) -> usize {
    ops.iter()
        .filter(|&&op| binding_named(f, op, "loom_v"))
        .count()
}

pub(super) fn has_let_named_in(f: &Function, ops: &[OpId], prefix: &str) -> bool {
    ops.iter().any(|&op| binding_named(f, op, prefix))
}

pub(super) fn let_count_named_in(f: &Function, ops: &[OpId], prefix: &str) -> usize {
    ops.iter()
        .filter(|&&op| binding_named(f, op, prefix))
        .count()
}

pub(super) fn has_red_add_in(f: &Function, ops: &[OpId]) -> bool {
    flat(f, ops).into_iter().any(|op| {
        matches!(
            f.op(op).kind,
            OpKind::Write {
                mode: WriteMode::Add,
                ..
            }
        )
    })
}

pub(super) fn ptx_u_loop(ptx: &str) -> &str {
    let loop_start = ptx.find("$Lloop_").expect("PTX has no U loop");
    let loop_ptx = &ptx[loop_start..];
    let exit_target_start = loop_ptx
        .find("bra $Ldone_")
        .expect("PTX U loop has no exit branch")
        + "bra ".len();
    let exit_target_end = exit_target_start
        + loop_ptx[exit_target_start..]
            .find(';')
            .expect("PTX U-loop exit branch has no terminator");
    let exit_target = &loop_ptx[exit_target_start..exit_target_end];
    let loop_end = loop_start
        + loop_ptx
            .find(&format!("{exit_target}:"))
            .expect("PTX U loop has no exit label");
    &ptx[loop_start..loop_end]
}

pub(super) fn assert_target_indices_load_is_outside_ptx_u_loop(kernel: &gpu_ir::Launch, ptx: &str) {
    assert_eq!(
        kernel
            .function
            .parameters()
            .iter()
            .filter(|param| matches!(param.ty, gpu_ir::ParamTy::Ptr(Scalar::S32)))
            .count(),
        1
    );
    let loop_start = ptx.find("$Lloop_").expect("PTX has no U loop");
    let loop_ptx = ptx_u_loop(ptx);
    assert!(ptx[..loop_start].contains("ld.global.b32"));
    assert!(!loop_ptx.contains("ld.global.b32"));
    assert!(loop_ptx.contains("red.global.add.f32"));
}

/// Whether the body writes at `position * u + lane`, the address a launch that
/// materializes a value writes its workspace at.
pub(super) fn has_position_store_in(f: &Function, ops: &[OpId], position: ValueId, u: i64) -> bool {
    flat(f, ops).into_iter().any(|op| {
        if !matches!(f.op(op).kind, OpKind::Write { .. }) {
            return false;
        }
        let Some(offset) = f.operand_op(f.op(op).operands[0]) else {
            return false;
        };
        if !matches!(f.op(offset).kind, OpKind::Offset) {
            return false;
        }
        let Some(sum) = f.operand_op(f.op(offset).operands[1]) else {
            return false;
        };
        if !matches!(f.op(sum).kind, OpKind::Bin(BinOp::Add)) {
            return false;
        }
        let Some(scaled) = f.operand_op(f.op(sum).operands[0]) else {
            return false;
        };
        matches!(f.op(scaled).kind, OpKind::Bin(BinOp::Mul))
            && f.op(scaled).operands[0] == position
            && is_kind(f, f.op(scaled).operands[1], |kind| {
                matches!(
                    kind,
                    OpKind::IntConst {
                        scalar: Scalar::S64,
                        value
                    } if *value == u
                )
            })
    })
}

/// The steps a body runs, for a question about the whole of it.
pub(super) fn body(f: &Function) -> Vec<OpId> {
    f.block_steps(f.entry()).collect()
}

pub(super) fn body_loads_materialized(f: &Function) -> bool {
    body_loads_materialized_in(f, &body(f))
}

pub(super) fn warp_guard_count(f: &Function) -> usize {
    warp_guard_count_in(f, &body(f))
}

pub(super) fn has_inactive_direct_zero_store(f: &Function) -> bool {
    has_inactive_direct_zero_store_in(f, &body(f))
}

pub(super) fn guarded_else_is_empty(f: &Function) -> bool {
    guarded_else_is_empty_in(f, &body(f))
}

pub(super) fn has_zero_initialized_output_accumulator(f: &Function) -> bool {
    has_zero_initialized_output_accumulator_in(f, &body(f))
}

pub(super) fn hidden_axis_loops(f: &Function) -> usize {
    hidden_axis_loops_in(f, &body(f))
}

pub(super) fn unprotected_load_count(f: &Function) -> usize {
    unprotected_load_count_in(f, &body(f))
}

pub(super) fn loom_value_bindings(f: &Function) -> usize {
    loom_value_bindings_in(f, &body(f))
}

pub(super) fn guarded_then_load_count(f: &Function) -> Option<usize> {
    guarded_then_load_count_in(f, &body(f))
}

pub(super) fn u_loop_body(f: &Function) -> Vec<OpId> {
    u_loop_body_in(f, &body(f))
}

pub(super) fn u_loop_prefix(f: &Function) -> Vec<OpId> {
    u_loop_prefix_in(f, &body(f))
}

pub(super) fn body_count_float_constant(f: &Function, expected: f64) -> usize {
    body_count_float_constant_in(f, &body(f), expected)
}

pub(super) fn has_position_store(f: &Function, position: ValueId, u: i64) -> bool {
    has_position_store_in(f, &body(f), position, u)
}

pub(super) fn warp_guard_loop_depth(f: &Function, depth: usize) -> Option<usize> {
    warp_guard_loop_depth_in(f, &body(f), depth)
}
