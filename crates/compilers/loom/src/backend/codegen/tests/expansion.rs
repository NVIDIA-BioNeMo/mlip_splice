// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where a kernel body is cut in two: the addresses a kernel binds, and the
//! arithmetic that reads them off their names.
//!
//! The expansion is cached under the lane width alone, so what it may not do is
//! say anything about where it is emitted. These read the two halves back and
//! check that the cut is where the key says it is.

use super::*;
use crate::backend::partition::statement_count;
use crate::domain::{BucketMaskValues, EmitCtx, ParameterValues};

/// The names a flat run of statements binds, in order.
///
/// Only for the two halves of the cut, which are flat: [`build_input_access`] binds
/// one let per address, and [`expand_program_for_lanes`] hands back the flat
/// lets the slice consumes. A body that has been placed is not flat, and is
/// read with [`bound_anywhere`].
fn bound(f: &gpu_ir::Function, ops: &[gpu_ir::OpId]) -> Vec<String> {
    for &op in ops {
        assert!(
            matches!(f.op(op).kind, gpu_ir::OpKind::Bind),
            "{:?} is not a binding, so this run is not flat",
            f.op(op).kind
        );
    }
    ops.iter()
        .filter_map(|&op| f.name(f.op(op).results[0]).map(str::to_owned))
        .collect()
}

/// Every name a body binds, however deep — an emitted kernel nests its loop and
/// its guards, so its bindings are not all at the top.
fn bound_anywhere(f: &gpu_ir::Function, ops: &[gpu_ir::OpId]) -> Vec<String> {
    let mut into = Vec::new();
    fn walk(f: &gpu_ir::Function, ops: &[gpu_ir::OpId], into: &mut Vec<String>) {
        for &op in ops {
            for &result in &f.op(op).results {
                if let Some(name) = f.name(result) {
                    into.push(name.to_string());
                }
            }
            for &region in &f.op(op).regions {
                for &block in &f.region(region).blocks {
                    for &param in &f.block(block).params {
                        if let Some(name) = f.name(param) {
                            into.push(name.to_string());
                        }
                    }
                    walk(f, &f.block(block).ops, into);
                }
            }
        }
    }
    walk(f, ops, &mut into);
    into
}

/// A fiber program whose second input bridges through a aligned view, so an
/// emitter that pre-bound the view element and one that did not address it
/// differently. Only the promoted placement walks the fiber, so only it bridges.
fn viewed() -> (LiftedProgram, Plan) {
    let loom = sender_scatter_program(8, 8);
    let program = Plan::prepare_fiber_direct(&loom).unwrap().unwrap();
    (loom, program)
}

/// A context of the shape an emitter builds, with the two things that make one
/// emission differ from another: whether the fiber's view element was pre-bound,
/// and the index the fiber is on.
struct Context {
    declared: ParameterValues,
    masks: BucketMaskValues,
    axes: Vec<gpu_ir::Value>,
    elements: Vec<gpu_ir::Value>,
    fiber: gpu_ir::Value,
}

fn context(body: &mut gpu_ir::Builder, program: &Plan, pre_bind_element: bool) -> Context {
    let declared = ParameterValues::declare(body, &program.layout, program.layout.params.len());
    let axes = (0..program.index.num_axes())
        .map(|_| body.i64(0))
        .collect::<Vec<_>>();
    let fiber = body.i32(0);
    let elements = if pre_bind_element {
        program
            .layout
            .used_views
            .iter()
            .map(|_| body.i32(0))
            .collect()
    } else {
        Vec::new()
    };
    Context {
        declared,
        masks: BucketMaskValues::none(),
        axes,
        elements,
        fiber,
    }
}

impl Context {
    fn emit_ctx<'a>(&'a self, program: &'a Plan) -> EmitCtx<'a, Index> {
        EmitCtx {
            index: &program.index,
            layout: &program.layout,
            declared: &self.declared,
            bucket_masks: &self.masks,
            axis_indices: &self.axes,
            view_elements: &self.elements,
            fiber_index: Some(self.fiber.clone()),
        }
    }
}

/// The cut itself: the expansion reads the operands by name and binds none of
/// those names, because binding them is what the emitting kernel does.
///
/// The second half is what keeps this from passing vacuously — an expansion
/// that mentioned no operand at all would bind no operand name either.
#[test]
fn the_cached_expansion_binds_no_address_and_reads_every_one() {
    let (loom, program) = viewed();
    let expansion = expand_program_for_lanes(&program, &loom, 1).unwrap();

    let f = expansion.body.function();
    let bound = bound(f, &expansion.body.steps());
    assert!(
        bound.iter().all(|name| !name.starts_with("loom_input_")),
        "the expansion binds an address: {bound:?}"
    );
    // An address is a parameter of the expansion's own arena, so what it reads
    // is what it declares — and declaring one is the opposite of binding it.
    let read = f
        .parameters()
        .iter()
        .map(|param| param.name.clone())
        .collect::<Vec<_>>();
    for operand in 0..loom.num_inputs {
        assert!(
            read.contains(&format!("loom_input_base_{operand}"))
                || read.contains(&format!("loom_input_lane_base_{operand}")),
            "the expansion never reads input {operand}"
        );
    }
}

/// Two emissions of one program address their inputs differently — one reads a
/// view element the emitter pre-bound, the other gathers it — and both are
/// served the same expansion, off the one key the lane width makes.
///
/// This is the property the key rests on. An expansion that still knew where it
/// was emitted would have to be either wrong for one of these or absent from the
/// cache for the other.
#[test]
fn one_expansion_serves_emissions_whose_addresses_differ() {
    let (loom, program) = viewed();
    assert!(
        !program.layout.used_views.is_empty(),
        "the fixture bridges through a view"
    );

    let access = |pre_bind_element| {
        let mut body = gpu_ir::Builder::new();
        let context = context(&mut body, &program, pre_bind_element);
        let ctx = context.emit_ctx(&program);
        let u = body.i32(0);
        let edge = context.fiber.clone();
        let (ops, _) =
            build_input_access(&mut body, &program, &loom, &ctx, &u, Some(&edge)).unwrap();
        let f = body.into_function();
        let bound = bound(&f, &ops);
        let shape = ops
            .iter()
            .map(|&op| format!("{:?}", f.op(op).kind))
            .collect::<Vec<_>>()
            .join(";");
        // How the address was reached is what differs, so the count of the
        // operations it took is what says the two emissions are not the same.
        let reached = ops
            .iter()
            .map(|&op| f.reads(f.op(op).operands[0]).len())
            .collect::<Vec<_>>();
        (bound, shape, reached)
    };
    let gathered = access(false);
    let pre_bound = access(true);

    assert_eq!(
        gathered.0, pre_bound.0,
        "both emissions bind the same addresses"
    );
    assert_ne!(
        (gathered.1.clone(), gathered.2.clone()),
        (pre_bound.1.clone(), pre_bound.2.clone()),
        "the two emissions address their inputs alike, so this proves nothing"
    );

    let first = expand_program_for_lanes(&program, &loom, 1).unwrap();
    let second = expand_program_for_lanes(&program, &loom, 1).unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "the lane width alone keys the cache, so both emissions get one expansion"
    );
    let other_width = expand_program_for_lanes(&program, &loom, 2).unwrap();
    assert!(
        !std::sync::Arc::ptr_eq(&first, &other_width),
        "the lane width is a key, so a second width is a second expansion"
    );
}

/// The access binding names one family per operand, and which members it binds is
/// decided by the operand alone: the lane fold when the operand carries lanes,
/// the mask when its bridges leave cells unmapped.
///
/// The expansion reads these names without seeing the operand, so this is the
/// interface between the halves, pinned.
#[test]
fn the_access_binding_names_the_families_the_expansion_reads() {
    let (loom, program) = viewed();
    let mut body = gpu_ir::Builder::new();
    let context = context(&mut body, &program, true);
    let ctx = context.emit_ctx(&program);
    let u = body.i32(0);
    let edge = context.fiber.clone();
    let (access, _) =
        build_input_access(&mut body, &program, &loom, &ctx, &u, Some(&edge)).unwrap();
    let f = body.into_function();

    let mut expected = Vec::new();
    for (operand, binding) in loom.operands[..loom.num_inputs].iter().enumerate() {
        expected.push(format!("loom_input_base_{operand}"));
        if binding.lane_extent != 1 {
            expected.push(format!("loom_input_lane_base_{operand}"));
        }
        if domain_index::operand_has_alive(&program.index, operand) {
            expected.push(format!("loom_input_alive_{operand}"));
        }
    }
    assert_eq!(bound(&f, &access), expected);
    // Every input here carries lanes and none is masked, so the fixture does
    // exercise the lane fold and says what the mask's absence looks like.
    assert!(expected.iter().any(|name| name.contains("lane_base")));
    assert!(!expected.iter().any(|name| name.contains("alive")));
}

/// The two halves are one stream when the slice runs, so what the slice keeps
/// is what it kept when they were built together: a body that reaches an
/// address binds the address.
///
/// Read off the names the body actually binds rather than off its rendering — a
/// substring search would be answered by any mention, including the read that
/// the missing binding would have left dangling.
#[test]
fn a_sliced_body_binds_the_addresses_its_arithmetic_reaches() {
    let (loom, program) = viewed();
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_kernel(&program, &loom, BLOCK_SIZES[0], 0, parts.len(), &parts[0]).unwrap();

    let names = bound_anywhere(kernel.body(), &steps_of(kernel.body()));
    for (operand, binding) in loom.operands[..loom.num_inputs].iter().enumerate() {
        let required = if binding.lane_extent == 1 {
            format!("loom_input_base_{operand}")
        } else {
            format!("loom_input_lane_base_{operand}")
        };
        assert!(
            names.contains(&required),
            "the emitted body never binds {required}, which its arithmetic reads: {names:?}"
        );
    }
}

/// The cut carries the liveness predicate the same way it carries an address:
/// the access binding makes it, because deciding it needs the context; the expansion
/// reads it, because selecting on it does not.
///
/// The access binding is what decides whether there is a predicate at all, so
/// a fixture with no masked input would pass every assertion here vacuously.
/// This one gathers its input through an index map that leaves cells unmapped.
#[test]
fn a_masked_input_is_cut_the_same_way_an_address_is() {
    let loom = gathered_input_program();
    let program = Plan::prepare(&loom).unwrap();
    assert!(
        domain_index::operand_has_alive(&program.index, 0),
        "the fixture masks its input"
    );

    let mut body = gpu_ir::Builder::new();
    let context = context(&mut body, &program, false);
    let ctx = context.emit_ctx(&program);
    let u = body.i32(0);
    let (access, _) = build_input_access(&mut body, &program, &loom, &ctx, &u, None).unwrap();
    let f = body.into_function();
    let made = bound(&f, &access);
    assert!(
        made.iter().any(|name| name == "loom_input_alive_0"),
        "the access binding makes the predicate: {made:?}"
    );

    let expansion = expand_program_for_lanes(&program, &loom, 1).unwrap();
    let held = expansion.body.function();
    assert!(
        !bound(held, &expansion.body.steps())
            .iter()
            .any(|name| name == "loom_input_alive_0"),
        "the expansion binds the predicate rather than reading it"
    );
    assert!(
        held.parameters()
            .iter()
            .any(|param| param.name == "loom_input_alive_0"),
        "the expansion never reads the predicate the access binding made"
    );

    // And the two meet in an emitted kernel: one binding, reached by the slice.
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let kernel = emit_kernel(&program, &loom, BLOCK_SIZES[0], 0, parts.len(), &parts[0]).unwrap();
    let names = bound_anywhere(kernel.body(), &steps_of(kernel.body()));
    assert_eq!(
        names.iter().filter(|n| *n == "loom_input_alive_0").count(),
        1,
        "the emitted body binds the predicate exactly once: {names:?}"
    );
}

/// The count a candidate is chosen by ([`candidate_statement_counts`],
/// [`lane_hoist_statement_counts`]). Both halves feed it, so a cut that
/// duplicated or dropped a binding would move it — which is a different kernel
/// being picked, not a cosmetic difference.
///
/// One count per placement, because the two address their inputs differently:
/// only the promoted one walks the fiber and bridges through the view.
///
/// [`candidate_statement_counts`]: super::super::super::compiler::candidate_statement_counts
/// [`lane_hoist_statement_counts`]: super::super::super::compiler::lane_hoist_statement_counts
#[test]
fn the_emitted_statement_count_is_what_candidate_selection_reads() {
    let loom = sender_scatter_program(8, 8);
    let parts = plan(&loom, DEFAULT_MAX_KERNEL_STATEMENTS).unwrap();
    let counts = [
        Plan::prepare(&loom).unwrap(),
        Plan::prepare_fiber_direct(&loom).unwrap().unwrap(),
    ]
    .map(|program| {
        let kernel =
            emit_kernel(&program, &loom, BLOCK_SIZES[0], 0, parts.len(), &parts[0]).unwrap();
        statement_count(kernel.body())
    });
    assert_eq!(counts, [15, 20]);
}
