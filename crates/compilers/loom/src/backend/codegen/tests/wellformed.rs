// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What every kernel Loom emits must be true of, asked one question at a time.
//!
//! Verification answers "is this a program" all at once, and its first refusal
//! is whichever rule it happens to reach first. That is too coarse to work
//! from: an operation belonging to no block makes every ordering question
//! moot, and an ordering fault makes a typing complaint noise.
//!
//! So these ask the cheap questions in the order they become meaningful —
//! stranded, then misordered, then the whole of verification — over the
//! smallest programs the emitter accepts. The first one that fails names the
//! stage that broke, and the ones after it stay quiet until it is fixed.

use super::super::super::iteration::Plan;
use super::super::super::partition::{DEFAULT_MAX_KERNEL_STATEMENTS, plan};
use super::super::{Flavor, emit_consumer_kernel_flavor};
use super::support::{gathered_input_program, multi_warp_program, sender_scatter_program};
use crate::program::LiftedProgram;

const BLOCK: u32 = 32;

/// Every program below, in the order they grow: one position, then many warps,
/// then a scatter that opens a region.
fn programs() -> Vec<(&'static str, LiftedProgram)> {
    vec![
        ("multi_warp", multi_warp_program()),
        ("gathered_input", gathered_input_program()),
        ("sender_scatter", sender_scatter_program(8, 8)),
    ]
}

fn emitted(program: &LiftedProgram) -> gpu_ir::Function {
    let prepared = Plan::prepare(program).expect("the program prepares");
    let parts = plan(program, DEFAULT_MAX_KERNEL_STATEMENTS).expect("it partitions");
    let kernel =
        emit_consumer_kernel_flavor(&prepared, program, &parts[0], Flavor::plain(BLOCK, 0, 1))
            .expect("it emits");
    kernel
        .kernel
        .function
        .body()
        .expect("a kernel is a body")
        .clone()
}

/// The first question: every operation belongs to a block. Nothing else is
/// worth asking of a body that fails this.
#[test]
fn every_emitted_operation_belongs_to_a_block() {
    for (name, program) in programs() {
        let body = emitted(&program);
        let stranded = body
            .stranded()
            .into_iter()
            .map(|op| body.op(op).kind.clone())
            .collect::<Vec<_>>();
        assert!(
            stranded.is_empty(),
            "{name}: {stranded:?} belong to no block"
        );
    }
}

/// The second: within a block, an operand stands before every operation that
/// reads it, because a value is readable after the operation defining it.
#[test]
fn every_emitted_operand_stands_before_its_reader() {
    for (name, program) in programs() {
        let body = emitted(&program);
        let misordered = body
            .misordered()
            .into_iter()
            .map(|(block, op)| (block, body.op(op).kind.clone()))
            .collect::<Vec<_>>();
        assert!(
            misordered.is_empty(),
            "{name}: these read something standing after them: {misordered:?}"
        );
    }
}

/// And then the whole of it, which is only meaningful once the two above hold.
#[test]
fn every_emitted_kernel_verifies() {
    for (name, program) in programs() {
        let prepared = Plan::prepare(&program).expect("the program prepares");
        let parts = plan(&program, DEFAULT_MAX_KERNEL_STATEMENTS).expect("it partitions");
        let kernel =
            emit_consumer_kernel_flavor(&prepared, &program, &parts[0], Flavor::plain(BLOCK, 0, 1))
                .expect("it emits");
        if let Err(why) = gpu_ir::verify::verify_launch(&kernel.kernel) {
            panic!("{name}: {why}");
        }
    }
}
