// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership, what a region hands back, and where a value may be read.

use super::*;
use crate::Site;

/// Naming an accumulator does not make a fold escape the region that made it:
/// the storage outlives the region, but whether the region ran does not, so a
/// value that stands in it is still the region's.
#[test]
fn a_fold_named_into_an_outer_place_still_does_not_escape_its_region() {
    let mut raw = raw();
    let zero = raw.float(0.0);
    let start = raw.place(
        OpKind::Bind,
        vec![zero],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let accumulator = raw.f.op(start).results[0];
    let one = raw.float(1.0);
    let sum = raw.operand(
        OpKind::Bin(BinOp::Add),
        vec![accumulator, one],
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    // A fold produced into the accumulator, inside a region of its own.
    let fold = raw.f.add_op(
        OpKind::Bind,
        true,
        vec![sum],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
        vec![Some(accumulator)],
        Vec::new(),
    );
    let folded = raw.f.op(fold).results[0];
    let inner = raw.f.add_block();
    raw.f.set_block_ops(inner, vec![fold]);
    raw.f.set_terminator(inner, Vec::new());
    let region = raw.f.add_region(vec![inner]);
    let scope = raw.f.add_op(
        OpKind::Scope,
        true,
        Vec::new(),
        &[],
        Vec::new(),
        vec![region],
    );
    let index = raw.int(0);
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, index],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, folded],
        &[],
    );
    assert_eq!(
        refuses(raw.body(vec![start, scope, write])).rule,
        Rule::Dominance
    );
}

/// The same fold, handed back through the region's own result: that is what a
/// value leaving a region is, and it reads cleanly afterwards.
#[test]
fn a_fold_a_guard_yields_reads_after_the_guard() {
    let mut b = Builder::new();
    let out = b.parameter(Param {
        name: "out".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let count = b.parameter(Param {
        name: "count".into(),
        ty: ParamTy::Scalar(Scalar::S32),
    });
    let zero = b.float(Scalar::F32, 0.0);
    let accumulator = b.bind_value_as("acc", Ty::scalar(Scalar::F32), zero);
    let tid = b.tid_x();
    let tid = b.cast(Scalar::S32, tid);
    let inside = b.lt(tid.clone(), count);
    let held = accumulator.clone();
    let folded = b
        .if_producing(
            inside,
            &[(Ty::scalar(Scalar::F32), Some(accumulator.clone()))],
            |arm| {
                let one = arm.float(Scalar::F32, 1.0);
                let sum = arm.add(held.clone(), one);
                vec![arm.update(held.clone(), sum)]
            },
            |_| vec![accumulator.clone()],
        )
        .into_iter()
        .next()
        .expect("one result");
    let at = b.offset(out, tid);
    b.store(at, folded);
    accepts(b.into_function());
}

/// A binding a region made is that region's own storage; outside it there is
/// nothing holding it. The builder's scopes refuse this where a caller writes
/// it, so what is proven here is that the rule holds of the arena too.
#[test]
fn reading_what_a_region_bound_from_outside_it_is_refused() {
    let mut raw = raw();
    let one = raw.float(1.0);
    let bound = raw.place(
        OpKind::Bind,
        vec![one],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let held = raw.f.op(bound).results[0];
    let inner = raw.f.add_block();
    raw.f.set_block_ops(inner, vec![bound]);
    raw.f.set_terminator(inner, Vec::new());
    let region = raw.f.add_region(vec![inner]);
    let scope = raw.f.add_op(
        OpKind::Scope,
        true,
        Vec::new(),
        &[],
        Vec::new(),
        vec![region],
    );
    let zero = raw.int(0);
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, zero],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, held],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![scope, write])).rule, Rule::Dominance);
}

/// An arm that yields the wrong number of values leaves a result of the `if`
/// with nothing to hold.
#[test]
fn an_arm_that_yields_the_wrong_arity_is_refused() {
    let mut raw = raw();
    let then_block = raw.f.add_block();
    let one = raw.float(1.0);
    raw.f.set_terminator(then_block, vec![one]);
    let then_region = raw.f.add_region(vec![then_block]);
    let else_block = raw.f.add_block();
    raw.f.set_terminator(else_block, Vec::new());
    let else_region = raw.f.add_region(vec![else_block]);
    let zero = raw.int(0);
    let pred = raw.operand(
        OpKind::Cmp(CmpOp::Lt),
        vec![zero, raw.count],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let conditional = raw.f.add_op(
        OpKind::If,
        true,
        vec![pred],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
        Vec::new(),
        vec![then_region, else_region],
    );
    assert_eq!(refuses(raw.body(vec![conditional])).rule, Rule::Region);
}

/// A step standing twice would define its values twice, and what they hold the
/// second time need not be what they held the first: the same read, either side
/// of a write to the cell it reads, is two values.
#[test]
fn a_step_standing_twice_in_one_block_is_refused() {
    let mut raw = raw();
    let zero = raw.int(0);
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, zero],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let read = raw.operand(
        OpKind::Load {
            lanes: 1,
            nc: false,
            evict: Evict::Normal,
        },
        vec![at],
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    let bind = raw.place(
        OpKind::Bind,
        vec![read],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let one = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, one],
        &[],
    );
    // Read, write, read again — through the one binding, which would hold the
    // cell before the write and after it.
    let refused = refuses(raw.body(vec![bind, write, bind]));
    assert_eq!(refused.rule, Rule::Ownership);
    assert!(refused.detail.contains("defined twice"), "{refused}");
}

/// Reading the same cell twice is two definitions, and saying so is all it
/// takes: each holds what its own read found.
#[test]
fn two_definitions_of_one_expression_are_accepted() {
    let mut raw = raw();
    let zero = raw.int(0);
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, zero],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let read = || OpKind::Load {
        lanes: 1,
        nc: false,
        evict: Evict::Normal,
    };
    let before = raw.operand(read(), vec![at], ValTy::Data(Ty::scalar(Scalar::F32)));
    let first = raw.place(
        OpKind::Bind,
        vec![before],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let one = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, one],
        &[],
    );
    let after = raw.operand(read(), vec![at], ValTy::Data(Ty::scalar(Scalar::F32)));
    let second = raw.place(
        OpKind::Bind,
        vec![after],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let held = raw.f.op(second).results[0];
    let out = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, held],
        &[],
    );
    accepts(raw.body(vec![first, write, second, out]));
}

/// One definition, one path: a step in two blocks would be defined on a path
/// that need not run.
#[test]
fn a_step_of_two_blocks_is_refused() {
    let mut raw = raw();
    let one = raw.float(1.0);
    let bound = raw.place(
        OpKind::Bind,
        vec![one],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
    );
    let inner = raw.f.add_block();
    raw.f.set_block_ops(inner, vec![bound]);
    raw.f.set_terminator(inner, Vec::new());
    let region = raw.f.add_region(vec![inner]);
    let scope = raw.f.add_op(
        OpKind::Scope,
        true,
        Vec::new(),
        &[],
        Vec::new(),
        vec![region],
    );
    assert_eq!(refuses(raw.body(vec![bound, scope])).rule, Rule::Ownership);
}

/// A guard is a step. One standing for a value would have no place to run, and
/// the renderer has nowhere to put it.
#[test]
fn an_if_that_is_an_operand_is_refused() {
    let mut raw = raw();
    let then_block = raw.f.add_block();
    let one = raw.float(1.0);
    raw.f.set_terminator(then_block, vec![one]);
    let then_region = raw.f.add_region(vec![then_block]);
    let else_block = raw.f.add_block();
    let other = raw.float(0.0);
    raw.f.set_terminator(else_block, vec![other]);
    let else_region = raw.f.add_region(vec![else_block]);
    let zero = raw.int(0);
    let pred = raw.operand(
        OpKind::Cmp(CmpOp::Lt),
        vec![zero, raw.count],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let held = raw.f.add_op(
        OpKind::If,
        false,
        vec![pred],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
        Vec::new(),
        vec![then_region, else_region],
    );
    let held = raw.f.op(held).results[0];
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, zero],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, held],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Region);
}

#[test]
fn a_use_def_cycle_is_refused_before_any_recursive_analysis() {
    let mut raw = raw();
    let one = raw.int(1);
    let sum = raw.operand(
        OpKind::Bin(BinOp::Add),
        vec![one, one],
        ValTy::Data(Ty::scalar(Scalar::S32)),
    );
    let add = raw.f.defining_op(sum).expect("the addition");
    raw.f.set_operand(add, 0, sum).expect("same arena");
    let refused = refuses(raw.body(Vec::new()));
    assert_eq!(refused.rule, Rule::Dominance);
    assert!(refused.detail.contains("use-def chain"), "{refused}");
}

#[test]
fn a_region_yielding_its_owners_result_is_a_use_def_cycle() {
    let mut raw = raw();
    let then_block = raw.f.new_block();
    let then_region = raw.f.new_region(vec![then_block]).expect("fresh block");
    let else_block = raw.f.new_block();
    let else_region = raw.f.new_region(vec![else_block]).expect("fresh block");
    let zero = raw.int(0);
    let predicate = raw.operand(
        OpKind::Cmp(CmpOp::Eq),
        vec![zero, raw.count],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let conditional = raw
        .f
        .new_op(
            OpKind::If,
            true,
            vec![predicate],
            &[ValTy::Data(Ty::scalar(Scalar::F32))],
            Vec::new(),
            vec![then_region, else_region],
        )
        .expect("structurally valid if");
    let result = raw.f.op(conditional).results[0];
    raw.f
        .set_yields(then_block, vec![result])
        .expect("same arena");
    raw.f
        .set_yields(else_block, vec![result])
        .expect("same arena");

    let refused = refuses(raw.body(vec![conditional]));
    assert_eq!(refused.rule, Rule::Dominance);
    assert!(refused.detail.contains("use-def chain"), "{refused}");
}

#[test]
fn orphan_blocks_and_regions_are_refused() {
    let mut raw = raw();
    let orphan = raw.f.new_block();
    assert_eq!(refuses(raw.body(Vec::new())).at, Site::Block(orphan));

    let mut raw = super::raw();
    let block = raw.f.new_block();
    let _orphan = raw.f.new_region(vec![block]).expect("fresh block");
    assert_eq!(
        refuses(raw.body(Vec::new())).at,
        Site::Block(block),
        "the orphaned block is diagnosed before its region"
    );
}
