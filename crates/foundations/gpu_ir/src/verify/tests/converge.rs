// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The threads a barrier needs, and what the arena can prove about
//! them.

use super::*;
use crate::Carry;

/// A barrier under a predicate the threads of a block need not agree on is one
/// the threads that skip the arm never arrive at.
#[test]
fn a_barrier_under_a_divergent_guard_is_refused() {
    let guarded = |per_thread: bool| {
        let mut b = Builder::new();
        let count = b.parameter(Param {
            name: "count".into(),
            ty: ParamTy::Scalar(Scalar::S32),
        });
        let held = if per_thread {
            let tid = b.tid_x();
            b.cast(Scalar::S32, tid)
        } else {
            let blocks = b.ctaid_x();
            b.cast(Scalar::S32, blocks)
        };
        let inside = b.lt(held, count);
        b.if_(inside, |arm| arm.barrier());
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        guarded(true)
            .expect_err("a thread index is one thread's")
            .rule,
        Rule::Convergence
    );
    guarded(false).expect("a block index is the whole block's");
}

/// A thread that returns has exited, and the barrier is satisfied by the
/// threads that remain — which is what makes the early-out guard every
/// grid-strided kernel opens with legal.
#[test]
fn a_barrier_after_a_partial_return_is_legal() {
    let mut b = Builder::new();
    let count = b.parameter(Param {
        name: "count".into(),
        ty: ParamTy::Scalar(Scalar::S32),
    });
    let tid = b.tid_x();
    let tid = b.cast(Scalar::S32, tid);
    let past = b.ge(tid, count);
    b.if_(past, |arm| arm.ret());
    b.barrier();
    accepts(b.into_function());
}

/// How often a loop goes round is how its bounds come out, which the arena
/// cannot read; the producer states it, and a barrier inside is admitted by
/// that statement.
#[test]
fn a_barrier_in_a_loop_needs_the_trip_count_stated() {
    let looped = |trip: Trip| {
        let mut b = Builder::new();
        let stop = b.i32(8);
        let step = b.i32(1);
        let zero = b.i32(0);
        let (body, ()) = b.build_loop_body(Scalar::S32, Vec::new(), |b, _, _| {
            b.barrier();
            (Vec::new(), ())
        });
        b.place_counted_loop(trip, zero, stop, step, body);
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        looped(Trip::PerThread)
            .expect_err("a count nobody stated")
            .rule,
        Rule::Convergence
    );
    looped(Trip::BlockUniform).expect("a stated count admits the barrier");
}

/// A warp collective reduces over the lanes that arrived, so it needs no such
/// statement — that is what makes it usable inside a partially retired warp.
#[test]
fn a_warp_collective_needs_no_convergence_statement() {
    let mut b = Builder::new();
    let out = b.parameter(Param {
        name: "out".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let count = b.parameter(Param {
        name: "count".into(),
        ty: ParamTy::Scalar(Scalar::S32),
    });
    let tid = b.tid_x();
    let tid = b.cast(Scalar::S32, tid);
    let inside = b.lt(tid.clone(), count);
    b.if_(inside, |arm| {
        let held = arm.float(Scalar::F32, 1.0);
        let delta = arm.i32(1);
        let other = arm.shfl_down(held.clone(), delta);
        let sum = arm.add(held, other);
        let at = arm.offset(out, tid);
        arm.store(at, sum);
    });
    accepts(b.into_function());
}

/// A loop that stands for what it reduces to is an *operand*: it runs wherever
/// it is read, not where a step sits. So the rules that admit a barrier in a
/// loop body apply to it there, and a walk that only visited placed steps would
/// leave every such loop unchecked — which is a barrier a per-thread trip count
/// reaches a different number of times, verified.
#[test]
fn a_barrier_in_an_unscheduled_fold_is_checked_where_the_fold_is_read() {
    let folded = |trip: Trip| {
        let mut b = Builder::new();
        let destination = b.parameter(Param {
            name: "destination".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let zero = b.i32(0);
        let four = b.i32(4);
        let one = b.i32(1);
        let start = b.float(Scalar::F32, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: start,
                into: None,
            }],
            |body, _index, given| {
                body.barrier();
                let one = body.float(Scalar::F32, 1.0);
                (vec![body.add(given[0].clone(), one)], ())
            },
        );
        // Never placed: it is read by the store below and evaluated there.
        let total = b.fold_loop(trip, zero.clone(), four, one, body);
        let at = b.offset(destination, zero);
        b.store(at, total.into_iter().next().expect("one carried value"));
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        folded(Trip::PerThread)
            .expect_err("a per-thread trip count reaches its barrier per-thread")
            .rule,
        Rule::Convergence
    );
    folded(Trip::BlockUniform).expect("a block-uniform trip count reaches it together");
}

/// The same cone reached through a function's result rather than through a
/// store, so the check does not depend on there being a placed reader.
#[test]
fn a_barrier_in_a_fold_a_function_hands_back_is_checked_too() {
    let mut b = Builder::new();
    let zero = b.i32(0);
    let four = b.i32(4);
    let one = b.i32(1);
    let start = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![Carry {
            ty: Ty::scalar(Scalar::F32),
            init: start,
            into: None,
        }],
        |body, _index, given| {
            body.barrier();
            let one = body.float(Scalar::F32, 1.0);
            (vec![body.add(given[0].clone(), one)], ())
        },
    );
    let total = b.fold_loop(Trip::PerThread, zero, four, one, body);
    b.returns(total);
    assert_eq!(
        verify_launch(&launched(b.into_function()))
            .expect_err("a returned fold is evaluated where it is returned")
            .rule,
        Rule::Convergence
    );
}

/// A fold whose result nothing reads is evaluated nowhere.
///
/// An operand runs where it is read, so one that is read nowhere runs nowhere,
/// and there is no block whose threads could fail to reach the barrier inside
/// it. Accepting it is the honest answer, and it is what makes the walk a walk
/// over *evaluated* cones rather than over every operation in the arena.
#[test]
fn a_fold_nothing_reads_is_evaluated_nowhere() {
    let mut b = Builder::new();
    let destination = b.parameter(Param {
        name: "destination".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let zero = b.i32(0);
    let four = b.i32(4);
    let one = b.i32(1);
    let start = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![Carry {
            ty: Ty::scalar(Scalar::F32),
            init: start,
            into: None,
        }],
        |body, _index, given| {
            body.barrier();
            let one = body.float(Scalar::F32, 1.0);
            (vec![body.add(given[0].clone(), one)], ())
        },
    );
    // Built and dropped: never stored, never returned, never an operand.
    let _unread = b.fold_loop(Trip::PerThread, zero.clone(), four, one, body);
    let at = b.offset(destination, zero);
    let held = b.float(Scalar::F32, 1.0);
    b.store(at, held);
    verify_launch(&launched(b.into_function())).expect("an operand nothing reads runs nowhere");
}

/// The same cone, this time read inside a conditional arm.
///
/// The guard is a block index, so every thread of the block agrees on it and
/// the arm itself is legal. What is left is the fold, which is evaluated inside
/// that arm — so the rule that admits a barrier in a loop applies there.
#[test]
fn a_barrier_bearing_fold_read_inside_a_guard_is_checked_in_that_arm() {
    let folded = |trip: Trip| {
        let mut b = Builder::new();
        let destination = b.parameter(Param {
            name: "destination".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let count = b.parameter(Param {
            name: "count".into(),
            ty: ParamTy::Scalar(Scalar::S32),
        });
        let blocks = b.ctaid_x();
        let blocks = b.cast(Scalar::S32, blocks);
        let uniform = b.lt(blocks, count);
        b.if_(uniform, |arm| {
            let zero = arm.i32(0);
            let four = arm.i32(4);
            let one = arm.i32(1);
            let start = arm.float(Scalar::F32, 0.0);
            let (body, ()) = arm.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: Ty::scalar(Scalar::F32),
                    init: start,
                    into: None,
                }],
                |body, _index, given| {
                    body.barrier();
                    let one = body.float(Scalar::F32, 1.0);
                    (vec![body.add(given[0].clone(), one)], ())
                },
            );
            let total = arm.fold_loop(trip, zero.clone(), four, one, body);
            let at = arm.offset(destination.clone(), zero);
            arm.store(at, total.into_iter().next().expect("one carried value"));
        });
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        folded(Trip::PerThread)
            .expect_err("the fold runs inside the arm, and its trip count is per thread")
            .rule,
        Rule::Convergence
    );
    folded(Trip::BlockUniform).expect("a block-uniform fold inside a uniform arm is legal");
}

/// And once more inside a placed loop's body, so the walk is not one level deep.
#[test]
fn a_barrier_bearing_fold_read_inside_a_loop_body_is_checked_there() {
    let folded = |trip: Trip| {
        let mut b = Builder::new();
        let destination = b.parameter(Param {
            name: "destination".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let outer_start = b.i32(0);
        let outer_stop = b.i32(2);
        let outer_step = b.i32(1);
        let (outer, ()) = b.build_loop_body(Scalar::S32, Vec::new(), |body, _index, _given| {
            let zero = body.i32(0);
            let four = body.i32(4);
            let one = body.i32(1);
            let start = body.float(Scalar::F32, 0.0);
            let (inner, ()) = body.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: Ty::scalar(Scalar::F32),
                    init: start,
                    into: None,
                }],
                |inner, _index, given| {
                    inner.barrier();
                    let one = inner.float(Scalar::F32, 1.0);
                    (vec![inner.add(given[0].clone(), one)], ())
                },
            );
            let total = body.fold_loop(trip, zero.clone(), four, one, inner);
            let at = body.offset(destination.clone(), zero);
            body.store(at, total.into_iter().next().expect("one carried value"));
            (Vec::new(), ())
        });
        b.place_counted_loop(
            Trip::BlockUniform,
            outer_start,
            outer_stop,
            outer_step,
            outer,
        );
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        folded(Trip::PerThread)
            .expect_err("a fold nested two levels down is still evaluated somewhere")
            .rule,
        Rule::Convergence
    );
    folded(Trip::BlockUniform).expect("a block-uniform fold nested inside a uniform loop is legal");
}

/// A divergent guard whose arm synchronises *only* through an unscheduled fold.
///
/// The fold itself is block-uniform, so the loop rule passes and cannot be what
/// refuses this. What is left is the guard: the arm holds a barrier, reachable
/// only by following the store's operand into the fold's body, and the threads
/// that skip the arm never arrive. This pins `block_synchronisation` following
/// operand cones — without that, the arm looks barrier-free and a per-thread
/// guard over it verifies.
#[test]
fn a_divergent_guard_over_an_arm_that_synchronises_only_through_a_fold_is_refused() {
    let guarded = |per_thread: bool| {
        let mut b = Builder::new();
        let destination = b.parameter(Param {
            name: "destination".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let count = b.parameter(Param {
            name: "count".into(),
            ty: ParamTy::Scalar(Scalar::S32),
        });
        let held = if per_thread {
            let tid = b.tid_x();
            b.cast(Scalar::S32, tid)
        } else {
            let blocks = b.ctaid_x();
            b.cast(Scalar::S32, blocks)
        };
        let guard = b.lt(held, count);
        b.if_(guard, |arm| {
            let zero = arm.i32(0);
            let four = arm.i32(4);
            let one = arm.i32(1);
            let start = arm.float(Scalar::F32, 0.0);
            let (body, ()) = arm.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: Ty::scalar(Scalar::F32),
                    init: start,
                    into: None,
                }],
                |body, _index, given| {
                    body.barrier();
                    let one = body.float(Scalar::F32, 1.0);
                    (vec![body.add(given[0].clone(), one)], ())
                },
            );
            // Block-uniform, so the loop rule is satisfied and the only thing
            // left to refuse this is the guard above it.
            let total = arm.fold_loop(Trip::BlockUniform, zero.clone(), four, one, body);
            let at = arm.offset(destination.clone(), zero);
            arm.store(at, total.into_iter().next().expect("one carried value"));
        });
        verify_launch(&launched(b.into_function()))
    };
    let refused = guarded(true).expect_err("the arm's only barrier is inside the fold it reads");
    assert_eq!(refused.rule, Rule::Convergence);
    assert!(
        refused.to_string().contains("skip the arm"),
        "the guard is what is refused, not the loop: {refused}"
    );
    guarded(false).expect("a block index is the whole block's");
}
