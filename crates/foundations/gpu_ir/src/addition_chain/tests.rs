// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What each way of summing the same terms costs in rounding depth.

use super::longest_addition_chain;
use crate::{
    Builder, Carry, Param, ParamTy, Scalar, SharedArr, Trip, Ty, Value, arena::Function,
    build::LoopBody,
};

fn f32() -> Ty {
    Ty::scalar(Scalar::F32)
}

fn ptr(b: &mut Builder, name: &str) -> Value {
    b.parameter(Param {
        name: name.into(),
        ty: ParamTy::Ptr(Scalar::F32),
    })
}

fn built(build: impl FnOnce(&mut Builder)) -> Function {
    let mut b = Builder::new();
    build(&mut b);
    b.into_function()
}

/// `total` terms summed by one thread, one `fma` after another.
fn sequential_sum(total: i64, stride: i64) -> Function {
    built(|b| {
        let source = ptr(b, "in");
        let out = ptr(b, "out");
        let empty = b.zero(Scalar::F32);
        let accumulator = b.bind_value_as("acc", f32(), empty);
        let start = b.tid_x();
        let start = b.cast(Scalar::S64, start);
        let stop = b.i64(total);
        let step = b.i64(stride);
        let open = b.open_loop_body(
            Scalar::S64,
            vec![Carry {
                ty: f32(),
                init: accumulator.clone(),
                into: Some(accumulator),
            }],
        );
        let index = open.index();
        let running = open.given()[0].clone();
        let at = b.offset(source.clone(), index);
        let term = b.load(at, 1, false);
        let one = b.float(Scalar::F32, 1.0);
        let folded = b.fma(term, one, running);
        let body: LoopBody = b.close_loop_body(open, vec![folded]);
        let summed = b.place_counted_loop(Trip::PerThread, start, stop, step, body);
        let zero = b.i64(0);
        let cell = b.offset(out, zero);
        b.store(cell, summed[0].clone());
    })
}

#[test]
fn a_sequential_sum_is_as_deep_as_it_is_long() {
    let chain = longest_addition_chain(&sequential_sum(1000, 1)).unwrap();
    assert_eq!(chain.within_thread, 1000);
    assert!(!chain.folds_across_threads);
}

#[test]
fn splitting_the_same_terms_over_threads_shortens_every_chain() {
    let chain = longest_addition_chain(&sequential_sum(1000, 128)).unwrap();
    assert_eq!(chain.within_thread, 8);
}

#[test]
fn a_warp_halving_is_as_deep_as_the_number_of_steps() {
    let function = built(|b| {
        let source = ptr(b, "in");
        let out = ptr(b, "out");
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let at = b.offset(source, tid);
        let mut sum = b.load(at, 1, false);
        for offset in [16, 8, 4, 2, 1] {
            let delta = b.i32(offset);
            let other = b.shfl_down(sum.clone(), delta);
            sum = b.add(sum, other);
        }
        let zero = b.i64(0);
        let cell = b.offset(out, zero);
        b.store(cell, sum);
    });
    let chain = longest_addition_chain(&function).unwrap();
    assert_eq!(chain.within_thread, 5);
}

#[test]
fn a_reduction_through_shared_memory_keeps_its_chain() {
    let function = built(|b| {
        let source = ptr(b, "in");
        let out = ptr(b, "out");
        let partials = b.shared_array(SharedArr {
            name: "partials".into(),
            elem: Scalar::F32,
            len: 4,
        });
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let at = b.offset(source, tid.clone());
        let first = b.load(at.clone(), 1, false);
        let second = b.load(at, 1, false);
        let mine = b.add(first, second);
        let slot = b.offset(partials.clone(), tid.clone());
        b.store(slot, mine);
        b.barrier();
        let mut total = b.zero(Scalar::F32);
        for lane in 0..4 {
            let index = b.i32(lane);
            let cell = b.offset(partials.clone(), index);
            let partial = b.load(cell, 1, false);
            total = b.add(total, partial);
        }
        let zero = b.i64(0);
        let cell = b.offset(out, zero);
        b.store(cell, total);
    });
    let chain = longest_addition_chain(&function).unwrap();
    // One addition before the store, then four to fold the four slots.
    assert_eq!(chain.within_thread, 5);
}

#[test]
fn folding_into_global_memory_is_reported_as_a_launch_wide_fold() {
    let function = built(|b| {
        let source = ptr(b, "in");
        let out = ptr(b, "out");
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let at = b.offset(source, tid);
        let term = b.load(at, 1, false);
        let zero = b.i64(0);
        let cell = b.offset(out, zero);
        b.fetch_add(cell, term);
    });
    let chain = longest_addition_chain(&function).unwrap();
    assert_eq!(chain.within_thread, 0);
    assert!(chain.folds_across_threads);
}

#[test]
fn a_loop_no_constant_bounds_admits_no_depth() {
    let function = built(|b| {
        let source = ptr(b, "in");
        let out = ptr(b, "out");
        let empty = b.zero(Scalar::F32);
        let accumulator = b.bind_value_as("acc", f32(), empty);
        let start = b.i64(0);
        let stop = b.tid_x();
        let stop = b.cast(Scalar::S64, stop);
        let step = b.i64(1);
        let open = b.open_loop_body(
            Scalar::S64,
            vec![Carry {
                ty: f32(),
                init: accumulator.clone(),
                into: Some(accumulator),
            }],
        );
        let index = open.index();
        let running = open.given()[0].clone();
        let at = b.offset(source, index);
        let term = b.load(at, 1, false);
        let folded = b.add(running, term);
        let body = b.close_loop_body(open, vec![folded]);
        let summed = b.place_counted_loop(Trip::PerThread, start, stop, step, body);
        let zero = b.i64(0);
        let cell = b.offset(out, zero);
        b.store(cell, summed[0].clone());
    });
    assert!(longest_addition_chain(&function).is_none());
}
