// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One function called by another, and the body that arrives when it is run
//! where it stands.

use std::sync::Arc;

use super::{Builder, kinds, pointer};
use crate::func::{DeviceFunction, Launch};
use crate::op::OpKind;
use crate::ptx::render_ptx_module;
use crate::{Param, ParamTy, Scalar, Ty, inline_calls, verify_launch};

/// `f(x) = x + 1`, over one scalar, handed back.
fn add_one() -> Arc<DeviceFunction> {
    let mut b = Builder::new();
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let one = b.float(Scalar::F32, 1.0);
    let sum = b.add(x, one);
    let sum = b.bind_value_as("sum", Ty::scalar(Scalar::F32), sum);
    b.returns(vec![sum]);
    Arc::new(DeviceFunction::new("add_one", b.into_function()))
}

/// A kernel that stores `add_one(in[0])` at `out[0]`, calling it or not.
fn caller(callee: Option<Arc<DeviceFunction>>) -> Launch {
    let mut b = Builder::new();
    let source = pointer(&mut b, "in");
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let at = b.offset(source, zero.clone());
    let read = b.load(at, 1, true);
    let read = b.bind_value_as("read", Ty::scalar(Scalar::F32), read);
    let value = match callee {
        Some(callee) => b
            .call(callee, vec![read])
            .into_iter()
            .next()
            .expect("one result"),
        None => {
            let one = b.float(Scalar::F32, 1.0);
            let sum = b.add(read, one);
            b.bind_value_as("sum", Ty::scalar(Scalar::F32), sum)
        }
    };
    let to = b.offset(out, zero);
    b.store(to, value);
    Launch::new(DeviceFunction::new("k", b.into_function()), 64, None)
}

#[test]
fn a_call_produces_what_its_callee_hands_back() {
    let launch = caller(Some(add_one()));
    let f = launch.body();
    let call = f
        .block(f.entry())
        .ops
        .iter()
        .copied()
        .find(|&op| matches!(f.op(op).kind, OpKind::Call(_)))
        .expect("the body calls");
    assert_eq!(f.op(call).results.len(), 1);
    assert_eq!(f.data_ty(f.op(call).results[0]), Ty::scalar(Scalar::F32));
    verify_launch(&launch).expect("a call agrees with its callee");
}

/// The callee's body arrives where the call stood, with the argument standing
/// in for the parameter, and nothing is left to call.
#[test]
fn inlining_runs_the_callee_where_it_was_called() {
    let mut launch = caller(Some(add_one()));
    inline_calls(&mut launch.function).expect("the callee is inlinable");
    let f = launch.body();
    let steps = f.block(f.entry()).ops.clone();
    assert!(
        !kinds(f, &steps)
            .iter()
            .any(|kind| matches!(kind, OpKind::Call(_))),
        "inlining leaves no call"
    );
    let names = steps
        .iter()
        .flat_map(|&op| f.op(op).results.clone())
        .filter_map(|result| f.name(result).map(str::to_owned))
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["read", "sum"]);
}

/// The backend never sees a call, and what it emits for one is what it emits
/// for the body written out by hand.
#[test]
fn a_called_body_renders_as_the_same_body_written_inline() {
    let called = render_ptx_module(std::slice::from_ref(&caller(Some(add_one()))), 90);
    let direct = render_ptx_module(std::slice::from_ref(&caller(None)), 90);
    assert_eq!(called, direct);
}

/// A call is a use of a function, so two calls of one callee run it twice and
/// share nothing but the function.
#[test]
fn two_calls_of_one_callee_each_run_it() {
    let callee = add_one();
    let mut b = Builder::new();
    let source = pointer(&mut b, "in");
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let at = b.offset(source, zero.clone());
    let read = b.load(at, 1, true);
    let read = b.bind_value_as("read", Ty::scalar(Scalar::F32), read);
    let once = b
        .call(callee.clone(), vec![read])
        .into_iter()
        .next()
        .expect("one result");
    let twice = b
        .call(callee, vec![once])
        .into_iter()
        .next()
        .expect("one result");
    let to = b.offset(out, zero);
    b.store(to, twice);
    let mut launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    verify_launch(&launch).expect("both calls agree with the callee");
    inline_calls(&mut launch.function).expect("the callee is inlinable");
    let f = launch.body();
    let steps = f.block(f.entry()).ops.clone();
    let sums = steps
        .iter()
        .flat_map(|&op| f.op(op).results.clone())
        .filter(|&result| f.name(result) == Some("sum"))
        .count();
    assert_eq!(sums, 2, "each call runs the body");
}

/// An argument of the wrong type is refused where the call is, naming the
/// callee.
#[test]
fn a_call_that_disagrees_with_its_callee_is_refused() {
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let wrong = b.i32(1);
    let callee = add_one();
    let tys = callee.result_tys();
    let operands = vec![b.read(wrong)];
    let op = b.place(OpKind::Call(callee), operands, &tys, Vec::new(), Vec::new());
    let value = b.value_of(b.function().op(op).results[0]);
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    b.store(to, value);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let refused = verify_launch(&launch).expect_err("the argument is an integer");
    assert_eq!(refused.rule, crate::verify::Rule::Type);
    assert!(refused.detail.contains("add_one"), "{refused}");
}
