// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{DeviceFunction, Ty};

#[test]
fn moving_nested_guards_carries_their_operand_captures() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let (fragment, ()) = b.build_fragment(|f| {
        let bound = f.bind_value(Ty::scalar(Scalar::F32), x.clone());
        let captured = f.add(bound, x.clone());
        f.if_(pred.clone(), |arm| arm.store(p, captured));
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    verify_round_trip("nested_capture", b.into_function());
}

#[test]
fn moving_sibling_guards_preserves_shared_and_owned_operands() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let shared = b.mul(x.clone(), x.clone());
    let shared_op = b.function().operand_op(shared.id_in(b.function())).unwrap();
    let (fragment, ()) = b.build_fragment(|f| {
        let bound = f.bind_value(Ty::scalar(Scalar::F32), x.clone());
        let captured = f.add(bound, shared.clone());
        for _ in 0..2 {
            f.if_(pred.clone(), |arm| {
                let owned = arm.mul(captured.clone(), x.clone());
                arm.store(p.clone(), owned);
            });
        }
    });
    b.store(p, shared);
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    assert!(b.owned().contains(&shared_op));
    let function = b.into_function();
    assert!(function.stranded().is_empty());
    verify_round_trip("shared_captures", function);
}

#[test]
fn moving_a_loop_keeps_body_operands_inside_and_captures_outside() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let (fragment, ()) = b.build_fragment(|f| {
        let bound = f.bind_value(Ty::scalar(Scalar::F32), x.clone());
        let captured = f.add(bound, x.clone());
        let start = f.i32(0);
        let stop = f.i32(4);
        let step = f.i32(1);
        let open = f.open_loop_body(Scalar::S32, Vec::new());
        let lane = open.index();
        let index = f.cast(Scalar::F32, lane);
        let value = f.add(captured, index);
        f.if_(pred.clone(), |arm| arm.store(p, value));
        let body = f.close_loop_body(open, Vec::new());
        f.place_counted_loop(crate::Trip::PerThread, start, stop, step, body);
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    let function = b.into_function();
    assert!(function.stranded().is_empty());
    verify_round_trip("loop_captures", function);
}

#[test]
fn moving_a_yielding_guard_carries_its_yield_expression() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let (fragment, ()) = b.build_fragment(|f| {
        let bound = f.bind_value(Ty::scalar(Scalar::F32), x.clone());
        let captured = f.add(bound, x.clone());
        f.open_arm();
        let taken = f.close_arm(vec![captured]);
        f.open_arm();
        let untaken = f.close_arm(vec![x]);
        let results = f.if_arms(
            pred.clone(),
            &[(Ty::scalar(Scalar::F32), None)],
            taken,
            untaken,
        );
        f.store(p, results[0].clone());
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    verify_round_trip("yield_captures", b.into_function());
}

#[test]
fn moving_a_guard_preserves_an_unmoved_siblings_yield_capture() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let shared = b.add(x.clone(), x.clone());
    let captured = b.function().operand_op(shared.id_in(b.function())).unwrap();
    b.open_arm();
    let taken = b.close_arm(vec![shared.clone()]);
    b.open_arm();
    let untaken = b.close_arm(vec![x]);
    let results = b.if_arms(
        pred.clone(),
        &[(Ty::scalar(Scalar::F32), None)],
        taken,
        untaken,
    );
    b.store(p.clone(), results[0].clone());
    let (fragment, ()) = b.build_fragment(|f| {
        f.if_(pred.clone(), |arm| arm.store(p, shared));
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    assert!(b.owned().contains(&captured));
    let f = b.into_function();
    verify_round_trip("sibling_yield", f);
}

#[test]
fn moving_a_guard_preserves_its_functions_return_capture() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let shared = b.add(x.clone(), x.clone());
    let captured = b.function().operand_op(shared.id_in(b.function())).unwrap();
    b.returns(vec![shared.clone()]);
    let (fragment, ()) = b.build_fragment(|f| {
        f.if_(pred.clone(), |arm| arm.store(p, shared));
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    assert!(b.owned().contains(&captured));
    let f = b.into_function();
    verify_round_trip("return_capture", f);
}

fn verify_round_trip(name: &str, function: Function) {
    assert!(function.stranded().is_empty());
    assert!(function.misordered().is_empty());
    let text = crate::dialect::print(&function).unwrap();
    let parsed = crate::dialect::parse(&text).unwrap();
    let diagnostics = crate::dialect::validate(&parsed);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    assert_eq!(weave::print(&parsed).unwrap(), text);
    crate::verify::verify(&DeviceFunction::new(name, function)).unwrap();
}

#[test]
fn moving_a_loop_preserves_carried_values_and_an_external_destination() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let doubled = b.add(x.clone(), x.clone());
    let destination = b.bind_value(Ty::scalar(Scalar::F32), doubled);
    let (fragment, ()) = b.build_fragment(|f| {
        let start = f.i32(0);
        let stop = f.i32(4);
        let step = f.i32(1);
        let open = f.open_loop_body(
            Scalar::S32,
            vec![crate::Carry {
                ty: Ty::scalar(Scalar::F32),
                init: x.clone(),
                into: Some(destination),
            }],
        );
        let given = open.given()[0].clone();
        let added = f.add(given.clone(), x);
        let reached = f.update(given, added);
        let body = f.close_loop_body(open, vec![reached]);
        let produced = f.place_counted_loop(crate::Trip::PerThread, start, stop, step, body);
        f.store(p, produced[0].clone());
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    verify_round_trip("carried_destination", b.into_function());
}

#[test]
fn moving_a_guard_keeps_a_deep_shared_capture_chain() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let p = pointer(&mut b, "out");
    let pred = b.lt(x.clone(), x.clone());
    let pred = b.bind_value(Ty::scalar(Scalar::Pred), pred);
    let mut shared = x.clone();
    let mut chain = Vec::new();
    for _ in 0..256 {
        shared = b.add(shared, x.clone());
        chain.push(b.function().operand_op(shared.id_in(b.function())).unwrap());
    }
    b.store(p.clone(), shared.clone());
    let (fragment, ()) = b.build_fragment(|f| {
        f.if_(pred.clone(), |arm| arm.store(p, shared));
    });
    let moved = b.place_fragment(fragment);
    b.if_(pred, |arm| arm.place_all(moved));
    assert!(chain.iter().all(|op| b.owned().contains(op)));
    verify_round_trip("shared_chain", b.into_function());
}
