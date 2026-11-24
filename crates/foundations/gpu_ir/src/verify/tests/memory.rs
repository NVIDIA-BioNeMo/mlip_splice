// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Addressing, the widths a space admits, the alignment a vector form
//! owes, and the storage a result runs in.

use super::*;

/// The address an operation touches memory through is the operand its effect
/// names, and it holds an address.
#[test]
fn a_write_through_a_datum_is_refused() {
    let mut raw = raw();
    let value = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![value, value],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Effect);
}

/// `red.global.add` is a global reduction with no eviction hint and no form for
/// sub-word integers.
#[test]
fn an_atomic_add_states_what_it_has_a_form_for() {
    let mut b = Builder::new();
    let shared = b.shared_array(SharedArr {
        name: "stripe".into(),
        elem: Scalar::F32,
        len: 32,
    });
    let zero = b.i32(0);
    let at = b.offset(shared, zero);
    let one = b.float(Scalar::F32, 1.0);
    b.red_add(at, one);
    assert_eq!(refuses(b.into_function()).rule, Rule::Memory);

    let mut b = Builder::new();
    let buffer = b.parameter(Param {
        name: "buffer".into(),
        ty: ParamTy::Ptr(Scalar::S8),
    });
    let zero = b.i32(0);
    let at = b.offset(buffer, zero);
    let one = b.int(Scalar::S8, 1);
    b.red_add(at, one);
    assert_eq!(refuses(b.into_function()).rule, Rule::Memory);
}

/// The addition that hands back what was there answers to the same two rules,
/// because it is the same addition.
#[test]
fn an_atomic_add_that_hands_back_states_what_it_has_a_form_for() {
    let mut b = Builder::new();
    let shared = b.shared_array(SharedArr {
        name: "stripe".into(),
        elem: Scalar::F32,
        len: 32,
    });
    let zero = b.i32(0);
    let at = b.offset(shared, zero);
    let one = b.float(Scalar::F32, 1.0);
    b.fetch_add(at, one);
    assert_eq!(refuses(b.into_function()).rule, Rule::Memory);

    let mut b = Builder::new();
    let buffer = b.parameter(Param {
        name: "buffer".into(),
        ty: ParamTy::Ptr(Scalar::S8),
    });
    let zero = b.i32(0);
    let at = b.offset(buffer, zero);
    let one = b.int(Scalar::S8, 1);
    b.fetch_add(at, one);
    assert_eq!(refuses(b.into_function()).rule, Rule::Memory);

    let mut b = Builder::new();
    let buffer = b.parameter(Param {
        name: "buffer".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let zero = b.i32(0);
    let at = b.offset(buffer, zero);
    let one = b.float(Scalar::F32, 1.0);
    b.fetch_add(at, one);
    accepts(b.into_function());
}

#[test]
fn a_non_coherent_read_of_shared_memory_is_refused() {
    let mut b = Builder::new();
    let shared = b.shared_array(SharedArr {
        name: "stripe".into(),
        elem: Scalar::F32,
        len: 32,
    });
    let out = b.parameter(Param {
        name: "out".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let zero = b.i32(0);
    let at = b.offset(shared, zero.clone());
    let read = b.load(at, 1, true);
    let to = b.offset(out, zero);
    b.store(to, read);
    assert_eq!(refuses(b.into_function()).rule, Rule::Memory);
}

/// A four-lane access renders as `ld.global.v4`, whose address must land on a
/// four-element boundary. The offset half of that is decided from the arena.
#[test]
fn a_wide_access_at_an_unproven_offset_is_refused() {
    let wide = |scale: i64| {
        let mut b = Builder::new();
        let source = b.parameter(Param {
            name: "in".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let out = b.parameter(Param {
            name: "out".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let factor = b.i32(scale);
        let at = b.mul(tid, factor);
        let from = b.offset(source, at);
        let read = b.load(from, 4, true);
        let read = b.bind_value_as("read", Ty::vec(Scalar::F32, 4), read);
        let zero = b.i32(0);
        let to = b.offset(out, zero);
        b.store(to, read);
        verify_launch(&launched(b.into_function()))
    };
    assert_eq!(
        wide(1).expect_err("one element apart proves nothing").rule,
        Rule::Alignment
    );
    wide(4).expect("four elements apart is a four-element boundary");
}

/// A loop index steps from its start by its step, so what both are multiples of
/// is what every value it takes is a multiple of — which is how a strided walk
/// discharges a vector access.
#[test]
fn a_wide_access_at_a_loop_index_is_proven_from_the_bounds() {
    let strided = |step: i64| {
        let mut b = Builder::new();
        let source = b.parameter(Param {
            name: "in".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let out = b.parameter(Param {
            name: "out".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let four = b.i32(4);
        let start = b.mul(tid, four);
        let stop = b.i32(256);
        let stride = b.i32(step);
        let zero = b.float(Scalar::F32, 0.0);
        let zero = b.splat(4, zero);
        let [_folded] = b.counted_loop(
            Counted {
                index: Scalar::S32,
                start,
                stop,
                step: stride,
                trip: Trip::PerThread,
            },
            [(Ty::vec(Scalar::F32, 4), zero)],
            |b, index, [held]| {
                let from = b.offset(source, index);
                let read = b.load(from, 4, true);
                [b.add(held, read)]
            },
        );
        let zero = b.i32(0);
        let to = b.offset(out, zero);
        let one = b.float(Scalar::F32, 1.0);
        b.store(to, one);
        verify_launch(&launched(b.into_function()))
    };
    strided(4).expect("a four-element start and a four-element step stay aligned");
    assert_eq!(
        strided(3)
            .expect_err("a three-element step leaves the boundary")
            .rule,
        Rule::Alignment
    );
}

/// A binding runs in the storage it names, with no arm to fall back to, so the
/// two hold the same thing. A branch's destination is only a request, and the
/// backend declines one it cannot honour.
#[test]
fn a_binding_into_storage_of_another_type_is_refused() {
    let mut raw = raw();
    let one = raw.float(1.0);
    let bind = raw.f.add_op(
        OpKind::Bind,
        true,
        vec![one],
        &[ValTy::Data(Ty::scalar(Scalar::F32))],
        Vec::new(),
        Vec::new(),
    );
    let place = raw.f.op(bind).results[0];
    let wide = raw.operand(
        OpKind::Splat(2),
        vec![one],
        ValTy::Data(Ty::vec(Scalar::F32, 2)),
    );
    let bound = raw.f.add_op(
        OpKind::Bind,
        true,
        vec![wide],
        &[ValTy::Data(Ty::vec(Scalar::F32, 2))],
        vec![Some(place)],
        Vec::new(),
    );
    assert_eq!(refuses(raw.body(vec![bind, bound])).rule, Rule::Aliasing);
}
