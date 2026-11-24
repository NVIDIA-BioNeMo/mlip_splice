// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One indivisible addition, in the form what reads its result asks for.
//!
//! `atom` and `red` are the same addition. Which one is emitted is decided from
//! the program — whether anything reads the value that was there before — and
//! never from a flag a caller sets, so the two can never disagree about what the
//! program means.

use super::{kernel, ptr, render};
use crate::arena::Function;
use crate::build::Builder;
use crate::func::{DeviceFunction, Launch};
use crate::op::{OpKind, WriteMode};
use crate::{Evict, Scalar, Ty};

/// `counter[0] += 1`, with the previous value stored where `read` says.
fn counting(read: bool) -> Launch {
    kernel(move |b| {
        let counter = ptr(b, "counter", Scalar::S32);
        let out = ptr(b, "out", Scalar::S32);
        let one = b.int(Scalar::S32, 1);
        let before = b.fetch_add(counter, one);
        if read {
            b.store(out, before);
        }
    })
}

/// The words a kernel declares of one register class, which is what says
/// whether a result was given storage.
fn declared(ptx: &str, class: &str) -> usize {
    let line = ptx
        .lines()
        .find(|line| line.contains(&format!("{class}<")))
        .unwrap_or_else(|| panic!("no {class} registers: {ptx}"));
    line.rsplit_once('<')
        .and_then(|(_, held)| held.trim_end_matches(">;").parse().ok())
        .expect("a register count")
}

#[test]
fn a_result_that_is_read_comes_back_in_a_register() {
    let ptx = render(&counting(true), 90);
    assert_eq!(ptx.matches("atom.global.add.u32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("red.global.add").count(), 0, "{ptx}");
    let held = ptx
        .lines()
        .find(|line| line.contains("atom.global.add"))
        .expect("one atomic");
    let destination = held
        .trim()
        .trim_start_matches("atom.global.add.u32")
        .trim()
        .split(',')
        .next()
        .expect("a destination")
        .to_string();
    assert!(destination.starts_with("%r"), "{ptx}");
    assert!(
        ptx.contains(&format!("st.global.b32 [%rd4], {destination};")),
        "{ptx}"
    );
}

/// Nothing reads it, so nothing asks the device to hand it back — and no
/// register is spent holding what nothing reads.
#[test]
fn a_result_nothing_reads_lowers_to_the_reduction() {
    let read = render(&counting(true), 90);
    let ptx = render(&counting(false), 90);
    assert_eq!(ptx.matches("red.global.add.u32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("atom.global.add").count(), 0, "{ptx}");
    let held = ptx
        .lines()
        .find(|line| line.contains("red.global.add"))
        .expect("one reduction");
    assert!(held.trim().ends_with("[%rd2], %r1;"), "{ptx}");
    assert_eq!(
        declared(&read, "%r") - declared(&ptx, "%r"),
        1,
        "the result register is the whole difference:\n{read}\n{ptx}"
    );
}

/// The two forms are one decision over one program, so moving the program
/// between them moves the instruction — which is what makes the pair above a
/// proof rather than two kernels that happen to differ.
#[test]
fn taking_the_use_away_and_putting_it_back_moves_the_instruction() {
    let mut b = Builder::new();
    let counter = ptr(&mut b, "counter", Scalar::S32);
    let out = ptr(&mut b, "out", Scalar::S32);
    let one = b.int(Scalar::S32, 1);
    let before = b.fetch_add(counter, one);
    b.store(out.clone(), before.clone());
    let mut body = b.into_function();

    let read = |body: &Function| {
        let launch = Launch::new(DeviceFunction::new("k", body.clone()), 64, None);
        render(&launch, 90)
    };
    assert!(read(&body).contains("atom.global.add"), "as built");

    // The store is the only use of what the addition produced. Take it out of
    // the program and the value is read nowhere.
    let entry = body.entry();
    let store = *body.block(entry).ops.last().expect("a step");
    body.detach_step(store).expect("it is a step");
    body.delete_op(store)
        .expect("nothing reads what it produced");
    let without = read(&body);
    assert!(without.contains("red.global.add"), "{without}");
    assert!(!without.contains("atom.global.add"), "{without}");

    // Give it a reader again and the previous value is asked for again.
    let adding = body
        .live_ops()
        .into_iter()
        .find(|&op| matches!(body.op(op).kind, OpKind::FetchAdd))
        .expect("the addition");
    let produced = body.op(adding).results[0];
    let destination = body.declared(1);
    let again = body
        .new_op(
            OpKind::Write {
                mode: WriteMode::Set,
                evict: Evict::Normal,
            },
            true,
            vec![destination, produced],
            &[],
            Vec::new(),
            Vec::new(),
        )
        .expect("a write of what the addition produced");
    let position = body.block(entry).ops.len();
    body.insert_step(entry, position, again).expect("a step");
    let restored = read(&body);
    assert!(restored.contains("atom.global.add"), "{restored}");
    assert!(!restored.contains("red.global.add"), "{restored}");
}

/// Both forms spell one scalar the same way, because they are one addition.
#[test]
fn both_forms_spell_the_scalar_the_way_the_reduction_does() {
    for (elem, spelled) in [
        (Scalar::S32, "u32"),
        (Scalar::U32, "u32"),
        (Scalar::S64, "u64"),
        (Scalar::U64, "u64"),
        (Scalar::F32, "f32"),
        (Scalar::F64, "f64"),
        (Scalar::F16, "noftz.f16"),
        (Scalar::BF16, "noftz.bf16"),
    ] {
        let held = |read: bool| {
            let k = kernel(move |b| {
                let counter = ptr(b, "counter", elem);
                let out = ptr(b, "out", elem);
                let one = if elem.is_float() {
                    b.float(elem, 1.0)
                } else {
                    b.int(elem, 1)
                };
                let before = b.fetch_add(counter, one);
                if read {
                    let doubled = b.add(before.clone(), before);
                    let bound = b.bind_value_as("doubled", Ty::scalar(elem), doubled);
                    b.store(out, bound);
                }
            });
            render(&k, 90)
        };
        assert!(
            held(true).contains(&format!("atom.global.add.{spelled} ")),
            "{elem:?}"
        );
        assert!(
            held(false).contains(&format!("red.global.add.{spelled} ")),
            "{elem:?}"
        );
    }
}
