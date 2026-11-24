// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the renderer emits for the forms that produce values: which register a
//! result lands in, whose it may take over, and what a loop carries.

use super::{f32, f32_load, io_kernel, lane_below_four, render};
use crate::build::{Builder, Value};
use crate::{Carry, Scalar, Trip};

/// The message rendering refused this body with.
fn refused(build: impl FnOnce(&mut Builder, Value, Value)) -> String {
    let k = io_kernel(build);
    let panic =
        std::panic::catch_unwind(|| render(&k, 120)).expect_err("this body must be refused");
    match panic.downcast_ref::<String>() {
        Some(message) => message.clone(),
        None => panic
            .downcast_ref::<&str>()
            .expect("a string panic")
            .to_string(),
    }
}

// ---- conditionals --------------------------------------------------------

/// One register per result, allocated before the branch, so every arm writes
/// the same one and the join copies nothing.
#[test]
fn both_arms_of_an_if_write_one_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let pred = lane_below_four(b);
            let read = f32_load(b, source);
            let chosen = b.if_value(
                pred,
                f32(),
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    b.add(read.clone(), one)
                },
                |b| b.float(Scalar::F32, 0.0),
            );
            b.store(out, chosen);
        }),
        120,
    );
    // The default is pure, so it lands before the branch and the then arm
    // overwrites it: one label, no join.
    assert_eq!(ptx.matches("$Lelse_").count(), 2, "one label: {ptx}");
    assert!(ptx.contains("$Lend_"), "{ptx}");
    let written = ptx
        .lines()
        .filter(|line| line.contains("mov.f32") || line.contains("add.rn.f32"))
        .map(|line| {
            line.split_whitespace()
                .nth(1)
                .unwrap()
                .trim_end_matches(',')
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(written.len(), 2, "{ptx}");
    assert_eq!(written[0], written[1], "{ptx}");
}

/// A default that reads memory may not be evaluated where the predicate fails,
/// so it stays in its arm and the branch needs a join.
#[test]
fn a_reading_default_stays_in_its_arm() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let pred = lane_below_four(b);
            let read = source.clone();
            let chosen = b.if_value(
                pred,
                f32(),
                |b| b.float(Scalar::F32, 1.0),
                move |b| f32_load(b, read.clone()),
            );
            b.store(out, chosen);
        }),
        120,
    );
    assert!(ptx.contains("$Lend_"), "the join is there: {ptx}");
    let else_label = ptx.find("$Lelse_1:").expect("an else label");
    let load = ptx.find("ld.global.f32").expect("the guarded load");
    assert!(load > else_label, "{ptx}");
}

/// Every result of a multi-result `if` is written by both arms.
#[test]
fn every_result_of_a_multi_result_if_is_written() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let pred = lane_below_four(b);
            let read = f32_load(b, source);
            let held = read.clone();
            let [left, right] = b.if_values(
                pred,
                [f32(), f32()],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    let up = b.add(held.clone(), one.clone());
                    [up, one]
                },
                move |b| {
                    let two = b.float(Scalar::F32, 2.0);
                    [read.clone(), two]
                },
            );
            let sum = b.add(left, right);
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains("add.rn.f32"), "{ptx}");
    // Two results, two registers, and the arms write both.
    assert!(ptx.contains(".reg .f32 %f<"), "{ptx}");
}

/// An `if` producing nothing keeps the statement form: one label, no register
/// to hand back.
#[test]
fn a_zero_result_if_keeps_the_statement_form() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let pred = lane_below_four(b);
            b.if_(pred, move |b| {
                let read = f32_load(b, source.clone());
                b.store(out.clone(), read);
            });
        }),
        120,
    );
    assert_eq!(ptx.matches("$Lelse_").count(), 2, "{ptx}");
    assert!(!ptx.contains("$Lend_"), "{ptx}");
}

/// A result is undefined where the arm producing it leaves, so a body written
/// that way is refused rather than rendered.
#[test]
fn an_arm_that_leaves_its_region_is_refused() {
    let why = refused(|b, _, out| {
        let pred = lane_below_four(b);
        let chosen = b.if_value(
            pred,
            f32(),
            |b| {
                b.ret();
                b.float(Scalar::F32, 1.0)
            },
            |b| b.float(Scalar::F32, 0.0),
        );
        b.store(out, chosen);
    });
    assert!(
        why.contains("undefined where its arm returns or breaks"),
        "{why}"
    );
}

/// A region ending in exactly what its last step produces produces it in place:
/// the region's registers are handed down, and nothing is copied at the yield.
#[test]
fn a_region_yielding_its_last_steps_results_produces_them_in_place() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let outer = lane_below_four(b);
            let read = f32_load(b, source);
            let inner_pred = {
                let tid = b.tid_x();
                let tid = b.cast(Scalar::S32, tid);
                let two = b.i32(2);
                b.lt(tid, two)
            };
            let held = read.clone();
            let chosen = b.if_value(
                outer,
                f32(),
                move |b| {
                    b.if_value(
                        inner_pred.clone(),
                        f32(),
                        {
                            let held = held.clone();
                            move |b| {
                                let one = b.float(Scalar::F32, 1.0);
                                b.add(held.clone(), one)
                            }
                        },
                        move |_| held.clone(),
                    )
                },
                move |b| b.float(Scalar::F32, 0.0),
            );
            b.store(out, chosen);
        }),
        120,
    );
    // Both arms of the nested `if` write the outer result's own register, and
    // nothing copies into it once they have.
    assert!(ptx.contains("add.rn.f32 %f1,"), "{ptx}");
    assert!(ptx.contains("ld.global.f32 %f1,"), "{ptx}");
    assert_eq!(
        ptx.matches("mov.f32").count(),
        1,
        "only the hoisted default: {ptx}"
    );
}

// ---- counted loops -------------------------------------------------------

/// One carried value, updated each iteration and read after the loop.
fn accumulating(b: &mut Builder, source: Value, into: Option<Value>, start: Value) -> Value {
    let (zero, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![Carry {
            ty: f32(),
            init: start,
            into,
        }],
        move |b, _, given| {
            let read = f32_load(b, source.clone());
            (vec![b.add(given[0].clone(), read)], ())
        },
    );
    b.place_counted_loop(Trip::PerThread, zero, stop, step, body)
        .into_iter()
        .next()
        .unwrap()
}

/// A loop carrying nothing is the counted loop and nothing else: an index, a
/// bound, a step.
#[test]
fn a_loop_carrying_nothing_is_the_counted_loop() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let (zero, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
            let (body, ()) = b.build_loop_body(Scalar::S32, Vec::new(), move |b, _, _| {
                let read = f32_load(b, source.clone());
                b.store(out.clone(), read);
                (Vec::new(), ())
            });
            b.place_counted_loop(Trip::PerThread, zero, stop, step, body);
        }),
        120,
    );
    assert_eq!(ptx.matches("setp.lt.s32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("add.s32").count(), 1, "the index step: {ptx}");
}

/// A carried value naming the binding it starts from runs in that binding's
/// register: the loop writes it at the backedge, so there is nothing to copy.
#[test]
fn a_carried_value_takes_over_the_binding_it_starts_from() {
    let taking_over = render(
        &io_kernel(|b, source, out| {
            let zero = b.float(Scalar::F32, 0.0);
            let place = b.bind_value_as("acc", f32(), zero);
            let total = accumulating(b, source, Some(place.clone()), place);
            b.store(out, total);
        }),
        120,
    );
    let apart = render(
        &io_kernel(|b, source, out| {
            let zero = b.float(Scalar::F32, 0.0);
            let place = b.bind_value_as("acc", f32(), zero);
            let total = accumulating(b, source, None, place.clone());
            let both = b.add(total, place);
            b.store(out, both);
        }),
        120,
    );
    let registers = |ptx: &str| {
        ptx.lines()
            .find(|line| line.contains(".reg .f32"))
            .expect("f32 registers")
            .to_owned()
    };
    assert_eq!(
        registers(&taking_over),
        "    .reg .f32 %f<3>;",
        "{taking_over}"
    );
    assert_eq!(registers(&apart), "    .reg .f32 %f<5>;", "{apart}");
}

/// Only a place this region gave storage to. A loop nested under a branch that
/// carries into a place bound outside it must not run in that place's register:
/// the branch may not be taken, and the loop's own value is a value of the
/// region it stands in, not of the one that outlives it. Coalescing there would
/// leave an ancestor's storage changed by a loop whose result was discarded.
#[test]
fn a_nested_loop_does_not_carry_in_an_ancestor_places_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let zero = b.float(Scalar::F32, 0.0);
            let place = b.bind_value_as("acc", f32(), zero);
            let pred = lane_below_four(b);
            let held = place.clone();
            b.if_(pred, move |b| {
                // The result is thrown away, so nothing may have been written
                // into the place the loop names.
                accumulating(b, source.clone(), Some(held.clone()), held.clone());
            });
            b.store(out, place);
        }),
        120,
    );
    let carried = ptx
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("add.rn.f32"))
        .expect("the fold")
        .split_whitespace()
        .nth(1)
        .unwrap()
        .trim_end_matches(',')
        .to_owned();
    let stored = ptx
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("st.global.f32"))
        .expect("the store")
        .rsplit(", ")
        .next()
        .unwrap()
        .trim_end_matches(';')
        .to_owned();
    assert_ne!(carried, stored, "the discarded loop folded into acc: {ptx}");
    assert!(ptx.contains("mov.f32 %f1, 0f00000000;"), "{ptx}");
}

/// Two carried values each get their own, and a body that reads one while
/// producing the other updates both at once rather than in sequence.
#[test]
fn carried_values_that_read_each_other_update_at_once() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let (zero, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
            let first = b.float(Scalar::F32, 0.0);
            let second = b.float(Scalar::F32, 1.0);
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![
                    Carry {
                        ty: f32(),
                        init: first,
                        into: None,
                    },
                    Carry {
                        ty: f32(),
                        init: second,
                        into: None,
                    },
                ],
                move |b, _, given| {
                    let read = f32_load(b, source.clone());
                    // Each next value reads the other's current one, so writing
                    // them in sequence would feed the second the first's update.
                    let next_first = b.add(given[1].clone(), read);
                    let next_second = b.mul(given[0].clone(), given[1].clone());
                    (vec![next_first, next_second], ())
                },
            );
            let carried = b.place_counted_loop(Trip::PerThread, zero, stop, step, body);
            let sum = b.add(carried[0].clone(), carried[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    // Both updates are staged and then moved in, so the pair changes together.
    assert_eq!(ptx.matches("mov.f32 %f").count(), 4, "{ptx}");
}

/// A loop that carries values has no value to leave it with, so breaking out of
/// one is refused.
#[test]
fn a_loop_that_carries_values_may_not_be_left() {
    let why = refused(|b, source, out| {
        let zero = b.float(Scalar::F32, 0.0);
        let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: f32(),
                init: zero,
                into: None,
            }],
            move |b, _, given| {
                b.break_();
                let read = f32_load(b, source.clone());
                (vec![b.add(given[0].clone(), read)], ())
            },
        );
        let total = b
            .place_counted_loop(Trip::PerThread, start, stop, step, body)
            .into_iter()
            .next()
            .unwrap();
        b.store(out, total);
    });
    assert!(why.contains("no value to leave it with"), "{why}");
}

/// A running loop writes its index where no step shows, so a binding taking it
/// over would be overwritten between iterations.
#[test]
fn a_binding_may_not_take_over_a_running_loops_index() {
    let why = refused(|b, _, out| {
        let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
        let (body, ()) = b.build_loop_body(Scalar::S32, Vec::new(), |b, index, _| {
            let one = b.i32(1);
            let shifted = b.add(index.clone(), one);
            b.update(index, shifted);
            (Vec::new(), ())
        });
        b.place_counted_loop(Trip::PerThread, start, stop, step, body);
        let zero = b.float(Scalar::F32, 0.0);
        b.store(out, zero);
    });
    assert!(
        why.contains("the index a running loop counts with"),
        "{why}"
    );
}
