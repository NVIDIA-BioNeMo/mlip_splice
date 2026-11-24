// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Names and the registers they stand in are not the same thing. What a branch
//! may be given depends on which storage its arms write and when, whatever the
//! writes and reads happen to call it.

use super::{f32, f32_load, io_kernel, lane_below_four, render};
use crate::{Carry, Counted, Scalar, Trip, Ty};

/// A loop in the arm runs its body again, so a read standing before the step
/// that writes a place comes after it once around the backedge. The branch may
/// not be given that storage, whatever the body looks like read top to bottom.
#[test]
fn an_arm_whose_loop_rereads_what_it_wrote_keeps_its_own_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (held, back) = (acc.clone(), acc.clone());
            let sink = out.clone();
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                    let (inner, sink) = (held.clone(), sink.clone());
                    let (body, ()) = b.build_loop_body(Scalar::S32, Vec::new(), move |b, _, _| {
                        // Reads the place, then writes it: the next iteration
                        // reads what this one wrote.
                        b.store(sink.clone(), inner.clone());
                        let one = b.float(Scalar::F32, 1.0);
                        let up = b.add(inner.clone(), one);
                        b.update(inner.clone(), up);
                        (Vec::new(), ())
                    });
                    b.place_counted_loop(Trip::PerThread, start, stop, step, body);
                    vec![held.clone()]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32 %f2, %f1;").count(), 2, "{ptx}");
    let last = ptx
        .lines()
        .rfind(|line| line.contains("st.global.f32"))
        .expect("the store after the branch");
    assert!(last.ends_with("%f2;"), "{ptx}");
}

/// A step naming a place another step of the arm has already produced into
/// writes storage a live name stands for. Following the names cannot say what
/// that storage holds afterwards, so the branch is not given it.
#[test]
fn an_arm_writing_a_superseded_name_keeps_its_own_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (first, second, back) = (acc.clone(), acc.clone(), acc.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    let up = b.add(first.clone(), one);
                    let bound = b.update(first.clone(), up);
                    // Names the place again, not what the first one produced.
                    let two = b.float(Scalar::F32, 2.0);
                    b.update(second.clone(), two);
                    vec![bound]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32 %f2, %f1;").count(), 2, "{ptx}");
    let last = ptx
        .lines()
        .rfind(|line| line.contains("st.global.f32"))
        .expect("the store after the branch");
    assert!(last.ends_with("%f2;"), "{ptx}");
}

/// A region nested in the arm ends by handing back a place the arm has already
/// written. Its end is a read like any other, so the branch is not given that
/// storage — the value that reached it is what the nested end names.
#[test]
fn an_arm_whose_nested_region_hands_back_a_written_place_keeps_its_own_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (write, stale, back) = (acc.clone(), acc.clone(), acc.clone());
            let sink = out.clone();
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    let up = b.add(write.clone(), one);
                    let bound = b.update(write.clone(), up);
                    let guard = lane_below_four(b);
                    let stale = stale.clone();
                    let chosen = b.if_value(
                        guard,
                        f32(),
                        move |_| stale.clone(),
                        |b| b.float(Scalar::F32, 0.0),
                    );
                    b.store(sink.clone(), chosen);
                    vec![bound]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<4>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32 %f2, %f1;").count(), 2, "{ptx}");
    let last = ptx
        .lines()
        .rfind(|line| line.contains("st.global.f32"))
        .expect("the store after the branch");
    assert!(last.ends_with("%f2;"), "{ptx}");
}

/// Two results naming one place are two values in one register, and the second
/// would land on the first. Each keeps its own storage, and the arm's two
/// constants both survive to be read.
#[test]
fn results_naming_one_place_do_not_share_its_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (one, two) = (acc.clone(), acc.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone())), (f32(), Some(acc.clone()))],
                |b| {
                    let a = b.float(Scalar::F32, 1.0);
                    let c = b.float(Scalar::F32, 2.0);
                    vec![a, c]
                },
                move |_| vec![one.clone(), two.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains("mov.f32 %f2, 0f3F800000;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f3, 0f40000000;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f4, %f2, %f3;"), "{ptx}");
}

/// What a loop carries is storage like any other, so whether a branch may be
/// given it is the one decision the whole result set makes. Two results that
/// hand back each other's carried value would each land on what the other
/// starts from, so neither is given its register and the swap survives.
#[test]
fn results_that_swap_carried_values_keep_their_own_registers() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let other = b.float(Scalar::F32, 1.0);
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![
                    Carry {
                        ty: f32(),
                        init: read,
                        into: None,
                    },
                    Carry {
                        ty: f32(),
                        init: other,
                        into: None,
                    },
                ],
                |b, _, given| {
                    let pred = lane_below_four(b);
                    let (first, second) = (given[0].clone(), given[1].clone());
                    let (back, forth) = (first.clone(), second.clone());
                    let swapped = b.if_producing(
                        pred,
                        &[(f32(), Some(first.clone())), (f32(), Some(second.clone()))],
                        move |_| vec![forth.clone(), back.clone()],
                        move |_| vec![first.clone(), second.clone()],
                    );
                    (swapped, ())
                },
            );
            let carried = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let sum = b.add(carried[0].clone(), carried[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    // The arm writes the two results in order, so the second may not read what
    // the first has already landed on — that would hand back one value twice.
    let arm = ptx
        .lines()
        .map(str::trim)
        .skip_while(|line| !line.starts_with("@!%p2 bra"))
        .filter(|line| line.starts_with("mov.f32"))
        .map(|line| {
            let mut held = line.split_whitespace().skip(1);
            let destination = held.next().unwrap().trim_end_matches(',').to_owned();
            let source = held.next().unwrap().trim_end_matches(';').to_owned();
            (destination, source)
        })
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(arm.len(), 2, "the arm hands back two values: {ptx}");
    assert_ne!(
        arm[1].1, arm[0].0,
        "the second result reads the register the first landed on: {ptx}"
    );
    assert!(ptx.contains("add.rn.f32"), "{ptx}");
}

/// A place reserved for a write in progress is not taken over by a branch
/// producing into it: the parts still to be written would read what the arm
/// left rather than what the write is landing.
#[test]
fn a_branch_does_not_take_over_a_reserved_place() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = super::storage::two_lanes(b, &source);
            let held = place.clone();
            let guarded = {
                let inner = place.clone();
                let pred = lane_below_four(b);
                let back = place.clone();
                b.if_producing(
                    pred,
                    &[(Ty::vec(Scalar::F32, 2), Some(place.clone()))],
                    move |b| {
                        let one = b.float(Scalar::F32, 1.0);
                        let one = b.splat(2, one);
                        vec![b.add(inner.clone(), one)]
                    },
                    move |_| vec![back.clone()],
                )
            };
            let guarded = guarded.into_iter().next().unwrap();
            // A permutation, so the whole pack stages and the place is reserved
            // while the parts land.
            let high = b.extract(1, held.clone());
            let low = b.extract(0, guarded);
            let next = b.pack(vec![high, low]);
            let next = b.update(held, next);
            b.store(out, next);
        }),
        120,
    );
    // The place is %f1,%f2 and the branch runs in %f3,%f4: the default copies
    // the place in rather than the arm writing over it.
    assert!(ptx.contains("mov.f32 %f3, %f1;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f4, %f2;"), "{ptx}");
    assert!(
        ptx.contains("mov.b64 {%f3, %f4},"),
        "the arm wrote the reserved place: {ptx}"
    );
}

/// Storage read under an older name is read all the same. A binding produced
/// into a place stands in that place's register, so a branch producing into the
/// binding may not take that register over while anything still names the place.
#[test]
fn a_place_read_under_an_older_name_keeps_what_it_held() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let two = b.float(Scalar::F32, 2.0);
            let renamed = b.update(acc.clone(), two);
            let pred = lane_below_four(b);
            let (held, back, older) = (renamed.clone(), renamed.clone(), acc.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(renamed.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    vec![b.add(held.clone(), one)]
                },
                move |_| vec![back.clone()],
            );
            // Names that storage under the name it had before the rename.
            let sum = b.add(produced.into_iter().next().unwrap(), older);
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<4>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f2, %f1;"), "{ptx}");
    assert!(
        ptx.contains("add.rn.f32 %f2, %f1,"),
        "the branch took over storage an older name still reads: {ptx}"
    );
}

/// What a loop carries into a binding is that binding's storage under the name
/// the body has for it. A step of the body naming it from outside reads the
/// running accumulator, not what it held going in, so the branch is not given
/// that storage and the binding keeps a register of its own.
#[test]
fn a_body_naming_what_it_carries_from_outside_keeps_the_binding_apart() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (held, back) = (acc.clone(), acc.clone());
            let sink = out.clone();
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let zero = b.float(Scalar::F32, 0.0);
                    let opened = b.update(held.clone(), zero);
                    let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                    let (outer, sink) = (opened.clone(), sink.clone());
                    b.counted_loop_producing(
                        Counted {
                            index: Scalar::S32,
                            start,
                            stop,
                            step,
                            trip: Trip::PerThread,
                        },
                        vec![Carry {
                            ty: f32(),
                            init: opened.clone(),
                            into: Some(opened),
                        }],
                        move |b, _, given| {
                            b.store(sink.clone(), outer.clone());
                            let one = b.float(Scalar::F32, 1.0);
                            let up = b.add(given[0].clone(), one);
                            vec![b.update(given[0].clone(), up)]
                        },
                    )
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    // The body reads the binding where it stands, and the fold runs elsewhere.
    assert!(ptx.contains("st.global.f32 [%rd4], %f1;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f2, %f2,"), "{ptx}");
    let last = ptx
        .lines()
        .rfind(|line| line.contains("st.global.f32"))
        .expect("the store after the branch");
    assert!(last.ends_with("%f2;"), "{ptx}");
}

/// A branch under an arm inside a loop body still has the rest of that body
/// after it in the same pass. What the body reads there is the value that
/// reached the branch, so the branch is not given the carried register — the
/// backedge is what may read the fold, not this pass.
#[test]
fn a_branch_under_an_arm_leaves_the_rest_of_the_pass_alone() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let sink = out.clone();
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: f32(),
                    init: read,
                    into: None,
                }],
                move |b, _, given| {
                    let carried = given[0].clone();
                    let (held, back, later) = (carried.clone(), carried.clone(), carried.clone());
                    let outer = lane_below_four(b);
                    let inner_sink = sink.clone();
                    b.if_(outer, move |b| {
                        let pred = lane_below_four(b);
                        let got = b.if_producing(
                            pred,
                            &[(f32(), Some(held.clone()))],
                            {
                                let held = held.clone();
                                move |b| {
                                    let one = b.float(Scalar::F32, 1.0);
                                    vec![b.add(held.clone(), one)]
                                }
                            },
                            move |_| vec![back.clone()],
                        );
                        b.store(inner_sink.clone(), got.into_iter().next().unwrap());
                    });
                    b.store(sink.clone(), later.clone());
                    (vec![carried], ())
                },
            );
            let carried = b.fold_loop(Trip::PerThread, from, stop, step, body);
            b.store(out, carried.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f2, %f1;"), "{ptx}");
    assert!(
        ptx.contains("add.rn.f32 %f2, %f1,"),
        "the branch took the carried register: {ptx}"
    );
}

/// Results land in the storage they name one after another, and a copy moves
/// nothing: the value copied from stays where it was. A later result reading a
/// place an earlier one landed in would read what that landing left, so none of
/// them is given its storage and every yield still reads what it named.
#[test]
fn a_later_result_reads_what_an_earlier_landing_would_have_replaced() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let a = b.bind_value_as("a", f32(), read.clone());
            let older = b.bind_value_as("b", f32(), read.clone());
            let two = b.float(Scalar::F32, 2.0);
            let newer = b.update(older.clone(), two);
            let d = b.bind_value_as("d", f32(), read);
            let pred = lane_below_four(b);
            let (first, third) = (older.clone(), older.clone());
            let (ba, bb, bd) = (a.clone(), newer.clone(), d.clone());
            let produced = b.if_producing(
                pred,
                &[
                    (f32(), Some(a.clone())),
                    (f32(), Some(newer.clone())),
                    (f32(), Some(d.clone())),
                ],
                move |b| {
                    let c = b.float(Scalar::F32, 7.0);
                    vec![first.clone(), c, third.clone()]
                },
                move |_| vec![ba.clone(), bb.clone(), bd.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[2].clone());
            b.store(out, sum);
        }),
        120,
    );
    // Three registers of their own, and both yields of the older name read it
    // where it stands rather than one of them reading a landing.
    assert!(ptx.contains(".reg .f32 %f<8>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f4, %f2;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f5, 0f40E00000;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f6, %f2;"), "{ptx}");
}

/// A destination naming the storage under an older name writes what a live name
/// stands for, which the walk cannot follow — so the branch is not given it.
#[test]
fn an_arm_writing_through_an_older_name_keeps_its_own_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let (write, back) = (acc.clone(), acc.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    let up = b.add(write.clone(), one);
                    let renamed = b.update(write.clone(), up);
                    // Names that storage under the name it had before.
                    let two = b.float(Scalar::F32, 2.0);
                    b.update(write.clone(), two);
                    vec![renamed]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32 %f2, %f1;").count(), 2, "{ptx}");
}

/// Two destinations may be distinct values and one register all the same. Each
/// result still gets storage of its own, so neither lands on the other.
#[test]
fn results_naming_one_register_under_two_names_do_not_share_it() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let older = b.bind_value_as("acc", f32(), read);
            let two = b.float(Scalar::F32, 2.0);
            let newer = b.update(older.clone(), two);
            let pred = lane_below_four(b);
            let (first, second) = (older.clone(), newer.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(older.clone())), (f32(), Some(newer.clone()))],
                |b| {
                    let x = b.float(Scalar::F32, 1.0);
                    let y = b.float(Scalar::F32, 3.0);
                    vec![x, y]
                },
                move |_| vec![first.clone(), second.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<5>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f2, 0f3F800000;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f3, 0f40400000;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f4, %f2, %f3;"), "{ptx}");
}

/// A destination may be a different value standing in storage the arm already
/// wrote. It writes what a live name denotes all the same, so the branch is not
/// given that storage.
#[test]
fn an_arm_writing_through_a_different_name_for_written_storage_keeps_its_own() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let older = b.bind_value_as("acc", f32(), read);
            let two = b.float(Scalar::F32, 2.0);
            let newer = b.update(older.clone(), two);
            let pred = lane_below_four(b);
            let (write, alias, back) = (newer.clone(), older.clone(), newer.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(newer.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    let up = b.add(write.clone(), one);
                    let renamed = b.update(write.clone(), up);
                    // A different value naming the same register.
                    let three = b.float(Scalar::F32, 3.0);
                    b.update(alias.clone(), three);
                    vec![renamed]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32 %f2, %f1;").count(), 2, "{ptx}");
}
