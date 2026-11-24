// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a loop carries and how results land in the storage they name: which
//! registers a body may be handed, what its steps may name from outside, and
//! the order the values a branch hands back are written in.

use super::{f32, f32_load, io_kernel, lane_below_four, render};
use crate::{Carry, Counted, Scalar, Trip, Ty};

/// An arm ending in exactly what a loop produces hands that loop its registers,
/// so the loop writes the slots whatever its carries name. A body step naming
/// the place from outside reads the running accumulator, so the branch is not
/// given that storage.
#[test]
fn a_handed_down_loop_writes_the_slots_whatever_its_carries_name() {
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
                    let (outer, sink) = (held.clone(), sink.clone());
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
                            init: held.clone(),
                            into: None,
                        }],
                        move |b, _, given| {
                            b.store(sink.clone(), outer.clone());
                            let one = b.float(Scalar::F32, 1.0);
                            vec![b.add(given[0].clone(), one)]
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
    // The body reads the place where it stands; the fold runs elsewhere.
    assert!(ptx.contains("st.global.f32 [%rd4], %f1;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f2, %f2,"), "{ptx}");
}

/// Carried values are initialized one after another into the registers they run
/// in. A starting value reading a register an earlier one lands in must be read
/// before that happens, so every starting value goes through a register of its
/// own and the carried ones are written together.
#[test]
fn crossed_starting_values_are_read_before_either_lands() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let one = b.float(Scalar::F32, 1.0);
            let a = b.bind_value_as("a", f32(), read);
            let c = b.bind_value_as("c", f32(), one);
            let pred = lane_below_four(b);
            let (ia, ic) = (a.clone(), c.clone());
            let (ba, bc) = (a.clone(), c.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(a.clone())), (f32(), Some(c.clone()))],
                move |b| {
                    let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                    // The second starts from a value reading the first's place,
                    // which is the register it lands in.
                    let crossed = b.add(ia.clone(), ic.clone());
                    b.counted_loop_producing(
                        Counted {
                            index: Scalar::S32,
                            start,
                            stop,
                            step,
                            trip: Trip::PerThread,
                        },
                        vec![
                            Carry {
                                ty: f32(),
                                init: ic.clone(),
                                into: None,
                            },
                            Carry {
                                ty: f32(),
                                init: crossed,
                                into: None,
                            },
                        ],
                        move |b, _, given| {
                            let two = b.float(Scalar::F32, 2.0);
                            let up = b.add(given[0].clone(), two.clone());
                            let down = b.add(given[1].clone(), two);
                            vec![up, down]
                        },
                    )
                },
                move |_| vec![ba.clone(), bc.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    let at = |needle: &str| {
        ptx.lines()
            .position(|line| line.trim() == needle)
            .unwrap_or_else(|| panic!("{needle} missing: {ptx}"))
    };
    // The crossed starting value is read before either carried register lands.
    let crossed = at("add.rn.f32 %f4, %f1, %f2;");
    assert!(crossed < at("mov.f32 %f1, %f3;"), "{ptx}");
    assert!(crossed < at("mov.f32 %f2, %f4;"), "{ptx}");
}

/// An arm ending in a nested branch hands that branch the slots' registers, and
/// the branch writes them at its own end one after another. Handing them back
/// crossed would land the first on what the second still reads, so neither
/// result is given its storage.
#[test]
fn a_nested_branch_handing_the_slots_back_crossed_keeps_its_own_registers() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let a = b.bind_value_as("a", f32(), read.clone());
            let c = b.bind_value_as("c", f32(), read);
            let pred = lane_below_four(b);
            let (ia, ic) = (a.clone(), c.clone());
            let (ba, bc) = (a.clone(), c.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(a.clone())), (f32(), Some(c.clone()))],
                move |b| {
                    let inner = lane_below_four(b);
                    let (ta, tc) = (ia.clone(), ic.clone());
                    let (ea, ec) = (ia.clone(), ic.clone());
                    b.if_producing(
                        inner,
                        &[(f32(), None), (f32(), None)],
                        move |_| vec![tc.clone(), ta.clone()],
                        move |_| vec![ea.clone(), ec.clone()],
                    )
                },
                move |_| vec![ba.clone(), bc.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<6>;"), "{ptx}");
    // The crossed hand-back reads the places where they stand.
    assert!(ptx.contains("mov.f32 %f3, %f2;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f4, %f1;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f5, %f3, %f4;"), "{ptx}");
}

/// A value is read where it stands, and what it takes to evaluate it happens
/// there too. An arm handing back a fold whose body writes another destination
/// would write that storage while landing the first result, so neither result
/// is given the storage it names.
#[test]
fn an_arm_yielding_a_fold_that_writes_a_sibling_keeps_its_own_registers() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let a = b.bind_value_as("a", f32(), read.clone());
            let c = b.bind_value_as("c", f32(), read);
            let pred = lane_below_four(b);
            let (ia, ic) = (a.clone(), c.clone());
            let (ba, bc) = (a.clone(), c.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(a.clone())), (f32(), Some(c.clone()))],
                move |b| {
                    let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                    let inner = ic.clone();
                    let (body, ()) = b.build_loop_body(
                        Scalar::S32,
                        vec![Carry {
                            ty: f32(),
                            init: ia.clone(),
                            into: None,
                        }],
                        move |b, _, given| {
                            // The fold's body writes the other destination.
                            let one = b.float(Scalar::F32, 1.0);
                            let bumped = b.add(inner.clone(), one.clone());
                            b.update(inner.clone(), bumped);
                            let up = b.add(given[0].clone(), one);
                            (vec![up], ())
                        },
                    );
                    let folded = b.fold_loop(Trip::PerThread, from, stop, step, body);
                    vec![folded.into_iter().next().unwrap(), ic.clone()]
                },
                move |_| vec![ba.clone(), bc.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<7>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f3, %f1;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f4, %f2;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f6, %f3, %f4;"), "{ptx}");
}

/// Evaluating the predicate may itself write a place. It runs between the
/// default and the arms, so a result given that storage would hold what the
/// predicate left whichever way the branch went. Reading a place is something
/// the arms do too and disqualifies nothing on its own; writing one disqualifies
/// every way of taking it over.
#[test]
fn a_predicate_that_writes_the_place_leaves_no_result_in_it() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = {
                let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                let inner = acc.clone();
                let start = lane_below_four(b);
                let (body, ()) = b.build_loop_body(
                    Scalar::S32,
                    vec![Carry {
                        ty: Ty::scalar(Scalar::Pred),
                        init: start,
                        into: None,
                    }],
                    move |b, _, given| {
                        let one = b.float(Scalar::F32, 1.0);
                        let up = b.add(inner.clone(), one);
                        b.update(inner.clone(), up);
                        (vec![given[0].clone()], ())
                    },
                );
                b.fold_loop(Trip::PerThread, from, stop, step, body)
                    .into_iter()
                    .next()
                    .unwrap()
            };
            let (held, back) = (acc.clone(), acc.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    vec![b.add(held.clone(), one)]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    // What the other arm hands back is copied out before the predicate runs.
    let at = |needle: &str| {
        ptx.lines()
            .position(|line| line.trim().starts_with(needle))
            .unwrap_or_else(|| panic!("{needle} missing: {ptx}"))
    };
    assert!(at("mov.f32 %f2, %f1;") < at("$Lloop_"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f2, %f1,"), "{ptx}");
    let last = ptx
        .lines()
        .rfind(|line| line.contains("st.global.f32"))
        .expect("the store");
    assert!(last.ends_with("%f2;"), "{ptx}");
}

/// A yield may write a carried register on its way to a value. The register it
/// lands in is written at the backedge, so what it wrote on the way must be its
/// own, and the landings must come after everything that reads them.
#[test]
fn a_yield_that_writes_a_carried_register_writes_one_of_its_own() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let one = b.float(Scalar::F32, 1.0);
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
                        init: one,
                        into: None,
                    },
                ],
                move |b, _, given| {
                    let other = given[1].clone();
                    let (inner_from, inner_stop, inner_step) = (b.i32(0), b.i32(2), b.i32(1));
                    let (inner, ()) = b.build_loop_body(
                        Scalar::S32,
                        vec![Carry {
                            ty: f32(),
                            init: given[0].clone(),
                            into: None,
                        }],
                        move |b, _, held| {
                            // Writes the other carried value on the way.
                            let two = b.float(Scalar::F32, 2.0);
                            let up = b.add(other.clone(), two.clone());
                            b.update(other.clone(), up);
                            (vec![b.add(held[0].clone(), two)], ())
                        },
                    );
                    let folded =
                        b.fold_loop(Trip::PerThread, inner_from, inner_stop, inner_step, inner);
                    (
                        vec![folded.into_iter().next().unwrap(), given[1].clone()],
                        (),
                    )
                },
            );
            let carried = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let sum = b.add(carried[0].clone(), carried[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    let at = |needle: &str| {
        ptx.lines()
            .position(|line| line.trim().starts_with(needle))
            .unwrap_or_else(|| panic!("{needle} missing: {ptx}"))
    };
    // The write on the way lands in a register of its own, and the carried ones
    // are written only once the inner walk has ended.
    assert!(ptx.contains("add.rn.f32 %f5, %f2,"), "{ptx}");
    assert!(at("$Ldone_4:") < at("mov.f32 %f1, %f3;"), "{ptx}");
    assert!(at("$Ldone_4:") < at("mov.f32 %f2, %f6;"), "{ptx}");
}

/// Carried values take their storage over together or not at all. A loop naming
/// one place this region gave storage to and one from further out may take over
/// neither: the outer one outlives the loop, and half a loop running in place is
/// not the decision either.
#[test]
fn carried_values_take_their_places_over_together_or_not_at_all() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let root = b.bind_value_as("root", f32(), read);
            let pred = lane_below_four(b);
            let sink = out.clone();
            let outer = root.clone();
            b.if_(pred, move |b| {
                let one = b.float(Scalar::F32, 1.0);
                let local = b.bind_value_as("local", f32(), one);
                let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
                let (body, ()) = b.build_loop_body(
                    Scalar::S32,
                    vec![
                        Carry {
                            ty: f32(),
                            init: outer.clone(),
                            into: Some(outer.clone()),
                        },
                        Carry {
                            ty: f32(),
                            init: local.clone(),
                            into: Some(local.clone()),
                        },
                    ],
                    |b, _, given| {
                        let two = b.float(Scalar::F32, 2.0);
                        let a = b.add(given[0].clone(), two.clone());
                        let c = b.add(given[1].clone(), two);
                        (vec![a, c], ())
                    },
                );
                let done = b.fold_loop(Trip::PerThread, from, stop, step, body);
                let sum = b.add(done[0].clone(), done[1].clone());
                b.store(sink.clone(), sum);
            });
            b.store(out, root);
        }),
        120,
    );
    // Both carried values are copied into registers of their own: the local one
    // does not run in its place just because the outer one cannot.
    assert!(ptx.contains("mov.f32 %f3, %f1;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f4, %f2;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f5, %f3,"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f6, %f4,"), "{ptx}");
}
