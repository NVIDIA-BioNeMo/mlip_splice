// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which storage a produced value is given: when a result may run in the
//! registers of the place it names, and when naming one is not enough.

use super::{f32, f32_load, io_kernel, lane_below_four, render};
use crate::build::Builder;
use crate::{Carry, Scalar, Trip, Ty};

/// A guarded fold hands its place back: the other arm yields the accumulator
/// unchanged, so the branch writes that storage whichever way it goes and the
/// result runs in the register the accumulator already runs in. No default is
/// written before the branch and no copy closes the arm — the fold is the one
/// instruction it reads and writes in place.
#[test]
fn a_result_folded_back_into_its_place_runs_in_that_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let held = acc.clone();
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                {
                    let held = held.clone();
                    move |b| {
                        let one = b.float(Scalar::F32, 1.0);
                        vec![b.add(held.clone(), one)]
                    }
                },
                move |_| vec![held.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<2>;"), "{ptx}");
    assert!(!ptx.contains("mov.f32"), "the join copies: {ptx}");
    assert!(ptx.contains("add.rn.f32 %f1, %f1,"), "{ptx}");
    assert!(ptx.contains("st.global.f32 [%rd4], %f1;"), "{ptx}");
}

/// A place read once the branch has produced into it is read wherever that read
/// stands: later in the same list, or past the region the branch sits in. The
/// value that reached the branch is what such a read means, so the result may
/// not run in that place's register and the place keeps what it held.
#[test]
fn a_place_read_after_the_branch_keeps_the_value_that_reached_it() {
    let folded = |past_the_region: bool| {
        io_kernel(move |b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let sink = out.clone();
            let held = acc.clone();
            let fold = move |b: &mut Builder| {
                let pred = lane_below_four(b);
                let inner = held.clone();
                let back = held.clone();
                let produced = b.if_producing(
                    pred,
                    &[(f32(), Some(held.clone()))],
                    move |b| {
                        let one = b.float(Scalar::F32, 1.0);
                        vec![b.add(inner.clone(), one)]
                    },
                    move |_| vec![back.clone()],
                );
                b.store(sink.clone(), produced.into_iter().next().unwrap());
            };
            if past_the_region {
                let outer = lane_below_four(b);
                b.if_(outer, fold);
            } else {
                fold(b);
            }
            // Names the value that reached the branch, not what it produced.
            b.store(out, acc);
        })
    };
    for past_the_region in [false, true] {
        let ptx = render(&folded(past_the_region), 120);
        // The fold runs in a register of its own and the place is untouched, so
        // the store after it reads what was loaded.
        assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
        assert!(ptx.contains("mov.f32 %f2, %f1;"), "{ptx}");
        assert!(ptx.contains("add.rn.f32 %f2, %f1,"), "{ptx}");
        let last = ptx
            .lines()
            .rfind(|line| line.contains("st.global.f32"))
            .expect("the store after the branch");
        assert!(last.ends_with("%f1;"), "{ptx}");
    }
}

/// The arm is walked in the order it runs, not read off its yields: a step that
/// reads a place another step of the arm has already written would see the
/// storage as the branch left it, where what it names is the value that reached
/// the branch.
#[test]
fn an_arm_that_reads_a_place_it_already_wrote_keeps_its_own_register() {
    let arm = |nested: bool| {
        io_kernel(move |b, source, out| {
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
                    // The write, plainly or under a region of its own.
                    let bound = if nested {
                        let guard = lane_below_four(b);
                        let (inner, up) = (write.clone(), up.clone());
                        b.if_(guard, move |b| {
                            b.update(inner.clone(), up.clone());
                        });
                        write.clone()
                    } else {
                        b.update(write.clone(), up)
                    };
                    // And a step of the arm reading the place after it.
                    b.store(sink.clone(), stale.clone());
                    vec![bound]
                },
                move |_| vec![back.clone()],
            );
            b.store(out, produced.into_iter().next().unwrap());
        })
    };
    for nested in [false, true] {
        let ptx = render(&arm(nested), 120);
        assert!(
            ptx.contains(".reg .f32 %f<3>;"),
            "the arm folded in place: {ptx}"
        );
        assert!(ptx.contains("mov.f32 %f2, %f1;"), "{ptx}");
    }
}

/// An arm that replaces the place rather than folding into it does not hand it
/// back, so the default written before the branch would clobber what the other
/// arm reads. The result gets its own register there.
#[test]
fn a_result_that_replaces_the_place_it_reads_gets_its_own_register() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let acc = b.bind_value_as("acc", f32(), read);
            let pred = lane_below_four(b);
            let held = acc.clone();
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(acc.clone()))],
                move |b| {
                    let one = b.float(Scalar::F32, 1.0);
                    vec![b.add(held.clone(), one)]
                },
                |b| vec![b.float(Scalar::F32, 0.0)],
            );
            b.store(out, produced.into_iter().next().unwrap());
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<3>;"), "{ptx}");
    assert!(ptx.contains("mov.f32 %f2, 0f00000000;"), "{ptx}");
    assert!(ptx.contains("add.rn.f32 %f2, %f1,"), "{ptx}");
}

/// Several results at once are written in the order they are handed back, so a
/// later one may not read a place an earlier one has already been produced
/// into. Where one does, none takes its place over.
#[test]
fn results_that_read_each_others_places_keep_their_own_registers() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let read = f32_load(b, source);
            let first = b.bind_value_as("first", f32(), read.clone());
            let second = b.bind_value_as("second", f32(), read);
            let pred = lane_below_four(b);
            let (a, c) = (first.clone(), second.clone());
            let produced = b.if_producing(
                pred,
                &[(f32(), Some(first.clone())), (f32(), Some(second.clone()))],
                {
                    let (a, c) = (a.clone(), c.clone());
                    move |b| {
                        let one = b.float(Scalar::F32, 1.0);
                        let up = b.add(a.clone(), one);
                        // The second result reads the first's place, which the
                        // join has already written by the time it lands.
                        let swapped = b.add(a.clone(), c.clone());
                        vec![up, swapped]
                    }
                },
                move |_| vec![a.clone(), c.clone()],
            );
            let sum = b.add(produced[0].clone(), produced[1].clone());
            b.store(out, sum);
        }),
        120,
    );
    // Two fresh registers for the results, both written before the branch.
    assert!(ptx.contains(".reg .f32 %f<6>;"), "{ptx}");
    assert_eq!(ptx.matches("mov.f32").count(), 2, "{ptx}");
}

/// A place read once the region holding the `if` has ended is still live, and
/// the result may not be allocated over it — the later reader would see what
/// the arms wrote. Read only through the result, the place is dead from the
/// branch on and the result runs in its register.
#[test]
fn a_result_leaves_a_place_alive_past_its_region_alone() {
    let guarded = |read_the_place: bool| {
        io_kernel(move |b, source, out| {
            let one = b.float(Scalar::F32, 1.0);
            let acc = b.bind_value_as("acc", f32(), one);
            let outer = lane_below_four(b);
            let held = acc.clone();
            let sink = out.clone();
            b.if_(outer, move |b| {
                let inner = lane_below_four(b);
                let source = source.clone();
                let produced = b.if_producing(
                    inner,
                    &[(f32(), Some(held.clone()))],
                    move |b| vec![f32_load(b, source.clone())],
                    |b| vec![b.float(Scalar::F32, 0.0)],
                );
                if !read_the_place {
                    b.store(sink.clone(), produced.into_iter().next().unwrap());
                }
            });
            if read_the_place {
                b.store(out, acc);
            }
        })
    };
    let alive = render(&guarded(true), 120);
    assert!(alive.contains(".reg .f32 %f<3>;"), "{alive}");
    let dead = render(&guarded(false), 120);
    assert!(dead.contains(".reg .f32 %f<2>;"), "{dead}");
}

/// Inside a loop the body runs again, so a place read before the `if` is read
/// after it too — once around the backedge.
#[test]
fn a_result_in_a_loop_body_respects_the_backedge() {
    let looped = |read_the_place: bool| {
        io_kernel(move |b, source, out| {
            let one = b.float(Scalar::F32, 1.0);
            let acc = b.bind_value_as("acc", f32(), one);
            let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
            let held = acc.clone();
            let (body, ()) = b.build_loop_body(Scalar::S32, Vec::new(), move |b, _, _| {
                if read_the_place {
                    b.store(out.clone(), held.clone());
                }
                let pred = lane_below_four(b);
                let source = source.clone();
                let produced = b.if_producing(
                    pred,
                    &[(f32(), Some(held.clone()))],
                    move |b| vec![f32_load(b, source.clone())],
                    |b| vec![b.float(Scalar::F32, 0.0)],
                );
                if !read_the_place {
                    b.store(out.clone(), produced.into_iter().next().unwrap());
                }
                (Vec::new(), ())
            });
            b.place_counted_loop(Trip::PerThread, start, stop, step, body);
        })
    };
    // Read before the `if`: the place is live across the backedge, so the
    // result gets its own register.
    let before = render(&looped(true), 120);
    assert!(before.contains(".reg .f32 %f<3>;"), "{before}");
    // Read only through the result: the result stands for the place from there
    // on, and takes it over.
    let after = render(&looped(false), 120);
    assert!(after.contains(".reg .f32 %f<2>;"), "{after}");
}

/// A result of a different type cannot share the registers of the place it
/// names, whatever the liveness says.
#[test]
fn a_result_of_another_type_gets_its_own_register() {
    let ptx = render(
        &io_kernel(|b, _, out| {
            let one = b.float(Scalar::F32, 1.0);
            let acc = b.bind_value_as("acc", f32(), one);
            let pred = lane_below_four(b);
            let wide = Ty::vec(Scalar::F32, 2);
            let produced = b.if_producing(
                pred,
                &[(wide, Some(acc))],
                |b| {
                    let two = b.float(Scalar::F32, 2.0);
                    vec![b.splat(2, two)]
                },
                |b| {
                    let three = b.float(Scalar::F32, 3.0);
                    vec![b.splat(2, three)]
                },
            );
            let value = produced.into_iter().next().unwrap();
            let lane = b.extract(0, value);
            b.store(out, lane);
        }),
        120,
    );
    assert!(ptx.contains(".reg .f32 %f<4>;"), "{ptx}");
}

/// An operand is evaluated where it is read, so a fold read twice runs twice.
/// Giving its result storage would make the second read free, which is the
/// opposite of what an unscheduled operand says.
#[test]
fn a_fold_read_twice_is_emitted_twice() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let start = b.float(Scalar::F32, 0.0);
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: f32(),
                    init: start,
                    into: None,
                }],
                move |b, index, given| {
                    let at = b.offset(source.clone(), index);
                    let read = b.load(at, 1, true);
                    (vec![b.add(given[0].clone(), read)], ())
                },
            );
            let (from, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
            let folded = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let folded = folded.into_iter().next().unwrap();
            let doubled = b.add(folded.clone(), folded);
            b.store(out, doubled);
        }),
        120,
    );
    assert_eq!(
        ptx.lines()
            .filter(|line| line.starts_with("$Lloop_") && line.ends_with(':'))
            .count(),
        2,
        "the fold runs once per read: {ptx}"
    );
    assert_eq!(
        ptx.matches("ld.global.nc.f32").count(),
        2,
        "and so does what it gathers: {ptx}"
    );
}

/// A loop that is a step of the body produces places, so what it carried is
/// read from the registers it left them in — once, however often it is read.
#[test]
fn a_placed_loops_results_are_read_from_their_registers() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let start = b.float(Scalar::F32, 0.0);
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: f32(),
                    init: start,
                    into: None,
                }],
                move |b, index, given| {
                    let at = b.offset(source.clone(), index);
                    let read = b.load(at, 1, true);
                    (vec![b.add(given[0].clone(), read)], ())
                },
            );
            let (from, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
            let carried = b.place_counted_loop(Trip::PerThread, from, stop, step, body);
            let carried = carried.into_iter().next().unwrap();
            let doubled = b.add(carried.clone(), carried);
            b.store(out, doubled);
        }),
        120,
    );
    assert_eq!(
        ptx.lines()
            .filter(|line| line.starts_with("$Lloop_") && line.ends_with(':'))
            .count(),
        1,
        "a placed loop runs once: {ptx}"
    );
}

/// One lowering, one pair of stems.
///
/// A loop is a loop whether the body runs it as a step or reads it for the
/// value it reduces to, and the labels say so. The stems are part of the text a
/// program's identity is taken over, so they are pinned here rather than left
/// to whichever emitter happened to print them.
#[test]
fn a_loop_labels_its_top_and_its_end_the_same_way_placed_or_read() {
    let counted = |placed: bool| {
        render(
            &io_kernel(move |b, source, out| {
                let start = b.float(Scalar::F32, 0.0);
                let (body, ()) = b.build_loop_body(
                    Scalar::S32,
                    vec![Carry {
                        ty: f32(),
                        init: start,
                        into: None,
                    }],
                    move |b, index, given| {
                        let at = b.offset(source.clone(), index);
                        let read = b.load(at, 1, true);
                        (vec![b.add(given[0].clone(), read)], ())
                    },
                );
                let (from, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
                let folded = if placed {
                    b.place_counted_loop(Trip::PerThread, from, stop, step, body)
                } else {
                    b.fold_loop(Trip::PerThread, from, stop, step, body)
                };
                b.store(out, folded.into_iter().next().unwrap());
            }),
            120,
        )
    };
    for placed in [true, false] {
        let ptx = counted(placed);
        assert!(ptx.contains("$Lloop_1:"), "placed={placed}: {ptx}");
        assert!(ptx.contains("bra $Lloop_1;"), "placed={placed}: {ptx}");
        assert!(ptx.contains("$Ldone_2:"), "placed={placed}: {ptx}");
        assert!(ptx.contains("bra $Ldone_2;"), "placed={placed}: {ptx}");
    }
}
