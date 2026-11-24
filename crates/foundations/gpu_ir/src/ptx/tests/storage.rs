// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writing a value into storage that value also reads.
//!
//! A pack lands its parts in order, so a part reading a register an earlier one
//! writes would read what that part put there. These pin which packs are proven
//! safe to write straight through and which are read out of the way first.

use super::{f32, io_kernel, render};
use crate::build::{Builder, Value};
use crate::{Carry, Scalar, Trip, Ty};

/// The moves a body makes, as `(destination, source)` pairs in order.
fn moves(ptx: &str) -> Vec<(String, String)> {
    ptx.lines()
        .filter(|line| line.trim_start().starts_with("mov.f32"))
        .map(|line| {
            let operands = line
                .trim()
                .trim_end_matches(';')
                .split_once(' ')
                .unwrap()
                .1
                .split(", ")
                .map(str::to_owned)
                .collect::<Vec<_>>();
            (operands[0].clone(), operands[1].clone())
        })
        .collect()
}

/// The pair a two-lane store writes, as `(lane 0, lane 1)`.
fn stored_pair(ptx: &str) -> (String, String) {
    let store = ptx
        .lines()
        .find(|line| line.contains("st.global.v2"))
        .expect("a two-lane store");
    let held = store
        .rsplit_once('{')
        .expect("a lane list")
        .1
        .trim_end_matches(&['}', ';', ' '][..])
        .split(", ")
        .map(str::to_owned)
        .collect::<Vec<_>>();
    (held[0].clone(), held[1].clone())
}

/// Two lanes loaded into one place, for a test that then writes that place.
pub(super) fn two_lanes(b: &mut Builder, source: &Value) -> Value {
    let zero = b.i32(0);
    let one = b.i32(1);
    let first = b.offset(source.clone(), zero);
    let first = b.load(first, 1, true);
    let second = b.offset(source.clone(), one);
    let second = b.load(second, 1, true);
    let both = b.pack(vec![first, second]);
    b.bind_value_as("place", Ty::vec(Scalar::F32, 2), both)
}

/// Writing a pack writes its parts in order, so a part that reads a register an
/// earlier part writes has to be read before any of them move. A swap is the
/// smallest case: written straight through it collapses to `[b, b]`.
#[test]
fn a_pack_that_permutes_its_own_storage_is_staged_whole() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let high = b.extract(1, place.clone());
            let low = b.extract(0, place.clone());
            let swapped = b.pack(vec![high, low]);
            let swapped = b.update(place, swapped);
            b.store(out, swapped);
        }),
        120,
    );

    let (low, high) = stored_pair(&ptx);
    let moves = moves(&ptx);
    assert_eq!(moves.len(), 4, "both parts are read, then both land: {ptx}");
    // Every part first, into registers of its own, while the place still holds
    // what it came in with.
    let (first, second) = (moves[0].clone(), moves[1].clone());
    assert_eq!(first.1, high, "the first part reads lane 1");
    assert_eq!(second.1, low, "the second reads lane 0");
    for staged in [&first.0, &second.0] {
        assert!(
            *staged != low && *staged != high,
            "staged somewhere the place does not occupy: {ptx}"
        );
    }
    // Then the landings, in the same order.
    assert_eq!(moves[2], (low.clone(), first.0.clone()), "{ptx}");
    assert_eq!(moves[3], (high.clone(), second.0.clone()), "{ptx}");
}

/// Nothing is staged where every lane reads its own: a part landing where it
/// already sits, or one combining its own lane, writes straight through.
#[test]
fn a_pack_that_updates_one_lane_in_place_copies_nothing() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let two = b.i32(2);
            let addend = b.offset(source, two);
            let addend = b.load(addend, 1, true);
            let low = b.extract(0, place.clone());
            let high = b.extract(1, place.clone());
            let high = b.add(high, addend);
            let next = b.pack(vec![low, high]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );
    assert!(
        !ptx.contains("mov.f32"),
        "an in-place update copies nothing: {ptx}"
    );
    let combine = ptx
        .lines()
        .find(|line| line.contains("add.rn.f32"))
        .expect("the combine");
    let operands = combine
        .trim()
        .trim_end_matches(';')
        .split_once(' ')
        .unwrap()
        .1
        .split(", ")
        .collect::<Vec<_>>();
    assert_eq!(
        operands[0], operands[1],
        "the combined lane is written where it was read: {ptx}"
    );
}

/// Four lanes loaded into one place.
fn four_lanes(b: &mut Builder, source: &Value) -> Value {
    let read = (0..4)
        .map(|lane| {
            let at = b.i32(lane);
            let at = b.offset(source.clone(), at);
            b.load(at, 1, true)
        })
        .collect::<Vec<_>>();
    let all = b.pack(read);
    b.bind_value_as("place", Ty::vec(Scalar::F32, 4), all)
}

/// The lanes a four-lane store writes, in order.
fn stored_quad(ptx: &str) -> Vec<String> {
    let store = ptx
        .lines()
        .find(|line| line.contains("st.global.v4"))
        .expect("a four-lane store");
    store
        .rsplit_once('{')
        .expect("a lane list")
        .1
        .trim_end_matches(&['}', ';', ' '][..])
        .split(", ")
        .map(str::to_owned)
        .collect()
}

/// One part of a pack can be cross-lane inside itself: a broadcast puts one
/// lane in two, so lane 1 of that part reads a register lane 0 of it lands in.
/// A lane count says nothing about which lane, so the whole pack stages.
#[test]
fn a_pack_whose_part_is_cross_lane_inside_itself_is_staged_whole() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = four_lanes(b, &source);
            // Lanes 0 and 1 both become lane 2; lanes 2 and 3 become lanes 0
            // and 1. Written straight through, the second half would read what
            // the first half had already put there.
            let third = b.extract(2, place.clone());
            let spread = b.splat(2, third);
            let low = b.extract(0, place.clone());
            let high = b.extract(1, place.clone());
            let moved = b.pack(vec![low, high]);
            let next = b.pack(vec![spread, moved]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );
    let held = stored_quad(&ptx);
    let moves = moves(&ptx);
    // Both parts are read into registers of their own, then all four land.
    let landings = moves
        .iter()
        .position(|(destination, _)| held.contains(destination))
        .expect("the lanes land");
    let (staging, landing) = moves.split_at(landings);
    assert_eq!(landing.len(), 4, "four lanes land: {ptx}");
    for (destination, _) in staging {
        assert!(
            !held.contains(destination),
            "nothing lands while the parts are still being read: {ptx}"
        );
    }
    for (at, (destination, source)) in landing.iter().enumerate() {
        assert_eq!(*destination, held[at], "lane {at} lands in its own: {ptx}");
        assert!(
            !held.contains(source),
            "lane {at} lands from a register of its own: {ptx}"
        );
    }
}

/// A part whose region captures a destination register is not lane-local: what
/// a body reads is not a lane of anything, so the whole pack stages.
#[test]
fn a_pack_whose_part_holds_a_region_reading_the_destination_is_staged() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let low = b.extract(0, place.clone());
            let seed = b.float(Scalar::F32, 0.0);
            // A walk whose body reads lane 0 of the place it will be written
            // into. Landing lane 0 first would change what the walk sums.
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: f32(),
                    init: seed,
                    into: None,
                }],
                move |b, _, given| (vec![b.add(given[0].clone(), low.clone())], ()),
            );
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let walked = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let walked = walked.into_iter().next().unwrap();
            let high = b.extract(1, place.clone());
            let next = b.pack(vec![walked, high]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );
    let (low, _) = stored_pair(&ptx);
    // The walk runs before anything lands, and the sum it accumulates is not
    // the register it is going to be written into.
    let walk = ptx
        .lines()
        .position(|line| line.starts_with("$Lloop_"))
        .expect("the walk");
    let landing = ptx
        .lines()
        .position(|line| line.trim_start().starts_with(&format!("mov.f32 {low},")))
        .expect("lane 0 lands");
    assert!(walk < landing, "the walk runs before its lane lands: {ptx}");
    let combine = ptx
        .lines()
        .find(|line| line.contains("add.rn.f32"))
        .expect("the walk's sum");
    assert!(
        !combine.contains(&format!("{low},")),
        "the walk does not accumulate into the lane it will land in: {ptx}"
    );
}

/// A structured operand may name a place to produce into. Where that place is
/// the very storage the pack is landing in, staging it would write what the
/// other parts are still to read, so the place is reserved for the write and
/// the walk carries its accumulator somewhere else.
#[test]
fn a_staged_walk_does_not_write_the_place_the_pack_is_landing_in() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            // A walk over the place itself, naming it as what to produce into.
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: Ty::vec(Scalar::F32, 2),
                    init: place.clone(),
                    into: Some(place.clone()),
                }],
                move |b, index, given| {
                    let widened = b.cast(Scalar::F32, index);
                    let widened = b.splat(2, widened);
                    (vec![b.add(given[0].clone(), widened)], ())
                },
            );
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let walked = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let walked = walked.into_iter().next().unwrap();
            // The walk's two lanes, swapped, landing back in the place it named.
            let high = b.extract(1, walked.clone());
            let low = b.extract(0, walked);
            let next = b.pack(vec![high, low]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );
    let (low, high) = stored_pair(&ptx);
    // The walk carries its accumulator somewhere the pack is not landing.
    let carried = ptx
        .lines()
        .filter(|line| line.contains("add.rn.f32"))
        .map(|line| {
            line.split_whitespace()
                .nth(1)
                .unwrap()
                .trim_end_matches(',')
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert!(!carried.is_empty(), "the walk sums: {ptx}");
    for register in &carried {
        assert!(
            *register != low && *register != high,
            "the staged walk wrote a register the pack was still to read: {ptx}"
        );
    }
    // And the place is written only once the walk is over.
    let lines = ptx.lines().collect::<Vec<_>>();
    let done = lines
        .iter()
        .position(|line| line.starts_with("$Ldone_"))
        .expect("the walk ends");
    for (at, line) in lines.iter().enumerate() {
        let writes = line
            .split_whitespace()
            .nth(1)
            .map(|held| held.trim_end_matches(',').to_owned());
        if writes.is_some_and(|held| held == low || held == high) && !line.contains("ld.global") {
            assert!(
                at > done,
                "the place is written before the walk is over: {ptx}"
            );
        }
    }
}

/// Staging keeps the order the source holds: a part that loads runs where it
/// was written, not after a part that comes later.
#[test]
fn staging_a_pack_keeps_the_order_its_parts_were_written_in() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            // Lane 0 reads a third element; lane 1 permutes, which is what
            // forces the whole pack to stage.
            let three = b.i32(3);
            let far = b.offset(source, three);
            let far = b.load(far, 1, true);
            let low = b.extract(0, place.clone());
            let next = b.pack(vec![far, low]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );
    let (low, _) = stored_pair(&ptx);
    let lines = ptx.lines().collect::<Vec<_>>();
    let far = lines
        .iter()
        .rposition(|line| line.contains("ld.global.nc.f32"))
        .expect("the third element");
    let read_low = lines
        .iter()
        .position(|line| {
            line.trim_start().starts_with("mov.f32") && line.ends_with(&format!(", {low};"))
        })
        .expect("lane 0 read out of the way");
    let landing = lines
        .iter()
        .position(|line| line.trim_start().starts_with(&format!("mov.f32 {low},")))
        .expect("lane 0 lands");
    assert!(
        far < landing && read_low < landing,
        "every part is evaluated before any of them land: {ptx}"
    );
}

/// A capture two regions deep still counts. The part is a walk whose body
/// branches, and the branch's arm is what reads a landing register: an analysis
/// that stopped at the walk's own operands and yields would miss it and write
/// the lane the arm is still to read.
#[test]
fn a_pack_whose_part_captures_a_destination_two_regions_down_is_staged() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let low = b.extract(0, place.clone());
            let seed = b.float(Scalar::F32, 0.0);
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: f32(),
                    init: seed,
                    into: None,
                }],
                move |b, index, given| {
                    // The capture is inside the arm of a branch inside the
                    // walk's body — two regions below the part itself.
                    let two = b.i32(2);
                    let taken = b.lt(index, two);
                    let running = given[0].clone();
                    let stepped = b.if_value(
                        taken,
                        f32(),
                        {
                            let running = running.clone();
                            move |b| b.add(running.clone(), low.clone())
                        },
                        move |_| running.clone(),
                    );
                    (vec![stepped], ())
                },
            );
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let walked = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let walked = walked.into_iter().next().unwrap();
            // Landing at lane 1, so the capture of lane 0 is a register an
            // earlier part writes.
            let high = b.extract(1, place.clone());
            let next = b.pack(vec![high, walked]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );

    let (low, high) = stored_pair(&ptx);
    let lines = ptx.lines().collect::<Vec<_>>();
    let walk = lines
        .iter()
        .position(|line| line.starts_with("$Lloop_"))
        .expect("the walk");
    let landing = lines
        .iter()
        .position(|line| {
            let held = line.split_whitespace().nth(1);
            line.trim_start().starts_with("mov.f32")
                && held.is_some_and(|held| held.trim_end_matches(',') == low)
        })
        .expect("lane 0 lands");
    assert!(
        walk < landing,
        "the walk reads lane 0 before anything lands in it: {ptx}"
    );
    // And the arm accumulates somewhere neither lane occupies.
    let combine = lines
        .iter()
        .find(|line| line.contains("add.rn.f32"))
        .expect("the arm's sum");
    let accumulator = combine
        .split_whitespace()
        .nth(1)
        .unwrap()
        .trim_end_matches(',')
        .to_owned();
    assert!(
        accumulator != low && accumulator != high,
        "the walk accumulates in a register the pack is landing in: {ptx}"
    );
}

/// A step of a staged region may itself name a place to produce into. Where
/// that place is one the pack is landing in, the step gets storage of its own,
/// so the walk cannot mutate what the other parts are still to read.
#[test]
fn a_staged_regions_own_binding_does_not_write_a_reserved_place() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let (body, ()) = b.build_loop_body(
                Scalar::S32,
                vec![Carry {
                    ty: Ty::vec(Scalar::F32, 2),
                    init: place.clone(),
                    into: None,
                }],
                {
                    let place = place.clone();
                    move |b, index, given| {
                        let widened = b.cast(Scalar::F32, index);
                        let widened = b.splat(2, widened);
                        let stepped = b.add(given[0].clone(), widened);
                        // A step of the body produced into the very place the
                        // pack is landing in.
                        let held = b.update(place.clone(), stepped);
                        (vec![held], ())
                    }
                },
            );
            let (from, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
            let walked = b.fold_loop(Trip::PerThread, from, stop, step, body);
            let walked = walked.into_iter().next().unwrap();
            // A permutation, so the whole pack stages.
            let high = b.extract(1, walked.clone());
            let low = b.extract(0, walked);
            let next = b.pack(vec![high, low]);
            let next = b.update(place, next);
            b.store(out, next);
        }),
        120,
    );

    let (low, high) = stored_pair(&ptx);
    let lines = ptx.lines().collect::<Vec<_>>();
    let done = lines
        .iter()
        .rposition(|line| line.starts_with("$Ldone_"))
        .expect("the walks end");
    for (at, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with("mov.f32") {
            continue;
        }
        let written = line
            .split_whitespace()
            .nth(1)
            .unwrap()
            .trim_end_matches(',')
            .to_owned();
        if written == low || written == high {
            assert!(
                at > done,
                "a step inside the staged walk wrote a place the pack is landing in: {ptx}"
            );
        }
    }
}

/// Reserving is a stack that releases: a pack landing inside a body leaves the
/// bindings after it free to take their places over, exactly as they would with
/// no pack anywhere.
#[test]
fn a_binding_after_a_staged_pack_still_takes_its_place_over() {
    let ptx = render(
        &io_kernel(|b, source, out| {
            let place = two_lanes(b, &source);
            let high = b.extract(1, place.clone());
            let low = b.extract(0, place.clone());
            let swapped = b.pack(vec![high, low]);
            let swapped = b.update(place.clone(), swapped);
            // After the pack has landed, an update of the same place takes its
            // registers over as it always would.
            let one = b.float(Scalar::F32, 1.0);
            let one = b.splat(2, one);
            let stepped = b.add(swapped, one);
            let stepped = b.update(place, stepped);
            b.store(out, stepped);
        }),
        // Below the packed-pair form, so each lane's step reads as its own
        // instruction and the register it writes is plain to see.
        90,
    );
    let (low, high) = stored_pair(&ptx);
    for (held, lane) in [(&low, 0), (&high, 1)] {
        assert!(
            ptx.contains(&format!("add.rn.f32 {held}, {held},")),
            "lane {lane} was stepped somewhere other than its own register: {ptx}"
        );
    }
    assert_eq!(
        moves(&ptx).len(),
        4,
        "the swap staged and nothing else copied: {ptx}"
    );
}
