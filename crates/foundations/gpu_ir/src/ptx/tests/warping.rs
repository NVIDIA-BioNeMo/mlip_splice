// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which lanes of a warp are there, and reading what one of them holds.

use super::{kernel, ptr, render};
use crate::func::Launch;
use crate::{Scalar, Ty};

/// Reads one `elem`, shuffles it out of the lane `source` names over the lanes
/// that are taking part, and stores what came back.
fn shuffled(elem: Scalar, source: i64) -> Launch {
    kernel(move |b| {
        let read = ptr(b, "in", elem);
        let out = ptr(b, "out", elem);
        let held = b.load(read, 1, false);
        let held = b.bind_value_as("held", Ty::scalar(elem), held);
        let mask = b.active_mask();
        let lane = b.int(Scalar::U32, source);
        let moved = b.shfl_from(held, lane, mask);
        b.store(out, moved);
    })
}

/// The register the mask was bound into, which the shuffles must name.
fn mask_register(ptx: &str) -> String {
    let line = ptx
        .lines()
        .find(|line| line.contains("activemask.b32"))
        .expect("one active mask");
    line.trim()
        .trim_start_matches("activemask.b32")
        .trim()
        .trim_end_matches(';')
        .to_string()
}

#[test]
fn an_active_mask_is_one_instruction_and_one_word() {
    let ptx = render(&shuffled(Scalar::F32, 3), 90);
    assert_eq!(ptx.matches("activemask.b32").count(), 1, "{ptx}");
    assert!(mask_register(&ptx).starts_with("%r"), "{ptx}");
}

/// The mask a shuffle moves between is the one the program stated, and the lane
/// it reads is the one it named — neither is a default the backend supplies.
#[test]
fn a_shuffle_names_the_lane_it_reads_and_the_lanes_that_take_part() {
    let ptx = render(&shuffled(Scalar::F32, 7), 90);
    let mask = mask_register(&ptx);
    let shuffles = ptx
        .lines()
        .filter(|line| line.contains("shfl.sync.idx.b32"))
        .collect::<Vec<_>>();
    assert_eq!(shuffles.len(), 1, "{ptx}");
    assert!(shuffles[0].ends_with(&format!(", 7, 31, {mask};")), "{ptx}");
    assert!(!ptx.contains("0xffffffff"), "no default mask: {ptx}");
    assert!(!ptx.contains("shfl.sync.down"), "{ptx}");
}

/// A shuffle moves 32 bits, so a value rides as many words as it occupies: one
/// for anything up to four bytes, two for eight.
#[test]
fn every_width_rides_the_words_it_occupies() {
    for (elem, words) in [
        (Scalar::S8, 1),
        (Scalar::U8, 1),
        (Scalar::F16, 1),
        (Scalar::BF16, 1),
        (Scalar::S32, 1),
        (Scalar::U32, 1),
        (Scalar::F32, 1),
        (Scalar::S64, 2),
        (Scalar::U64, 2),
        (Scalar::F64, 2),
    ] {
        let ptx = render(&shuffled(elem, 1), 90);
        assert_eq!(
            ptx.matches("shfl.sync.idx.b32").count(),
            words,
            "{elem:?}: {ptx}"
        );
    }
}

/// The shuffle Loom already emits is another operation, and adding this one
/// leaves it exactly where it was.
#[test]
fn a_shuffle_down_still_moves_over_the_whole_warp() {
    let k = kernel(|b| {
        let read = ptr(b, "in", Scalar::F32);
        let out = ptr(b, "out", Scalar::F32);
        let held = b.load(read, 1, false);
        let delta = b.i32(1);
        let moved = b.shfl_down(held, delta);
        b.store(out, moved);
    });
    let ptx = render(&k, 90);
    assert!(
        ptx.contains("shfl.sync.down.b32 %r2, %r1, 1, 31, 0xffffffff;"),
        "{ptx}"
    );
    assert!(!ptx.contains("activemask"), "{ptx}");
    assert!(!ptx.contains("shfl.sync.idx"), "{ptx}");
}
