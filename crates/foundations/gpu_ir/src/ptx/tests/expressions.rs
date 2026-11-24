// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Expression-tree evaluation order and depth.

use super::{kernel, ptr};
use crate::ptx::kernel::K;
use crate::{Carry, Scalar, Trip, Ty};

fn render_without_verification(kernel: &crate::func::Launch) -> String {
    let mut lower = K::new(kernel.body(), 120, String::new());
    let mut out = String::new();
    lower.kernel(kernel, &mut out);
    out
}

#[test]
fn deep_fma_operands_use_a_bounded_call_stack() {
    const DEPTH: usize = 20_000;
    for position in 0..3 {
        let kernel = kernel(|b| {
            let out = ptr(b, "out", Scalar::F32);
            let one = b.float(Scalar::F32, 1.0);
            let mut value = one.clone();
            for _ in 0..DEPTH {
                let mut inputs = [one.clone(), one.clone(), one.clone()];
                inputs[position] = value;
                let [left, right, addend] = inputs;
                value = b.fma(left, right, addend);
            }
            b.store(out, value);
        });
        let ptx = render_without_verification(&kernel);
        assert_eq!(ptx.matches("fma.rn.f32 ").count(), DEPTH);
    }
}

#[test]
fn each_operand_occurrence_is_evaluated_in_source_order() {
    let kernel = kernel(|b| {
        let source = ptr(b, "in", Scalar::F32);
        let out = ptr(b, "out", Scalar::F32);
        let loaded = b.load(source, 1, true);
        let sum = b.add(loaded.clone(), loaded.clone());
        let one = b.float(Scalar::F32, 1.0);
        let result = b.fma(one, loaded, sum);
        b.store(out, result);
    });

    let ptx = render_without_verification(&kernel);
    let operations = ptx
        .lines()
        .filter_map(|line| {
            ["ld.global.nc.f32", "add.rn.f32", "fma.rn.f32"]
                .into_iter()
                .find(|instruction| line.contains(instruction))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        operations,
        [
            "ld.global.nc.f32",
            "ld.global.nc.f32",
            "ld.global.nc.f32",
            "add.rn.f32",
            "fma.rn.f32",
        ],
        "{ptx}"
    );
}

#[test]
fn deep_selects_and_packs_use_a_bounded_call_stack() {
    const DEPTH: usize = 10_000;
    let kernel = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let one = b.float(Scalar::F32, 1.0);
        let zero = b.float(Scalar::F32, 0.0);
        let predicate = b.gt(one.clone(), zero.clone());
        let mut value = one;
        for _ in 0..DEPTH {
            value = b.select(predicate.clone(), value, zero.clone());
            let packed = b.pack(vec![value]);
            value = b.extract(0, packed);
        }
        b.store(out, value);
    });
    let ptx = render_without_verification(&kernel);
    assert_eq!(ptx.matches("selp.f32 ").count(), DEPTH);
}

#[test]
fn held_loads_are_read_once_while_unheld_loads_repeat() {
    for held in [false, true] {
        let kernel = kernel(|b| {
            let source = ptr(b, "in", Scalar::F32);
            let out = ptr(b, "out", Scalar::F32);
            let value = b.load(source, 1, true);
            let value = if held {
                b.bind_value_as("loaded", Ty::scalar(Scalar::F32), value)
            } else {
                value
            };
            let sum = b.add(value.clone(), value);
            b.store(out, sum);
        });
        let ptx = render_without_verification(&kernel);
        assert_eq!(
            ptx.matches("ld.global.nc.f32 ").count(),
            if held { 1 } else { 2 }
        );
    }
}

#[test]
fn shuffle_occurrences_read_source_lane_and_mask_in_order() {
    let kernel = kernel(|b| {
        let source = ptr(b, "in", Scalar::F32);
        let lane = ptr(b, "lane", Scalar::U32);
        let out = ptr(b, "out", Scalar::F32);
        let source = b.load(source, 1, true);
        let lane = b.load(lane, 1, true);
        let mask = b.active_mask();
        let moved = b.shfl_from(source, lane, mask);
        let sum = b.add(moved.clone(), moved);
        b.store(out, sum);
    });
    let ptx = render_without_verification(&kernel);
    let instructions = ptx
        .lines()
        .filter_map(|line| {
            [
                "ld.global.nc.f32",
                "ld.global.nc.b32",
                "activemask.b32",
                "shfl.sync.idx.b32",
                "add.rn.f32",
            ]
            .into_iter()
            .find(|instruction| line.contains(instruction))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        instructions,
        [
            "ld.global.nc.f32",
            "ld.global.nc.b32",
            "activemask.b32",
            "shfl.sync.idx.b32",
            "ld.global.nc.f32",
            "ld.global.nc.b32",
            "activemask.b32",
            "shfl.sync.idx.b32",
            "add.rn.f32",
        ]
    );
}

#[test]
fn warp_votes_are_emitted_for_each_unheld_occurrence() {
    let kernel = kernel(|b| {
        let source = ptr(b, "in", Scalar::F32);
        let out = ptr(b, "out", Scalar::F32);
        let value = b.load(source, 1, true);
        let zero = b.float(Scalar::F32, 0.0);
        let one = b.float(Scalar::F32, 1.0);
        let predicate = b.gt(value, zero.clone());
        let vote = b.warp_any(predicate);
        let a = b.select(vote.clone(), one.clone(), zero.clone());
        let c = b.select(vote, one, zero);
        let sum = b.add(a, c);
        b.store(out, sum);
    });
    let ptx = render_without_verification(&kernel);
    let instructions = ptx
        .lines()
        .filter_map(|line| {
            [
                "ld.global.nc.f32",
                "activemask.b32",
                "redux.sync.or.b32",
                "selp.f32",
                "add.rn.f32",
            ]
            .into_iter()
            .find(|instruction| line.contains(instruction))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        instructions,
        [
            "ld.global.nc.f32",
            "activemask.b32",
            "redux.sync.or.b32",
            "selp.f32",
            "ld.global.nc.f32",
            "activemask.b32",
            "redux.sync.or.b32",
            "selp.f32",
            "add.rn.f32",
        ]
    );
}

#[test]
fn each_unheld_loop_result_keeps_its_own_region() {
    let kernel = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let seed = b.float(Scalar::F32, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: seed,
                into: None,
            }],
            |b, _, given| {
                let one = b.float(Scalar::F32, 1.0);
                (vec![b.add(given[0].clone(), one)], ())
            },
        );
        let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
        let mut values = b.fold_loop(Trip::PerThread, start, stop, step, body);
        let value = values.remove(0);
        let sum = b.add(value.clone(), value);
        b.store(out, sum);
    });
    let ptx = render_without_verification(&kernel);
    let instructions = ptx
        .lines()
        .filter_map(|line| {
            if line.starts_with("$Lloop_") {
                Some("loop")
            } else if line.starts_with("$Ldone_") {
                Some("done")
            } else if line.contains("add.rn.f32") {
                Some("add")
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        instructions,
        ["loop", "add", "done", "loop", "add", "done", "add"]
    );
}
