// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::time::{Duration, Instant};

use weave::{Ir, OperationName, Operations, Type};

const OPERATIONS: usize = 100_000;

fn timed<T>(work: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let result = work();
    (result, start.elapsed())
}

fn main() {
    let name = OperationName::new("bench.operation").unwrap();
    let ty = Type::opaque("i32").unwrap();
    let mut ir = Ir::new();
    let source = ir.create_operation(name.clone(), [], [ty.clone()]).unwrap();
    let replacement = ir.create_operation(name.clone(), [], [ty.clone()]).unwrap();
    let old = ir.operation(source).unwrap().results[0];
    let new = ir.operation(replacement).unwrap().results[0];
    let root = ir.create_operation(name.clone(), [], []).unwrap();
    ir.add_root(root).unwrap();
    let region = ir.append_region(root).unwrap();
    let block = ir.append_block(region, []).unwrap();
    ir.append_operation(block, source).unwrap();
    ir.append_operation(block, replacement).unwrap();

    let (_, build) = timed(|| {
        for _ in 0..OPERATIONS {
            let operation = ir.create_operation(name.clone(), [old], []).unwrap();
            ir.append_operation(block, operation).unwrap();
            black_box(operation);
        }
    });
    let (_, index) = timed(|| black_box(ir.uses().of_value(old.slot() as usize).len()));
    let (_, cached) = timed(|| black_box(ir.uses().of_value(old.slot() as usize).len()));
    let (replaced, rauw) = timed(|| ir.replace_all_uses_with(old, new).unwrap());
    let ((frozen, _), freeze) = timed(|| ir.freeze_with_map().unwrap());

    assert_eq!(replaced, OPERATIONS);
    assert_eq!(frozen.operation_count(), OPERATIONS + 3);
    println!("operations={OPERATIONS}");
    println!(
        "build_ns_per_op={:.1}",
        build.as_nanos() as f64 / OPERATIONS as f64
    );
    println!("build_uses_ms={:.3}", index.as_secs_f64() * 1e3);
    println!("cached_uses_ns={}", cached.as_nanos());
    println!(
        "rauw_ns_per_use={:.1}",
        rauw.as_nanos() as f64 / OPERATIONS as f64
    );
    println!(
        "freeze_ns_per_op={:.1}",
        freeze.as_nanos() as f64 / OPERATIONS as f64
    );
}
