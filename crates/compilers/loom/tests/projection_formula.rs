// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_driver::DeviceBuffer;
use domain_index::{Domain, IndexMap, SlotKind};
use kiln::{Bindings, ResidentRecipe, execute, tune};
use std::num::NonZeroUsize;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::shape::Shape;

mod common;

use common::{default_stream, gpu_available};
use loom::{CompileOptions, Lanes, Operand, Program, compile};

const LANES: i64 = 4;
const TARGET_INDICES: [i32; 8] = [0, 0, 0, 1, 1, 1, 2, 2];
const PREIMAGE_OFFSETS: [i32; 4] = [0, 3, 6, 8];
const PREIMAGE_SOURCE_INDICES: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

fn request() -> Program {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("projection_formula");
    let coefficient = builder.param(Shape::scalar(DType::F32));
    let values = builder.param(Shape::scalar(DType::F32));
    let coefficient = builder.add(coefficient, coefficient);
    let product = builder.multiply(coefficient, values);
    let entry = builder.end_func(vec![product]);
    let module = builder.finish(entry).unwrap();
    let operand = |axis_domain, lanes| Operand {
        dtype: DType::F32,
        element_shape: Vec::new(),
        lanes,
        axis_domains: vec![Some(axis_domain)],
    };
    Program {
        name: "projection_formula".into(),
        module,
        lane_count: NonZeroUsize::new(LANES as usize).unwrap(),
        graph: domain_index::Graph {
            domains: vec![Domain { extent: 3 }, Domain { extent: 8 }],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: true,
                may_have_unmapped: false,
            }],
            aligned_views: Vec::new(),
        },
        inputs: vec![operand(0, Lanes::Shared), operand(1, Lanes::PerLane)],
        outputs: vec![operand(0, Lanes::Shared)],
    }
}

#[test]
#[serial_test::serial(gpu)]
fn coarse_factor_is_applied_after_domain_and_lane_sums() {
    if !gpu_available() {
        eprintln!("coarse_factor_is_applied_after_domain_and_lane_sums: no GPU, skipping");
        return;
    }
    let (compiled, _) = compile(request(), CompileOptions::default()).unwrap();
    let coefficients = [2.0f32, -3.0, 0.5];
    let values = (0..8 * LANES)
        .map(|index| (index as f32 - 9.0) / 7.0)
        .collect::<Vec<_>>();
    let ctx = cuda_driver::Context::current().raw();
    let coefficient_buffer = DeviceBuffer::from_host(ctx, &coefficients).unwrap();
    let value_buffer = DeviceBuffer::from_host(ctx, &values).unwrap();
    let mut slot_buffers = Vec::new();
    let mut inputs = vec![coefficient_buffer.ptr(), value_buffer.ptr()];
    for slot in &compiled.index_arrays {
        let arrays: Vec<&[i32]> = match slot.kind {
            SlotKind::TargetIndices => vec![&TARGET_INDICES],
            SlotKind::PreimageSorted => vec![&PREIMAGE_OFFSETS],
            SlotKind::Preimage => vec![&PREIMAGE_OFFSETS, &PREIMAGE_SOURCE_INDICES],
            other => panic!("unexpected projection-formula slot {other:?}"),
        };
        for array in arrays {
            slot_buffers.push(DeviceBuffer::from_host(ctx, array).unwrap());
            inputs.push(slot_buffers.last().unwrap().ptr());
        }
    }
    assert_eq!(inputs.len(), compiled.input_count);
    let output = DeviceBuffer::<f32>::alloc(ctx, coefficients.len()).unwrap();
    let workspace = DeviceBuffer::<u8>::alloc(
        ctx,
        recipe::algebra::workspace_upper_bound(&compiled.recipe),
    )
    .unwrap();
    let outputs = [output.ptr()];
    let bindings = Bindings {
        inputs: &inputs,
        outputs: &outputs,
        workspace: workspace.ptr(),
    };
    let flat = unsafe { tune(&compiled.recipe, &bindings, default_stream()) }.unwrap();
    let mut resident = unsafe { ResidentRecipe::new(ctx, flat) }.unwrap();
    unsafe { execute(&mut resident, &bindings, default_stream()) }.unwrap();
    let got = output.to_host().unwrap();

    for coarse in 0..coefficients.len() {
        let sum = TARGET_INDICES
            .iter()
            .enumerate()
            .filter(|(_, target)| **target as usize == coarse)
            .flat_map(|(fine, _)| &values[fine * LANES as usize..(fine + 1) * LANES as usize])
            .map(|&value| value as f64)
            .sum::<f64>();
        let expected = 2.0 * coefficients[coarse] as f64 * sum;
        assert!(
            (got[coarse] as f64 - expected).abs() <= 1e-5 * expected.abs().max(1.0),
            "coarse={coarse}: {} vs {expected}",
            got[coarse]
        );
    }
}
