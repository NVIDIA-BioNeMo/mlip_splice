// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn mixed_output_request(u: i64, may_have_unmapped: bool) -> crate::FlatProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("mixed_outputs");
    let weight = builder.param(Shape::new(DType::F32, [9]));
    let value = builder.param(Shape::new(DType::F32, [9]));
    let product = builder.multiply(weight, value);
    let negative = builder.op(xir::ops::Negate, &[product])[0];
    let doubled = builder.add(product, product);
    let main = builder.end_func(vec![product, negative, doubled]);
    let binding = |domain, lane_extent| FlatOperand {
        dtype: DType::F32,
        element_shape: vec![9],
        lane_extent,
        axis_domains: vec![domain],
    };
    crate::FlatProgram {
        name: "mixed_outputs".into(),
        module: builder.finish(main).unwrap(),
        lane_count: u,
        domains: vec![Domain { extent: 37 }, Domain { extent: 7 }],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped,
        }],
        num_axes: 1,
        operands: vec![
            binding(1, 1),
            binding(0, u),
            binding(0, u),
            binding(1, u),
            binding(1, 1),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

#[test]
fn mixed_lane_outputs_verify() {
    for u in [1, 4, 32, 35, 128] {
        for may_have_unmapped in [false, true] {
            let (mut frontier, _) = crate::frontier::ProgramFrontier::extract(
                mixed_output_request(u, may_have_unmapped),
                crate::CompileOptions::default(),
                &mut || Ok(()),
            )
            .unwrap();
            let mut verified = 0;
            let mut refusals = Vec::new();
            while let Some(result) = frontier.next(&mut || Ok(())) {
                match result {
                    Ok((loom, _)) => {
                        let kernel = emit_test_kernel(&loom);
                        gpu_ir::verify::verify_launch(&kernel).unwrap();
                        gpu_ir::dialect::print(kernel.function.body().unwrap()).unwrap();
                        verified += 1;
                    }
                    Err(error) => refusals.push(error),
                }
            }
            assert!(
                verified > 0,
                "u={u}, unmapped={may_have_unmapped}: {refusals:?}"
            );
        }
    }
}
