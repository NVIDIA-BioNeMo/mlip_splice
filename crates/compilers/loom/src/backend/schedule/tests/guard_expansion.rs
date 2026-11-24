// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn additive_expansion_retains_distinct_exact_zero_regions() {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let left_predicate = builder.param(Shape::scalar(DType::Bool));
    let right_predicate = builder.param(Shape::scalar(DType::Bool));
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let left_polynomial = builder.add(x, y);
    let right_polynomial = builder.add(y, x);
    let left = builder.op1(Select, &[left_predicate, left_polynomial, zero]);
    let right = builder.op1(Select, &[right_predicate, right_polynomial, zero]);
    let output = builder.add(left, right);
    let main = builder.end_func(vec![output]);
    let loom = loom_program! {
        name: "expanded_guards".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![
            Domain {
                extent: 17,
            },
            Domain {
                extent: 5,
            },
        ],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: false,
        }],
        num_axes: 1,
        operands: vec![
            binding(Dtype::Bool, 0),
            binding(Dtype::Bool, 0),
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 1),
        ],
        num_inputs: 4,
        available_views: Vec::new(),
    };

    let plan = baseline_plan(&loom);
    let candidate = &plan.candidates[0];
    let guarded = candidate.guards[0]
        .as_ref()
        .expect("expanded exact-zero terms lost their guards");
    assert_eq!(guarded.regions.len(), 2);
    assert!(guarded.remainder.elements.iter().all(Vec::is_empty));

    let Launch::Consumer { part, .. } = &candidate.launches[0] else {
        panic!("guarded launch is not a consumer");
    };
    let kernel = emit_consumer_kernel_flavor(
        &plan.domain,
        &loom,
        part,
        Flavor {
            materialized: &candidate.materialized,
            guard: Some(guarded),
            io_lanes: 1,
            ..Flavor::plain(BLOCK_SIZES[0], 0, 1)
        },
    )
    .map(|consumer| consumer.kernel)
    .unwrap();
    assert_eq!(warp_guard_count(kernel.body()), 2);
}
