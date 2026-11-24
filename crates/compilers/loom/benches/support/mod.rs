// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{Domain, IndexMap};
use loom::{Lanes, Operand, Program};
use std::num::NonZeroUsize;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::shape::Shape;

const LANE: NonZeroUsize = NonZeroUsize::new(8).unwrap();

fn operand(element_shape: Vec<i64>, lanes: Lanes, axis_domain: Option<usize>) -> Operand {
    Operand {
        dtype: DType::F32,
        element_shape,
        lanes,
        axis_domains: vec![axis_domain],
    }
}

fn broadcast(positions: i64, elements: i64) -> Program {
    let lane = LANE.get() as i64;
    assert_eq!(elements % lane, 0);
    let element_shape = vec![elements / lane];
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::new(DType::F32, element_shape.clone()));
    let squared = builder.multiply(x, x);
    let main = builder.end_func(vec![squared]);
    let module = builder.finish(main).unwrap();
    Program {
        name: "broadcast_reduction".into(),
        module,
        lane_count: LANE,
        graph: domain_index::Graph {
            domains: vec![Domain { extent: positions }],
            index_maps: Vec::new(),
            aligned_views: Vec::new(),
        },
        inputs: vec![operand(element_shape.clone(), Lanes::PerLane, Some(0))],
        outputs: vec![operand(element_shape, Lanes::PerLane, None)],
    }
}

fn projection(coarse: i64, fine: i64) -> Program {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("projection_formula");
    let coefficient = builder.param(Shape::scalar(DType::F32));
    let values = builder.param(Shape::scalar(DType::F32));
    let coefficient = builder.add(coefficient, coefficient);
    let product = builder.multiply(coefficient, values);
    let entry = builder.end_func(vec![product]);
    let module = builder.finish(entry).unwrap();
    Program {
        name: "projection_formula".into(),
        module,
        lane_count: LANE,
        graph: domain_index::Graph {
            domains: vec![Domain { extent: coarse }, Domain { extent: fine }],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: true,
                may_have_unmapped: false,
            }],
            aligned_views: Vec::new(),
        },
        inputs: vec![
            operand(Vec::new(), Lanes::Shared, Some(0)),
            operand(Vec::new(), Lanes::PerLane, Some(1)),
        ],
        outputs: vec![operand(Vec::new(), Lanes::Shared, Some(0))],
    }
}

pub(crate) fn request(arguments: &[String]) -> (&str, Program) {
    let named = arguments.first().map(String::as_str).unwrap_or("broadcast");
    let size = |index: usize, fallback: i64| {
        arguments
            .get(index)
            .map(|given| given.parse().expect("a size is an integer"))
            .unwrap_or(fallback)
    };
    let request = match named {
        "broadcast" => broadcast(size(1, 128), size(2, 4096)),
        "projection" => projection(size(1, 3), size(2, 8)),
        other => panic!("no kernel named {other}: broadcast or projection"),
    };
    (named, request)
}

pub(crate) fn prepare() {
    cuda_toolchain::populate_from_python_for_tests();
    cuda_driver::ensure_context().expect("CUDA context");
}

#[allow(dead_code)]
pub(crate) fn compile(arguments: &[String]) -> loom::CompileProfile {
    let (_, request) = request(arguments);
    prepare();
    let (_, profile) = loom::compile(
        request,
        loom::CompileOptions {
            keep_ptx: true,
            ..Default::default()
        },
    )
    .expect("compile");
    profile
}
