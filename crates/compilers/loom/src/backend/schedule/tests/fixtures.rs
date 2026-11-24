// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn binding(dtype: Dtype, domain: i32) -> FlatOperand {
    FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![domain],
    }
}

pub(super) fn diamond(
    dtype: Dtype,
    xir_dtype: DType,
    domains: Vec<Domain>,
    index_maps: Vec<IndexMap>,
    output_domain: i32,
) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(xir_dtype));
    let y = builder.param(Shape::scalar(xir_dtype));
    let mut shared = builder.multiply(x, y);
    for _ in 0..12 {
        shared = builder.multiply(shared, y);
    }
    let left = builder.add(shared, x);
    let right = builder.add(shared, y);
    let main = builder.end_func(vec![left, right]);
    let module = builder.finish(main).unwrap();
    loom_program! {
        name: "plan_diamond".into(),
        module,
        lane_count: 8,
        domains,
        index_maps,
        num_axes: 1,
        operands: vec![
            binding(dtype, 0),
            binding(dtype, 0),
            binding(dtype, output_domain),
            binding(dtype, output_domain),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn asymmetric_additive_terms() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let mut left = builder.multiply(x, y);
    for _ in 0..24 {
        left = builder.multiply(left, y);
    }
    let right = builder.multiply(x, y);
    let result = builder.add(left, right);
    let main = builder.end_func(vec![result]);
    let module = builder.finish(main).unwrap();
    loom_program! {
        name: "asymmetric_additive_terms".into(),
        module,
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn shaped_binding(dtype: Dtype, shape: &[i64]) -> FlatOperand {
    FlatOperand {
        dtype,
        element_shape: shape.to_vec(),
        lane_extent: 8,
        axis_domains: vec![0],
    }
}

pub(super) fn pointwise_add_chain() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let mut result = builder.add(x, y);
    for _ in 0..24 {
        result = builder.add(result, x);
    }
    let main = builder.end_func(vec![result]);
    loom_program! {
        name: "pointwise_add_chain".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
            binding(Dtype::F32, 0),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn pointwise_multiply_add(guarded: bool) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let predicate = guarded.then(|| builder.param(Shape::scalar(DType::Bool)));
    let a = builder.param(Shape::scalar(DType::F32));
    let b = builder.param(Shape::scalar(DType::F32));
    let c = builder.param(Shape::scalar(DType::F32));
    let product = builder.multiply(a, b);
    let mut result = builder.add(product, c);
    if let Some(predicate) = predicate {
        let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
        result = builder.op1(Select, &[predicate, result, zero]);
    }
    let main = builder.end_func(vec![result]);
    let mut operands = Vec::new();
    if guarded {
        operands.push(binding(Dtype::Bool, 0));
    }
    operands.extend([
        binding(Dtype::F32, 0),
        binding(Dtype::F32, 0),
        binding(Dtype::F32, 0),
        binding(Dtype::F32, 0),
    ]);
    loom_program! {
        name: if guarded {
            "guarded_multiply_add"
        } else {
            "pointwise_multiply_add"
        }
        .into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands,
        num_inputs: 3 + usize::from(guarded),
        available_views: Vec::new(),
    }
}

pub(super) fn reduce_sum_program(dtype: Dtype, xir_dtype: DType, init: Literal) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let input = builder.param(Shape::new(xir_dtype, [32]));
    let init = builder.constant(init);
    let reducer = builder.func("add", |function| {
        let left = function.param(Shape::scalar(xir_dtype));
        let right = function.param(Shape::scalar(xir_dtype));
        vec![function.add(left, right)]
    });
    let result = builder.reduce(input, init, &[0], reducer);
    let main = builder.end_func(vec![result]);
    loom_program! {
        name: "reduce_sum".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![shaped_binding(dtype, &[32]), shaped_binding(dtype, &[])],
        num_inputs: 1,
        available_views: Vec::new(),
    }
}

pub(super) fn reduce_pointwise_sum_program(computed_init: bool) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let left = builder.param(Shape::new(DType::F32, [8]));
    let right = builder.param(Shape::new(DType::F32, [8]));
    let values = builder.add(left, right);
    let (init, num_inputs) = if computed_init {
        let init_left = builder.param(Shape::scalar(DType::F32));
        let init_right = builder.param(Shape::scalar(DType::F32));
        (builder.add(init_left, init_right), 4)
    } else {
        (
            builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap()),
            2,
        )
    };
    let reducer = builder.func("add", |function| {
        let left = function.param(Shape::scalar(DType::F32));
        let right = function.param(Shape::scalar(DType::F32));
        vec![function.add(left, right)]
    });
    let result = builder.reduce(values, init, &[0], reducer);
    let main = builder.end_func(vec![result]);
    let mut operands = vec![
        shaped_binding(Dtype::F32, &[8]),
        shaped_binding(Dtype::F32, &[8]),
    ];
    if computed_init {
        operands.extend([
            shaped_binding(Dtype::F32, &[]),
            shaped_binding(Dtype::F32, &[]),
        ]);
    }
    operands.push(shaped_binding(Dtype::F32, &[]));
    loom_program! {
        name: "reduce_pointwise_sum".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands,
        num_inputs,
        available_views: Vec::new(),
    }
}

pub(super) fn dot_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let left = builder.param(Shape::new(DType::F32, [32]));
    let right = builder.param(Shape::new(DType::F32, [32]));
    let result = builder.dot_general(
        left,
        right,
        DotDimensionNumbers::new(&[0], &[0], &[], &[]),
        DotAlgorithm::f32_f32_f32(),
    );
    let main = builder.end_func(vec![result]);
    loom_program! {
        name: "dot".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 3,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            shaped_binding(Dtype::F32, &[32]),
            shaped_binding(Dtype::F32, &[32]),
            shaped_binding(Dtype::F32, &[]),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn guarded_select_program(
    u: i64,
    zero_on_false: bool,
    num_outputs: usize,
    zero_by_multiply: bool,
    downstream_infinity: bool,
    fiber: bool,
) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let predicate = builder.param(Shape::scalar(DType::Bool));
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let inactive = if zero_by_multiply {
        builder.multiply(x, zero)
    } else {
        zero
    };
    let mut results = Vec::new();
    for output in 0..num_outputs {
        let mut expensive = builder.multiply(x, y);
        for _ in 0..8 + output {
            expensive = builder.multiply(expensive, y);
        }
        let selected = if zero_on_false {
            builder.op1(Select, &[predicate, expensive, inactive])
        } else {
            builder.op1(Select, &[predicate, inactive, expensive])
        };
        let selected = if downstream_infinity {
            let infinity = builder.constant(Literal::new(
                Shape::scalar(DType::F32),
                f32::INFINITY.to_le_bytes().to_vec(),
            ));
            builder.multiply(selected, infinity)
        } else {
            selected
        };
        results.push(selected);
    }
    let main = builder.end_func(results);
    let (domains, index_maps, output_domain) = if fiber {
        (
            vec![Domain { extent: 7 }, Domain { extent: 3 }],
            vec![IndexMap {
                source_id: 0,
                target_id: 1,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            1,
        )
    } else {
        (vec![Domain { extent: 3 }], Vec::new(), 0)
    };
    let lane_binding = |dtype, domain| FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: u,
        axis_domains: vec![domain],
    };
    let mut operands = vec![
        lane_binding(Dtype::Bool, 0),
        lane_binding(Dtype::F32, 0),
        lane_binding(Dtype::F32, 0),
    ];
    operands.extend((0..num_outputs).map(|_| lane_binding(Dtype::F32, output_domain)));
    loom_program! {
        name: "guarded_select".into(),
        module: builder.finish(main).unwrap(),
        lane_count: u,
        domains,
        index_maps,
        num_axes: 1,
        operands,
        num_inputs: 3,
        available_views: Vec::new(),
    }
}

pub(super) fn guarded_additive_program(num_terms: usize) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let predicate = builder.param(Shape::scalar(DType::Bool));
    let x = builder.param(Shape::scalar(DType::F32));
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let mut sum = zero;
    for _ in 0..num_terms {
        let term = builder.multiply(x, x);
        let guarded = builder.op1(Select, &[predicate, term, zero]);
        sum = builder.add(sum, guarded);
    }
    let main = builder.end_func(vec![sum]);
    let binding = |dtype| FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![0],
    };
    loom_program! {
        name: "guarded_additive".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 8,
        domains: vec![Domain {
            extent: 17,
        }],
        index_maps: Vec::new(),
        num_axes: 1,
        operands: vec![
            binding(Dtype::Bool),
            binding(Dtype::F32),
            binding(Dtype::F32),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn multi_guard_fiber_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let left_predicate = builder.param(Shape::scalar(DType::Bool));
    let right_predicate = builder.param(Shape::scalar(DType::Bool));
    let x = builder.param(Shape::scalar(DType::F32));
    let y = builder.param(Shape::scalar(DType::F32));
    let zero = builder.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
    let left_value = builder.multiply(x, x);
    let right_value = builder.multiply(y, y);
    let left = builder.op1(Select, &[left_predicate, left_value, zero]);
    let right = builder.op1(Select, &[right_predicate, right_value, zero]);
    let output = builder.add(left, right);
    let main = builder.end_func(vec![output]);
    let binding = |dtype, domain| FlatOperand {
        dtype,
        element_shape: Vec::new(),
        lane_extent: 8,
        axis_domains: vec![domain],
    };
    loom_program! {
        name: "multi_guard_fiber".into(),
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
    }
}

pub(super) fn lane_hoist_program(
    u: i64,
    output_width_one: bool,
    fiber: bool,
    computed_stage: bool,
) -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let narrow = builder.param(Shape::scalar(DType::F32));
    let wide = builder.param(Shape::scalar(DType::F32));
    let output = if computed_stage {
        let narrow_stage = builder.multiply(narrow, narrow);
        builder.multiply(narrow_stage, wide)
    } else {
        builder.multiply(narrow, wide)
    };
    let main = builder.end_func(vec![output]);
    let (domains, index_maps, output_domain) = if fiber {
        (
            vec![Domain { extent: 7 }, Domain { extent: 3 }],
            vec![IndexMap {
                source_id: 0,
                target_id: 1,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            1,
        )
    } else {
        (vec![Domain { extent: 3 }], Vec::new(), 0)
    };
    let binding = |lane_extent, domain| FlatOperand {
        dtype: Dtype::F32,
        element_shape: Vec::new(),
        lane_extent,
        axis_domains: vec![domain],
    };
    loom_program! {
        name: "lane_hoist".into(),
        module: builder.finish(main).unwrap(),
        lane_count: u,
        domains,
        index_maps,
        num_axes: 1,
        operands: vec![
            binding(1, 0),
            binding(u, 0),
            binding(if output_width_one { 1 } else { u }, output_domain),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn mixed_lane_hoist_program() -> LiftedProgram {
    let mut builder = ModuleBuilder::new();
    builder.begin_func("main");
    let narrow = builder.param(Shape::new(DType::F32, [3]));
    let wide = builder.param(Shape::new(DType::F32, [3]));
    let narrow_stage = builder.multiply(narrow, narrow);
    let output = builder.multiply(narrow_stage, wide);
    let main = builder.end_func(vec![output, output]);
    let binding = |lane_extent, domain| FlatOperand {
        dtype: Dtype::F32,
        element_shape: vec![3],
        lane_extent,
        axis_domains: vec![domain],
    };
    loom_program! {
        name: "mixed_lane_hoist".into(),
        module: builder.finish(main).unwrap(),
        lane_count: 128,
        domains: vec![
            Domain {
                extent: 7,
            },
            Domain {
                extent: 1,
            },
        ],
        index_maps: vec![IndexMap {
            source_id: 0,
            target_id: 1,
            is_injective: false,
            is_sorted: false,
            may_have_unmapped: true,
        }],
        num_axes: 1,
        operands: vec![
            binding(1, 0),
            binding(128, 0),
            binding(1, 0),
            binding(128, 1),
        ],
        num_inputs: 2,
        available_views: Vec::new(),
    }
}

pub(super) fn split_parts(loom: &LiftedProgram) -> Vec<KernelPart> {
    (73..256)
        .find_map(|cap| plan(loom, cap).ok().filter(|parts| parts.len() == 2))
        .expect("diamond admits a deterministic two-consumer split")
}

pub(super) fn cut_plan(loom: &LiftedProgram, register_cap: usize) -> PlanIr {
    let domain = DomainPlan::prepare(loom).unwrap();
    PlanIr::build(loom, domain, split_parts(loom), register_cap, 1).unwrap()
}

pub(super) fn bounded_materialized_plan(loom: &LiftedProgram) -> (usize, PlanIr) {
    (1..1024)
        .find_map(|statement_cap| {
            if plan(loom, statement_cap).is_ok() {
                return None;
            }
            let domain = DomainPlan::prepare(loom).ok()?;
            let plan =
                PlanIr::build_bounded(loom, domain, statement_cap, 0, 16, &OnceCell::new(), true)
                    .ok()?;
            if plan.candidates[0].materialized.is_empty() {
                None
            } else {
                Some((statement_cap, plan))
            }
        })
        .expect("materialization makes the indivisible diamond fit")
}

pub(super) fn split_reduction_plan(
    loom: &LiftedProgram,
    provenance: ReductionProvenance,
) -> (usize, PlanIr) {
    (73..1024)
        .find_map(|statement_cap| {
            let domain = DomainPlan::prepare(loom).ok()?;
            let plan = PlanIr::build_bounded(
                loom,
                domain,
                statement_cap,
                usize::MAX,
                16,
                &OnceCell::new(),
                true,
            )
            .ok()?;
            plan.candidates[0]
                .reductions
                .iter()
                .any(|region| region.provenance == provenance && region.launches.len() > 1)
                .then_some((statement_cap, plan))
        })
        .expect("reduction region admits more than one bounded launch")
}

pub(super) fn baseline_plan(loom: &LiftedProgram) -> PlanIr {
    let domain = DomainPlan::prepare(loom).unwrap();
    PlanIr::build(loom, domain, plan(loom, usize::MAX).unwrap(), usize::MAX, 0).unwrap()
}
