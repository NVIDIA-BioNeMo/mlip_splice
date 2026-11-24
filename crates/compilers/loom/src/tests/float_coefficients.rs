// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{FlatOperand, FlatProgram, lower_program};
use algebraic_real::AlgebraicReal;
use domain_index::Domain;
use xir::analysis::constant_literal;
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ops::Literal;
use xir::shape::Shape;

#[test]
fn lowering_cancels_ordinary_and_mixed_exact_coefficients() {
    for dims in [vec![], vec![4], vec![2, 3]] {
        let shape = Shape::new(DType::F32, dims.clone());
        let elements = shape.element_count().unwrap();
        let mut builder = ModuleBuilder::new();
        let main = builder.func("cancellation", |func| {
            let x = func.param(shape.clone());
            let two = func.constant(Literal::new(
                shape.clone(),
                2.0f32.to_le_bytes().repeat(elements),
            ));
            let root = func.constant(
                Literal::exact(shape, vec![AlgebraicReal::sqrt(2u32); elements]).unwrap(),
            );
            let doubled = func.multiply(two, x);
            let difference = func.binary("subtract", doubled, x).unwrap();
            let ordinary = func.binary("subtract", difference, x).unwrap();
            let scaled = func.multiply(root, x);
            let squared = func.multiply(root, scaled);
            let mixed = func.binary("subtract", squared, doubled).unwrap();
            vec![ordinary, mixed]
        });
        let binding = || FlatOperand {
            dtype: DType::F32,
            element_shape: dims.clone(),
            lane_extent: 8,
            axis_domains: vec![0],
        };
        let program = lower_program(FlatProgram {
            module: builder.finish(main).unwrap(),
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding(), binding()],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "cancellation".into(),
        })
        .unwrap();
        let func = program.module.entry_func();
        for &result in &func.results {
            let literal = constant_literal(func, result).expect("constant zero result");
            assert!(literal.data.iter().all(|&byte| byte == 0));
        }
        assert!(func.nodes.iter().all(|node| !matches!(
            node.op.mnemonic(),
            "add" | "subtract" | "multiply" | "negate"
        )));
    }
}
