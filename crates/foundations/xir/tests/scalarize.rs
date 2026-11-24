// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xir::builder::ModuleBuilder;
use xir::dim_numbers::DotDimensionNumbers;
use xir::dtype::DType;
use xir::ops::{Compare, ComparisonDirection, DotAlgorithm, Maximum, Select};
use xir::shape::Shape;

#[test]
fn generic_scalarization_preserves_elementwise_semantics() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::new(DType::F32, [2]);
        let left = func.param(shape.clone());
        let right = func.param(shape);
        let maximum = func.maximum(left, right);
        let greater = func.compare(ComparisonDirection::Gt, None, left, right);
        vec![maximum, greater]
    });
    let mut module = builder.finish(main).unwrap();

    xir::scalarize::run(&mut module).unwrap();

    let func = module.entry_func();
    let maxima = func
        .nodes
        .iter()
        .filter(|node| node.op.as_any().is::<Maximum>())
        .collect::<Vec<_>>();
    assert_eq!(maxima.len(), 2);
    assert!(
        maxima
            .iter()
            .all(|node| func.value_shape(node.results[0]).is_scalar())
    );
    let comparisons = func
        .nodes
        .iter()
        .filter_map(|node| node.op.as_any().downcast_ref::<Compare>())
        .collect::<Vec<_>>();
    assert_eq!(comparisons.len(), 2);
    assert!(
        comparisons
            .iter()
            .all(|op| op.comparison_direction == ComparisonDirection::Gt)
    );
}

#[test]
fn generic_scalarization_broadcasts_a_scalar_select_predicate() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let predicate = func.param(Shape::scalar(DType::Bool));
        let shape = Shape::new(DType::F32, [2]);
        let on_true = func.param(shape.clone());
        let on_false = func.param(shape);
        vec![func.op1(Select, &[predicate, on_true, on_false])]
    });
    let mut module = builder.finish(main).unwrap();

    xir::scalarize::run(&mut module).unwrap();

    let selects = module
        .entry_func()
        .nodes
        .iter()
        .filter(|node| node.op.as_any().is::<Select>())
        .count();
    assert_eq!(selects, 2);
}

#[test]
fn generic_scalarization_eliminates_tensor_views() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let value = func.param(Shape::new(DType::F32, [2, 2]));
        vec![func.transpose(value, &[1, 0])]
    });
    let mut module = builder.finish(main).unwrap();

    xir::scalarize::run(&mut module).unwrap();

    assert!(
        module
            .entry_func()
            .nodes
            .iter()
            .all(|node| node.op.mnemonic() != "transpose")
    );
}

#[test]
fn generic_scalarization_preserves_the_module_function_table() {
    let mut builder = ModuleBuilder::new();
    builder.func("helper", |func| {
        let left = func.param(Shape::scalar(DType::F32));
        let right = func.param(Shape::scalar(DType::F32));
        vec![func.add(left, right)]
    });
    let main = builder.func("main", |func| {
        let shape = Shape::new(DType::F32, [2]);
        let left = func.param(shape.clone());
        let right = func.param(shape);
        vec![func.maximum(left, right)]
    });
    let mut module = builder.finish(main).unwrap();

    xir::scalarize::run(&mut module).unwrap();

    assert_eq!(module.funcs.len(), 2);
    assert_eq!(module.entry.index(), 1);
    assert_eq!(module.funcs[0].name, "helper");
    assert_eq!(module.funcs[1].name, "main");
}

#[test]
fn target_specific_ops_require_a_custom_lowering() {
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        let shape = Shape::new(DType::F32, [2]);
        let left = func.param(shape.clone());
        let right = func.param(shape);
        vec![func.dot_general(
            left,
            right,
            DotDimensionNumbers::new(&[0], &[0], &[], &[]),
            DotAlgorithm::f32_f32_f32(),
        )]
    });
    let mut module = builder.finish(main).unwrap();

    let error = xir::scalarize::run(&mut module).unwrap_err();

    assert!(error.contains("dot_general"), "{error}");
}

#[test]
fn scalarization_does_not_expand_domain_axes() {
    let mut graph = domain_index::FiniteMapGraph::default();
    let domain = graph.add_set(1_000_000).unwrap();
    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    let main = builder.func("main", |func| {
        let shape = Shape::new(DType::F32, [2]).with_domain_axes([domain]);
        let left = func.param(shape.clone());
        let right = func.param(shape);
        vec![func.add(left, right)]
    });
    let mut module = builder.finish(main).unwrap();

    xir::scalarize::run(&mut module).unwrap();

    let func = module.entry_func();
    assert_eq!(
        func.nodes
            .iter()
            .filter(|node| node.op.mnemonic() == "add")
            .count(),
        2
    );
    assert!(
        func.values
            .iter()
            .all(|value| value.shape.domain_axes.as_slice() == [domain])
    );
    assert_eq!(func.value_shape(func.results[0]).dims.as_slice(), [2]);
}

#[test]
fn custom_lowering_receives_aligned_scalar_inputs_for_every_rank() {
    use xir::ir::Node;
    use xir::scalarize::{Context, CustomLowering, ScalarizedResults};

    struct CheckAlignment {
        count: usize,
        visited: usize,
    }

    impl CustomLowering for CheckAlignment {
        fn lower(
            &mut self,
            context: &mut Context<'_>,
            node: &Node,
        ) -> Result<Option<ScalarizedResults>, String> {
            if !node.op.as_any().is::<Select>() {
                return Ok(None);
            }
            let inputs = context.aligned_scalar_inputs(node)?;
            assert_eq!(inputs.len(), 3);
            assert_eq!(
                inputs[0],
                vec![context.mapped(node.inputs[0])?[0]; self.count]
            );
            for (aligned, &original) in inputs[1..].iter().zip(&node.inputs[1..]) {
                assert_eq!(aligned, context.mapped(original)?);
                assert_eq!(aligned.len(), self.count);
            }
            let output = (0..self.count)
                .map(|i| {
                    context
                        .builder()
                        .op1(Select, &[inputs[0][i], inputs[1][i], inputs[2][i]])
                })
                .collect();
            self.visited += 1;
            Ok(Some(vec![output]))
        }
    }

    for dims in [&[][..], &[3], &[2, 3]] {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let predicate = func.param(Shape::scalar(DType::Bool));
            let left = func.param(Shape::new(DType::F32, dims.iter().copied()));
            let right = func.param(Shape::new(DType::F32, dims.iter().copied()));
            vec![func.op1(Select, &[predicate, left, right])]
        });
        let mut module = builder.finish(main).unwrap();
        let count = dims.iter().map(|&dim| dim as usize).product();
        let mut lowering = CheckAlignment { count, visited: 0 };
        xir::scalarize::run_with(&mut module, &mut lowering).unwrap();
        assert_eq!(lowering.visited, 1);
        let func = module.entry_func();
        assert_eq!(func.value_shape(func.results[0]).dims.as_slice(), dims);
        assert_eq!(
            func.nodes
                .iter()
                .filter(|node| node.op.as_any().is::<Select>())
                .count(),
            count
        );
    }
}

#[test]
fn scalar_input_alignment_rejects_result_arity_and_element_count_mismatches() {
    use xir::ir::Node;
    use xir::scalarize::{Context, CustomLowering, ScalarizedResults};

    struct CheckErrors;
    impl CustomLowering for CheckErrors {
        fn lower(
            &mut self,
            context: &mut Context<'_>,
            node: &Node,
        ) -> Result<Option<ScalarizedResults>, String> {
            let mut invalid = node.clone();
            invalid.results.clear();
            assert!(
                context
                    .aligned_scalar_inputs(&invalid)
                    .unwrap_err()
                    .contains("requires one result")
            );
            invalid.results.extend([node.results[0]; 2]);
            assert!(
                context
                    .aligned_scalar_inputs(&invalid)
                    .unwrap_err()
                    .contains("requires one result")
            );
            invalid.results = node.results.clone();
            invalid.inputs[0] = context.func().params[0];
            assert!(
                context
                    .aligned_scalar_inputs(&invalid)
                    .unwrap_err()
                    .contains("element count mismatch")
            );
            Ok(None)
        }
    }
    let mut builder = ModuleBuilder::new();
    let main = builder.func("main", |func| {
        func.param(Shape::new(DType::F32, [2]));
        let left = func.param(Shape::new(DType::F32, [3]));
        let right = func.param(Shape::new(DType::F32, [3]));
        vec![func.add(left, right)]
    });
    let mut module = builder.finish(main).unwrap();
    xir::scalarize::run_with(&mut module, &mut CheckErrors).unwrap();
}

#[test]
fn scalarization_rejects_zero_length_axes_without_changing_the_module() {
    for dims in [&[0][..], &[2, 0, 3]] {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            vec![func.param(Shape::new(DType::F32, dims.iter().copied()))]
        });
        let mut module = builder.finish(main).unwrap();
        let before = format!("{module:?}");
        assert!(
            xir::scalarize::run(&mut module)
                .unwrap_err()
                .contains("zero-sized")
        );
        assert_eq!(format!("{module:?}"), before);
    }
}
