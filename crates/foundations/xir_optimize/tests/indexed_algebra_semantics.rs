// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{FiniteMapGraph, FiniteMapId, FiniteSetId};
use xir::builder::ModuleBuilder;
use xir::dtype::DType;
use xir::ir::{Func, Module, ValueId};
use xir::ops::{
    Constant, GatherAlongMap, Literal, Negate, ScalarOp, Select, Subtract, SumAlongMap,
};
use xir::shape::Shape;
use xir_optimize::{
    DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK,
    extract_algebraic_frontier_with_limits,
};

#[derive(Clone, Debug, PartialEq)]
struct Tensor {
    axes: Vec<FiniteSetId>,
    values: Vec<f64>,
}

type RuntimeMaps = Vec<Vec<i32>>;

fn domain_shape(axes: impl IntoIterator<Item = FiniteSetId>) -> Shape {
    Shape::scalar(DType::F64).with_domain_axes(axes)
}

fn bool_domain_shape(axes: impl IntoIterator<Item = FiniteSetId>) -> Shape {
    Shape::scalar(DType::Bool).with_domain_axes(axes)
}

fn f64_constant(builder: &mut ModuleBuilder, shape: Shape, values: &[f64]) -> ValueId {
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    builder.constant(Literal::new(shape, data))
}

fn extent(graph: &FiniteMapGraph, set: FiniteSetId) -> usize {
    usize::try_from(graph.set(set).unwrap().extent).unwrap()
}

fn tensor_size(graph: &FiniteMapGraph, axes: &[FiniteSetId]) -> usize {
    axes.iter()
        .map(|&axis| extent(graph, axis))
        .product::<usize>()
}

fn parameter_tensor(
    graph: &FiniteMapGraph,
    func: &Func,
    parameter: ValueId,
    values: &[f64],
) -> Tensor {
    let shape = func.value_shape(parameter);
    assert!(
        shape.dims.is_empty(),
        "test interpreter only needs scalar cells"
    );
    assert_eq!(values.len(), tensor_size(graph, &shape.domain_axes));
    Tensor {
        axes: shape.domain_axes.to_vec(),
        values: values.to_vec(),
    }
}

fn constant_tensor(graph: &FiniteMapGraph, literal: &Literal) -> Tensor {
    assert!(
        literal.shape.dims.is_empty(),
        "test interpreter only needs scalar constants"
    );
    let scalar = match literal.shape.dtype {
        DType::F64 => {
            assert_eq!(literal.data.len(), 8);
            f64::from_le_bytes(literal.data.as_slice().try_into().unwrap())
        }
        DType::Bool => {
            assert_eq!(literal.data.len(), 1);
            if literal.data[0] == 0 { 0.0 } else { 1.0 }
        }
        dtype => panic!("unsupported constant dtype {dtype:?}"),
    };
    let axes = literal.shape.domain_axes.to_vec();
    Tensor {
        values: vec![scalar; tensor_size(graph, &axes)],
        axes,
    }
}

fn gather(
    graph: &FiniteMapGraph,
    maps: &RuntimeMaps,
    map: FiniteMapId,
    axis: usize,
    input: &Tensor,
) -> Tensor {
    let declared = graph.map(map).unwrap();
    assert_eq!(input.axes[axis], declared.target);
    let target_extent = extent(graph, declared.target);
    let source_extent = extent(graph, declared.source);
    assert_eq!(maps[map.index()].len(), source_extent);
    let mut axes = input.axes.clone();
    axes[axis] = declared.source;
    let output_len = tensor_size(graph, &axes);
    let stride = axes[axis + 1..]
        .iter()
        .map(|&set| extent(graph, set))
        .product::<usize>();
    let values = (0..output_len)
        .map(|output_index| {
            let source_index = (output_index / stride) % source_extent;
            let mapped = maps[map.index()][source_index];
            if mapped < 0 {
                return 0.0;
            }
            let mapped = usize::try_from(mapped).unwrap();
            assert!(mapped < target_extent);
            let outer = output_index / (source_extent * stride);
            let inner = output_index % stride;
            input.values[outer * target_extent * stride + mapped * stride + inner]
        })
        .collect();
    Tensor { axes, values }
}

fn sum(
    graph: &FiniteMapGraph,
    maps: &RuntimeMaps,
    map: FiniteMapId,
    axis: usize,
    input: &Tensor,
) -> Tensor {
    let declared = graph.map(map).unwrap();
    assert_eq!(input.axes[axis], declared.source);
    let source_extent = extent(graph, declared.source);
    let target_extent = extent(graph, declared.target);
    assert_eq!(maps[map.index()].len(), source_extent);
    let stride = input.axes[axis + 1..]
        .iter()
        .map(|&set| extent(graph, set))
        .product::<usize>();
    let mut axes = input.axes.clone();
    axes[axis] = declared.target;
    let mut values = vec![0.0; tensor_size(graph, &axes)];
    for input_index in 0..input.values.len() {
        let source_index = (input_index / stride) % source_extent;
        let mapped = maps[map.index()][source_index];
        if mapped < 0 {
            continue;
        }
        let mapped = usize::try_from(mapped).unwrap();
        assert!(mapped < target_extent);
        let outer = input_index / (source_extent * stride);
        let inner = input_index % stride;
        let output_index = outer * target_extent * stride + mapped * stride + inner;
        values[output_index] += input.values[input_index];
    }
    Tensor { axes, values }
}

fn elementwise(op: ScalarOp, inputs: &[&Tensor]) -> Tensor {
    assert!(!inputs.is_empty());
    for input in &inputs[1..] {
        assert_eq!(input.axes, inputs[0].axes);
        assert_eq!(input.values.len(), inputs[0].values.len());
    }
    let values = (0..inputs[0].values.len())
        .map(|index| match op {
            ScalarOp::Negate => -inputs[0].values[index],
            ScalarOp::Add => inputs[0].values[index] + inputs[1].values[index],
            ScalarOp::Subtract => inputs[0].values[index] - inputs[1].values[index],
            ScalarOp::Multiply => inputs[0].values[index] * inputs[1].values[index],
            ScalarOp::Select => {
                if inputs[0].values[index] != 0.0 {
                    inputs[1].values[index]
                } else {
                    inputs[2].values[index]
                }
            }
            other => panic!("unsupported elementwise op {other:?}"),
        })
        .collect();
    Tensor {
        axes: inputs[0].axes.clone(),
        values,
    }
}

fn evaluate(module: &Module, maps: &RuntimeMaps, parameters: &[Vec<f64>]) -> Vec<Tensor> {
    assert_eq!(module.finite_map_graph.maps.len(), maps.len());
    let func = module.entry_func();
    assert_eq!(func.params.len(), parameters.len());
    let mut values = vec![None; func.values.len()];
    for (&parameter, data) in func.params.iter().zip(parameters) {
        values[parameter.index()] = Some(parameter_tensor(
            &module.finite_map_graph,
            func,
            parameter,
            data,
        ));
    }
    for node in &func.nodes {
        assert_eq!(node.results.len(), 1);
        let result = if let Some(op) = node.op.as_any().downcast_ref::<Constant>() {
            constant_tensor(&module.finite_map_graph, &op.literal)
        } else if let Some(op) = node.op.as_any().downcast_ref::<GatherAlongMap>() {
            gather(
                &module.finite_map_graph,
                maps,
                op.map,
                op.axis as usize,
                values[node.inputs[0].index()].as_ref().unwrap(),
            )
        } else if let Some(op) = node.op.as_any().downcast_ref::<SumAlongMap>() {
            sum(
                &module.finite_map_graph,
                maps,
                op.map,
                op.axis as usize,
                values[node.inputs[0].index()].as_ref().unwrap(),
            )
        } else if let Some(op) = node.op.as_elementwise() {
            let inputs = node
                .inputs
                .iter()
                .map(|input| values[input.index()].as_ref().unwrap())
                .collect::<Vec<_>>();
            elementwise(op.scalar(), &inputs)
        } else {
            panic!("unsupported op {}", node.op.mnemonic());
        };
        values[node.results[0].index()] = Some(result);
    }
    func.results
        .iter()
        .map(|result| values[result.index()].clone().unwrap())
        .collect()
}

fn assert_outputs(actual: &[Tensor], expected: &[Vec<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.values.as_slice(),
            expected.as_slice(),
            "{context}, output {index}"
        );
    }
}

fn assert_frontier_semantics(
    module: &Module,
    maps: &RuntimeMaps,
    parameters: &[Vec<f64>],
    expected: &[Vec<f64>],
) {
    let original = evaluate(module, maps, parameters);
    assert_outputs(&original, expected, "original");
    let (frontier, _) = extract_algebraic_frontier_with_limits(module, 1 << 20, 1 << 18).unwrap();
    assert!(!frontier.is_empty());
    for (index, alternative) in frontier.iter().enumerate() {
        let actual = evaluate(&alternative.module, maps, parameters);
        assert_eq!(actual, original, "frontier alternative {index}");
    }
}

#[test]
fn partial_collision_projection_empty_fibers_and_zero_fill_stay_exact() {
    let mut graph = FiniteMapGraph::default();
    let source = graph.add_set(5).unwrap();
    let target = graph.add_set(4).unwrap();
    let map = graph.add_map(source, target).unwrap();
    let maps = vec![vec![0, 0, -1, 3, 0]];

    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("partial_projection");
    let coefficient = builder.param(domain_shape([target]));
    let other_coefficient = builder.param(domain_shape([target]));
    let x = builder.param(domain_shape([source]));
    let y = builder.param(domain_shape([source]));
    let predicate = builder.param(bool_domain_shape([source]));
    let target_predicate = builder.param(bool_domain_shape([target]));
    let gathered = builder.gather_along_map(coefficient, map, 0);
    let product = builder.multiply(gathered, x);
    let projection = builder.sum_along_map(product, map, 0);
    let selected_projection = builder.op1(Select, &[target_predicate, projection, coefficient]);
    let shared_projection = builder.add(projection, selected_projection);

    let gathered_other = builder.gather_along_map(other_coefficient, map, 0);
    let target_add = builder.add(coefficient, other_coefficient);
    let gather_add_left = builder.gather_along_map(target_add, map, 0);
    let gather_add_right = builder.add(gathered, gathered_other);
    let gather_add = builder.op1(Subtract, &[gather_add_left, gather_add_right]);
    let gather_add = builder.sum_along_map(gather_add, map, 0);
    let target_mul = builder.multiply(coefficient, other_coefficient);
    let gather_mul_left = builder.gather_along_map(target_mul, map, 0);
    let gather_mul_right = builder.multiply(gathered, gathered_other);
    let gather_mul = builder.op1(Subtract, &[gather_mul_left, gather_mul_right]);
    let gather_mul = builder.sum_along_map(gather_mul, map, 0);
    let target_neg = builder.op1(Negate, &[coefficient]);
    let gather_neg_left = builder.gather_along_map(target_neg, map, 0);
    let gather_neg_right = builder.op1(Negate, &[gathered]);
    let gather_neg = builder.op1(Subtract, &[gather_neg_left, gather_neg_right]);
    let gather_neg = builder.sum_along_map(gather_neg, map, 0);
    let source_add = builder.add(x, y);
    let sum_add_left = builder.sum_along_map(source_add, map, 0);
    let sum_x = builder.sum_along_map(x, map, 0);
    let sum_y = builder.sum_along_map(y, map, 0);
    let sum_add_right = builder.add(sum_x, sum_y);
    let sum_linearity = builder.op1(Subtract, &[sum_add_left, sum_add_right]);
    let multiplicity = builder.sum_along_map(gathered, map, 0);
    let negated_y = builder.op1(Negate, &[y]);
    let selected = builder.op1(Select, &[predicate, x, negated_y]);
    let selected = builder.sum_along_map(selected, map, 0);
    let cancelled = builder.op1(Negate, &[product]);
    let cancelled = builder.add(product, cancelled);
    let cancelled = builder.sum_along_map(cancelled, map, 0);
    let one = f64_constant(&mut builder, domain_shape([target]), &[1.0]);
    let gathered_one = builder.gather_along_map(one, map, 0);
    let zero = f64_constant(&mut builder, domain_shape([target]), &[0.0]);
    let zero = builder.gather_along_map(zero, map, 0);
    let zero = builder.sum_along_map(zero, map, 0);
    let entry = builder.end_func(vec![
        projection,
        shared_projection,
        gather_add,
        gather_mul,
        gather_neg,
        sum_linearity,
        multiplicity,
        selected,
        cancelled,
        gathered_one,
        zero,
    ]);
    let module = builder.finish(entry).unwrap();

    assert_frontier_semantics(
        &module,
        &maps,
        &[
            vec![2.0, 5.0, 7.0, 11.0],
            vec![3.0, -2.0, 4.0, 6.0],
            vec![3.0, -4.0, 13.0, 2.0, 5.0],
            vec![1.0, 6.0, -2.0, 4.0, 8.0],
            vec![1.0, 0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ],
        &[
            vec![8.0, 0.0, 0.0, 22.0],
            vec![16.0, 5.0, 0.0, 33.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![6.0, 0.0, 0.0, 11.0],
            vec![2.0, 0.0, 0.0, -4.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ],
    );
}

#[test]
fn parallel_maps_with_the_same_endpoints_keep_nominal_identity() {
    let mut graph = FiniteMapGraph::default();
    let source = graph.add_set(4).unwrap();
    let target = graph.add_set(3).unwrap();
    let first = graph.add_map(source, target).unwrap();
    let second = graph.add_map(source, target).unwrap();
    let maps = vec![vec![0, 0, 2, -1], vec![1, 2, 2, -1]];

    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("parallel_maps");
    let coefficient = builder.param(domain_shape([target]));
    let through_first = builder.gather_along_map(coefficient, first, 0);
    let through_second = builder.gather_along_map(coefficient, second, 0);
    let different = builder.op1(Subtract, &[through_second, through_first]);
    let different = builder.sum_along_map(different, first, 0);
    let same = builder.op1(Subtract, &[through_first, through_first]);
    let same = builder.sum_along_map(same, first, 0);
    let entry = builder.end_func(vec![different, same]);
    let module = builder.finish(entry).unwrap();

    let original = evaluate(&module, &maps, &[vec![2.0, 3.0, 5.0]]);
    assert_outputs(
        &original,
        &[vec![4.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
        "original",
    );
    let frontier = extract_algebraic_frontier_with_limits(
        &module,
        DEFAULT_MAX_ALGEBRAIC_WORK,
        DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
    )
    .unwrap()
    .0;
    assert!(!frontier.is_empty());
    for (index, alternative) in frontier.iter().enumerate() {
        assert_eq!(
            evaluate(&alternative.module, &maps, &[vec![2.0, 3.0, 5.0]]),
            original,
            "frontier alternative {index}"
        );
    }
}

#[test]
fn equal_cardinality_domains_remain_distinct_types() {
    let mut graph = FiniteMapGraph::default();
    let source = graph.add_set(4).unwrap();
    let left_domain = graph.add_set(3).unwrap();
    let right_domain = graph.add_set(3).unwrap();
    let left_map = graph.add_map(source, left_domain).unwrap();
    let right_map = graph.add_map(source, right_domain).unwrap();
    let maps = vec![vec![0, 1, 1, -1], vec![2, 2, 0, -1]];

    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("equal_cardinality_domains");
    let left = builder.param(domain_shape([left_domain]));
    let right = builder.param(domain_shape([right_domain]));
    let left = builder.gather_along_map(left, left_map, 0);
    let right = builder.gather_along_map(right, right_map, 0);
    let difference = builder.op1(Subtract, &[left, right]);
    let to_left = builder.sum_along_map(difference, left_map, 0);
    let to_right = builder.sum_along_map(difference, right_map, 0);
    let entry = builder.end_func(vec![to_left, to_right]);
    let module = builder.finish(entry).unwrap();

    assert_frontier_semantics(
        &module,
        &maps,
        &[vec![2.0, 3.0, 5.0], vec![7.0, 11.0, 13.0]],
        &[vec![-11.0, -14.0, 0.0], vec![-4.0, 0.0, -21.0]],
    );
}

#[test]
fn a_direct_map_is_not_assumed_to_equal_a_path_composition() {
    let mut graph = FiniteMapGraph::default();
    let source = graph.add_set(5).unwrap();
    let middle = graph.add_set(4).unwrap();
    let target = graph.add_set(3).unwrap();
    let first = graph.add_map(source, middle).unwrap();
    let second = graph.add_map(middle, target).unwrap();
    let direct = graph.add_map(source, target).unwrap();
    let maps = vec![
        vec![0, 1, 1, 3, -1],
        vec![2, 0, -1, 1],
        vec![0, 0, 2, 2, -1],
    ];

    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("composition_provenance");
    let coefficient = builder.param(domain_shape([target]));
    let source_value = builder.param(domain_shape([source]));
    let nested = builder.gather_along_map(coefficient, second, 0);
    let nested = builder.gather_along_map(nested, first, 0);
    let through_direct = builder.gather_along_map(coefficient, direct, 0);
    let gather_difference = builder.op1(Subtract, &[nested, through_direct]);
    let gather_difference = builder.sum_along_map(gather_difference, direct, 0);
    let nested_sum = builder.sum_along_map(source_value, first, 0);
    let nested_sum = builder.sum_along_map(nested_sum, second, 0);
    let direct_sum = builder.sum_along_map(source_value, direct, 0);
    let sum_difference = builder.op1(Subtract, &[nested_sum, direct_sum]);
    let entry = builder.end_func(vec![gather_difference, sum_difference]);
    let module = builder.finish(entry).unwrap();

    assert_frontier_semantics(
        &module,
        &maps,
        &[vec![2.0, 3.0, 5.0], vec![1.0, 2.0, 4.0, 8.0, 16.0]],
        &[vec![3.0, 0.0, -5.0], vec![3.0, 8.0, -11.0]],
    );
}

#[test]
fn two_domain_axes_keep_their_axis_specific_maps() {
    let mut graph = FiniteMapGraph::default();
    let source0 = graph.add_set(3).unwrap();
    let target0 = graph.add_set(2).unwrap();
    let source1 = graph.add_set(4).unwrap();
    let target1 = graph.add_set(3).unwrap();
    let map0 = graph.add_map(source0, target0).unwrap();
    let map1 = graph.add_map(source1, target1).unwrap();
    let maps = vec![vec![0, 0, -1], vec![2, 0, 2, -1]];

    let mut builder = ModuleBuilder::with_finite_map_graph(graph);
    builder.begin_func("two_axes");
    let coefficient = builder.param(domain_shape([target0, target1]));
    let value = builder.param(domain_shape([source0, source1]));
    let gathered = builder.gather_along_map(coefficient, map1, 1);
    let gathered = builder.gather_along_map(gathered, map0, 0);
    let product = builder.multiply(gathered, value);
    let summed = builder.sum_along_map(product, map0, 0);
    let summed = builder.sum_along_map(summed, map1, 1);
    let entry = builder.end_func(vec![summed]);
    let module = builder.finish(entry).unwrap();

    assert_frontier_semantics(
        &module,
        &maps,
        &[
            vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            (1..=12).map(f64::from).collect(),
        ],
        &[vec![16.0, 0.0, 80.0, 0.0, 0.0, 0.0]],
    );
}
