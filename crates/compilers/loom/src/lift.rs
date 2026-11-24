// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lift a pointwise Loom program over its finite domain maps.

use std::collections::HashMap;

use domain_index::{FiniteMapGraph, FiniteMapId, FiniteSetId, Graph};
use xir::builder::ModuleBuilder;
use xir::ir::{Def, Func, Module, ValueId};
use xir::ops::{
    Concatenate, Constant, Convert, GatherAlongMap, Literal, Multiply, Negate, Reshape, Slice,
    SumAlongMap,
};

use super::LiftedProgram;
use super::program::{InputElement, OutputFactor};

struct Domains {
    graph: FiniteMapGraph,
    maps: HashMap<(FiniteSetId, FiniteSetId), FiniteMapId>,
    unit: FiniteSetId,
    iteration: Vec<FiniteSetId>,
}

impl Domains {
    fn for_program(program: &LiftedProgram) -> Result<Self, String> {
        let physical = Graph {
            domains: program.domains.clone(),
            index_maps: program.index_maps.clone(),
            aligned_views: program.available_views.clone(),
        };
        let mut graph = physical.finite_map_graph()?;
        let mut maps = graph
            .maps
            .iter()
            .enumerate()
            .map(|(index, map)| {
                Ok((
                    (map.source, map.target),
                    FiniteMapId(u32::try_from(index).map_err(|error| error.to_string())?),
                ))
            })
            .collect::<Result<HashMap<_, _>, String>>()?;
        if maps.len() != graph.maps.len() {
            return Err("parallel index_maps cannot name one finite map".into());
        }
        let unit = graph.add_set(1)?;
        let lane = if program.lane_count == 1 {
            unit
        } else {
            graph.add_set(program.lane_count)?
        };
        let mut iteration = Vec::with_capacity(program.num_axes + 1);
        for axis in 0..program.num_axes {
            let present = program
                .operands
                .iter()
                .filter_map(|binding| {
                    let domain = binding.axis_domains[axis];
                    (domain >= 0).then_some(domain)
                })
                .collect::<Vec<_>>();
            let domain = if present.is_empty() {
                unit
            } else {
                FiniteSetId(
                    u32::try_from(physical.common_source(present)?)
                        .map_err(|error| error.to_string())?,
                )
            };
            iteration.push(domain);
        }
        iteration.push(lane);
        for &source in &iteration {
            if source == unit || maps.contains_key(&(source, unit)) {
                continue;
            }
            let map = graph.add_map(source, unit)?;
            maps.insert((source, unit), map);
        }
        Ok(Self {
            graph,
            maps,
            unit,
            iteration,
        })
    }

    fn operand_axes(
        &self,
        program: &LiftedProgram,
        operand: usize,
    ) -> Result<Vec<FiniteSetId>, String> {
        let binding = &program.operands[operand];
        let mut axes = binding
            .axis_domains
            .iter()
            .map(|&domain| {
                if domain < 0 {
                    Ok(self.unit)
                } else {
                    Ok(FiniteSetId(
                        u32::try_from(domain).map_err(|error| error.to_string())?,
                    ))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        axes.push(if binding.lane_extent == 1 {
            self.unit
        } else {
            *self
                .iteration
                .last()
                .expect("iteration always has a lane axis")
        });
        Ok(axes)
    }

    fn map(&self, source: FiniteSetId, target: FiniteSetId) -> Result<FiniteMapId, String> {
        self.maps.get(&(source, target)).copied().ok_or_else(|| {
            format!(
                "no direct finite map from set {} to set {}",
                source.index(),
                target.index()
            )
        })
    }
}

fn gather_to_iteration(
    builder: &mut ModuleBuilder,
    domains: &Domains,
    mut value: ValueId,
    physical: &[FiniteSetId],
) -> Result<ValueId, String> {
    for (axis, (&source, &target)) in domains.iteration.iter().zip(physical).enumerate().rev() {
        if source != target {
            value = builder.gather_along_map(
                value,
                domains.map(source, target)?,
                u32::try_from(axis).map_err(|error| error.to_string())?,
            );
        }
    }
    Ok(value)
}

fn sum_to_operand(
    builder: &mut ModuleBuilder,
    domains: &Domains,
    mut value: ValueId,
    physical: &[FiniteSetId],
) -> Result<ValueId, String> {
    for (axis, (&source, &target)) in domains.iteration.iter().zip(physical).enumerate() {
        if source != target {
            value = builder.sum_along_map(
                value,
                domains.map(source, target)?,
                u32::try_from(axis).map_err(|error| error.to_string())?,
            );
        }
    }
    Ok(value)
}

pub(super) fn lift_over_domains(program: &LiftedProgram) -> Result<Module, String> {
    let domains = Domains::for_program(program)?;
    let old = program.module.entry_func();
    if program.module.entry.index() + 1 != program.module.funcs.len() {
        return Err("Loom XIR entry function must follow every called function".into());
    }
    let prefix = program.module.funcs[..program.module.entry.index()].to_vec();
    let mut builder = ModuleBuilder::with_function_prefix(domains.graph.clone(), prefix);
    builder.begin_func(&old.name);
    let mut values = vec![None; old.values.len()];
    for (operand, &parameter) in old.params.iter().enumerate() {
        let physical = domains.operand_axes(program, operand)?;
        let shape = old
            .value_shape(parameter)
            .clone()
            .with_domain_axes(physical.iter().copied());
        let value = builder.param(shape);
        values[parameter.index()] = Some(gather_to_iteration(
            &mut builder,
            &domains,
            value,
            &physical,
        )?);
    }
    for node in &old.nodes {
        let inputs = node
            .inputs
            .iter()
            .map(|input| {
                values[input.index()]
                    .ok_or_else(|| format!("value %{} has not been lifted", input.index()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let results = if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
            let mut literal: Literal = constant.literal.clone();
            literal.shape = literal
                .shape
                .with_domain_axes(domains.iteration.iter().copied());
            builder.op(Constant { literal }, &[])
        } else {
            if inputs.is_empty() {
                return Err(format!(
                    "cannot lift zero-operand XIR op `{}` over domains",
                    node.op.mnemonic()
                ));
            }
            builder
                .try_op_boxed(node.op.clone(), &inputs)
                .map_err(|error| format!("lifting `{}`: {error}", node.op.mnemonic()))?
        };
        if results.len() != node.results.len() {
            return Err(format!(
                "lifting `{}` produced {} results, expected {}",
                node.op.mnemonic(),
                results.len(),
                node.results.len()
            ));
        }
        for (&old_result, &result) in node.results.iter().zip(&results) {
            values[old_result.index()] = Some(result);
        }
    }
    let results = old
        .results
        .iter()
        .enumerate()
        .map(|(output, result)| {
            let value = values[result.index()]
                .ok_or_else(|| format!("result %{} has not been lifted", result.index()))?;
            let physical = domains.operand_axes(program, program.num_inputs + output)?;
            sum_to_operand(&mut builder, &domains, value, &physical)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let entry = builder.end_func(results);
    builder
        .finish(entry)
        .map_err(|error| format!("lifted Loom XIR verification: {error}"))
}

fn producer(func: &Func, value: ValueId) -> Option<&xir::ir::Node> {
    let Def::NodeResult { node, slot: 0 } = func.values[value.index()].def else {
        return None;
    };
    func.nodes.get(node)
}

fn input_element(func: &Func, value: ValueId) -> Option<InputElement> {
    match func.values[value.index()].def {
        Def::Param(operand) if func.value_shape(value).is_scalar() => {
            return Some(InputElement {
                operand,
                element: 0,
            });
        }
        Def::Param(_) => return None,
        Def::NodeResult { .. } => {}
    }
    let node = producer(func, value)?;
    if node.op.as_any().is::<Reshape>() || node.op.as_any().is::<GatherAlongMap>() {
        return input_element(func, node.inputs[0]);
    }
    let slice = node.op.as_any().downcast_ref::<Slice>()?;
    if func.value_shape(value).num_elements() != Some(1) {
        return None;
    }
    let source = node.inputs[0];
    let Def::Param(operand) = func.values[source.index()].def else {
        return None;
    };
    let dims = &func.value_shape(source).dims;
    if dims.len() != slice.start_indices.len() {
        return None;
    }
    let element = dims.iter().zip(&slice.start_indices).try_fold(
        0usize,
        |flat, (&extent, &coordinate)| {
            let extent = usize::try_from(extent).ok()?;
            let coordinate = usize::try_from(coordinate).ok()?;
            (coordinate < extent).then_some(flat.checked_mul(extent)?.checked_add(coordinate)?)
        },
    )?;
    Some(InputElement { operand, element })
}

fn output_factor(func: &Func, value: ValueId) -> Option<OutputFactor> {
    if let Some(input) = input_element(func, value) {
        return Some(OutputFactor::Input(input));
    }
    let node = producer(func, value)?;
    if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
        return Some(OutputFactor::Constant(constant.literal.clone()));
    }
    if node.op.as_any().is::<Convert>() {
        return Some(OutputFactor::Convert {
            from: func.value_shape(node.inputs[0]).dtype,
            to: func.value_shape(value).dtype,
            input: Box::new(output_factor(func, node.inputs[0])?),
        });
    }
    let op = node.op.as_elementwise()?.scalar();
    if !matches!(
        op,
        xir::ops::ScalarOp::Negate
            | xir::ops::ScalarOp::Abs
            | xir::ops::ScalarOp::Sqrt
            | xir::ops::ScalarOp::Rsqrt
            | xir::ops::ScalarOp::Not
            | xir::ops::ScalarOp::Add
            | xir::ops::ScalarOp::Subtract
            | xir::ops::ScalarOp::Multiply
            | xir::ops::ScalarOp::Divide
            | xir::ops::ScalarOp::Maximum
            | xir::ops::ScalarOp::Minimum
            | xir::ops::ScalarOp::Remainder
            | xir::ops::ScalarOp::And
            | xir::ops::ScalarOp::Or
            | xir::ops::ScalarOp::Xor
            | xir::ops::ScalarOp::ShiftLeft
            | xir::ops::ScalarOp::ShiftRightArithmetic
            | xir::ops::ScalarOp::ShiftRightLogical
            | xir::ops::ScalarOp::Compare(_)
            | xir::ops::ScalarOp::Select
    ) {
        return None;
    }
    let inputs = node
        .inputs
        .iter()
        .map(|&input| output_factor(func, input))
        .collect::<Option<Vec<_>>>()?;
    Some(OutputFactor::Elementwise {
        op,
        dtype: func.value_shape(value).dtype,
        inputs,
    })
}

fn factorized_sum(func: &Func, value: ValueId) -> Option<(ValueId, Vec<OutputFactor>)> {
    let node = producer(func, value)?;
    if node.op.as_any().is::<SumAlongMap>() {
        let mut input = node.inputs[0];
        while producer(func, input).is_some_and(|node| node.op.as_any().is::<SumAlongMap>()) {
            input = producer(func, input)?.inputs[0];
        }
        return Some((input, Vec::new()));
    }
    if !node.op.as_any().is::<Multiply>() {
        return None;
    }
    for (factor, summed) in [
        (node.inputs[0], node.inputs[1]),
        (node.inputs[1], node.inputs[0]),
    ] {
        if let Some(factor) = output_factor(func, factor)
            && let Some((body, mut factors)) = factorized_sum(func, summed)
        {
            factors.push(factor);
            return Some((body, factors));
        }
    }
    None
}

fn scalar_leaves(func: &Func, value: ValueId, leaves: &mut Vec<ValueId>) {
    let Some(node) = producer(func, value) else {
        leaves.push(value);
        return;
    };
    if node.op.as_any().is::<Reshape>() {
        scalar_leaves(func, node.inputs[0], leaves);
    } else if node.op.as_any().is::<Concatenate>() {
        for &input in &node.inputs {
            scalar_leaves(func, input, leaves);
        }
    } else {
        leaves.push(value);
    }
}

fn depends_on_sum(
    func: &Func,
    value: ValueId,
    seen: &mut std::collections::HashSet<ValueId>,
) -> bool {
    if !seen.insert(value) {
        return false;
    }
    let Some(node) = producer(func, value) else {
        return false;
    };
    node.op.as_any().is::<SumAlongMap>()
        || node
            .inputs
            .iter()
            .any(|&input| depends_on_sum(func, input, seen))
}

fn nested_accumulation_is_lowerable(func: &Func, value: ValueId) -> bool {
    if !depends_on_sum(func, value, &mut std::collections::HashSet::new()) {
        return true;
    }
    let Some(node) = producer(func, value) else {
        return false;
    };
    if node.op.as_any().is::<SumAlongMap>() || node.op.as_any().is::<Negate>() {
        return nested_accumulation_is_lowerable(func, node.inputs[0]);
    }
    if node.op.as_any().is::<Multiply>() {
        let [left, right] = node.inputs.as_slice() else {
            return false;
        };
        let left_has_sum = depends_on_sum(func, *left, &mut std::collections::HashSet::new());
        let right_has_sum = depends_on_sum(func, *right, &mut std::collections::HashSet::new());
        return match (left_has_sum, right_has_sum) {
            (true, false) => nested_accumulation_is_lowerable(func, *left),
            (false, true) => nested_accumulation_is_lowerable(func, *right),
            (false, false) | (true, true) => false,
        };
    }
    false
}

pub(super) fn validate_domain_map_lowering(module: &Module) -> Result<(), String> {
    let func = module.entry_func();
    for (output, &result) in func.results.iter().enumerate() {
        let mut leaves = Vec::new();
        scalar_leaves(func, result, &mut leaves);
        for (element, leaf) in leaves.into_iter().enumerate() {
            match factorized_sum(func, leaf) {
                Some((body, _)) if !nested_accumulation_is_lowerable(func, body) => {
                    return Err(format!(
                        "Loom output {output} element {element} has a nested accumulation that cannot be flattened exactly"
                    ));
                }
                None if depends_on_sum(func, leaf, &mut std::collections::HashSet::new()) => {
                    return Err(format!(
                        "Loom output {output} element {element} has no representable accumulation chain"
                    ));
                }
                Some(_) | None => {}
            }
        }
    }
    Ok(())
}

pub(super) fn lower_domain_maps(
    module: &mut Module,
) -> Result<Vec<Vec<Vec<OutputFactor>>>, String> {
    validate_domain_map_lowering(module)?;
    let old = module.entry_func();
    let mut replacements = HashMap::new();
    let mut output_factors = Vec::with_capacity(old.results.len());
    for &result in &old.results {
        let mut leaves = Vec::new();
        scalar_leaves(old, result, &mut leaves);
        let factors = leaves
            .into_iter()
            .map(|leaf| {
                let Some((body, factors)) = factorized_sum(old, leaf) else {
                    return Vec::new();
                };
                if !factors.is_empty() {
                    replacements.insert(leaf, body);
                }
                factors
            })
            .collect();
        output_factors.push(factors);
    }

    if module.entry.index() + 1 != module.funcs.len() {
        return Err("Loom XIR entry function must follow every called function".into());
    }
    let prefix = module.funcs[..module.entry.index()].to_vec();
    let mut builder = ModuleBuilder::with_function_prefix(module.finite_map_graph.clone(), prefix);
    builder.begin_func(&old.name);
    let mut values = vec![None; old.values.len()];
    for &parameter in &old.params {
        let shape = old.value_shape(parameter).clone().with_domain_axes([]);
        values[parameter.index()] = Some(builder.param(shape));
    }
    for node in &old.nodes {
        if node.results.len() == 1 {
            let result = node.results[0];
            let alias = replacements.get(&result).copied().or_else(|| {
                (node.op.as_any().is::<GatherAlongMap>() || node.op.as_any().is::<SumAlongMap>())
                    .then(|| node.inputs[0])
            });
            if let Some(alias) = alias {
                values[result.index()] = values[alias.index()];
                continue;
            }
        }
        let inputs = node
            .inputs
            .iter()
            .map(|input| {
                values[input.index()]
                    .ok_or_else(|| format!("value %{} has not been lowered", input.index()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let results = if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
            let mut literal = constant.literal.clone();
            literal.shape = literal.shape.with_domain_axes([]);
            builder.op(Constant { literal }, &[])
        } else {
            builder
                .try_op_boxed(node.op.clone(), &inputs)
                .map_err(|error| format!("lowering `{}`: {error}", node.op.mnemonic()))?
        };
        if results.len() != node.results.len() {
            return Err(format!(
                "lowering `{}` produced {} results, expected {}",
                node.op.mnemonic(),
                results.len(),
                node.results.len()
            ));
        }
        for (&old_result, &result) in node.results.iter().zip(&results) {
            values[old_result.index()] = Some(result);
        }
    }
    let results = old
        .results
        .iter()
        .map(|result| {
            values[result.index()]
                .ok_or_else(|| format!("result %{} has not been lowered", result.index()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let entry = builder.end_func(results);
    *module = builder
        .finish(entry)
        .map_err(|error| format!("domain-lowered Loom XIR verification: {error}"))?;
    Ok(output_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatOperand, FlatProgram};
    use algebraic_real::AlgebraicReal;
    use domain_index::{Domain, IndexMap};
    use xir::dtype::DType as Dtype;
    use xir::dtype::DType;
    use xir::shape::Shape;
    use xir_optimize::algebraic_pipeline;

    fn projection_request() -> FlatProgram {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("projection");
        let coefficient = builder.param(Shape::scalar(DType::F32));
        let values = builder.param(Shape::scalar(DType::F32));
        let coefficient = builder.add(coefficient, coefficient);
        let product = builder.multiply(coefficient, values);
        let entry = builder.end_func(vec![product]);
        let module = builder.finish(entry).unwrap();
        let binding = |axis_domain, lane_extent| FlatOperand {
            dtype: Dtype::F32,
            element_shape: Vec::new(),
            lane_extent,
            axis_domains: vec![axis_domain],
        };
        FlatProgram {
            module,
            lane_count: 4,
            domains: vec![Domain { extent: 3 }, Domain { extent: 8 }],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            num_axes: 1,
            operands: vec![binding(0, 1), binding(1, 4), binding(0, 1)],
            num_inputs: 2,
            available_views: Vec::new(),
            name: "projection".into(),
        }
    }

    #[derive(Clone, Copy)]
    enum AccumulationForm {
        AddBeforeSum,
        AddSums,
        MultiplySums,
    }

    fn accumulation_module(form: AccumulationForm) -> Module {
        let mut graph = FiniteMapGraph::default();
        let fine = graph.add_set(4).unwrap();
        let coarse = graph.add_set(2).unwrap();
        let unit = graph.add_set(1).unwrap();
        let map = graph.add_map(fine, coarse).unwrap();
        let final_map = graph.add_map(coarse, unit).unwrap();
        let mut builder = ModuleBuilder::with_finite_map_graph(graph);
        builder.begin_func("accumulation_form");
        let shape = Shape::scalar(DType::F32).with_domain_axes([fine]);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let result = match form {
            AccumulationForm::AddBeforeSum => {
                let sum = builder.add(left, right);
                builder.sum_along_map(sum, map, 0)
            }
            AccumulationForm::AddSums => {
                let left = builder.sum_along_map(left, map, 0);
                let right = builder.sum_along_map(right, map, 0);
                builder.add(left, right)
            }
            AccumulationForm::MultiplySums => {
                let left = builder.sum_along_map(left, map, 0);
                let right = builder.sum_along_map(right, map, 0);
                let product = builder.multiply(left, right);
                builder.sum_along_map(product, final_map, 0)
            }
        };
        let entry = builder.end_func(vec![result]);
        builder.finish(entry).unwrap()
    }

    #[test]
    fn addition_below_one_sum_chain_is_lowerable() {
        let mut module = accumulation_module(AccumulationForm::AddBeforeSum);

        let factors = lower_domain_maps(&mut module).unwrap();

        assert_eq!(factors, vec![vec![Vec::new()]]);
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .all(|node| !node.op.as_any().is::<SumAlongMap>())
        );
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<xir::ops::Add>())
        );
    }

    #[test]
    fn addition_of_sum_chains_is_rejected_without_reassociation() {
        let mut module = accumulation_module(AccumulationForm::AddSums);
        let before = module.to_string();

        let error = lower_domain_maps(&mut module).unwrap_err();

        assert!(
            error.contains("no representable accumulation chain"),
            "{error}"
        );
        assert_eq!(module.to_string(), before);
    }

    #[test]
    fn product_of_sum_chains_is_not_lowered_as_a_sum_of_products() {
        let mut module = accumulation_module(AccumulationForm::MultiplySums);
        let before = module.to_string();

        let error = lower_domain_maps(&mut module).unwrap_err();

        assert!(error.contains("cannot be flattened exactly"), "{error}");
        assert_eq!(module.to_string(), before);
    }

    #[test]
    fn lifted_projection_formula_moves_a_coarse_factor_after_the_fiber_sum() {
        let program = LiftedProgram::try_from(projection_request()).unwrap();
        let mut lifted = lift_over_domains(&program).unwrap();

        algebraic_pipeline().run(&mut lifted).unwrap();

        xir::verify::verify(&lifted).unwrap();
        let func = lifted.entry_func();
        assert_eq!(func.nodes.len(), 4);
        let producer = |value: ValueId| {
            let xir::ir::Def::NodeResult { node, slot: 0 } = func.values[value.index()].def else {
                panic!("expected one-result producer")
            };
            &func.nodes[node]
        };
        let product = producer(func.results[0]);
        assert!(product.op.as_any().is::<xir::ops::Multiply>());
        let factor_position = product
            .inputs
            .iter()
            .position(|&value| producer(value).op.as_any().is::<xir::ops::Add>())
            .unwrap();
        let add = producer(product.inputs[factor_position]);
        assert_eq!(add.inputs.as_slice(), &[func.params[0], func.params[0]]);
        assert_eq!(
            func.value_shape(add.results[0]),
            func.value_shape(func.results[0])
        );
        let lane_sum = producer(product.inputs[1 - factor_position]);
        let lane_op = lane_sum.op.as_any().downcast_ref::<SumAlongMap>().unwrap();
        assert_eq!((lane_op.map.index(), lane_op.axis), (2, 1));
        let fiber_sum = producer(lane_sum.inputs[0]);
        let fiber_op = fiber_sum.op.as_any().downcast_ref::<SumAlongMap>().unwrap();
        assert_eq!((fiber_op.map.index(), fiber_op.axis), (0, 0));
        assert_eq!(fiber_sum.inputs.as_slice(), &[func.params[1]]);

        crate::scalarize::run(&mut lifted).unwrap();
        algebraic_pipeline().run(&mut lifted).unwrap();
        let factors = lower_domain_maps(&mut lifted).unwrap();
        algebraic_pipeline().run(&mut lifted).unwrap();

        assert_eq!(
            factors,
            vec![vec![vec![OutputFactor::Elementwise {
                op: xir::ops::ScalarOp::Add,
                dtype: DType::F32,
                inputs: vec![
                    OutputFactor::Input(InputElement {
                        operand: 0,
                        element: 0
                    }),
                    OutputFactor::Input(InputElement {
                        operand: 0,
                        element: 0
                    })
                ]
            }]]]
        );
        assert!(lifted.entry_func().nodes.iter().all(|node| !matches!(
            node.op.mnemonic(),
            "gather_along_map" | "sum_along_map" | "multiply"
        )));
    }

    #[test]
    fn algebraic_lowering_owes_the_coarse_input_at_the_output_write() {
        let program = crate::lower_program(projection_request()).unwrap();

        assert_eq!(
            program.output_factors(0, 0),
            &[OutputFactor::Elementwise {
                op: xir::ops::ScalarOp::Add,
                dtype: DType::F32,
                inputs: vec![
                    OutputFactor::Input(InputElement {
                        operand: 0,
                        element: 0
                    }),
                    OutputFactor::Input(InputElement {
                        operand: 0,
                        element: 0
                    })
                ]
            }]
        );
        assert!(
            program
                .module
                .entry_func()
                .nodes
                .iter()
                .all(|node| !matches!(
                    node.op.mnemonic(),
                    "gather_along_map" | "sum_along_map" | "multiply"
                ))
        );

        crate::search::compile_flat(
            projection_request(),
            crate::CompileOptions {
                shortlist_budget: None,
                target: Some(crate::backend::compiler::test_target()),
                only_element_order: Some(crate::ElementOrder::Trace),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn algebraic_lowering_keeps_both_projection_sides_for_recipe_choice() {
        let mut alternatives = crate::lower_program_frontier(projection_request()).unwrap();
        alternatives.sort_by_key(|alternative| {
            (
                alternative.finite_map_work().before_accumulation,
                alternative.finite_map_work().after_accumulation,
            )
        });

        assert!(alternatives.len() >= 2);
        let after = alternatives.first().unwrap();
        let before = alternatives.last().unwrap();
        assert_eq!(after.finite_map_work().before_accumulation, 0);
        assert!(after.finite_map_work().after_accumulation > 0);
        assert!(!after.output_factors(0, 0).is_empty());
        assert!(before.finite_map_work().before_accumulation > 0);
        assert_eq!(before.finite_map_work().after_accumulation, 0);
        assert!(before.output_factors(0, 0).is_empty());

        let (compiled, profile) = crate::search::compile_flat(
            projection_request(),
            crate::CompileOptions {
                shortlist_budget: None,
                target: Some(crate::backend::compiler::test_target()),
                only_element_order: Some(crate::ElementOrder::Trace),
                ..Default::default()
            },
        )
        .unwrap();
        let recipe::Work::Choice { candidates, .. } = &compiled.recipe.body().nodes()[0].work()
        else {
            panic!("graph frontier must lower to a recipe choice");
        };
        assert!(candidates.len() >= 3);
        assert_eq!(profile.arms[0].graph, 0);
        assert!(
            profile
                .frontier_graphs
                .iter()
                .any(|graph| graph.after_accumulation > 0)
        );
        assert!(
            profile
                .frontier_graphs
                .iter()
                .any(|graph| graph.before_accumulation > 0 && graph.after_accumulation == 0)
        );
    }

    #[test]
    fn unsupported_cheaper_sum_placements_do_not_starve_the_original() {
        let mut request = projection_request();
        let mut builder = ModuleBuilder::new();
        builder.begin_func("sum_linearity_frontier");
        let x = builder.param(Shape::scalar(DType::F32));
        let y = builder.param(Shape::scalar(DType::F32));
        let sum = builder.add(x, y);
        let entry = builder.end_func(vec![sum]);
        request.module = builder.finish(entry).unwrap();
        request.operands[0] = request.operands[1].clone();

        let program = LiftedProgram::try_from(request.clone()).unwrap();
        let mut lifted = lift_over_domains(&program).unwrap();
        crate::scalarize::run(&mut lifted).unwrap();
        let proofs = xir_optimize::extract_algebraic_frontier_with_limits(
            &lifted,
            xir_optimize::DEFAULT_MAX_ALGEBRAIC_WORK,
            xir_optimize::DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
        )
        .unwrap()
        .0;
        assert!(proofs.iter().any(|alternative| {
            alternative.work.before_accumulation == 0
                && validate_domain_map_lowering(&alternative.module).is_err()
        }));
        assert!(
            proofs
                .iter()
                .any(|alternative| { validate_domain_map_lowering(&alternative.module).is_ok() })
        );

        let alternatives = crate::lower_program_frontier(request.clone()).unwrap();
        assert!(!alternatives.is_empty());
        assert!(
            alternatives
                .iter()
                .all(|alternative| { alternative.finite_map_work().before_accumulation > 0 })
        );
        crate::search::compile_flat(
            request,
            crate::CompileOptions {
                shortlist_budget: None,
                target: Some(crate::backend::compiler::test_target()),
                only_element_order: Some(crate::ElementOrder::Trace),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn algebraic_lowering_owes_exact_content_as_a_constant_factor() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("exact_projection");
        let shape = Shape::scalar(DType::F32);
        let value = builder.param(shape.clone());
        let coefficient =
            builder.constant(Literal::exact(shape, vec![AlgebraicReal::sqrt(2u32)]).unwrap());
        let result = builder.multiply(coefficient, value);
        let entry = builder.end_func(vec![result]);
        let module = builder.finish(entry).unwrap();
        let binding = |axis_domain| FlatOperand {
            dtype: Dtype::F32,
            element_shape: Vec::new(),
            lane_extent: 1,
            axis_domains: vec![axis_domain],
        };
        let request = FlatProgram {
            module,
            lane_count: 1,
            domains: vec![Domain { extent: 3 }, Domain { extent: 8 }],
            index_maps: vec![IndexMap {
                source_id: 1,
                target_id: 0,
                is_injective: false,
                is_sorted: false,
                may_have_unmapped: false,
            }],
            num_axes: 1,
            operands: vec![binding(1), binding(0)],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "exact_projection".into(),
        };

        let program = crate::lower_program(request).unwrap();

        let [OutputFactor::Constant(literal)] = program.output_factors(0, 0) else {
            panic!("expected one constant output factor")
        };
        assert_eq!(
            literal.exact.as_ref().unwrap(),
            &[AlgebraicReal::sqrt(2u32)]
        );
    }

    #[test]
    fn lifting_preserves_called_functions_at_their_ids() {
        let mut builder = ModuleBuilder::new();
        let reducer = builder.func("add", |func| {
            let shape = Shape::scalar(DType::F32);
            let left = func.param(shape.clone());
            let right = func.param(shape);
            vec![func.add(left, right)]
        });
        let entry = builder.func("main", |func| {
            let input = func.param(Shape::new(DType::F32, [2]));
            let init = func.constant(
                Literal::exact(Shape::scalar(DType::F32), vec![AlgebraicReal::zero()]).unwrap(),
            );
            vec![func.reduce(input, init, &[0], reducer)]
        });
        let module = builder.finish(entry).unwrap();
        let binding = |visible_shape| FlatOperand {
            dtype: Dtype::F32,
            element_shape: visible_shape,
            lane_extent: 1,
            axis_domains: Vec::new(),
        };
        let program = LiftedProgram::try_from(FlatProgram {
            module,
            lane_count: 1,
            domains: Vec::new(),
            index_maps: Vec::new(),
            num_axes: 0,
            operands: vec![binding(vec![2]), binding(Vec::new())],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "called_function".into(),
        })
        .unwrap();

        let lifted = lift_over_domains(&program).unwrap();

        assert_eq!(lifted.entry.index(), 1);
        assert_eq!(lifted.funcs.len(), 2);
        assert_eq!(
            lifted.entry_func().nodes.last().unwrap().op.called_funcs(),
            &[reducer]
        );

        let mut lowered = lifted;
        lower_domain_maps(&mut lowered).unwrap();
        assert_eq!(lowered.entry.index(), 1);
        assert_eq!(lowered.funcs.len(), 2);
        assert_eq!(
            lowered.entry_func().nodes.last().unwrap().op.called_funcs(),
            &[reducer]
        );
    }
}
