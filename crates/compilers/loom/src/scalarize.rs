// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use xir::dtype::DType;
use xir::element_mapping::{DotPlan, axes, gather_map, num_elements, reduction_map, scatter_map};
use xir::ir::{FuncId, Module, Node, ValueId};
use xir::ops::{
    Add, Compare, ComparisonDirection, Constant, Convert, DotAlgorithm, DotGeneral, Gather,
    Maximum, Minimum, Multiply, PrecisionType, Reduce, Scatter, Select,
};
use xir::scalarize::{Context, CustomLowering, ScalarizedResults};
use xir::shape::Shape;

pub(super) fn validate_dots(module: &Module) -> Result<(), String> {
    let func = module.entry_func();
    for node in &func.nodes {
        let Some(dot) = node.op.as_any().downcast_ref::<DotGeneral>() else {
            continue;
        };
        validate_dot_algorithm(
            dot,
            func.value_shape(node.inputs[0]),
            func.value_shape(node.inputs[1]),
            func.value_shape(node.results[0]),
        )?;
    }
    Ok(())
}

fn validate_dot_algorithm(
    dot: &DotGeneral,
    lhs: &Shape,
    rhs: &Shape,
    output: &Shape,
) -> Result<(), String> {
    let precision = match lhs.dtype {
        DType::F16 => PrecisionType::F16,
        DType::Bf16 => PrecisionType::Bf16,
        DType::F32 => PrecisionType::F32,
        DType::F64 => PrecisionType::F64,
        dtype => return Err(format!("loom dot does not support {dtype:?} operands")),
    };
    let supported = DotAlgorithm {
        lhs: precision,
        rhs: precision,
        accum: precision,
        lhs_components: 1,
        rhs_components: 1,
        num_primitive_ops: 1,
        allow_imprecise_accum: false,
    };
    if dot.algorithm != supported || rhs.dtype != lhs.dtype || output.dtype != lhs.dtype {
        return Err(format!(
            "loom dot algorithm {:?} is unsupported for {} operands; expected {:?}",
            dot.algorithm,
            lhs.dtype.name(),
            supported
        ));
    }
    Ok(())
}

pub(super) fn run(module: &mut Module) -> Result<(), String> {
    validate_dots(module)?;
    xir::scalarize::run_with(module, &mut AlgebraicLowering)
        .map_err(|error| format!("loom {error}"))
}

struct AlgebraicLowering;

impl CustomLowering for AlgebraicLowering {
    fn lower(
        &mut self,
        context: &mut Context<'_>,
        node: &Node,
    ) -> Result<Option<ScalarizedResults>, String> {
        if let Some(op) = node.op.as_any().downcast_ref::<Reduce>() {
            return lower_reduce(context, node, op).map(Some);
        }
        if let Some(op) = node.op.as_any().downcast_ref::<Gather>() {
            return lower_gather(context, node, op).map(|values| Some(vec![values]));
        }
        if let Some(op) = node.op.as_any().downcast_ref::<Scatter>() {
            return lower_scatter(context, node, op).map(|values| Some(vec![values]));
        }
        if let Some(op) = node.op.as_any().downcast_ref::<DotGeneral>() {
            return lower_dot(context, node, op).map(|values| Some(vec![values]));
        }
        if let Some(compare) = node.op.as_any().downcast_ref::<Compare>()
            && matches!(
                compare.comparison_direction,
                ComparisonDirection::Gt | ComparisonDirection::Ge
            )
        {
            return lower_comparison(context, node, compare).map(|values| Some(vec![values]));
        }
        if node.op.as_any().is::<Minimum>() || node.op.as_any().is::<Maximum>() {
            return lower_minmax(context, node).map(|values| Some(vec![values]));
        }
        Ok(None)
    }
}

fn lower_comparison(
    context: &mut Context<'_>,
    node: &Node,
    compare: &Compare,
) -> Result<Vec<ValueId>, String> {
    let inputs = context.aligned_scalar_inputs(node)?;
    let direction = match compare.comparison_direction {
        ComparisonDirection::Gt => ComparisonDirection::Lt,
        ComparisonDirection::Ge => ComparisonDirection::Le,
        _ => unreachable!("only Gt and Ge reach Loom comparison lowering"),
    };
    (0..inputs[0].len())
        .map(|element| {
            context
                .builder()
                .try_op1(
                    Compare {
                        comparison_direction: direction,
                        compare_type: compare.compare_type,
                    },
                    &[inputs[1][element], inputs[0][element]],
                )
                .map_err(|error| format!("loom scalarization compare: {error}"))
        })
        .collect()
}

fn lower_minmax(context: &mut Context<'_>, node: &Node) -> Result<Vec<ValueId>, String> {
    let inputs = context.aligned_scalar_inputs(node)?;
    let [left, right] = inputs.as_slice() else {
        return Err("loom scalarization min/max arity mismatch".to_string());
    };
    (0..left.len())
        .map(|element| {
            let predicate = context
                .builder()
                .try_op1(
                    Compare {
                        comparison_direction: ComparisonDirection::Lt,
                        compare_type: None,
                    },
                    &[left[element], right[element]],
                )
                .map_err(|error| format!("loom scalarization compare: {error}"))?;
            let branches = if node.op.as_any().is::<Minimum>() {
                [predicate, left[element], right[element]]
            } else {
                [predicate, right[element], left[element]]
            };
            context
                .builder()
                .try_op1(Select, &branches)
                .map_err(|error| format!("loom scalarization select: {error}"))
        })
        .collect()
}

fn lower_reduce(
    context: &mut Context<'_>,
    node: &Node,
    op: &Reduce,
) -> Result<ScalarizedResults, String> {
    if node.inputs.len() != 2 || node.results.len() != 1 {
        return Err("loom scalarization supports one-input, one-result reduce".into());
    }
    let input = context.mapped(node.inputs[0])?.to_vec();
    let init = context.mapped(node.inputs[1])?.to_vec();
    if init.len() != 1 {
        return Err("loom scalarization reduce init is not scalar".into());
    }
    let in_dims = context.func().value_shape(node.inputs[0]).dims.clone();
    let (out_dims, cells) = reduction_map(&in_dims, &axes(&op.dimensions))?;
    let mut output = vec![init[0]; num_elements(&out_dims)?];
    for (flat, &value) in input.iter().enumerate() {
        let out = &mut output[cells[flat]];
        *out = lower_scalar_region(context, op.reducer, &[*out, value])?;
    }
    Ok(vec![output])
}

fn literal_i64(literal: &xir::ops::Literal) -> Result<Vec<i64>, String> {
    let bytes = literal.shape.dtype.size_bytes();
    literal
        .data
        .chunks_exact(bytes)
        .map(|value| {
            Ok(match literal.shape.dtype {
                DType::I8 => i8::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::I16 => i16::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::I32 => i32::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::I64 => i64::from_le_bytes(value.try_into().unwrap()),
                DType::U8 => u8::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::U16 => u16::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::U32 => u32::from_le_bytes(value.try_into().unwrap()) as i64,
                DType::U64 => i64::try_from(u64::from_le_bytes(value.try_into().unwrap()))
                    .map_err(|_| "index constant does not fit i64")?,
                dtype => return Err(format!("index constant has non-integer dtype {dtype:?}")),
            })
        })
        .collect()
}

fn constant_indices(context: &Context<'_>, value: ValueId) -> Result<Vec<i64>, String> {
    let literal = xir::analysis::constant_literal(context.func(), value)
        .ok_or_else(|| "indexed operation requires compile-time constant indices".to_string())?;
    literal_i64(literal)
}

fn lower_gather(
    context: &mut Context<'_>,
    node: &Node,
    op: &Gather,
) -> Result<Vec<ValueId>, String> {
    let input = context.mapped(node.inputs[0])?.to_vec();
    let indices = constant_indices(context, node.inputs[1])?;
    let map = gather_map(
        &context.func().value_shape(node.inputs[0]).dims,
        &context.func().value_shape(node.inputs[1]).dims,
        &indices,
        &context.func().value_shape(node.results[0]).dims,
        op,
    )?;
    Ok(map.into_iter().map(|index| input[index]).collect())
}

fn lower_scatter(
    context: &mut Context<'_>,
    node: &Node,
    op: &Scatter,
) -> Result<Vec<ValueId>, String> {
    let mut output = context.mapped(node.inputs[0])?.to_vec();
    let indices = constant_indices(context, node.inputs[1])?;
    let updates = context.mapped(node.inputs[2])?.to_vec();
    let map = scatter_map(
        &context.func().value_shape(node.inputs[0]).dims,
        &context.func().value_shape(node.inputs[1]).dims,
        &indices,
        &context.func().value_shape(node.inputs[2]).dims,
        op,
    )?;
    for (update, destination) in updates.into_iter().zip(map) {
        let Some(destination) = destination else {
            continue;
        };
        output[destination] =
            lower_scalar_region(context, op.update, &[output[destination], update])?;
    }
    Ok(output)
}

fn lower_scalar_region(
    context: &mut Context<'_>,
    function: FuncId,
    params: &[ValueId],
) -> Result<ValueId, String> {
    let func = context.module().func(function).clone();
    if func.params.len() != params.len() || func.results.len() != 1 {
        return Err("scalar region arity mismatch".into());
    }
    let mut values = func
        .params
        .iter()
        .copied()
        .zip(params.iter().copied())
        .collect::<std::collections::HashMap<_, _>>();
    for node in &func.nodes {
        if node.results.len() != 1 {
            return Err(format!(
                "scalar region `{}` must produce one result",
                node.op.mnemonic()
            ));
        }
        let inputs =
            node.inputs
                .iter()
                .map(|input| {
                    values.get(input).copied().ok_or_else(|| {
                        format!("scalar region input %{} is unavailable", input.index())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        if !node.op.as_any().is::<Constant>()
            && !node.op.as_any().is::<Convert>()
            && node.op.as_elementwise().is_none()
        {
            return Err(format!(
                "unsupported scalar region op `{}`",
                node.op.mnemonic()
            ));
        }
        let result = context
            .builder()
            .try_op1_boxed(node.op.clone(), &inputs)
            .map_err(|error| format!("scalar region `{}`: {error}", node.op.mnemonic()))?;
        values.insert(node.results[0], result);
    }
    values
        .get(&func.results[0])
        .copied()
        .ok_or_else(|| "scalar region result is unavailable".into())
}

fn lower_dot(
    context: &mut Context<'_>,
    node: &Node,
    op: &DotGeneral,
) -> Result<Vec<ValueId>, String> {
    let lhs_shape = context.func().value_shape(node.inputs[0]).clone();
    let rhs_shape = context.func().value_shape(node.inputs[1]).clone();
    let output_shape = context.func().value_shape(node.results[0]).clone();
    validate_dot_algorithm(op, &lhs_shape, &rhs_shape, &output_shape)?;
    let lhs = context.mapped(node.inputs[0])?.to_vec();
    let rhs = context.mapped(node.inputs[1])?.to_vec();
    let plan = DotPlan::new(&lhs_shape.dims, &rhs_shape.dims, op)?;
    let cells = num_elements(&plan.out_dims)?;
    let mut output = Vec::with_capacity(cells);
    for flat in 0..cells {
        let mut sum = None;
        for (left, right) in plan.contraction(flat)? {
            let product = context.builder().op1(Multiply, &[lhs[left], rhs[right]]);
            sum = Some(match sum {
                None => product,
                Some(sum) => context.builder().op1(Add, &[sum, product]),
            });
        }
        output.push(sum.ok_or_else(|| "loom dot has an empty contraction".to_string())?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use algebraic_real::AlgebraicReal;
    use xir::builder::ModuleBuilder;
    use xir::dim_numbers::DotDimensionNumbers;
    use xir::dtype::DType;
    use xir::ops::{Compare, ComparisonDirection, Constant, DotAlgorithm, Literal, Select};
    use xir::shape::Shape;

    use super::{run, validate_dots};
    use crate::optimization_pipeline;

    #[test]
    fn exact_polynomials_cross_tensor_views() {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let shape = Shape::new(DType::F32, [2, 2]);
            let x = func.param(shape.clone());
            let two = func.constant(
                Literal::exact(shape.clone(), vec![AlgebraicReal::integer(2); 4]).unwrap(),
            );
            let three =
                func.constant(Literal::exact(shape, vec![AlgebraicReal::integer(3); 4]).unwrap());
            let twice = func.multiply(x, two);
            let thrice = func.multiply(x, three);
            let twice_t = func.transpose(twice, &[1, 0]);
            let thrice_t = func.transpose(thrice, &[1, 0]);
            vec![func.add(twice_t, thrice_t)]
        });
        let mut module = builder.finish(main).unwrap();

        run(&mut module).unwrap();
        optimization_pipeline(crate::CompileOptions::default())
            .run(&mut module)
            .unwrap();

        let nodes = &module.entry_func().nodes;
        assert!(nodes.iter().all(|node| node.op.mnemonic() != "transpose"));
        assert!(nodes.iter().all(|node| node.op.mnemonic() != "add"));
        assert!(nodes.iter().any(|node| {
            node.op
                .as_any()
                .downcast_ref::<Constant>()
                .and_then(|constant| constant.literal.exact.as_ref())
                .is_some_and(|values| values == &[AlgebraicReal::integer(5)])
        }));
    }

    #[test]
    fn native_dot_is_scalarized() {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let lhs = func.param(Shape::new(DType::F32, [2]));
            let rhs = func.param(Shape::new(DType::F32, [2]));
            vec![func.dot_general(
                lhs,
                rhs,
                DotDimensionNumbers::new(&[0], &[0], &[], &[]),
                DotAlgorithm::f32_f32_f32(),
            )]
        });
        let mut module = builder.finish(main).unwrap();

        run(&mut module).unwrap();

        let nodes = &module.entry_func().nodes;
        assert!(nodes.iter().all(|node| node.op.mnemonic() != "dot_general"));
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "multiply")
                .count(),
            2
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "add")
                .count(),
            1
        );
    }

    #[test]
    fn additive_reduce_uses_the_loom_hook() {
        let mut builder = ModuleBuilder::new();
        let reducer = builder.func("add", |func| {
            let left = func.param(Shape::scalar(DType::F32));
            let right = func.param(Shape::scalar(DType::F32));
            vec![func.add(left, right)]
        });
        let main = builder.func("main", |func| {
            let input = func.param(Shape::new(DType::F32, [4]));
            let init = func.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
            vec![func.reduce(input, init, &[0], reducer)]
        });
        let mut module = builder.finish(main).unwrap();

        run(&mut module).unwrap();

        let entry = module.entry_func();
        assert!(
            entry
                .nodes
                .iter()
                .all(|node| node.op.mnemonic() != "reduce")
        );
        assert_eq!(
            entry
                .nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "add")
                .count(),
            4
        );
    }

    #[test]
    fn normalizes_ordered_comparisons_and_minmax() {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let shape = Shape::new(DType::F32, [2]);
            let left = func.param(shape.clone());
            let right = func.param(shape);
            let minimum = func.binary("minimum", left, right).unwrap();
            let maximum = func.maximum(left, right);
            let greater = func.compare(ComparisonDirection::Gt, None, left, right);
            let greater_equal = func.compare(ComparisonDirection::Ge, None, left, right);
            vec![minimum, maximum, greater, greater_equal]
        });
        let mut module = builder.finish(main).unwrap();

        run(&mut module).unwrap();

        let nodes = &module.entry_func().nodes;
        assert!(
            nodes
                .iter()
                .all(|node| { !matches!(node.op.mnemonic(), "minimum" | "maximum") })
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "select")
                .count(),
            4
        );
        assert!(
            nodes
                .iter()
                .filter_map(|node| {
                    node.op
                        .as_any()
                        .downcast_ref::<Compare>()
                        .map(|compare| compare.comparison_direction)
                })
                .all(|direction| matches!(
                    direction,
                    ComparisonDirection::Lt | ComparisonDirection::Le
                ))
        );
    }

    #[test]
    fn piecewise_optimization_crosses_scalarization() {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let shape = Shape::new(DType::F32, [2]);
            let a = func.param(shape.clone());
            let b = func.param(shape.clone());
            let c = func.param(shape.clone());
            let d = func.param(shape);
            let predicate = func.compare(ComparisonDirection::Lt, None, a, b);
            let left = func.op1(Select, &[predicate, a, b]);
            let right = func.op1(Select, &[predicate, c, d]);
            vec![func.add(left, right)]
        });
        let mut module = builder.finish(main).unwrap();

        run(&mut module).unwrap();
        optimization_pipeline(crate::CompileOptions::default())
            .run(&mut module)
            .unwrap();

        assert_eq!(
            module
                .entry_func()
                .nodes
                .iter()
                .filter(|node| node.op.mnemonic() == "select")
                .count(),
            2
        );
    }

    #[test]
    fn non_native_dot_algorithm_is_rejected() {
        let module = vector_dot_module(2, DotAlgorithm::tf32_tf32_f32());

        let error = validate_dots(&module).unwrap_err();

        assert!(error.contains("unsupported"), "{error}");
        assert!(error.contains("Tf32"), "{error}");
    }

    fn vector_dot_module(size: i64, algorithm: DotAlgorithm) -> xir::ir::Module {
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let lhs = func.param(Shape::new(DType::F32, [size]));
            let rhs = func.param(Shape::new(DType::F32, [size]));
            vec![func.dot_general(
                lhs,
                rhs,
                DotDimensionNumbers::new(&[0], &[0], &[], &[]),
                algorithm,
            )]
        });
        builder.finish(main).unwrap()
    }
}
