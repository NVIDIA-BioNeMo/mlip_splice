// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantics-preserving tensor-to-scalar decomposition with target hooks for
//! operations whose lowering depends on the caller's math policy.

use crate::builder::ModuleBuilder;
use crate::element_mapping::{
    broadcast_elements, concatenate_elements, num_elements, pad_elements, slice_elements,
    transpose_elements, unravel,
};
use crate::ir::{Func, Module, Node, ValueId};
use crate::ops::{
    BroadcastInDim, Concatenate, Constant, Convert, GatherAlongMap, Literal, Pad, Reshape, Slice,
    SumAlongMap, Transpose,
};
use crate::shape::{Dims, Shape};
use crate::verify;

pub type ScalarizedResults = Vec<Vec<ValueId>>;

pub trait CustomLowering {
    /// Return `None` without emitting nodes when the generic lowering should
    /// handle `node`.
    fn lower(
        &mut self,
        _context: &mut Context<'_>,
        _node: &Node,
    ) -> Result<Option<ScalarizedResults>, String> {
        Ok(None)
    }
}

pub struct Context<'a> {
    builder: &'a mut ModuleBuilder,
    module: &'a Module,
    func: &'a Func,
    values: &'a [Option<Vec<ValueId>>],
}

impl Context<'_> {
    pub fn builder(&mut self) -> &mut ModuleBuilder {
        self.builder
    }

    pub fn module(&self) -> &Module {
        self.module
    }

    pub fn func(&self) -> &Func {
        self.func
    }

    pub fn mapped(&self, value: ValueId) -> Result<&[ValueId], String> {
        self.values[value.index()]
            .as_deref()
            .ok_or_else(|| format!("scalarization value %{} is unavailable", value.index()))
    }

    /// Return one row-major scalar vector per input, aligned to the node's
    /// single result. A one-element input repeats across the result; all other
    /// inputs must have the result's element count. Domain axes are not expanded.
    pub fn aligned_scalar_inputs(&self, node: &Node) -> Result<Vec<Vec<ValueId>>, String> {
        if node.results.len() != 1 {
            return Err(format!(
                "scalarization requires one result for `{}`",
                node.op.mnemonic()
            ));
        }
        let count = num_elements(&self.func().value_shape(node.results[0]).dims)?;
        let inputs = node
            .inputs
            .iter()
            .map(|&input| self.mapped(input).map(<[ValueId]>::to_vec))
            .collect::<Result<Vec<_>, String>>()?;
        if inputs
            .iter()
            .any(|input| input.len() != 1 && input.len() != count)
        {
            return Err(format!(
                "scalarization element count mismatch for `{}`",
                node.op.mnemonic()
            ));
        }
        Ok(inputs
            .into_iter()
            .map(|input| {
                if input.len() == 1 {
                    vec![input[0]; count]
                } else {
                    input
                }
            })
            .collect())
    }
}

pub fn run(module: &mut Module) -> Result<(), String> {
    run_with(module, &mut PreserveSemantics)
}

struct PreserveSemantics;

impl CustomLowering for PreserveSemantics {}

pub fn run_with(module: &mut Module, custom: &mut impl CustomLowering) -> Result<(), String> {
    let funcs = module
        .funcs
        .iter()
        .map(|old| scalarize_func(module, old, custom))
        .collect::<Result<Vec<_>, _>>()?;
    let scalarized = Module {
        finite_map_graph: module.finite_map_graph.clone(),
        funcs,
        entry: module.entry,
    };
    verify::verify(&scalarized).map_err(|error| format!("scalarized XIR verification: {error}"))?;
    *module = scalarized;
    Ok(())
}

fn scalarize_func(
    module: &Module,
    old: &Func,
    custom: &mut impl CustomLowering,
) -> Result<Func, String> {
    let mut builder = ModuleBuilder::with_finite_map_graph(module.finite_map_graph.clone());
    builder.begin_func(&old.name);
    let mut values = vec![None; old.values.len()];
    for &param in &old.params {
        let shape = old.value_shape(param);
        let new = builder.param(shape.clone());
        values[param.index()] = Some(unpack(&mut builder, new, shape)?);
    }

    for node in &old.nodes {
        let mut context = Context {
            builder: &mut builder,
            module,
            func: old,
            values: &values,
        };
        let lowered = lower_node(&mut context, node, custom)?;
        if lowered.len() != node.results.len() {
            return Err(format!(
                "scalarization of `{}` produced {} results, expected {}",
                node.op.mnemonic(),
                lowered.len(),
                node.results.len()
            ));
        }
        for (&result, elements) in node.results.iter().zip(lowered) {
            values[result.index()] = Some(elements);
        }
    }

    let results = old
        .results
        .iter()
        .map(|&result| {
            pack(
                &mut builder,
                mapped(&values, result)?,
                old.value_shape(result),
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let entry = builder.end_func(results);
    let mut funcs = builder
        .finish(entry)
        .map_err(|error| format!("scalarized function verification: {error}"))?
        .funcs;
    Ok(funcs.remove(0))
}

fn mapped(values: &[Option<Vec<ValueId>>], value: ValueId) -> Result<&[ValueId], String> {
    values[value.index()]
        .as_deref()
        .ok_or_else(|| format!("scalarization value %{} is unavailable", value.index()))
}

fn lower_node(
    context: &mut Context<'_>,
    node: &Node,
    custom: &mut impl CustomLowering,
) -> Result<ScalarizedResults, String> {
    if let Some(lowered) = custom.lower(context, node)? {
        return Ok(lowered);
    }
    if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
        return Ok(vec![scalar_constants(context.builder, &constant.literal)?]);
    }
    if node.op.as_any().is::<Reshape>() {
        return Ok(vec![context.mapped(node.inputs[0])?.to_vec()]);
    }
    if let Some(op) = node.op.as_any().downcast_ref::<BroadcastInDim>() {
        return Ok(vec![broadcast_elements(
            context.mapped(node.inputs[0])?,
            &context.func.value_shape(node.inputs[0]).dims,
            &op.out_dims,
            &op.broadcast_dimensions,
        )?]);
    }
    if let Some(op) = node.op.as_any().downcast_ref::<Transpose>() {
        return Ok(vec![transpose_elements(
            context.mapped(node.inputs[0])?,
            &context.func.value_shape(node.inputs[0]).dims,
            &op.permutation,
        )?]);
    }
    if let Some(op) = node.op.as_any().downcast_ref::<Slice>() {
        return Ok(vec![slice_elements(
            context.mapped(node.inputs[0])?,
            &context.func.value_shape(node.inputs[0]).dims,
            op,
        )?]);
    }
    if let Some(op) = node.op.as_any().downcast_ref::<Concatenate>() {
        let inputs = node
            .inputs
            .iter()
            .map(|&input| {
                Ok((
                    context.mapped(input)?,
                    &context.func.value_shape(input).dims,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        return Ok(vec![concatenate_elements(&inputs, op.dimension as usize)?]);
    }
    if let Some(op) = node.op.as_any().downcast_ref::<Pad>() {
        let pad = context.mapped(node.inputs[1])?;
        if pad.len() != 1 {
            return Err("scalarization pad value is not scalar".into());
        }
        return Ok(vec![pad_elements(
            context.mapped(node.inputs[0])?,
            &context.func.value_shape(node.inputs[0]).dims,
            &pad[0],
            op,
        )?]);
    }
    if node.op.as_any().is::<Convert>() || node.op.as_elementwise().is_some() {
        let inputs = context.aligned_scalar_inputs(node)?;
        let count = num_elements(&context.func.value_shape(node.results[0]).dims)?;
        let output = (0..count)
            .map(|element| {
                let scalar_inputs = inputs
                    .iter()
                    .map(|input| input[element])
                    .collect::<Vec<_>>();
                context
                    .builder
                    .try_op1_boxed(node.op.clone(), &scalar_inputs)
                    .map_err(|error| format!("scalarization `{}`: {error}", node.op.mnemonic()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        return Ok(vec![output]);
    }
    if node.op.as_any().is::<GatherAlongMap>() || node.op.as_any().is::<SumAlongMap>() {
        let inputs = context.mapped(node.inputs[0])?.to_vec();
        let output = inputs
            .iter()
            .map(|&input| {
                context
                    .builder
                    .try_op1_boxed(node.op.clone(), &[input])
                    .map_err(|error| format!("scalarization `{}`: {error}", node.op.mnemonic()))
            })
            .collect::<Result<Vec<_>, String>>()?;
        return Ok(vec![output]);
    }
    Err(format!(
        "unsupported XIR op during scalarization: `{}`",
        node.op.mnemonic()
    ))
}

fn scalar_constants(
    builder: &mut ModuleBuilder,
    literal: &Literal,
) -> Result<Vec<ValueId>, String> {
    let count = num_elements(&literal.shape.dims)?;
    let bytes = literal.shape.dtype.size_bytes();
    (0..count)
        .map(|element| {
            let shape = Shape::scalar(literal.shape.dtype)
                .with_domain_axes(literal.shape.domain_axes.iter().copied());
            let scalar = match &literal.exact {
                Some(exact) => Literal::exact(shape, vec![exact[element].clone()])?,
                None => Literal::new(
                    shape,
                    literal.data[element * bytes..(element + 1) * bytes].to_vec(),
                ),
            };
            Ok(builder.op1(Constant { literal: scalar }, &[]))
        })
        .collect()
}

fn unpack(
    builder: &mut ModuleBuilder,
    value: ValueId,
    shape: &Shape,
) -> Result<Vec<ValueId>, String> {
    if shape.is_scalar() {
        return Ok(vec![value]);
    }
    let count = num_elements(&shape.dims)?;
    (0..count)
        .map(|flat| {
            let coordinate = unravel(flat, &shape.dims)?;
            let start_indices = coordinate.iter().map(|&value| value as i64).collect();
            let limit_indices = coordinate.iter().map(|&value| value as i64 + 1).collect();
            let strides = std::iter::repeat_n(1, shape.rank()).collect();
            let slice = builder.op1(
                Slice {
                    start_indices,
                    limit_indices,
                    strides,
                },
                &[value],
            );
            Ok(builder.op1(
                Reshape {
                    out_dims: Dims::new(),
                },
                &[slice],
            ))
        })
        .collect()
}

fn pack(
    builder: &mut ModuleBuilder,
    elements: &[ValueId],
    shape: &Shape,
) -> Result<ValueId, String> {
    if shape.is_scalar() {
        return elements
            .first()
            .copied()
            .ok_or_else(|| "scalarization produced no scalar output".into());
    }
    if elements.len() != num_elements(&shape.dims)? {
        return Err("scalarization output element count mismatch".into());
    }
    let vectors = elements
        .iter()
        .map(|&element| {
            builder.op1(
                Reshape {
                    out_dims: Dims::from_slice(&[1]),
                },
                &[element],
            )
        })
        .collect::<Vec<_>>();
    let flat = if vectors.len() == 1 {
        vectors[0]
    } else {
        builder.op1(Concatenate { dimension: 0 }, &vectors)
    };
    Ok(builder.op1(
        Reshape {
            out_dims: shape.dims.clone(),
        },
        &[flat],
    ))
}
