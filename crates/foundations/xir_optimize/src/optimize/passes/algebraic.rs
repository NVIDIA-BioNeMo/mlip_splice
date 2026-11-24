// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::SmallVec;

use super::fold::literal_bytes;
use crate::ir::{Module, ValueId};
use crate::op::Op;
use crate::ops::{Add, BroadcastInDim, Constant, Literal, Multiply, Negate, ScalarOp, Subtract};
use crate::optimize::rebuild::Rebuilder;

use crate::optimize::AlgebraicWorkBudget;
use crate::shape::Shape;

/// Algebraic canonicalization treats floating-point values as approximate real
/// algebra through bounded local rewrites.
const ALGEBRAIC_MAX_SWEEPS: usize = 16;

pub(in crate::optimize) fn algebraic_canonicalize(
    module: &mut Module,
    max_literal_bytes: usize,
    work: &mut AlgebraicWorkBudget,
) -> Result<bool, String> {
    let mut changed = false;
    for _ in 0..ALGEBRAIC_MAX_SWEEPS {
        if !algebraic_sweep(module, max_literal_bytes, work)? {
            break;
        }
        changed = true;
    }
    Ok(changed)
}

fn algebraic_sweep(
    module: &mut Module,
    max_literal_bytes: usize,
    work: &mut AlgebraicWorkBudget,
) -> Result<bool, String> {
    let mut changed = false;
    let funcs = module
        .funcs
        .iter()
        .enumerate()
        .map(|(function_index, old)| {
            let mut new = Rebuilder::new(old);
            for node in &old.nodes {
                let inputs = new.mapped_inputs(node)?;
                let rewrite = (node.results.len() == 1)
                    .then(|| {
                        algebraic_rewrite(
                            &new,
                            node.op.as_ref(),
                            &inputs,
                            old.value_shape(node.results[0]),
                            max_literal_bytes,
                            (function_index, node.results[0].0),
                            work,
                        )
                    })
                    .flatten();
                match rewrite {
                    Some(AlgebraicRewrite::Alias(value)) => {
                        new.alias(&node.results, &[value])?;
                        changed = true;
                    }
                    Some(AlgebraicRewrite::Materialize(literal)) => {
                        new.emit(
                            old,
                            &node.results,
                            std::sync::Arc::new(Constant { literal }),
                            SmallVec::new(),
                        )?;
                        changed = true;
                    }
                    Some(AlgebraicRewrite::Rebuild(op, inputs)) => {
                        new.emit(old, &node.results, op.into(), inputs)?;
                        changed = true;
                    }
                    None => {
                        new.emit(old, &node.results, node.op.clone(), inputs)?;
                    }
                }
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}

enum AlgebraicRewrite {
    Alias(ValueId),
    Materialize(Literal),
    Rebuild(Box<dyn Op>, SmallVec<[ValueId; 4]>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FloatUniform {
    Zero,
    One,
}

fn algebraic_rewrite(
    values: &Rebuilder,
    op: &dyn Op,
    inputs: &[ValueId],
    output_shape: &Shape,
    max_literal_bytes: usize,
    expression: (usize, u32),
    work: &mut AlgebraicWorkBudget,
) -> Option<AlgebraicRewrite> {
    let scalar = op.as_elementwise()?.scalar();
    if !output_shape.dtype.is_float() {
        return None;
    }
    if let Some(literal) = fold_exact_elementwise(
        values,
        inputs,
        output_shape,
        scalar,
        max_literal_bytes,
        expression,
        work,
    ) {
        return Some(AlgebraicRewrite::Materialize(literal));
    }
    match scalar {
        ScalarOp::Add => {
            if float_uniform(values, inputs[1]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Alias(inputs[0]));
            }
            if float_uniform(values, inputs[0]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Alias(inputs[1]));
            }
            if let Some((_, y)) = values.producer::<Negate>(inputs[1]) {
                return Some(AlgebraicRewrite::Rebuild(
                    Box::new(Subtract),
                    smallvec_pair(inputs[0], y),
                ));
            }
            if let Some((_, x)) = values.producer::<Negate>(inputs[0]) {
                return Some(AlgebraicRewrite::Rebuild(
                    Box::new(Subtract),
                    smallvec_pair(inputs[1], x),
                ));
            }
            commute(Box::new(Add), inputs)
        }
        ScalarOp::Subtract => {
            if float_uniform(values, inputs[1]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Alias(inputs[0]));
            }
            if float_uniform(values, inputs[0]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Rebuild(
                    Box::new(Negate),
                    [inputs[1]].into_iter().collect(),
                ));
            }
            if let Some((_, y)) = values.producer::<Negate>(inputs[1]) {
                return Some(AlgebraicRewrite::Rebuild(
                    Box::new(Add),
                    smallvec_pair(inputs[0], y),
                ));
            }
            None
        }
        ScalarOp::Multiply => {
            if float_uniform(values, inputs[0]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Alias(inputs[0]));
            }
            if float_uniform(values, inputs[1]) == Some(FloatUniform::Zero) {
                return Some(AlgebraicRewrite::Alias(inputs[1]));
            }
            if float_uniform(values, inputs[1]) == Some(FloatUniform::One) {
                return Some(AlgebraicRewrite::Alias(inputs[0]));
            }
            if float_uniform(values, inputs[0]) == Some(FloatUniform::One) {
                return Some(AlgebraicRewrite::Alias(inputs[1]));
            }
            commute(Box::new(Multiply), inputs)
        }
        ScalarOp::Divide => {
            if float_uniform(values, inputs[1]) == Some(FloatUniform::One) {
                return Some(AlgebraicRewrite::Alias(inputs[0]));
            }
            None
        }
        ScalarOp::Negate => values
            .producer::<Negate>(inputs[0])
            .map(|(_, x)| AlgebraicRewrite::Alias(x)),
        _ => None,
    }
}

fn fold_exact_elementwise(
    values: &Rebuilder,
    inputs: &[ValueId],
    output_shape: &Shape,
    op: ScalarOp,
    max_literal_bytes: usize,
    expression: (usize, u32),
    work: &mut AlgebraicWorkBudget,
) -> Option<Literal> {
    if literal_bytes(output_shape).ok().flatten()? > max_literal_bytes {
        return None;
    }
    let literals = inputs
        .iter()
        .map(|&input| values.literal(input))
        .collect::<Option<Vec<_>>>()?;
    let exact = literals
        .iter()
        .map(Literal::algebraic_values)
        .collect::<Option<Vec<_>>>()?;
    let elements = output_shape.num_elements()?;
    if exact.iter().any(|values| values.len() != elements) {
        return None;
    }
    let operands = (0..elements)
        .map(|element| {
            exact
                .iter()
                .map(|values| &values[element])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cost = operands.iter().try_fold(0usize, |cost, operands| {
        let element_cost = match op {
            ScalarOp::Negate => operands[0].estimated_work(),
            ScalarOp::Add | ScalarOp::Subtract => operands[0]
                .estimated_work()
                .checked_add(operands[1].estimated_work())?,
            ScalarOp::Multiply | ScalarOp::Divide => operands[0]
                .estimated_work()
                .checked_mul(operands[1].estimated_work())?,
            _ => return None,
        };
        cost.checked_add(element_cost)
    })?;
    if !work.reserve(expression, cost) {
        return None;
    }
    let mut result = Vec::with_capacity(elements);
    for operands in operands {
        let value = match op {
            ScalarOp::Negate => -operands[0],
            ScalarOp::Add => operands[0] + operands[1],
            ScalarOp::Subtract => operands[0] - operands[1],
            ScalarOp::Multiply => operands[0] * operands[1],
            ScalarOp::Divide => {
                let reciprocal = operands[1].recip()?;
                operands[0] * &reciprocal
            }
            _ => return None,
        };
        result.push(value);
    }
    Literal::exact(output_shape.clone(), result).ok()
}

/// Order a commutative binary op's operands by value id so that `a ∘ b` and
/// `b ∘ a` become the same node and collapse under CSE.
fn commute(op: Box<dyn Op>, inputs: &[ValueId]) -> Option<AlgebraicRewrite> {
    (inputs[0].index() > inputs[1].index())
        .then(|| AlgebraicRewrite::Rebuild(op, smallvec_pair(inputs[1], inputs[0])))
}

fn smallvec_pair(a: ValueId, b: ValueId) -> SmallVec<[ValueId; 4]> {
    [a, b].into_iter().collect()
}

/// The uniform floating value carried by `value`, when it provably holds `+0.0`
/// or `1.0` in every element (directly as a constant or through a broadcast).
fn float_uniform(values: &Rebuilder, value: ValueId) -> Option<FloatUniform> {
    if let Some(uniform) = values
        .literal(value)
        .and_then(|literal| literal_uniform(&literal))
    {
        return Some(uniform);
    }
    let (_, source) = values.producer::<BroadcastInDim>(value)?;
    values
        .literal(source)
        .and_then(|literal| literal_uniform(&literal))
}

fn literal_uniform(literal: &Literal) -> Option<FloatUniform> {
    let dtype = literal.shape.dtype;
    if !dtype.is_float() {
        return None;
    }
    let bytes = dtype.size_bytes();
    if literal.data.is_empty() || !literal.data.len().is_multiple_of(bytes) {
        return None;
    }
    let mut elements = literal.data.chunks_exact(bytes);
    if elements
        .clone()
        .all(|element| element.iter().all(|&b| b == 0))
    {
        return Some(FloatUniform::Zero);
    }
    let one = float_one_bytes(dtype);
    if elements.all(|element| element == one.as_slice()) {
        return Some(FloatUniform::One);
    }
    None
}

/// Little-endian bit pattern of `1.0` in each floating dtype.
fn float_one_bytes(dtype: crate::dtype::DType) -> Vec<u8> {
    use crate::dtype::DType;
    match dtype {
        DType::F16 => 0x3C00u16.to_le_bytes().to_vec(),
        DType::Bf16 => 0x3F80u16.to_le_bytes().to_vec(),
        DType::F32 => 1.0f32.to_le_bytes().to_vec(),
        DType::F64 => 1.0f64.to_le_bytes().to_vec(),
        _ => Vec::new(),
    }
}
