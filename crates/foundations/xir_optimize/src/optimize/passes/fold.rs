// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use algebraic_real::AlgebraicReal;
use smallvec::SmallVec;

use crate::ir::{Module, ValueId};
use crate::op::Op;
use crate::ops::{
    BroadcastInDim, Compare, ComparisonDirection, ComparisonType, Concatenate, Constant, Literal,
    Pad, Reshape, ScalarOp, Select, Slice, Transpose,
};
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;

pub(in crate::optimize) fn constant_folding(
    module: &mut Module,
    max_literal_bytes: usize,
) -> Result<bool, String> {
    let mut changed = false;
    let funcs = module
        .funcs
        .iter()
        .map(|old| {
            let mut new = Rebuilder::new(old);
            for node in &old.nodes {
                let inputs = new.mapped_inputs(node)?;
                // Every folding rule reads at least one operand as a constant
                // literal, so a node whose rebuilt operands are all non-constant
                // cannot fold. Skipping fold_node here avoids its op-downcast
                // chain on the overwhelming majority of nodes in a large scalar
                // program, where nothing is constant.
                let has_constant_input = inputs.iter().any(|&value| new.literal(value).is_some());
                let folded = (has_constant_input && node.results.len() == 1)
                    .then(|| {
                        fold_node(
                            &new,
                            node.op.as_ref(),
                            &inputs,
                            old.value_shape(node.results[0]),
                            max_literal_bytes,
                        )
                    })
                    .flatten();
                if let Some(literal) = folded {
                    new.emit(
                        old,
                        &node.results,
                        std::sync::Arc::new(Constant { literal }),
                        SmallVec::new(),
                    )?;
                    changed = true;
                } else {
                    new.emit(old, &node.results, node.op.clone(), inputs)?;
                }
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}

/// The literal `op` folds to over already-rebuilt constant operands, or `None`
/// when it does not fold — an operand is not constant, the result would exceed
/// `max_literal_bytes`, or the op is not one of the foldable forms. This is a
/// total best-effort query: constant folding treats every `None` as "leave this
/// node unchanged", so a node it cannot fold skips only its own rewrite and can
/// never abort the pass. The remaining structural guards inside the fold
/// routines can only fail on IR that entry verification already rejected; a
/// failure there is likewise reported as "does not fold".
pub(in crate::optimize) fn fold_node(
    values: &Rebuilder,
    op: &dyn Op,
    inputs: &[ValueId],
    output_shape: &Shape,
    max_literal_bytes: usize,
) -> Option<Literal> {
    let output_bytes = literal_bytes(output_shape).ok().flatten()?;
    if output_bytes > max_literal_bytes || op.as_any().is::<Constant>() {
        return None;
    }

    let folded = if op.as_any().is::<Reshape>() {
        values.literal(inputs[0]).map(|input| input.data)
    } else if let Some(transpose) = op.as_any().downcast_ref::<Transpose>() {
        values
            .literal(inputs[0])
            .and_then(|input| transpose_literal(&input, output_shape, &transpose.permutation).ok())
    } else if let Some(broadcast) = op.as_any().downcast_ref::<BroadcastInDim>() {
        values.literal(inputs[0]).and_then(|input| {
            broadcast_literal(&input, output_shape, &broadcast.broadcast_dimensions).ok()
        })
    } else if let Some(slice) = op.as_any().downcast_ref::<Slice>() {
        values
            .literal(inputs[0])
            .and_then(|input| slice_literal(&input, output_shape, slice).ok())
    } else if let Some(concatenate) = op.as_any().downcast_ref::<Concatenate>() {
        inputs
            .iter()
            .map(|&input| values.literal(input))
            .collect::<Option<Vec<_>>>()
            .and_then(|inputs| {
                concatenate_literals(&inputs, output_shape, concatenate.dimension as usize).ok()
            })
    } else if let Some(pad) = op.as_any().downcast_ref::<Pad>() {
        values
            .literal(inputs[0])
            .zip(values.literal(inputs[1]))
            .and_then(|(input, padding)| pad_literal(&input, &padding, output_shape, pad).ok())
    } else if op.as_any().is::<Select>() {
        fold_select(values, inputs, output_shape).ok().flatten()
    } else if let Some(elementwise) = op.as_elementwise() {
        fold_integer_elementwise(
            values,
            inputs,
            output_shape,
            elementwise.scalar(),
            op.as_any()
                .downcast_ref::<Compare>()
                .and_then(|compare| compare.compare_type),
        )
        .ok()
        .flatten()
    } else {
        None
    };
    let exact = fold_structural_exact(values, op, inputs, output_shape)
        .ok()
        .flatten();
    folded.and_then(|data| {
        let literal = Literal::new(output_shape.clone(), data);
        match exact {
            Some(values) => literal.with_exact(values).ok(),
            None => Some(literal),
        }
    })
}

fn fold_structural_exact(
    values: &Rebuilder,
    op: &dyn Op,
    inputs: &[ValueId],
    output_shape: &Shape,
) -> Result<Option<Vec<AlgebraicReal>>, String> {
    if op.as_any().is::<Reshape>() {
        return Ok(values.literal(inputs[0]).and_then(|literal| literal.exact));
    }
    if let Some(transpose) = op.as_any().downcast_ref::<Transpose>() {
        return remap_exact(values.literal(inputs[0]), output_shape, |output| {
            let mut source = vec![0usize; output.len()];
            for (axis, &coordinate) in output.iter().enumerate() {
                source[transpose.permutation[axis] as usize] = coordinate;
            }
            Ok(source)
        });
    }
    if let Some(broadcast) = op.as_any().downcast_ref::<BroadcastInDim>() {
        let input = values.literal(inputs[0]);
        return remap_exact(input.clone(), output_shape, |output| {
            let input = input
                .as_ref()
                .ok_or_else(|| "missing literal".to_string())?;
            Ok(broadcast
                .broadcast_dimensions
                .iter()
                .enumerate()
                .map(|(axis, &output_axis)| {
                    if input.shape.dims[axis] == 1 {
                        0
                    } else {
                        output[output_axis as usize]
                    }
                })
                .collect())
        });
    }
    if let Some(slice) = op.as_any().downcast_ref::<Slice>() {
        return remap_exact(values.literal(inputs[0]), output_shape, |output| {
            output
                .iter()
                .enumerate()
                .map(|(axis, &coordinate)| {
                    usize::try_from(
                        slice.start_indices[axis]
                            + i64::try_from(coordinate).map_err(|error| error.to_string())?
                                * slice.strides[axis],
                    )
                    .map_err(|error| error.to_string())
                })
                .collect()
        });
    }
    if let Some(concatenate) = op.as_any().downcast_ref::<Concatenate>() {
        let Some(literals) = inputs
            .iter()
            .map(|&input| values.literal(input))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        return concatenate_exact(&literals, output_shape, concatenate.dimension as usize);
    }
    if let Some(pad) = op.as_any().downcast_ref::<Pad>() {
        let Some((input, padding)) = values.literal(inputs[0]).zip(values.literal(inputs[1]))
        else {
            return Ok(None);
        };
        return pad_exact(&input, &padding, output_shape, pad);
    }
    if op.as_any().is::<Select>() {
        return select_exact(values, inputs, output_shape);
    }
    Ok(None)
}

fn remap_exact(
    input: Option<Literal>,
    output_shape: &Shape,
    source_coordinate: impl Fn(&[usize]) -> Result<Vec<usize>, String>,
) -> Result<Option<Vec<AlgebraicReal>>, String> {
    let Some(input) = input else {
        return Ok(None);
    };
    let Some(values) = input.exact else {
        return Ok(None);
    };
    let output_elements = output_shape.element_count()?;
    let mut exact = Vec::with_capacity(output_elements);
    for flat in 0..output_elements {
        let output = unravel(flat, &output_shape.dims)?;
        let source = ravel(&source_coordinate(&output)?, &input.shape.dims)?;
        exact.push(values[source].clone());
    }
    Ok(Some(exact))
}

fn concatenate_exact(
    inputs: &[Literal],
    output_shape: &Shape,
    dimension: usize,
) -> Result<Option<Vec<AlgebraicReal>>, String> {
    let Some(values) = inputs
        .iter()
        .map(|literal| literal.exact.as_ref())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let mut exact = Vec::with_capacity(output_shape.element_count()?);
    for flat in 0..output_shape.element_count()? {
        let mut coordinate = unravel(flat, &output_shape.dims)?;
        let mut offset = coordinate[dimension];
        for (input, values) in inputs.iter().zip(&values) {
            let extent =
                usize::try_from(input.shape.dims[dimension]).map_err(|error| error.to_string())?;
            if offset < extent {
                coordinate[dimension] = offset;
                exact.push(values[ravel(&coordinate, &input.shape.dims)?].clone());
                break;
            }
            offset -= extent;
        }
    }
    Ok(Some(exact))
}

fn pad_exact(
    input: &Literal,
    padding: &Literal,
    output_shape: &Shape,
    pad: &Pad,
) -> Result<Option<Vec<AlgebraicReal>>, String> {
    let (Some(input_values), Some(padding_values)) = (&input.exact, &padding.exact) else {
        return Ok(None);
    };
    if padding_values.len() != 1 {
        return Err("XIR exact pad literal is not scalar".into());
    }
    let mut exact = Vec::with_capacity(output_shape.element_count()?);
    for flat in 0..output_shape.element_count()? {
        let output = unravel(flat, &output_shape.dims)?;
        let mut source = Vec::with_capacity(output.len());
        let mut is_padding = false;
        for (axis, &coordinate) in output.iter().enumerate() {
            let coordinate = i64::try_from(coordinate).map_err(|error| error.to_string())?;
            let expanded = coordinate - pad.edge_padding_low[axis];
            let step = pad.interior_padding[axis] + 1;
            if expanded < 0 || expanded % step != 0 {
                is_padding = true;
                break;
            }
            let input_coordinate = expanded / step;
            if input_coordinate >= input.shape.dims[axis] {
                is_padding = true;
                break;
            }
            source.push(usize::try_from(input_coordinate).map_err(|error| error.to_string())?);
        }
        exact.push(if is_padding {
            padding_values[0].clone()
        } else {
            input_values[ravel(&source, &input.shape.dims)?].clone()
        });
    }
    Ok(Some(exact))
}

fn select_exact(
    values: &Rebuilder,
    inputs: &[ValueId],
    output_shape: &Shape,
) -> Result<Option<Vec<AlgebraicReal>>, String> {
    let Some(predicate) = values.literal(inputs[0]) else {
        return Ok(None);
    };
    if predicate.shape.is_scalar() {
        let selected = if predicate.data[0] == 0 {
            inputs[2]
        } else {
            inputs[1]
        };
        return Ok(values.literal(selected).and_then(|literal| literal.exact));
    }
    let (Some(yes), Some(no)) = (values.literal(inputs[1]), values.literal(inputs[2])) else {
        return Ok(None);
    };
    let (Some(yes_values), Some(no_values)) = (yes.exact, no.exact) else {
        return Ok(None);
    };
    let mut exact = Vec::with_capacity(output_shape.element_count()?);
    for index in 0..output_shape.element_count()? {
        exact.push(if predicate.data[index] == 0 {
            no_values[index].clone()
        } else {
            yes_values[index].clone()
        });
    }
    Ok(Some(exact))
}

pub(in crate::optimize) fn literal_bytes(shape: &Shape) -> Result<Option<usize>, String> {
    if shape.dims.contains(&0) {
        return Ok(None);
    }
    let elements = shape.dims.iter().try_fold(1usize, |product, &dim| {
        let dim = usize::try_from(dim).map_err(|_| format!("negative XIR dimension {dim}"))?;
        product
            .checked_mul(dim)
            .ok_or_else(|| format!("XIR shape {:?} overflows usize", shape.dims))
    })?;
    Ok(elements.checked_mul(shape.dtype.size_bytes()))
}

fn transpose_literal(
    input: &Literal,
    output_shape: &Shape,
    permutation: &[u32],
) -> Result<Vec<u8>, String> {
    remap_literal(input, output_shape, |output| {
        let mut source = vec![0usize; output.len()];
        for (axis, &coordinate) in output.iter().enumerate() {
            source[permutation[axis] as usize] = coordinate;
        }
        Ok(source)
    })
}

fn broadcast_literal(
    input: &Literal,
    output_shape: &Shape,
    dimensions: &[u32],
) -> Result<Vec<u8>, String> {
    remap_literal(input, output_shape, |output| {
        Ok(dimensions
            .iter()
            .enumerate()
            .map(|(axis, &output_axis)| {
                if input.shape.dims[axis] == 1 {
                    0
                } else {
                    output[output_axis as usize]
                }
            })
            .collect())
    })
}

fn slice_literal(input: &Literal, output_shape: &Shape, slice: &Slice) -> Result<Vec<u8>, String> {
    remap_literal(input, output_shape, |output| {
        output
            .iter()
            .enumerate()
            .map(|(axis, &coordinate)| {
                usize::try_from(
                    slice.start_indices[axis]
                        + i64::try_from(coordinate).map_err(|error| error.to_string())?
                            * slice.strides[axis],
                )
                .map_err(|error| error.to_string())
            })
            .collect()
    })
}

fn concatenate_literals(
    inputs: &[Literal],
    output_shape: &Shape,
    dimension: usize,
) -> Result<Vec<u8>, String> {
    let element_bytes = output_shape.dtype.size_bytes();
    let output_elements = literal_bytes(output_shape)?
        .ok_or_else(|| "cannot fold zero-sized concatenate".to_string())?
        / element_bytes;
    let mut data = Vec::with_capacity(output_elements * element_bytes);
    for flat in 0..output_elements {
        let mut coordinate = unravel(flat, &output_shape.dims)?;
        let mut offset = coordinate[dimension];
        let mut selected = None;
        for input in inputs {
            let extent =
                usize::try_from(input.shape.dims[dimension]).map_err(|error| error.to_string())?;
            if offset < extent {
                coordinate[dimension] = offset;
                selected = Some((input, ravel(&coordinate, &input.shape.dims)?));
                break;
            }
            offset -= extent;
        }
        let (input, source) = selected.ok_or_else(|| {
            format!(
                "concatenate coordinate {} exceeds input extents",
                coordinate[dimension]
            )
        })?;
        push_element(&mut data, input, source)?;
    }
    Ok(data)
}

fn pad_literal(
    input: &Literal,
    padding: &Literal,
    output_shape: &Shape,
    pad: &Pad,
) -> Result<Vec<u8>, String> {
    let element_bytes = output_shape.dtype.size_bytes();
    if padding.data.len() != element_bytes {
        return Err("XIR pad literal is not scalar".into());
    }
    let output_elements = literal_bytes(output_shape)?
        .ok_or_else(|| "cannot fold zero-sized pad".to_string())?
        / element_bytes;
    let mut data = Vec::with_capacity(output_elements * element_bytes);
    for flat in 0..output_elements {
        let output = unravel(flat, &output_shape.dims)?;
        let mut source = Vec::with_capacity(output.len());
        let mut is_padding = false;
        for (axis, &coordinate) in output.iter().enumerate() {
            let coordinate = i64::try_from(coordinate).map_err(|error| error.to_string())?;
            let expanded = coordinate - pad.edge_padding_low[axis];
            let step = pad.interior_padding[axis] + 1;
            if expanded < 0 || expanded % step != 0 {
                is_padding = true;
                break;
            }
            let input_coordinate = expanded / step;
            if input_coordinate >= input.shape.dims[axis] {
                is_padding = true;
                break;
            }
            source.push(usize::try_from(input_coordinate).map_err(|error| error.to_string())?);
        }
        if is_padding {
            data.extend_from_slice(&padding.data);
        } else {
            push_element(&mut data, input, ravel(&source, &input.shape.dims)?)?;
        }
    }
    Ok(data)
}

fn fold_select(
    values: &Rebuilder,
    inputs: &[ValueId],
    output_shape: &Shape,
) -> Result<Option<Vec<u8>>, String> {
    let Some(predicate) = values.literal(inputs[0]) else {
        return Ok(None);
    };
    let predicate_scalar = predicate.shape.is_scalar();
    if predicate_scalar {
        let selected = if predicate.data[0] == 0 {
            inputs[2]
        } else {
            inputs[1]
        };
        return Ok(values.literal(selected).map(|literal| literal.data));
    }
    let Some(yes) = values.literal(inputs[1]) else {
        return Ok(None);
    };
    let Some(no) = values.literal(inputs[2]) else {
        return Ok(None);
    };
    let element_bytes = output_shape.dtype.size_bytes();
    let elements = literal_bytes(output_shape)?
        .ok_or_else(|| "cannot fold zero-sized select".to_string())?
        / element_bytes;
    let mut data = Vec::with_capacity(elements * element_bytes);
    for index in 0..elements {
        push_element(
            &mut data,
            if predicate.data[index] == 0 {
                &no
            } else {
                &yes
            },
            index,
        )?;
    }
    Ok(Some(data))
}

fn fold_integer_elementwise(
    values: &Rebuilder,
    inputs: &[ValueId],
    output_shape: &Shape,
    op: ScalarOp,
    compare_type: Option<ComparisonType>,
) -> Result<Option<Vec<u8>>, String> {
    let Some(literals) = inputs
        .iter()
        .map(|&input| values.literal(input))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some(&input_dtype) = literals.first().map(|literal| &literal.shape.dtype) else {
        return Ok(None);
    };
    if !(input_dtype.is_integer() || input_dtype.is_bool()) {
        return Ok(None);
    }
    let input_bytes = input_dtype.size_bytes();
    let elements = literals[0].data.len() / input_bytes;
    if literals.iter().any(|literal| {
        literal.shape.dtype != input_dtype || literal.data.len() / input_bytes != elements
    }) {
        return Ok(None);
    }
    let mut data = Vec::with_capacity(
        elements
            .checked_mul(output_shape.dtype.size_bytes())
            .ok_or_else(|| "folded elementwise literal size overflow".to_string())?,
    );
    for element in 0..elements {
        let operands = literals
            .iter()
            .map(|literal| decode_integer(literal, element))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(result) = fold_integer_scalar(op, &operands, input_dtype, compare_type) else {
            return Ok(None);
        };
        encode_integer(&mut data, result, output_shape.dtype)?;
    }
    Ok(Some(data))
}

fn fold_integer_scalar(
    op: ScalarOp,
    values: &[u64],
    dtype: crate::dtype::DType,
    compare_type: Option<ComparisonType>,
) -> Option<u64> {
    let width = (dtype.size_bytes() * 8) as u32;
    let mask = integer_mask(width);
    let bits = |index: usize| values[index] & mask;
    let signed = |index: usize| sign_extend(bits(index), width);
    let result = if dtype.is_bool() {
        let a = bits(0) != 0;
        match op {
            ScalarOp::Not => u64::from(!a),
            ScalarOp::And => u64::from(a && bits(1) != 0),
            ScalarOp::Or => u64::from(a || bits(1) != 0),
            ScalarOp::Xor => u64::from(a ^ (bits(1) != 0)),
            ScalarOp::Compare(direction) => {
                return compare_integer(
                    direction,
                    u64::from(a),
                    u64::from(bits(1) != 0),
                    8,
                    false,
                    compare_type,
                )
                .map(u64::from);
            }
            _ => return None,
        }
    } else {
        match op {
            ScalarOp::Negate => 0u64.wrapping_sub(bits(0)) & mask,
            ScalarOp::Abs if dtype.is_signed_integer() => {
                if signed(0) < 0 {
                    0u64.wrapping_sub(bits(0)) & mask
                } else {
                    bits(0)
                }
            }
            ScalarOp::Abs => bits(0),
            ScalarOp::Add => bits(0).wrapping_add(bits(1)) & mask,
            ScalarOp::Subtract => bits(0).wrapping_sub(bits(1)) & mask,
            ScalarOp::Multiply => bits(0).wrapping_mul(bits(1)) & mask,
            ScalarOp::Divide if dtype.is_signed_integer() => {
                let (a, b) = (signed(0), signed(1));
                if b == 0 || (a == i64::MIN && b == -1) {
                    return None;
                }
                (a / b) as u64 & mask
            }
            ScalarOp::Divide => {
                if bits(1) == 0 {
                    return None;
                }
                bits(0) / bits(1)
            }
            ScalarOp::Remainder if dtype.is_signed_integer() => {
                let (a, b) = (signed(0), signed(1));
                if b == 0 || (a == i64::MIN && b == -1) {
                    return None;
                }
                (a % b) as u64 & mask
            }
            ScalarOp::Remainder => {
                if bits(1) == 0 {
                    return None;
                }
                bits(0) % bits(1)
            }
            ScalarOp::Maximum if dtype.is_signed_integer() => {
                if signed(0) >= signed(1) {
                    bits(0)
                } else {
                    bits(1)
                }
            }
            ScalarOp::Maximum => bits(0).max(bits(1)),
            ScalarOp::Minimum if dtype.is_signed_integer() => {
                if signed(0) <= signed(1) {
                    bits(0)
                } else {
                    bits(1)
                }
            }
            ScalarOp::Minimum => bits(0).min(bits(1)),
            ScalarOp::Not => !bits(0) & mask,
            ScalarOp::And => bits(0) & bits(1),
            ScalarOp::Or => bits(0) | bits(1),
            ScalarOp::Xor => bits(0) ^ bits(1),
            ScalarOp::ShiftLeft => {
                let shift = u32::try_from(bits(1)).ok()?;
                if shift >= width {
                    return None;
                }
                bits(0).wrapping_shl(shift) & mask
            }
            ScalarOp::ShiftRightArithmetic => {
                let shift = u32::try_from(bits(1)).ok()?;
                if shift >= width {
                    return None;
                }
                (signed(0) >> shift) as u64 & mask
            }
            ScalarOp::ShiftRightLogical => {
                let shift = u32::try_from(bits(1)).ok()?;
                if shift >= width {
                    return None;
                }
                bits(0) >> shift
            }
            ScalarOp::Power => {
                if dtype.is_signed_integer() && signed(1) < 0 {
                    return None;
                }
                let exponent = u32::try_from(bits(1)).ok()?;
                bits(0).wrapping_pow(exponent) & mask
            }
            ScalarOp::Compare(direction) => {
                return compare_integer(
                    direction,
                    bits(0),
                    bits(1),
                    width,
                    dtype.is_signed_integer(),
                    compare_type,
                )
                .map(u64::from);
            }
            _ => return None,
        }
    };
    Some(result)
}

fn compare_integer(
    direction: ComparisonDirection,
    a: u64,
    b: u64,
    width: u32,
    signed_default: bool,
    compare_type: Option<ComparisonType>,
) -> Option<bool> {
    let signed = match compare_type {
        None => signed_default,
        Some(ComparisonType::Signed) => true,
        Some(ComparisonType::Unsigned) => false,
        Some(ComparisonType::Float | ComparisonType::TotalOrder) => return None,
    };
    Some(if signed {
        compare(direction, sign_extend(a, width), sign_extend(b, width))
    } else {
        compare(direction, a, b)
    })
}

fn compare<T: PartialEq + PartialOrd>(direction: ComparisonDirection, a: T, b: T) -> bool {
    match direction {
        ComparisonDirection::Eq => a == b,
        ComparisonDirection::Ne => a != b,
        ComparisonDirection::Lt => a < b,
        ComparisonDirection::Le => a <= b,
        ComparisonDirection::Gt => a > b,
        ComparisonDirection::Ge => a >= b,
    }
}

fn decode_integer(literal: &Literal, element: usize) -> Result<u64, String> {
    let bytes = literal.shape.dtype.size_bytes();
    let start = element
        .checked_mul(bytes)
        .ok_or_else(|| "integer literal offset overflow".to_string())?;
    let mut value = [0u8; 8];
    value[..bytes].copy_from_slice(
        literal
            .data
            .get(start..start + bytes)
            .ok_or_else(|| format!("integer literal element {element} is out of bounds"))?,
    );
    Ok(u64::from_le_bytes(value))
}

fn encode_integer(
    data: &mut Vec<u8>,
    value: u64,
    dtype: crate::dtype::DType,
) -> Result<(), String> {
    if !(dtype.is_integer() || dtype.is_bool()) {
        return Err(format!("cannot encode folded integer as {dtype:?}"));
    }
    let bytes = value.to_le_bytes();
    data.extend_from_slice(&bytes[..dtype.size_bytes()]);
    Ok(())
}

fn integer_mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn sign_extend(value: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((value << shift) as i64) >> shift
}

fn remap_literal(
    input: &Literal,
    output_shape: &Shape,
    source_coordinate: impl Fn(&[usize]) -> Result<Vec<usize>, String>,
) -> Result<Vec<u8>, String> {
    let element_bytes = output_shape.dtype.size_bytes();
    let output_elements = literal_bytes(output_shape)?
        .ok_or_else(|| "cannot fold zero-sized shape operation".to_string())?
        / element_bytes;
    let mut data = Vec::with_capacity(output_elements * element_bytes);
    for flat in 0..output_elements {
        let output = unravel(flat, &output_shape.dims)?;
        let source = ravel(&source_coordinate(&output)?, &input.shape.dims)?;
        push_element(&mut data, input, source)?;
    }
    Ok(data)
}

fn push_element(data: &mut Vec<u8>, literal: &Literal, flat: usize) -> Result<(), String> {
    let element_bytes = literal.shape.dtype.size_bytes();
    let start = flat
        .checked_mul(element_bytes)
        .ok_or_else(|| "literal element offset overflow".to_string())?;
    let end = start + element_bytes;
    let bytes = literal
        .data
        .get(start..end)
        .ok_or_else(|| format!("literal element {flat} is out of bounds"))?;
    data.extend_from_slice(bytes);
    Ok(())
}

fn unravel(mut flat: usize, dims: &[i64]) -> Result<Vec<usize>, String> {
    let mut coordinate = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        let dim = usize::try_from(dims[axis]).map_err(|error| error.to_string())?;
        coordinate[axis] = flat % dim;
        flat /= dim;
    }
    Ok(coordinate)
}

fn ravel(coordinate: &[usize], dims: &[i64]) -> Result<usize, String> {
    coordinate
        .iter()
        .zip(dims)
        .try_fold(0usize, |flat, (&coordinate, &dim)| {
            let dim = usize::try_from(dim).map_err(|error| error.to_string())?;
            if coordinate >= dim {
                return Err(format!("coordinate {coordinate} exceeds dimension {dim}"));
            }
            flat.checked_mul(dim)
                .and_then(|flat| flat.checked_add(coordinate))
                .ok_or_else(|| "literal coordinate overflow".to_string())
        })
}
