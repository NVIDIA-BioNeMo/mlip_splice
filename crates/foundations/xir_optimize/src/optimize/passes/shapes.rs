// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::SmallVec;

use crate::analysis::op_key;
use crate::ir::Module;
use crate::ops::{BroadcastInDim, Concatenate, Convert, Pad, Reshape, Slice, Transpose};
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;

pub(in crate::optimize) fn canonicalize_shapes(module: &mut Module) -> Result<bool, String> {
    let mut changed = false;
    let funcs = module
        .funcs
        .iter()
        .map(|old| {
            let mut new = Rebuilder::new(old);
            for node in &old.nodes {
                let mut inputs = new.mapped_inputs(node)?;
                let mut op = node.op.clone();
                let output_shape = node.results.first().map(|&result| old.value_shape(result));
                let alias = if node.results.len() != 1 {
                    None
                } else if node.op.as_any().downcast_ref::<Convert>().is_some()
                    && new.func.value_shape(inputs[0]) == output_shape.unwrap()
                {
                    Some(inputs[0])
                } else if let Some(reshape) = node.op.as_any().downcast_ref::<Reshape>() {
                    if new.func.value_shape(inputs[0]) == output_shape.unwrap() {
                        Some(inputs[0])
                    } else if let Some((_, source)) = new.producer::<Reshape>(inputs[0]) {
                        inputs[0] = source;
                        if new.func.value_shape(source) == output_shape.unwrap() {
                            Some(source)
                        } else {
                            op = std::sync::Arc::new(reshape.clone());
                            None
                        }
                    } else {
                        None
                    }
                } else if let Some(transpose) = node.op.as_any().downcast_ref::<Transpose>() {
                    if is_identity_permutation(&transpose.permutation) {
                        Some(inputs[0])
                    } else if let Some((inner, source)) = new.producer::<Transpose>(inputs[0]) {
                        let permutation = transpose
                            .permutation
                            .iter()
                            .map(|&axis| inner.permutation[axis as usize])
                            .collect::<SmallVec<_>>();
                        inputs[0] = source;
                        if is_identity_permutation(&permutation) {
                            Some(source)
                        } else {
                            op = std::sync::Arc::new(Transpose { permutation });
                            None
                        }
                    } else {
                        None
                    }
                } else if let Some(broadcast) = node.op.as_any().downcast_ref::<BroadcastInDim>() {
                    if is_identity_broadcast(
                        new.func.value_shape(inputs[0]),
                        output_shape.unwrap(),
                        &broadcast.broadcast_dimensions,
                    ) {
                        Some(inputs[0])
                    } else if let Some((inner, source)) = new.producer::<BroadcastInDim>(inputs[0])
                    {
                        let broadcast_dimensions = inner
                            .broadcast_dimensions
                            .iter()
                            .map(|&axis| broadcast.broadcast_dimensions[axis as usize])
                            .collect::<SmallVec<_>>();
                        inputs[0] = source;
                        if is_identity_broadcast(
                            new.func.value_shape(source),
                            output_shape.unwrap(),
                            &broadcast_dimensions,
                        ) {
                            Some(source)
                        } else {
                            op = std::sync::Arc::new(BroadcastInDim {
                                out_dims: broadcast.out_dims.clone(),
                                broadcast_dimensions,
                            });
                            None
                        }
                    } else {
                        None
                    }
                } else if let Some(slice) = node.op.as_any().downcast_ref::<Slice>() {
                    let input_shape = new.func.value_shape(inputs[0]);
                    (slice.start_indices.iter().all(|&index| index == 0)
                        && slice.limit_indices == input_shape.dims
                        && slice.strides.iter().all(|&stride| stride == 1))
                    .then_some(inputs[0])
                } else if node.op.as_any().downcast_ref::<Concatenate>().is_some()
                    && inputs.len() == 1
                {
                    Some(inputs[0])
                } else if let Some(pad) = node.op.as_any().downcast_ref::<Pad>() {
                    (pad.edge_padding_low.iter().all(|&value| value == 0)
                        && pad.edge_padding_high.iter().all(|&value| value == 0)
                        && pad.interior_padding.iter().all(|&value| value == 0))
                    .then_some(inputs[0])
                } else {
                    None
                };

                if let Some(value) = alias {
                    new.alias(&node.results, &[value])?;
                    changed = true;
                } else {
                    changed |=
                        op_key(op.as_ref()) != op_key(node.op.as_ref()) || inputs != node.inputs;
                    new.emit(old, &node.results, op, inputs)?;
                }
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}

fn is_identity_permutation(permutation: &[u32]) -> bool {
    permutation
        .iter()
        .enumerate()
        .all(|(index, &axis)| index == axis as usize)
}

fn is_identity_broadcast(input: &Shape, output: &Shape, dimensions: &[u32]) -> bool {
    input == output && is_identity_permutation(dimensions)
}
