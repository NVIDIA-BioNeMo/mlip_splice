// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static row-major element semantics, independent of scalar arithmetic.
//! Callers supply verified XIR shapes and dimension numbers, with nonzero axes.

use crate::ops::{DotGeneral, Gather, Pad, Scatter, Slice};
use crate::shape::Dims;

/// Broadcast into `out_dims`; each input axis maps to its named output axis.
pub fn broadcast_elements<T: Clone>(
    input: &[T],
    in_dims: &[i64],
    out_dims: &[i64],
    broadcast_dimensions: &[u32],
) -> Result<Vec<T>, String> {
    let mut output = Vec::with_capacity(num_elements(out_dims)?);
    for flat in 0..num_elements(out_dims)? {
        let out_coord = unravel(flat, out_dims)?;
        let in_coord = broadcast_dimensions
            .iter()
            .enumerate()
            .map(|(axis, &out_axis)| {
                if in_dims[axis] == 1 {
                    0
                } else {
                    out_coord[out_axis as usize]
                }
            })
            .collect::<Vec<_>>();
        output.push(input[ravel(&in_coord, in_dims)?].clone());
    }
    Ok(output)
}

/// Reorder axes so output axis `i` is input axis `permutation[i]`.
pub fn transpose_elements<T: Clone>(
    input: &[T],
    in_dims: &[i64],
    permutation: &[u32],
) -> Result<Vec<T>, String> {
    let out_dims = permutation
        .iter()
        .map(|&axis| in_dims[axis as usize])
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(num_elements(&out_dims)?);
    for flat in 0..num_elements(&out_dims)? {
        let out_coord = unravel(flat, &out_dims)?;
        let mut in_coord = vec![0; in_dims.len()];
        for (out_axis, &in_axis) in permutation.iter().enumerate() {
            in_coord[in_axis as usize] = out_coord[out_axis];
        }
        output.push(input[ravel(&in_coord, in_dims)?].clone());
    }
    Ok(output)
}

/// Select input coordinates `start + output_coordinate * stride`.
pub fn slice_elements<T: Clone>(
    input: &[T],
    in_dims: &[i64],
    op: &Slice,
) -> Result<Vec<T>, String> {
    let out_dims = op
        .start_indices
        .iter()
        .zip(&op.limit_indices)
        .zip(&op.strides)
        .map(|((&start, &limit), &stride)| (limit - start + stride - 1) / stride)
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(num_elements(&out_dims)?);
    for flat in 0..num_elements(&out_dims)? {
        let out_coord = unravel(flat, &out_dims)?;
        let in_coord = out_coord
            .iter()
            .enumerate()
            .map(|(axis, &coord)| {
                op.start_indices[axis] as usize + coord * op.strides[axis] as usize
            })
            .collect::<Vec<_>>();
        output.push(input[ravel(&in_coord, in_dims)?].clone());
    }
    Ok(output)
}

/// Join equally ranked tensors along `axis`, preserving input order.
pub fn concatenate_elements<T: Clone>(
    inputs: &[(&[T], &Dims)],
    axis: usize,
) -> Result<Vec<T>, String> {
    let mut out_dims = inputs[0].1.to_vec();
    out_dims[axis] = inputs.iter().map(|(_, dims)| dims[axis]).sum();
    let mut output = Vec::with_capacity(num_elements(&out_dims)?);
    for flat in 0..num_elements(&out_dims)? {
        let mut coord = unravel(flat, &out_dims)?;
        let mut along = coord[axis];
        let mut found = None;
        for &(values, dims) in inputs {
            if along < dims[axis] as usize {
                coord[axis] = along;
                found = Some(values[ravel(&coord, dims)?].clone());
                break;
            }
            along -= dims[axis] as usize;
        }
        output.push(found.ok_or_else(|| "concatenate coordinate is out of range".to_string())?);
    }
    Ok(output)
}

/// Place input coordinates at `low + input_coordinate * (interior + 1)`.
/// Fill other output cells with `pad_value`; negative edges crop the input.
pub fn pad_elements<T: Clone>(
    input: &[T],
    in_dims: &[i64],
    pad_value: &T,
    op: &Pad,
) -> Result<Vec<T>, String> {
    let out_dims = in_dims
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            op.edge_padding_low[axis]
                + op.edge_padding_high[axis]
                + dim
                + (dim - 1).max(0) * op.interior_padding[axis]
        })
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(num_elements(&out_dims)?);
    for flat in 0..num_elements(&out_dims)? {
        let out_coord = unravel(flat, &out_dims)?;
        let mut in_coord = Vec::with_capacity(in_dims.len());
        let mut inside = true;
        for (axis, &coordinate) in out_coord.iter().enumerate() {
            let shifted = coordinate as i64 - op.edge_padding_low[axis];
            let step = op.interior_padding[axis] + 1;
            if shifted < 0 || shifted % step != 0 || shifted / step >= in_dims[axis] {
                inside = false;
                break;
            }
            in_coord.push((shifted / step) as usize);
        }
        output.push(if inside {
            input[ravel(&in_coord, in_dims)?].clone()
        } else {
            pad_value.clone()
        });
    }
    Ok(output)
}

/// Row-major lhs/rhs element pairs contributing to each `dot_general` result.
/// Result axes are batching, lhs free, then rhs free; contraction axes follow
/// the lhs contracting dimension numbers.
pub struct DotPlan {
    lhs_dims: Vec<i64>,
    rhs_dims: Vec<i64>,
    lhs_batch: Vec<usize>,
    rhs_batch: Vec<usize>,
    lhs_contract: Vec<usize>,
    rhs_contract: Vec<usize>,
    lhs_free: Vec<usize>,
    rhs_free: Vec<usize>,
    pub out_dims: Vec<i64>,
    pub contract_dims: Vec<i64>,
}

impl DotPlan {
    pub fn new(lhs_dims: &[i64], rhs_dims: &[i64], op: &DotGeneral) -> Result<Self, String> {
        let d = &op.dnums;
        let lhs_batch = axes(&d.lhs_batching_dimensions);
        let rhs_batch = axes(&d.rhs_batching_dimensions);
        let lhs_contract = axes(&d.lhs_contracting_dimensions);
        let rhs_contract = axes(&d.rhs_contracting_dimensions);
        let free = |rank: usize, batch: &[usize], contract: &[usize]| {
            (0..rank)
                .filter(|axis| !batch.contains(axis) && !contract.contains(axis))
                .collect::<Vec<_>>()
        };
        let lhs_free = free(lhs_dims.len(), &lhs_batch, &lhs_contract);
        let rhs_free = free(rhs_dims.len(), &rhs_batch, &rhs_contract);
        // The result lays its axes out batch, then lhs free, then rhs free.
        let out_dims = lhs_batch
            .iter()
            .map(|&axis| lhs_dims[axis])
            .chain(lhs_free.iter().map(|&axis| lhs_dims[axis]))
            .chain(rhs_free.iter().map(|&axis| rhs_dims[axis]))
            .collect();
        let contract_dims = lhs_contract.iter().map(|&axis| lhs_dims[axis]).collect();
        Ok(Self {
            lhs_dims: lhs_dims.to_vec(),
            rhs_dims: rhs_dims.to_vec(),
            lhs_batch,
            rhs_batch,
            lhs_contract,
            rhs_contract,
            lhs_free,
            rhs_free,
            out_dims,
            contract_dims,
        })
    }

    /// The `(lhs cell, rhs cell)` pairs whose products sum into result cell
    /// `flat`, in contraction order.
    pub fn contraction(&self, flat: usize) -> Result<Vec<(usize, usize)>, String> {
        let out_coord = unravel(flat, &self.out_dims)?;
        let mut lhs_coord = vec![0; self.lhs_dims.len()];
        let mut rhs_coord = vec![0; self.rhs_dims.len()];
        let mut cursor = 0;
        for (&lhs_axis, &rhs_axis) in self.lhs_batch.iter().zip(&self.rhs_batch) {
            lhs_coord[lhs_axis] = out_coord[cursor];
            rhs_coord[rhs_axis] = out_coord[cursor];
            cursor += 1;
        }
        for &axis in &self.lhs_free {
            lhs_coord[axis] = out_coord[cursor];
            cursor += 1;
        }
        for &axis in &self.rhs_free {
            rhs_coord[axis] = out_coord[cursor];
            cursor += 1;
        }
        (0..num_elements(&self.contract_dims)?)
            .map(|contract_flat| {
                let contract_coord = unravel(contract_flat, &self.contract_dims)?;
                for (k, (&lhs_axis, &rhs_axis)) in
                    self.lhs_contract.iter().zip(&self.rhs_contract).enumerate()
                {
                    lhs_coord[lhs_axis] = contract_coord[k];
                    rhs_coord[rhs_axis] = contract_coord[k];
                }
                Ok((
                    ravel(&lhs_coord, &self.lhs_dims)?,
                    ravel(&rhs_coord, &self.rhs_dims)?,
                ))
            })
            .collect()
    }
}

/// XIR dimension numbers as usize axes.
pub fn axes(dimensions: &[u32]) -> Vec<usize> {
    dimensions.iter().map(|&axis| axis as usize).collect()
}

/// A reduction's output shape, and which output cell each input cell folds
/// into. The reduced axes drop out; the rest keep their order.
pub fn reduction_map(in_dims: &[i64], reduced: &[usize]) -> Result<(Vec<i64>, Vec<usize>), String> {
    let keep = |axis: usize| !reduced.contains(&axis);
    let out_dims = in_dims
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| keep(axis).then_some(dim))
        .collect::<Vec<_>>();
    let cells = (0..num_elements(in_dims)?)
        .map(|flat| {
            let out_coord = unravel(flat, in_dims)?
                .into_iter()
                .enumerate()
                .filter_map(|(axis, index)| keep(axis).then_some(index))
                .collect::<Vec<_>>();
            ravel(&out_coord, &out_dims)
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((out_dims, cells))
}

/// Map each output element to its operand element, clamping slice starts.
/// Operand batching dimensions are not supported.
pub fn gather_map(
    operand_dims: &[i64],
    start_dims: &[i64],
    starts: &[i64],
    output_dims: &[i64],
    op: &Gather,
) -> Result<Vec<usize>, String> {
    let d = &op.dnums;
    let index_vector_dim = d.index_vector_dim as usize;
    let explicit_index_vector = index_vector_dim < start_dims.len();
    let start_batch_axes = (0..start_dims.len())
        .filter(|&axis| axis != index_vector_dim)
        .collect::<Vec<_>>();
    let output_batch_axes = (0..output_dims.len())
        .filter(|axis| !d.offset_dims.contains(&(*axis as u32)))
        .collect::<Vec<_>>();
    if start_batch_axes.len() != output_batch_axes.len() {
        return Err("gather batch rank does not match output batch rank".into());
    }
    let window_operand_axes = (0..operand_dims.len())
        .filter(|axis| !d.collapsed_slice_dims.contains(&(*axis as u32)))
        .collect::<Vec<_>>();
    if window_operand_axes.len() != d.offset_dims.len() {
        return Err("gather window rank does not match offset_dims".into());
    }
    if starts.len() != num_elements(start_dims)? {
        return Err("gather index value count does not match its shape".into());
    }

    (0..num_elements(output_dims)?)
        .map(|flat| {
            let output_coord = unravel(flat, output_dims)?;
            let mut start_coord = vec![0; start_dims.len()];
            for (&start_axis, &output_axis) in start_batch_axes.iter().zip(&output_batch_axes) {
                start_coord[start_axis] = output_coord[output_axis];
            }
            let mut read_start = |component: usize| -> Result<i64, String> {
                if explicit_index_vector {
                    start_coord[index_vector_dim] = component;
                } else if component != 0 {
                    return Err("implicit gather index vector must have size one".into());
                }
                starts
                    .get(ravel(&start_coord, start_dims)?)
                    .copied()
                    .ok_or_else(|| "gather index component is out of range".into())
            };

            let mut operand_coord = vec![0; operand_dims.len()];
            for (component, &operand_axis) in d.start_index_map.iter().enumerate() {
                let operand_axis = operand_axis as usize;
                let max_start = operand_dims[operand_axis] - op.slice_sizes[operand_axis];
                operand_coord[operand_axis] = read_start(component)?.clamp(0, max_start) as usize;
            }
            for (&operand_axis, &start_axis) in d
                .operand_batching_dims
                .iter()
                .zip(&d.start_indices_batching_dims)
            {
                let start_axis = start_axis as usize;
                let output_axis = output_batch_axes[start_batch_axes
                    .iter()
                    .position(|&axis| axis == start_axis)
                    .ok_or_else(|| "gather batching axis is the index-vector axis".to_string())?];
                operand_coord[operand_axis as usize] = output_coord[output_axis];
            }
            for (&operand_axis, &output_axis) in window_operand_axes.iter().zip(&d.offset_dims) {
                operand_coord[operand_axis] += output_coord[output_axis as usize];
            }
            ravel(&operand_coord, operand_dims)
        })
        .collect()
}

/// Map each update element to its operand element; out-of-bounds updates
/// have no destination. Repeated destinations retain update order.
pub fn scatter_map(
    operand_dims: &[i64],
    scatter_dims: &[i64],
    indices: &[i64],
    update_dims: &[i64],
    op: &Scatter,
) -> Result<Vec<Option<usize>>, String> {
    let d = &op.dnums;
    let index_vector_dim = d.index_vector_dim as usize;
    let explicit_index_vector = index_vector_dim < scatter_dims.len();
    let scatter_batch_axes = (0..scatter_dims.len())
        .filter(|&axis| axis != index_vector_dim)
        .collect::<Vec<_>>();
    let update_scatter_axes = (0..update_dims.len())
        .filter(|axis| !d.update_window_dims.contains(&(*axis as u32)))
        .collect::<Vec<_>>();
    if scatter_batch_axes.len() != update_scatter_axes.len() {
        return Err("scatter batch rank does not match update scatter rank".into());
    }
    let window_operand_axes = (0..operand_dims.len())
        .filter(|axis| {
            !d.inserted_window_dims.contains(&(*axis as u32))
                && !d.input_batching_dims.contains(&(*axis as u32))
        })
        .collect::<Vec<_>>();
    if window_operand_axes.len() != d.update_window_dims.len() {
        return Err("scatter window rank does not match update_window_dims".into());
    }
    if indices.len() != num_elements(scatter_dims)? {
        return Err("scatter index value count does not match its shape".into());
    }

    (0..num_elements(update_dims)?)
        .map(|flat| {
            let update_coord = unravel(flat, update_dims)?;
            let mut scatter_coord = vec![0; scatter_dims.len()];
            for (&scatter_axis, &update_axis) in scatter_batch_axes.iter().zip(&update_scatter_axes)
            {
                scatter_coord[scatter_axis] = update_coord[update_axis];
            }
            let mut read_index = |component: usize| -> Result<i64, String> {
                if explicit_index_vector {
                    scatter_coord[index_vector_dim] = component;
                } else if component != 0 {
                    return Err("implicit scatter index vector must have size one".into());
                }
                indices
                    .get(ravel(&scatter_coord, scatter_dims)?)
                    .copied()
                    .ok_or_else(|| "scatter index component is out of range".into())
            };

            let mut operand_coord = vec![0i64; operand_dims.len()];
            for (component, &operand_axis) in d.scatter_dims_to_operand_dims.iter().enumerate() {
                operand_coord[operand_axis as usize] = read_index(component)?;
            }
            for (&operand_axis, &scatter_axis) in d
                .input_batching_dims
                .iter()
                .zip(&d.scatter_indices_batching_dims)
            {
                let update_axis = update_scatter_axes[scatter_batch_axes
                    .iter()
                    .position(|&axis| axis == scatter_axis as usize)
                    .ok_or_else(|| "scatter batching axis is the index-vector axis".to_string())?];
                operand_coord[operand_axis as usize] = update_coord[update_axis] as i64;
            }
            for (&operand_axis, &update_axis) in
                window_operand_axes.iter().zip(&d.update_window_dims)
            {
                operand_coord[operand_axis] += update_coord[update_axis as usize] as i64;
            }
            if operand_coord
                .iter()
                .zip(operand_dims)
                .any(|(&coord, &dim)| coord < 0 || coord >= dim)
            {
                return Ok(None);
            }
            let coordinate = operand_coord
                .into_iter()
                .map(|coord| coord as usize)
                .collect::<Vec<_>>();
            ravel(&coordinate, operand_dims).map(Some)
        })
        .collect()
}

/// Count row-major elements. Rank zero has one element.
/// Zero-length axes are not supported by element expansion.
pub fn num_elements(dims: &[i64]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |product, &dim| {
        let dim = usize::try_from(dim).map_err(|_| format!("negative dimension {dim}"))?;
        if dim == 0 {
            return Err("zero-sized visible dimensions are not supported".into());
        }
        product
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {dims:?} overflows usize"))
    })
}

pub fn unravel(mut flat: usize, dims: &[i64]) -> Result<Vec<usize>, String> {
    if flat >= num_elements(dims)? {
        return Err(format!("flat index {flat} is outside shape {dims:?}"));
    }
    let mut coord = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        let dim = dims[axis] as usize;
        coord[axis] = flat % dim;
        flat /= dim;
    }
    Ok(coord)
}

pub fn ravel(coord: &[usize], dims: &[i64]) -> Result<usize, String> {
    if coord.len() != dims.len() {
        return Err(format!(
            "coordinate rank {} differs from shape rank {}",
            coord.len(),
            dims.len()
        ));
    }
    coord
        .iter()
        .zip(dims)
        .try_fold(0usize, |flat, (&coordinate, &dim)| {
            let dim = usize::try_from(dim).map_err(|_| format!("negative dimension {dim}"))?;
            if coordinate >= dim {
                return Err(format!(
                    "coordinate {coordinate} is outside dimension {dim}"
                ));
            }
            flat.checked_mul(dim)
                .and_then(|flat| flat.checked_add(coordinate))
                .ok_or_else(|| "coordinate overflow".to_string())
        })
}
