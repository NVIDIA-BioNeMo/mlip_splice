// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use smallvec::smallvec;

use crate::dim_numbers::GatherDimensionNumbers;
use crate::op::def_ops;
use crate::shape::{Dims, Shape, ShapeError};

use super::infer_shapes::{
    axes_str, dims_str, expect_arity, expect_axes, expect_disjoint_axes, expect_scalar, same_dtype,
};

fn gather_infer(
    operand: &Shape,
    starts: &Shape,
    d: &GatherDimensionNumbers,
    slice_sizes: &[i64],
) -> Result<Shape, ShapeError> {
    let op = "gather";
    let ro = operand.rank();
    let rs = starts.rank();
    if !starts.dtype.is_integer() {
        return Err(ShapeError::new(
            op,
            "start_indices must have an integer dtype",
        ));
    }
    if slice_sizes.len() != ro {
        return Err(ShapeError::new(
            op,
            format!("slice_sizes len {} != operand rank {ro}", slice_sizes.len()),
        ));
    }
    if slice_sizes
        .iter()
        .zip(&operand.dims)
        .any(|(&size, &extent)| size < 0 || size > extent)
    {
        return Err(ShapeError::new(
            op,
            "slice_sizes must fit within the operand",
        ));
    }
    let ivd = d.index_vector_dim as usize;
    if ivd > rs {
        return Err(ShapeError::new(op, "index_vector_dim out of range"));
    }
    let index_count = if ivd < rs {
        usize::try_from(starts.dims[ivd])
            .map_err(|_| ShapeError::new(op, "index vector size is negative"))?
    } else {
        1
    };
    if d.start_index_map.len() != index_count {
        return Err(ShapeError::new(
            op,
            "start_index_map length must match the index vector size",
        ));
    }
    expect_axes(
        op,
        "collapsed_slice_dims",
        &d.collapsed_slice_dims,
        ro,
        true,
    )?;
    expect_axes(
        op,
        "operand_batching_dims",
        &d.operand_batching_dims,
        ro,
        true,
    )?;
    expect_disjoint_axes(
        op,
        "collapsed_slice_dims",
        &d.collapsed_slice_dims,
        "operand_batching_dims",
        &d.operand_batching_dims,
    )?;
    if ro != d.offset_dims.len() + d.collapsed_slice_dims.len() + d.operand_batching_dims.len() {
        return Err(ShapeError::new(
            op,
            "operand rank must equal offset, collapsed, and batching dimension counts",
        ));
    }
    if d.collapsed_slice_dims
        .iter()
        .chain(&d.operand_batching_dims)
        .any(|&axis| slice_sizes[axis as usize] > 1)
    {
        return Err(ShapeError::new(
            op,
            "collapsed and batching slice sizes must be at most one",
        ));
    }
    expect_axes(
        op,
        "start_indices_batching_dims",
        &d.start_indices_batching_dims,
        rs,
        false,
    )?;
    if d.start_indices_batching_dims.contains(&d.index_vector_dim) {
        return Err(ShapeError::new(
            op,
            "index_vector_dim must not be a start_indices batching dimension",
        ));
    }
    if d.operand_batching_dims.len() != d.start_indices_batching_dims.len() {
        return Err(ShapeError::new(op, "batching dimension count mismatch"));
    }
    for (&operand_axis, &starts_axis) in d
        .operand_batching_dims
        .iter()
        .zip(&d.start_indices_batching_dims)
    {
        if operand.dims[operand_axis as usize] != starts.dims[starts_axis as usize] {
            return Err(ShapeError::new(op, "batching dimension sizes differ"));
        }
    }
    expect_axes(op, "start_index_map", &d.start_index_map, ro, false)?;
    expect_disjoint_axes(
        op,
        "start_index_map",
        &d.start_index_map,
        "operand_batching_dims",
        &d.operand_batching_dims,
    )?;
    let batch_sizes: Vec<i64> = (0..rs)
        .filter(|&x| x != ivd)
        .map(|x| starts.dims[x])
        .collect();
    let offset_sizes: Vec<i64> = (0..ro)
        .filter(|&x| {
            !d.collapsed_slice_dims.contains(&(x as u32))
                && !d.operand_batching_dims.contains(&(x as u32))
        })
        .map(|x| slice_sizes[x])
        .collect();
    if d.offset_dims.len() != offset_sizes.len() {
        return Err(ShapeError::new(
            op,
            format!(
                "offset_dims len {} != non-collapsed operand dims {}",
                d.offset_dims.len(),
                offset_sizes.len()
            ),
        ));
    }
    let out_rank = batch_sizes.len() + d.offset_dims.len();
    expect_axes(op, "offset_dims", &d.offset_dims, out_rank, true)?;
    let mut out = vec![0i64; out_rank];
    let mut is_offset = vec![false; out_rank];
    for (k, &pos) in d.offset_dims.iter().enumerate() {
        let p = pos as usize;
        if p >= out_rank {
            return Err(ShapeError::new(op, "offset dim position out of range"));
        }
        out[p] = offset_sizes[k];
        is_offset[p] = true;
    }
    let mut bi = 0;
    for p in 0..out_rank {
        if !is_offset[p] {
            out[p] = batch_sizes[bi];
            bi += 1;
        }
    }
    Ok(operand.with_dims(out))
}

def_ops! {
    /// Extract a `slice_sizes` window; operand 0 is the source, operands 1.. are
    /// scalar start indices (one per operand dim).
    DynamicSlice { slice_sizes: Dims }
    mnemonic = "dynamic_slice"
    attrs = |this| { vec![("slice_sizes", dims_str(&this.slice_sizes))] }
    infer = |this, inputs| {
        if inputs.is_empty() {
            return Err(ShapeError::new("dynamic_slice", "needs an operand"));
        }
        let x = &inputs[0];
        let r = x.rank();
        if inputs.len() != r + 1 {
            return Err(ShapeError::new(
                "dynamic_slice",
                format!("expected {} start indices, got {}", r, inputs.len() - 1),
            ));
        }
        if this.slice_sizes.len() != r {
            return Err(ShapeError::new("dynamic_slice", "slice_sizes rank mismatch"));
        }
        for (i, idx) in inputs[1..].iter().enumerate() {
            expect_scalar("dynamic_slice", idx, &format!("start index {i}"))?;
        }
        for i in 0..r {
            if this.slice_sizes[i] < 0 || this.slice_sizes[i] > x.dims[i] {
                return Err(ShapeError::new("dynamic_slice", format!("slice size out of range on dim {i}")));
            }
        }
        Ok(smallvec![x.with_dims(this.slice_sizes.iter().copied())])
    }

    /// Write `update` into `operand` at scalar start indices; result is the
    /// operand shape. Operands: operand, update, then one scalar index per dim.
    DynamicUpdateSlice { }
    mnemonic = "dynamic_update_slice"
    attrs = |_this| { vec![] }
    infer = |_this, inputs| {
        if inputs.len() < 2 {
            return Err(ShapeError::new("dynamic_update_slice", "needs operand and update"));
        }
        let x = &inputs[0];
        let upd = &inputs[1];
        let r = x.rank();
        if inputs.len() != r + 2 {
            return Err(ShapeError::new(
                "dynamic_update_slice",
                format!("expected {} start indices, got {}", r, inputs.len() - 2),
            ));
        }
        same_dtype("dynamic_update_slice", x, upd, "operand and update")?;
        if upd.rank() != r {
            return Err(ShapeError::new("dynamic_update_slice", "update rank mismatch"));
        }
        for i in 0..r {
            if upd.dims[i] > x.dims[i] {
                return Err(ShapeError::new("dynamic_update_slice", format!("update larger than operand on dim {i}")));
            }
        }
        for (i, idx) in inputs[2..].iter().enumerate() {
            expect_scalar("dynamic_update_slice", idx, &format!("start index {i}"))?;
        }
        Ok(smallvec![x.clone()])
    }

    /// Indexed gather. Operands: operand, start_indices.
    Gather { dnums: GatherDimensionNumbers, slice_sizes: Dims, indices_are_sorted: bool }
    mnemonic = "gather"
    attrs = |this| {
        vec![
            ("offset_dims", axes_str(&this.dnums.offset_dims)),
            ("collapsed_slice_dims", axes_str(&this.dnums.collapsed_slice_dims)),
            ("operand_batching_dims", axes_str(&this.dnums.operand_batching_dims)),
            ("start_indices_batching_dims", axes_str(&this.dnums.start_indices_batching_dims)),
            ("start_index_map", axes_str(&this.dnums.start_index_map)),
            ("index_vector_dim", this.dnums.index_vector_dim.to_string()),
            ("slice_sizes", dims_str(&this.slice_sizes)),
            ("indices_are_sorted", this.indices_are_sorted.to_string()),
        ]
    }
    infer = |this, inputs| {
        expect_arity("gather", inputs, 2)?;
        Ok(smallvec![gather_infer(&inputs[0], &inputs[1], &this.dnums, &this.slice_sizes)?])
    }
}
