// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use smallvec::SmallVec;

use crate::shape::{Shape, ShapeError};

pub(super) fn expect_axes(
    op: &'static str,
    name: &str,
    axes: &[u32],
    rank: usize,
    sorted: bool,
) -> Result<(), ShapeError> {
    if axes.iter().any(|&axis| axis as usize >= rank) {
        return Err(ShapeError::new(
            op,
            format!("{name} contains an axis outside rank {rank}"),
        ));
    }
    let unique = axes.iter().copied().collect::<HashSet<_>>().len() == axes.len();
    if !unique {
        return Err(ShapeError::new(op, format!("{name} must be unique")));
    }
    if sorted && !axes.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(ShapeError::new(op, format!("{name} must be sorted")));
    }
    Ok(())
}

pub(super) fn expect_disjoint_axes(
    op: &'static str,
    left_name: &str,
    left: &[u32],
    right_name: &str,
    right: &[u32],
) -> Result<(), ShapeError> {
    if left.iter().any(|axis| right.contains(axis)) {
        return Err(ShapeError::new(
            op,
            format!("{left_name} and {right_name} must be disjoint"),
        ));
    }
    Ok(())
}

pub(super) fn expect_arity(op: &'static str, inputs: &[Shape], n: usize) -> Result<(), ShapeError> {
    if inputs.len() != n {
        return Err(ShapeError::new(
            op,
            format!("expected {n} operands, got {}", inputs.len()),
        ));
    }
    if let Some(first) = inputs.first() {
        for input in &inputs[1..] {
            same_domain_axes(op, first, input, "operand")?;
        }
    }
    Ok(())
}

pub(super) fn expect_scalar(op: &'static str, s: &Shape, what: &str) -> Result<(), ShapeError> {
    if !s.is_scalar() {
        return Err(ShapeError::new(
            op,
            format!("{what} must be a scalar, got rank {}", s.rank()),
        ));
    }
    Ok(())
}

pub(super) fn same_domain_axes(
    op: &'static str,
    a: &Shape,
    b: &Shape,
    what: &str,
) -> Result<(), ShapeError> {
    if a.domain_axes != b.domain_axes {
        return Err(ShapeError::new(
            op,
            format!(
                "{what} domain axes differ: {:?} vs {:?}",
                a.domain_axes, b.domain_axes
            ),
        ));
    }
    Ok(())
}

pub(super) fn same_axes(
    op: &'static str,
    a: &Shape,
    b: &Shape,
    what: &str,
) -> Result<(), ShapeError> {
    same_domain_axes(op, a, b, what)?;
    if a.dims != b.dims {
        return Err(ShapeError::new(
            op,
            format!("{what} shapes differ: {:?} vs {:?}", a.dims, b.dims),
        ));
    }
    Ok(())
}

pub(super) fn same_dtype(
    op: &'static str,
    a: &Shape,
    b: &Shape,
    what: &str,
) -> Result<(), ShapeError> {
    if a.dtype != b.dtype {
        return Err(ShapeError::new(
            op,
            format!(
                "{what} dtypes differ: {} vs {}",
                a.dtype.name(),
                b.dtype.name()
            ),
        ));
    }
    Ok(())
}

pub(super) fn axes_str(a: &[u32]) -> String {
    format!("{a:?}")
}

pub(super) fn dims_str(d: &[i64]) -> String {
    format!("{d:?}")
}

pub(super) fn bools_str(b: &[bool]) -> String {
    format!("{b:?}")
}

/// Per-spatial-dim `[low, high]` window padding, StableHLO's single `padding`
/// (Nx2) form, shared by `convolution` and `reduce_window`.
pub type Padding = SmallVec<[[i64; 2]; 4]>;

pub(super) fn padding_str(p: &[[i64; 2]]) -> String {
    let body = p
        .iter()
        .map(|[lo, hi]| format!("[{lo}, {hi}]"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

// ----------------------------------------------------------------------------
