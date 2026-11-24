// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dimension-number descriptors, copied (semantics) from StableHLO/HLO. These
//! are what make `dot`/`gather`/`scatter` fully general rather than narrow
//! special cases.

use smallvec::SmallVec;

pub type Axes = SmallVec<[u32; 4]>;

fn axes(slice: &[u32]) -> Axes {
    SmallVec::from_slice(slice)
}

/// Contracting + batch dimensions on each side of a `dot`. Generalizes matmul,
/// batched matmul, and tensordot into one operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct DotDimensionNumbers {
    pub lhs_contracting_dimensions: Axes,
    pub rhs_contracting_dimensions: Axes,
    pub lhs_batching_dimensions: Axes,
    pub rhs_batching_dimensions: Axes,
}

impl DotDimensionNumbers {
    /// Plain 2-D matmul `[m,k] · [k,n] -> [m,n]`.
    pub fn matmul() -> DotDimensionNumbers {
        DotDimensionNumbers {
            lhs_contracting_dimensions: axes(&[1]),
            rhs_contracting_dimensions: axes(&[0]),
            lhs_batching_dimensions: Axes::new(),
            rhs_batching_dimensions: Axes::new(),
        }
    }

    pub fn new(
        lhs_contracting_dimensions: &[u32],
        rhs_contracting_dimensions: &[u32],
        lhs_batching_dimensions: &[u32],
        rhs_batching_dimensions: &[u32],
    ) -> DotDimensionNumbers {
        DotDimensionNumbers {
            lhs_contracting_dimensions: axes(lhs_contracting_dimensions),
            rhs_contracting_dimensions: axes(rhs_contracting_dimensions),
            lhs_batching_dimensions: axes(lhs_batching_dimensions),
            rhs_batching_dimensions: axes(rhs_batching_dimensions),
        }
    }
}

/// Maps the structured `start_indices` operand of a `gather` onto slices of the
/// operand. Field meanings follow StableHLO's `GatherDimensionNumbers`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct GatherDimensionNumbers {
    pub offset_dims: Axes,
    pub collapsed_slice_dims: Axes,
    pub operand_batching_dims: Axes,
    pub start_indices_batching_dims: Axes,
    pub start_index_map: Axes,
    pub index_vector_dim: u32,
}

/// Maps the structured `scatter_indices` operand of a `scatter` onto windows of
/// the operand. Field meanings follow StableHLO's `ScatterDimensionNumbers`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ScatterDimensionNumbers {
    pub update_window_dims: Axes,
    pub inserted_window_dims: Axes,
    pub input_batching_dims: Axes,
    pub scatter_indices_batching_dims: Axes,
    pub scatter_dims_to_operand_dims: Axes,
    pub index_vector_dim: u32,
}

/// Which axes of a `convolution`'s input, kernel, and output play which role.
/// Field meanings follow XLA's `ConvolutionDimensionNumbers`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ConvDimensionNumbers {
    pub input_batch_dimension: u32,
    pub input_feature_dimension: u32,
    pub input_spatial_dimensions: Axes,
    pub kernel_input_feature_dimension: u32,
    pub kernel_output_feature_dimension: u32,
    pub kernel_spatial_dimensions: Axes,
    pub output_batch_dimension: u32,
    pub output_feature_dimension: u32,
    pub output_spatial_dimensions: Axes,
}
