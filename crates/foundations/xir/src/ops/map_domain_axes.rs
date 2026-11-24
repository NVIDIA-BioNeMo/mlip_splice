// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;

use domain_index::FiniteMapId;
use smallvec::{SmallVec, smallvec};

use crate::op::{InferCtx, Op};
use crate::shape::{Shape, ShapeError};

use super::infer_shapes::expect_arity;

fn map_axis_shape(
    op: &'static str,
    input: &Shape,
    axis: u32,
    input_set: domain_index::FiniteSetId,
    output_set: domain_index::FiniteSetId,
) -> Result<Shape, ShapeError> {
    let axis = axis as usize;
    let Some(&actual) = input.domain_axes.get(axis) else {
        return Err(ShapeError::new(
            op,
            format!("domain axis {axis} out of range"),
        ));
    };
    if actual != input_set {
        return Err(ShapeError::new(
            op,
            format!(
                "domain axis {axis} names set {}, expected {}",
                actual.index(),
                input_set.index()
            ),
        ));
    }
    let mut output = input.clone();
    output.domain_axes[axis] = output_set;
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherAlongMap {
    pub map: FiniteMapId,
    pub axis: u32,
}

impl Op for GatherAlongMap {
    fn mnemonic(&self) -> &'static str {
        "gather_along_map"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        expect_arity(self.mnemonic(), ctx.inputs(), 1)?;
        let map = ctx
            .finite_map_graph()
            .map(self.map)
            .map_err(|error| ShapeError::new(self.mnemonic(), error))?;
        Ok(smallvec![map_axis_shape(
            self.mnemonic(),
            &ctx.inputs()[0],
            self.axis,
            map.target,
            map.source,
        )?])
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("map", self.map.index().to_string()),
            ("axis", self.axis.to_string()),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumAlongMap {
    pub map: FiniteMapId,
    pub axis: u32,
}

impl Op for SumAlongMap {
    fn mnemonic(&self) -> &'static str {
        "sum_along_map"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        expect_arity(self.mnemonic(), ctx.inputs(), 1)?;
        let map = ctx
            .finite_map_graph()
            .map(self.map)
            .map_err(|error| ShapeError::new(self.mnemonic(), error))?;
        Ok(smallvec![map_axis_shape(
            self.mnemonic(),
            &ctx.inputs()[0],
            self.axis,
            map.source,
            map.target,
        )?])
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("map", self.map.index().to_string()),
            ("axis", self.axis.to_string()),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}
