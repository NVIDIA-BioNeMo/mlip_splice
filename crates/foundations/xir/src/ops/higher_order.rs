// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;

use smallvec::{SmallVec, smallvec};

use crate::dim_numbers::ScatterDimensionNumbers;
use crate::dtype::DType;
use crate::ir::FuncId;
use crate::op::{InferCtx, Op};
use crate::shape::{Dims, Shape, ShapeError};

use super::infer_shapes::{
    Padding, axes_str, dims_str, expect_arity, expect_axes, expect_disjoint_axes, expect_scalar,
    padding_str, same_axes, same_dtype,
};

/// Reduce `operand` over `dimensions` using `reducer`. Operands: operand, init
/// scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduce {
    pub dimensions: SmallVec<[u32; 4]>,
    pub reducer: FuncId,
}

impl Op for Reduce {
    fn mnemonic(&self) -> &'static str {
        "reduce"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        // Variadic: the operands are K inputs followed by K scalar init values,
        // reduced jointly over `dims` to K results (e.g. argmax reduces a value
        // tensor and an index tensor together).
        let inputs = ctx.inputs();
        if inputs.is_empty() || !inputs.len().is_multiple_of(2) {
            return Err(ShapeError::new(
                "reduce",
                "expected an even number of operands (K inputs then K inits)",
            ));
        }
        let k = inputs.len() / 2;
        let (data, inits) = inputs.split_at(k);
        let rank = data[0].rank();
        let mut keep = vec![true; rank];
        for &dd in &self.dimensions {
            let dd = dd as usize;
            if dd >= rank {
                return Err(ShapeError::new("reduce", "reduced dim out of range"));
            }
            keep[dd] = false;
        }
        let mut results = SmallVec::new();
        for i in 0..k {
            if data[i].dims != data[0].dims {
                return Err(ShapeError::new(
                    "reduce",
                    "all reduced inputs must share a shape",
                ));
            }
            expect_scalar("reduce", &inits[i], "init value")?;
            same_dtype("reduce", &data[i], &inits[i], "input and init")?;
            let dims = (0..rank).filter(|&j| keep[j]).map(|j| data[i].dims[j]);
            results.push(data[i].with_dims(dims));
        }
        let Some(reducer) = ctx.func_sig(self.reducer) else {
            return Err(ShapeError::new("reduce", "reducer is out of range"));
        };
        let scalar_params = data
            .iter()
            .chain(inits)
            .map(|shape| Shape::scalar(shape.dtype))
            .collect::<Vec<_>>();
        let scalar_results = data
            .iter()
            .map(|shape| Shape::scalar(shape.dtype))
            .collect::<Vec<_>>();
        if reducer.params != scalar_params || reducer.results != scalar_results {
            return Err(ShapeError::new(
                "reduce",
                "reducer signature must be (K accumulators, K values) -> K accumulators",
            ));
        }
        Ok(results)
    }

    fn called_funcs(&self) -> &[FuncId] {
        std::slice::from_ref(&self.reducer)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("dimensions", axes_str(&self.dimensions)),
            ("reducer", format!("@{}", self.reducer.index())),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// Windowed reduction: slide a window over each input and reduce it with
/// `reducer`, jointly over K inputs (+ K scalar inits) → K results. The
/// per-dimension vectors all have the operand rank. Covers cumulative scans
/// and pooling.
#[derive(Clone, Debug, PartialEq)]
pub struct ReduceWindow {
    pub window_dimensions: Dims,
    pub window_strides: Dims,
    pub base_dilations: Dims,
    pub window_dilations: Dims,
    pub padding: Padding,
    pub reducer: FuncId,
}

impl Op for ReduceWindow {
    fn mnemonic(&self) -> &'static str {
        "reduce_window"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let op = "reduce_window";
        let inputs = ctx.inputs();
        if inputs.is_empty() || !inputs.len().is_multiple_of(2) {
            return Err(ShapeError::new(op, "expected K inputs then K inits"));
        }
        let k = inputs.len() / 2;
        let (data, inits) = inputs.split_at(k);
        let rank = data[0].rank();
        for v in [
            &self.window_dimensions,
            &self.window_strides,
            &self.base_dilations,
            &self.window_dilations,
        ] {
            if v.len() != rank {
                return Err(ShapeError::new(
                    op,
                    "window vectors must have the operand rank",
                ));
            }
        }
        if self.padding.len() != rank {
            return Err(ShapeError::new(op, "padding must have the operand rank"));
        }
        let mut out_dims = Vec::with_capacity(rank);
        for i in 0..rank {
            let input = data[0].dims[i];
            let dilated_input = if input == 0 {
                0
            } else {
                (input - 1) * self.base_dilations[i] + 1
            };
            let [pad_low, pad_high] = self.padding[i];
            let padded = dilated_input + pad_low + pad_high;
            let dilated_window = (self.window_dimensions[i] - 1) * self.window_dilations[i] + 1;
            let stride = self.window_strides[i];
            if stride < 1 {
                return Err(ShapeError::new(op, "window stride must be >= 1"));
            }
            out_dims.push(if padded < dilated_window {
                0
            } else {
                (padded - dilated_window) / stride + 1
            });
        }
        let mut results = SmallVec::new();
        for i in 0..k {
            if data[i].dims != data[0].dims {
                return Err(ShapeError::new(op, "all inputs must share a shape"));
            }
            expect_scalar(op, &inits[i], "init value")?;
            same_dtype(op, &data[i], &inits[i], "input and init")?;
            results.push(data[i].with_dims(out_dims.iter().copied()));
        }
        let Some(reducer) = ctx.func_sig(self.reducer) else {
            return Err(ShapeError::new(op, "reducer is out of range"));
        };
        let scalar_params = data
            .iter()
            .chain(inits)
            .map(|shape| {
                Shape::scalar(shape.dtype).with_domain_axes(shape.domain_axes.iter().copied())
            })
            .collect::<Vec<_>>();
        let scalar_results = data
            .iter()
            .map(|shape| {
                Shape::scalar(shape.dtype).with_domain_axes(shape.domain_axes.iter().copied())
            })
            .collect::<Vec<_>>();
        if reducer.params != scalar_params || reducer.results != scalar_results {
            return Err(ShapeError::new(
                op,
                "reducer signature must be (K accumulators, K values) -> K accumulators",
            ));
        }
        Ok(results)
    }

    fn called_funcs(&self) -> &[FuncId] {
        std::slice::from_ref(&self.reducer)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("window_dimensions", dims_str(&self.window_dimensions)),
            ("window_strides", dims_str(&self.window_strides)),
            ("base_dilations", dims_str(&self.base_dilations)),
            ("window_dilations", dims_str(&self.window_dilations)),
            ("padding", padding_str(&self.padding)),
            ("reducer", format!("@{}", self.reducer.index())),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// Scatter `updates` into `operand` at `scatter_indices`, combining with
/// `update`. Operands: operand, scatter_indices, updates. Result is the operand.
fn scatter_infer(
    ctx: &InferCtx<'_>,
    d: &ScatterDimensionNumbers,
    update: FuncId,
) -> Result<Shape, ShapeError> {
    let op = "scatter";
    let inputs = ctx.inputs();
    expect_arity(op, inputs, 3)?;
    let operand = &inputs[0];
    let indices = &inputs[1];
    let updates = &inputs[2];
    same_dtype(op, operand, updates, "operand and updates")?;
    if !indices.dtype.is_integer() {
        return Err(ShapeError::new(
            op,
            "scatter_indices must have an integer dtype",
        ));
    }

    let operand_rank = operand.rank();
    let indices_rank = indices.rank();
    let updates_rank = updates.rank();
    let ivd = d.index_vector_dim as usize;
    if ivd > indices_rank {
        return Err(ShapeError::new(op, "index_vector_dim out of range"));
    }
    expect_axes(
        op,
        "update_window_dims",
        &d.update_window_dims,
        updates_rank,
        true,
    )?;
    expect_axes(
        op,
        "inserted_window_dims",
        &d.inserted_window_dims,
        operand_rank,
        true,
    )?;
    expect_axes(
        op,
        "input_batching_dims",
        &d.input_batching_dims,
        operand_rank,
        true,
    )?;
    expect_disjoint_axes(
        op,
        "inserted_window_dims",
        &d.inserted_window_dims,
        "input_batching_dims",
        &d.input_batching_dims,
    )?;
    if operand_rank
        != d.update_window_dims.len() + d.inserted_window_dims.len() + d.input_batching_dims.len()
    {
        return Err(ShapeError::new(
            op,
            "operand rank must equal update window, inserted, and batching dimension counts",
        ));
    }
    expect_axes(
        op,
        "scatter_indices_batching_dims",
        &d.scatter_indices_batching_dims,
        indices_rank,
        false,
    )?;
    if d.scatter_indices_batching_dims
        .contains(&d.index_vector_dim)
    {
        return Err(ShapeError::new(
            op,
            "index_vector_dim must not be a scatter_indices batching dimension",
        ));
    }
    if d.input_batching_dims.len() != d.scatter_indices_batching_dims.len() {
        return Err(ShapeError::new(op, "batching dimension count mismatch"));
    }
    for (&operand_axis, &indices_axis) in d
        .input_batching_dims
        .iter()
        .zip(&d.scatter_indices_batching_dims)
    {
        if operand.dims[operand_axis as usize] != indices.dims[indices_axis as usize] {
            return Err(ShapeError::new(op, "batching dimension sizes differ"));
        }
    }
    expect_axes(
        op,
        "scatter_dims_to_operand_dims",
        &d.scatter_dims_to_operand_dims,
        operand_rank,
        false,
    )?;
    expect_disjoint_axes(
        op,
        "scatter_dims_to_operand_dims",
        &d.scatter_dims_to_operand_dims,
        "input_batching_dims",
        &d.input_batching_dims,
    )?;
    let index_count = if ivd < indices_rank {
        usize::try_from(indices.dims[ivd])
            .map_err(|_| ShapeError::new(op, "index vector size is negative"))?
    } else {
        1
    };
    if d.scatter_dims_to_operand_dims.len() != index_count {
        return Err(ShapeError::new(
            op,
            "scatter_dims_to_operand_dims length must match the index vector size",
        ));
    }

    let update_scatter_dims = (0..updates_rank)
        .filter(|axis| !d.update_window_dims.contains(&(*axis as u32)))
        .collect::<Vec<_>>();
    let indices_scatter_dims = (0..indices_rank)
        .filter(|&axis| axis != ivd)
        .collect::<Vec<_>>();
    if update_scatter_dims.len() != indices_scatter_dims.len()
        || update_scatter_dims.iter().zip(&indices_scatter_dims).any(
            |(&update_axis, &indices_axis)| updates.dims[update_axis] != indices.dims[indices_axis],
        )
    {
        return Err(ShapeError::new(
            op,
            "update scatter dimensions must match scatter_indices",
        ));
    }
    let operand_window_dims = (0..operand_rank)
        .filter(|axis| {
            !d.inserted_window_dims.contains(&(*axis as u32))
                && !d.input_batching_dims.contains(&(*axis as u32))
        })
        .collect::<Vec<_>>();
    if d.update_window_dims
        .iter()
        .zip(&operand_window_dims)
        .any(|(&update_axis, &operand_axis)| {
            updates.dims[update_axis as usize] > operand.dims[operand_axis]
        })
    {
        return Err(ShapeError::new(
            op,
            "update window dimensions must fit within the operand",
        ));
    }

    let Some(update_sig) = ctx.func_sig(update) else {
        return Err(ShapeError::new(op, "update computation is out of range"));
    };
    let scalar = Shape::scalar(operand.dtype).with_domain_axes(operand.domain_axes.iter().copied());
    if update_sig.params != [scalar.clone(), scalar.clone()] || update_sig.results != [scalar] {
        return Err(ShapeError::new(
            op,
            "update computation must have type (scalar<E>, scalar<E>) -> scalar<E>",
        ));
    }
    Ok(operand.clone())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scatter {
    pub dnums: ScatterDimensionNumbers,
    pub indices_are_sorted: bool,
    pub unique_indices: bool,
    pub update: FuncId,
}

impl Op for Scatter {
    fn mnemonic(&self) -> &'static str {
        "scatter"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let operand = scatter_infer(ctx, &self.dnums, self.update)?;
        Ok(smallvec![operand])
    }

    fn called_funcs(&self) -> &[FuncId] {
        std::slice::from_ref(&self.update)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "update_window_dims",
                axes_str(&self.dnums.update_window_dims),
            ),
            (
                "inserted_window_dims",
                axes_str(&self.dnums.inserted_window_dims),
            ),
            (
                "input_batching_dims",
                axes_str(&self.dnums.input_batching_dims),
            ),
            (
                "scatter_indices_batching_dims",
                axes_str(&self.dnums.scatter_indices_batching_dims),
            ),
            (
                "scatter_dims_to_operand_dims",
                axes_str(&self.dnums.scatter_dims_to_operand_dims),
            ),
            ("index_vector_dim", self.dnums.index_vector_dim.to_string()),
            ("indices_are_sorted", self.indices_are_sorted.to_string()),
            ("unique_indices", self.unique_indices.to_string()),
            ("update", format!("@{}", self.update.index())),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// Sort the (variadic) operands jointly along `dimension` by `comparator`. Each
/// operand yields a like-shaped result.
///
/// `comparator` takes the `2N` scalars `(lhs_0, rhs_0, …, lhs_(N-1), rhs_(N-1))`
/// — a pair per operand, interleaved as StableHLO orders them — and returns
/// whether `lhs` must precede `rhs`. A comparator that answers `false` both ways
/// leaves the two elements in their original order.
#[derive(Clone, Debug, PartialEq)]
pub struct Sort {
    pub dimension: u32,
    pub is_stable: bool,
    pub comparator: FuncId,
}

impl Op for Sort {
    fn mnemonic(&self) -> &'static str {
        "sort"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        if inputs.is_empty() {
            return Err(ShapeError::new("sort", "needs at least one operand"));
        }
        let first = &inputs[0];
        if self.dimension as usize >= first.rank() {
            return Err(ShapeError::new("sort", "dim out of range"));
        }
        for x in inputs {
            same_axes("sort", first, x, "operand")?;
        }
        let Some(comparator) = ctx.func_sig(self.comparator) else {
            return Err(ShapeError::new("sort", "comparator is out of range"));
        };
        let scalar_inputs = inputs
            .iter()
            .map(|shape| {
                Shape::scalar(shape.dtype).with_domain_axes(shape.domain_axes.iter().copied())
            })
            .collect::<Vec<_>>();
        let parameters = scalar_inputs
            .iter()
            .flat_map(|shape| [shape.clone(), shape.clone()])
            .collect::<Vec<_>>();
        if comparator.params != parameters
            || comparator.results
                != [Shape::scalar(DType::Bool).with_domain_axes(first.domain_axes.iter().copied())]
        {
            return Err(ShapeError::new(
                "sort",
                "comparator signature must compare scalar pairs and return a bool scalar",
            ));
        }
        Ok(inputs.iter().cloned().collect())
    }

    fn called_funcs(&self) -> &[FuncId] {
        std::slice::from_ref(&self.comparator)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("dimension", self.dimension.to_string()),
            ("is_stable", self.is_stable.to_string()),
            ("comparator", format!("@{}", self.comparator.index())),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// `while cond(carried) { carried = body(carried) }`. Operands and results are
/// the loop-carried values; `funcs = [cond, body]`.
#[derive(Clone, Debug, PartialEq)]
pub struct While {
    pub funcs: [FuncId; 2],
}

impl While {
    pub fn new(cond: FuncId, body: FuncId) -> While {
        While {
            funcs: [cond, body],
        }
    }

    pub fn cond(&self) -> FuncId {
        self.funcs[0]
    }

    pub fn body(&self) -> FuncId {
        self.funcs[1]
    }
}

impl Op for While {
    fn mnemonic(&self) -> &'static str {
        "while"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        let cond_p = ctx.func_params(self.cond());
        let cond_r = ctx.func_results(self.cond());
        let body_p = ctx.func_params(self.body());
        let body_r = ctx.func_results(self.body());
        if cond_p != inputs || body_p != inputs {
            return Err(ShapeError::new(
                "while",
                "cond/body params must match carried values",
            ));
        }
        if body_r != inputs {
            return Err(ShapeError::new(
                "while",
                "body results must match carried values",
            ));
        }
        if cond_r.len() != 1 || !cond_r[0].is_scalar() || cond_r[0].dtype != DType::Bool {
            return Err(ShapeError::new("while", "cond must return a bool scalar"));
        }
        Ok(inputs.iter().cloned().collect())
    }

    fn called_funcs(&self) -> &[FuncId] {
        &self.funcs
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("cond", format!("@{}", self.cond().index())),
            ("body", format!("@{}", self.body().index())),
        ]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// Select one of `branches` by a scalar selector and run it on the operands:
/// `branches[selector](operands)`. The selector (operand 0) is a scalar integer
/// index (or a bool, where false/true pick branch 0/1); operands 1.. are passed
/// to the chosen branch. Every branch shares the operand signature and result
/// shapes. Covers both StableHLO `case` (index) and `if` (bool).
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub branches: SmallVec<[FuncId; 2]>,
}

impl Op for Case {
    fn mnemonic(&self) -> &'static str {
        "case"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        if inputs.is_empty() {
            return Err(ShapeError::new("case", "needs a selector operand"));
        }
        let selector = &inputs[0];
        if !selector.is_scalar() || !(selector.dtype == DType::Bool || selector.dtype.is_integer())
        {
            return Err(ShapeError::new(
                "case",
                "selector must be a scalar integer or bool",
            ));
        }
        if self.branches.is_empty() {
            return Err(ShapeError::new("case", "needs at least one branch"));
        }
        let operands = &inputs[1..];
        let result = ctx.func_results(self.branches[0]).to_vec();
        for &b in &self.branches {
            if ctx.func_params(b) != operands {
                return Err(ShapeError::new(
                    "case",
                    "branch params must match the operands",
                ));
            }
            if ctx.func_results(b) != result.as_slice() {
                return Err(ShapeError::new(
                    "case",
                    "branches must return the same shapes",
                ));
            }
        }
        Ok(result.into_iter().collect())
    }

    fn called_funcs(&self) -> &[FuncId] {
        &self.branches
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        let names = self
            .branches
            .iter()
            .map(|b| format!("@{}", b.index()))
            .collect::<Vec<_>>()
            .join(", ");
        vec![("branches", format!("[{names}]"))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

/// Call `callee` with the operands; results are the callee's results.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub callee: FuncId,
}

impl Op for Call {
    fn mnemonic(&self) -> &'static str {
        "call"
    }

    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError> {
        let inputs = ctx.inputs();
        if ctx.func_params(self.callee) != inputs {
            return Err(ShapeError::new("call", "operands must match callee params"));
        }
        Ok(ctx.func_results(self.callee).iter().cloned().collect())
    }

    fn called_funcs(&self) -> &[FuncId] {
        std::slice::from_ref(&self.callee)
    }

    fn attrs(&self) -> Vec<(&'static str, String)> {
        vec![("callee", format!("@{}", self.callee.index()))]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}
