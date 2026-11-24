// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! pyo3 pyclasses over [`xir::builder::ModuleBuilder`]. The Python StableHLO
//! walker drives this imperatively — `begin_func`, `param`, one method per op
//! (operands as `u32` value ids, attributes as plain data, dtypes/kinds as short
//! strings), `end_func`, `finish` — then reads the built module's text. Shape
//! errors surface as `ValueError`.
//!
//! This lives in an rlib (no `#[pymodule]`) so multiple pyo3 extension modules
//! can each compile the same pyclass source into their own `.so`, giving every
//! extension its own dylib-local type identity for the built [`xir::ir::Module`].

#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    unsafe_op_in_unsafe_fn
)]

use algebraic_real::{AlgebraicReal, AlgebraicRealData};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use xir::builder::{ModuleBuilder, binary_from_name, unary_from_name};
use xir::dim_numbers::{
    Axes, DotDimensionNumbers, GatherDimensionNumbers, ScatterDimensionNumbers,
};
use xir::dtype::DType;
use xir::ir::{FuncId, Module, ValueId};
use xir::ops::{
    BroadcastInDim, Clamp, Compare, ComparisonDirection, ComparisonType, Concatenate, Constant,
    Convert, DotAlgorithm, DotGeneral, Gather, Literal, Pad, PrecisionType, Reduce, Reshape,
    Scatter, Select, Slice, Sort, Transpose, While,
};
use xir::shape::{Dims, Shape, ShapeError};

// ---- small mappings from the wire forms (strings / ints) to xir types ----

fn dtype_from_str(s: &str) -> PyResult<DType> {
    Ok(match s {
        "f16" => DType::F16,
        "bf16" => DType::Bf16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "i1" | "bool" => DType::Bool,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "u8" => DType::U8,
        "u16" => DType::U16,
        "u32" => DType::U32,
        "u64" => DType::U64,
        "c64" => DType::Complex64,
        "c128" => DType::Complex128,
        other => return Err(PyValueError::new_err(format!("unknown dtype {other:?}"))),
    })
}

fn precision_type(s: &str) -> PyResult<PrecisionType> {
    PrecisionType::parse(s)
        .ok_or_else(|| PyValueError::new_err(format!("unknown precision type {s:?}")))
}

fn compare_dir(s: &str) -> PyResult<ComparisonDirection> {
    use ComparisonDirection::*;
    Ok(match s {
        "eq" => Eq,
        "ne" => Ne,
        "lt" => Lt,
        "le" => Le,
        "gt" => Gt,
        "ge" => Ge,
        other => return Err(PyValueError::new_err(format!("unknown compare {other:?}"))),
    })
}

/// StableHLO `ComparisonType` from its enum name; `""` = `None` (infer).
fn compare_type(s: &str) -> PyResult<Option<ComparisonType>> {
    use ComparisonType::*;
    Ok(match s {
        "" => None,
        "FLOAT" => Some(Float),
        "TOTALORDER" => Some(TotalOrder),
        "SIGNED" => Some(Signed),
        "UNSIGNED" => Some(Unsigned),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown compare type {other:?}"
            )));
        }
    })
}

fn to_py(e: ShapeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn vids(xs: &[u32]) -> Vec<ValueId> {
    xs.iter().map(|&x| ValueId(x)).collect()
}

#[pyclass(unsendable)]
pub struct XirBuilder {
    inner: ModuleBuilder,
}

impl XirBuilder {
    fn one(&mut self, op: impl xir::op::Op + 'static, inputs: &[ValueId]) -> PyResult<u32> {
        self.inner.try_op1(op, inputs).map(|v| v.0).map_err(to_py)
    }

    fn one_boxed(&mut self, op: Box<dyn xir::op::Op>, inputs: &[ValueId]) -> PyResult<u32> {
        self.inner
            .try_op1_boxed(op.into(), inputs)
            .map(|v| v.0)
            .map_err(to_py)
    }

    fn many(&mut self, op: impl xir::op::Op + 'static, inputs: &[ValueId]) -> PyResult<Vec<u32>> {
        self.inner
            .try_op(op, inputs)
            .map(|r| r.iter().map(|v| v.0).collect())
            .map_err(to_py)
    }
}

#[pymethods]
impl XirBuilder {
    #[new]
    fn new() -> XirBuilder {
        XirBuilder {
            inner: ModuleBuilder::new(),
        }
    }

    // ---- function lifecycle ----

    fn begin_func(&mut self, name: &str) {
        self.inner.begin_func(name);
    }

    fn end_func(&mut self, results: Vec<u32>) -> u32 {
        self.inner.end_func(vids(&results)).0
    }

    fn param(&mut self, dtype: &str, dims: Vec<i64>) -> PyResult<u32> {
        let shape = Shape::new(dtype_from_str(dtype)?, dims);
        Ok(self.inner.param(shape).0)
    }

    fn finish(&mut self, entry: u32) -> PyResult<XirModule> {
        let mb = std::mem::take(&mut self.inner);
        mb.finish(FuncId(entry))
            .map(|m| XirModule { inner: m })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // ---- elementwise ----

    fn unary(&mut self, kind: &str, x: u32) -> PyResult<u32> {
        let op = unary_from_name(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown unary {kind:?}")))?;
        self.one_boxed(op, &[ValueId(x)])
    }

    fn binary(&mut self, kind: &str, a: u32, b: u32) -> PyResult<u32> {
        let op = binary_from_name(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown binary {kind:?}")))?;
        self.one_boxed(op, &[ValueId(a), ValueId(b)])
    }

    fn compare(&mut self, dir: &str, compare_type_name: &str, a: u32, b: u32) -> PyResult<u32> {
        self.one(
            Compare {
                comparison_direction: compare_dir(dir)?,
                compare_type: compare_type(compare_type_name)?,
            },
            &[ValueId(a), ValueId(b)],
        )
    }

    fn select(&mut self, pred: u32, on_true: u32, on_false: u32) -> PyResult<u32> {
        self.one(
            Select,
            &[ValueId(pred), ValueId(on_true), ValueId(on_false)],
        )
    }

    fn clamp(&mut self, low: u32, x: u32, high: u32) -> PyResult<u32> {
        self.one(Clamp, &[ValueId(low), ValueId(x), ValueId(high)])
    }

    fn convert(&mut self, x: u32, dtype: &str) -> PyResult<u32> {
        let dtype = dtype_from_str(dtype)?;
        self.one(Convert { dtype }, &[ValueId(x)])
    }

    // ---- structural ----

    fn broadcast(
        &mut self,
        x: u32,
        out_dims: Vec<i64>,
        broadcast_dimensions: Vec<u32>,
    ) -> PyResult<u32> {
        self.one(
            BroadcastInDim {
                out_dims: Dims::from_slice(&out_dims),
                broadcast_dimensions: Axes::from_slice(&broadcast_dimensions),
            },
            &[ValueId(x)],
        )
    }

    fn transpose(&mut self, x: u32, permutation: Vec<u32>) -> PyResult<u32> {
        self.one(
            Transpose {
                permutation: Axes::from_slice(&permutation),
            },
            &[ValueId(x)],
        )
    }

    fn reshape(&mut self, x: u32, out_dims: Vec<i64>) -> PyResult<u32> {
        self.one(
            Reshape {
                out_dims: Dims::from_slice(&out_dims),
            },
            &[ValueId(x)],
        )
    }

    fn slice(
        &mut self,
        x: u32,
        start_indices: Vec<i64>,
        limit_indices: Vec<i64>,
        strides: Vec<i64>,
    ) -> PyResult<u32> {
        self.one(
            Slice {
                start_indices: Dims::from_slice(&start_indices),
                limit_indices: Dims::from_slice(&limit_indices),
                strides: Dims::from_slice(&strides),
            },
            &[ValueId(x)],
        )
    }

    fn concatenate(&mut self, inputs: Vec<u32>, dimension: u32) -> PyResult<u32> {
        self.one(Concatenate { dimension }, &vids(&inputs))
    }

    fn pad(
        &mut self,
        x: u32,
        pad_value: u32,
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<i64>,
    ) -> PyResult<u32> {
        self.one(
            Pad {
                edge_padding_low: Dims::from_slice(&edge_padding_low),
                edge_padding_high: Dims::from_slice(&edge_padding_high),
                interior_padding: Dims::from_slice(&interior_padding),
            },
            &[ValueId(x), ValueId(pad_value)],
        )
    }

    fn constant(&mut self, dtype: &str, dims: Vec<i64>, data: Vec<u8>) -> PyResult<u32> {
        let shape = Shape::new(dtype_from_str(dtype)?, dims);
        self.one(
            Constant {
                literal: Literal::new(shape, data),
            },
            &[],
        )
    }

    fn exact_constant(
        &mut self,
        dtype: &str,
        dims: Vec<i64>,
        data: Vec<u8>,
        values: &Bound<'_, PyList>,
    ) -> PyResult<u32> {
        let shape = Shape::new(dtype_from_str(dtype)?, dims);
        let exact = values
            .iter()
            .map(|value| -> PyResult<AlgebraicReal> {
                let record = value
                    .downcast::<PyList>()
                    .map_err(|_| PyValueError::new_err("algebraic value must be a list"))?;
                let tag: String = record.get_item(0)?.extract()?;
                let encoded = match tag.as_str() {
                    "m" => {
                        let terms = record.get_item(1)?.downcast_into::<PyList>()?;
                        AlgebraicRealData::Multiquadratic(
                            terms
                                .iter()
                                .map(|term| {
                                    let term = term.downcast_into::<PyList>()?;
                                    let n: String = term.get_item(0)?.extract()?;
                                    let d: String = term.get_item(1)?.extract()?;
                                    let r: String = term.get_item(2)?.extract()?;
                                    Ok((
                                        n.parse().map_err(|_| {
                                            PyValueError::new_err("invalid numerator")
                                        })?,
                                        d.parse().map_err(|_| {
                                            PyValueError::new_err("invalid denominator")
                                        })?,
                                        r.parse().map_err(|_| {
                                            PyValueError::new_err("invalid radicand")
                                        })?,
                                    ))
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        )
                    }
                    "p" => {
                        let coefficients: Vec<String> = record.get_item(1)?.extract()?;
                        AlgebraicRealData::Polynomial {
                            coefficients: coefficients
                                .into_iter()
                                .map(|value| {
                                    value
                                        .parse()
                                        .map_err(|_| PyValueError::new_err("invalid coefficient"))
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                            root_index: record.get_item(2)?.extract()?,
                        }
                    }
                    _ => {
                        return Err(PyValueError::new_err(
                            "unknown AlgebraicReal representation",
                        ));
                    }
                };
                AlgebraicReal::from_data(encoded)
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let literal = Literal::new(shape, data)
            .with_exact(exact)
            .map_err(PyValueError::new_err)?;
        self.one(Constant { literal }, &[])
    }

    // ---- contraction ----

    #[allow(clippy::too_many_arguments)]
    fn dot(
        &mut self,
        lhs: u32,
        rhs: u32,
        lhs_contracting: Vec<u32>,
        rhs_contracting: Vec<u32>,
        lhs_batch: Vec<u32>,
        rhs_batch: Vec<u32>,
        lhs_precision: &str,
        rhs_precision: &str,
        accum_precision: &str,
        lhs_components: u8,
        rhs_components: u8,
        num_primitive_ops: u8,
        allow_imprecise_accum: bool,
    ) -> PyResult<u32> {
        let dnums =
            DotDimensionNumbers::new(&lhs_contracting, &rhs_contracting, &lhs_batch, &rhs_batch);
        let algorithm = DotAlgorithm {
            lhs: precision_type(lhs_precision)?,
            rhs: precision_type(rhs_precision)?,
            accum: precision_type(accum_precision)?,
            lhs_components,
            rhs_components,
            num_primitive_ops,
            allow_imprecise_accum,
        };
        self.one(
            DotGeneral { dnums, algorithm },
            &[ValueId(lhs), ValueId(rhs)],
        )
    }

    // ---- indexed ----

    fn gather(
        &mut self,
        operand: u32,
        start_indices: u32,
        offset_dims: Vec<u32>,
        collapsed_slice_dims: Vec<u32>,
        operand_batching_dims: Vec<u32>,
        start_indices_batching_dims: Vec<u32>,
        start_index_map: Vec<u32>,
        index_vector_dim: u32,
        slice_sizes: Vec<i64>,
        indices_are_sorted: bool,
    ) -> PyResult<u32> {
        let dnums = GatherDimensionNumbers {
            offset_dims: Axes::from_slice(&offset_dims),
            collapsed_slice_dims: Axes::from_slice(&collapsed_slice_dims),
            operand_batching_dims: Axes::from_slice(&operand_batching_dims),
            start_indices_batching_dims: Axes::from_slice(&start_indices_batching_dims),
            start_index_map: Axes::from_slice(&start_index_map),
            index_vector_dim,
        };
        self.one(
            Gather {
                dnums,
                slice_sizes: Dims::from_slice(&slice_sizes),
                indices_are_sorted,
            },
            &[ValueId(operand), ValueId(start_indices)],
        )
    }

    fn scatter(
        &mut self,
        operand: u32,
        scatter_indices: u32,
        updates: u32,
        update_window_dims: Vec<u32>,
        inserted_window_dims: Vec<u32>,
        input_batching_dims: Vec<u32>,
        scatter_indices_batching_dims: Vec<u32>,
        scatter_dims_to_operand_dims: Vec<u32>,
        index_vector_dim: u32,
        indices_are_sorted: bool,
        unique_indices: bool,
        update: u32,
    ) -> PyResult<u32> {
        let dnums = ScatterDimensionNumbers {
            update_window_dims: Axes::from_slice(&update_window_dims),
            inserted_window_dims: Axes::from_slice(&inserted_window_dims),
            input_batching_dims: Axes::from_slice(&input_batching_dims),
            scatter_indices_batching_dims: Axes::from_slice(&scatter_indices_batching_dims),
            scatter_dims_to_operand_dims: Axes::from_slice(&scatter_dims_to_operand_dims),
            index_vector_dim,
        };
        self.one(
            Scatter {
                dnums,
                indices_are_sorted,
                unique_indices,
                update: FuncId(update),
            },
            &[ValueId(operand), ValueId(scatter_indices), ValueId(updates)],
        )
    }

    // ---- higher-order ----

    fn reduce(
        &mut self,
        inputs: Vec<u32>,
        inits: Vec<u32>,
        dimensions: Vec<u32>,
        reducer: u32,
    ) -> PyResult<Vec<u32>> {
        let mut all = vids(&inputs);
        all.extend(vids(&inits));
        self.many(
            Reduce {
                dimensions: Axes::from_slice(&dimensions),
                reducer: FuncId(reducer),
            },
            &all,
        )
    }

    fn sort(
        &mut self,
        inputs: Vec<u32>,
        dimension: u32,
        is_stable: bool,
        comparator: u32,
    ) -> PyResult<Vec<u32>> {
        self.many(
            Sort {
                dimension,
                is_stable,
                comparator: FuncId(comparator),
            },
            &vids(&inputs),
        )
    }

    fn repeat_while(&mut self, carried: Vec<u32>, cond: u32, body: u32) -> PyResult<Vec<u32>> {
        self.many(While::new(FuncId(cond), FuncId(body)), &vids(&carried))
    }
}

#[pyclass(unsendable)]
pub struct XirModule {
    pub inner: Module,
}

#[pymethods]
impl XirModule {
    fn text(&self) -> String {
        xir::print::module_to_string(&self.inner)
    }

    fn function_names(&self) -> Vec<String> {
        self.inner
            .funcs
            .iter()
            .map(|function| function.name.clone())
            .collect()
    }

    fn entry_name(&self) -> String {
        self.inner.entry_func().name.clone()
    }

    fn operation_names(&self, function: usize) -> PyResult<Vec<String>> {
        let function = self
            .inner
            .funcs
            .get(function)
            .ok_or_else(|| PyValueError::new_err("xir function index out of range"))?;
        Ok(function
            .nodes
            .iter()
            .map(|node| node.op.mnemonic().to_string())
            .collect())
    }

    fn __str__(&self) -> String {
        self.text()
    }

    fn __repr__(&self) -> String {
        format!("<xir.Module: {} func(s)>", self.inner.funcs.len())
    }
}
