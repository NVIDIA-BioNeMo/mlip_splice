// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Emitting an indexed tensor: one element of it, as an expression.
//!
//! [`super::IndexedTensor`] says what a value is; this walks that description
//! down to the reads and arithmetic that produce one element of it, following
//! the schedule [`super::schedule`] planned. The recursion is over the
//! expression, not over the element: an element index is an argument, so a
//! constant one folds a whole branch away while a dynamic one selects.

use gpu_ir::{Carry, CmpOp, Scalar, Trip, Ty, Value};
use xir::ops::Scatter;

use super::super::{
    Evaluator, convert, lower_scalar_region, math_scalar, output_factor_elementwise, zero,
};
use super::{
    IndexedExpression, IndexedTensor, STATIC_REDUCTION_TERMS_PER_BINDING, coordinate, ravel_index,
    schedule, unrolls,
};

impl Evaluator<'_> {
    fn indexed_name(&mut self, stem: &str) -> String {
        let index = self.indexed_names;
        self.indexed_names += 1;
        format!("loom_{stem}_{index}")
    }

    /// [`Evaluator::emit_indexed`] at a constant element, which is how a plan
    /// names one and how an output is expanded.
    pub(in crate::eval) fn emit_indexed_at(
        &mut self,
        tensor: &IndexedTensor,
        element: usize,
    ) -> Result<Value, String> {
        let index = self.stmts.i32(element as i64);
        self.emit_indexed(tensor, index)
    }

    pub(in crate::eval) fn emit_indexed(
        &mut self,
        tensor: &IndexedTensor,
        index: Value,
    ) -> Result<Value, String> {
        let key = self
            .stmts
            .as_int_const(&index)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&element| element < tensor.elements)
            .map(|element| tensor.element_key(element));
        if let Some(key) = key
            && let Some(value) = self.indexed_cache.get(&key)
        {
            return Ok(value.clone());
        }
        let value = self.emit_indexed_expression(tensor, index)?;
        let Some(key) = key.filter(|key| self.names_a_shared_value(*key, tensor)) else {
            return Ok(value);
        };
        let Some(ty) = self.binding_ty(tensor)? else {
            return Ok(value);
        };
        let name = self.indexed_name("indexed_value");
        let bound = self.stmts.bind_value_as(name, ty, value);
        self.indexed_cache.insert(key, bound.clone());
        Ok(bound)
    }

    /// Whether the plan named this value. Naming is decided once, by weighing
    /// what rebuilding a value costs against the register it would hold; the
    /// emitter only reads that decision.
    fn names_a_shared_value(&self, key: (usize, usize), _tensor: &IndexedTensor) -> bool {
        self.indexed_named.contains(&key)
    }

    /// The type a named indexed value holds. `None` for values that are always
    /// folded into their consumer's operand and so never named.
    fn binding_ty(&self, tensor: &IndexedTensor) -> Result<Option<Ty>, String> {
        Ok(Some(match tensor.expression.as_ref() {
            IndexedExpression::Elementwise { dtype, .. } => {
                Ty::vec(math_scalar(*dtype)?, self.lanes)
            }
            IndexedExpression::Convert { to, .. } => Ty::vec(math_scalar(*to)?, self.lanes),
            IndexedExpression::Pointer {
                scalar,
                bool_storage,
                load_lanes,
                splat_lanes,
                ..
            } => Ty::vec(
                if *bool_storage { Scalar::Pred } else { *scalar },
                (*load_lanes).max(*splat_lanes),
            ),
            IndexedExpression::Reduce { reducer, .. } => {
                let func = self.module.func(*reducer);
                Ty::vec(
                    math_scalar(func.value_shape(func.results[0]).dtype)?,
                    self.lanes,
                )
            }
            IndexedExpression::Gather { input, .. } => return self.binding_ty(input),
            IndexedExpression::Dense(_)
            | IndexedExpression::Constant { .. }
            | IndexedExpression::Reindex { .. }
            | IndexedExpression::Concatenate { .. }
            | IndexedExpression::Pad { .. } => return Ok(None),
        }))
    }

    /// Plans the whole indexed program before any of it is emitted: which shared
    /// values to name, and in what order, so the live count stays inside the
    /// budget without rebuilding more than the budget requires.
    ///
    /// The plan is the only analysis. It resolves re-indexing, concatenation and
    /// padding before hash-consing, so a value reached through two different
    /// aliases is one node and its reads are counted together; keying on the
    /// aliases instead would credit each of them once and miss the sharing
    /// underneath. The set it names is exactly the set emitted, so the live count
    /// it measured is the live count the program has.
    pub(in crate::eval) fn prepare_indexed_plan(
        &mut self,
        roots: &[(IndexedTensor, usize)],
    ) -> Result<(), String> {
        let dag = schedule::plan(roots, self.indexed_register_budget)?;
        self.indexed_named.clear();
        for node in 0..dag.nodes.len() {
            if dag.names(node) {
                self.indexed_named.insert(dag.key(node));
            }
        }
        // Emitting in the planned order is what makes the live count match what
        // the plan measured; the per-output expansion then only reads names.
        for &node in &dag.order {
            if !dag.names(node) {
                continue;
            }
            let scalar_node = &dag.nodes[node];
            let (tensor, element) = (scalar_node.tensor.clone(), scalar_node.element);
            self.emit_indexed_at(&tensor, element)?;
        }
        Ok(())
    }

    fn emit_indexed_expression(
        &mut self,
        tensor: &IndexedTensor,
        index: Value,
    ) -> Result<Value, String> {
        let lanes = self.lanes;
        match tensor.expression.as_ref() {
            IndexedExpression::Dense(values) => {
                if let Some(value) = self.stmts.as_int_const(&index)
                    && let Ok(element) = usize::try_from(value)
                    && let Some(value) = values.get(element)
                {
                    return Ok(value.clone());
                }
                let values = values.clone();
                let mut selected = values
                    .last()
                    .cloned()
                    .ok_or_else(|| "indexed dense tensor is empty".to_string())?;
                for (element, value) in values[..values.len() - 1].iter().enumerate().rev() {
                    let wanted = self.stmts.i32(element as i64);
                    let predicate = self.stmts.cmp(CmpOp::Eq, index.clone(), wanted);
                    selected = self.stmts.select(predicate, value.clone(), selected);
                }
                Ok(selected)
            }
            IndexedExpression::Pointer {
                base,
                scalar,
                stride,
                load_lanes,
                splat_lanes,
                bool_storage,
                alive,
                evict,
            } => {
                let (scalar, stride) = (*scalar, *stride);
                let (load_lanes, splat_lanes) = (*load_lanes, *splat_lanes);
                let (bool_storage, evict) = (*bool_storage, *evict);
                let (base, alive) = (base.clone(), alive.clone());
                let b = &mut self.stmts;
                let wide = b.cast(Scalar::S64, index);
                let stride = b.i64(stride);
                let offset = b.mul(wide, stride);
                let cell = b.offset(base, offset);
                let mut value = b.load_evict(cell, load_lanes, true, evict);
                if splat_lanes > load_lanes {
                    value = b.splat(splat_lanes, value);
                }
                if bool_storage {
                    let bits = b.cast(Scalar::U32, value);
                    let zero = b.int(Scalar::U32, 0);
                    let zero = b.splat(splat_lanes, zero);
                    value = b.cmp(CmpOp::Ne, bits, zero);
                }
                Ok(match alive {
                    Some(predicate) if scalar == Scalar::Pred => {
                        let predicate = b.splat(splat_lanes, predicate);
                        b.and(predicate, value)
                    }
                    Some(predicate) => {
                        let zero = zero(b, scalar);
                        let zero = b.splat(splat_lanes, zero);
                        b.select(predicate, value, zero)
                    }
                    None => value,
                })
            }
            IndexedExpression::Constant {
                name,
                scalar,
                data,
                immediate,
            } => {
                if let Some(value) = self.stmts.as_int_const(&index)
                    && let Ok(element) = usize::try_from(value)
                    && let Some(value) = immediate.get(element)
                {
                    return Ok(value.clone());
                }
                let value =
                    self.stmts
                        .const_load_shared(name.clone(), *scalar, data.clone(), index);
                Ok(if lanes == 1 {
                    value
                } else {
                    self.stmts.splat(lanes, value)
                })
            }
            IndexedExpression::Reindex { input, indices } => {
                let index = indices.load(&mut self.stmts, index);
                self.emit_indexed(input, index)
            }
            IndexedExpression::Concatenate { inputs, offsets } => {
                if let Some(value) = self.stmts.as_int_const(&index)
                    && let Ok(element) = usize::try_from(value)
                    && element < tensor.elements
                {
                    let input = offsets.partition_point(|&offset| offset <= element) - 1;
                    let inputs = inputs.clone();
                    return self.emit_indexed_at(&inputs[input], element - offsets[input]);
                }
                let inputs = inputs.clone();
                let offsets = offsets.clone();
                let mut selected = None;
                for (input, &[start, end]) in inputs.iter().zip(offsets.array_windows::<2>()).rev()
                {
                    let start = self.stmts.i32(start as i64);
                    let shifted = self.stmts.sub(index.clone(), start);
                    let floor = self.stmts.i32(0);
                    let raised = self.stmts.max(shifted, floor);
                    let ceiling = self.stmts.i32(input.elements as i64 - 1);
                    let local = self.stmts.min(raised, ceiling);
                    let value = self.emit_indexed(input, local)?;
                    selected = Some(match selected {
                        None => value,
                        Some(selected) => {
                            let limit = self.stmts.i32(end as i64);
                            let within = self.stmts.lt(index.clone(), limit);
                            self.stmts.select(within, value, selected)
                        }
                    });
                }
                selected.ok_or_else(|| "indexed concatenate has no inputs".to_string())
            }
            IndexedExpression::Gather {
                input,
                starts,
                operand_dims,
                start_dims,
                output_dims,
                op,
            } => {
                let d = &op.dnums;
                let index_vector_dim = d.index_vector_dim as usize;
                let start_batch_axes = (0..start_dims.len())
                    .filter(|&axis| axis != index_vector_dim)
                    .collect::<Vec<_>>();
                let output_batch_axes = (0..output_dims.len())
                    .filter(|axis| !d.offset_dims.contains(&(*axis as u32)))
                    .collect::<Vec<_>>();
                let window_operand_axes = (0..operand_dims.len())
                    .filter(|axis| !d.collapsed_slice_dims.contains(&(*axis as u32)))
                    .collect::<Vec<_>>();
                let output_coord = (0..output_dims.len())
                    .map(|axis| coordinate(&mut self.stmts, index.clone(), output_dims, axis))
                    .collect::<Vec<_>>();
                let origin = self.stmts.i32(0);
                let mut start_coord = vec![origin.clone(); start_dims.len()];
                for (&start_axis, &output_axis) in start_batch_axes.iter().zip(&output_batch_axes) {
                    start_coord[start_axis] = output_coord[output_axis].clone();
                }
                let mut operand_coord = vec![origin; operand_dims.len()];
                let starts = starts.clone();
                for (component, &operand_axis) in d.start_index_map.iter().enumerate() {
                    if index_vector_dim < start_dims.len() {
                        start_coord[index_vector_dim] = self.stmts.i32(component as i64);
                    }
                    let flat = ravel_index(&mut self.stmts, &start_coord, start_dims);
                    let mut start = self.emit_indexed(&starts, flat)?;
                    if lanes > 1 {
                        start = self.stmts.extract(0, start);
                    }
                    let operand_axis = operand_axis as usize;
                    let floor = self.stmts.i32(0);
                    let raised = self.stmts.max(start, floor);
                    let ceiling = self
                        .stmts
                        .i32(operand_dims[operand_axis] - op.slice_sizes[operand_axis]);
                    operand_coord[operand_axis] = self.stmts.min(raised, ceiling);
                }
                for (&operand_axis, &start_axis) in d
                    .operand_batching_dims
                    .iter()
                    .zip(&d.start_indices_batching_dims)
                {
                    let batch = start_batch_axes
                        .iter()
                        .position(|&axis| axis == start_axis as usize)
                        .ok_or_else(|| {
                            "gather batching axis is the index-vector axis".to_string()
                        })?;
                    operand_coord[operand_axis as usize] =
                        output_coord[output_batch_axes[batch]].clone();
                }
                for (&operand_axis, &output_axis) in window_operand_axes.iter().zip(&d.offset_dims)
                {
                    operand_coord[operand_axis] = self.stmts.add(
                        operand_coord[operand_axis].clone(),
                        output_coord[output_axis as usize].clone(),
                    );
                }
                let input = input.clone();
                let flat = ravel_index(&mut self.stmts, &operand_coord, operand_dims);
                self.emit_indexed(&input, flat)
            }
            IndexedExpression::Elementwise { op, dtype, inputs } => {
                let (op, dtype) = (*op, *dtype);
                let inputs = inputs.clone();
                let mut values = Vec::with_capacity(inputs.len());
                for input in &inputs {
                    let input_index = if input.elements == 1 {
                        self.stmts.i32(0)
                    } else {
                        index.clone()
                    };
                    values.push(self.emit_indexed(input, input_index)?);
                }
                output_factor_elementwise(&mut self.stmts, op, dtype, values, lanes)
            }
            IndexedExpression::Convert { from, to, input } => {
                let (from, to) = (*from, *to);
                let input = input.clone();
                let value = self.emit_indexed(&input, index)?;
                convert(&mut self.stmts, value, from, to, lanes)
            }
            IndexedExpression::Pad {
                input,
                pad,
                input_indices,
                inside,
            } => {
                let input_index = input_indices.load(&mut self.stmts, index.clone());
                let flag = inside.load(&mut self.stmts, index);
                let one = self.stmts.i32(1);
                let predicate = self.stmts.eq(flag, one);
                let (input, pad) = (input.clone(), pad.clone());
                let taken = self.emit_indexed(&input, input_index)?;
                let padded = self.emit_indexed_at(&pad, 0)?;
                Ok(self.stmts.select(predicate, taken, padded))
            }
            IndexedExpression::Reduce {
                input,
                init,
                init_per_output,
                reducer,
                input_indices,
                offsets,
                uniform_terms,
            } => {
                let (init_per_output, reducer, uniform_terms) =
                    (*init_per_output, *reducer, *uniform_terms);
                let (input, init) = (input.clone(), init.clone());
                let (input_indices, offsets) = (input_indices.clone(), offsets.clone());
                let module = self.module;
                let init_index = if init_per_output {
                    index.clone()
                } else {
                    self.stmts.i32(0)
                };
                let mut accumulator = self.emit_indexed(&init, init_index)?;
                if let Some(value) = self.stmts.as_int_const(&index)
                    && let Ok(output) = usize::try_from(value)
                    && let (Some(&start), Some(&stop)) =
                        (offsets.values.get(output), offsets.values.get(output + 1))
                    && unrolls(stop - start)
                {
                    let scalar = math_scalar(
                        module
                            .func(reducer)
                            .value_shape(module.func(reducer).results[0])
                            .dtype,
                    )?;
                    let indices = input_indices.values[start..stop].to_vec();
                    let total = indices.len();
                    for (term, input_index) in indices.into_iter().enumerate() {
                        let value = self.emit_indexed_at(&input, input_index)?;
                        accumulator = lower_scalar_region(
                            &mut self.stmts,
                            module,
                            reducer,
                            vec![accumulator, value],
                            lanes,
                        )?;
                        if (term + 1).is_multiple_of(STATIC_REDUCTION_TERMS_PER_BINDING)
                            && term + 1 != total
                        {
                            let name = self.indexed_name("indexed_partial_reduce");
                            accumulator =
                                self.stmts
                                    .bind_value_as(name, Ty::vec(scalar, lanes), accumulator);
                        }
                    }
                    return Ok(accumulator);
                }
                if uniform_terms.is_some_and(|terms| terms <= 8) {
                    let terms_per_output = uniform_terms.unwrap();
                    for term in 0..terms_per_output {
                        let width = self.stmts.i32(terms_per_output as i64);
                        let base = self.stmts.mul(index.clone(), width);
                        let offset = self.stmts.i32(term as i64);
                        let lookup = self.stmts.add(base, offset);
                        let lookup = input_indices.load(&mut self.stmts, lookup);
                        let value = self.emit_indexed(&input, lookup)?;
                        accumulator = lower_scalar_region(
                            &mut self.stmts,
                            module,
                            reducer,
                            vec![accumulator, value],
                            lanes,
                        )?;
                    }
                    return Ok(accumulator);
                }

                // A reduction whose term count is only known at runtime is a
                // counted loop standing for the value it reduces to: it is an
                // operand, so it nests wherever the expression that reads it is.
                let (start, stop) = match uniform_terms {
                    Some(terms) => {
                        let width = self.stmts.i32(terms as i64);
                        let start = self.stmts.mul(index, width.clone());
                        let stop = self.stmts.add(start.clone(), width);
                        (start, stop)
                    }
                    None => {
                        let start = offsets.load(&mut self.stmts, index.clone());
                        let one = self.stmts.i32(1);
                        let next = self.stmts.add(index, one);
                        let stop = offsets.load(&mut self.stmts, next);
                        (start, stop)
                    }
                };
                let step = self.stmts.i32(1);
                let ty = self.stmts.value_ty(&accumulator).data();
                let open = self.stmts.open_loop_body(
                    Scalar::S32,
                    vec![Carry {
                        ty,
                        init: accumulator,
                        into: None,
                    }],
                );
                let term = open.index();
                let carried = open.given()[0].clone();
                let lookup = input_indices.load(&mut self.stmts, term);
                let value = self.emit_indexed(&input, lookup)?;
                let updated = lower_scalar_region(
                    &mut self.stmts,
                    module,
                    reducer,
                    vec![carried, value],
                    lanes,
                )?;
                let body = self.stmts.close_loop_body(open, vec![updated]);
                let folded = self
                    .stmts
                    .fold_loop(Trip::PerThread, start, stop, step, body);
                folded
                    .into_iter()
                    .next()
                    .ok_or_else(|| "a reduction loop carries its accumulator".to_string())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::eval) fn emit_scatter_destination(
        &mut self,
        indices: &IndexedTensor,
        update_index: Value,
        operand_dims: &[i64],
        scatter_dims: &[i64],
        update_dims: &[i64],
        op: &Scatter,
        sentinel: usize,
    ) -> Result<Value, String> {
        let lanes = self.lanes;
        let d = &op.dnums;
        let index_vector_dim = d.index_vector_dim as usize;
        let scatter_batch_axes = (0..scatter_dims.len())
            .filter(|&axis| axis != index_vector_dim)
            .collect::<Vec<_>>();
        let update_scatter_axes = (0..update_dims.len())
            .filter(|axis| !d.update_window_dims.contains(&(*axis as u32)))
            .collect::<Vec<_>>();
        let window_operand_axes = (0..operand_dims.len())
            .filter(|axis| {
                !d.inserted_window_dims.contains(&(*axis as u32))
                    && !d.input_batching_dims.contains(&(*axis as u32))
            })
            .collect::<Vec<_>>();
        let update_coord = (0..update_dims.len())
            .map(|axis| coordinate(&mut self.stmts, update_index.clone(), update_dims, axis))
            .collect::<Vec<_>>();
        let origin = self.stmts.i32(0);
        let mut scatter_coord = vec![origin.clone(); scatter_dims.len()];
        for (&scatter_axis, &update_axis) in scatter_batch_axes.iter().zip(&update_scatter_axes) {
            scatter_coord[scatter_axis] = update_coord[update_axis].clone();
        }
        let mut operand_coord = vec![origin; operand_dims.len()];
        let indices = indices.clone();
        for (component, &operand_axis) in d.scatter_dims_to_operand_dims.iter().enumerate() {
            if index_vector_dim < scatter_dims.len() {
                scatter_coord[index_vector_dim] = self.stmts.i32(component as i64);
            }
            let flat = ravel_index(&mut self.stmts, &scatter_coord, scatter_dims);
            let mut value = self.emit_indexed(&indices, flat)?;
            if lanes > 1 {
                value = self.stmts.extract(0, value);
            }
            operand_coord[operand_axis as usize] = value;
        }
        for (&operand_axis, &scatter_axis) in d
            .input_batching_dims
            .iter()
            .zip(&d.scatter_indices_batching_dims)
        {
            let batch = scatter_batch_axes
                .iter()
                .position(|&axis| axis == scatter_axis as usize)
                .ok_or_else(|| "scatter batching axis is the index-vector axis".to_string())?;
            operand_coord[operand_axis as usize] = update_coord[update_scatter_axes[batch]].clone();
        }
        for (&operand_axis, &update_axis) in window_operand_axes.iter().zip(&d.update_window_dims) {
            operand_coord[operand_axis] = self.stmts.add(
                operand_coord[operand_axis].clone(),
                update_coord[update_axis as usize].clone(),
            );
        }
        let mut out_of_bounds: Option<Value> = None;
        for (coordinate, &dim) in operand_coord.iter().zip(operand_dims) {
            let floor = self.stmts.i32(0);
            let below = self.stmts.lt(coordinate.clone(), floor);
            let extent = self.stmts.i32(dim);
            let above = self.stmts.cmp(CmpOp::Ge, coordinate.clone(), extent);
            let outside = self.stmts.or(below, above);
            out_of_bounds = Some(match out_of_bounds {
                None => outside,
                Some(held) => self.stmts.or(held, outside),
            });
        }
        let out_of_bounds = match out_of_bounds {
            Some(out_of_bounds) => out_of_bounds,
            None => {
                let zero = self.stmts.i32(0);
                let one = self.stmts.i32(1);
                self.stmts.eq(zero, one)
            }
        };
        let sentinel = self.stmts.i32(sentinel as i64);
        let flat = ravel_index(&mut self.stmts, &operand_coord, operand_dims);
        Ok(self.stmts.select(out_of_bounds, sentinel, flat))
    }
}
