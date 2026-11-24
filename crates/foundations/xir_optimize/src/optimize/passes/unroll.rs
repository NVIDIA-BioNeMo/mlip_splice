// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replacing an operation that carries a region with the circuit it denotes.
//!
//! Two operations state a computation as a region applied a fixed number of
//! times. A `sort` along a dimension of length `n` is a sorting network: `n`
//! rounds of compare-exchange gates between adjacent positions, one application
//! of the comparator per gate. A `while` is the composition of its body with
//! itself as many times as its predicate holds, and the predicate is read by
//! propagating constants through the region. Both are finite acyclic
//! compositions of operations the region already contains, so unrolling them
//! introduces nothing new to lower.
//!
//! Exchanging only adjacent positions is what makes one network serve both
//! sorts: a gate swaps when the later element strictly precedes the earlier one,
//! so equal elements never pass each other and the result is stable whether or
//! not stability was asked for.
//!
//! What is not decided is left alone. A predicate that constant propagation
//! cannot read is a loop with no count to unroll, and the `while` is rebuilt
//! unchanged; so is an unrolling that would exceed [`MAX_UNROLLED_NODES`].

use smallvec::SmallVec;

use crate::dtype::DType;
use crate::ir::{FuncId, Module, Node, ValueId};
use crate::op::Op;
use crate::ops::{BroadcastInDim, Concatenate, Constant, Select, Slice, Sort, While};
use crate::shape::{Dims, Shape};

use super::fold::fold_node;
use crate::optimize::rebuild::Rebuilder;

/// Nodes one unrolled operation may emit. A sort of `n` elements costs
/// `n(n-1)/2` gates, so the bound is what stops a long axis from unrolling into
/// a program no machine holds.
pub(in crate::optimize) const MAX_UNROLLED_NODES: usize = 1 << 20;

/// Iterations one loop may take before its count is treated as unread. A
/// predicate that never turns false states a computation that does not finish,
/// and there is no count to find.
pub(in crate::optimize) const MAX_LOOP_ITERATIONS: usize = 1 << 20;

pub(in crate::optimize) fn unroll_static_regions(
    module: &mut Module,
    max_literal_bytes: usize,
) -> Result<bool, String> {
    let source: &Module = module;
    let mut changed = false;
    let funcs = source
        .funcs
        .iter()
        .map(|old| {
            let mut new = Rebuilder::new(old);
            let mut unroller = Unroller {
                module: source,
                max_literal_bytes,
                active: Vec::new(),
            };
            for node in &old.nodes {
                let inputs = new.mapped_inputs(node)?;
                let results = node
                    .results
                    .iter()
                    .map(|&result| old.value_shape(result).clone())
                    .collect::<Vec<_>>();
                match unroller.unroll(&mut new, node.op.as_ref(), &inputs, &results)? {
                    Some(values) => {
                        new.alias(&node.results, &values)?;
                        changed = true;
                    }
                    None => {
                        new.emit(old, &node.results, node.op.clone(), inputs)?;
                    }
                }
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}

/// How a region's own shapes appear once its operations are emitted in the
/// caller. A loop's regions take the carried values themselves, so their shapes
/// are unchanged; a comparator is written over scalars and is applied to a whole
/// hyperplane at once, so each of its shapes gains that hyperplane's dimensions.
type ShapeLift<'a> = &'a dyn Fn(&Shape) -> Shape;

struct Unroller<'a> {
    module: &'a Module,
    max_literal_bytes: usize,
    active: Vec<FuncId>,
}

impl Unroller<'_> {
    /// The circuit `op` denotes, or `None` when it denotes none or its extent
    /// cannot be read.
    fn unroll(
        &mut self,
        out: &mut Rebuilder,
        op: &dyn Op,
        inputs: &[ValueId],
        results: &[Shape],
    ) -> Result<Option<Vec<ValueId>>, String> {
        if let Some(sort) = op.as_any().downcast_ref::<Sort>() {
            return self.unroll_sort(out, sort, inputs, results);
        }
        if let Some(repeat) = op.as_any().downcast_ref::<While>() {
            return self.unroll_while(out, repeat, inputs, results);
        }
        Ok(None)
    }

    fn unroll_sort(
        &mut self,
        out: &mut Rebuilder,
        sort: &Sort,
        inputs: &[ValueId],
        results: &[Shape],
    ) -> Result<Option<Vec<ValueId>>, String> {
        if inputs.is_empty() || inputs.len() != results.len() {
            return Ok(None);
        }
        if self.active.contains(&sort.comparator) {
            return Ok(None);
        }
        let axis = sort.dimension as usize;
        if axis >= results[0].rank() {
            return Ok(None);
        }
        let Ok(length) = usize::try_from(results[0].dims[axis]) else {
            return Ok(None);
        };
        let comparator = self.module.func(sort.comparator);
        // The comparator is written over scalars, and every one of its shapes is
        // lifted to a hyperplane by dimension alone. A region holding a shape
        // that is not scalar has no such lift.
        if comparator
            .values
            .iter()
            .any(|value| !value.shape.is_scalar())
            || comparator.results.len() != 1
            || comparator.params.len() != 2 * inputs.len()
        {
            return Ok(None);
        }
        // One round per position, each round exchanging every other adjacent
        // pair: n(n-1)/2 gates in total.
        let gates = length.saturating_mul(length.saturating_sub(1)) / 2;
        let per_gate = comparator.nodes.len() + 2 * inputs.len();
        let emitted = (length * inputs.len())
            .saturating_add(gates.saturating_mul(per_gate))
            .saturating_add(inputs.len());
        if emitted > MAX_UNROLLED_NODES {
            return Ok(None);
        }
        if length <= 1 {
            return Ok(Some(inputs.to_vec()));
        }

        let mut plane_dims = results[0].dims.clone();
        plane_dims[axis] = 1;
        let plane = |dtype: DType| {
            results[0]
                .with_dims(plane_dims.iter().copied())
                .with_dtype(dtype)
        };

        let mut lanes = Vec::with_capacity(inputs.len());
        for (operand, &input) in inputs.iter().enumerate() {
            let shape = plane(results[operand].dtype);
            lanes.push(
                (0..length)
                    .map(|position| {
                        let op = hyperplane(&results[operand].dims, axis, position);
                        out.emit_fresh(shape.clone(), op, [input])
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let lift = |shape: &Shape| plane(shape.dtype);
        for round in 0..length {
            let mut position = round % 2;
            while position + 1 < length {
                // A gate asks whether the later element must precede the
                // earlier one, which is the only question whose answer may move
                // them.
                let arguments = (0..inputs.len())
                    .flat_map(|operand| [lanes[operand][position + 1], lanes[operand][position]])
                    .collect::<Vec<_>>();
                let swap = self.inline(out, sort.comparator, &arguments, &lift)?;
                let [swap] = swap.as_slice() else {
                    return Err("sort comparator does not return one value".into());
                };
                for operand in 0..inputs.len() {
                    let shape = plane(results[operand].dtype);
                    let (earlier, later) = (lanes[operand][position], lanes[operand][position + 1]);
                    lanes[operand][position] =
                        out.emit_fresh(shape.clone(), Select, [*swap, later, earlier]);
                    lanes[operand][position + 1] =
                        out.emit_fresh(shape, Select, [*swap, earlier, later]);
                }
                position += 2;
            }
        }

        Ok(Some(
            (0..inputs.len())
                .map(|operand| {
                    out.emit_fresh(
                        results[operand].clone(),
                        Concatenate {
                            dimension: sort.dimension,
                        },
                        lanes[operand].iter().copied(),
                    )
                })
                .collect(),
        ))
    }

    fn unroll_while(
        &mut self,
        out: &mut Rebuilder,
        repeat: &While,
        inputs: &[ValueId],
        results: &[Shape],
    ) -> Result<Option<Vec<ValueId>>, String> {
        if inputs.len() != results.len() {
            return Ok(None);
        }
        if self.active.contains(&repeat.cond()) || self.active.contains(&repeat.body()) {
            return Ok(None);
        }
        let same = |shape: &Shape| shape.clone();
        let mark = out.mark();
        let mut carried = inputs.to_vec();
        for _ in 0..MAX_LOOP_ITERATIONS {
            let predicate = self.inline(out, repeat.cond(), &carried, &same)?;
            let [predicate] = predicate.as_slice() else {
                out.rewind(mark);
                return Ok(None);
            };
            let Some(held) = out.literal(*predicate) else {
                out.rewind(mark);
                return Ok(None);
            };
            if held.shape.dtype != DType::Bool || !held.shape.is_scalar() || held.data.len() != 1 {
                out.rewind(mark);
                return Ok(None);
            }
            if held.data[0] == 0 {
                return Ok(Some(carried));
            }
            if out.nodes_since(mark) > MAX_UNROLLED_NODES {
                out.rewind(mark);
                return Ok(None);
            }
            carried = self.inline(out, repeat.body(), &carried, &same)?;
        }
        out.rewind(mark);
        Ok(None)
    }

    /// Emit `region`'s operations in the caller, reading `arguments` for its
    /// parameters, and answer with its results.
    fn inline(
        &mut self,
        out: &mut Rebuilder,
        region: FuncId,
        arguments: &[ValueId],
        lift: ShapeLift<'_>,
    ) -> Result<Vec<ValueId>, String> {
        let func = self.module.func(region);
        if func.params.len() != arguments.len() {
            return Err(format!(
                "region `{}` takes {} parameters, applied to {}",
                func.name,
                func.params.len(),
                arguments.len()
            ));
        }
        let mut values = vec![None; func.values.len()];
        for (&param, &argument) in func.params.iter().zip(arguments) {
            values[param.index()] = Some(argument);
        }
        self.active.push(region);
        for node in &func.nodes {
            let inputs = node
                .inputs
                .iter()
                .map(|&input| {
                    values[input.index()].ok_or_else(|| {
                        format!("region `{}` reads unbuilt %{}", func.name, input.index())
                    })
                })
                .collect::<Result<SmallVec<[ValueId; 4]>, String>>()?;
            let results = node
                .results
                .iter()
                .map(|&result| lift(func.value_shape(result)))
                .collect::<Vec<_>>();
            let emitted = match self.unroll(out, node.op.as_ref(), &inputs, &results)? {
                Some(values) => values,
                None => self.emit(out, node, &inputs, &results)?,
            };
            for (&result, value) in node.results.iter().zip(emitted) {
                values[result.index()] = Some(value);
            }
        }
        self.active.pop();
        func.results
            .iter()
            .map(|&result| {
                values[result.index()].ok_or_else(|| {
                    format!("region `{}` returns unbuilt %{}", func.name, result.index())
                })
            })
            .collect()
    }

    /// One region operation in the caller, at the lifted shapes.
    fn emit(
        &self,
        out: &mut Rebuilder,
        node: &Node,
        inputs: &[ValueId],
        results: &[Shape],
    ) -> Result<Vec<ValueId>, String> {
        // A constant holds its own elements, so it is emitted at the shape those
        // elements fill and then spread over the lifted one.
        if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
            let held = constant.literal.shape.clone();
            let value = out.emit_fresh(held.clone(), constant.clone(), []);
            if held == results[0] {
                return Ok(vec![value]);
            }
            if !held.is_scalar() {
                return Err(format!(
                    "region constant of shape {:?} does not spread to {:?}",
                    held.dims, results[0].dims
                ));
            }
            return Ok(vec![out.emit_fresh(
                results[0].clone(),
                BroadcastInDim {
                    out_dims: results[0].dims.clone(),
                    broadcast_dimensions: SmallVec::new(),
                },
                [value],
            )]);
        }
        // The values a region computes from constants alone are constants, and
        // reading a loop's predicate is exactly that.
        if let [result] = results
            && inputs.iter().any(|&input| out.literal(input).is_some())
            && let Some(literal) = fold_node(
                out,
                node.op.as_ref(),
                inputs,
                result,
                self.max_literal_bytes,
            )
        {
            return Ok(vec![out.emit_fresh(
                result.clone(),
                Constant { literal },
                [],
            )]);
        }
        Ok(out
            .emit_fresh_results(results, node.op.clone(), inputs.iter().copied())
            .into_vec())
    }
}

/// The slice taking `position` along `axis`, keeping the axis at length one so
/// the pieces concatenate back into the whole.
fn hyperplane(dims: &Dims, axis: usize, position: usize) -> Slice {
    let position = position as i64;
    Slice {
        start_indices: dims
            .iter()
            .enumerate()
            .map(|(index, _)| if index == axis { position } else { 0 })
            .collect(),
        limit_indices: dims
            .iter()
            .enumerate()
            .map(|(index, &dim)| if index == axis { position + 1 } else { dim })
            .collect(),
        strides: dims.iter().map(|_| 1).collect(),
    }
}
