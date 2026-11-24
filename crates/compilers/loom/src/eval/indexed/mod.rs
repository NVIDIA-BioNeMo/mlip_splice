// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use gpu_ir::{Builder, Evict, Scalar, Value};
use xir::dtype::DType;
use xir::ir::FuncId;
use xir::ops::{Gather, ScalarOp};

mod emit;
mod schedule;

const STATIC_REDUCTION_TERMS_PER_BINDING: usize = 8;

/// Terms a reduction may name individually before it is walked with a loop.
///
/// Unrolling a reduction writes one statement per term, so a recovered
/// map/reduce region would put its whole term list back into the program text —
/// exactly the size the recovery existed to remove. Below this many terms the
/// unrolled form is worth its space: it exposes each term to common
/// subexpression elimination and to fused multiply-add. Above it, the loop reads
/// its terms through the index tables instead and the program stops growing with
/// the term count.
const MAX_UNROLLED_REDUCTION_TERMS: usize = 16;

fn unrolls(terms: usize) -> bool {
    terms <= MAX_UNROLLED_REDUCTION_TERMS
}

#[derive(Clone)]
pub(crate) struct IndexedTensor {
    pub elements: usize,
    expression: Arc<IndexedExpression>,
}

#[derive(Clone)]
struct IndexTable {
    name: String,
    data: Arc<[u8]>,
    values: Arc<[usize]>,
}

impl IndexTable {
    fn new(name: String, values: &[usize]) -> Result<Self, String> {
        let data = values
            .iter()
            .map(|&value| {
                i32::try_from(value)
                    .map(i32::to_le_bytes)
                    .map_err(|_| format!("indexed table value {value} does not fit i32"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into();
        Ok(Self {
            name,
            data,
            values: values.into(),
        })
    }

    fn load(&self, b: &mut Builder, index: Value) -> Value {
        if let Some(value) = b.as_int_const(&index)
            && let Ok(index) = usize::try_from(value)
            && let Some(&value) = self.values.get(index)
        {
            return b.i32(value as i64);
        }
        b.const_load_shared(&self.name, Scalar::S32, self.data.clone(), index)
    }
}

#[derive(Clone)]
enum IndexedExpression {
    Dense(Vec<Value>),
    Pointer {
        base: Value,
        scalar: Scalar,
        stride: i64,
        load_lanes: u8,
        splat_lanes: u8,
        bool_storage: bool,
        alive: Option<Value>,
        evict: Evict,
    },
    Constant {
        name: String,
        scalar: Scalar,
        data: Arc<[u8]>,
        immediate: Vec<Value>,
    },
    Reindex {
        input: IndexedTensor,
        indices: IndexTable,
    },
    Concatenate {
        inputs: Vec<IndexedTensor>,
        offsets: Vec<usize>,
    },
    Gather {
        input: IndexedTensor,
        starts: IndexedTensor,
        operand_dims: Vec<i64>,
        start_dims: Vec<i64>,
        output_dims: Vec<i64>,
        op: Gather,
    },
    Elementwise {
        op: ScalarOp,
        dtype: DType,
        inputs: Vec<IndexedTensor>,
    },
    Convert {
        from: DType,
        to: DType,
        input: IndexedTensor,
    },
    Pad {
        input: IndexedTensor,
        pad: IndexedTensor,
        input_indices: IndexTable,
        inside: IndexTable,
    },
    Reduce {
        input: IndexedTensor,
        init: IndexedTensor,
        init_per_output: bool,
        reducer: FuncId,
        input_indices: IndexTable,
        offsets: IndexTable,
        uniform_terms: Option<usize>,
    },
}

/// Dismantles a value without recursing once per level.
///
/// A tensor owns the tensors it reads, so the drop glue the compiler writes
/// descends the whole DAG on the machine stack — and these DAGs are as deep as
/// the arithmetic they came from, deep enough to abort the process on a program
/// that is merely long. Taking the expression out and dropping what it held from
/// a list costs one enum write per value and no frames at all.
impl Drop for IndexedTensor {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        detach(&mut self.expression, &mut pending);
        while let Some(mut tensor) = pending.pop() {
            detach(&mut tensor.expression, &mut pending);
        }
    }
}

/// Moves the tensors an expression owns onto `pending`, leaving it empty.
/// A shared expression is left alone: another owner still reads it.
fn detach(expression: &mut Arc<IndexedExpression>, pending: &mut Vec<IndexedTensor>) {
    let Some(owner) = Arc::get_mut(expression) else {
        return;
    };
    match std::mem::replace(owner, IndexedExpression::Dense(Vec::new())) {
        IndexedExpression::Reindex { input, .. } | IndexedExpression::Convert { input, .. } => {
            pending.push(input);
        }
        IndexedExpression::Concatenate { inputs, .. }
        | IndexedExpression::Elementwise { inputs, .. } => pending.extend(inputs),
        IndexedExpression::Gather { input, starts, .. } => {
            pending.push(input);
            pending.push(starts);
        }
        IndexedExpression::Pad { input, pad, .. } => {
            pending.push(input);
            pending.push(pad);
        }
        IndexedExpression::Reduce { input, init, .. } => {
            pending.push(input);
            pending.push(init);
        }
        IndexedExpression::Dense(_)
        | IndexedExpression::Pointer { .. }
        | IndexedExpression::Constant { .. } => {}
    }
}

impl IndexedTensor {
    fn element_key(&self, element: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.expression) as usize, element)
    }

    pub(super) fn dense(elements: Vec<Value>) -> Self {
        Self {
            elements: elements.len(),
            expression: Arc::new(IndexedExpression::Dense(elements)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pointer(
        base: Value,
        scalar: Scalar,
        stride: i64,
        load_lanes: u8,
        splat_lanes: u8,
        bool_storage: bool,
        alive: Option<Value>,
        evict: Evict,
        elements: usize,
    ) -> Self {
        Self {
            elements,
            expression: Arc::new(IndexedExpression::Pointer {
                base,
                scalar,
                stride,
                load_lanes,
                splat_lanes,
                bool_storage,
                alive,
                evict,
            }),
        }
    }

    pub(super) fn constant(
        name: String,
        scalar: Scalar,
        data: Vec<u8>,
        elements: usize,
        immediate: Vec<Value>,
    ) -> Self {
        let expression = if elements <= 8 {
            IndexedExpression::Dense(immediate)
        } else {
            IndexedExpression::Constant {
                name,
                scalar,
                data: data.into(),
                immediate,
            }
        };
        Self {
            elements,
            expression: Arc::new(expression),
        }
    }

    pub(super) fn reindex(&self, name: String, indices: Vec<usize>) -> Result<Self, String> {
        Ok(Self {
            elements: indices.len(),
            expression: Arc::new(IndexedExpression::Reindex {
                input: self.clone(),
                indices: IndexTable::new(name, &indices)?,
            }),
        })
    }

    pub(super) fn concatenate(inputs: Vec<IndexedTensor>) -> Result<Self, String> {
        if inputs.is_empty() {
            return Err("indexed concatenate has no inputs".into());
        }
        let mut flattened = Vec::new();
        for input in inputs {
            match input.expression.as_ref() {
                IndexedExpression::Concatenate { inputs, .. } => {
                    flattened.extend(inputs.iter().cloned());
                }
                _ => flattened.push(input),
            }
        }
        let mut offsets = Vec::with_capacity(flattened.len() + 1);
        offsets.push(0usize);
        for input in &flattened {
            offsets.push(
                offsets
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(input.elements)
                    .ok_or_else(|| "indexed concatenate size overflows usize".to_string())?,
            );
        }
        Ok(Self {
            elements: *offsets.last().unwrap(),
            expression: Arc::new(IndexedExpression::Concatenate {
                inputs: flattened,
                offsets,
            }),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather(
        input: IndexedTensor,
        starts: IndexedTensor,
        operand_dims: Vec<i64>,
        start_dims: Vec<i64>,
        output_dims: Vec<i64>,
        op: Gather,
        elements: usize,
    ) -> Self {
        Self {
            elements,
            expression: Arc::new(IndexedExpression::Gather {
                input,
                starts,
                operand_dims,
                start_dims,
                output_dims,
                op,
            }),
        }
    }

    pub(super) fn elementwise(
        op: ScalarOp,
        dtype: DType,
        inputs: Vec<IndexedTensor>,
        elements: usize,
    ) -> Self {
        Self {
            elements,
            expression: Arc::new(IndexedExpression::Elementwise { op, dtype, inputs }),
        }
    }

    pub(super) fn convert(from: DType, to: DType, input: IndexedTensor) -> Self {
        Self {
            elements: input.elements,
            expression: Arc::new(IndexedExpression::Convert { from, to, input }),
        }
    }

    pub(super) fn pad(
        name: String,
        input: IndexedTensor,
        pad: IndexedTensor,
        input_indices: Vec<usize>,
        inside: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            elements: input_indices.len(),
            expression: Arc::new(IndexedExpression::Pad {
                input,
                pad,
                input_indices: IndexTable::new(name.clone(), &input_indices)?,
                inside: IndexTable::new(format!("{name}_inside"), &inside)?,
            }),
        })
    }

    pub(super) fn reduce(
        name: String,
        input: IndexedTensor,
        init: IndexedTensor,
        reducer: FuncId,
        cells: &[usize],
        outputs: usize,
    ) -> Result<Self, String> {
        if cells.len() != input.elements {
            return Err(format!(
                "reduction maps {} cells from {} input elements",
                cells.len(),
                input.elements
            ));
        }
        let mut terms = vec![Vec::new(); outputs];
        for (input, &output) in cells.iter().enumerate() {
            if output >= outputs {
                return Err(format!("reduction output index {output} is out of range"));
            }
            terms[output].push(input);
        }
        Self::reduction(name, input, init, false, reducer, terms)
    }

    pub(super) fn scatter(
        name: String,
        operand: IndexedTensor,
        updates: IndexedTensor,
        reducer: FuncId,
        destinations: &[Option<usize>],
    ) -> Result<Self, String> {
        if destinations.len() != updates.elements {
            return Err(format!(
                "scatter maps {} destinations from {} updates",
                destinations.len(),
                updates.elements
            ));
        }
        let mut terms = vec![Vec::new(); operand.elements];
        for (update, &destination) in destinations.iter().enumerate() {
            if let Some(destination) = destination {
                let output = terms
                    .get_mut(destination)
                    .ok_or_else(|| format!("scatter destination {destination} is out of range"))?;
                output.push(update);
            }
        }
        Self::reduction(name, updates, operand, true, reducer, terms)
    }

    fn reduction(
        name: String,
        input: IndexedTensor,
        init: IndexedTensor,
        init_per_output: bool,
        reducer: FuncId,
        terms: Vec<Vec<usize>>,
    ) -> Result<Self, String> {
        let outputs = terms.len();
        let expected_init_elements = if init_per_output { outputs } else { 1 };
        if init.elements != expected_init_elements {
            return Err(format!(
                "indexed reduction has {} init elements for {outputs} outputs",
                init.elements
            ));
        }
        let uniform_terms = terms
            .first()
            .map(Vec::len)
            .filter(|&count| terms.iter().all(|terms| terms.len() == count));
        let mut offsets = Vec::with_capacity(outputs + 1);
        let mut input_indices = Vec::new();
        offsets.push(0);
        for terms in terms {
            input_indices.extend(terms);
            offsets.push(input_indices.len());
        }
        Ok(Self {
            elements: outputs,
            expression: Arc::new(IndexedExpression::Reduce {
                input,
                init,
                init_per_output,
                reducer,
                input_indices: IndexTable::new(name.clone(), &input_indices)?,
                offsets: IndexTable::new(format!("{name}_offsets"), &offsets)?,
                uniform_terms,
            }),
        })
    }
}

fn coordinate(b: &mut Builder, flat: Value, dims: &[i64], axis: usize) -> Value {
    let stride = dims[axis + 1..].iter().product::<i64>();
    let mut value = flat;
    if stride != 1 {
        let stride = b.i32(stride);
        value = b.div(value, stride);
    }
    if axis != 0 {
        let extent = b.i32(dims[axis]);
        value = b.rem(value, extent);
    }
    value
}

fn ravel_index(b: &mut Builder, coordinates: &[Value], dims: &[i64]) -> Value {
    let mut flat = b.i32(0);
    for (coordinate, &dim) in coordinates.iter().zip(dims) {
        let dim = b.i32(dim);
        let scaled = b.mul(flat, dim);
        flat = b.add(scaled, coordinate.clone());
    }
    flat
}

#[cfg(test)]
mod tests;
