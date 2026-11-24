// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use gpu_ir::{BinOp, Builder, CmpOp, Scalar, Ty};
use xir::analysis::FuncFacts;
use xir::dtype::DType;
use xir::element_mapping::{
    broadcast_elements, concatenate_elements, pad_elements, slice_elements, transpose_elements,
};
use xir::ir::{Func, FuncId, Module, ValueId};
use xir::ops::{
    BroadcastInDim, Concatenate, Constant, Convert, DotGeneral, Gather, Pad, Reduce, Reshape,
    ScalarOp, Scatter, Slice, Transpose,
};

mod constant;
mod indexed;
mod ops;
mod scalar;
mod shape;

use constant::{Const, cast_const, constant_value, convert, eval_const};
pub(super) use indexed::IndexedTensor;
pub(super) use scalar::{math_scalar, zero};

/// Named indexed values a thread may hold at once, or zero to hold every shared
/// value.
///
/// This is not the register file. A thread on this target has 255 registers and
/// the kernels here use all of them; the plan allocates none of them and cannot
/// know what ptxas will do with its bindings. What this bounds is how many
/// values the plan keeps live before it starts rebuilding the cheapest of them,
/// which is a proxy for pressure rather than a count of registers.
const INDEXED_REGISTER_BUDGET: usize = 96;

pub(super) fn scalar_constant(
    b: &mut Builder,
    literal: &xir::ops::Literal,
    lanes: u8,
) -> Result<gpu_ir::Value, String> {
    let value = constant_value(b, literal)?;
    let [element] = value.elems.as_slice() else {
        return Err("output factor constant must be scalar".into());
    };
    let element = element.clone();
    Ok(b.splat(lanes, element))
}

pub(super) fn convert_output_factor(
    b: &mut Builder,
    value: gpu_ir::Value,
    from: DType,
    to: DType,
    lanes: u8,
) -> Result<gpu_ir::Value, String> {
    convert(b, value, from, to, lanes)
}

pub(super) fn output_factor_elementwise(
    b: &mut Builder,
    op: ScalarOp,
    dtype: DType,
    inputs: Vec<gpu_ir::Value>,
    lanes: u8,
) -> Result<gpu_ir::Value, String> {
    let scalar = math_scalar(dtype)?;
    let mut inputs = inputs.into_iter().map(Some).collect::<Vec<_>>();
    fn input(
        inputs: &mut [Option<gpu_ir::Value>],
        op: ScalarOp,
        index: usize,
    ) -> Result<gpu_ir::Value, String> {
        inputs
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| format!("output factor `{op:?}` has no operand {index}"))
    }
    let pair = |inputs: &mut [Option<gpu_ir::Value>]| {
        Ok::<_, String>((input(inputs, op, 0)?, input(inputs, op, 1)?))
    };
    Ok(match op {
        ScalarOp::Negate => {
            let value = input(&mut inputs, op, 0)?;
            b.neg(value)
        }
        ScalarOp::Abs => {
            let value = input(&mut inputs, op, 0)?;
            let negated = b.neg(value.clone());
            b.max(value, negated)
        }
        ScalarOp::Sign => {
            let value = input(&mut inputs, op, 0)?;
            let zero = scalar::zero(b, scalar);
            let zero = b.splat(lanes, zero);
            let one = scalar::one(b, scalar);
            let one = b.splat(lanes, one);
            let below = b.cmp(CmpOp::Lt, value.clone(), zero.clone());
            let above = b.cmp(CmpOp::Gt, value.clone(), zero);
            let positive = b.select(above, one.clone(), value);
            let negative = b.neg(one);
            b.select(below, negative, positive)
        }
        ScalarOp::Sqrt => {
            let value = input(&mut inputs, op, 0)?;
            b.sqrt(value)
        }
        ScalarOp::Rsqrt => {
            let one = scalar::one(b, scalar);
            let one = b.splat(lanes, one);
            let value = input(&mut inputs, op, 0)?;
            let root = b.sqrt(value);
            b.div(one, root)
        }
        ScalarOp::Not => {
            let value = input(&mut inputs, op, 0)?;
            b.not(value)
        }
        ScalarOp::Add => {
            let (left, right) = pair(&mut inputs)?;
            b.add(left, right)
        }
        ScalarOp::Subtract => {
            let (left, right) = pair(&mut inputs)?;
            b.sub(left, right)
        }
        ScalarOp::Multiply => {
            let (left, right) = pair(&mut inputs)?;
            b.mul(left, right)
        }
        ScalarOp::Divide => {
            let (left, right) = pair(&mut inputs)?;
            b.div(left, right)
        }
        ScalarOp::Maximum => {
            let (left, right) = pair(&mut inputs)?;
            b.max(left, right)
        }
        ScalarOp::Minimum => {
            let (left, right) = pair(&mut inputs)?;
            b.min(left, right)
        }
        ScalarOp::Remainder => {
            let (left, right) = pair(&mut inputs)?;
            b.rem(left, right)
        }
        ScalarOp::And => {
            let (left, right) = pair(&mut inputs)?;
            b.bin(BinOp::And, left, right)
        }
        ScalarOp::Or => {
            let (left, right) = pair(&mut inputs)?;
            b.bin(BinOp::Or, left, right)
        }
        ScalarOp::Xor => {
            let (left, right) = pair(&mut inputs)?;
            b.bin(BinOp::Xor, left, right)
        }
        ScalarOp::ShiftLeft => {
            let (left, right) = pair(&mut inputs)?;
            let amount = b.cast(Scalar::U32, right);
            b.bin(BinOp::Shl, left, amount)
        }
        ScalarOp::ShiftRightArithmetic => {
            let (left, right) = pair(&mut inputs)?;
            let signed = b.cast(scalar::signed_integer(scalar)?, left);
            let amount = b.cast(Scalar::U32, right);
            let shifted = b.bin(BinOp::Shr, signed, amount);
            b.cast(scalar, shifted)
        }
        ScalarOp::ShiftRightLogical => {
            let (left, right) = pair(&mut inputs)?;
            let unsigned = b.cast(scalar::unsigned_integer(scalar)?, left);
            let amount = b.cast(Scalar::U32, right);
            let shifted = b.bin(BinOp::Shr, unsigned, amount);
            b.cast(scalar, shifted)
        }
        ScalarOp::Compare(direction) => {
            let (left, right) = pair(&mut inputs)?;
            b.cmp(scalar::comparison(direction), left, right)
        }
        ScalarOp::Select => {
            let predicate = input(&mut inputs, op, 0)?;
            let on_true = input(&mut inputs, op, 1)?;
            let on_false = input(&mut inputs, op, 2)?;
            b.select(predicate, on_true, on_false)
        }
        other => return Err(format!("unsupported output factor scalar op {other:?}")),
    })
}

/// The scalar type an output is computed in — its math type, which the
/// operand's storage dtype may narrow.
pub(super) fn output_scalar(
    loom: &super::LiftedProgram,
    output: usize,
) -> Result<gpu_ir::Scalar, String> {
    let func = loom.module.entry_func();
    math_scalar(func.value_shape(func.results[output]).dtype)
}
use shape::shapes;

fn lower_scalar_region(
    b: &mut Builder,
    module: &Module,
    function: FuncId,
    params: Vec<gpu_ir::Value>,
    lanes: u8,
) -> Result<gpu_ir::Value, String> {
    let func = module.func(function);
    if func.params.len() != params.len() || func.results.len() != 1 {
        return Err("scalar region arity mismatch".into());
    }
    if let [node] = func.nodes.as_slice()
        && node.inputs.as_slice() == func.params
        && node.results.as_slice() == func.results
        && let Some(elementwise) = node.op.as_elementwise()
    {
        return output_factor_elementwise(
            b,
            elementwise.scalar(),
            func.value_shape(node.results[0]).dtype,
            params,
            lanes,
        );
    }
    let mut values = func
        .params
        .iter()
        .copied()
        .zip(params)
        .collect::<HashMap<_, _>>();
    for node in &func.nodes {
        if node.results.len() != 1 {
            return Err(format!(
                "scalar region `{}` must produce one result",
                node.op.mnemonic()
            ));
        }
        let result = node.results[0];
        let value = if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
            scalar_constant(b, &constant.literal, lanes)?
        } else if node.op.as_any().downcast_ref::<Convert>().is_some() {
            let input = values
                .get(&node.inputs[0])
                .cloned()
                .ok_or_else(|| "scalar region convert input is unavailable".to_string())?;
            convert(
                b,
                input,
                func.value_shape(node.inputs[0]).dtype,
                func.value_shape(result).dtype,
                lanes,
            )?
        } else if let Some(elementwise) = node.op.as_elementwise() {
            let inputs = node
                .inputs
                .iter()
                .map(|input| {
                    values.get(input).cloned().ok_or_else(|| {
                        format!("scalar region input %{} is unavailable", input.index())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            output_factor_elementwise(
                b,
                elementwise.scalar(),
                func.value_shape(result).dtype,
                inputs,
                lanes,
            )?
        } else {
            return Err(format!(
                "unsupported scalar region op `{}`",
                node.op.mnemonic()
            ));
        };
        values.insert(result, value);
    }
    values
        .get(&func.results[0])
        .cloned()
        .ok_or_else(|| "scalar region result is unavailable".into())
}

fn scalar_region_is_add(module: &Module, function: FuncId) -> bool {
    let func = module.func(function);
    func.params.len() == 2
        && func.results.len() == 1
        && func.nodes.len() == 1
        && func.nodes[0].op.as_elementwise().map(|op| op.scalar()) == Some(ScalarOp::Add)
        && func.nodes[0].inputs.as_slice() == func.params
        && func.nodes[0].results.as_slice() == func.results
}

#[derive(Clone)]
pub(super) struct Value {
    elems: Vec<gpu_ir::Value>,
    constants: Option<Vec<Const>>,
    indexed: Option<IndexedTensor>,
    reductions: Vec<Option<ReductionKind>>,
    guards: Vec<Option<GuardProvenance>>,
}

/// A shape op reindexes a value's elements; the provenance travelling with
/// them — constants, reduction terms, guards — is reindexed identically, so
/// the op is applied once and the value carries it through every component.
enum Reindex<'a> {
    Broadcast(&'a BroadcastInDim),
    Transpose(&'a Transpose),
    Slice(&'a Slice),
}

impl Reindex<'_> {
    fn of(node: &xir::ir::Node) -> Option<Reindex<'_>> {
        let op = node.op.as_any();
        op.downcast_ref::<BroadcastInDim>()
            .map(Reindex::Broadcast)
            .or_else(|| op.downcast_ref::<Transpose>().map(Reindex::Transpose))
            .or_else(|| op.downcast_ref::<Slice>().map(Reindex::Slice))
    }

    fn apply<T: Clone>(&self, values: &[T], dims: &[i64]) -> Result<Vec<T>, String> {
        match self {
            Reindex::Broadcast(op) => {
                broadcast_elements(values, dims, &op.out_dims, &op.broadcast_dimensions)
            }
            Reindex::Transpose(op) => transpose_elements(values, dims, &op.permutation),
            Reindex::Slice(op) => slice_elements(values, dims, op),
        }
    }
}

impl Value {
    fn reindexed(self, by: &Reindex<'_>, dims: &[i64], name: String) -> Result<Self, String> {
        let indices = by.apply(&(0..self.elems.len()).collect::<Vec<_>>(), dims)?;
        Ok(Value {
            elems: by.apply(&self.elems, dims)?,
            indexed: self
                .indexed
                .map(|value| value.reindex(name, indices))
                .transpose()?,
            constants: self
                .constants
                .as_deref()
                .map(|values| by.apply(values, dims))
                .transpose()?,
            reductions: by.apply(&self.reductions, dims)?,
            guards: by.apply(&self.guards, dims)?,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct GuardProvenance {
    pub predicate: gpu_ir::ValueId,
    pub key: GuardKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GuardKey {
    pub value: ValueId,
    pub element: usize,
    pub inverted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReductionKind {
    Reduce,
    Dot,
}

/// One evaluation of a Loom program: the arena it built, the outputs' values in
/// it, and the provenance travelling with them.
///
/// Values belong to the arena that defined them, so every consumer of one of
/// these reads its `body` and nothing else — a value of one evaluation cannot be
/// read against another.
pub(super) struct Evaluation {
    /// The value each input element is read through, per input and element.
    pub inputs: Vec<Vec<gpu_ir::ValueId>>,
    pub outputs: Vec<Vec<gpu_ir::ValueId>>,
    pub body: gpu_ir::Body,
    pub reductions: Vec<Vec<Option<ReductionKind>>>,
    pub guards: Vec<Vec<Option<GuardProvenance>>>,
    /// Per binding this evaluation guarded, the guard it was built under.
    pub statement_guards: BTreeMap<gpu_ir::ValueId, GuardProvenance>,
}

pub(super) struct Evaluator<'a> {
    module: &'a Module,
    func: &'a Func,
    facts: FuncFacts,
    values: HashMap<ValueId, Value>,
    inputs: Vec<Vec<gpu_ir::ValueId>>,
    pub stmts: gpu_ir::Builder,
    statement_guards: BTreeMap<gpu_ir::ValueId, GuardProvenance>,
    lanes: u8,
    indexed_mode: bool,
    indexed_names: usize,
    indexed_cache: HashMap<(usize, usize), gpu_ir::Value>,
    /// Values the plan chose to name; every other value is rebuilt at each read.
    indexed_named: std::collections::HashSet<(usize, usize)>,
    /// Values that may be live at once before the plan starts giving some back.
    indexed_register_budget: usize,
}

impl<'a> Evaluator<'a> {
    /// A fresh evaluator over `module`, with nothing bound yet.
    ///
    /// The arena it writes is its own, so what the program reads has to be
    /// built here: a caller declares the reads through [`Evaluator::builder`]
    /// and hands the elements back to [`Evaluator::bind_params`]. A value of
    /// another arena cannot stand in, which is what keeps one evaluation's
    /// values out of another's.
    pub(crate) fn new(module: &'a Module, lanes: u8) -> Evaluator<'a> {
        let func = module.entry_func();
        Evaluator {
            module,
            func,
            facts: FuncFacts::analyze(func),
            values: HashMap::new(),
            inputs: Vec::new(),
            stmts: gpu_ir::Builder::new(),
            statement_guards: BTreeMap::new(),
            lanes,
            indexed_mode: false,
            indexed_names: 0,
            indexed_cache: HashMap::new(),
            indexed_named: std::collections::HashSet::new(),
            indexed_register_budget: INDEXED_REGISTER_BUDGET / lanes.max(1) as usize,
        }
    }

    /// The body this evaluation writes, for a caller declaring what it reads.
    pub(crate) fn builder(&mut self) -> &mut gpu_ir::Builder {
        &mut self.stmts
    }

    /// Binds each parameter of the entry function to the elements a caller
    /// built in this evaluation's own arena.
    pub(crate) fn bind_params(
        &mut self,
        params: Vec<Vec<gpu_ir::Value>>,
        indexed: Option<Vec<IndexedTensor>>,
    ) -> Result<(), String> {
        if params.len() != self.func.params.len() {
            return Err(format!(
                "evaluator got {} parameters for {} function parameters",
                params.len(),
                self.func.params.len()
            ));
        }
        if indexed
            .as_ref()
            .is_some_and(|values| values.len() != params.len())
        {
            return Err("indexed evaluator parameter count mismatch".into());
        }
        self.indexed_mode = indexed.is_some();
        self.inputs = params
            .iter()
            .map(|elems| {
                elems
                    .iter()
                    .map(|elem| elem.id_in(self.stmts.function()))
                    .collect()
            })
            .collect();
        self.values = self
            .func
            .params
            .iter()
            .copied()
            .zip(params.into_iter().enumerate())
            .map(|(id, (parameter, elems))| {
                let indexed = indexed.as_ref().map(|values| values[parameter].clone());
                (
                    id,
                    Value {
                        reductions: vec![None; elems.len()],
                        guards: vec![None; elems.len()],
                        elems,
                        constants: None,
                        indexed,
                    },
                )
            })
            .collect();
        Ok(())
    }

    pub(crate) fn run_with_provenance(mut self) -> Result<Evaluation, String> {
        for (node_index, node) in self.func.nodes.iter().enumerate() {
            self.lower_node(node_index, node)
                .map_err(|e| format!("loom node {node_index} `{}`: {e}", node.op.mnemonic()))?;
        }
        // Use counts are taken over every output before any of them is emitted:
        // a value shared between two outputs is as worth naming as one shared
        // inside a single output, and a per-output count cannot see the first.
        let outputs = self
            .func
            .results
            .iter()
            .map(|&result| {
                self.value(result)
                    .map(|value| (result, value.indexed.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let roots = outputs
            .iter()
            .filter_map(|(_, indexed)| indexed.as_ref())
            .flat_map(|indexed| (0..indexed.elements).map(|element| (indexed.clone(), element)))
            .collect::<Vec<_>>();
        if !roots.is_empty() {
            self.prepare_indexed_plan(&roots)?;
        }
        for (result, indexed) in outputs {
            let Some(indexed) = indexed else {
                continue;
            };
            let elems = (0..indexed.elements)
                .map(|element| self.emit_indexed_at(&indexed, element))
                .collect::<Result<Vec<_>, _>>()?;
            self.values.get_mut(&result).unwrap().elems = elems;
        }
        let outputs = self
            .func
            .results
            .iter()
            .map(|id| {
                self.value(*id).map(|value| {
                    value
                        .elems
                        .iter()
                        .map(|element| element.id_in(self.stmts.function()))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reductions = self
            .func
            .results
            .iter()
            .map(|id| self.value(*id).map(|value| value.reductions.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let guards = self
            .func
            .results
            .iter()
            .map(|id| self.value(*id).map(|value| value.guards.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Evaluation {
            inputs: self.inputs,
            outputs,
            body: self.stmts.finish_owned(),
            reductions,
            guards,
            statement_guards: self.statement_guards,
        })
    }

    pub(super) fn value(&self, id: ValueId) -> Result<&Value, String> {
        self.values
            .get(&id)
            .ok_or_else(|| format!("value %{} has not been lowered", id.index()))
    }

    pub(super) fn placeholders(
        &mut self,
        dtype: DType,
        elements: usize,
    ) -> Result<Vec<gpu_ir::Value>, String> {
        let scalar = math_scalar(dtype)?;
        let value = zero(&mut self.stmts, scalar);
        let value = self.stmts.splat(self.lanes, value);
        Ok(vec![value; elements])
    }

    fn lower_node(&mut self, node_index: usize, node: &xir::ir::Node) -> Result<(), String> {
        let values = if let Some(constant) = node.op.as_any().downcast_ref::<Constant>() {
            let mut value = constant_value(&mut self.stmts, &constant.literal)?;
            if self.lanes > 1 {
                value.elems = value
                    .elems
                    .into_iter()
                    .map(|element| self.stmts.splat(self.lanes, element))
                    .collect();
            }
            if self.indexed_mode {
                value.indexed = Some(IndexedTensor::constant(
                    format!("loom_constant_{node_index}"),
                    math_scalar(constant.literal.shape.dtype)?,
                    constant.literal.data.clone(),
                    value.elems.len(),
                    value.elems.clone(),
                ));
            }
            vec![value]
        } else if node.op.as_any().downcast_ref::<Convert>().is_some() {
            let input = self.value(node.inputs[0])?.clone();
            let src = self.func.value_shape(node.inputs[0]).dtype;
            let dst = self.func.value_shape(node.results[0]).dtype;
            let elems = input
                .elems
                .into_iter()
                .map(|element| convert(&mut self.stmts, element, src, dst, self.lanes))
                .collect::<Result<Vec<_>, _>>()?;
            vec![Value {
                reductions: vec![None; elems.len()],
                guards: input.guards,
                elems,
                constants: input.constants.map(|values| {
                    values
                        .into_iter()
                        .map(|value| cast_const(value, dst))
                        .collect()
                }),
                indexed: input
                    .indexed
                    .map(|value| IndexedTensor::convert(src, dst, value)),
            }]
        } else if node.op.as_any().downcast_ref::<Reshape>().is_some() {
            vec![self.value(node.inputs[0])?.clone()]
        } else if let Some(reindex) = Reindex::of(node) {
            let dims = self.func.value_shape(node.inputs[0]).dims.clone();
            vec![self.value(node.inputs[0])?.clone().reindexed(
                &reindex,
                &dims,
                format!("loom_reindex_{node_index}"),
            )?]
        } else if let Some(op) = node.op.as_any().downcast_ref::<Concatenate>() {
            let mut values = node
                .inputs
                .iter()
                .map(|&input| self.value(input).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            let output_shape = self.func.value_shape(node.results[0]);
            if op.dimension == 0
                && output_shape.rank() == 1
                && node
                    .inputs
                    .iter()
                    .all(|&input| self.func.value_shape(input).rank() == 1)
            {
                let guards = values
                    .iter()
                    .flat_map(|value| value.guards.iter().cloned())
                    .collect::<Vec<_>>();
                let inputs = values
                    .into_iter()
                    .map(|value| {
                        value
                            .indexed
                            .unwrap_or_else(|| IndexedTensor::dense(value.elems))
                    })
                    .collect();
                let elements = num_elements(&output_shape.dims)?;
                vec![Value {
                    elems: self.placeholders(output_shape.dtype, elements)?,
                    constants: None,
                    indexed: Some(IndexedTensor::concatenate(inputs)?),
                    reductions: vec![None; elements],
                    guards,
                }]
            } else {
                for value in &mut values {
                    if let Some(indexed) = value.indexed.take() {
                        value.elems = (0..indexed.elements)
                            .map(|element| self.emit_indexed_at(&indexed, element))
                            .collect::<Result<Vec<_>, _>>()?;
                    }
                }
                let inputs = values
                    .iter()
                    .zip(&node.inputs)
                    .map(|(value, &id)| (value.elems.as_slice(), &self.func.value_shape(id).dims))
                    .collect::<Vec<_>>();
                vec![Value {
                    elems: concatenate_elements(&inputs, op.dimension as usize)?,
                    constants: None,
                    indexed: None,
                    reductions: concatenate_elements(
                        &values
                            .iter()
                            .zip(&node.inputs)
                            .map(|(value, &id)| {
                                (value.reductions.as_slice(), &self.func.value_shape(id).dims)
                            })
                            .collect::<Vec<_>>(),
                        op.dimension as usize,
                    )?,
                    guards: concatenate_elements(
                        &values
                            .iter()
                            .zip(&node.inputs)
                            .map(|(value, &id)| {
                                (value.guards.as_slice(), &self.func.value_shape(id).dims)
                            })
                            .collect::<Vec<_>>(),
                        op.dimension as usize,
                    )?,
                }]
            }
        } else if let Some(op) = node.op.as_any().downcast_ref::<Pad>() {
            let input = self.value(node.inputs[0])?;
            let pad = self.value(node.inputs[1])?;
            if pad.elems.len() != 1 {
                return Err("pad value is not scalar".into());
            }
            let elems = pad_elements(
                &input.elems,
                &self.func.value_shape(node.inputs[0]).dims,
                &pad.elems[0],
                op,
            )?;
            let indexed = input
                .indexed
                .clone()
                .map(|indexed| {
                    let mapping = pad_elements(
                        &(0..input.elems.len()).map(Some).collect::<Vec<_>>(),
                        &self.func.value_shape(node.inputs[0]).dims,
                        &None,
                        op,
                    )?;
                    let input_indices = mapping.iter().map(|index| index.unwrap_or(0)).collect();
                    let inside = mapping
                        .iter()
                        .map(|index| usize::from(index.is_some()))
                        .collect();
                    IndexedTensor::pad(
                        format!("loom_pad_{}", node.results[0].index()),
                        indexed,
                        pad.indexed
                            .clone()
                            .unwrap_or_else(|| IndexedTensor::dense(pad.elems.clone())),
                        input_indices,
                        inside,
                    )
                })
                .transpose()?;
            vec![Value {
                elems,
                constants: None,
                indexed,
                reductions: pad_elements(
                    &input.reductions,
                    &self.func.value_shape(node.inputs[0]).dims,
                    &None,
                    op,
                )?,
                guards: pad_elements(
                    &input.guards,
                    &self.func.value_shape(node.inputs[0]).dims,
                    &None,
                    op,
                )?,
            }]
        } else if let Some(op) = node.op.as_any().downcast_ref::<Reduce>() {
            vec![self.lower_reduce(node, op)?]
        } else if let Some(op) = node.op.as_any().downcast_ref::<DotGeneral>() {
            vec![self.lower_dot(node, op)?]
        } else if let Some(op) = node.op.as_any().downcast_ref::<Gather>() {
            vec![self.lower_gather(node, op)?]
        } else if let Some(op) = node.op.as_any().downcast_ref::<Scatter>() {
            vec![self.lower_scatter(node, op)?]
        } else if let Some(elementwise) = node.op.as_elementwise() {
            vec![self.lower_elementwise(node, elementwise.scalar())?]
        } else {
            return Err(format!(
                "unsupported XIR op; inputs={}, results={}",
                shapes(self.func, &node.inputs),
                shapes(self.func, &node.results)
            ));
        };
        if values.len() != node.results.len() {
            return Err(format!(
                "lowering produced {} values for {} results",
                values.len(),
                node.results.len()
            ));
        }
        for (&id, value) in node.results.iter().zip(values) {
            let dtype = self.func.value_shape(id).dtype;
            let value = self.bind(node_index, id, dtype, value)?;
            self.values.insert(id, value);
        }
        Ok(())
    }

    fn bind(
        &mut self,
        node_index: usize,
        id: ValueId,
        dtype: DType,
        value: Value,
    ) -> Result<Value, String> {
        if value.constants.is_some()
            || value.indexed.is_some()
            || self.func.nodes[node_index]
                .op
                .as_any()
                .downcast_ref::<Reshape>()
                .is_some()
        {
            return Ok(value);
        }
        let scalar = math_scalar(dtype)?;
        let mut elems = Vec::with_capacity(value.elems.len());
        for (element, (built, guard)) in value.elems.into_iter().zip(&value.guards).enumerate() {
            let name = format!("loom_v{}_{}", id.index(), element);
            let bound = self
                .stmts
                .bind_value_as(name, Ty::vec(scalar, self.lanes), built);
            if let Some(guard) = guard {
                let id = bound.id_in(self.stmts.function());
                self.statement_guards.insert(id, guard.clone());
            }
            elems.push(bound);
        }
        Ok(Value {
            elems,
            constants: None,
            indexed: None,
            reductions: value.reductions,
            guards: value.guards,
        })
    }
}

pub(super) use xir::element_mapping::num_elements;
