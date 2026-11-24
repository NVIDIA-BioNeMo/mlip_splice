// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{BinOp, Carry, CmpOp, Scalar, Trip, Ty};
use xir::element_mapping::{axes, gather_map, num_elements, reduction_map, scatter_map};
use xir::ops::{DotGeneral, Gather, Reduce, ScalarOp, Scatter};

use super::scalar::{
    approximate_scalar, arc_tangent_of_ratio, comparison, exponential, half_power, magnitude,
    math_scalar, natural_log, one, sign_of, signed_integer, spread_float, unsigned_integer,
};
use super::shape::{lower_dot_values, zip_exact};
use super::{
    Evaluator, GuardKey, GuardProvenance, IndexedTensor, ReductionKind, Value, lower_scalar_region,
    scalar_region_is_add,
};

impl Evaluator<'_> {
    pub(super) fn lower_elementwise(
        &mut self,
        node: &xir::ir::Node,
        op: ScalarOp,
    ) -> Result<Value, String> {
        let dtype = self.func.value_shape(node.results[0]).dtype;
        let scalar = math_scalar(dtype)?;
        let lanes = self.lanes;
        let elements = num_elements(&self.func.value_shape(node.results[0]).dims)?;
        let inputs = node
            .inputs
            .iter()
            .map(|&input| self.value(input).cloned())
            .collect::<Result<Vec<_>, String>>()?;
        if inputs.iter().any(|input| input.indexed.is_some()) {
            let indexed = inputs
                .iter()
                .map(|input| {
                    input
                        .indexed
                        .clone()
                        .unwrap_or_else(|| IndexedTensor::dense(input.elems.clone()))
                })
                .collect();
            return Ok(Value {
                elems: self.placeholders(dtype, elements)?,
                constants: None,
                indexed: Some(IndexedTensor::elementwise(op, dtype, indexed, elements)),
                reductions: vec![None; elements],
                guards: vec![None; elements],
            });
        }
        let b = &mut self.stmts;
        let spread_one = |b: &mut gpu_ir::Builder| {
            let one = one(b, scalar);
            b.splat(lanes, one)
        };
        let elems: Vec<gpu_ir::Value> = match op {
            ScalarOp::Negate => inputs[0]
                .elems
                .iter()
                .map(|value| b.neg(value.clone()))
                .collect(),
            ScalarOp::Abs => inputs[0]
                .elems
                .iter()
                .map(|value| magnitude(b, value.clone()))
                .collect(),
            ScalarOp::Sign => inputs[0]
                .elems
                .iter()
                .map(|value| sign_of(b, value.clone(), scalar, lanes))
                .collect(),
            ScalarOp::Sqrt => inputs[0]
                .elems
                .iter()
                .map(|value| b.sqrt(value.clone()))
                .collect(),
            ScalarOp::Rsqrt => inputs[0]
                .elems
                .iter()
                .map(|value| {
                    let one = spread_one(b);
                    let root = b.sqrt(value.clone());
                    b.div(one, root)
                })
                .collect(),
            ScalarOp::Not => inputs[0]
                .elems
                .iter()
                .map(|value| b.not(value.clone()))
                .collect(),
            ScalarOp::Add
            | ScalarOp::Subtract
            | ScalarOp::Multiply
            | ScalarOp::Divide
            | ScalarOp::Maximum
            | ScalarOp::Minimum
            | ScalarOp::Remainder
            | ScalarOp::And
            | ScalarOp::Or
            | ScalarOp::Xor
            | ScalarOp::ShiftLeft
            | ScalarOp::ShiftRightArithmetic
            | ScalarOp::ShiftRightLogical => {
                let pairs = zip_exact(&inputs[0].elems, &inputs[1].elems)?
                    .map(|(lhs, rhs)| (lhs.clone(), rhs.clone()))
                    .collect::<Vec<_>>();
                pairs
                    .into_iter()
                    .map(|(lhs, rhs)| {
                        Ok(match op {
                            ScalarOp::Add => b.add(lhs, rhs),
                            ScalarOp::Subtract => b.sub(lhs, rhs),
                            ScalarOp::Multiply => b.mul(lhs, rhs),
                            ScalarOp::Divide => b.div(lhs, rhs),
                            ScalarOp::Maximum => b.max(lhs, rhs),
                            ScalarOp::Minimum => b.min(lhs, rhs),
                            ScalarOp::Remainder => b.rem(lhs, rhs),
                            ScalarOp::And => b.bin(BinOp::And, lhs, rhs),
                            ScalarOp::Or => b.bin(BinOp::Or, lhs, rhs),
                            ScalarOp::Xor => b.bin(BinOp::Xor, lhs, rhs),
                            ScalarOp::ShiftLeft => {
                                let amount = b.cast(Scalar::U32, rhs);
                                b.bin(BinOp::Shl, lhs, amount)
                            }
                            ScalarOp::ShiftRightArithmetic => {
                                let wide = b.cast(signed_integer(scalar)?, lhs);
                                let amount = b.cast(Scalar::U32, rhs);
                                let shifted = b.bin(BinOp::Shr, wide, amount);
                                b.cast(scalar, shifted)
                            }
                            ScalarOp::ShiftRightLogical => {
                                let wide = b.cast(unsigned_integer(scalar)?, lhs);
                                let amount = b.cast(Scalar::U32, rhs);
                                let shifted = b.bin(BinOp::Shr, wide, amount);
                                b.cast(scalar, shifted)
                            }
                            _ => unreachable!(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?
            }
            ScalarOp::Power => {
                let constants = inputs[1]
                    .constants
                    .as_ref()
                    .ok_or_else(|| "power exponent must be compile-time constant".to_string())?;
                let pairs = zip_exact(&inputs[0].elems, constants)?
                    .map(|(base, exponent)| (base.clone(), *exponent))
                    .collect::<Vec<_>>();
                pairs
                    .into_iter()
                    .map(|(base, exponent)| half_power(b, base, exponent.as_f64()?, scalar, lanes))
                    .collect::<Result<Vec<_>, String>>()?
            }
            ScalarOp::Compare(direction) => {
                let pairs = zip_exact(&inputs[0].elems, &inputs[1].elems)?
                    .map(|(lhs, rhs)| (lhs.clone(), rhs.clone()))
                    .collect::<Vec<_>>();
                pairs
                    .into_iter()
                    .map(|(lhs, rhs)| b.cmp(comparison(direction), lhs, rhs))
                    .collect()
            }
            ScalarOp::Select => {
                let (pred, yes, no) = (&inputs[0].elems, &inputs[1].elems, &inputs[2].elems);
                // A select whose predicate is one element chooses the same way
                // for every element it governs, which is what a compact-support
                // cutoff is: one radius decides a whole path's worth of terms.
                // StableHLO states that as a rank-0 predicate, so a single
                // element stands for the whole width here rather than being a
                // shape disagreement.
                let triples = broadcast_triples(pred, yes, no)?;
                triples
                    .into_iter()
                    .map(|(pred, yes, no)| b.select(pred, yes, no))
                    .collect()
            }
            ScalarOp::Exponential
            | ScalarOp::ExponentialMinusOne
            | ScalarOp::Log
            | ScalarOp::LogPlusOne
            | ScalarOp::Logistic
            | ScalarOp::Cbrt
            | ScalarOp::Tanh
            | ScalarOp::Sine
            | ScalarOp::Cosine
            | ScalarOp::Tan => {
                // Every one of these is one hardware approximation, or a few
                // composed from it. They are opaque to exact rewriting, which
                // recognizes the field operations alone, so each is a boundary
                // of the island the rewriting works inside.
                approximate_scalar(scalar)?;
                inputs[0]
                    .elems
                    .iter()
                    .map(|value| {
                        let value = value.clone();
                        match op {
                            ScalarOp::Exponential => exponential(b, value, scalar, lanes),
                            ScalarOp::ExponentialMinusOne => {
                                let raised = exponential(b, value, scalar, lanes);
                                let one = spread_one(b);
                                b.sub(raised, one)
                            }
                            ScalarOp::Log => natural_log(b, value, scalar, lanes),
                            ScalarOp::LogPlusOne => {
                                let one = spread_one(b);
                                let shifted = b.add(one, value);
                                natural_log(b, shifted, scalar, lanes)
                            }
                            ScalarOp::Logistic => {
                                let negated = b.neg(value);
                                let raised = exponential(b, negated, scalar, lanes);
                                let one = spread_one(b);
                                let denominator = b.add(one.clone(), raised);
                                b.div(one, denominator)
                            }
                            // At zero the chain is exact without a guard:
                            // log2(0) is minus infinity, two raised to it is
                            // zero, and the sign of zero is zero.
                            ScalarOp::Cbrt => {
                                let magnitude = magnitude(b, value.clone());
                                let logarithm = natural_log(b, magnitude, scalar, lanes);
                                let third = spread_float(b, 1.0 / 3.0, scalar, lanes);
                                let scaled = b.mul(logarithm, third);
                                let root = exponential(b, scaled, scalar, lanes);
                                let sign = sign_of(b, value, scalar, lanes);
                                b.mul(sign, root)
                            }
                            ScalarOp::Tanh => b.un(gpu_ir::UnOp::Tanh, value),
                            ScalarOp::Sine => b.un(gpu_ir::UnOp::Sin, value),
                            ScalarOp::Cosine => b.un(gpu_ir::UnOp::Cos, value),
                            _ => {
                                let sine = b.un(gpu_ir::UnOp::Sin, value.clone());
                                let cosine = b.un(gpu_ir::UnOp::Cos, value);
                                b.div(sine, cosine)
                            }
                        }
                    })
                    .collect()
            }
            ScalarOp::Floor | ScalarOp::Ceil | ScalarOp::RoundNearestEven => {
                let rounding = match op {
                    ScalarOp::Floor => gpu_ir::UnOp::Floor,
                    ScalarOp::Ceil => gpu_ir::UnOp::Ceil,
                    _ => gpu_ir::UnOp::RoundEven,
                };
                inputs[0]
                    .elems
                    .iter()
                    .map(|value| b.un(rounding, value.clone()))
                    .collect()
            }
            // Halves away from zero, which no conversion mode spells: round the
            // magnitude down after a half step and put the sign back.
            ScalarOp::RoundNearestAfz => inputs[0]
                .elems
                .iter()
                .map(|value| {
                    let value = value.clone();
                    let magnitude = magnitude(b, value.clone());
                    let half = spread_float(b, 0.5, scalar, lanes);
                    let stepped = b.add(magnitude, half);
                    let rounded = b.un(gpu_ir::UnOp::Floor, stepped);
                    let sign = sign_of(b, value, scalar, lanes);
                    b.mul(sign, rounded)
                })
                .collect(),
            // Infinity compares false against itself under `lt`, and every
            // comparison with a NaN is false, so one comparison decides both.
            // The bound belongs to the operand's type, not the bool result's.
            ScalarOp::IsFinite => {
                let operand_scalar = math_scalar(self.func.value_shape(node.inputs[0]).dtype)?;
                inputs[0]
                    .elems
                    .iter()
                    .map(|value| {
                        let magnitude = magnitude(b, value.clone());
                        let infinite = spread_float(b, f64::INFINITY, operand_scalar, lanes);
                        b.cmp(CmpOp::Lt, magnitude, infinite)
                    })
                    .collect()
            }
            ScalarOp::Atan2 => {
                approximate_scalar(scalar)?;
                let pairs = zip_exact(&inputs[0].elems, &inputs[1].elems)?
                    .map(|(y, x)| (y.clone(), x.clone()))
                    .collect::<Vec<_>>();
                pairs
                    .into_iter()
                    .map(|(y, x)| arc_tangent_of_ratio(b, y, x, scalar, lanes))
                    .collect()
            }
            ScalarOp::CountLeadingZeros | ScalarOp::Popcnt => {
                let counting = if op == ScalarOp::CountLeadingZeros {
                    gpu_ir::UnOp::LeadingZeros
                } else {
                    gpu_ir::UnOp::OneBits
                };
                inputs[0]
                    .elems
                    .iter()
                    .map(|value| b.un(counting, value.clone()))
                    .collect()
            }
            // StableHLO orders the operands low, value, high.
            ScalarOp::Clamp => {
                let triples =
                    broadcast_triples(&inputs[1].elems, &inputs[0].elems, &inputs[2].elems)?;
                triples
                    .into_iter()
                    .map(|(value, low, high)| {
                        let bounded = b.min(value, high);
                        b.max(bounded, low)
                    })
                    .collect()
            }
        };
        let guards = (0..elems.len())
            .map(|element| {
                if op == ScalarOp::Select {
                    let yes_zero = self.facts.value(node.inputs[1]).exact_zero;
                    let no_zero = self.facts.value(node.inputs[2]).exact_zero;
                    // A one-element predicate governs every element, so its
                    // guard is the same one throughout, and the key names that
                    // element rather than this one.
                    let predicate_element = if inputs[0].elems.len() == 1 {
                        0
                    } else {
                        element
                    };
                    return match (yes_zero, no_zero) {
                        (false, true) => Some(GuardProvenance {
                            predicate: inputs[0].elems[predicate_element]
                                .id_in(self.stmts.function()),
                            key: GuardKey {
                                value: node.inputs[0],
                                element: predicate_element,
                                inverted: false,
                            },
                        }),
                        (true, false) => Some(GuardProvenance {
                            predicate: {
                                let inverted =
                                    self.stmts.not(inputs[0].elems[predicate_element].clone());
                                inverted.id_in(self.stmts.function())
                            },
                            key: GuardKey {
                                value: node.inputs[0],
                                element: predicate_element,
                                inverted: true,
                            },
                        }),
                        _ => None,
                    };
                }
                let guard = |index: usize| {
                    inputs
                        .get(index)?
                        .guards
                        .get(element)
                        .and_then(Clone::clone)
                };
                match op {
                    ScalarOp::Multiply => guard(0).or_else(|| guard(1)),
                    ScalarOp::Add | ScalarOp::Subtract => {
                        let lhs = guard(0);
                        let rhs = guard(1);
                        match (lhs, rhs) {
                            (Some(lhs), Some(rhs)) if lhs.key == rhs.key => Some(lhs),
                            (Some(lhs), None) if self.facts.value(node.inputs[1]).exact_zero => {
                                Some(lhs)
                            }
                            (None, Some(rhs)) if self.facts.value(node.inputs[0]).exact_zero => {
                                Some(rhs)
                            }
                            _ => None,
                        }
                    }
                    ScalarOp::Negate | ScalarOp::Abs => guard(0),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let constants = inputs
            .iter()
            .map(|input| input.constants.as_deref())
            .collect::<Option<Vec<_>>>()
            .and_then(|inputs| {
                (0..elems.len())
                    .map(|element| {
                        let values = inputs
                            .iter()
                            .map(|input| input[if input.len() == 1 { 0 } else { element }])
                            .collect::<Vec<_>>();
                        super::eval_const(op, &values, dtype).ok()
                    })
                    .collect::<Option<Vec<_>>>()
            });
        Ok(Value {
            reductions: vec![None; elems.len()],
            guards,
            indexed: None,
            elems,
            constants,
        })
    }

    pub(super) fn lower_reduce(
        &mut self,
        node: &xir::ir::Node,
        op: &Reduce,
    ) -> Result<Value, String> {
        if node.inputs.len() != 2 || node.results.len() != 1 {
            return Err("only one-input, one-result reduce is supported".into());
        }
        let input = self.value(node.inputs[0])?.clone();
        let init = self.value(node.inputs[1])?.clone();
        if init.elems.len() != 1 {
            return Err("reduce init must be scalar".into());
        }
        let in_dims = self.func.value_shape(node.inputs[0]).dims.to_vec();
        let (out_dims, cells) = reduction_map(&in_dims, &axes(&op.dimensions))?;
        let output_elements = num_elements(&out_dims)?;
        if let Some(indexed_input) = input.indexed.clone() {
            return Ok(Value {
                elems: self.placeholders(
                    self.func.value_shape(node.results[0]).dtype,
                    output_elements,
                )?,
                constants: None,
                indexed: Some(IndexedTensor::reduce(
                    format!("loom_reduce_{}", node.results[0].index()),
                    indexed_input,
                    init.indexed
                        .clone()
                        .unwrap_or_else(|| IndexedTensor::dense(init.elems.clone())),
                    op.reducer,
                    &cells,
                    output_elements,
                )?),
                reductions: vec![None; output_elements],
                guards: vec![None; output_elements],
            });
        }
        let module = self.module;
        let lanes = self.lanes;
        let mut output = vec![init.elems[0].clone(); output_elements];
        let mut term_guards = vec![Vec::new(); output.len()];
        for (flat, value) in input.elems.iter().enumerate() {
            let out_index = cells[flat];
            let folded = lower_scalar_region(
                &mut self.stmts,
                module,
                op.reducer,
                vec![output[out_index].clone(), value.clone()],
                lanes,
            )?;
            output[out_index] = folded;
            term_guards[out_index].push(input.guards[flat].clone());
        }
        let guards = term_guards
            .into_iter()
            .map(|guards| {
                if !self.facts.value(node.inputs[1]).exact_zero {
                    return None;
                }
                let mut guards = guards.into_iter();
                let first = guards.next().flatten()?;
                guards
                    .all(|guard| guard.is_some_and(|guard| first.key == guard.key))
                    .then_some(first)
            })
            .collect();
        Ok(Value {
            guards,
            reductions: vec![
                scalar_region_is_add(module, op.reducer)
                    .then_some(ReductionKind::Reduce);
                output.len()
            ],
            elems: output,
            constants: None,
            indexed: None,
        })
    }

    pub(super) fn lower_dot(
        &mut self,
        node: &xir::ir::Node,
        op: &DotGeneral,
    ) -> Result<Value, String> {
        let lhs = self.value(node.inputs[0])?.clone();
        let rhs = self.value(node.inputs[1])?.clone();
        if lhs.indexed.is_some() || rhs.indexed.is_some() {
            return Err("dot_general of indexed values is unsupported".into());
        }
        let lhs_dims = self.func.value_shape(node.inputs[0]).dims.to_vec();
        let rhs_dims = self.func.value_shape(node.inputs[1]).dims.to_vec();
        let elems = lower_dot_values(
            &mut self.stmts,
            &lhs.elems,
            &lhs_dims,
            &rhs.elems,
            &rhs_dims,
            op,
        )?;
        let count = elems.len();
        Ok(Value {
            elems,
            constants: None,
            indexed: None,
            guards: vec![None; count],
            reductions: vec![Some(ReductionKind::Dot); count],
        })
    }

    pub(super) fn lower_gather(
        &mut self,
        node: &xir::ir::Node,
        op: &Gather,
    ) -> Result<Value, String> {
        let input = self.value(node.inputs[0])?.clone();
        let starts = self.value(node.inputs[1])?.clone();
        let input_indexed = input
            .indexed
            .clone()
            .unwrap_or_else(|| IndexedTensor::dense(input.elems.clone()));
        let output_elements = num_elements(&self.func.value_shape(node.results[0]).dims)?;
        let indexed = if let Some(constants) = &starts.constants {
            let indices = constants
                .iter()
                .copied()
                .map(|value| value.as_i64())
                .collect::<Result<Vec<_>, _>>()?;
            let map = gather_map(
                &self.func.value_shape(node.inputs[0]).dims,
                &self.func.value_shape(node.inputs[1]).dims,
                &indices,
                &self.func.value_shape(node.results[0]).dims,
                op,
            )?;
            input_indexed.reindex(format!("loom_gather_{}", node.results[0].index()), map)?
        } else {
            IndexedTensor::gather(
                input_indexed,
                starts
                    .indexed
                    .clone()
                    .unwrap_or_else(|| IndexedTensor::dense(starts.elems.clone())),
                self.func.value_shape(node.inputs[0]).dims.to_vec(),
                self.func.value_shape(node.inputs[1]).dims.to_vec(),
                self.func.value_shape(node.results[0]).dims.to_vec(),
                op.clone(),
                output_elements,
            )
        };
        Ok(Value {
            elems: self.placeholders(
                self.func.value_shape(node.results[0]).dtype,
                output_elements,
            )?,
            indexed: Some(indexed),
            constants: None,
            reductions: vec![None; output_elements],
            guards: vec![None; output_elements],
        })
    }

    /// A scatter with runtime indices: one counted loop per chunk of outputs,
    /// carrying that chunk's accumulators.
    ///
    /// Each iteration computes where the update lands and then, per output the
    /// chunk holds, produces into that accumulator's own storage where the
    /// destination matches it and passes it through where it does not. The
    /// branch is what keeps the combine off the outputs it does not address; a
    /// select would evaluate it for every one of them.
    pub(super) fn lower_scatter(
        &mut self,
        node: &xir::ir::Node,
        op: &Scatter,
    ) -> Result<Value, String> {
        let operand = self.value(node.inputs[0])?.clone();
        let indices = self.value(node.inputs[1])?.clone();
        let updates = self.value(node.inputs[2])?.clone();
        if let Some(constants) = &indices.constants {
            let indices = constants
                .iter()
                .copied()
                .map(super::Const::as_i64)
                .collect::<Result<Vec<_>, _>>()?;
            let destinations = scatter_map(
                &self.func.value_shape(node.inputs[0]).dims,
                &self.func.value_shape(node.inputs[1]).dims,
                &indices,
                &self.func.value_shape(node.inputs[2]).dims,
                op,
            )?;
            let elements = operand.elems.len();
            let indexed = IndexedTensor::scatter(
                format!("loom_scatter_{}", node.results[0].index()),
                operand
                    .indexed
                    .unwrap_or_else(|| IndexedTensor::dense(operand.elems)),
                updates
                    .indexed
                    .unwrap_or_else(|| IndexedTensor::dense(updates.elems)),
                op.update,
                &destinations,
            )?;
            return Ok(Value {
                indexed: Some(indexed),
                elems: self.placeholders(self.func.value_shape(node.results[0]).dtype, elements)?,
                constants: None,
                reductions: vec![None; elements],
                guards: vec![None; elements],
            });
        }
        let initial = if let Some(indexed) = &operand.indexed {
            let indexed = indexed.clone();
            (0..indexed.elements)
                .map(|element| self.emit_indexed_at(&indexed, element))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            operand.elems.clone()
        };
        let mut elems = Vec::with_capacity(initial.len());
        let updates_indexed = updates
            .indexed
            .clone()
            .unwrap_or_else(|| IndexedTensor::dense(updates.elems.clone()));
        let indices_indexed = indices
            .indexed
            .clone()
            .unwrap_or_else(|| IndexedTensor::dense(indices.elems.clone()));
        let scalar = math_scalar(self.func.value_shape(node.results[0]).dtype)?;
        let module = self.module;
        let lanes = self.lanes;
        let element_ty = Ty::vec(scalar, lanes);
        let operand_dims = self.func.value_shape(node.inputs[0]).dims.to_vec();
        let indices_dims = self.func.value_shape(node.inputs[1]).dims.to_vec();
        let updates_dims = self.func.value_shape(node.inputs[2]).dims.to_vec();
        let operand_elements = operand.elems.len();
        let stop = i32::try_from(updates.elems.len())
            .map_err(|_| "scatter update count does not fit i32")?;
        let outputs_per_fold = usize::from(u8::MAX / lanes).max(1);
        for (chunk, initial) in initial.chunks(outputs_per_fold).enumerate() {
            let first = i32::try_from(chunk * outputs_per_fold)
                .map_err(|_| "scatter output index does not fit i32")?;
            let slots = u8::try_from(initial.len()).map_err(|_| "scatter chunk is too wide")?;
            let packed_ty = Ty::vec(
                scalar,
                slots
                    .checked_mul(lanes)
                    .ok_or("scatter chunk is too wide")?,
            );
            // The accumulators stay packed: one value of `slots × lanes`, one
            // thing the loop carries, and one thing it reduces to. Each slot is
            // a chunk of it that its own branch updates in place.
            let start = self.stmts.pack(initial.to_vec());
            let carried = vec![Carry {
                ty: packed_ty,
                init: start,
                into: None,
            }];
            let start = self.stmts.i32(0);
            let limit = self.stmts.i32(i64::from(stop));
            let step = self.stmts.i32(1);
            let open = self.stmts.open_loop_body(Scalar::S32, carried);
            let index = open.index();
            let accumulator = open.given()[0].clone();
            let update = self.emit_indexed(&updates_indexed, index.clone())?;
            let update = self.stmts.bind_value(element_ty, update);
            let destination = self.emit_scatter_destination(
                &indices_indexed,
                index,
                &operand_dims,
                &indices_dims,
                &updates_dims,
                op,
                operand_elements,
            )?;
            let destination_ty = self.stmts.value_ty(&destination).data();
            let destination = self.stmts.bind_value(destination_ty, destination);
            let mut held = accumulator.clone();
            for slot in 0..slots {
                let chunk_of = |b: &mut gpu_ir::Builder, held: &gpu_ir::Value, chunk: u8| {
                    let read = (0..lanes)
                        .map(|lane| b.extract(chunk * lanes + lane, held.clone()))
                        .collect::<Vec<_>>();
                    b.pack(read)
                };
                let wanted = i64::from(first) + i64::from(slot);
                let wanted = self.stmts.int(destination_ty.scalar, wanted);
                let matches = self.stmts.eq(destination.clone(), wanted);
                self.stmts.open_arm();
                let mine = chunk_of(&mut self.stmts, &held, slot);
                let combined = lower_scalar_region(
                    &mut self.stmts,
                    module,
                    op.update,
                    vec![mine, update.clone()],
                    lanes,
                )?;
                // Every other chunk arrives as it stands, so the branch writes
                // this one and nothing else.
                let parts = (0..slots)
                    .map(|other| {
                        if other == slot {
                            combined.clone()
                        } else {
                            chunk_of(&mut self.stmts, &held, other)
                        }
                    })
                    .collect::<Vec<_>>();
                let next = self.stmts.pack(parts);
                let then_ = self.stmts.close_arm(vec![next]);
                self.stmts.open_arm();
                let else_ = self.stmts.close_arm(vec![held.clone()]);
                let produced = self.stmts.if_arms(
                    matches,
                    &[(packed_ty, Some(accumulator.clone()))],
                    then_,
                    else_,
                );
                held = produced.into_iter().next().expect("one packed result");
            }
            let body = self.stmts.close_loop_body(open, vec![held]);
            // The loop stands for the value it reduces to rather than being a
            // step: the profile's body is bindings, and the whole scatter is one
            // of them — one definition of one packed value, costing one, with
            // the destination and the update read once per iteration inside it.
            let folded = self
                .stmts
                .fold_loop(Trip::PerThread, start, limit, step, body);
            let packed = self.stmts.bind_value_as(
                format!("loom_scatter_fold_{}_{chunk}", node.results[0].index()),
                packed_ty,
                folded.into_iter().next().expect("one packed result"),
            );
            // Slots are read back out of the one packed binding, as they were
            // when the fold was one expression.
            for element in 0..slots {
                let read = (0..lanes)
                    .map(|lane| self.stmts.extract(element * lanes + lane, packed.clone()))
                    .collect::<Vec<_>>();
                elems.push(if lanes == 1 {
                    read.into_iter().next().expect("one lane")
                } else {
                    self.stmts.pack(read)
                });
            }
        }
        let count = elems.len();
        Ok(Value {
            indexed: None,
            elems,
            constants: None,
            reductions: vec![None; count],
            guards: vec![None; count],
        })
    }
}

/// The `(predicate, yes, no)` triples a select evaluates, one per element it
/// produces.
///
/// An operand of one element governs every element of the result: that is what a
/// rank-0 StableHLO predicate means, and what a scalar arm means. Any other
/// disagreement in element count is a genuine shape error and is refused.
fn broadcast_triples(
    pred: &[gpu_ir::Value],
    yes: &[gpu_ir::Value],
    no: &[gpu_ir::Value],
) -> Result<Vec<(gpu_ir::Value, gpu_ir::Value, gpu_ir::Value)>, String> {
    let width = pred.len().max(yes.len()).max(no.len());
    let at = |operand: &[gpu_ir::Value], element: usize| -> Result<gpu_ir::Value, String> {
        match operand.len() {
            length if length == width => Ok(operand[element].clone()),
            1 => Ok(operand[0].clone()),
            length => Err(format!(
                "select operand has {length} elements, expected 1 or {width}"
            )),
        }
    };
    (0..width)
        .map(|element| Ok((at(pred, element)?, at(yes, element)?, at(no, element)?)))
        .collect()
}
