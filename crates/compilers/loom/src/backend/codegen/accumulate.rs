// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Folding a launch's evaluated values into its output accumulators.
//!
//! An accumulation is placed right after the last step its value depends on,
//! not at the end of the body: the value dies at the accumulator instead of
//! staying live to the bottom, which is what keeps register pressure bounded.
//! Under the algebraic policy a single-use product folds into the accumulator's
//! `fma` rather than being materialized first.
//!
//! An accumulator is a place: each fold defines a new value produced into the
//! storage the last one held, so the chain stays in one register and every later
//! read names the value the chain reached.

use std::collections::BTreeMap;

use gpu_ir::{Builder, OpId, OpKind, Value, ValueId};

use gpu_ir::{BinOp, ValTy};

use super::{LiftedProgram, Scalar, SelectedOutputs, Ty, output_scalar};

/// Sum `value` across the warp into lane 0, in the fixed halving order every
/// Loom reduction uses. The other lanes keep partial sums.
pub(super) fn warp_sum(b: &mut Builder, value: Value) -> Value {
    let mut sum = value;
    for offset in [16, 8, 4, 2, 1] {
        let delta = b.i32(offset);
        let other = b.shfl_down(sum.clone(), delta);
        let next = b.add(sum.clone(), other);
        sum = b.update(sum, next);
    }
    sum
}

/// How a block's threads are arranged for a cross-warp reduction: the values
/// holding this thread's lane and warp, and how many warps the block has.
#[derive(Clone)]
pub(super) struct Warps {
    pub lane: Value,
    pub warp: Value,
    pub count: u32,
}

/// Sum an accumulator across the block, given each warp already summed its own
/// lanes: lane 0 of every warp deposits into a shared stripe, warp 0 re-reduces
/// them, and `finish` runs inside warp 0 with the block total in hand.
///
/// The total is what the accumulator holds from here on, so `finish` is built
/// where it can read it rather than handed in already built against the value
/// the block started from.
pub(super) fn push_block_sum(
    b: &mut Builder,
    place: Value,
    partial: String,
    scalar: Scalar,
    warps: Warps,
    finish: impl FnOnce(&mut Builder, Value) -> Result<(), String>,
) -> Result<Value, String> {
    let stripe = b.shared_array(gpu_ir::SharedArr {
        name: partial,
        elem: scalar,
        len: i64::from(warps.count),
    });
    {
        let zero = b.i32(0);
        let leader = b.eq(warps.lane.clone(), zero);
        let (deposit, ()) = b.build_fragment(|f| {
            let slot = f.offset(stripe.clone(), warps.warp.clone());
            f.store(slot, place.clone());
        });
        let deposit = deposit.into_ops();
        b.if_(leader, |arm| arm.place_all(deposit));
    }
    b.barrier();
    let ty = Ty::scalar(scalar);
    let results = [(ty, Some(place.clone()))];
    b.open_arm();
    // Warps that deposited a partial contribute it; the rest contribute zero.
    let combined = {
        let count = b.i32(i64::from(warps.count));
        let within = b.lt(warps.lane.clone(), count);
        b.open_arm();
        let slot = b.offset(stripe, warps.lane.clone());
        let loaded = b.load(slot, 1, false);
        let taken = b.close_arm(vec![loaded]);
        b.open_arm();
        let empty = b.zero(scalar);
        let untaken = b.close_arm(vec![empty]);
        b.if_arms(within, &results, taken, untaken)
    };
    let held = combined.into_iter().next().expect("one result");
    let summed = warp_sum(b, held);
    let finished = finish(b, summed.clone());
    let taken = b.close_arm(vec![summed]);
    b.open_arm();
    let untaken = b.close_arm(vec![place]);
    let zero = b.i32(0);
    let first_warp = b.eq(warps.warp.clone(), zero);
    let produced = b.if_arms(first_warp, &results, taken, untaken);
    finished?;
    Ok(produced.into_iter().next().expect("one result"))
}

/// The scalar an output accumulates in: its math type, widened when packed half
/// lanes would lose the sum.
pub(super) fn accumulator_scalar(
    loom: &LiftedProgram,
    output: usize,
    lanes: u8,
) -> Result<Scalar, String> {
    Ok(vector_reduction_scalar(output_scalar(loom, output)?, lanes))
}

pub(super) fn vector_reduction_scalar(scalar: Scalar, lanes: u8) -> Scalar {
    if lanes > 1 && matches!(scalar, Scalar::F16 | Scalar::BF16) {
        Scalar::F32
    } else {
        scalar
    }
}

/// One accumulation per lane, folding `value` into the accumulator, and the
/// value the accumulator reaches.
pub(super) fn scalar_accumulation(
    b: &mut Builder,
    accumulator: Value,
    value: Value,
    lanes: u8,
    accumulator_scalar: Scalar,
    contract: bool,
    value_is_accumulator_type: bool,
) -> Value {
    let mut accumulator = accumulator;
    for lane in 0..lanes {
        let value = if lanes == 1 {
            value.clone()
        } else {
            b.extract(lane, value.clone())
        };
        let value = if value_is_accumulator_type {
            value
        } else {
            b.cast(accumulator_scalar, value)
        };
        let folded = accumulate(b, accumulator.clone(), value, contract);
        accumulator = b.update(accumulator, folded);
    }
    accumulator
}

/// `accumulator + value`, contracted into one `fma` where the value is a product
/// the caller is free to absorb.
pub(super) fn accumulate(
    b: &mut Builder,
    accumulator: Value,
    value: Value,
    contract: bool,
) -> Value {
    if contract
        && let Some(op) = b.function().operand_op(value.id_in(b.function()))
        && matches!(b.function().op(op).kind, OpKind::Bin(BinOp::Mul))
    {
        let (left, right) = (
            b.function().op(op).operands[0],
            b.function().op(op).operands[1],
        );
        let (left, right) = (b.value_of(left), b.value_of(right));
        return b.fma(left, right, accumulator);
    }
    b.add(accumulator, value)
}

/// One product an accumulation may absorb whole: where the body binds it, the
/// steps that go with it, and the factors it holds.
struct Product {
    steps: Vec<usize>,
    left: ValueId,
    right: ValueId,
    ty: Ty,
    root: usize,
}

/// The places a set of steps reads, counted once per step: a step reading one
/// place twice still counts once.
fn read_once_per_step(f: &gpu_ir::Function, ops: &[OpId], uses: &mut BTreeMap<ValueId, usize>) {
    for &op in ops {
        let mut read = std::collections::BTreeSet::new();
        f.add_op_reads(op, &mut read);
        for value in read {
            *uses.entry(value).or_default() += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn interleave_accumulations(
    b: &mut Builder,
    body: Vec<OpId>,
    outputs: SelectedOutputs,
    scalar: &mut [Vec<Option<Value>>],
    wide: &mut [Vec<Option<Value>>],
    io_lanes: u8,
    accumulator_scalars: &[Scalar],
    contract: bool,
) -> Vec<OpId> {
    let f = b.function();
    let mut output_uses = BTreeMap::<ValueId, usize>::new();
    for value in outputs.iter().flatten().flatten() {
        for read in f.reads(value.id_in(f)) {
            *output_uses.entry(read).or_default() += 1;
        }
    }
    let mut body_uses = BTreeMap::<ValueId, usize>::new();
    read_once_per_step(f, &body, &mut body_uses);

    let mut products = BTreeMap::<ValueId, Product>::new();
    if contract {
        let mut multiplications = BTreeMap::<ValueId, (usize, ValueId, ValueId, Ty)>::new();
        for (index, &op) in body.iter().enumerate() {
            if !matches!(f.op(op).kind, OpKind::Bind) {
                continue;
            }
            let defined = f.op(op).results[0];
            let ValTy::Data(ty) = f.ty(defined) else {
                continue;
            };
            let Some(product) = f.operand_op(f.op(op).operands[0]) else {
                continue;
            };
            if !matches!(f.op(product).kind, OpKind::Bin(BinOp::Mul)) || !ty.scalar.is_float() {
                continue;
            }
            multiplications.insert(
                defined,
                (
                    index,
                    f.op(product).operands[0],
                    f.op(product).operands[1],
                    ty,
                ),
            );
        }
        for (&defined, &(index, left, right, ty)) in &multiplications {
            if output_uses.get(&defined) == Some(&1) && !body_uses.contains_key(&defined) {
                products.insert(
                    defined,
                    Product {
                        steps: vec![index],
                        left,
                        right,
                        ty,
                        root: index,
                    },
                );
            }
        }
        // A binding that only renames a product — a copy, or a cast that changes
        // nothing — is the product as far as an accumulation is concerned.
        for (alias_index, &op) in body.iter().enumerate() {
            if !matches!(f.op(op).kind, OpKind::Bind) {
                continue;
            }
            let defined = f.op(op).results[0];
            let ValTy::Data(alias_ty) = f.ty(defined) else {
                continue;
            };
            let init = f.op(op).operands[0];
            let source = match f.operand_op(init) {
                None => init,
                Some(cast)
                    if matches!(f.op(cast).kind, OpKind::Cast(scalar) if scalar == alias_ty.scalar)
                        && f.is_materialized(f.op(cast).operands[0]) =>
                {
                    f.op(cast).operands[0]
                }
                Some(_) => continue,
            };
            let Some(&(product_index, left, right, product_ty)) = multiplications.get(&source)
            else {
                continue;
            };
            if alias_ty == product_ty
                && output_uses.get(&defined) == Some(&1)
                && !body_uses.contains_key(&defined)
                && !output_uses.contains_key(&source)
                && body_uses.get(&source) == Some(&1)
            {
                products.insert(
                    defined,
                    Product {
                        steps: vec![product_index, alias_index],
                        left,
                        right,
                        ty: alias_ty,
                        root: product_index,
                    },
                );
            }
        }
    }

    let definitions = body
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(f.op(**op).kind, OpKind::Bind))
        .map(|(index, op)| (f.op(*op).results[0], index))
        .collect::<BTreeMap<_, _>>();

    let mut insertions: Vec<Vec<OpId>> = (0..=body.len()).map(|_| Vec::new()).collect();
    let mut removed = vec![false; body.len()];
    for (output, values) in outputs.into_iter().enumerate() {
        for (element, value) in values.into_iter().enumerate() {
            let Some(value) = value else {
                continue;
            };
            let f = b.function();
            let mut value_is_accumulator_type = false;
            let mut root_index = None;
            let mut folded = None;
            let held = value.id_in(f);
            if let Some(product) = products.get(&held)
                && (wide[output][element].is_some()
                    || (io_lanes == 1 && product.ty.scalar == accumulator_scalars[output]))
            {
                for &index in &product.steps {
                    removed[index] = true;
                }
                value_is_accumulator_type = product.ty.scalar == accumulator_scalars[output];
                root_index = Some(product.root);
                folded = Some((product.left, product.right));
            }
            let after = root_index.unwrap_or_else(|| {
                f.reads(held)
                    .iter()
                    .filter_map(|value| definitions.get(value))
                    .max()
                    .map_or(0, |index| index + 1)
            });
            let (fragment, ()) = b.build_fragment(|g| {
                let value = match folded {
                    Some((left, right)) => {
                        let (left, right) = (g.value_of(left), g.value_of(right));
                        g.mul(left, right)
                    }
                    None => value,
                };
                if let Some(place) = wide[output][element].clone() {
                    let sum = accumulate(g, place.clone(), value, contract);
                    wide[output][element] = Some(g.update(place, sum));
                } else if let Some(place) = scalar[output][element].clone() {
                    scalar[output][element] = Some(scalar_accumulation(
                        g,
                        place,
                        value,
                        io_lanes,
                        accumulator_scalars[output],
                        contract,
                        value_is_accumulator_type,
                    ));
                }
            });
            insertions[after].extend(fragment.into_ops());
        }
    }
    let mut interleaved =
        Vec::with_capacity(body.len() + insertions.iter().map(Vec::len).sum::<usize>());
    let mut insertions = insertions.into_iter();
    interleaved.extend(insertions.next().expect("one insertion boundary"));
    for ((index, op), after) in body.into_iter().enumerate().zip(insertions) {
        if !removed[index] {
            interleaved.push(op);
        }
        interleaved.extend(after);
    }
    interleaved
}
