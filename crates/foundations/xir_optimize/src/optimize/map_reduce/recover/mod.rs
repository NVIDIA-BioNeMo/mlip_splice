// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recovers map/reduce regions from a scalarized program.
//!
//! A map/reduce region is a wide reduction over many structurally identical
//! terms. Re-expressed as gathers that choose each lane's operands, the shared
//! operation sequence over a lane axis, and a **non-unique** Scatter under the
//! reducer, it becomes something the backend can walk with a loop. Non-unique is
//! the whole point: many lanes land on one element. A unique Scatter is a
//! permutation and resolves straight back to scalar work.
//!
//! The region is usually not written down. The extractor that chose this
//! program's form minimizes node count, so it factors — `a*(b + c)` rather than
//! `a*b + a*c` — and the factored form has no wide reduction anywhere, only
//! short sums over heavily shared intermediates. Both forms are the same
//! e-class, but only the distributed one is a map/reduce. So the recognizer
//! distributes for itself, under a declared algebraic law and a hard bound, and
//! throws the expansion away once it has chosen a batching. See [`distribute`].
//!
//! Nothing here knows what the arithmetic means. Terms are grouped by [`infer_shapes`]
//! isomorphism and by which pool each operand is drawn from — not by degree,
//! monomial, or coefficient — so the same recovery applies to whatever the
//! e-graph leaves behind, and it does not care whether the program is a forward
//! pass or the transpose autodiff produced from one.
//!
//! Reassociating and distributing are sound only under the algebraic math
//! policy, which is this pass's sole caller.

use std::collections::{HashMap, HashSet};

use smallvec::{SmallVec, smallvec};

use crate::dim_numbers::ScatterDimensionNumbers;
use crate::dtype::DType;
use crate::ir::{Def, Func, FuncId, Module, Node, ValueDef, ValueId};
use crate::op::Op;
use crate::ops::{
    Add, And, Constant, Literal, Maximum, Minimum, Multiply, Or, Reshape, ScalarOp, Scatter,
};
use crate::shape::Shape;

use super::super::rebuild::Rebuilder;

mod distribute;
mod infer_shapes;
mod operand_pools;

use distribute::{
    Expansion, distributed_product, region_leaves, scalar_region, sole_consumers, use_counts,
};
use infer_shapes::{ShapeId, Shapes, producer};
use operand_pools::{Group, LeafPools, group_terms, index_vector, take_vector};

/// How large the tree unfolding of one atom may get. A batch replays its atoms
/// per lane, so an atom that only looks small because of shared subexpressions
/// would trade program size for work at a ruinous rate.
const MAX_SHAPE_UNFOLDING: u64 = 1 << 12;

/// The reducers a region may be recovered under. Each is associative and
/// commutative, so the scatter may visit terms in any order, and each carries an
/// identity XIR can build, so the scatter has somewhere to start.
///
/// The recovery reads none of this — it batches by structure alone — so widening
/// these functions and [`distribute::distributes_over`] is all another algebra
/// needs.
fn reducers() -> [ScalarOp; 1] {
    [ScalarOp::Add]
}

fn scalar_op(scalar: ScalarOp) -> Option<Box<dyn Op>> {
    Some(match scalar {
        ScalarOp::Negate => Box::new(crate::ops::Negate),
        ScalarOp::Add => Box::new(Add),
        ScalarOp::Multiply => Box::new(Multiply),
        ScalarOp::And => Box::new(And),
        ScalarOp::Or => Box::new(Or),
        ScalarOp::Maximum => Box::new(Maximum),
        ScalarOp::Minimum => Box::new(Minimum),
        _ => return None,
    })
}

fn reduction_identity(scalar: ScalarOp, shape: &Shape) -> Option<Literal> {
    match scalar {
        ScalarOp::Add => Literal::zeros(shape.clone()).ok(),
        _ => None,
    }
}

/// One recovered region: the value it replaces, the reducer combining its terms,
/// the product combining each term's atoms, and the terms themselves.
struct Region {
    root: ValueId,
    reducer: ScalarOp,
    product: ScalarOp,
    terms: Vec<Vec<ValueId>>,
    /// Per term, the unary operations wrapping it. Terms that differ here take
    /// different batches, because one batch emits one operation sequence.
    wrappers: Vec<Vec<ScalarOp>>,
}

pub(in crate::optimize) fn recover_map_reduce_regions(
    module: &mut Module,
    max_terms: usize,
) -> Result<bool, String> {
    if max_terms == 0 {
        return Ok(false);
    }
    let old = module.entry_func().clone();
    let scalar = scalar_region(&old);
    let uses = use_counts(&old);
    let consumer = sole_consumers(&old);

    let (shapes, assigned) =
        infer_shapes::assign(&old, MAX_SHAPE_UNFOLDING, |value| !scalar[value.index()]);
    let mut rebuilt = Rebuilder::new(&old);
    let pools = LeafPools::build(&mut rebuilt, &old, &region_leaves(&old, &scalar))?;

    // Roots are taken outermost first, and a root that distribution cannot pay
    // for hands its own interior reductions back to the queue. That fallback is
    // what keeps a factorization the extractor was right about: when scaling a
    // sum by one factor would cost a multiply per term, the scale stays where it
    // is and only the sum underneath becomes a loop.
    let mut regions: Vec<Region> = Vec::new();
    let mut accepted: HashMap<ValueId, (usize, Vec<Group>)> = HashMap::new();
    let mut claimed: HashSet<ValueId> = HashSet::new();
    for reducer in reducers() {
        let mut expansion = Expansion::new(reducer, max_terms);
        let mut queue = old
            .nodes
            .iter()
            .filter_map(|node| {
                let [root] = node.results.as_slice() else {
                    return None;
                };
                distribute::is_expansion_root(&old, *root, &scalar, reducer, &uses, &consumer)
                    .then_some(*root)
            })
            .collect::<Vec<_>>();
        while let Some(root) = queue.pop() {
            if !claimed.insert(root) {
                continue;
            }
            let Some(product) = distributed_product(&old, root, reducer) else {
                continue;
            };
            let Some(terms) = expansion.terms(&old, root, &scalar) else {
                queue.extend(distribute::interior_reductions(
                    &old, root, &scalar, reducer,
                ));
                continue;
            };
            // One term is not a reduction, and a term with no atoms is empty.
            if terms.len() < 2 || terms.iter().any(|term| term.factors.is_empty()) {
                continue;
            }
            // Every term costs at least one reducer step, so a region with more
            // terms than the work it replaces can never pay. Reject it before
            // paying for the grouping.
            if terms.len() as u64 > replaced_work(&old, root, &scalar, &uses) {
                queue.extend(distribute::interior_reductions(
                    &old, root, &scalar, reducer,
                ));
                continue;
            }
            let mut candidate = Region {
                root,
                reducer,
                product,
                wrappers: terms.iter().map(|term| term.wrappers.clone()).collect(),
                terms: terms.into_iter().map(|term| term.factors).collect(),
            };
            let groups = group_terms(&old, &mut candidate, &shapes, &assigned, &pools);
            let payable = groups.as_ref().is_some_and(|groups| {
                profitable(&old, &candidate, groups, &shapes, &scalar, &uses)
            });
            if payable {
                accepted.insert(root, (regions.len(), groups.expect("grouped")));
                regions.push(candidate);
            } else {
                queue.extend(distribute::interior_reductions(
                    &old, root, &scalar, reducer,
                ));
            }
        }
    }
    if accepted.is_empty() {
        return Ok(false);
    }

    for node in &old.nodes {
        if let [result] = node.results.as_slice()
            && let Some((index, groups)) = accepted.get(result)
        {
            let dtype = old.value_shape(*result).dtype;
            let reducer = module_reducer(module, regions[*index].reducer, dtype)?;
            let replacement = emit_region(
                &mut rebuilt,
                &old,
                &regions[*index],
                groups,
                &shapes,
                &assigned,
                &pools,
                reducer,
            )?;
            rebuilt.alias(&[*result], &[replacement])?;
            continue;
        }
        let inputs = rebuilt.mapped_inputs(node)?;
        rebuilt.emit(&old, &node.results, node.op.clone(), inputs)?;
    }
    module.funcs[module.entry.index()] = rebuilt.finish(&old)?;
    Ok(true)
}

/// Accepts a recovery only when it is cheaper to *run*, not merely smaller to
/// print.
///
/// This is the distinction distribution makes it easy to lose. The extractor
/// factored the program precisely so that a shared subexpression is evaluated
/// once; distributing multiplies that subexpression back out across every term
/// that contained it. The recovered XIR still looks small — a handful of gathers
/// and one loop — but the loop body runs once per term, in sequence, so the
/// operations the thread actually performs can be many times what the factored
/// cone performed. Counting nodes cannot see that; only counting executed
/// operations can.
///
/// So a region is recovered only when distribution did not invent work: when the
/// factored form was already a wide sum of distinct terms with nothing to share.
/// Anything the extractor genuinely factored is left exactly as it wrote it.
fn profitable(
    func: &Func,
    region: &Region,
    groups: &[Group],
    shapes: &Shapes,
    scalar: &[bool],
    uses: &[usize],
) -> bool {
    let executed = recovered_work(groups, shapes);
    let factored = replaced_work(func, region.root, scalar, uses);
    let distinct = region.terms.iter().collect::<HashSet<_>>().len();
    // One reducer step is inherent, however many batches the region takes:
    // `emit_region` seeds a single accumulator with the identity, where the
    // factored tree started from its first term. That is the only allowance —
    // anything beyond it is work distribution invented.
    distinct > 1
        && factored > 0
        && executed <= factored + 1
        && emitted_size(groups, shapes) < factored
}

/// Operations one thread performs to evaluate the recovered region: every term
/// replays its atoms, joins them with the product, applies its wrappers, and
/// takes one reducer step. The backend walks the terms with a loop, so this is a
/// sequential count, not a parallel one.
fn recovered_work(groups: &[Group], shapes: &Shapes) -> u64 {
    groups
        .iter()
        .map(|group| {
            let per_term = group
                .key
                .atoms
                .iter()
                .map(|&atom| shapes.size(atom))
                .sum::<u64>()
                + (group.key.atoms.len() as u64).saturating_sub(1)
                + group.key.wrappers.len() as u64
                + 1;
            (group.terms.len() as u64).saturating_mul(per_term)
        })
        .sum()
}

/// Nodes the recovered region adds to the program: the gathers and their index
/// vectors, each batch's operations once, and the scatter.
fn emitted_size(groups: &[Group], shapes: &Shapes) -> u64 {
    groups
        .iter()
        .map(|group| {
            let replay = group
                .key
                .atoms
                .iter()
                .map(|&atom| shapes.size(atom))
                .sum::<u64>();
            2 * group.key.pools.len() as u64
                + replay
                + (group.key.atoms.len() as u64).saturating_sub(1)
                + group.key.wrappers.len() as u64
                + 3
        })
        .sum()
}

/// Operations recovery actually removes.
///
/// Not every node under the root disappears when the root is replaced. A value
/// the cone shares with a consumer outside it still has to be computed, so
/// counting the whole cone credits recovery with work it does not delete and
/// lets a region look cheaper than it is. Only values whose every use lies
/// inside the cone are really replaced.
fn replaced_work(func: &Func, root: ValueId, scalar: &[bool], uses: &[usize]) -> u64 {
    let mut inside = vec![false; func.values.len()];
    let mut cone = Vec::new();
    let mut stack = vec![root];
    while let Some(value) = stack.pop() {
        if !scalar[value.index()] || std::mem::replace(&mut inside[value.index()], true) {
            continue;
        }
        cone.push(value);
        if let Some(node) = producer(func, value) {
            stack.extend(node.inputs.iter().copied());
        }
    }
    let mut from_inside = vec![0usize; func.values.len()];
    for &value in &cone {
        if let Some(node) = producer(func, value) {
            for &input in &node.inputs {
                from_inside[input.index()] += 1;
            }
        }
    }
    // Being read from outside retains a value, and retaining it retains
    // everything it reads: those operands still have to be computed for it. Only
    // the root escapes this, because every one of its uses is aliased to the
    // recovered value. Without the propagation a value one level below an escape
    // looks privately owned and is wrongly counted as deleted.
    let mut retained = vec![false; func.values.len()];
    let mut frontier = cone
        .iter()
        .copied()
        .filter(|&value| value != root && uses[value.index()] > from_inside[value.index()])
        .collect::<Vec<_>>();
    for &value in &frontier {
        retained[value.index()] = true;
    }
    while let Some(value) = frontier.pop() {
        let Some(node) = producer(func, value) else {
            continue;
        };
        for &input in &node.inputs {
            if inside[input.index()] && !std::mem::replace(&mut retained[input.index()], true) {
                frontier.push(input);
            }
        }
    }
    cone.iter()
        .filter(|&&value| !retained[value.index()])
        .count() as u64
}

#[allow(clippy::too_many_arguments)]
fn emit_region(
    rebuilt: &mut Rebuilder,
    old: &Func,
    region: &Region,
    groups: &[Group],
    shapes: &Shapes,
    assigned: &[Option<ShapeId>],
    pools: &LeafPools,
    reducer: FuncId,
) -> Result<ValueId, String> {
    let output_shape = old.value_shape(region.root);
    let flat_shape = output_shape.with_dims([1]);
    let identity = reduction_identity(region.reducer, &flat_shape)
        .ok_or_else(|| format!("reducer {:?} has no identity", region.reducer))?;
    let mut accumulator =
        rebuilt.emit_fresh(flat_shape.clone(), Constant { literal: identity }, []);
    for group in groups {
        let lanes = group.terms.len();
        let lane_shape = output_shape.with_dims([lanes as i64]);

        // One gather per leaf position: each lane's choice of operand.
        let mut gathered = Vec::with_capacity(group.key.pools.len());
        for position in 0..group.key.pools.len() {
            let indices = group
                .leaves
                .iter()
                .map(|leaves| {
                    pools
                        .location(leaves[position])
                        .map(|location| location.element)
                        .ok_or_else(|| "a batched leaf lost its pool".to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            let index = index_vector(rebuilt, &indices, &lane_shape.domain_axes)?;
            let leaf_shape =
                lane_shape.with_dtype(old.value_shape(group.leaves[0][position]).dtype);
            gathered.push(rebuilt.emit_fresh(
                leaf_shape,
                take_vector(),
                [pools.value(group.key.pools[position]), index],
            ));
        }

        // Replay each atom over the lane axis, then join them with the product.
        let mut next_leaf = 0usize;
        let mut updates = None;
        for &atom in &region.terms[group.terms[0]] {
            let value = replay(
                rebuilt,
                old,
                atom,
                shapes,
                assigned,
                &gathered,
                &mut next_leaf,
                &lane_shape,
            )?;
            updates = Some(match updates {
                None => value,
                Some(left) => {
                    let op = scalar_op(region.product)
                        .ok_or_else(|| format!("product {:?} has no operation", region.product))?;
                    rebuilt.emit_fresh_op(lane_shape.clone(), op.into(), [left, value])
                }
            });
        }
        let mut updates = updates.ok_or_else(|| "a batched term has no atoms".to_string())?;
        for &wrapper in &region.wrappers[group.terms[0]] {
            let op = scalar_op(wrapper)
                .ok_or_else(|| format!("wrapper {wrapper:?} has no operation"))?;
            updates = rebuilt.emit_fresh_op(lane_shape.clone(), op.into(), [updates]);
        }
        let destinations = index_vector(rebuilt, &vec![0; lanes], &lane_shape.domain_axes)?;
        accumulator = rebuilt.emit_fresh(
            flat_shape.clone(),
            Scatter {
                dnums: ScatterDimensionNumbers {
                    update_window_dims: SmallVec::new(),
                    inserted_window_dims: smallvec![0],
                    input_batching_dims: SmallVec::new(),
                    scatter_indices_batching_dims: SmallVec::new(),
                    scatter_dims_to_operand_dims: smallvec![0],
                    index_vector_dim: 1,
                },
                indices_are_sorted: false,
                // Many lanes land on one element. This is what lets the backend
                // walk them with a loop instead of naming every lane.
                unique_indices: false,
                update: reducer,
            },
            [accumulator, destinations, updates],
        );
    }
    Ok(rebuilt.emit_fresh(
        output_shape.clone(),
        Reshape {
            out_dims: output_shape.dims.clone(),
        },
        [accumulator],
    ))
}

/// Rebuilds one atom over the lane axis, reading each leaf position from the
/// gather that supplies it. The program holds the atom once, however many terms
/// share it.
#[allow(clippy::too_many_arguments)]
fn replay(
    rebuilt: &mut Rebuilder,
    old: &Func,
    value: ValueId,
    shapes: &Shapes,
    assigned: &[Option<ShapeId>],
    gathered: &[ValueId],
    next_leaf: &mut usize,
    lane_shape: &Shape,
) -> Result<ValueId, String> {
    let shape = assigned[value.index()]
        .ok_or_else(|| format!("value %{} has no recovered shape", value.index()))?;
    if shapes.is_leaf(shape) {
        let gathered = *gathered
            .get(*next_leaf)
            .ok_or_else(|| "batched replay ran past its gathered leaves".to_string())?;
        *next_leaf += 1;
        return Ok(gathered);
    }
    let node = producer(old, value)
        .ok_or_else(|| format!("value %{} has no producer to replay", value.index()))?;
    let mut inputs = Vec::with_capacity(node.inputs.len());
    for &input in &node.inputs {
        inputs.push(replay(
            rebuilt, old, input, shapes, assigned, gathered, next_leaf, lane_shape,
        )?);
    }
    let result_shape = lane_shape.with_dtype(old.value_shape(node.results[0]).dtype);
    Ok(rebuilt.emit_fresh_op(result_shape, node.op.clone(), inputs))
}

fn module_reducer(module: &mut Module, scalar: ScalarOp, dtype: DType) -> Result<FuncId, String> {
    if let Some(found) = module.funcs.iter().enumerate().find_map(|(index, func)| {
        let [left, right] = func.params.as_slice() else {
            return None;
        };
        let [node] = func.nodes.as_slice() else {
            return None;
        };
        let [result] = func.results.as_slice() else {
            return None;
        };
        (node.op.as_elementwise().map(|op| op.scalar()) == Some(scalar)
            && node.inputs.as_slice() == [*left, *right]
            && node.results.as_slice() == [*result]
            && func.value_shape(*left) == &Shape::scalar(dtype)
            && func.value_shape(*right) == &Shape::scalar(dtype)
            && func.value_shape(*result) == &Shape::scalar(dtype))
            .then_some(FuncId(index as u32))
    }) {
        return Ok(found);
    }
    if module.entry.index() + 1 != module.funcs.len() {
        return Err("map/reduce recovery requires the entry function to be last".into());
    }
    let op = scalar_op(scalar).ok_or_else(|| format!("reducer {scalar:?} has no operation"))?;
    let name = format!("indexed_reduce_{}_{}", op.mnemonic(), dtype.name());
    let entry = module.funcs.pop().expect("module has an entry function");
    let reducer = FuncId(module.funcs.len() as u32);
    let shape = Shape::scalar(dtype);
    let left = ValueId(0);
    let right = ValueId(1);
    let result = ValueId(2);
    module.funcs.push(Func {
        name,
        params: vec![left, right],
        nodes: vec![Node {
            op: op.into(),
            inputs: smallvec![left, right],
            results: smallvec![result],
        }],
        results: vec![result],
        values: vec![
            ValueDef {
                shape: shape.clone(),
                def: Def::Param(0),
            },
            ValueDef {
                shape: shape.clone(),
                def: Def::Param(1),
            },
            ValueDef {
                shape,
                def: Def::NodeResult { node: 0, slot: 0 },
            },
        ],
    });
    module.funcs.push(entry);
    module.entry = FuncId(module.funcs.len() as u32 - 1);
    Ok(reducer)
}

#[cfg(test)]
mod tests;
