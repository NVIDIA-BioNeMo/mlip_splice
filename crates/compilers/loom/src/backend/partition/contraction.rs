// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Folding a single-use product into the sum that consumes it.
//!
//! Contraction is a rebuild, not an edit: an operation in the arena is what it
//! is, so a contracted body is built beside the one it came from and the
//! bindings that follow read the values the rebuild produced. What the rebuild
//! does not reach stays unreferenced, and the slice at the end is what decides
//! which of it the body runs.

use gpu_ir::{BinOp, Builder, Carry, Function, OpId, OpKind, Scalar, Ty, ValTy, Value, ValueId};

use super::{BTreeMap, slice_bindings};

/// How many times each place is read, counting every edge — a place read twice
/// by one operand cone is read twice.
fn count_reads(f: &Function, value: ValueId, uses: &mut BTreeMap<ValueId, usize>) {
    match f.operand_op(value) {
        None => *uses.entry(value).or_default() += 1,
        Some(op) => {
            for index in 0..f.op(op).operands.len() {
                count_reads(f, f.op(op).operands[index], uses);
            }
            for index in 0..f.op(op).regions.len() {
                let region = f.op(op).regions[index];
                for slot in 0..f.region(region).blocks.len() {
                    let block = f.region(region).blocks[slot];
                    for step in 0..f.block(block).ops.len() {
                        let inner = f.block(block).ops[step];
                        for operand in 0..f.op(inner).operands.len() {
                            count_reads(f, f.op(inner).operands[operand], uses);
                        }
                    }
                    for yielded in 0..f.block(block).terminator.yields.len() {
                        count_reads(f, f.block(block).terminator.yields[yielded], uses);
                    }
                }
            }
        }
    }
}

/// One product a sum may absorb: what it was bound as, and its two factors.
struct Product {
    ty: ValTy,
    left: ValueId,
    right: ValueId,
}

pub(in super::super) fn contract_fma_bindings(
    b: &mut Builder,
    bindings: Vec<OpId>,
    roots: &mut [Value],
) -> Result<Vec<OpId>, String> {
    let f = b.function();
    // A multi-binding rebuild may make a later binding refer into an earlier
    // copied region. Until regions are cloned with a scope-local value map,
    // leave that body unchanged rather than creating cross-region references.
    let has_region = bindings.len() > 1
        && bindings.iter().any(|&op| {
            fn visit(
                f: &Function,
                value: ValueId,
                seen: &mut std::collections::BTreeSet<ValueId>,
            ) -> bool {
                if !seen.insert(value) {
                    return false;
                }
                let Some(op) = f.operand_op(value) else {
                    return false;
                };
                if !f.op(op).regions.is_empty() {
                    return true;
                }
                f.op(op).operands.iter().any(|&v| visit(f, v, seen))
            }
            visit(
                f,
                f.op(op).operands[0],
                &mut std::collections::BTreeSet::new(),
            )
        });
    if has_region {
        return Ok(bindings);
    }
    let mut uses = BTreeMap::new();
    for &op in &bindings {
        if !matches!(f.op(op).kind, OpKind::Bind) {
            return Err("loom FMA contraction expects flat bindings".into());
        }
        count_reads(f, f.op(op).operands[0], &mut uses);
    }
    let root_values = roots.iter().map(|root| root.id_in(f)).collect::<Vec<_>>();
    for &root in &root_values {
        count_reads(f, root, &mut uses);
    }

    let mut types = BTreeMap::new();
    let mut products = BTreeMap::new();
    for &op in &bindings {
        let held = f.op(op);
        let (value, init) = (held.results[0], held.operands[0]);
        let ty = f.ty(value);
        types.insert(value, ty);
        let ValTy::Data(data) = ty else { continue };
        let Some(product) = f.operand_op(init) else {
            continue;
        };
        if !matches!(f.op(product).kind, OpKind::Bin(BinOp::Mul))
            || !data.scalar.is_float()
            || uses.get(&value) != Some(&1)
        {
            continue;
        }
        products.insert(
            value,
            Product {
                ty,
                left: f.op(product).operands[0],
                right: f.op(product).operands[1],
            },
        );
    }

    let mut remap = BTreeMap::new();
    let mut contracted = Vec::with_capacity(bindings.len());
    for op in bindings {
        let f = b.function();
        let held = f.op(op);
        let (value, init) = (held.results[0], held.operands[0]);
        let ty = f.ty(value);
        let name = f.name(value).map(str::to_owned);
        let rebuilt = contract(b, init, ty, &products, &types, &mut remap)?;
        let bound = match name {
            Some(name) => match ty {
                ValTy::Data(ty) => b.bind_value_as(name, ty, rebuilt),
                ValTy::Ptr(ty) => b.bind_pointer_as(name, ty, rebuilt),
            },
            None => match ty {
                ValTy::Data(ty) => b.bind_value(ty, rebuilt),
                ValTy::Ptr(ty) => b.bind_pointer(ty, rebuilt),
            },
        };
        remap.insert(value, bound.id_in(b.function()));
        contracted.push(b.defining_op(&bound).expect("a binding is an operation"));
    }

    let mut rebuilt_roots = Vec::with_capacity(root_values.len());
    for root in root_values {
        let ty = expression_ty(b.function(), root, &types);
        let value = match ty {
            Some(ty) => contract(b, root, ty, &products, &types, &mut remap)?,
            None => substituted(b, root, &mut remap)?,
        };
        rebuilt_roots.push(value);
    }
    for (root, rebuilt) in roots.iter_mut().zip(rebuilt_roots) {
        *root = rebuilt;
    }

    let f = b.function();
    let root_ids = roots.iter().map(|root| root.id_in(f)).collect::<Vec<_>>();
    slice_bindings(f, &contracted, &root_ids)
}

/// A value with every rebuilt binding read in place of the one it replaced, and
/// nothing else changed.
fn substituted(
    b: &mut Builder,
    value: ValueId,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<Value, String> {
    if let Some(&held) = remap.get(&value) {
        return Ok(b.value_of(held));
    }
    let f = b.function();
    let Some(op) = f.operand_op(value) else {
        return Ok(b.value_of(value));
    };
    if !f.op(op).regions.is_empty() {
        return rebuild_loop(b, op, value, None, remap);
    }
    if stands_as_it_is(f, value, remap) {
        return Ok(b.value_of(value));
    }
    let operands = f.op(op).operands.clone();
    let mut rebuilt = Vec::with_capacity(operands.len());
    for operand in operands {
        rebuilt.push(substituted(b, operand, remap)?);
    }
    Ok(rebuild(b, op, value, rebuilt))
}

/// Whether an expression reaches nothing this rebuild replaced, so it stands as
/// it is rather than being built again beside itself.
///
/// An operation owning a region never does: what its body reads is not part of
/// its operand cone, so only rebuilding it can say.
fn stands_as_it_is(f: &Function, value: ValueId, remap: &BTreeMap<ValueId, ValueId>) -> bool {
    if remap.contains_key(&value) {
        return false;
    }
    match f.operand_op(value) {
        None => true,
        Some(op) => {
            f.op(op).regions.is_empty()
                && f.op(op)
                    .operands
                    .iter()
                    .all(|&operand| stands_as_it_is(f, operand, remap))
        }
    }
}

/// `value` with every `a * b + c` whose product this body binds for one use
/// folded into one `fma`.
#[allow(clippy::too_many_lines)]
fn contract(
    b: &mut Builder,
    value: ValueId,
    expected: ValTy,
    products: &BTreeMap<ValueId, Product>,
    types: &BTreeMap<ValueId, ValTy>,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<Value, String> {
    if let Some(&held) = remap.get(&value) {
        return Ok(b.value_of(held));
    }
    let f = b.function();
    let Some(op) = f.operand_op(value) else {
        return Ok(b.value_of(value));
    };
    let kind = f.op(op).kind.clone();
    let operands = f.op(op).operands.clone();
    // A loop standing for the value it reduces to is rebuilt through its region:
    // its body is an expression like any other, and a product written inside it
    // contracts where it stands.
    if !f.op(op).regions.is_empty() {
        return rebuild_loop(b, op, value, Some((products, types)), remap);
    }
    fn recurse(
        b: &mut Builder,
        operand: ValueId,
        expected: ValTy,
        products: &BTreeMap<ValueId, Product>,
        types: &BTreeMap<ValueId, ValTy>,
        remap: &mut BTreeMap<ValueId, ValueId>,
    ) -> Result<Value, String> {
        contract(b, operand, expected, products, types, remap)
    }
    Ok(match kind {
        OpKind::Un(unop) => {
            let inner = recurse(b, operands[0], expected, products, types, remap)?;
            b.un(unop, inner)
        }
        OpKind::Bin(binop @ (BinOp::Add | BinOp::Sub)) if matches!(expected, ValTy::Data(ty) if ty.scalar.is_float()) =>
        {
            let negate = binop == BinOp::Sub;
            if let Some((left, right)) = product(b, operands[0], expected, products, types, remap)?
            {
                let addend = recurse(b, operands[1], expected, products, types, remap)?;
                let addend = if negate { b.neg(addend) } else { addend };
                return Ok(b.fma(left, right, addend));
            }
            if let Some((left, right)) = product(b, operands[1], expected, products, types, remap)?
            {
                let left = if negate { b.neg(left) } else { left };
                let addend = recurse(b, operands[0], expected, products, types, remap)?;
                return Ok(b.fma(left, right, addend));
            }
            let left = recurse(b, operands[0], expected, products, types, remap)?;
            let right = recurse(b, operands[1], expected, products, types, remap)?;
            b.bin(binop, left, right)
        }
        OpKind::Bin(binop) => {
            let left = recurse(b, operands[0], expected, products, types, remap)?;
            let right = recurse(b, operands[1], expected, products, types, remap)?;
            b.bin(binop, left, right)
        }
        OpKind::Fma => {
            let left = recurse(b, operands[0], expected, products, types, remap)?;
            let right = recurse(b, operands[1], expected, products, types, remap)?;
            let addend = recurse(b, operands[2], expected, products, types, remap)?;
            b.fma(left, right, addend)
        }
        OpKind::Select => {
            let pred = substituted(b, operands[0], remap)?;
            let taken = recurse(b, operands[1], expected, products, types, remap)?;
            let other = recurse(b, operands[2], expected, products, types, remap)?;
            b.select(pred, taken, other)
        }
        OpKind::Cast(scalar) => {
            let input = expression_ty(b.function(), operands[0], types).unwrap_or(expected);
            let inner = contract(b, operands[0], input, products, types, remap)?;
            b.cast(scalar, inner)
        }
        OpKind::Splat(lanes) => {
            let input = match expected {
                ValTy::Data(ty) => ValTy::Data(Ty::scalar(ty.scalar)),
                ty => ty,
            };
            let inner = contract(b, operands[0], input, products, types, remap)?;
            b.splat(lanes, inner)
        }
        OpKind::Extract(lane) => {
            let input = expression_ty(b.function(), operands[0], types).unwrap_or(expected);
            let inner = contract(b, operands[0], input, products, types, remap)?;
            b.extract(lane, inner)
        }
        // A pack is its parts end to end, and a part carries its own width: a
        // vector part contracts as a vector, not as one lane of one.
        OpKind::Pack => {
            let mut parts = Vec::with_capacity(operands.len());
            for operand in operands {
                let held = expression_ty(b.function(), operand, types)
                    .unwrap_or_else(|| b.function().ty(operand));
                parts.push(contract(b, operand, held, products, types, remap)?);
            }
            b.pack(parts)
        }
        OpKind::ShflDown => {
            let inner = recurse(b, operands[0], expected, products, types, remap)?;
            let delta = contract(
                b,
                operands[1],
                ValTy::Data(Ty::scalar(Scalar::S32)),
                products,
                types,
                remap,
            )?;
            b.shfl_down(inner, delta)
        }
        _ => substituted(b, value, remap)?,
    })
}

/// What a rebuild does to the expressions it copies: contract products into
/// fused multiply-adds, or leave them as they stand.
type Contracting<'a> = Option<(&'a BTreeMap<ValueId, Product>, &'a BTreeMap<ValueId, ValTy>)>;

/// One expression, rebuilt under `contracting`.
fn rebuilt(
    b: &mut Builder,
    value: ValueId,
    contracting: Contracting<'_>,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<Value, String> {
    match contracting {
        Some((products, types)) => {
            let expected =
                expression_ty(b.function(), value, types).unwrap_or_else(|| b.function().ty(value));
            contract(b, value, expected, products, types, remap)
        }
        None => substituted(b, value, remap),
    }
}

/// Rebuilds a loop standing for the value it reduces to, region and all.
///
/// Its bounds and starting values are expressions here; its body is one in a
/// scope of its own, whose index and carried values stand for the source's.
/// Copying that scope is what carries a substitution — or a product the update
/// absorbs — through a fold rather than around it.
fn rebuild_loop(
    b: &mut Builder,
    op: OpId,
    value: ValueId,
    contracting: Contracting<'_>,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<Value, String> {
    let f = b.function();
    let OpKind::Loop { index, trip } = f.op(op).kind else {
        return Err("loom FMA contraction expects a loop to own the region".into());
    };
    let ordinal = f
        .op(op)
        .results
        .iter()
        .position(|&result| result == value)
        .ok_or_else(|| "loom FMA contraction lost which result it rebuilds".to_string())?;
    let operands = f.op(op).operands.clone();
    let into = f.op(op).into.clone();
    let tys = f
        .op(op)
        .results
        .iter()
        .map(|&result| f.data_ty(result))
        .collect::<Vec<_>>();
    let block = f.region_block(f.op(op).regions[0]);
    let params = f.block(block).params.clone();
    let steps = f.block_steps(block).collect::<Vec<_>>();
    let yields = f.block(block).terminator.yields.clone();

    let start = rebuilt(b, operands[0], contracting, remap)?;
    let stop = rebuilt(b, operands[1], contracting, remap)?;
    let step = rebuilt(b, operands[2], contracting, remap)?;
    let mut carried = Vec::with_capacity(tys.len());
    for (slot, &ty) in tys.iter().enumerate() {
        let init = rebuilt(b, operands[3 + slot], contracting, remap)?;
        let place = match into.get(slot).copied().flatten() {
            Some(place) => Some(substituted(b, place, remap)?),
            None => None,
        };
        carried.push(Carry {
            ty,
            init,
            into: place,
        });
    }

    let open = b.open_loop_body(index, carried);
    let bound = std::iter::once(open.index().clone())
        .chain(open.given().iter().cloned())
        .collect::<Vec<_>>();
    for (&param, held) in params.iter().zip(&bound) {
        let held = held.id_in(b.function());
        remap.insert(param, held);
    }
    let built = (|b: &mut Builder| {
        for step in steps {
            rebuild_step(b, step, contracting, remap)?;
        }
        let mut produced = Vec::with_capacity(yields.len());
        for yielded in yields {
            produced.push(rebuilt(b, yielded, contracting, remap)?);
        }
        Ok::<_, String>(produced)
    })(b);
    let body = b.close_loop_body(open, built?);
    let results = b.fold_loop(trip, start, stop, step, body);
    // One operation, rebuilt once. Every result it produces stands for the one
    // opposite it from here on, so a sibling asked for later reads this copy
    // rather than building a second.
    let f = b.function();
    let sources = f.op(op).results.clone();
    for (&source, held) in sources.iter().zip(&results) {
        let held = held.id_in(f);
        remap.insert(source, held);
    }
    Ok(results
        .into_iter()
        .nth(ordinal)
        .expect("one value per carried slot"))
}

/// Rebuilds one step of a region being copied, and records what its results
/// stand for so the steps after it read them.
fn rebuild_step(
    b: &mut Builder,
    op: OpId,
    contracting: Contracting<'_>,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<(), String> {
    let f = b.function();
    let kind = f.op(op).kind.clone();
    let operands = f.op(op).operands.clone();
    let results = f.op(op).results.clone();
    let into = f.op(op).into.clone();
    match kind {
        OpKind::Bind => {
            let ty = f.ty(results[0]);
            let name = f.name(results[0]).map(str::to_owned);
            let place = into.first().copied().flatten();
            let init = rebuilt(b, operands[0], contracting, remap)?;
            // A binding produced into a place is produced into the rebuilt one:
            // dropping the destination would give the copy storage of its own
            // and leave the accumulator it was updating unwritten.
            let bound = match place {
                Some(place) => {
                    let place = substituted(b, place, remap)?;
                    b.update(place, init)
                }
                None => match (name.clone(), ty) {
                    (Some(name), ValTy::Data(ty)) => b.bind_value_as(name, ty, init),
                    (Some(name), ValTy::Ptr(ty)) => b.bind_pointer_as(name, ty, init),
                    (None, ValTy::Data(ty)) => b.bind_value(ty, init),
                    (None, ValTy::Ptr(ty)) => b.bind_pointer(ty, init),
                },
            };
            if place.is_some()
                && let Some(name) = name
            {
                b.name_value(&bound, name);
            }
            let bound = bound.id_in(b.function());
            remap.insert(results[0], bound);
            Ok(())
        }
        OpKind::If => {
            let f = b.function();
            let arms = f.op(op).regions.clone();
            let tys = results
                .iter()
                .map(|&result| f.data_ty(result))
                .collect::<Vec<_>>();
            let predicate = rebuilt(b, operands[0], contracting, remap)?;
            let mut built = Vec::with_capacity(arms.len());
            for arm in arms {
                let block = b.function().region_block(arm);
                let steps = b.function().block_steps(block).collect::<Vec<_>>();
                let yields = b.function().block(block).terminator.yields.clone();
                b.open_arm();
                let produced = (|b: &mut Builder| {
                    for step in steps {
                        rebuild_step(b, step, contracting, remap)?;
                    }
                    let mut produced = Vec::with_capacity(yields.len());
                    for yielded in yields {
                        produced.push(rebuilt(b, yielded, contracting, remap)?);
                    }
                    Ok::<_, String>(produced)
                })(b);
                built.push(b.close_arm(produced?));
            }
            let mut arms = built.into_iter();
            let (then_, else_) = (
                arms.next().expect("a then arm"),
                arms.next().expect("an else arm"),
            );
            let mut places = Vec::with_capacity(tys.len());
            for (slot, &ty) in tys.iter().enumerate() {
                let place = match into.get(slot).copied().flatten() {
                    Some(place) => Some(substituted(b, place, remap)?),
                    None => None,
                };
                places.push((ty, place));
            }
            let produced = b.if_arms(predicate, &places, then_, else_);
            for (&result, held) in results.iter().zip(&produced) {
                let held = held.id_in(b.function());
                remap.insert(result, held);
            }
            Ok(())
        }
        // A region holding a write or a jump is not an expression, and this
        // rebuilds expressions.
        kind => Err(format!(
            "loom FMA contraction cannot rebuild {kind:?} inside a region"
        )),
    }
}

/// The two factors of a product this sum may absorb: a binding the body holds
/// for one use, or a multiply written in place. A negation moves onto the left
/// factor, where it costs nothing.
fn product(
    b: &mut Builder,
    value: ValueId,
    expected: ValTy,
    products: &BTreeMap<ValueId, Product>,
    types: &BTreeMap<ValueId, ValTy>,
    remap: &mut BTreeMap<ValueId, ValueId>,
) -> Result<Option<(Value, Value)>, String> {
    let f = b.function();
    let (value, negate) = match f.operand_op(value) {
        Some(op) if matches!(f.op(op).kind, OpKind::Un(gpu_ir::UnOp::Neg)) => {
            (f.op(op).operands[0], true)
        }
        _ => (value, false),
    };
    let (left, right) = match f.operand_op(value) {
        Some(op) if matches!(f.op(op).kind, OpKind::Bin(BinOp::Mul)) => {
            (f.op(op).operands[0], f.op(op).operands[1])
        }
        Some(_) => return Ok(None),
        None => {
            let Some(product) = products.get(&value) else {
                return Ok(None);
            };
            if product.ty != expected {
                return Ok(None);
            }
            (product.left, product.right)
        }
    };
    let left = contract(b, left, expected, products, types, remap)?;
    let left = if negate { b.neg(left) } else { left };
    let right = contract(b, right, expected, products, types, remap)?;
    Ok(Some((left, right)))
}

/// Rebuilds one region-free operand operation over already-rebuilt operands.
///
/// `value` is the result being rebuilt, so a kind with more than one of them
/// says which comes back.
fn rebuild(b: &mut Builder, op: OpId, value: ValueId, operands: Vec<Value>) -> Value {
    let kind = b.function().op(op).kind.clone();
    let mut operands = operands.into_iter();
    let mut next = || operands.next().expect("the operand count is the kind's");
    match kind {
        OpKind::Un(unop) => {
            let a = next();
            b.un(unop, a)
        }
        OpKind::Bin(binop) => {
            let (x, y) = (next(), next());
            b.bin(binop, x, y)
        }
        OpKind::Fma => {
            let (x, y, z) = (next(), next(), next());
            b.fma(x, y, z)
        }
        OpKind::Cmp(cmpop) => {
            let (x, y) = (next(), next());
            b.cmp(cmpop, x, y)
        }
        OpKind::Select => {
            let (p, x, y) = (next(), next(), next());
            b.select(p, x, y)
        }
        OpKind::Cast(scalar) => {
            let a = next();
            b.cast(scalar, a)
        }
        OpKind::Splat(lanes) => {
            let a = next();
            b.splat(lanes, a)
        }
        OpKind::Extract(lane) => {
            let a = next();
            b.extract(lane, a)
        }
        OpKind::Pack => {
            let lanes = operands.collect::<Vec<_>>();
            b.pack(lanes)
        }
        OpKind::ShflDown => {
            let (a, delta) = (next(), next());
            b.shfl_down(a, delta)
        }
        OpKind::WarpAny => {
            let a = next();
            b.warp_any(a)
        }
        OpKind::Offset => {
            let (pointer, elements) = (next(), next());
            b.offset(pointer, elements)
        }
        OpKind::Load { lanes, nc, evict } => {
            let pointer = next();
            b.load_evict(pointer, lanes, nc, evict)
        }
        OpKind::ConstLoad(array) => {
            let index = next();
            b.const_load_shared(array.name.clone(), array.scalar, array.data.clone(), index)
        }
        // A constant and a special register read nothing, so they come through
        // as they are.
        _ => b.value_of(value),
    }
}

/// The type a value carries, as far as the bindings this body holds decide it.
fn expression_ty(f: &Function, value: ValueId, types: &BTreeMap<ValueId, ValTy>) -> Option<ValTy> {
    if f.is_materialized(value) {
        return types.get(&value).copied();
    }
    let op = f.operand_op(value)?;
    let operands = &f.op(op).operands;
    match f.op(op).kind {
        OpKind::IntConst { scalar, .. } | OpKind::FloatConst { scalar, .. } => {
            Some(ValTy::Data(Ty::scalar(scalar)))
        }
        OpKind::Un(_) => expression_ty(f, operands[0], types),
        OpKind::Bin(_) => match (
            expression_ty(f, operands[0], types),
            expression_ty(f, operands[1], types),
        ) {
            (Some(a), Some(b)) => (a == b).then_some(a),
            (Some(ty), None) | (None, Some(ty)) => Some(ty),
            (None, None) => None,
        },
        OpKind::Fma => {
            let known = [
                expression_ty(f, operands[0], types),
                expression_ty(f, operands[1], types),
                expression_ty(f, operands[2], types),
            ];
            let ty = known.iter().flatten().next().copied()?;
            known
                .into_iter()
                .flatten()
                .all(|known| known == ty)
                .then_some(ty)
        }
        OpKind::Cast(scalar) => expression_ty(f, operands[0], types)
            .map(ValTy::data)
            .map(|ty| ValTy::Data(Ty::vec(scalar, ty.lanes))),
        OpKind::Splat(lanes) => expression_ty(f, operands[0], types)
            .map(ValTy::data)
            .map(|ty| ValTy::Data(Ty::vec(ty.scalar, lanes))),
        OpKind::Extract(_) => expression_ty(f, operands[0], types)
            .map(ValTy::data)
            .map(|ty| ValTy::Data(Ty::scalar(ty.scalar))),
        _ => None,
    }
}
