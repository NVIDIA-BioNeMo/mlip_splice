// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which sums absorb the product beside them, and which are left alone.

use gpu_ir::{Builder, OpKind, Param, ParamTy, Scalar, Ty, UnOp, Value, ValueId};

use super::super::*;
use gpu_ir::Trip;

/// What a contracted body looks like, for a test that asks about its shape.
struct Contracted {
    body: gpu_ir::Function,
    ops: Vec<gpu_ir::OpId>,
    root: ValueId,
}

impl Contracted {
    fn kind(&self, value: ValueId) -> Option<&OpKind> {
        self.body.operand_op(value).map(|op| &self.body.op(op).kind)
    }

    fn operands(&self, value: ValueId) -> Vec<ValueId> {
        let op = self.body.operand_op(value).expect("an operand");
        self.body.op(op).operands.clone()
    }

    /// The value a binding of this name defines in the contracted body.
    ///
    /// A rebuild carries the name it came from, so the arena holds the original
    /// under it too; the contracted list is what says which one runs.
    fn named(&self, name: &str) -> ValueId {
        self.ops
            .iter()
            .map(|&op| self.body.op(op).results[0])
            .find(|&value| self.body.name(value) == Some(name))
            .unwrap_or_else(|| panic!("{name} is not bound"))
    }

    fn is_named(&self, value: ValueId, name: &str) -> bool {
        self.body.name(value) == Some(name)
    }

    fn init(&self, name: &str) -> ValueId {
        let value = self.named(name);
        let op = self.body.defining_op(value).expect("a binding");
        self.body.op(op).operands[0]
    }
}

/// `product = a * b` bound once, then `result` built from it by `build`.
fn contract(
    scalar: Scalar,
    lanes: u8,
    build: impl FnOnce(&mut Builder, Value, Value, Value) -> Value,
) -> Contracted {
    let mut b = Builder::new();
    let ty = Ty::vec(scalar, lanes);
    let declare = |b: &mut Builder, name: &str| {
        let value = b.parameter(Param {
            name: name.to_string(),
            ty: ParamTy::Scalar(scalar),
        });
        if lanes == 1 {
            value
        } else {
            b.splat(lanes, value)
        }
    };
    let a = declare(&mut b, "a");
    let second = declare(&mut b, "b");
    let c = declare(&mut b, "c");
    let raw = b.mul(a, second);
    let product = b.bind_value_as("product", ty, raw);
    let init = build(&mut b, product.clone(), c, product);
    let bound = b.bind_value_as("result", ty, init);
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();
    let [root] = roots;
    let root = root.id_in(b.function());
    Contracted {
        body: b.into_function(),
        ops,
        root,
    }
}

/// The four ways a product meets a sum, and where the negation lands.
#[test]
fn algebraic_fma_contraction_handles_add_and_subtract_forms() {
    let held = contract(Scalar::F32, 1, |b, product, c, _| b.add(product, c));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Fma)));
    let [a, second, addend] = held.operands(init)[..] else {
        panic!("an fma takes three")
    };
    assert!(held.is_named(a, "a") && held.is_named(second, "b") && held.is_named(addend, "c"));

    let held = contract(Scalar::F32, 1, |b, product, c, _| b.add(c, product));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Fma)));

    // `product - c` keeps the product and negates what it adds.
    let held = contract(Scalar::F32, 1, |b, product, c, _| b.sub(product, c));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Fma)));
    let addend = held.operands(init)[2];
    assert!(matches!(held.kind(addend), Some(OpKind::Un(UnOp::Neg))));

    // `c - product` negates the product's left factor instead.
    let held = contract(Scalar::F32, 1, |b, product, c, _| b.sub(c, product));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Fma)));
    let left = held.operands(init)[0];
    assert!(matches!(held.kind(left), Some(OpKind::Un(UnOp::Neg))));
    assert!(held.is_named(held.operands(init)[2], "c"));
}

/// Absorbing the product is what makes its binding dead, so the slice drops it
/// and the root names what the sum became.
#[test]
fn algebraic_fma_contraction_rewrites_roots_and_drops_product() {
    let held = contract(Scalar::F32, 1, |b, product, c, _| b.add(product, c));
    assert_eq!(held.ops.len(), 1, "only the sum survives");
    assert_eq!(held.root, held.named("result"));
    assert!(
        !held
            .ops
            .iter()
            .any(|&op| held.body.name(held.body.op(op).results[0]) == Some("product")),
        "the product binding is not placed"
    );
}

/// A product two sums read is not one either may absorb: contracting it would
/// compute it twice.
#[test]
fn algebraic_fma_contraction_preserves_shared_products() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let declare = |b: &mut Builder, name: &str| {
        b.parameter(Param {
            name: name.to_string(),
            ty: ParamTy::Scalar(Scalar::F32),
        })
    };
    let a = declare(&mut b, "a");
    let second = declare(&mut b, "b");
    let c = declare(&mut b, "c");
    let d = declare(&mut b, "d");
    let raw = b.mul(a, second);
    let product = b.bind_value_as("product", ty, raw);
    let first = b.add(product.clone(), c);
    let first = b.bind_value_as("first", ty, first);
    let second_sum = b.add(product, d);
    let second_sum = b.bind_value_as("second", ty, second_sum);
    let ops = b.steps();
    let mut roots = [first, second_sum];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();
    let f = b.function();
    assert_eq!(ops.len(), 3, "the product stays bound");
    for &op in &ops {
        assert!(
            !matches!(f.op(op).kind, OpKind::Bind)
                || !matches!(
                    f.operand_op(f.op(op).operands[0])
                        .map(|held| &f.op(held).kind),
                    Some(OpKind::Fma)
                ),
            "a shared product was contracted"
        );
    }
}

/// A product bound at one lane count cannot be absorbed by a sum of another:
/// the two are different values, whatever their scalars agree on.
#[test]
fn algebraic_fma_contraction_requires_matching_lane_types() {
    let mut b = Builder::new();
    let a = b.parameter(Param {
        name: "a".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let second = b.parameter(Param {
        name: "b".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let c = b.parameter(Param {
        name: "c".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let raw = b.mul(a, second);
    let product = b.bind_value_as("product", Ty::scalar(Scalar::F32), raw);
    let widened = b.splat(2, product);
    let addend = b.splat(2, c);
    let sum = b.add(widened, addend);
    let result = b.bind_value_as("result", Ty::vec(Scalar::F32, 2), sum);
    let ops = b.steps();
    let mut roots = [result];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();
    assert_eq!(ops.len(), 2, "a lane mismatch keeps the product bound");
}

/// A product written into the sum rather than bound beside it contracts too:
/// what matters is that nothing else reads it.
#[test]
fn an_inline_product_contracts_into_a_fused_multiply_add() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let a = b.parameter(Param {
        name: "a".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let second = b.parameter(Param {
        name: "b".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let c = b.parameter(Param {
        name: "c".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let product = b.mul(a, second);
    let sum = b.add(product, c);
    let result = b.bind_value_as("result", ty, sum);
    let ops = b.steps();
    let mut roots = [result];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();
    let f = b.function();
    assert_eq!(ops.len(), 1);
    let init = f.op(ops[0]).operands[0];
    assert!(matches!(
        f.operand_op(init).map(|op| &f.op(op).kind),
        Some(OpKind::Fma)
    ));
}

/// Packed lanes contract exactly as scalars do — the rewrite is over values, and
/// a vector is one value.
#[test]
fn packed_lanes_contract_their_multiply_adds() {
    let held = contract(Scalar::F32, 2, |b, product, c, _| b.add(product, c));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Fma)));
}

/// Integers have no fused multiply-add, so their sums keep both operations.
#[test]
fn integer_multiply_add_is_not_fused() {
    let held = contract(Scalar::S32, 1, |b, product, c, _| b.add(product, c));
    let init = held.init("result");
    assert!(matches!(held.kind(init), Some(OpKind::Bin(BinOp::Add))));
    assert_eq!(held.ops.len(), 2, "the product stays bound");
}

/// A fold is an expression like any other: the sum its update ends with absorbs
/// the product beside it, inside the loop body where it stands.
#[test]
fn a_fold_body_keeps_its_fused_multiply_add() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let lane = b.parameter(Param {
        name: "lane".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let weight = b.parameter(Param {
        name: "weight".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let zero = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![gpu_ir::Carry {
            ty,
            init: zero,
            into: None,
        }],
        |b, _, given| {
            let product = b.mul(lane.clone(), weight.clone());
            (vec![b.add(product, given[0].clone())], ())
        },
    );
    let (start, stop, step) = (b.i32(0), b.i32(32), b.i32(1));
    let folded = b.fold_loop(Trip::PerThread, start, stop, step, body);
    let bound = b.bind_value_as("result", ty, folded.into_iter().next().unwrap());
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();

    let f = b.function();
    assert_eq!(ops.len(), 1, "one binding holds the fold");
    let init = f.op(ops[0]).operands[0];
    let fold = f
        .operand_op(init)
        .filter(|&op| matches!(f.op(op).kind, OpKind::Loop { .. }))
        .expect("the fold was rewritten into something else");
    let block = f.region_block(f.op(fold).regions[0]);
    let yielded = f.block(block).terminator.yields[0];
    let update = f
        .operand_op(yielded)
        .expect("the update is an expression the body yields");
    assert!(
        matches!(f.op(update).kind, OpKind::Fma),
        "a fold body lost its fused multiply-add: {:?}",
        f.op(update).kind
    );
    let [x, y, z] = f.op(update).operands[..] else {
        panic!("an fma takes three")
    };
    assert_eq!(f.name(x), Some("lane"));
    assert_eq!(f.name(y), Some("weight"));
    assert_eq!(z, f.block(block).params[1], "the accumulator it carries");
}

/// A branch inside a fold body is rebuilt with it: the arm's own sum absorbs
/// the product beside it, the arm still produces into the value the loop
/// carries, and the loop hands back the result the reader asked for.
#[test]
fn a_guarded_fold_body_contracts_inside_its_arm() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let lane = b.parameter(Param {
        name: "lane".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let weight = b.parameter(Param {
        name: "weight".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let (first, second) = (b.float(Scalar::F32, 0.0), b.float(Scalar::F32, 1.0));
    let carried = [first, second]
        .into_iter()
        .map(|init| gpu_ir::Carry {
            ty,
            init,
            into: None,
        })
        .collect::<Vec<_>>();
    let (body, ()) = b.build_loop_body(Scalar::S32, carried, |b, index, given| {
        let zero = b.i32(0);
        let taken = b.gt(index, zero);
        b.open_arm();
        let product = b.mul(lane.clone(), weight.clone());
        let combined = b.add(product, given[0].clone());
        let then_ = b.close_arm(vec![combined]);
        b.open_arm();
        let else_ = b.close_arm(vec![given[0].clone()]);
        let produced = b.if_arms(taken, &[(ty, Some(given[0].clone()))], then_, else_);
        (
            vec![produced.into_iter().next().unwrap(), given[1].clone()],
            (),
        )
    });
    let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
    let folded = b.fold_loop(Trip::PerThread, start, stop, step, body);
    // The reader wants the *second* slot, so a rebuild that hands back the
    // first would be reading another accumulator.
    let bound = b.bind_value_as("result", ty, folded[1].clone());
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();

    let f = b.function();
    let init = f.op(ops[0]).operands[0];
    let fold = f.operand_op(init).expect("the fold");
    assert_eq!(f.op(fold).results.len(), 2, "both slots are carried");
    assert_eq!(
        f.op(fold).results[1],
        init,
        "the binding reads the slot it asked for"
    );
    let block = f.region_block(f.op(fold).regions[0]);
    let guard = f.block_steps(block).last().expect("the branch");
    assert!(matches!(f.op(guard).kind, OpKind::If));
    assert_eq!(
        f.op(guard).destination(0),
        Some(f.block(block).params[1]),
        "the result still produces into the value the loop carries"
    );
    let arm = f.region_block(f.op(guard).regions[0]);
    let combined = f.block(arm).terminator.yields[0];
    assert!(
        matches!(
            f.operand_op(combined).map(|op| &f.op(op).kind),
            Some(OpKind::Fma)
        ),
        "the arm's sum did not absorb the product beside it"
    );
}

/// A value the rebuild replaced is read through the fold that reads it: the
/// region is copied, not stepped over.
#[test]
fn a_substituted_binding_reaches_inside_a_fold() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    // `scaled` is a binding the fold reads; contraction rebuilds every binding,
    // so the fold must come out reading the rebuilt one.
    let doubled = b.add(x.clone(), x);
    let scaled = b.bind_value_as("scaled", ty, doubled);
    let zero = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![gpu_ir::Carry {
            ty,
            init: zero,
            into: None,
        }],
        |b, _, given| (vec![b.add(given[0].clone(), scaled.clone())], ()),
    );
    let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
    let folded = b.fold_loop(Trip::PerThread, start, stop, step, body);
    let bound = b.bind_value_as("result", ty, folded.into_iter().next().unwrap());
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();

    let f = b.function();
    assert_eq!(ops.len(), 2, "the binding it reads and the fold");
    let rebuilt_scaled = f.op(ops[0]).results[0];
    let fold = f.operand_op(f.op(ops[1]).operands[0]).expect("the fold");
    let block = f.region_block(f.op(fold).regions[0]);
    let yielded = f.block(block).terminator.yields[0];
    assert!(
        f.reads(yielded).contains(&rebuilt_scaled),
        "the rebuilt fold still reads the binding it was built beside"
    );
}

/// Fusing keeps the width it fused: a two-lane sum contracts and stays two
/// lanes wide.
#[test]
fn contraction_preserves_a_two_lane_vector_binding() {
    let held = contract(Scalar::F32, 2, |b, product, c, _| b.add(product, c));
    let result = held.named("result");
    assert_eq!(
        held.body.data_ty(result),
        Ty::vec(Scalar::F32, 2),
        "the fused binding lost its lanes"
    );
    assert!(
        matches!(held.kind(held.init("result")), Some(OpKind::Fma)),
        "a two-lane multiply-add was not contracted"
    );
}

/// A binding inside a loop body that is produced into what the loop carries
/// keeps its destination through the rebuild. Losing it would give the copy
/// storage of its own and leave the accumulator unwritten.
#[test]
fn a_rebuilt_loop_body_keeps_a_bindings_destination() {
    let mut b = Builder::new();
    let ty = Ty::scalar(Scalar::F32);
    let lane = b.parameter(Param {
        name: "lane".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let weight = b.parameter(Param {
        name: "weight".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let zero = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![gpu_ir::Carry {
            ty,
            init: zero,
            into: None,
        }],
        |b, _, given| {
            // A step of the body, produced into the value the loop carries.
            let product = b.mul(lane.clone(), weight.clone());
            let stepped = b.add(product, given[0].clone());
            let updated = b.update(given[0].clone(), stepped);
            b.name_value(&updated, "running");
            (vec![updated], ())
        },
    );
    let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
    let folded = b.fold_loop(Trip::PerThread, start, stop, step, body);
    let bound = b.bind_value_as("result", ty, folded.into_iter().next().unwrap());
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();

    let f = b.function();
    let fold = f.operand_op(f.op(ops[0]).operands[0]).expect("the fold");
    let block = f.region_block(f.op(fold).regions[0]);
    let step = f.block_steps(block).next().expect("the binding");
    assert!(matches!(f.op(step).kind, OpKind::Bind));
    assert_eq!(
        f.op(step).destination(0),
        Some(f.block(block).params[1]),
        "the rebuilt binding lost the place it is produced into"
    );
    assert_eq!(
        f.name(f.op(step).results[0]),
        Some("running"),
        "and the name it carried"
    );
    assert!(
        matches!(
            f.operand_op(f.op(step).operands[0])
                .map(|op| &f.op(op).kind),
            Some(OpKind::Fma)
        ),
        "the body still contracted"
    );
}

/// A pack is its parts end to end, whatever width each part is: a two-lane part
/// beside a three-lane one is one five-lane value, and each part contracts as
/// the vector it is.
#[test]
fn a_pack_of_vector_parts_keeps_their_widths_and_contracts_each() {
    let mut b = Builder::new();
    let declare = |b: &mut Builder, name: &str, lanes: u8| {
        let value = b.parameter(Param {
            name: name.to_string(),
            ty: ParamTy::Scalar(Scalar::F32),
        });
        b.splat(lanes, value)
    };
    let (a2, b2, c2) = (
        declare(&mut b, "a2", 2),
        declare(&mut b, "b2", 2),
        declare(&mut b, "c2", 2),
    );
    let (a3, b3, c3) = (
        declare(&mut b, "a3", 3),
        declare(&mut b, "b3", 3),
        declare(&mut b, "c3", 3),
    );
    let narrow = {
        let product = b.mul(a2, b2);
        b.add(product, c2)
    };
    let wide = {
        let product = b.mul(a3, b3);
        b.add(product, c3)
    };
    let packed = b.pack(vec![narrow, wide]);
    let five = Ty::vec(Scalar::F32, 5);
    assert_eq!(
        b.value_ty(&packed).data(),
        five,
        "a two-lane part beside a three-lane one is five lanes"
    );
    let bound = b.bind_value_as("packed", five, packed);
    let ops = b.steps();
    let mut roots = [bound];
    let ops = contract_fma_bindings(&mut b, ops, &mut roots).unwrap();

    let f = b.function();
    let init = f.op(ops[0]).operands[0];
    assert_eq!(f.data_ty(init), five, "the rebuilt pack lost its lanes");
    let pack = f.operand_op(init).expect("a pack");
    assert!(matches!(f.op(pack).kind, OpKind::Pack));
    for (part, lanes) in f.op(pack).operands.iter().zip([2u8, 3]) {
        assert_eq!(f.data_ty(*part), Ty::vec(Scalar::F32, lanes));
        assert!(
            matches!(
                f.operand_op(*part).map(|op| &f.op(op).kind),
                Some(OpKind::Fma)
            ),
            "a {lanes}-lane part of a pack did not contract"
        );
    }
}
