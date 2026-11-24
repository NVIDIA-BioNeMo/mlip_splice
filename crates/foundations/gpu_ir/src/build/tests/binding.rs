// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binding a value: what a step of the body is, and what it is not.

use super::super::Builder;
use super::{pointer, scalar};
use crate::op::OpKind;
use crate::{PtrTy, Scalar, Space, Ty, ValTy};

/// A binding is a step of the body and its result is a place: it is computed
/// once, wherever the binding sits, and every read of it reads that storage.
#[test]
fn a_binding_is_a_step_whose_result_is_a_place() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let doubled = b.add(x.clone(), x);
    let bound = b.bind_value(Ty::scalar(Scalar::F32), doubled);
    let f = b.function();
    let id = bound.id_in(f);
    assert!(f.is_materialized(id));
    assert_eq!(f.ty(id), ValTy::Data(Ty::scalar(Scalar::F32)));
    let op = f.defining_op(id).unwrap();
    assert!(matches!(f.op(op).kind, OpKind::Bind));
    assert!(f.op(op).placed);
}

/// Names carry meaning the arena does not, so they are a side table: naming a
/// value defines nothing and a name is found by asking for it.
#[test]
fn a_name_is_a_side_table_over_values() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let bound = b.bind_value_as("total", Ty::scalar(Scalar::F32), x.clone());
    let ops = b.owned().len();
    let values = b.function().num_values();
    b.name_value(&bound, "renamed");
    assert_eq!(b.owned().len(), ops, "naming is not a step");
    assert_eq!(b.function().num_values(), values, "naming defines nothing");
    let f = b.function();
    assert_eq!(f.name(bound.id_in(f)), Some("renamed"));
    assert_eq!(f.value_named("total"), None);
    assert_eq!(f.value_named("renamed"), Some(bound.id_in(f)));
}

/// A pointer binding is typed as a pointer and keeps the space it addresses.
#[test]
fn a_pointer_binding_stays_a_pointer() {
    let mut b = Builder::new();
    let source = pointer(&mut b, "source");
    let four = b.i32(4);
    let at = b.offset(source, four);
    let bound = b.bind_pointer_as(
        "row",
        PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        },
        at,
    );
    let f = b.function();
    assert_eq!(
        f.ty(bound.id_in(f)),
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        })
    );
}

/// Rebinding is renaming: the update is a new value that happens to be
/// produced into the storage the old one held, which is what makes the emitted
/// code write one register instead of copying between two.
#[test]
fn an_update_is_a_new_value_in_the_old_storage() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let place = b.bind_value_as("acc", Ty::scalar(Scalar::F32), x.clone());
    let stepped = b.add(place.clone(), x);
    let updated = b.update(place.clone(), stepped);
    let f = b.function();
    let (place, updated) = (place.id_in(f), updated.id_in(f));
    assert_ne!(place, updated);
    assert_eq!(f.ty(place), f.ty(updated));
    let op = f.defining_op(updated).unwrap();
    assert!(matches!(f.op(op).kind, OpKind::Bind));
    assert_eq!(f.op(op).destination(0), Some(place));
}

/// A constant a caller can specialise on is answered as one, and anything else
/// is not guessed at.
#[test]
fn an_integer_constant_answers_for_itself() {
    let mut b = Builder::new();
    let five = b.i32(5);
    assert_eq!(b.as_int_const(&five), Some(5));
    let x = scalar(&mut b, "x");
    assert_eq!(b.as_int_const(&x), None);
    let sum = b.add(five.clone(), five);
    assert_eq!(b.as_int_const(&sum), None, "nothing here folds constants");
}

/// The builder records what it is handed. It does not reassociate, contract, or
/// share a repeated subexpression, because the rounding sequence a caller wrote
/// is the one that must reach the device.
#[test]
fn the_builder_records_rather_than_rewrites() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let y = scalar(&mut b, "y");
    let product = b.mul(x.clone(), y.clone());
    let sum = b.add(product, y.clone());
    let bound = b.bind_value(Ty::scalar(Scalar::F32), sum);
    let f = b.function();
    let op = f.defining_op(bound.id_in(f)).unwrap();
    let init = f.op(op).operands[0];
    assert!(matches!(
        f.op(f.operand_op(init).unwrap()).kind,
        OpKind::Bin(crate::BinOp::Add)
    ));

    // The same expression written twice is two operations, since sharing one is
    // a caller's decision made where it can be read.
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let first = b.add(x.clone(), x.clone());
    let second = b.add(x.clone(), x);
    let f = b.function();
    assert_ne!(first.id_in(f), second.id_in(f));
    assert!(f.same_value(first.id_in(f), second.id_in(f)));
}

/// An analysis over a finished arena speaks in ids; handing one back stamps it
/// with the scope being built, so the two ways of naming a value meet.
#[test]
fn an_id_from_an_analysis_becomes_a_value_again() {
    let mut b = Builder::new();
    let x = scalar(&mut b, "x");
    let bound = b.bind_value_as("held", Ty::scalar(Scalar::F32), x);
    let id = b.function().value_named("held").unwrap();
    assert_eq!(id, bound.id_in(b.function()));
    let again = b.value_of(id);
    let doubled = b.add(again, bound);
    let f = b.function();
    assert_eq!(f.reads(doubled.id_in(f)), [id].into_iter().collect());
}
