// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carrying a program from one arena into another: source order, substitutions
//! as cut points, and what a copy is allowed to reach.

use super::super::{Body, Builder, Carry, Value};
use super::{bound_names, scalar};
use crate::op::OpKind;
use crate::{Scalar, Trip, Ty};

/// `a` and `b` in, `first = a + b`, `second = first * b`, `third = second + a`
/// out — three named steps whose order is the thing to preserve.
fn chain() -> Body {
    let mut b = Builder::new();
    let left = scalar(&mut b, "a");
    let right = scalar(&mut b, "b");
    let sum = b.add(left.clone(), right.clone());
    let first = b.bind_value_as("first", Ty::scalar(Scalar::F32), sum);
    let scaled = b.mul(first.clone(), right);
    let second = b.bind_value_as("second", Ty::scalar(Scalar::F32), scaled);
    let shifted = b.add(second, left);
    b.bind_value_as("third", Ty::scalar(Scalar::F32), shifted);
    b.finish_owned()
}

/// Importing copies in the source's own order. The emitted text is a function
/// of the order values are defined, so a fresh topological order here would
/// renumber every register for the same program.
/// Every operation belongs to a block, so an import holds what it copies from
/// the moment it arrives. Which of them the body *runs* is the caller's to say,
/// and it says so with the step list; one held by nobody would be live and
/// homeless, which is not a program.
/// Every operation an import creates belongs to a block, the operands it
/// copies on demand included: those are made while an operation reading them
/// is copied, and the caller never asked for them by name.
///
/// This states the invariant. It does not yet catch its violation — the shapes
/// that strand such an operand come from Loom, not from a body written here.
#[test]
fn importing_a_region_places_the_operands_it_copies_within() {
    let mut source = Builder::new();
    let x = scalar(&mut source, "x");
    let pred = source.lt(x.clone(), x.clone());
    source.if_(pred, |arm| {
        // A nested region, so the copy meets the operand two levels down,
        // where neither the caller nor the outer region lists what it makes.
        let inner = arm.lt(x.clone(), x.clone());
        arm.if_(inner, |deeper| {
            let doubled = deeper.add(x.clone(), x.clone());
            deeper.bind_value(Ty::scalar(Scalar::F32), doubled);
        });
    });
    let held = x.id_in(source.function());
    let source = source.finish_owned();

    let mut into = Builder::new();
    let y = scalar(&mut into, "y");
    into.import(&source, &[(held, y)]).expect("the import");
    let f = into.into_function();
    for op in f.live_ops() {
        assert!(
            f.live_blocks()
                .into_iter()
                .any(|b| f.block(b).ops.contains(&op)),
            "operation {op:?} {:?} is held by no block",
            f.op(op).kind
        );
    }
}

#[test]
fn an_import_holds_everything_it_copies() {
    let mut source = Builder::new();
    let x = scalar(&mut source, "x");
    let doubled = source.add(x.clone(), x.clone());
    source.bind_value(Ty::scalar(Scalar::F32), doubled);
    let held = x.id_in(source.function());
    let source = source.finish_owned();

    let mut into = Builder::new();
    let y = scalar(&mut into, "y");
    let import = into.import(&source, &[(held, y)]).expect("the import");
    let copied = import.into_ops();
    let f = into.into_function();
    let listed = f.block(f.entry()).ops.clone();
    for op in copied {
        assert!(listed.contains(&op), "operation {op:?} is held by no block");
    }
}

#[test]
fn an_import_keeps_the_order_the_source_holds() {
    let source = chain();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let f = source.function();
    let substitutions = vec![(f.declared(0), a), (f.declared(1), outer_b)];
    let import = b.import(&source, &substitutions).unwrap();
    let ops = import.ops().to_vec();
    b.place_all(ops.clone());
    let held = b.finish_owned();
    assert_eq!(
        bound_names(held.function(), &ops),
        ["first", "second", "third"]
    );
}

/// A substituted definition is a cut: it is read rather than recomputed, so
/// neither it nor what it used to read arrives. That is how a value written out
/// by one kernel and loaded back by another crosses.
#[test]
fn a_substituted_binding_is_a_cut() {
    let source = chain();
    let f = source.function();
    let second = f.value_named("second").unwrap();

    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let loaded = scalar(&mut b, "loaded_second");
    let substitutions = vec![
        (f.declared(0), a),
        (f.declared(1), scalar(&mut b, "outer_b")),
        (second, loaded.clone()),
    ];
    let import = b.import(&source, &substitutions).unwrap();
    let ops = import.ops().to_vec();
    b.place_all(ops.clone());
    let held = b.finish_owned();
    assert_eq!(
        bound_names(held.function(), &ops),
        ["first", "third"],
        "the cut binding did not arrive"
    );
    // And what stands for it downstream is the value it was cut to.
    assert_eq!(
        import.value(second).id_in(held.function()),
        loaded.id_in(held.function())
    );
}

/// Importing needs a value for everything the source reads from outside itself,
/// because there is nothing in the destination for it to mean otherwise.
#[test]
fn an_unsubstituted_parameter_is_refused() {
    let source = chain();
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let refused = b.import(&source, &[(f.declared(0), a)]);
    assert_eq!(
        refused.err().as_deref(),
        Some("importing left a parameter of the source unsubstituted")
    );
}

/// Two imports of one body are two copies: nothing they define is shared, so
/// changing what one reads cannot reach the other.
#[test]
fn importing_the_same_body_twice_keeps_the_two_apart() {
    let source = chain();
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let substitutions = vec![(f.declared(0), a), (f.declared(1), outer_b)];
    let first = b.import(&source, &substitutions).unwrap();
    let second = b.import(&source, &substitutions).unwrap();
    let third = f.value_named("third").unwrap();
    let held = b.function();
    assert_ne!(
        first.value(third).id_in(held),
        second.value(third).id_in(held)
    );
    assert_eq!(first.ops().len(), second.ops().len());
}

/// A value of a finished body, for a caller handing roots to an import.
fn root_of(held: Body, name: &str) -> (Body, Value) {
    let b = Builder::resuming(held);
    let id = b
        .function()
        .value_named(name)
        .expect("a name the body binds");
    let root = b.value_of(id);
    (b.finish_owned(), root)
}

/// A closure import brings only what the roots reach, in the source's own
/// order.
#[test]
fn a_closure_import_brings_what_its_roots_reach() {
    let (source, root) = root_of(chain(), "second");
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let substitutions = vec![(f.declared(0), a), (f.declared(1), outer_b)];
    let produced = b.import_closure(&source, &[root], &substitutions).unwrap();
    let placed = b.owned().to_vec();
    let held = b.finish_owned();
    assert_eq!(
        bound_names(held.function(), &placed),
        ["first", "second"],
        "reaching `second` brings `first`, and leaves `third` behind"
    );
    assert_eq!(produced.len(), 1);
    assert!(
        held.function()
            .is_materialized(produced[0].id_in(held.function()))
    );
}

/// A cut stops the closure from walking further back, which is what makes a
/// substitution a cut in the graph rather than an extra definition.
#[test]
fn a_closure_import_stops_at_its_cuts() {
    let (source, root) = root_of(chain(), "third");
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let loaded = scalar(&mut b, "loaded_second");
    let substitutions = vec![
        (f.declared(0), a),
        (f.declared(1), outer_b),
        (f.value_named("second").unwrap(), loaded),
    ];
    b.import_closure(&source, &[root], &substitutions).unwrap();
    let placed = b.owned().to_vec();
    let held = b.finish_owned();
    assert_eq!(
        bound_names(held.function(), &placed),
        ["third"],
        "`first` was only reachable through the cut"
    );
}

/// An operand loop is copied whole — its region and what its body reads —
/// because it stands for the value it reduces to wherever that value is read.
#[test]
fn an_import_carries_an_operand_loop_and_its_region() {
    let mut b = Builder::new();
    let source = super::pointer(&mut b, "source");
    let start = b.float(Scalar::F32, 0.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![Carry {
            ty: Ty::scalar(Scalar::F32),
            init: start,
            into: None,
        }],
        |b, index, given| {
            let at = b.offset(source.clone(), index);
            let read = b.load(at, 1, true);
            (vec![b.add(given[0].clone(), read)], ())
        },
    );
    let (zero, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
    let folded = b.fold_loop(Trip::PerThread, zero, stop, step, body);
    b.bind_value_as(
        "total",
        Ty::scalar(Scalar::F32),
        folded.into_iter().next().unwrap(),
    );
    let held = b.finish_owned();

    let f = held.function();
    let mut b = Builder::new();
    let outer = super::pointer(&mut b, "outer_source");
    let import = b.import(&held, &[(f.declared(0), outer)]).unwrap();
    let ops = import.ops().to_vec();
    b.place_all(ops.clone());
    let copied = b.finish_owned();
    let f = copied.function();
    assert_eq!(bound_names(f, &ops), ["total"]);
    let total = f.value_named("total").unwrap();
    let fold = f.op(f.defining_op(total).unwrap()).operands[0];
    let op = f.operand_op(fold).expect("the fold arrived as an operand");
    assert!(matches!(f.op(op).kind, OpKind::Loop { .. }));
    let block = f.region_block(f.op(op).regions[0]);
    assert_eq!(
        f.block(block).params.len(),
        2,
        "its index and what it carries"
    );
    assert_eq!(f.block(block).terminator.yields.len(), 1);
}

/// A copy is a different arena, so a value of the source does not address it.
/// The import's own map is the only way across.
#[test]
#[should_panic(expected = "a value of")]
fn a_source_value_does_not_address_the_copy() {
    let source = chain();
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let import = b
        .import(&source, &[(f.declared(0), a), (f.declared(1), outer_b)])
        .unwrap();
    let ops = import.ops().to_vec();
    b.place_all(ops);
    let copied = b.finish_owned();
    let _ = copied.function().ty(f.value_named("third").unwrap());
}

/// What an import reached is a question with an answer, so a caller that cut
/// part of a program away can ask rather than assume.
#[test]
fn an_import_says_what_it_reached() {
    let source = chain();
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let import = b
        .import(&source, &[(f.declared(0), a), (f.declared(1), outer_b)])
        .unwrap();
    for name in ["first", "second", "third"] {
        assert!(import.reached(f.value_named(name).unwrap()), "{name}");
    }
}

/// The import map is indexed by source ids, so it must reject an equally
/// numbered id issued by another arena rather than returning a real copied
/// value for it.
#[test]
#[should_panic(expected = "a value of")]
fn an_import_refuses_an_id_of_another_source() {
    let source = chain();
    let f = source.function();
    let mut b = Builder::new();
    let a = scalar(&mut b, "outer_a");
    let outer_b = scalar(&mut b, "outer_b");
    let import = b
        .import(&source, &[(f.declared(0), a), (f.declared(1), outer_b)])
        .unwrap();

    let elsewhere = chain();
    let foreign = elsewhere.function().value_named("first").unwrap();
    let _ = import.value(foreign);
}

/// What an import copies is held in creation order, and creation order already
/// stands every operand ahead of its reader: an operand is copied through the
/// operation reading it, so it exists first. That is what lets an import hold
/// what it made without asking the whole arena which operations travel.
#[test]
fn an_import_writes_each_operand_before_its_reader() {
    let mut source = Builder::new();
    let x = scalar(&mut source, "x");
    // A chain of operands under one named step, so a reader stands after
    // several operands that arrive only because it reads them.
    let doubled = source.add(x.clone(), x.clone());
    let squared = source.mul(doubled.clone(), doubled);
    let shifted = source.add(squared, x.clone());
    source.bind_value(Ty::scalar(Scalar::F32), shifted);
    let held = x.id_in(source.function());
    let source = source.finish_owned();

    let mut into = Builder::new();
    let y = scalar(&mut into, "y");
    into.import(&source, &[(held, y)]).expect("the import");
    let f = into.into_function();
    assert!(
        f.misordered().is_empty(),
        "an operand of the import stands after something reading it: {:?}",
        f.misordered()
    );
    assert!(
        f.stranded().is_empty(),
        "the import left an operation held by no block: {:?}",
        f.stranded()
    );
}
