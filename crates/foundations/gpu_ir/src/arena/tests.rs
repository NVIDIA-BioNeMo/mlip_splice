// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the arena guarantees on its own: one owner, ids that only address the
//! arena that issued them, use-def edges that see through operands and into
//! regions, and a `into` that says which storage a result is produced in.

use crate::build::{Body, Builder, Carry, Value};
use crate::op::OpKind;
use crate::{Param, ParamTy, Scalar, Trip, Ty, ValTy};

use super::Def;

/// One body binding `sum = a + b`, with `a` and `b` its parameters.
fn sum() -> Body {
    let mut b = Builder::new();
    let left = b.parameter(Param {
        name: "a".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let right = b.parameter(Param {
        name: "b".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let added = b.add(left, right);
    let bound = b.bind_value(Ty::scalar(Scalar::F32), added);
    b.name_value(&bound, "sum");
    b.finish_owned()
}

/// A value of one arena addresses that arena and no other, whatever the two
/// happen to be numbered. Two functions both number from zero, so this is the
/// mistake the tag exists to catch.
#[test]
#[should_panic(expected = "a value of")]
fn a_value_of_another_arena_is_refused() {
    let held = sum();
    let elsewhere = Builder::new().into_function();
    let _ = elsewhere.ty(held.function().declared(0));
}

/// Cloning re-tags, because a clone is made in order to diverge from what it
/// was cloned from.
#[test]
#[should_panic(expected = "a value of")]
fn a_clone_does_not_answer_to_the_original() {
    let held = sum();
    let copy = held.function().clone();
    let _ = copy.ty(held.function().declared(0));
}

/// The map is the way across, and it is by index: the copy holds the same
/// program, so the value at each index means what it meant.
#[test]
fn clone_with_map_carries_every_value_across() {
    let held = sum();
    let source = held.function();
    let (copy, mapped) = source.clone_with_map();
    assert_eq!(mapped.len(), source.num_values());
    assert_eq!(copy.num_ops(), source.num_ops());
    for (index, &carried) in mapped.iter().enumerate() {
        let original = super::ValueId::new(source.tag(), index);
        assert_eq!(copy.ty(carried), source.ty(original));
        assert_eq!(copy.name(carried), source.name(original));
        assert_eq!(
            copy.is_materialized(carried),
            source.is_materialized(original)
        );
    }
    let named = copy
        .value_named("sum")
        .expect("the copy holds the names the source chose");
    assert_eq!(named, mapped[source.value_named("sum").unwrap().index()]);
}

/// What the arena writes this function down as.
fn written_down(function: &super::Function) -> String {
    let mut text = String::new();
    crate::arena::text::render_function(function, &|_| 0, "", &mut text);
    text
}

/// The arena that text says.
fn parsed(function: &super::Function) -> super::Function {
    let text = written_down(function);
    crate::arena::text::parse_function(&mut crate::arena::text::Reader::new(&text))
        .expect("what the arena rendered, the arena parses")
        .build(&|_| None)
        .expect("it is the arena's own text")
}

/// Every arena a caller can come by wears its own tag, so none of them answers
/// to another's ids — whichever way it was come by.
#[test]
fn no_two_functions_a_caller_can_hold_share_a_tag() {
    let body = sum();
    let source = body.function();
    // A transaction keeps the tag of what it writes, which is exactly why it
    // must replace it: what wore the tag before is gone, so the count is still
    // one arena per tag.
    let mut written = source.clone();
    let kept = written.tag();
    written.transact(|b| vec![b.int(Scalar::S32, 7)]);
    assert_eq!(written.tag(), kept, "the same arena, written");

    let tags: Vec<super::ArenaTag> = vec![
        source.tag(),
        kept,
        Builder::new().into_function().tag(),
        source.clone().tag(),
        source.clone_with_map().0.tag(),
        crate::arena::wire::decode_function(&crate::arena::wire::encode_function(source))
            .expect("what the arena wrote, the arena reads")
            .tag(),
        parsed(source).tag(),
    ];
    let distinct: std::collections::BTreeSet<_> = tags.iter().collect();
    assert_eq!(distinct.len(), tags.len(), "each is its own arena");
}

/// And two arenas refuse each other's ids, however alike they are: `written`
/// holds every value `source` does, at the same indices, and answers to none of
/// them.
#[test]
#[should_panic(expected = "a value of")]
fn one_arena_does_not_answer_to_another_that_holds_the_same_program() {
    let body = sum();
    let source = body.function();
    let mut written = source.clone();
    written.transact(|b| vec![b.int(Scalar::S32, 7)]);
    let _ = written.ty(source.value_at(0));
}

/// A write is all of what it did or none of it, and the arena is what says so.
///
/// The copy it writes into wears this arena's own tag, so an id taken before the
/// write goes on naming what it named. That is sound because the copy is never
/// handed out: `transact` holds `&mut self` for as long as it lives, and what
/// comes back is ids, never an arena. There is no way to end up holding two.
#[test]
fn a_transaction_keeps_every_id_and_hands_back_the_ones_it_added() {
    let mut function = sum().into_function();
    let tag = function.tag();
    let before: Vec<String> = (0..function.num_values())
        .map(|index| format!("{:?}", function.ty(function.value_at(index))))
        .collect();
    let steps = function.block(function.entry()).ops.len();

    let written = function.transact(|b| {
        let seven = b.int(Scalar::S32, 7);
        let doubled = b.add(seven.clone(), seven);
        vec![doubled]
    });

    assert_eq!(function.tag(), tag, "the same arena, written");
    for (index, ty) in before.iter().enumerate() {
        let held = super::ValueId::new(tag, index);
        assert_eq!(&format!("{:?}", function.ty(held)), ty, "id {index} moved");
    }
    assert_eq!(written.values.len(), 1);
    assert!(written.values[0].index() >= before.len(), "it is new");
    assert_eq!(
        function.ty(written.values[0]),
        ValTy::Data(Ty::scalar(Scalar::S32))
    );
    assert_eq!(
        written.steps,
        function.block(function.entry()).ops[steps..].to_vec(),
        "the steps it appended, and only those"
    );
}

/// A write that stops partway leaves nothing behind: the copy is dropped without
/// being installed, so the function is the one it was — the same bytes, the same
/// text, and the same ids answering to it.
#[test]
fn a_transaction_that_panics_leaves_the_function_it_was_given() {
    let mut function = sum().into_function();
    let tag = function.tag();
    let bytes = crate::arena::wire::encode_function(&function);
    let text = written_down(&function);
    let held = function.value_at(0);

    let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        function.transact(|b| {
            b.int(Scalar::S32, 7);
            panic!("the caller stopped halfway");
        })
    }));
    assert!(stopped.is_err(), "it stopped");

    assert_eq!(function.tag(), tag, "still the arena it was");
    assert_eq!(crate::arena::wire::encode_function(&function), bytes);
    assert_eq!(written_down(&function), text);
    // The id taken before it still answers, and to what it named.
    assert_eq!(function.ty(held), ValTy::Data(Ty::scalar(Scalar::F32)));
}

#[test]
fn unattached_structure_is_deleted_without_weakening_verification() {
    let mut function = sum().into_function();
    let block = function.new_block();
    let parameter = function
        .new_block_param(block, ValTy::Data(Ty::scalar(Scalar::F32)))
        .expect("an ordinary block");
    let region = function.new_region(vec![block]).expect("a fresh block");
    let device = crate::DeviceFunction::new("sum", function.clone());
    assert!(
        crate::verify(&device).is_err(),
        "the verifier sees the orphan"
    );

    function
        .delete_unattached()
        .expect("the orphan is temporary");

    assert!(!function.holds_value(parameter));
    assert!(!function.holds_block(block));
    assert!(!function.holds_region(region));
    crate::verify(&crate::DeviceFunction::new("sum", function)).expect("the program remains");
}

#[test]
fn unattached_structure_is_deleted_transactionally() {
    let mut function = sum().into_function();
    let block = function.new_block();
    let parameter = function
        .new_block_param(block, ValTy::Data(Ty::scalar(Scalar::F32)))
        .expect("an ordinary block");
    let region = function.new_region(vec![block]).expect("a fresh block");
    let reader = function
        .live_ops()
        .into_iter()
        .find(|&op| !function.op(op).operands.is_empty())
        .expect("the sum reads operands");
    function
        .set_operand(reader, 0, parameter)
        .expect("the same arena");

    assert!(function.delete_unattached().is_err());

    assert!(function.holds_value(parameter));
    assert!(function.holds_block(block));
    assert!(function.holds_region(region));
}

/// A block parameter is a place; the result of a placed operation is a place;
/// the result of an operand operation is not. That distinction is what makes a
/// value read twice cost twice.
#[test]
fn only_a_parameter_or_a_placed_result_is_a_place() {
    let held = sum();
    let f = held.function();
    let a = f.declared(0);
    assert!(f.is_materialized(a));
    assert!(matches!(f.value(a).def, Def::Param { index: 0, .. }));

    let bound = f.value_named("sum").unwrap();
    assert!(f.is_materialized(bound));
    assert!(f.operand_op(bound).is_none());
    let op = f.defining_op(bound).expect("a binding defines its result");
    assert!(matches!(f.op(op).kind, OpKind::Bind));

    let added = f.op(op).operands[0];
    assert!(!f.is_materialized(added));
    assert!(f.operand_op(added).is_some());
}

/// What a value reads is the places its operand cone reaches, not the operands
/// on the way — an expression has no storage to read from.
#[test]
fn reads_are_the_places_an_operand_cone_reaches() {
    let held = sum();
    let f = held.function();
    let bound = f.value_named("sum").unwrap();
    let added = f.op(f.defining_op(bound).unwrap()).operands[0];
    assert_eq!(
        f.reads(added),
        [f.declared(0), f.declared(1)].into_iter().collect()
    );
    // A place reads itself: that is the storage a caller loads from.
    assert_eq!(f.reads(bound), [bound].into_iter().collect());
}

/// A loop standing for the value it reduces to reads what its body gathers,
/// less what the region binds for itself. Counting the index or the
/// accumulator would announce a place nothing outside the loop can name.
#[test]
fn an_operand_region_reads_what_it_does_not_bind_itself() {
    let mut b = Builder::new();
    let source = b.parameter(Param {
        name: "source".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let count = b.parameter(Param {
        name: "count".into(),
        ty: ParamTy::Scalar(Scalar::S32),
    });
    let zero = b.i32(0);
    let one = b.i32(1);
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
    let folded = b.fold_loop(Trip::PerThread, zero, count.clone(), one, body);
    let bound = b.bind_value(Ty::scalar(Scalar::F32), folded.into_iter().next().unwrap());
    b.name_value(&bound, "total");
    let held = b.finish_owned();

    let f = held.function();
    let total = f.value_named("total").unwrap();
    let fold = f.op(f.defining_op(total).unwrap()).operands[0];
    assert_eq!(
        f.reads(fold),
        [f.declared(0), f.declared(1)].into_iter().collect(),
        "the fold reads the pointer it gathers through and the bound it counts to"
    );
}

/// A result stating a destination says which storage it is produced in. That
/// replaces the name a tree re-used, and no liveness analysis recovers it.
#[test]
fn a_result_states_the_place_it_is_produced_into() {
    let mut b = Builder::new();
    let seed = b.float(Scalar::F32, 0.0);
    let accumulator = b.bind_value(Ty::scalar(Scalar::F32), seed);
    b.name_value(&accumulator, "acc");
    let one = b.float(Scalar::F32, 1.0);
    let stepped = b.add(accumulator.clone(), one);
    let updated = b.update(accumulator.clone(), stepped);
    b.name_value(&updated, "acc_next");
    let held = b.finish_owned();

    let f = held.function();
    let place = f.value_named("acc").unwrap();
    let next = f.value_named("acc_next").unwrap();
    assert_ne!(place, next, "an update is a new value, not the old one");
    let op = f.defining_op(next).unwrap();
    assert_eq!(
        f.op(op).destination(0),
        Some(place),
        "the update names the storage it writes"
    );
    assert!(
        f.reads(f.op(op).operands[0]).contains(&place),
        "and it still reads what that storage held"
    );
}

/// A result taking nothing over says so, rather than defaulting to a place the
/// backend would have to guess.
#[test]
fn a_result_taking_nothing_over_states_no_destination() {
    let held = sum();
    let f = held.function();
    let bound = f.value_named("sum").unwrap();
    let op = f.defining_op(bound).unwrap();
    assert_eq!(f.op(op).destination(0), None);
}

/// Slicing a body chooses which operations run. Nothing is copied and no value
/// is renumbered, which is what lets an analysis speak of the same values the
/// emitter will.
#[test]
fn slicing_a_block_renumbers_nothing() {
    let mut b = Builder::new();
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let first = b.bind_value(Ty::scalar(Scalar::F32), x.clone());
    let second = b.bind_value(Ty::scalar(Scalar::F32), x);
    b.name_value(&first, "first");
    b.name_value(&second, "second");
    let mut f = b.into_function();

    let before = f.num_values();
    let entry = f.entry();
    let kept = vec![f.block(entry).ops[1]];
    f.set_block_ops(entry, kept.clone());
    assert_eq!(f.block(entry).ops, kept);
    assert_eq!(f.num_values(), before);
    assert_eq!(f.name(f.value_named("first").unwrap()), Some("first"));
}

/// Every operation, including one whose result is re-evaluated at each use,
/// occupies a position in a block.
#[test]
fn an_operand_belongs_to_a_block() {
    let mut b = Builder::new();
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let doubled = b.add(x.clone(), x);
    let f = b.into_function();
    let operand = f.operand_op(doubled.id_in(&f)).unwrap();
    assert!(f.block(f.entry()).ops.contains(&operand));
}

#[test]
fn refusing_to_delete_a_used_expression_keeps_its_block_position() {
    let mut b = Builder::new();
    let one = b.float(Scalar::F32, 1.0);
    b.bind_value(Ty::scalar(Scalar::F32), one.clone());
    let mut f = b.into_function();
    let op = f.defining_op(one.id_in(&f)).expect("the constant");
    let before = f.step_of(op).expect("a live operation");
    assert!(f.delete_op(op).is_err());
    assert_eq!(f.step_of(op).expect("the operation remains"), before);
}

/// Structural sameness sees through operands and stops at places, which is
/// exactly what a caller asking "is this the same expression" means.
#[test]
fn structural_sameness_compares_expressions_not_storage() {
    let mut b = Builder::new();
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let one = b.float(Scalar::F32, 1.0);
    let other = b.float(Scalar::F32, 1.0);
    let left = b.add(x.clone(), one);
    let right = b.add(x.clone(), other);
    let two = b.float(Scalar::F32, 2.0);
    let apart = b.add(x, two);
    let f = b.into_function();
    assert!(f.same_value(left.id_in(&f), right.id_in(&f)));
    assert!(!f.same_value(left.id_in(&f), apart.id_in(&f)));
}

/// A load may not be moved to where it would not otherwise run; the arithmetic
/// around it may.
#[test]
fn speculating_stops_at_a_read_of_memory() {
    let mut b = Builder::new();
    let source = b.parameter(Param {
        name: "source".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let zero = b.i32(0);
    let one = b.float(Scalar::F32, 1.0);
    let at = b.offset(source, zero);
    let read = b.load(at, 1, true);
    let sum = b.add(read.clone(), one.clone());
    let f = b.into_function();
    assert!(f.is_speculatable(one.id_in(&f)));
    assert!(!f.is_speculatable(read.id_in(&f)));
    assert!(!f.is_speculatable(sum.id_in(&f)));
}

/// Two results of one operation are two values. Comparing them by the
/// operation alone would say a loop's first accumulator is its second.
#[test]
fn different_results_of_one_operation_are_not_the_same_value() {
    let mut b = Builder::new();
    let first = b.float(Scalar::F32, 0.0);
    let second = b.float(Scalar::F32, 1.0);
    let (body, ()) = b.build_loop_body(
        Scalar::S32,
        vec![
            Carry {
                ty: Ty::scalar(Scalar::F32),
                init: first,
                into: None,
            },
            Carry {
                ty: Ty::scalar(Scalar::F32),
                init: second,
                into: None,
            },
        ],
        |b, _, given| {
            let one = b.float(Scalar::F32, 1.0);
            (
                vec![
                    b.add(given[0].clone(), one.clone()),
                    b.add(given[1].clone(), one),
                ],
                (),
            )
        },
    );
    let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
    let carried = b.fold_loop(Trip::PerThread, start, stop, step, body);
    let f = b.function();
    let (left, right) = (carried[0].id_in(f), carried[1].id_in(f));
    assert!(f.same_value(left, left), "a value is itself");
    assert!(f.same_value(right, right));
    assert!(
        !f.same_value(left, right),
        "two accumulators of one loop are not one value"
    );
}

/// A loop is its body as much as its bounds: two that count the same way and
/// reduce differently are different expressions, and treating them as one
/// would let a division by a different fold cancel to one.
#[test]
fn loops_with_the_same_header_and_different_bodies_differ() {
    let fold = |b: &mut Builder, source: &Value, negate: bool| {
        let start = b.float(Scalar::F32, 0.0);
        let source = source.clone();
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: start,
                into: None,
            }],
            move |b, index, given| {
                let at = b.offset(source.clone(), index);
                let read = b.load(at, 1, true);
                let read = if negate { b.neg(read) } else { read };
                (vec![b.add(given[0].clone(), read)], ())
            },
        );
        let (start, stop, step) = (b.i32(0), b.i32(8), b.i32(1));
        b.fold_loop(Trip::PerThread, start, stop, step, body)
            .into_iter()
            .next()
            .unwrap()
    };

    let mut b = Builder::new();
    let source = b.parameter(Param {
        name: "source".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let summed = fold(&mut b, &source, false);
    let same = fold(&mut b, &source, false);
    let negated = fold(&mut b, &source, true);
    let f = b.function();
    assert!(
        f.same_value(summed.id_in(f), summed.id_in(f)),
        "a fold is itself"
    );
    assert!(
        f.same_value(summed.id_in(f), same.id_in(f)),
        "two folds written alike are the same expression"
    );
    assert!(
        !f.same_value(summed.id_in(f), negated.id_in(f)),
        "two folds that reduce differently are not"
    );
}

/// A value of one type is not a value of another, whatever operation produced
/// it: an `f32` sum and an `f64` sum over widened operands are two values.
#[test]
fn values_of_different_types_are_not_the_same() {
    let mut b = Builder::new();
    let x = b.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let narrow = b.add(x.clone(), x.clone());
    let wide = b.splat(2, x);
    let wide = b.add(wide.clone(), wide);
    let f = b.function();
    assert!(!f.same_value(narrow.id_in(f), wide.id_in(f)));
}

/// A sibling nothing reads is still part of what an operation is. Two loops
/// alike in the result a caller asked for and different in one it did not are
/// two operations, and pairing their bodies' values would say otherwise.
#[test]
fn an_unread_result_of_another_type_makes_two_operations_differ() {
    let fold = |b: &mut Builder, second: Scalar| {
        let first = b.float(Scalar::F32, 0.0);
        let other = b.float(second, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![
                Carry {
                    ty: Ty::scalar(Scalar::F32),
                    init: first,
                    into: None,
                },
                Carry {
                    ty: Ty::scalar(second),
                    init: other,
                    into: None,
                },
            ],
            move |b, _, given| {
                let one = b.float(Scalar::F32, 1.0);
                let other = b.float(second, 1.0);
                (
                    vec![b.add(given[0].clone(), one), b.add(given[1].clone(), other)],
                    (),
                )
            },
        );
        let (start, stop, step) = (b.i32(0), b.i32(4), b.i32(1));
        b.fold_loop(Trip::PerThread, start, stop, step, body)
            .into_iter()
            .next()
            .unwrap()
    };

    let mut b = Builder::new();
    let read = fold(&mut b, Scalar::F32);
    let alike = fold(&mut b, Scalar::F32);
    let apart = fold(&mut b, Scalar::F64);
    let f = b.function();
    assert!(
        f.same_value(read.id_in(f), alike.id_in(f)),
        "two loops written alike are the same expression"
    );
    assert!(
        !f.same_value(read.id_in(f), apart.id_in(f)),
        "a sibling of another type is another operation, read or not"
    );
}

/// Whether a value is read at all, counted over every way there is to read one:
/// an operand, the storage a result is produced into, and what a block hands
/// back — which for the entry block is what the function hands back.
///
/// This is what decides whether an indivisible add asks for the value that was
/// there, so it is one question with one answer and not a rule per reader.
#[test]
fn a_value_is_used_wherever_uses_says_it_is() {
    let mut b = Builder::new();
    let left = b.parameter(Param {
        name: "a".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let ty = Ty::scalar(Scalar::F32);
    let place = b.bind_value(ty, left.clone());
    let operand = b.bind_value(ty, left.clone());
    let destination = b.bind_value(ty, left.clone());
    let yielded = b.bind_value(ty, left.clone());
    let unread = b.bind_value(ty, left);

    // an operand
    let doubled = b.add(operand.clone(), operand.clone());
    b.bind_value(ty, doubled);
    // the storage a result is produced into
    b.update(destination.clone(), place.clone());
    // what the function hands back
    b.returns(vec![yielded.clone()]);
    let f = b.into_function();

    for held in [&operand, &destination, &yielded] {
        let id = held.clone().id_in(&f);
        assert!(f.is_used(id), "%{} is read", id.index());
        assert!(!f.uses(id).expect("a value of this arena").is_empty());
    }
    let id = unread.id_in(&f);
    assert!(!f.is_used(id), "%{} is read nowhere", id.index());
    assert!(f.uses(id).expect("a value of this arena").is_empty());
}

#[test]
fn freezing_builds_dense_typed_indexes_and_shared_analyses() {
    let mut builder = Builder::new();
    let left = builder.int(Scalar::S32, 1);
    let right = builder.int(Scalar::S32, 2);
    let sum = builder.add(left, right);
    let held = builder.bind_value(Ty::scalar(Scalar::S32), sum);
    let operation = builder.defining_op(&held).expect("the binding");
    let frozen = builder.into_function().freeze();
    let frozen_operation = frozen.frozen_operation(operation);
    let owner = frozen
        .operation_block(frozen_operation)
        .expect("the operation belongs to a block");

    assert!(matches!(
        frozen.operation(frozen_operation).kind,
        crate::OpKind::Bind
    ));
    assert!(
        frozen
            .block(owner)
            .ops
            .iter()
            .any(|op| op.index() == operation.index())
    );
    assert_eq!(
        frozen
            .uses()
            .of_value(held.id_in(frozen.function()).index()),
        []
    );
}
