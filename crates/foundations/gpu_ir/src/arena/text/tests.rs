// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the device half of the text says, and that it says all of it.

use std::sync::Arc;

use super::{
    ALL_BINARY, ALL_COMPARISONS, ALL_EVICTS, ALL_SCALARS, ALL_SPACES, ALL_SREGS, ALL_SYNCS,
    ALL_TRIPS, ALL_UNARY, ALL_WRITE_MODES, Reader, binary_name, binary_named, comparison_name,
    comparison_named, escaped, evict_name, evict_named, parse_device_function, parse_function,
    render_device_function, render_function, scalar_name, scalar_named, space_name, space_named,
    sreg_name, sreg_named, sync_name, sync_named, trip_name, trip_named, unary_name, unary_named,
    write_mode_name, write_mode_named,
};
use crate::arena::Function;
use crate::effect::Sync;
use crate::func::{Access, DeviceFunction, Signature};
use crate::{Scalar, Ty};

/// One arena as text, and the arena that text comes back as.
fn round_trip(f: &Function) -> (String, Function) {
    let mut text = String::new();
    render_function(f, &|_| 0, "  ", &mut text);
    let mut reader = Reader::new(&text);
    let raw = parse_function(&mut reader).expect("what was just written reads back");
    assert!(reader.at_end(), "the whole body was read");
    let back = raw.build(&|_| None).expect("a body with no calls resolves");
    (text, back)
}

fn rendered(f: &Function) -> String {
    let mut text = String::new();
    render_function(f, &|_| 0, "  ", &mut text);
    text
}

/// The body every test starts from: a guard, a load, arithmetic and a store.
fn add_one() -> Function {
    let mut b = crate::Builder::new();
    let source = b.parameter(crate::Param {
        name: "source".to_string(),
        ty: crate::ParamTy::Ptr(Scalar::F32),
    });
    let destination = b.parameter(crate::Param {
        name: "destination".to_string(),
        ty: crate::ParamTy::Ptr(Scalar::F32),
    });
    let count = b.parameter(crate::Param {
        name: "count".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::S32),
    });
    let ctaid = b.ctaid_x();
    let ntid = b.ntid_x();
    let tid = b.tid_x();
    let flat = b.mul(ctaid, ntid);
    let flat = b.add(flat, tid);
    let flat = b.cast(Scalar::S32, flat);
    let element = b.bind_value_as("element", Ty::scalar(Scalar::S32), flat);
    let inside = b.lt(element.clone(), count);
    b.if_(inside, |b| {
        let at = b.offset(source, element.clone());
        let read = b.load(at, 1, true);
        let one = b.float(Scalar::F32, 1.0);
        let sum = b.add(read, one);
        let to = b.offset(destination, element);
        b.store(to, sum);
    });
    b.into_function()
}

#[test]
fn a_rendered_body_reads_back_as_the_same_bytes() {
    let f = add_one();
    let (text, back) = round_trip(&f);
    assert_eq!(
        crate::arena::wire::encode_function(&f),
        crate::arena::wire::encode_function(&back),
        "the text carries everything the bytes do"
    );
    assert_eq!(text, rendered(&back), "and rendering it again is exact");
}

#[test]
fn comments_and_whitespace_are_normalized_away() {
    let f = add_one();
    let text = rendered(&f);
    let annotated = text
        .lines()
        .map(|line| format!("\n\n   {line}   # what this line is\n"))
        .collect::<String>();
    let mut reader = Reader::new(&annotated);
    let raw = parse_function(&mut reader).expect("comments are nothing");
    let back = raw.build(&|_| None).expect("a body with no calls resolves");
    assert_eq!(rendered(&back), text);
}

/// Every float that a decimal spelling would round, lose or confuse.
#[test]
fn a_float_constant_keeps_its_bits() {
    for scalar in [Scalar::F16, Scalar::BF16, Scalar::F32, Scalar::F64] {
        for value in [
            0.0f64,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_dead_beef),
            1.0 / 3.0,
            f64::MIN_POSITIVE,
        ] {
            let mut b = crate::Builder::new();
            let held = b.float(scalar, value);
            b.returns(vec![held]);
            let f = b.into_function();
            let (_, back) = round_trip(&f);
            let held = match back.op(back.op_at(0)).kind {
                crate::OpKind::FloatConst { value, .. } => value,
                ref other => panic!("a float constant became {other:?}"),
            };
            assert_eq!(
                held.to_bits(),
                value.to_bits(),
                "{scalar:?} {value} changed bits"
            );
        }
    }
}

#[test]
fn a_constant_array_keeps_its_bytes() {
    let data: Vec<u8> = (0..=255u8).collect();
    let mut b = crate::Builder::new();
    let index = b.i32(0);
    let held = b.const_load("table \"quoted\"\n", Scalar::U8, data.clone(), index);
    b.returns(vec![held]);
    let f = b.into_function();
    let (_, back) = round_trip(&f);
    match &back.op(back.op_at(1)).kind {
        crate::OpKind::ConstLoad(array) => {
            assert_eq!(&array.data[..], &data[..]);
            assert_eq!(array.name, "table \"quoted\"\n");
        }
        other => panic!("a constant load became {other:?}"),
    }
}

/// A name is a caller's own string, so every one of them has to survive.
#[test]
fn arbitrary_names_escape_and_unescape_exactly() {
    for held in [
        "",
        "plain",
        "a \"quoted\" one",
        "back\\slash",
        "line\nbreak\ttab\r",
        "\u{1}\u{7f}control",
        "θερμοκρασία 温度 🌡",
    ] {
        assert_eq!(
            Reader::new(&escaped(held)).quoted().expect("it reads back"),
            held
        );
    }
}

#[test]
fn a_call_names_the_symbol_it_was_rendered_with() {
    let mut callee = crate::Builder::new();
    let x = callee.parameter(crate::Param {
        name: "x".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::F32),
    });
    let doubled = callee.add(x.clone(), x);
    callee.returns(vec![doubled]);
    let callee = Arc::new(DeviceFunction::new("twice", callee.into_function()));

    let mut b = crate::Builder::new();
    let x = b.parameter(crate::Param {
        name: "x".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::F32),
    });
    let produced = b.call(callee.clone(), vec![x]);
    b.returns(produced);
    let f = b.into_function();

    let mut text = String::new();
    render_function(&f, &|_| 7, "", &mut text);
    assert!(text.contains("call @7"), "{text}");
    let mut reader = Reader::new(&text);
    let raw = parse_function(&mut reader).expect("it reads back");
    assert_eq!(raw.calls(), vec![(0, 7)]);
    let back = raw
        .build(&|symbol| (symbol == 7).then(|| callee.clone()))
        .expect("the symbol is answered");
    assert_eq!(
        crate::arena::wire::encode_function(&f),
        crate::arena::wire::encode_function(&back)
    );
    let mut reader = Reader::new(&text);
    let raw = parse_function(&mut reader).expect("it reads back");
    assert!(
        raw.build(&|_| None).is_err(),
        "an unanswered symbol refuses"
    );
}

/// `scale(dst, x) -> f32`: a contract stating each thing a signature holds —
/// a touched pointer at a width, an untouched datum, a result and a barrier.
fn scale() -> Signature {
    Signature {
        parameters: vec![
            crate::Param {
                name: "dst".to_string(),
                ty: crate::ParamTy::Ptr(Scalar::F32),
            },
            crate::Param {
                name: "x".to_string(),
                ty: crate::ParamTy::Scalar(Scalar::F32),
            },
        ],
        results: vec![crate::ValTy::Data(Ty::scalar(Scalar::F32))],
        accesses: vec![
            Access {
                read: true,
                written: true,
                lanes: 4,
            },
            Access::default(),
        ],
        sync: Some(Sync::Block),
    }
}

/// One device function as text, and the function that text comes back as, with
/// no symbol answered from outside it.
fn device_round_trip(f: &DeviceFunction) -> (String, DeviceFunction) {
    let mut text = String::new();
    render_device_function(f, &|_| 0, "  ", &mut text);
    let mut reader = Reader::new(&text);
    let raw = parse_device_function(&mut reader).expect("what was just written reads back");
    assert!(reader.at_end(), "the whole function was read");
    let back = raw
        .build(&|_| None)
        .expect("nothing outside this text is named");
    (text, back)
}

/// Everything a declaration states survives the text and comes back from the
/// text alone: no copy of the original is supplied, and the reader is handed no
/// symbol to answer.
#[test]
fn a_declaration_is_the_whole_contract_the_text_states() {
    let held = DeviceFunction::declared("scale", scale());
    let (text, back) = device_round_trip(&held);
    assert_eq!(back.name, "scale");
    assert_eq!(back.signature(), Some(&scale()));
    assert!(back.body().is_none(), "a declaration has no body");
    let mut again = String::new();
    render_device_function(&back, &|_| 0, "  ", &mut again);
    assert_eq!(text, again, "the text is a function of the contract");
    crate::verify(&back).expect("what was verified before is verified after");
}

/// The three lists of a signature are stated apart, so a text that names a
/// different number of each says so rather than being reconciled on the way in
/// or out. That is what an editor between two valid contracts is holding, and
/// [`crate::verify`] is where it is refused.
#[test]
fn a_signature_whose_lists_disagree_comes_through_and_is_refused() {
    let mut wrong = scale();
    wrong.accesses.pop();
    let held = DeviceFunction::declared("scale", wrong.clone());
    let (_, back) = device_round_trip(&held);
    assert_eq!(back.signature(), Some(&wrong));
    let refused = crate::verify(&back).expect_err("it states what it does to one of two arguments");
    assert_eq!(refused.rule, crate::Rule::Abi);
}

/// A body is written whole under the same word, and a call in it still names a
/// symbol — which function that is remains the module's fact.
#[test]
fn a_body_is_a_device_function_the_text_states_too() {
    let mut b = crate::Builder::new();
    let x = b.parameter(crate::Param {
        name: "x".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::F32),
    });
    let doubled = b.add(x.clone(), x);
    b.returns(vec![doubled]);
    let held = DeviceFunction::new("twice", b.into_function());
    let (text, back) = device_round_trip(&held);
    assert!(text.contains("function \"twice\" body"), "{text}");
    assert_eq!(back.name, "twice");
    assert_eq!(
        crate::arena::wire::encode_function(held.body().unwrap()),
        crate::arena::wire::encode_function(back.body().unwrap())
    );
}

/// A caller and the function it calls, both written down, and the call resolved
/// from the declaration *the text carries* rather than from a copy the reader
/// was already holding. Nothing of the contract comes from outside these bytes.
#[test]
fn a_call_is_answered_by_the_declaration_the_text_states() {
    let callee = Arc::new(DeviceFunction::declared("scale", scale()));
    let mut b = crate::Builder::new();
    let dst = b.parameter(crate::Param {
        name: "dst".to_string(),
        ty: crate::ParamTy::Ptr(Scalar::F32),
    });
    let x = b.parameter(crate::Param {
        name: "x".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::F32),
    });
    let produced = b.call(callee.clone(), vec![dst, x]);
    b.returns(produced);
    let caller = DeviceFunction::new("calls_scale", b.into_function());

    let mut text = String::new();
    render_device_function(&callee, &|_| 0, "", &mut text);
    render_device_function(&caller, &|_| 0, "", &mut text);

    // From here on the originals are gone: everything is read out of `text`.
    let mut reader = Reader::new(&text);
    let first = parse_device_function(&mut reader).expect("the declaration reads back");
    let second = parse_device_function(&mut reader).expect("the caller reads back");
    assert!(reader.at_end(), "the whole text was read");
    let module = [Arc::new(
        first.build(&|_| None).expect("a declaration names nothing"),
    )];
    let held = second
        .build(&|symbol| module.get(symbol).cloned())
        .expect("the symbol is answered from the text");

    let called = held
        .body()
        .unwrap()
        .live_ops()
        .into_iter()
        .find_map(|op| match &held.body().unwrap().op(op).kind {
            crate::OpKind::Call(callee) => Some(callee.clone()),
            _ => None,
        })
        .expect("the body calls");
    assert_eq!(called.name, "scale");
    assert_eq!(called.signature(), Some(&scale()));
    // The recovered contract is the one the caller is type-checked against, so
    // a body rebuilt from text alone verifies exactly as the original did.
    crate::verify(&held).expect("the call agrees with the contract the text carried");
    let mut again = String::new();
    render_device_function(&module[0], &|_| 0, "", &mut again);
    render_device_function(&held, &|_| 0, "", &mut again);
    assert_eq!(text, again, "the text is a function of what it states");
}

/// The refusals a caller writing a contract by hand will meet.
#[test]
fn a_broken_contract_says_where_it_broke() {
    let refused = |text: &str| {
        let mut reader = Reader::new(text);
        let raw = parse_device_function(&mut reader);
        match raw {
            Err(held) => held,
            Ok(_) if reader.at_end() => panic!("{text:?} should not read"),
            // A word this grammar does not have ends the contract, and the
            // trailing text is the caller's to meet.
            Ok(_) => panic!("{text:?} left text behind rather than refusing"),
        }
    };
    let held = refused("function \"f\" nonsense\n");
    assert!(held.detail.contains("\"body\" belongs here"), "{held}");
    let held = refused("function \"f\" declared access sideways lanes 1\n");
    assert!(held.detail.contains("no way of touching memory"), "{held}");
    let held = refused("function \"f\" declared access read sync none\n");
    assert!(held.detail.contains("\"lanes\" belongs here"), "{held}");
    let held = refused("function \"f\" declared sync everyone\n");
    assert!(held.detail.contains("no synchronisation"), "{held}");
    let held = refused("function \"f\" declared sync none sync none\n");
    assert!(held.detail.contains("states a second"), "{held}");
    let held = refused("function f body\n");
    assert!(held.detail.contains("quoted string"), "{held}");
}

/// The refusals a caller writing this text by hand will meet.
#[test]
fn broken_text_says_where_it_broke() {
    let refused = |text: &str| {
        let mut reader = Reader::new(text);
        parse_function(&mut reader)
            .err()
            .unwrap_or_else(|| panic!("{text:?} should not read"))
    };
    let held = refused("value %1 data s32 x1 result *0 0\n");
    assert_eq!((held.line, held.column), (1, 7));
    assert!(held.detail.contains("%0 belongs here"), "{held}");
    let held = refused("op *0 step nonsense operands [] results [] into [] regions []\n");
    assert_eq!(held.line, 1);
    assert!(held.detail.contains("no operation"), "{held}");
    let held = refused("value %0 data nope x1 result *0 0\n");
    assert!(held.detail.contains("no scalar"), "{held}");
    let held = refused("constant $0 \"a\" f32 \"0\"\n");
    assert!(held.detail.contains("two hexadecimal digits"), "{held}");
    let held = refused("op *0 step floatconst f32 0x00 operands []\n");
    assert!(held.detail.contains("16 hexadecimal digits"), "{held}");
}

#[test]
fn a_body_that_enters_no_block_is_no_body() {
    let mut reader = Reader::new("value %0 data s32 x1 result *0 0\n");
    let raw = parse_function(&mut reader).expect("it reads");
    let refused = raw.build(&|_| None).expect_err("it states no entry");
    assert!(
        refused.detail.contains("which block it enters"),
        "{refused}"
    );
}

#[test]
fn a_text_operation_without_a_block_is_refused() {
    let text = "value %0 data s32 x1 result *0 0\n\
                op *0 operand intconst s32 1 operands [] results [%0] into [] regions []\n\
                block ^0 region none params [] ops [] yields []\n\
                entry ^0\n";
    let mut reader = Reader::new(text);
    let raw = parse_function(&mut reader).expect("the syntax reads");
    let refused = raw
        .build(&|_| None)
        .expect_err("every operation belongs to a block");
    assert!(refused.detail.contains("belongs to no block"), "{refused}");
}

/// Every operation this language has, written down and read back.
///
/// The kinds are stated here rather than derived, so a kind added to the
/// language is one this test does not carry — and the `match` at the bottom is
/// what says so at compile time.
#[test]
fn every_kind_of_operation_round_trips() {
    let callee = Arc::new(DeviceFunction::new("callee", Function::new()));
    let array = Arc::new(crate::op::ConstArray {
        name: "table".to_string(),
        scalar: Scalar::F32,
        data: vec![0u8; 4].into(),
    });
    let kinds = vec![
        crate::OpKind::IntConst {
            scalar: Scalar::S32,
            value: -7,
        },
        crate::OpKind::FloatConst {
            scalar: Scalar::F64,
            value: -0.0,
        },
        crate::OpKind::Sreg(crate::Sreg::CtaidY),
        crate::OpKind::Load {
            lanes: 4,
            nc: false,
            evict: crate::Evict::Last,
        },
        crate::OpKind::Offset,
        crate::OpKind::Un(crate::UnOp::Sqrt),
        crate::OpKind::Bin(crate::BinOp::Shr),
        crate::OpKind::Fma,
        crate::OpKind::Cmp(crate::CmpOp::Ge),
        crate::OpKind::Select,
        crate::OpKind::Cast(Scalar::BF16),
        crate::OpKind::Splat(3),
        crate::OpKind::Extract(2),
        crate::OpKind::Pack,
        crate::OpKind::ShflDown,
        crate::OpKind::ActiveMask,
        crate::OpKind::ShflFrom,
        crate::OpKind::WarpAny,
        crate::OpKind::ConstLoad(array.clone()),
        crate::OpKind::Bind,
        crate::OpKind::Write {
            mode: crate::WriteMode::Add,
            evict: crate::Evict::First,
        },
        crate::OpKind::FetchAdd,
        crate::OpKind::If,
        crate::OpKind::Loop {
            index: Scalar::U64,
            trip: crate::Trip::BlockUniform,
        },
        crate::OpKind::Scope,
        crate::OpKind::Call(callee.clone()),
        crate::OpKind::Break,
        crate::OpKind::Ret,
        crate::OpKind::Bar,
    ];

    let mut f = Function::new();
    let held = f
        .declare_param(crate::Param {
            name: "p".to_string(),
            ty: crate::ParamTy::Scalar(Scalar::F32),
        })
        .expect("one declaration");
    let data = crate::ValTy::Data(Ty::scalar(Scalar::F32));
    for kind in &kinds {
        // A loop and a conditional are either a step or an operand, so both
        // spellings of `placed` are written for them and one for everything
        // else.
        let placements: &[bool] = match kind.may_be_either() {
            true => &[false, true],
            false => match kind.must_be_placed() {
                true => &[true],
                false => &[false],
            },
        };
        for &placed in placements {
            let regions = (0..kind.regions())
                .map(|_| {
                    let block = f.new_block();
                    f.new_region(vec![block]).expect("a fresh block")
                })
                .collect();
            let op = f
                .new_op(
                    kind.clone(),
                    placed,
                    vec![held, held],
                    &[data],
                    vec![Some(held)],
                    regions,
                )
                .expect("the shape each kind states");
            let entry = f.entry();
            let position = f.block(entry).ops.len();
            f.insert_step(entry, position, op)
                .expect("every operation has a block");
        }
    }

    let mut text = String::new();
    render_function(&f, &|_| 0, "", &mut text);
    let mut reader = Reader::new(&text);
    let raw = parse_function(&mut reader).expect("what was just written reads back");
    let back = raw
        .build(&|symbol| (symbol == 0).then(|| callee.clone()))
        .expect("the one symbol is answered");
    assert_eq!(
        crate::arena::wire::encode_function(&f),
        crate::arena::wire::encode_function(&back),
        "{text}"
    );
    assert!(reader.at_end(), "the whole body was read");

    // Every variant is listed above: the match is what fails to compile when
    // one is added, and the count is what fails when one is dropped.
    for kind in &kinds {
        match kind {
            crate::OpKind::IntConst { .. }
            | crate::OpKind::FloatConst { .. }
            | crate::OpKind::Sreg(_)
            | crate::OpKind::Load { .. }
            | crate::OpKind::Offset
            | crate::OpKind::Un(_)
            | crate::OpKind::Bin(_)
            | crate::OpKind::Fma
            | crate::OpKind::Cmp(_)
            | crate::OpKind::Select
            | crate::OpKind::Cast(_)
            | crate::OpKind::Splat(_)
            | crate::OpKind::Extract(_)
            | crate::OpKind::Pack
            | crate::OpKind::ShflDown
            | crate::OpKind::ActiveMask
            | crate::OpKind::ShflFrom
            | crate::OpKind::WarpAny
            | crate::OpKind::ConstLoad(_)
            | crate::OpKind::Bind
            | crate::OpKind::Write { .. }
            | crate::OpKind::FetchAdd
            | crate::OpKind::If
            | crate::OpKind::Loop { .. }
            | crate::OpKind::Scope
            | crate::OpKind::Call(_)
            | crate::OpKind::Break
            | crate::OpKind::Ret
            | crate::OpKind::Bar => {}
        }
    }
    assert_eq!(kinds.len(), 29, "one of every kind");
}

/// A body whose blocks nest: a loop, a guard inside it, a barrier and a break
/// under that, and a call beside them.
#[test]
fn nested_blocks_and_regions_round_trip() {
    let mut b = crate::Builder::new();
    let source = b.parameter(crate::Param {
        name: "source".to_string(),
        ty: crate::ParamTy::Ptr(Scalar::F32),
    });
    let trips = b.parameter(crate::Param {
        name: "trips".to_string(),
        ty: crate::ParamTy::Scalar(Scalar::S32),
    });
    let (start, step) = (b.i32(0), b.i32(1));
    let zero = b.float(Scalar::F32, 0.0);
    let total = b.counted_loop(
        crate::Counted {
            index: Scalar::S32,
            start,
            stop: trips.clone(),
            step,
            trip: crate::Trip::BlockUniform,
        },
        [(Ty::scalar(Scalar::F32), zero)],
        |b, index, [carried]| {
            b.barrier();
            let inside = b.lt(index.clone(), trips.clone());
            b.if_(inside, |b| {
                b.scoped(|b| b.break_());
            });
            let at = b.offset(source.clone(), index);
            let read = b.load(at, 1, true);
            [b.add(carried, read)]
        },
    );
    b.returns(total.to_vec());
    let f = b.into_function();
    let (text, back) = round_trip(&f);
    assert_eq!(
        crate::arena::wire::encode_function(&f),
        crate::arena::wire::encode_function(&back),
        "{text}"
    );
    assert!(
        back.num_regions() >= 4,
        "a loop, a guard's two arms, a scope"
    );
}

/// One operation is a step of one block, whichever way a caller states the
/// steps of one — and a refused statement leaves both blocks as they were.
#[test]
fn one_block_cannot_take_a_step_another_runs() {
    let mut f = add_one();
    let guard = (0..f.num_ops())
        .find(|&index| matches!(f.op(f.op_at(index)).kind, crate::OpKind::If))
        .expect("the shown body guards");
    let guard = f.op_at(guard);
    let then_ = f.region_block(f.op(guard).regions[0]);
    let else_ = f.region_block(f.op(guard).regions[1]);
    let write = f.block_steps(then_).next().expect("a step");
    let before = crate::arena::wire::encode_function(&f);

    let refused = f
        .set_steps(else_, vec![write])
        .expect_err("the then arm runs it");
    assert!(
        matches!(refused, crate::arena::edit::EditError::AlreadyAStep { .. }),
        "{refused}"
    );
    assert_eq!(
        crate::arena::wire::encode_function(&f),
        before,
        "a refused statement states nothing"
    );
    assert!(
        f.block(then_).ops.contains(&write),
        "and the arm still runs it"
    );

    // Moving is the atomic way to change the block position.
    f.move_step(write, else_, 0).expect("it moves as one edit");
    assert_eq!(f.block(else_).ops, vec![write]);
    assert!(!f.block(then_).ops.contains(&write));
    // And the list it is given may not name one thing twice.
    assert!(f.set_steps(else_, vec![write, write]).is_err());
}

/// What is written down is the program, so what an editor deleted is not in
/// the bytes and does not come back out of them.
#[test]
fn deleted_operations_are_not_in_the_bytes() {
    let mut f = add_one();
    // The constant this body adds, once nothing reads it.
    let one = (0..f.num_ops())
        .find(|&index| matches!(f.op(f.op_at(index)).kind, crate::OpKind::FloatConst { .. }))
        .expect("the shown body adds a constant");
    let one = f.op_at(one);
    let two = f
        .new_op(
            crate::OpKind::FloatConst {
                scalar: Scalar::F32,
                value: 2.0,
            },
            false,
            Vec::new(),
            &[crate::ValTy::Data(Ty::scalar(Scalar::F32))],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let two = f.op(two).results[0];
    assert_eq!(f.replace_uses(f.op(one).results[0], two).unwrap(), 1);
    f.delete_op(one).expect("nothing reads it now");

    let bytes = crate::arena::wire::encode_function(&f);
    assert_eq!(
        bytes,
        crate::arena::wire::encode_function(&f.compacted()),
        "the bytes are the program's, with the holes closed up"
    );
    let back = crate::arena::wire::decode_function(&bytes).expect("its own bytes");
    assert_eq!(back.num_ops(), f.live_ops().len(), "no hole came back");
    let constants: Vec<f64> = (0..back.num_ops())
        .filter_map(|index| match back.op(back.op_at(index)).kind {
            crate::OpKind::FloatConst { value, .. } => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(constants, vec![2.0]);
    // And what a whole device function is written as says the same.
    let held = DeviceFunction::new("add_one", f);
    let written = crate::arena::wire::encode_device_function(&held);
    let read = crate::arena::wire::decode_device_function(&written).expect("its own bytes");
    assert_eq!(read.body().unwrap().num_ops(), back.num_ops());
    assert_eq!(
        written,
        crate::arena::wire::encode_device_function(&DeviceFunction::new(
            "add_one",
            held.body().unwrap().compacted()
        ))
    );
}

/// A region-owning operation goes with what it holds, and none of it is in the
/// bytes afterwards.
#[test]
fn a_deleted_conditional_leaves_no_region_in_the_bytes() {
    let mut f = add_one();
    let guard = (0..f.num_ops())
        .find(|&index| matches!(f.op(f.op_at(index)).kind, crate::OpKind::If))
        .expect("the shown body guards");
    let guard = f.op_at(guard);
    for block in [
        f.region_block(f.op(guard).regions[0]),
        f.region_block(f.op(guard).regions[1]),
    ] {
        for op in f.block(block).ops.clone().into_iter().rev() {
            if f.op(op).placed {
                f.detach_step(op).unwrap();
            }
            f.delete_op(op).unwrap();
        }
    }
    f.detach_step(guard).unwrap();
    f.delete_op(guard).expect("both arms are empty");

    let bytes = crate::arena::wire::encode_function(&f);
    let back = crate::arena::wire::decode_function(&bytes).expect("its own bytes");
    assert_eq!(back.num_regions(), 0, "no region came back");
    assert_eq!(back.num_blocks(), 1, "only the entry block came back");
    assert!(
        !(0..back.num_ops()).any(|index| matches!(
            back.op(back.op_at(index)).kind,
            crate::OpKind::If | crate::OpKind::Write { .. }
        )),
        "neither the guard nor what it held came back"
    );
    assert_eq!(bytes, crate::arena::wire::encode_function(&f.compacted()));
}

/// One spelling each way, for every variant the language has.
#[test]
fn every_spelling_reads_back_as_what_it_spells() {
    macro_rules! both_ways {
        ($all:ident, $name:ident, $named:ident) => {
            for held in $all {
                assert_eq!($named($name(*held)), Some(*held), "{:?}", held);
            }
        };
    }
    both_ways!(ALL_SCALARS, scalar_name, scalar_named);
    both_ways!(ALL_SPACES, space_name, space_named);
    both_ways!(ALL_EVICTS, evict_name, evict_named);
    both_ways!(ALL_SREGS, sreg_name, sreg_named);
    both_ways!(ALL_UNARY, unary_name, unary_named);
    both_ways!(ALL_BINARY, binary_name, binary_named);
    both_ways!(ALL_COMPARISONS, comparison_name, comparison_named);
    both_ways!(ALL_WRITE_MODES, write_mode_name, write_mode_named);
    both_ways!(ALL_TRIPS, trip_name, trip_named);
    both_ways!(ALL_SYNCS, sync_name, sync_named);
    assert_eq!(scalar_named("nope"), None);
}
