// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::{
    WireError, decode_device_function, decode_function, encode_device_function, encode_function,
};
use crate::build::Counted;
use crate::{
    Access, Builder, DeviceFunction, Function, OpKind, Param, ParamTy, Scalar, SharedArr,
    Signature, Trip, Ty, verify,
};

/// `scale(dst, x) -> f32`: a callee with no body here, whose every fact is what
/// its declaration says.
fn scale() -> DeviceFunction {
    DeviceFunction::declared(
        "scale",
        Signature {
            parameters: vec![
                Param {
                    name: "dst".to_string(),
                    ty: ParamTy::Ptr(Scalar::F32),
                },
                Param {
                    name: "x".to_string(),
                    ty: ParamTy::Scalar(Scalar::F32),
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
            sync: Some(crate::effect::Sync::Block),
        },
    )
}

/// A body that calls `scale` and hands back what it produced.
fn calls_scale() -> DeviceFunction {
    let mut b = Builder::new();
    let dst = b.parameter(Param {
        name: "dst".to_string(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let x = b.parameter(Param {
        name: "x".to_string(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let handed = b
        .call(Arc::new(scale()), vec![dst, x])
        .pop()
        .expect("one result");
    b.returns(vec![handed]);
    DeviceFunction::new("calls_scale", b.into_function())
}

fn twice() -> Arc<DeviceFunction> {
    let mut callee = Builder::new();
    let taken = callee.parameter(Param {
        name: "taken".to_string(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let doubled = callee.add(taken.clone(), taken);
    callee.returns(vec![doubled]);
    Arc::new(DeviceFunction::new("twice", callee.into_function()))
}

/// A body reaching across the language: declarations of both kinds, a guarded
/// region, a carried loop, a call, a constant array, a warp collective, an
/// atomic add and a barrier.
fn everything() -> DeviceFunction {
    let callee = twice();
    let mut b = Builder::new();
    let values = b.parameter(Param {
        name: "values".to_string(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let total = b.parameter(Param {
        name: "total".to_string(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let count = b.parameter(Param {
        name: "count".to_string(),
        ty: ParamTy::Scalar(Scalar::S32),
    });
    let staging = b.shared_array(SharedArr {
        name: "staging".to_string(),
        elem: Scalar::F32,
        len: 32,
    });

    let tid = b.tid_x();
    let element = b.cast(Scalar::S32, tid);
    let element = b.bind_value_as("element", Ty::scalar(Scalar::S32), element);
    let zero = b.float(Scalar::F32, 0.0);

    let start = b.i32(0);
    let step = b.i32(1);
    let [sum] = b.counted_loop(
        Counted {
            index: Scalar::S32,
            start,
            stop: count.clone(),
            step,
            trip: Trip::BlockUniform,
        },
        [(Ty::scalar(Scalar::F32), zero.clone())],
        |b, index, [carried]| {
            let at = b.offset(values.clone(), index);
            let read = b.load(at, 1, true);
            let read = b
                .call(callee.clone(), vec![read])
                .pop()
                .expect("one result");
            [b.add(carried, read)]
        },
    );

    let looked_up = b.const_load("table", Scalar::F32, vec![0u8; 16], element.clone());
    let shared_at = b.offset(staging, element.clone());
    b.store(shared_at, looked_up);
    b.barrier();

    let inside = b.lt(element, count);
    b.if_(inside, |b| {
        let positive = b.gt(sum.clone(), zero.clone());
        let any = b.warp_any(positive);
        let selected = b.select(any, sum.clone(), zero.clone());
        b.red_add(total.clone(), selected);
    });
    b.ret();
    DeviceFunction::new("everything", b.into_function())
}

fn add_one() -> DeviceFunction {
    let mut b = Builder::new();
    let source = b.parameter(Param {
        name: "source".to_string(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let destination = b.parameter(Param {
        name: "destination".to_string(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    let at = b.i32(0);
    let read = b.offset(source, at.clone());
    let read = b.load(read, 1, true);
    let one = b.float(Scalar::F32, 1.0);
    let sum = b.add(read, one);
    let written = b.offset(destination, at);
    b.store(written, sum);
    DeviceFunction::new("add_one", b.into_function())
}

/// A debug rendering with the arena tags taken out. Two arenas holding the same
/// program carry different tags by construction, and the tag is the one thing a
/// round trip must not preserve.
fn tagless(value: &impl std::fmt::Debug) -> String {
    let text = format!("{value:?}");
    let mut left = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(at) = rest.find("ArenaTag(") {
        left.push_str(&rest[..at]);
        let end = rest[at..].find(')').expect("a tag closes") + at + 1;
        rest = &rest[end..];
    }
    left.push_str(rest);
    left
}

/// What a round trip has to preserve is what the arena is: every value, its
/// type and where it comes from; every operation, what it does, what it reads
/// and what it produces into; every block and region; and the names a caller
/// chose. Ids are compared as indices because the copy is a different arena.
fn same(left: &Function, right: &Function) {
    assert_eq!(left.num_values(), right.num_values());
    assert_eq!(left.num_ops(), right.num_ops());
    assert_eq!(left.declarations(), right.declarations());
    assert_eq!(left.entry().index(), right.entry().index());
    for index in 0..left.num_values() {
        let (a, b) = (left.value_at(index), right.value_at(index));
        assert_eq!(left.ty(a), right.ty(b), "value {index}");
        assert_eq!(left.name(a), right.name(b), "value {index}");
        assert_eq!(
            left.is_materialized(a),
            right.is_materialized(b),
            "value {index}"
        );
    }
    let indices = |ids: &[crate::ValueId]| ids.iter().map(|id| id.index()).collect::<Vec<_>>();
    for index in 0..left.num_ops() {
        let (a, b) = (left.op(left.op_at(index)), right.op(right.op_at(index)));
        assert_eq!(tagless(&a.kind), tagless(&b.kind), "op {index}");
        assert_eq!(a.placed, b.placed, "op {index}");
        assert_eq!(indices(&a.operands), indices(&b.operands), "op {index}");
        assert_eq!(indices(&a.results), indices(&b.results), "op {index}");
        assert_eq!(
            a.into
                .iter()
                .map(|id| id.map(|id| id.index()))
                .collect::<Vec<_>>(),
            b.into
                .iter()
                .map(|id| id.map(|id| id.index()))
                .collect::<Vec<_>>(),
            "op {index}"
        );
        assert_eq!(
            a.regions.iter().map(|id| id.index()).collect::<Vec<_>>(),
            b.regions.iter().map(|id| id.index()).collect::<Vec<_>>(),
            "op {index}"
        );
        for (&region, &copied) in a.regions.iter().zip(&b.regions) {
            let blocks = |f: &Function, region| {
                f.region(region)
                    .blocks
                    .iter()
                    .map(|&block| {
                        (
                            indices(&f.block(block).params),
                            f.block(block)
                                .ops
                                .iter()
                                .map(|op| op.index())
                                .collect::<Vec<_>>(),
                            indices(&f.block(block).terminator.yields),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(blocks(left, region), blocks(right, copied), "op {index}");
        }
    }
}

#[test]
fn a_body_is_the_same_arena_after_a_round_trip() {
    for function in [add_one(), everything()] {
        let bytes = encode_function(function.body().unwrap());
        let back = decode_function(&bytes).unwrap();
        same(function.body().unwrap(), &back);
        assert_eq!(
            encode_function(&back),
            bytes,
            "the bytes are a function of the arena"
        );
    }
}

/// The arena's tag is not among the bytes: two arenas holding the same program
/// are the same program, and cloning one re-tags it.
#[test]
fn the_bytes_are_the_program_and_not_the_arena_holding_it() {
    let function = everything();
    let copy = function.body().unwrap().clone();
    assert_ne!(
        function.body().unwrap().tag(),
        copy.tag(),
        "cloning re-tags"
    );
    assert_eq!(
        encode_function(function.body().unwrap()),
        encode_function(&copy)
    );
}

#[test]
fn a_decoded_body_still_verifies() {
    let function = everything();
    verify(&function).unwrap();
    let back = decode_device_function(&encode_device_function(&function)).unwrap();
    assert_eq!(back.name, "everything");
    verify(&back).unwrap();
}

/// One `Arc` used twice is one entry in the pool and one `Arc` on the way back:
/// a body that calls a function twice does not carry it twice.
#[test]
fn a_shared_callee_is_written_once_and_shared_again() {
    let callee = twice();
    let calling = |times: usize| {
        let mut b = Builder::new();
        let mut value = b.parameter(Param {
            name: "value".to_string(),
            ty: ParamTy::Scalar(Scalar::F32),
        });
        for _ in 0..times {
            value = b
                .call(callee.clone(), vec![value])
                .pop()
                .expect("one result");
        }
        b.returns(vec![value]);
        encode_function(&b.into_function())
    };

    let bytes = calling(2);
    let back = decode_function(&bytes).unwrap();
    let callees: Vec<Arc<DeviceFunction>> = (0..back.num_ops())
        .filter_map(|index| match &back.op(back.op_at(index)).kind {
            OpKind::Call(callee) => Some(callee.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(callees.len(), 2);
    assert!(
        Arc::ptr_eq(&callees[0], &callees[1]),
        "two uses of one function stay one function"
    );
    // What a second use costs is one operation, not a second body.
    let body = encode_function(callee.body().unwrap()).len();
    assert!(
        bytes.len() - calling(1).len() < body,
        "the second call wrote {} bytes and the callee's body is {body}",
        bytes.len() - calling(1).len()
    );
}

/// A callee with no body here is written as the contract it states, and comes
/// back as the same contract — every parameter, result, access and the threads
/// it needs.
#[test]
fn a_declaration_is_the_same_contract_after_a_round_trip() {
    let callee = scale();
    let bytes = encode_device_function(&callee);
    let back = decode_device_function(&bytes).unwrap();
    assert_eq!(back.name, "scale");
    assert_eq!(back.signature(), callee.signature());
    assert!(back.body().is_none(), "a declaration has no body");
    assert_eq!(
        encode_device_function(&back),
        bytes,
        "the bytes are a function of the contract"
    );
    verify(&back).unwrap();
}

/// A body that calls a declaration carries it in the same pool a body-backed
/// callee rides, so a retained call comes back as the call it was.
#[test]
fn a_body_carries_the_declaration_it_calls() {
    let function = calls_scale();
    let bytes = encode_function(function.body().unwrap());
    let back = decode_function(&bytes).unwrap();
    same(function.body().unwrap(), &back);
    let called = (0..back.num_ops())
        .filter_map(|index| match &back.op(back.op_at(index)).kind {
            OpKind::Call(callee) => Some(callee.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(called.len(), 1);
    assert_eq!(called[0].signature(), scale().signature());
    assert_eq!(
        encode_function(&back),
        bytes,
        "the bytes are a function of the arena"
    );
}

/// Every byte of a body that calls a declaration, flipped two ways: whatever
/// decodes is a structure, and what it claims about the contract is the
/// verifier's to refuse rather than the decoder's to believe.
#[test]
fn no_corrupted_declaration_escapes_as_a_verified_contract() {
    let bytes = encode_function(calls_scale().body().unwrap());
    let mut decoded = 0;
    for index in 0..bytes.len() {
        for flip in [0x01u8, 0x40] {
            let mut corrupted = bytes.clone();
            corrupted[index] ^= flip;
            let Ok(function) = decode_function(&corrupted) else {
                continue;
            };
            decoded += 1;
            for op in 0..function.num_ops() {
                let OpKind::Call(callee) = &function.op(function.op_at(op)).kind else {
                    continue;
                };
                let Some(signature) = callee.signature() else {
                    continue;
                };
                // Whatever the bytes said, what a caller may rely on is only
                // what the rules admit.
                if verify(callee).is_ok() {
                    assert_eq!(signature.accesses.len(), signature.parameters.len());
                    assert!(signature.results.len() <= 1);
                }
            }
        }
    }
    assert!(decoded > 0, "no flip decoded; this test proves nothing");
}

/// An implementation is a body or a declaration and there is no third thing.
#[test]
fn an_implementation_this_language_does_not_have_is_refused() {
    let mut bytes = encode_device_function(&scale());
    // The name is written first, as its length and then its bytes, so the tag
    // stands right after it.
    let tag = 8 + "scale".len();
    assert_eq!(bytes[tag], 1, "a declaration");
    bytes[tag] = 2;
    assert!(matches!(
        decode_device_function(&bytes),
        Err(WireError::UnknownTag {
            what: "implementation",
            tag: 2
        })
    ));
}

#[test]
fn bytes_that_are_not_an_arena_are_refused() {
    assert!(matches!(
        decode_function(&[]),
        Err(WireError::Truncated { .. })
    ));
    let bytes = encode_function(add_one().body().unwrap());
    assert!(matches!(
        decode_function(&bytes[..bytes.len() - 1]),
        Err(WireError::Truncated { .. })
    ));
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        decode_function(&trailing),
        Err(WireError::Trailing { left: 1 })
    ));
}

/// Every byte of a real body, flipped two ways: none may panic the decoder, and
/// whatever does decode is addressable — every id it holds indexes the arena it
/// belongs to, which is what stops a walk of it reaching outside.
#[test]
fn no_corrupted_byte_escapes_as_an_unaddressable_arena() {
    let bytes = encode_function(everything().body().unwrap());
    let mut decoded = 0;
    for index in 0..bytes.len() {
        for flip in [0x01u8, 0x40] {
            let mut corrupted = bytes.clone();
            corrupted[index] ^= flip;
            let Ok(function) = decode_function(&corrupted) else {
                continue;
            };
            decoded += 1;
            for value in 0..function.num_values() {
                let _ = function.ty(function.value_at(value));
            }
            for op in 0..function.num_ops() {
                let held = function.op(function.op_at(op));
                for &operand in &held.operands {
                    let _ = function.ty(operand);
                }
                for &region in &held.regions {
                    for &block in &function.region(region).blocks {
                        let _ = function.block(block).ops.len();
                    }
                }
            }
        }
    }
    assert!(
        decoded > 0,
        "a flip inside a name or a constant is still an arena; if none decoded \
         this test proves nothing"
    );
}

/// A tag no operation has is refused where it is read, rather than becoming
/// some other operation.
#[test]
fn an_operation_this_language_does_not_have_is_refused() {
    let mut b = Builder::new();
    let value = b.parameter(Param {
        name: "value".to_string(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let doubled = b.add(value.clone(), value);
    b.returns(vec![doubled]);
    let bytes = encode_function(&b.into_function());
    // The one operation's kind: `Bin(Add)` is tag 6 followed by tag 0, and the
    // arithmetic is the only place those two bytes are adjacent.
    let kind = bytes
        .windows(2)
        .position(|pair| pair == [6, 0])
        .expect("the binary operation is written");
    let mut corrupted = bytes;
    corrupted[kind] = 200;
    assert!(matches!(
        decode_function(&corrupted),
        Err(WireError::UnknownTag {
            what: "operation",
            tag: 200
        })
    ));
}
