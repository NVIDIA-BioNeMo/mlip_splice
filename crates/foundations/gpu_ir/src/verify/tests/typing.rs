// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What each operation's operands and results must hold, and which
//! arithmetic has a meaning at a scalar.

use super::*;

#[test]
fn arithmetic_over_two_types_is_refused() {
    let mut raw = raw();
    let held = raw.float(1.0);
    let index = raw.int(1);
    let sum = raw.operand(
        OpKind::Bin(BinOp::Add),
        vec![held, index],
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, index],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, sum],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Type);
}

/// The remainder is integral: there is no rounded float form to name, so a
/// float remainder is refused rather than lowered into something else.
#[test]
fn a_float_remainder_has_no_contract_to_name() {
    for (op, rule) in [
        (BinOp::Rem, Some(Rule::Type)),
        (BinOp::And, Some(Rule::Type)),
        (BinOp::Div, None),
        (BinOp::Max, None),
    ] {
        let mut raw = raw();
        let a = raw.float(3.0);
        let b = raw.float(2.0);
        let held = raw.operand(
            OpKind::Bin(op),
            vec![a, b],
            ValTy::Data(Ty::scalar(Scalar::F32)),
        );
        let zero = raw.int(0);
        let at = raw.operand(
            OpKind::Offset,
            vec![raw.buffer, zero],
            ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            }),
        );
        let write = raw.place(
            OpKind::Write {
                mode: WriteMode::Set,
                evict: Evict::Normal,
            },
            vec![at, held],
            &[],
        );
        let held = verify_launch(&launched(raw.body(vec![write])));
        assert_eq!(held.err().map(|why| why.rule), rule, "{op:?} on floats");
    }
}

/// A complement is bitwise on an integer and logical on a predicate. A float
/// has no bit pattern the operation is defined over, so it is refused.
#[test]
fn a_complement_of_a_float_is_refused() {
    for (scalar, rule) in [(Scalar::S32, None), (Scalar::F32, Some(Rule::Type))] {
        let mut raw = raw();
        let source = if scalar == Scalar::F32 {
            raw.float(1.0)
        } else {
            raw.count
        };
        let other = if scalar == Scalar::F32 {
            raw.float(0.0)
        } else {
            raw.count
        };
        let complement = raw.operand(
            OpKind::Un(UnOp::Not),
            vec![source],
            ValTy::Data(Ty::scalar(scalar)),
        );
        let below = raw.operand(
            OpKind::Cmp(CmpOp::Lt),
            vec![complement, other],
            ValTy::Data(Ty::scalar(Scalar::Pred)),
        );
        let zero = raw.int(0);
        let at = raw.operand(
            OpKind::Offset,
            vec![raw.buffer, zero],
            ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            }),
        );
        let held = raw.float(1.0);
        let none = raw.float(0.0);
        let value = raw.operand(
            OpKind::Select,
            vec![below, held, none],
            ValTy::Data(Ty::scalar(Scalar::F32)),
        );
        let write = raw.place(
            OpKind::Write {
                mode: WriteMode::Set,
                evict: Evict::Normal,
            },
            vec![at, value],
            &[],
        );
        let outcome = verify_launch(&launched(raw.body(vec![write])));
        assert_eq!(
            outcome.err().map(|why| why.rule),
            rule,
            "a complement of {scalar:?}"
        );
    }
}

/// A fused multiply-add rounds once, which only a float does.
#[test]
fn an_integer_fused_multiply_add_is_refused() {
    let mut raw = raw();
    let a = raw.int(2);
    let sum = raw.operand(
        OpKind::Fma,
        vec![a, a, a],
        ValTy::Data(Ty::scalar(Scalar::S32)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, sum],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let one = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, one],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Type);
}

/// A predicate has no bit pattern to move between lanes.
#[test]
fn a_shuffle_of_predicates_is_refused() {
    let mut raw = raw();
    let zero = raw.int(0);
    let pred = raw.operand(
        OpKind::Cmp(CmpOp::Lt),
        vec![zero, raw.count],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let one = raw.int(1);
    let moved = raw.operand(
        OpKind::ShflDown,
        vec![pred, one],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let selected = raw.operand(
        OpKind::Un(UnOp::Not),
        vec![moved],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, zero],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let held = raw.float(1.0);
    let other = raw.float(0.0);
    let value = raw.operand(
        OpKind::Select,
        vec![selected, held, other],
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, value],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Type);
}

/// A shuffle names one lane it reads from and the lanes taking part, and each
/// of them is a whole number of the shape the instruction takes.
#[test]
fn a_shuffle_from_a_lane_states_a_lane_and_a_mask() {
    // Each operand is built by an operation that really produces the type it
    // is given, so what a refusal is about is the shuffle and not the operand
    // beside it — which is why the site is asserted as well as the rule.
    let refuse = |value: Scalar, lane: Ty, mask: Ty| {
        let mut raw = raw();
        let moved = match value {
            Scalar::Pred => raw.operand(
                OpKind::Cmp(CmpOp::Lt),
                vec![raw.count, raw.count],
                ValTy::Data(Ty::scalar(Scalar::Pred)),
            ),
            held if held.is_float() => raw.float(1.0),
            held => raw.operand(
                OpKind::IntConst {
                    scalar: held,
                    value: 1,
                },
                Vec::new(),
                ValTy::Data(Ty::scalar(held)),
            ),
        };
        let held = |raw: &mut Raw, ty: Ty| {
            let scalar = if ty.scalar == Scalar::Pred {
                raw.operand(
                    OpKind::Cmp(CmpOp::Lt),
                    vec![raw.count, raw.count],
                    ValTy::Data(Ty::scalar(Scalar::Pred)),
                )
            } else if ty.scalar.is_float() {
                raw.operand(
                    OpKind::FloatConst {
                        scalar: ty.scalar,
                        value: 0.0,
                    },
                    Vec::new(),
                    ValTy::Data(Ty::scalar(ty.scalar)),
                )
            } else {
                raw.operand(
                    OpKind::IntConst {
                        scalar: ty.scalar,
                        value: 0,
                    },
                    Vec::new(),
                    ValTy::Data(Ty::scalar(ty.scalar)),
                )
            };
            match ty.lanes {
                1 => scalar,
                lanes => raw.operand(
                    OpKind::Splat(lanes),
                    vec![scalar],
                    ValTy::Data(Ty::vec(ty.scalar, lanes)),
                ),
            }
        };
        let lane = held(&mut raw, lane);
        let mask = held(&mut raw, mask);
        let shuffled = raw.operand(
            OpKind::ShflFrom,
            vec![moved, lane, mask],
            ValTy::Data(Ty::scalar(value)),
        );
        let zero = raw.int(0);
        let at = raw.operand(
            OpKind::Offset,
            vec![raw.buffer, zero],
            ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            }),
        );
        let cast = raw.operand(
            OpKind::Cast(Scalar::F32),
            vec![shuffled],
            ValTy::Data(Ty::scalar(Scalar::F32)),
        );
        let write = raw.place(
            OpKind::Write {
                mode: WriteMode::Set,
                evict: Evict::Normal,
            },
            vec![at, cast],
            &[],
        );
        let held = raw
            .f
            .defining_op(shuffled)
            .expect("the shuffle produced it");
        verify_launch(&launched(raw.body(vec![write])))
            .err()
            .map(|why| (why.rule, why.at == crate::verify::Site::Op(held)))
    };
    let word = Ty::scalar(Scalar::U32);
    // A lane and a mask the instruction has a form for.
    assert_eq!(refuse(Scalar::F32, word, word), None);
    assert_eq!(refuse(Scalar::F32, Ty::scalar(Scalar::S32), word), None);
    // A predicate has no bit pattern to move between lanes.
    assert_eq!(refuse(Scalar::Pred, word, word), Some((Rule::Type, true)));
    // `shfl.sync` takes its lane in a 32-bit word whatever it moves, so
    // anything narrower, wider or floating is no lane it can be given.
    for held in [
        Scalar::Pred,
        Scalar::S8,
        Scalar::U8,
        Scalar::S16,
        Scalar::U16,
        Scalar::S64,
        Scalar::U64,
        Scalar::F32,
        Scalar::F64,
    ] {
        assert_eq!(
            refuse(Scalar::F32, Ty::scalar(held), word),
            Some((Rule::Type, true)),
            "a source lane of {held:?}"
        );
    }
    // A vector of lanes names no one lane.
    assert_eq!(
        refuse(Scalar::F32, Ty::vec(Scalar::U32, 2), word),
        Some((Rule::Type, true))
    );
    // The mask is a bit per lane, not a vector of them and not another width.
    for held in [
        Ty::vec(Scalar::U32, 2),
        Ty::scalar(Scalar::F32),
        Ty::scalar(Scalar::S32),
        Ty::scalar(Scalar::U64),
    ] {
        assert_eq!(
            refuse(Scalar::F32, word, held),
            Some((Rule::Type, true)),
            "participants held as {held:?}"
        );
    }
}

/// The lanes that are there are a word, whatever a caller says they are.
#[test]
fn an_active_mask_of_another_type_is_refused() {
    let mut raw = raw();
    let held = raw.operand(
        OpKind::ActiveMask,
        Vec::new(),
        ValTy::Data(Ty::scalar(Scalar::S64)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, held],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let one = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, one],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Type);
}

/// One address is one cell: a wide value would be several additions and several
/// previous values, which is not one indivisible step.
#[test]
fn an_atomic_add_of_a_vector_is_refused() {
    let mut raw = raw();
    let one = raw.float(1.0);
    let wide = raw.operand(
        OpKind::Splat(2),
        vec![one],
        ValTy::Data(Ty::vec(Scalar::F32, 2)),
    );
    let add = raw.place(
        OpKind::FetchAdd,
        vec![raw.buffer, wide],
        &[ValTy::Data(Ty::vec(Scalar::F32, 2))],
    );
    assert_eq!(refuses(raw.body(vec![add])).rule, Rule::Type);
}

/// A special register reads as an unsigned word, whatever a caller says it
/// produces.
#[test]
fn a_special_register_of_another_type_is_refused() {
    let mut raw = raw();
    let held = raw.operand(
        OpKind::Sreg(Sreg::TidX),
        Vec::new(),
        ValTy::Data(Ty::scalar(Scalar::S64)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, held],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let one = raw.float(1.0);
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, one],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Type);
}

/// A constant array is read at a whole number of its own elements.
#[test]
fn a_constant_array_of_partial_elements_is_refused() {
    let mut raw = raw();
    let index = raw.int(0);
    let held = raw.operand(
        OpKind::ConstLoad(Arc::new(ConstArray {
            name: "table".into(),
            scalar: Scalar::F32,
            data: vec![0u8; 6].into(),
        })),
        vec![index],
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    let at = raw.operand(
        OpKind::Offset,
        vec![raw.buffer, index],
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let write = raw.place(
        OpKind::Write {
            mode: WriteMode::Set,
            evict: Evict::Normal,
        },
        vec![at, held],
        &[],
    );
    assert_eq!(refuses(raw.body(vec![write])).rule, Rule::Memory);
}

#[test]
fn an_if_requires_a_scalar_predicate() {
    let mut raw = raw();
    let then_block = raw.f.new_block();
    let then_region = raw.f.new_region(vec![then_block]).expect("fresh block");
    let else_block = raw.f.new_block();
    let else_region = raw.f.new_region(vec![else_block]).expect("fresh block");
    let condition = raw.int(1);
    let conditional = raw
        .f
        .new_op(
            OpKind::If,
            true,
            vec![condition],
            &[],
            Vec::new(),
            vec![then_region, else_region],
        )
        .expect("structurally valid if");
    assert_eq!(refuses(raw.body(vec![conditional])).rule, Rule::Type);
}

#[test]
fn a_loop_requires_an_integral_index_and_matching_header() {
    let mut raw = raw();
    let inner = raw.f.new_block();
    raw.f
        .new_block_param(inner, ValTy::Data(Ty::scalar(Scalar::F32)))
        .expect("fresh block");
    let region = raw.f.new_region(vec![inner]).expect("fresh block");
    let start = raw.float(0.0);
    let stop = raw.float(4.0);
    let step = raw.float(1.0);
    let looped = raw
        .f
        .new_op(
            OpKind::Loop {
                index: Scalar::F32,
                trip: Trip::PerThread,
            },
            true,
            vec![start, stop, step],
            &[],
            Vec::new(),
            vec![region],
        )
        .expect("structurally valid loop");
    assert_eq!(refuses(raw.body(vec![looped])).rule, Rule::Type);
}

#[test]
fn region_results_cannot_be_pointers() {
    let mut raw = raw();
    let inner = raw.f.new_block();
    raw.f
        .new_block_param(inner, ValTy::Data(Ty::scalar(Scalar::S32)))
        .expect("loop index");
    raw.f
        .new_block_param(
            inner,
            ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            }),
        )
        .expect("carried pointer");
    raw.f
        .set_yields(inner, vec![raw.buffer])
        .expect("same arena");
    let region = raw.f.new_region(vec![inner]).expect("fresh block");
    let start = raw.int(0);
    let stop = raw.int(4);
    let step = raw.int(1);
    let looped = raw
        .f
        .new_op(
            OpKind::Loop {
                index: Scalar::S32,
                trip: Trip::PerThread,
            },
            true,
            vec![start, stop, step, raw.buffer],
            &[ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            })],
            Vec::new(),
            vec![region],
        )
        .expect("structurally valid loop");
    assert_eq!(refuses(raw.body(vec![looped])).rule, Rule::Type);
}

#[test]
fn a_scope_takes_and_produces_nothing() {
    let build = |has_operand: bool, results: Vec<ValTy>, parameter: Option<ValTy>| {
        let mut raw = raw();
        let inner = raw.f.new_block();
        if let Some(ty) = parameter {
            raw.f.new_block_param(inner, ty).expect("fresh block");
        }
        let region = raw.f.new_region(vec![inner]).expect("fresh block");
        let operands = has_operand.then_some(raw.count).into_iter().collect();
        let scope = raw
            .f
            .new_op(
                OpKind::Scope,
                true,
                operands,
                &results,
                Vec::new(),
                vec![region],
            )
            .expect("structurally valid scope");
        refuses(raw.body(vec![scope]))
    };

    assert_eq!(build(true, Vec::new(), None).rule, Rule::Type);
    assert_eq!(
        build(false, vec![ValTy::Data(Ty::scalar(Scalar::F32))], None,).rule,
        Rule::Type
    );
    assert_eq!(
        build(
            false,
            Vec::new(),
            Some(ValTy::Data(Ty::scalar(Scalar::S32))),
        )
        .rule,
        Rule::Region
    );
}

#[test]
fn an_if_cannot_produce_a_pointer() {
    let mut raw = raw();
    let then_block = raw.f.new_block();
    raw.f
        .set_yields(then_block, vec![raw.buffer])
        .expect("same arena");
    let then_region = raw.f.new_region(vec![then_block]).expect("fresh block");
    let else_block = raw.f.new_block();
    raw.f
        .set_yields(else_block, vec![raw.buffer])
        .expect("same arena");
    let else_region = raw.f.new_region(vec![else_block]).expect("fresh block");
    let zero = raw.int(0);
    let condition = raw.operand(
        OpKind::Cmp(CmpOp::Eq),
        vec![zero, raw.count],
        ValTy::Data(Ty::scalar(Scalar::Pred)),
    );
    let conditional = raw
        .f
        .new_op(
            OpKind::If,
            true,
            vec![condition],
            &[ValTy::Ptr(PtrTy {
                elem: Scalar::F32,
                space: Space::Global,
            })],
            Vec::new(),
            vec![then_region, else_region],
        )
        .expect("structurally valid if");
    assert_eq!(refuses(raw.body(vec![conditional])).rule, Rule::Type);
}
