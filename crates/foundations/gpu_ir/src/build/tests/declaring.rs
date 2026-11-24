// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A callee whose definition is elsewhere: the contract it states, the rules
//! that contract answers to, and the call the backend leaves for the linker.

use std::sync::Arc;

use super::{Builder, pointer};
use crate::effect::Sync;
use crate::func::{Access, DeviceFunction, Launch, Signature};
use crate::op::OpKind;
use crate::ptx::render_module;
use crate::verify::{Rule, verify_launch};
use crate::{Param, ParamTy, Scalar, Ty, ValTy, inline_calls};

/// `scale(out, x) -> f32`: writes one `f32` through its first argument, one lane
/// at a time, and hands back a value. Nothing about it is proven — every fact is
/// what the declaration says.
fn scale() -> Signature {
    Signature {
        parameters: vec![
            Param {
                name: "dst".into(),
                ty: ParamTy::Ptr(Scalar::F32),
            },
            Param {
                name: "x".into(),
                ty: ParamTy::Scalar(Scalar::F32),
            },
        ],
        results: vec![ValTy::Data(Ty::scalar(Scalar::F32))],
        accesses: vec![
            Access {
                read: false,
                written: true,
                lanes: 1,
            },
            Access::default(),
        ],
        sync: None,
    }
}

fn declared(name: &str, signature: Signature) -> Arc<DeviceFunction> {
    Arc::new(DeviceFunction::declared(name, signature))
}

/// A kernel that calls `scale` on its output buffer and stores what comes back.
fn caller(callee: Arc<DeviceFunction>) -> Launch {
    let mut b = Builder::new();
    let source = pointer(&mut b, "in");
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let at = b.offset(source, zero.clone());
    let read = b.load(at, 1, true);
    let read = b.bind_value_as("read", Ty::scalar(Scalar::F32), read);
    let to = b.offset(out, zero);
    let to = b.bind_pointer_as(
        "to",
        crate::PtrTy {
            elem: Scalar::F32,
            space: crate::Space::Global,
        },
        to,
    );
    let handed = b
        .call(callee, vec![to.clone(), read])
        .into_iter()
        .next()
        .expect("one result");
    b.store(to, handed);
    Launch::new(DeviceFunction::new("k", b.into_function()), 64, None)
}

#[test]
fn a_declared_callee_type_checks_where_it_is_called() {
    let launch = caller(declared("scale", scale()));
    verify_launch(&launch).expect("the call agrees with the declaration");
    let f = launch.body();
    let call = f
        .block(f.entry())
        .ops
        .iter()
        .copied()
        .find(|&op| matches!(f.op(op).kind, OpKind::Call(_)))
        .expect("the body calls");
    assert_eq!(f.data_ty(f.op(call).results[0]), Ty::scalar(Scalar::F32));
}

/// An argument of the wrong type is refused where the call is, exactly as it is
/// for a callee with a body.
#[test]
fn a_call_that_disagrees_with_a_declaration_is_refused() {
    let mut signature = scale();
    signature.parameters[1].ty = ParamTy::Scalar(Scalar::F64);
    let launch = caller(declared("scale", signature));
    let refused = verify_launch(&launch).expect_err("the argument is an f32");
    assert_eq!(refused.rule, Rule::Type);
    assert!(refused.detail.contains("scale"), "{refused}");
}

/// Nothing runs where the call stands: there is no body to run, so inlining
/// leaves it exactly where the builder wrote it.
#[test]
fn inlining_leaves_a_declared_call_alone() {
    let mut launch = caller(declared("scale", scale()));
    let before = crate::arena::wire::encode_function(launch.body());
    inline_calls(&mut launch.function).expect("nothing to inline");
    assert_eq!(
        before,
        crate::arena::wire::encode_function(launch.body()),
        "a body with nothing to inline is left as it was written"
    );
}

/// The module declares the symbol it calls and does not define, and the call
/// site moves the arguments through parameter space and reads the result back.
#[test]
fn a_declared_call_renders_as_an_extern_signature_and_a_call() {
    let launch = caller(declared("scale", scale()));
    let ptx = render_module(std::slice::from_ref(&launch), 90).expect("it renders");
    assert!(
        ptx.contains(".extern .func (.param .b32 scale_retval0) scale(.param .b64 dst_param_0, .param .b32 x_param_1);"),
        "{ptx}"
    );
    assert!(ptx.contains("st.param.b64 [scale_1_param_0]"), "{ptx}");
    assert!(ptx.contains("st.param.f32 [scale_1_param_1]"), "{ptx}");
    assert!(
        ptx.contains("call (scale_1_retval0), scale, (scale_1_param_0, scale_1_param_1);"),
        "{ptx}"
    );
    assert!(ptx.contains("ld.param.f32 "), "{ptx}");
    // The entry renderer is untouched: the kernel is still an entry with its
    // launch bound, and the declaration stands outside it.
    assert!(ptx.contains(".visible .entry k("), "{ptx}");
    assert!(ptx.contains(".maxntid 64, 1, 1"), "{ptx}");
    let extern_at = ptx.find(".extern .func").expect("a declaration");
    let entry_at = ptx.find(".visible .entry").expect("an entry");
    assert!(extern_at < entry_at, "{ptx}");
}

/// Two calls of one declaration are two calls and one declaration, and their
/// parameter spaces do not collide.
#[test]
fn two_calls_of_one_declaration_declare_it_once() {
    let callee = declared("scale", scale());
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    let to = b.bind_pointer_as(
        "to",
        crate::PtrTy {
            elem: Scalar::F32,
            space: crate::Space::Global,
        },
        to,
    );
    let one = b.float(Scalar::F32, 1.0);
    let once = b
        .call(callee.clone(), vec![to.clone(), one])
        .into_iter()
        .next()
        .expect("one result");
    let twice = b
        .call(callee, vec![to.clone(), once])
        .into_iter()
        .next()
        .expect("one result");
    b.store(to, twice);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let ptx = render_module(std::slice::from_ref(&launch), 90).expect("it renders");
    assert_eq!(ptx.matches(".extern .func").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("call (scale_").count(), 2, "{ptx}");
    assert!(ptx.contains("scale_1_param_0"), "{ptx}");
    assert!(ptx.contains("scale_2_param_0"), "{ptx}");
}

/// A declaration that hands nothing back renders as a call with no return
/// parameter, and one that takes nothing renders with no argument list.
#[test]
fn a_declaration_that_neither_takes_nor_hands_back_renders_bare() {
    let callee = declared(
        "tick",
        Signature {
            parameters: Vec::new(),
            results: Vec::new(),
            accesses: Vec::new(),
            sync: None,
        },
    );
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    b.call(callee, Vec::new());
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    let value = b.float(Scalar::F32, 1.0);
    b.store(to, value);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let ptx = render_module(std::slice::from_ref(&launch), 90).expect("it renders");
    assert!(ptx.contains(".extern .func tick();"), "{ptx}");
    assert!(ptx.contains("call tick;"), "{ptx}");
}

/// Each parameter is a slot of its own width, and a sub-word integer is written
/// and read back through the typed instructions that width has — not widened
/// into the register that happens to hold it.
#[test]
fn a_sub_word_argument_crosses_as_its_own_width() {
    let callee = declared(
        "narrow",
        Signature {
            parameters: vec![Param {
                name: "byte".into(),
                ty: ParamTy::Scalar(Scalar::S8),
            }],
            results: vec![ValTy::Data(Ty::scalar(Scalar::S8))],
            accesses: vec![Access::default()],
            sync: None,
        },
    );
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let byte = b.int(Scalar::S8, 3);
    let handed = b
        .call(callee, vec![byte])
        .into_iter()
        .next()
        .expect("one result");
    let widened = b.cast(Scalar::F32, handed);
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    b.store(to, widened);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let ptx = render_module(std::slice::from_ref(&launch), 90).expect("it renders");
    assert!(
        ptx.contains(".extern .func (.param .b8 narrow_retval0) narrow(.param .b8 byte_param_0);"),
        "{ptx}"
    );
    assert!(ptx.contains("st.param.s8 [narrow_1_param_0]"), "{ptx}");
    assert!(ptx.contains("ld.param.s8 "), "{ptx}");
}

/// A declaration is callable and not launchable: there is no body to enter.
#[test]
fn launching_a_declaration_is_refused() {
    let launch = Launch::new(DeviceFunction::declared("scale", scale()), 64, None);
    let refused = verify_launch(&launch).expect_err("a launch runs a body");
    assert_eq!(refused.rule, Rule::Abi);
    assert!(refused.detail.contains("a body"), "{refused}");
}

/// One symbol is one function: an entry and a declaration under one name would
/// be two definitions of it.
#[test]
fn a_declaration_that_shadows_an_entry_is_refused() {
    let launch = caller(declared("k", scale()));
    let refused = render_module(std::slice::from_ref(&launch), 90)
        .expect_err("the entry is called as an external function");
    assert_eq!(refused.rule, Rule::Abi);
    assert!(refused.detail.contains("entered here"), "{refused}");
}

/// One symbol has one contract, because the module declares it once.
#[test]
fn one_symbol_under_two_contracts_is_refused() {
    let mut other = scale();
    other.results = Vec::new();
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    let to = b.bind_pointer_as(
        "to",
        crate::PtrTy {
            elem: Scalar::F32,
            space: crate::Space::Global,
        },
        to,
    );
    let one = b.float(Scalar::F32, 1.0);
    let handed = b
        .call(declared("scale", scale()), vec![to.clone(), one.clone()])
        .into_iter()
        .next()
        .expect("one result");
    b.call(declared("scale", other), vec![to.clone(), one]);
    b.store(to, handed);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let refused = render_module(std::slice::from_ref(&launch), 90)
        .expect_err("one symbol is called under two contracts");
    assert_eq!(refused.rule, Rule::Abi);
    assert!(refused.detail.contains("two contracts"), "{refused}");
}

/// A declaration that says it reaches memory through an argument holding a
/// datum has said something unanswerable, and the effect rule says so.
#[test]
fn a_declared_access_through_a_datum_is_refused() {
    let mut signature = scale();
    signature.accesses[1] = Access {
        read: true,
        written: false,
        lanes: 1,
    };
    signature.parameters[1].ty = ParamTy::Ptr(Scalar::F32);
    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    let to = b.bind_pointer_as(
        "to",
        crate::PtrTy {
            elem: Scalar::F32,
            space: crate::Space::Global,
        },
        to,
    );
    // The call is given a datum where the declaration states a pointer, so the
    // effect it names cannot be answered.
    let one = b.float(Scalar::F32, 1.0);
    let handed = b
        .call(declared("scale", signature), vec![to.clone(), one])
        .into_iter()
        .next()
        .expect("one result");
    b.store(to, handed);
    let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    let refused = verify_launch(&launch).expect_err("operand 1 holds a datum");
    assert!(
        matches!(refused.rule, Rule::Effect | Rule::Type),
        "{refused}"
    );
}

/// Importing carries a retained call as one operation while it runs a
/// body-backed callee where it stood, so one body may hold both kinds and a
/// copy of it holds exactly what it held.
#[test]
fn importing_inlines_the_body_and_carries_the_declaration() {
    let mut c = Builder::new();
    let x = c.parameter(Param {
        name: "x".into(),
        ty: ParamTy::Scalar(Scalar::F32),
    });
    let one = c.float(Scalar::F32, 1.0);
    let sum = c.add(x, one);
    let sum = c.bind_value_as("sum", Ty::scalar(Scalar::F32), sum);
    c.returns(vec![sum]);
    let defined = Arc::new(DeviceFunction::new("add_one", c.into_function()));

    let mut b = Builder::new();
    let out = pointer(&mut b, "out");
    let zero = b.i32(0);
    let to = b.offset(out, zero);
    let to = b.bind_pointer_as(
        "to",
        crate::PtrTy {
            elem: Scalar::F32,
            space: crate::Space::Global,
        },
        to,
    );
    let start = b.float(Scalar::F32, 1.0);
    let stepped = b
        .call(defined, vec![start])
        .into_iter()
        .next()
        .expect("one result");
    let handed = b
        .call(declared("scale", scale()), vec![to.clone(), stepped])
        .into_iter()
        .next()
        .expect("one result");
    b.store(to, handed);
    let mut launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
    verify_launch(&launch).expect("both calls agree with their callees");
    inline_calls(&mut launch.function).expect("the defined callee is inlinable");
    let f = launch.body();
    let retained = f
        .live_ops()
        .into_iter()
        .filter_map(|op| match &f.op(op).kind {
            OpKind::Call(callee) => Some(callee.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(retained, vec!["scale".to_string()]);
    // The inlined body left its named step behind, which is what running it
    // where it stood means.
    assert!(
        f.live_ops()
            .into_iter()
            .flat_map(|op| f.op(op).results.clone())
            .any(|result| f.name(result) == Some("sum")),
        "the defined callee ran where it was called"
    );
}

/// An editor may point a call at a callee of either kind: which one it runs is
/// the module's fact, and the call is the same operation either way.
#[test]
fn an_editor_may_retarget_a_call_at_a_declaration() {
    let mut launch = caller(declared("scale", scale()));
    let f = launch
        .function
        .body_mut()
        .expect("a kernel is given by a body");
    let call = f
        .live_ops()
        .into_iter()
        .find(|&op| matches!(f.op(op).kind, OpKind::Call(_)))
        .expect("the body calls");
    let mut other = scale();
    other.parameters[1].ty = ParamTy::Scalar(Scalar::F64);
    f.set_callee(call, declared("other", other))
        .expect("a call runs whichever function it is pointed at");
    let refused = verify_launch(&launch).expect_err("the new contract takes an f64");
    assert_eq!(refused.rule, Rule::Type);
    assert!(refused.detail.contains("other"), "{refused}");
}

/// The threads a declared callee makes a caller wait for are the caller's to
/// answer for: a call to one under a predicate the block need not agree on is a
/// barrier the threads that skip it never reach.
#[test]
fn a_declared_barrier_is_the_callers_convergence_question() {
    let mut signature = scale();
    signature.sync = Some(Sync::Block);
    let callee = declared("scale", signature);
    let guarded = |uniform: bool| {
        let mut b = Builder::new();
        let out = pointer(&mut b, "out");
        let zero = b.i32(0);
        let to = b.offset(out, zero);
        let to = b.bind_pointer_as(
            "to",
            crate::PtrTy {
                elem: Scalar::F32,
                space: crate::Space::Global,
            },
            to,
        );
        let held = match uniform {
            true => b.ctaid_x(),
            false => b.tid_x(),
        };
        let held = b.cast(Scalar::S32, held);
        let four = b.i32(4);
        let inside = b.lt(held, four);
        let callee = callee.clone();
        let inner = to.clone();
        b.if_(inside, move |b| {
            let one = b.float(Scalar::F32, 1.0);
            b.call(callee, vec![inner, one]);
        });
        let one = b.float(Scalar::F32, 1.0);
        b.store(to, one);
        Launch::new(DeviceFunction::new("k", b.into_function()), 64, None)
    };
    let refused = verify_launch(&guarded(false)).expect_err("the threads need not agree");
    assert_eq!(refused.rule, Rule::Convergence);
    verify_launch(&guarded(true)).expect("every thread of the block takes the same arm");
}
