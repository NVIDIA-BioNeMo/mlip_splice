// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a declaration must state to be one, and the refusals a caller writing
//! one by hand will meet. Nothing here is about the bytes on the other side of
//! the link — only about whether the contract says something at all.

use super::{Rule, verify};
use crate::effect::Sync;
use crate::func::{Access, DeviceFunction, Signature};
use crate::{Param, ParamTy, Scalar, Ty, ValTy};

/// `scale(dst, x) -> f32`, the declaration every rule below is a deviation from.
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
                read: true,
                written: true,
                lanes: 4,
            },
            Access::default(),
        ],
        sync: Some(Sync::Block),
    }
}

fn refuses(name: &str, signature: Signature) -> String {
    let refused = verify(&DeviceFunction::declared(name, signature))
        .expect_err("this declaration states no contract");
    assert_eq!(refused.rule, Rule::Abi, "{refused}");
    refused.detail
}

#[test]
fn a_complete_declaration_is_accepted() {
    verify(&DeviceFunction::declared("scale", scale())).expect("it states a contract");
}

/// One access per argument. A list of another length answers about arguments
/// that are not there or leaves ones that are unanswered.
#[test]
fn an_access_list_of_another_length_is_refused() {
    let mut signature = scale();
    signature.accesses.pop();
    let held = refuses("scale", signature);
    assert!(held.contains("2 arguments"), "{held}");
}

/// A datum is not memory: there is nothing behind it to reach.
#[test]
fn memory_reached_through_a_datum_is_refused() {
    let mut signature = scale();
    signature.accesses[1] = Access {
        read: true,
        written: false,
        lanes: 1,
    };
    let held = refuses("scale", signature);
    assert!(held.contains("is a datum"), "{held}");
}

/// A width is a number of elements one instruction moves, so it is a power of
/// two, and an argument nothing reaches has none.
#[test]
fn an_access_width_that_is_no_width_is_refused() {
    let mut signature = scale();
    signature.accesses[0].lanes = 3;
    let held = refuses("scale", signature);
    assert!(held.contains("no access width"), "{held}");

    let mut signature = scale();
    signature.accesses[0] = Access {
        read: false,
        written: false,
        lanes: 4,
    };
    let held = refuses("scale", signature);
    assert!(held.contains("touches nothing"), "{held}");
}

/// PTX gives a function one return parameter.
#[test]
fn more_than_one_result_is_refused() {
    let mut signature = scale();
    signature.results.push(ValTy::Data(Ty::scalar(Scalar::F32)));
    let held = refuses("scale", signature);
    assert!(held.contains("at most one"), "{held}");
}

/// A return parameter holds one value, not a vector of them.
#[test]
fn a_vector_result_is_refused() {
    let mut signature = scale();
    signature.results = vec![ValTy::Data(Ty::vec(Scalar::F32, 4))];
    let held = refuses("scale", signature);
    assert!(held.contains("4 lanes"), "{held}");
}

/// A predicate is a register state rather than a width, so nothing carries one
/// across a call.
#[test]
fn a_predicate_across_the_call_boundary_is_refused() {
    let mut signature = scale();
    signature.parameters[1].ty = ParamTy::Scalar(Scalar::Pred);
    let held = refuses("scale", signature);
    assert!(held.contains("argument 1 is a predicate"), "{held}");

    let mut signature = scale();
    signature.results = vec![ValTy::Data(Ty::scalar(Scalar::Pred))];
    let held = refuses("scale", signature);
    assert!(held.contains("result 0 is a predicate"), "{held}");
}

/// An address into the block's own memory belongs to the kernel that was
/// launched, and nothing outside a body can be handed one.
#[test]
fn a_shared_memory_result_is_refused() {
    let mut signature = scale();
    signature.results = vec![ValTy::Ptr(crate::PtrTy {
        elem: Scalar::F32,
        space: crate::Space::Shared,
    })];
    let held = refuses("scale", signature);
    assert!(held.contains("Shared"), "{held}");
}

/// Every name in a declaration is printed into the module's text, so every one
/// must be a name PTX can be given.
#[test]
fn a_name_ptx_cannot_be_given_is_refused() {
    let held = refuses("sc ale", scale());
    assert!(held.contains("is a symbol"), "{held}");

    let mut signature = scale();
    signature.parameters[0].name = "d-st".into();
    let held = refuses("scale", signature);
    assert!(held.contains("argument 0 is called"), "{held}");
}
