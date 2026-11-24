// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a declaration must be, for a function whose definition is elsewhere.
//!
//! Nothing here is proven against a body — there is none, and inventing one from
//! the bytes of another language is exactly what this crate does not do. What
//! *is* decidable is whether the declaration says something at all: that it
//! states one access per parameter, that the access agrees with what the
//! parameter is, that what it hands back is what a PTX `.func` can hand back,
//! and that every name it carries is one the renderer can print. A declaration
//! past these rules is trusted; whether the bytes behind it really define a
//! symbol of this shape is NVIDIA's linker to say.

use super::module::is_ptx_identifier;
use super::{Diagnostic, Result, Rule, Site};
use crate::func::Signature;
use crate::{ParamTy, Scalar, Space, ValTy};

pub(super) fn check(name: &str, signature: &Signature) -> Result {
    let refuse = |detail: String| -> Diagnostic {
        Diagnostic {
            function: name.to_string(),
            rule: Rule::Abi,
            at: Site::Function,
            detail,
        }
    };
    let require = |held: bool, detail: &dyn Fn() -> String| -> Result {
        match held {
            true => Ok(()),
            false => Err(refuse(detail())),
        }
    };
    require(is_ptx_identifier(name), &|| {
        format!("{name:?} is a symbol and not a name PTX can be given")
    })?;
    let taken = signature.parameters.len();
    let stated = signature.accesses.len();
    require(taken == stated, &|| {
        format!("it takes {taken} arguments and states what it does to {stated}")
    })?;
    for (slot, param) in signature.parameters.iter().enumerate() {
        require(is_ptx_identifier(&param.name), &|| {
            format!(
                "argument {slot} is called {:?}, which is not a name PTX can be given",
                param.name
            )
        })?;
        let access = signature.accesses[slot];
        match param.ty {
            ParamTy::Scalar(scalar) => {
                require(!access.touched() && access.lanes == 0, &|| {
                    format!("argument {slot} is a datum and it states memory reached through it")
                })?;
                passable(scalar, &|| format!("argument {slot}"), &refuse)?;
            }
            ParamTy::Ptr(_) if access.touched() => {
                require(access.lanes.is_power_of_two(), &|| {
                    format!(
                        "it reaches argument {slot} {} elements at a time, which is no access width",
                        access.lanes
                    )
                })?;
            }
            ParamTy::Ptr(_) => {
                require(access.lanes == 0, &|| {
                    format!(
                        "it touches nothing through argument {slot} and states a width of {} \
                         elements there",
                        access.lanes
                    )
                })?;
            }
        }
    }
    // PTX gives a `.func` one return parameter, so one value is the most a
    // declaration can hand back.
    let results = signature.results.len();
    require(results <= 1, &|| {
        format!("it hands back {results} values and a PTX function hands back at most one")
    })?;
    for (slot, &result) in signature.results.iter().enumerate() {
        match result {
            ValTy::Data(ty) => {
                require(ty.lanes == 1, &|| {
                    format!(
                        "result {slot} holds {} lanes and a return parameter holds one value",
                        ty.lanes
                    )
                })?;
                passable(ty.scalar, &|| format!("result {slot}"), &refuse)?;
            }
            ValTy::Ptr(ptr) => {
                require(ptr.space == Space::Global, &|| {
                    format!("result {slot} is an address into {:?} memory", ptr.space)
                })?;
            }
        }
    }
    Ok(())
}

/// Which scalars cross a call. A predicate has no `.param` of its own — it is a
/// register state, not a width — so it is the one thing this ABI cannot carry.
fn passable(
    scalar: Scalar,
    what: &dyn Fn() -> String,
    refuse: &dyn Fn(String) -> Diagnostic,
) -> Result {
    match scalar {
        Scalar::Pred => Err(refuse(format!(
            "{} is a predicate, which no call parameter holds",
            what()
        ))),
        _ => Ok(()),
    }
}
