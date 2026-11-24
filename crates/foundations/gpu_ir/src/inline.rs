// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Running a callee's body where it was called.
//!
//! The backend never sees a call: this pass runs before rendering and leaves
//! one body per launched function. It is the two-arena copy with the arguments
//! standing in for the callee's parameters, so a call is expanded by the same
//! machinery that carries a program between arenas.
//!
//! A function with no call in it is left exactly as its builder wrote it —
//! same values, same order, same arena. Rebuilding one to no purpose would
//! renumber nothing that matters and cost what it costs, and a pass that does
//! nothing should do nothing.

use crate::arena::{BlockId, Declaration, Function};
use crate::build::{Body, Builder, Calls};
use crate::func::{DeviceFunction, Implementation};
use crate::op::OpKind;

/// Whether anything this function runs, at any depth, calls one with a body.
///
/// A call to a declaration is not one of them: there is no body to run here, so
/// the call is what the backend renders.
pub(crate) fn inlines_anywhere(f: &Function) -> bool {
    inlines(f, f.entry())
}

/// Whether anything this block runs, at any depth, calls a function with a body.
fn inlines(f: &Function, at: BlockId) -> bool {
    f.block(at).ops.iter().any(|&op| {
        matches!(&f.op(op).kind, OpKind::Call(callee) if callee.body().is_some())
            || f.op(op).regions.iter().any(|&region| {
                f.region(region)
                    .blocks
                    .iter()
                    .any(|&inner| inlines(f, inner))
            })
    })
}

/// Replaces every call to a defined function in `function` with the callee's
/// body, and every such call in those bodies too. A call to a declaration is
/// left where it stands.
pub fn inline_calls(function: &mut DeviceFunction) -> Result<(), String> {
    let Implementation::Body(held) = &mut function.implementation else {
        return Ok(());
    };
    if !inlines_anywhere(held) {
        return Ok(());
    }
    let body = Body::of(std::mem::take(held));
    let source = body.function();
    let mut b = Builder::new();
    let mut substitutions = Vec::with_capacity(source.declarations().len());
    for (index, declared) in source.declarations().iter().enumerate() {
        let here = match declared {
            Declaration::Param(param) => b.parameter(param.clone()),
            Declaration::Shared(array) => b.shared_array(array.clone()),
        };
        substitutions.push((source.declared(index), here));
    }
    let mut import = b.import_resolving(&body, &substitutions, Calls::Inlined)?;
    b.place_all(import.ops().to_vec());
    let returned = source.returned().to_vec();
    let mut handed = Vec::with_capacity(returned.len());
    for value in returned {
        handed.push(b.imported(&mut import, source, value)?);
    }
    b.returns(handed);
    function.implementation = Implementation::Body(b.into_function());
    Ok(())
}
