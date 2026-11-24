// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building an arena: who owns a value, what a binding is, and how a program
//! crosses from one arena into another.

mod binding;
mod calling;
mod declaring;
mod importing;
mod ownership;
mod placement;

use super::{Body, Builder, Value};
use crate::arena::{Function, OpId};
use crate::op::OpKind;
use crate::{Param, ParamTy, Scalar};

/// A scalar `f32` parameter, which is the cheapest thing to read.
fn scalar(b: &mut Builder, name: &str) -> Value {
    b.parameter(Param {
        name: name.into(),
        ty: ParamTy::Scalar(Scalar::F32),
    })
}

fn pointer(b: &mut Builder, name: &str) -> Value {
    b.parameter(Param {
        name: name.into(),
        ty: ParamTy::Ptr(Scalar::F32),
    })
}

/// The names the steps of a body bind, in order — unnamed steps skipped.
fn bound_names(f: &Function, ops: &[OpId]) -> Vec<String> {
    ops.iter()
        .flat_map(|&op| f.op(op).results.clone())
        .filter_map(|result| f.name(result).map(str::to_owned))
        .collect()
}

fn kinds(f: &Function, ops: &[OpId]) -> Vec<OpKind> {
    ops.iter().map(|&op| f.op(op).kind.clone()).collect()
}

/// The steps of a finished body.
fn steps(held: &Body) -> Vec<OpId> {
    held.ops().to_vec()
}
