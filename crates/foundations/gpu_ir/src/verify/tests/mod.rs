// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the verifier refuses, one rule at a time.
//!
//! Each of these is a program the renderer would otherwise meet halfway through
//! emitting text. Building them takes the arena directly, because the builder
//! refuses most of them at construction — which is the point: a rule holds
//! either by construction or by verification, and these prove the second.
//!
//! One file per rule group, each reading the harness below.

mod converge;
mod memory;
mod module;
mod signature;
mod structure;
mod typing;

use std::sync::Arc;

use super::{Rule, verify, verify_launch};
use crate::arena::{Declaration, Function, ValueId};
use crate::build::Builder;
use crate::func::{DeviceFunction, Launch};
use crate::op::{ConstArray, OpKind, Trip, WriteMode};
use crate::{
    BinOp, CmpOp, Counted, Evict, Param, ParamTy, PtrTy, Scalar, SharedArr, Space, Sreg, Ty, UnOp,
    ValTy,
};

/// A launch of `body` at 64 threads.
fn launched(body: Function) -> Launch {
    Launch::new(DeviceFunction::new("k", body), 64, None)
}

fn refuses(body: Function) -> super::Diagnostic {
    verify_launch(&launched(body)).expect_err("this program breaks a rule")
}

fn accepts(body: Function) {
    verify_launch(&launched(body)).expect("this program breaks no rule");
}

/// A function with one `f32` buffer parameter and one scalar, and the arena
/// under construction.
struct Raw {
    f: Function,
    buffer: ValueId,
    count: ValueId,
}

fn raw() -> Raw {
    let mut f = Function::new();
    let buffer = f.declare(
        Declaration::Param(Param {
            name: "buffer".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        }),
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Global,
        }),
    );
    let count = f.declare(
        Declaration::Param(Param {
            name: "count".into(),
            ty: ParamTy::Scalar(Scalar::S32),
        }),
        ValTy::Data(Ty::scalar(Scalar::S32)),
    );
    Raw { f, buffer, count }
}

impl Raw {
    fn operand(&mut self, kind: OpKind, operands: Vec<ValueId>, ty: ValTy) -> ValueId {
        let op = self
            .f
            .add_op(kind, false, operands, &[ty], Vec::new(), Vec::new());
        self.f.op(op).results[0]
    }

    fn int(&mut self, value: i64) -> ValueId {
        self.operand(
            OpKind::IntConst {
                scalar: Scalar::S32,
                value,
            },
            Vec::new(),
            ValTy::Data(Ty::scalar(Scalar::S32)),
        )
    }

    fn float(&mut self, value: f64) -> ValueId {
        self.operand(
            OpKind::FloatConst {
                scalar: Scalar::F32,
                value,
            },
            Vec::new(),
            ValTy::Data(Ty::scalar(Scalar::F32)),
        )
    }

    fn place(&mut self, kind: OpKind, operands: Vec<ValueId>, tys: &[ValTy]) -> crate::OpId {
        self.f
            .add_op(kind, true, operands, tys, Vec::new(), Vec::new())
    }

    /// Finishes with every operation in a block. The explicitly supplied
    /// operations retain their requested order after the value-producing
    /// operations the test constructed around them.
    fn body(mut self, steps: Vec<crate::OpId>) -> Function {
        let entry = self.f.entry();
        let explicit = steps
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let mut ops = self
            .f
            .live_ops()
            .into_iter()
            .filter(|op| {
                !explicit.contains(op) && self.f.step_of(*op).expect("a live operation").is_none()
            })
            .collect::<Vec<_>>();
        ops.extend(steps);
        self.f.set_block_ops(entry, ops);
        self.f
    }
}
