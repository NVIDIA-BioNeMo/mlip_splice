// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operands to instructions: which form a target has for an operation, and
//! which lanes ride which register.
//!
//! A value with storage is read from it. Everything else is an operand and is
//! evaluated here, at the read: an operand read twice is computed twice, which
//! is what makes a load or a warp collective happen exactly as often as it is
//! written.

use super::kernel::K;
use super::types::{Binding, MathK, Operand, Val, arith, load_type, mem, reg_class, subword_wide};
use crate::arena::{OpId, ValueId};
use crate::bits::{bf16_bits, f16_bits};
use crate::op::OpKind;
use crate::{BinOp, CmpOp, Scalar, Space, Sreg, Ty, UnOp};

impl<T: super::text::Text> K<'_, T> {
    /// What a value with storage of its own reads as.
    fn stored(binding: &Binding) -> Val {
        match binding {
            Binding::LoopState { regs, ty } | Binding::Data { regs, ty } => Val::Data {
                ops: regs.iter().cloned().map(Operand::Reg).collect(),
                ty: *ty,
            },
            Binding::Ptr { reg, ty } => Val::Ptr {
                reg: reg.clone(),
                byte_off: 0,
                ty: *ty,
            },
        }
    }

    pub(super) fn eval(&mut self, value: ValueId) -> Val {
        enum Frame {
            Evaluate(ValueId),
            Apply { op: OpId, arity: usize },
        }

        let mut frames = vec![Frame::Evaluate(value)];
        let mut values = Vec::new();
        while let Some(frame) = frames.pop() {
            match frame {
                Frame::Evaluate(value) => {
                    if let Some(binding) = self.held(value) {
                        values.push(Self::stored(binding));
                        continue;
                    }
                    let op = self
                        .f
                        .operand_op(value)
                        .expect("a value with no storage is an operand");
                    if matches!(&self.f.op(op).kind, OpKind::Loop { .. }) {
                        values.push(self.operand_loop(op, value));
                        continue;
                    }
                    let operands = &self.f.op(op).operands;
                    frames.push(Frame::Apply {
                        op,
                        arity: operands.len(),
                    });
                    frames.extend(operands.iter().rev().copied().map(Frame::Evaluate));
                }
                Frame::Apply { op, arity } => {
                    let first = values
                        .len()
                        .checked_sub(arity)
                        .expect("one value for every evaluated operand");
                    let result = self.eval_evaluated(op, values.drain(first..));
                    values.push(result);
                }
            }
        }
        assert_eq!(values.len(), 1, "one result from operand evaluation");
        values.pop().unwrap()
    }

    #[allow(clippy::cognitive_complexity)]
    fn eval_evaluated(&mut self, op: OpId, mut evaluated: impl Iterator<Item = Val>) -> Val {
        let f = self.f;
        let mut operand = || evaluated.next().expect("an evaluated operand");
        match &f.op(op).kind {
            OpKind::IntConst { scalar, value } => {
                assert!(
                    !scalar.is_float() && *scalar != Scalar::Pred,
                    "integer constant of {scalar:?}"
                );
                Val::Data {
                    ops: vec![Operand::ImmI(*value)],
                    ty: Ty::scalar(*scalar),
                }
            }
            OpKind::FloatConst { scalar, value } => {
                let op = match scalar {
                    Scalar::F32 | Scalar::F64 => Operand::ImmF(*value),
                    Scalar::F16 | Scalar::BF16 => {
                        let bits = if *scalar == Scalar::F16 {
                            f16_bits(*value)
                        } else {
                            bf16_bits(*value)
                        };
                        let r = self.reg(*scalar);
                        self.ins(format_args!("mov.b16 {r}, 0x{bits:04X};"));
                        Operand::Reg(r)
                    }
                    // Typing admits a float constant only of a float scalar.
                    other => unreachable!("a float constant of {other:?}"),
                };
                Val::Data {
                    ops: vec![op],
                    ty: Ty::scalar(*scalar),
                }
            }
            OpKind::Sreg(sreg) => {
                let name = match sreg {
                    Sreg::TidX => "%tid.x",
                    Sreg::CtaidX => "%ctaid.x",
                    Sreg::CtaidY => "%ctaid.y",
                    Sreg::NtidX => "%ntid.x",
                };
                let r = self.reg(Scalar::U32);
                self.ins(format_args!("mov.u32 {r}, {name};"));
                Val::Data {
                    ops: vec![Operand::Reg(r)],
                    ty: Ty::scalar(Scalar::U32),
                }
            }
            OpKind::ConstLoad(array) => {
                let (name, scalar) = (array.name.clone(), array.scalar);
                let index = operand();
                let (indices, index_ty) = index.data();
                assert_eq!(index_ty.lanes, 1, "constant index vector");
                let index = self.materialize(&indices[0], index_ty.scalar);
                let index64 = match index_ty.scalar {
                    Scalar::S32 | Scalar::U32 => {
                        let widened = self.reg(Scalar::U64);
                        self.ins(format_args!(
                            "cvt.u64.{} {widened}, {index};",
                            arith(index_ty.scalar)
                        ));
                        widened
                    }
                    Scalar::S64 | Scalar::U64 => index,
                    // Typing admits a constant read only at a 32- or 64-bit index.
                    other => unreachable!("a constant read at an index of {other:?}"),
                };
                let offset = self.reg(Scalar::U64);
                self.ins(format_args!(
                    "mul.lo.u64 {offset}, {index64}, {};",
                    scalar.size()
                ));
                let address = self.reg(Scalar::U64);
                self.ins(format_args!("mov.u64 {address}, const_{name};"));
                let pointer = self.reg(Scalar::U64);
                self.ins(format_args!("add.u64 {pointer}, {address}, {offset};"));
                let (regs, ty) = if scalar == Scalar::Pred {
                    let bits = self.reg(Scalar::U16);
                    self.ins(format_args!("ld.global.nc.u8 {bits}, [{pointer}];"));
                    let predicate = self.reg(Scalar::Pred);
                    self.ins(format_args!("setp.ne.u16 {predicate}, {bits}, 0;"));
                    (vec![predicate], Ty::scalar(Scalar::Pred))
                } else {
                    let ty = Ty::scalar(scalar);
                    let regs = self.regs(ty);
                    self.ins(format_args!(
                        "ld.global.nc.{} {}, [{pointer}];",
                        load_type(scalar),
                        regs[0]
                    ));
                    (regs, ty)
                };
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty,
                }
            }
            OpKind::Load { lanes, nc, evict } => {
                let (lanes, nc, evict) = (*lanes, *nc, *evict);
                let p = operand();
                let ty = Ty::vec(
                    match &p {
                        Val::Ptr { ty, .. } => ty.elem,
                        Val::Data { .. } => unreachable!("a load through a data value"),
                    },
                    lanes,
                );
                let regs = self.regs(ty);
                self.load_into(&p, nc, evict, &regs, ty);
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty,
                }
            }
            OpKind::Offset => {
                let p = operand();
                let Val::Ptr { reg, byte_off, ty } = p else {
                    unreachable!("an offset of a data value");
                };
                let (ops, ety) = operand().data();
                assert_eq!(ety.lanes, 1, "pointer offset by a vector");
                assert!(!ety.scalar.is_float(), "pointer offset by {:?}", ety.scalar);
                let scale = ty.elem.size();
                match &ops[0] {
                    Operand::ImmI(v) => Val::Ptr {
                        reg,
                        byte_off: byte_off.wrapping_add(v.wrapping_mul(scale)),
                        ty,
                    },
                    op => {
                        let (wide, mad, cvt_up): (Scalar, &str, Option<&str>) = match ty.space {
                            Space::Global => (
                                Scalar::S64,
                                "mad.lo.s64",
                                match ety.scalar {
                                    Scalar::S32 => Some("cvt.s64.s32"),
                                    Scalar::U32 => Some("cvt.u64.u32"),
                                    _ => None,
                                },
                            ),
                            Space::Shared => (
                                Scalar::S32,
                                "mad.lo.s32",
                                match ety.scalar {
                                    Scalar::S64 | Scalar::U64 => Some("cvt.u32.u64"),
                                    _ => None,
                                },
                            ),
                        };
                        let mut idx = self.materialize(op, ety.scalar);
                        if let Some(cvt) = cvt_up {
                            let w = self.reg(wide);
                            self.ins(format_args!("{cvt} {w}, {idx};"));
                            idx = w;
                        }
                        let r = self.reg(wide);
                        self.ins(format_args!("{mad} {r}, {idx}, {scale}, {reg};"));
                        Val::Ptr {
                            reg: r,
                            byte_off,
                            ty,
                        }
                    }
                }
            }
            OpKind::Un(unop) => {
                let unop = *unop;
                let (ops, ty) = operand().data();
                match unop {
                    UnOp::Neg => {
                        assert_ne!(ty.scalar, Scalar::Pred, "negating a predicate");
                        let regs = self.regs(ty);
                        for (dst, src) in regs.iter().cloned().zip(&ops) {
                            if let Some(wide) = subword_wide(ty.scalar) {
                                let input = self.reg(wide);
                                self.cvt_one(wide, ty.scalar, &input, src);
                                let output = self.reg(wide);
                                self.ins(format_args!("sub.{} {output}, 0, {input};", arith(wide)));
                                self.cvt_one(ty.scalar, wide, &dst, &Operand::Reg(output));
                            } else {
                                let s = self.operand(src, ty.scalar);
                                self.ins(format_args!("neg.{} {dst}, {s};", arith(ty.scalar)));
                            }
                        }
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::Not => {
                        assert!(!ty.scalar.is_float(), "a complement of {:?}", ty.scalar);
                        let regs = self.regs(ty);
                        for (dst, src) in regs.iter().cloned().zip(&ops) {
                            // `not` has a predicate and a bit-width form, and a
                            // sub-word integer has neither: complement it in the
                            // width it is held in and narrow back, which
                            // re-extends the sign or the zeros.
                            if let Some(wide) = subword_wide(ty.scalar) {
                                let input = self.reg(wide);
                                self.cvt_one(wide, ty.scalar, &input, src);
                                let output = self.reg(wide);
                                self.ins(format_args!("not.{} {output}, {input};", mem(wide)));
                                self.cvt_one(ty.scalar, wide, &dst, &Operand::Reg(output));
                            } else {
                                let s = self.materialize(src, ty.scalar);
                                self.ins(format_args!("not.{} {dst}, {s};", mem(ty.scalar)));
                            }
                        }
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::Sqrt => {
                        assert!(ty.scalar.is_float(), "sqrt of {:?}", ty.scalar);
                        // No packed or half-precision sqrt in PTX: lane-split,
                        // and f16/bf16 hop through f32.
                        let via_f32 = matches!(ty.scalar, Scalar::F16 | Scalar::BF16);
                        let regs = self.regs(ty);
                        for (dst, src) in regs.iter().cloned().zip(&ops) {
                            if via_f32 {
                                let w = self.up_f32(src, ty.scalar);
                                let r = self.reg(Scalar::F32);
                                self.ins(format_args!("sqrt.rn.f32 {r}, {w};"));
                                self.ins(format_args!(
                                    "cvt.rn.{}.f32 {dst}, {r};",
                                    arith(ty.scalar)
                                ));
                            } else {
                                let s = self.operand(src, ty.scalar);
                                self.ins(format_args!("sqrt.rn.{} {dst}, {s};", arith(ty.scalar)));
                            }
                        }
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::Exp2 | UnOp::Log2 | UnOp::Sin | UnOp::Cos => {
                        assert!(
                            ty.scalar.is_float() && ty.scalar != Scalar::F64,
                            "an approximate {unop:?} of {:?}",
                            ty.scalar
                        );
                        let instruction = match unop {
                            UnOp::Exp2 => "ex2.approx",
                            UnOp::Log2 => "lg2.approx",
                            UnOp::Sin => "sin.approx",
                            _ => "cos.approx",
                        };
                        let regs = self.approx_f32_lanes(instruction, &ops, ty);
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::Tanh => {
                        assert!(
                            ty.scalar.is_float() && ty.scalar != Scalar::F64,
                            "an approximate tanh of {:?}",
                            ty.scalar
                        );
                        let regs = self.tanh_lanes(&ops, ty);
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::Floor | UnOp::Ceil | UnOp::RoundEven => {
                        assert!(ty.scalar.is_float(), "a {unop:?} of {:?}", ty.scalar);
                        let mode = match unop {
                            UnOp::Floor => "rmi",
                            UnOp::Ceil => "rpi",
                            _ => "rni",
                        };
                        let regs = self.round_lanes(mode, &ops, ty);
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                    UnOp::LeadingZeros | UnOp::OneBits => {
                        assert!(
                            !ty.scalar.is_float() && ty.scalar != Scalar::Pred,
                            "a {unop:?} of {:?}",
                            ty.scalar
                        );
                        let instruction = if unop == UnOp::LeadingZeros {
                            "clz"
                        } else {
                            "popc"
                        };
                        let regs = self.bit_count_lanes(instruction, &ops, ty);
                        Val::Data {
                            ops: regs.into_iter().map(Operand::Reg).collect(),
                            ty,
                        }
                    }
                }
            }
            OpKind::Bin(binop) => {
                let binop = *binop;
                self.bin_evaluated(binop, operand(), operand(), None)
            }
            OpKind::Fma => self.fma_evaluated(operand(), operand(), operand(), None),
            OpKind::Cmp(cmpop) => {
                let cmpop = *cmpop;
                let (aops, aty) = operand().data();
                let (bops, bty) = operand().data();
                assert_eq!(aty, bty, "comparison type mismatch");
                let pty = Ty::vec(Scalar::Pred, aty.lanes);
                let regs = self.regs(pty);
                let cmp = match cmpop {
                    CmpOp::Eq => "eq",
                    CmpOp::Ne => "ne",
                    CmpOp::Lt => "lt",
                    CmpOp::Le => "le",
                    CmpOp::Gt => "gt",
                    CmpOp::Ge => "ge",
                };
                // bf16 setp needs sm_90; go through f32 below it (f16 is universal).
                let via_f32 = aty.scalar == Scalar::BF16 && self.sm < 90;
                let via_wide = subword_wide(aty.scalar);
                for lane in 0..aty.lanes as usize {
                    let (sa, sb, t) = if via_f32 {
                        (
                            self.up_f32(&aops[lane], aty.scalar),
                            self.up_f32(&bops[lane], bty.scalar),
                            "f32",
                        )
                    } else if let Some(wide) = via_wide {
                        let wa = self.reg(wide);
                        self.cvt_one(wide, aty.scalar, &wa, &aops[lane]);
                        let wb = self.reg(wide);
                        self.cvt_one(wide, bty.scalar, &wb, &bops[lane]);
                        (wa, wb, arith(wide))
                    } else {
                        (
                            self.operand(&aops[lane], aty.scalar),
                            self.operand(&bops[lane], bty.scalar),
                            arith(aty.scalar),
                        )
                    };
                    self.ins(format_args!("setp.{cmp}.{t} {}, {sa}, {sb};", regs[lane]));
                }
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty: pty,
                }
            }
            OpKind::Select => self.select_evaluated(operand(), operand(), operand(), None),
            OpKind::Cast(scalar) => {
                let scalar = *scalar;
                let (ops, ty) = operand().data();
                if reg_class(scalar) == reg_class(ty.scalar)
                    && !scalar.is_float()
                    && !ty.scalar.is_float()
                {
                    // same width, different signedness: the bits are the value
                    return Val::Data {
                        ops,
                        ty: Ty::vec(scalar, ty.lanes),
                    };
                }
                if scalar == ty.scalar {
                    return Val::Data { ops, ty };
                }
                let dty = Ty::vec(scalar, ty.lanes);
                let regs = self.regs(dty);
                for (dst, src) in regs.iter().cloned().zip(&ops) {
                    self.cvt_one(scalar, ty.scalar, &dst, src);
                }
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty: dty,
                }
            }
            OpKind::Splat(lanes) => {
                let lanes = *lanes;
                let (ops, ty) = operand().data();
                assert_eq!(ty.lanes, 1, "splat of a vector");
                Val::Data {
                    ops: vec![ops[0].clone(); lanes as usize],
                    ty: Ty::vec(ty.scalar, lanes),
                }
            }
            OpKind::Extract(lane) => {
                let lane = *lane;
                let (ops, ty) = operand().data();
                assert!(lane < ty.lanes, "lane {lane} of {ty:?}");
                Val::Data {
                    ops: vec![ops[lane as usize].clone()],
                    ty: Ty::scalar(ty.scalar),
                }
            }
            // Concatenation: a part contributes its own lanes, in order, so the
            // packed value is the parts' registers end to end and packing costs
            // nothing on its own.
            OpKind::Pack => {
                let mut ops = Vec::with_capacity(f.op(op).operands.len());
                let mut scalar = None;
                for part in evaluated.by_ref() {
                    let (o, ty) = part.data();
                    assert!(
                        scalar.is_none() || scalar == Some(ty.scalar),
                        "pack of mixed scalars"
                    );
                    scalar = Some(ty.scalar);
                    ops.extend(o.iter().cloned());
                }
                let lanes = u8::try_from(ops.len()).expect("pack of too many lanes");
                Val::Data {
                    ops,
                    ty: Ty::vec(scalar.expect("non-empty pack"), lanes),
                }
            }
            OpKind::ShflDown => {
                let (ops, ty) = operand().data();
                let (dops, dty) = operand().data();
                assert_eq!(dty.lanes, 1, "vector shuffle delta");
                let sd = self.operand(&dops[0], dty.scalar);
                let regs = self.regs(ty);
                for (dst, src) in regs.iter().cloned().zip(&ops) {
                    self.shfl_down_one(ty.scalar, &dst, src, &sd);
                }
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty,
                }
            }
            OpKind::ActiveMask => {
                let mask = self.reg(Scalar::U32);
                self.ins(format_args!("activemask.b32 {mask};"));
                Val::Data {
                    ops: vec![Operand::Reg(mask)],
                    ty: Ty::scalar(Scalar::U32),
                }
            }
            OpKind::ShflFrom => {
                let (ops, ty) = operand().data();
                let (lops, lty) = operand().data();
                assert!(
                    lty.lanes == 1 && matches!(lty.scalar, Scalar::S32 | Scalar::U32),
                    "shuffle source lane held as {lty:?}"
                );
                let lane = self.operand(&lops[0], lty.scalar);
                let (mops, mty) = operand().data();
                assert_eq!(
                    mty,
                    Ty::scalar(Scalar::U32),
                    "shuffle participants held as {mty:?}"
                );
                let mask = self.operand(&mops[0], Scalar::U32);
                let regs = self.regs(ty);
                for (dst, src) in regs.iter().cloned().zip(&ops) {
                    self.shfl_from_one(ty.scalar, &dst, src, &lane, &mask);
                }
                Val::Data {
                    ops: regs.into_iter().map(Operand::Reg).collect(),
                    ty,
                }
            }
            OpKind::WarpAny => {
                let (ops, ty) = operand().data();
                assert_eq!(ty.scalar, Scalar::Pred, "warp_any of {ty:?}");
                let mut p = self.materialize(&ops[0], Scalar::Pred);
                for op in &ops[1..] {
                    let lane = self.materialize(op, Scalar::Pred);
                    let merged = self.reg(Scalar::Pred);
                    self.ins(format_args!("or.pred {merged}, {p}, {lane};"));
                    p = merged;
                }
                let mask = self.reg(Scalar::U32);
                self.ins(format_args!("activemask.b32 {mask};"));
                let mut bits = self.reg(Scalar::U32);
                self.ins(format_args!("selp.u32 {bits}, 1, 0, {p};"));
                if self.sm >= 80 {
                    let r = self.reg(Scalar::U32);
                    self.ins(format_args!("redux.sync.or.b32 {r}, {bits}, {mask};"));
                    bits = r;
                } else {
                    // Pre-sm_80 has no redux; butterfly-`or` over the active
                    // mask. (Runs only for render coverage; the device is
                    // sm_80+.)
                    for delta in [16, 8, 4, 2, 1] {
                        let other = self.reg(Scalar::U32);
                        self.ins(format_args!(
                            "shfl.sync.bfly.b32 {other}, {bits}, {delta}, 31, {mask};"
                        ));
                        let merged = self.reg(Scalar::U32);
                        self.ins(format_args!("or.b32 {merged}, {bits}, {other};"));
                        bits = merged;
                    }
                }
                let out = self.reg(Scalar::Pred);
                self.ins(format_args!("setp.ne.u32 {out}, {bits}, 0;"));
                Val::Data {
                    ops: vec![Operand::Reg(out)],
                    ty: Ty::scalar(Scalar::Pred),
                }
            }
            OpKind::Loop { .. } => unreachable!("a loop is evaluated as a region"),
            // An if is a step, and every other kind here must be placed; both
            // are settled before anything is emitted.
            kind => unreachable!("{kind:?} is a step of a body, not an operand"),
        }
    }

    pub(super) fn bin(&mut self, op: BinOp, a: ValueId, b: ValueId, dst: Option<&[String]>) -> Val {
        let a = self.eval(a);
        let b = self.eval(b);
        self.bin_evaluated(op, a, b, dst)
    }

    fn bin_evaluated(&mut self, op: BinOp, a: Val, b: Val, dst: Option<&[String]>) -> Val {
        let (aops, aty) = a.data();
        let (bops, bty) = b.data();
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            assert_eq!(bty.lanes, 1, "vector shift amount");
            assert!(
                matches!(bty.scalar, Scalar::S32 | Scalar::U32),
                "shift amount of {:?}",
                bty.scalar
            );
        } else {
            assert_eq!(
                aty, bty,
                "operand type mismatch on {op:?}: {aty:?} vs {bty:?}"
            );
        }
        // Fold integer constants so address math stays out of the kernel.
        if let (Operand::ImmI(x), Operand::ImmI(y)) = (&aops[0], &bops[0])
            && aty.lanes == 1
            && !aty.scalar.is_float()
        {
            let (x, y) = (*x, *y);
            let v = match op {
                BinOp::Add => x.checked_add(y),
                BinOp::Sub => x.checked_sub(y),
                BinOp::Mul => x.checked_mul(y),
                BinOp::Div if y != 0 => x.checked_div(y),
                BinOp::Rem if y != 0 => x.checked_rem(y),
                BinOp::Min => Some(x.min(y)),
                BinOp::Max => Some(x.max(y)),
                _ => None,
            };
            if let Some(v) = v {
                return Val::Data {
                    ops: vec![Operand::ImmI(v)],
                    ty: aty,
                };
            }
        }
        let owned;
        let dst = match dst {
            Some(d) => d,
            None => {
                owned = self.regs(aty);
                &owned
            }
        };
        self.math_into(MathK::Bin(op), aty, &[&aops, &bops], dst);
        Val::Data {
            ops: dst.iter().cloned().map(Operand::Reg).collect(),
            ty: aty,
        }
    }

    pub(super) fn fma(
        &mut self,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        dst: Option<&[String]>,
    ) -> Val {
        let a = self.eval(a);
        let b = self.eval(b);
        let c = self.eval(c);
        self.fma_evaluated(a, b, c, dst)
    }

    fn fma_evaluated(&mut self, a: Val, b: Val, c: Val, dst: Option<&[String]>) -> Val {
        let (aops, aty) = a.data();
        let (bops, bty) = b.data();
        let (cops, cty) = c.data();
        assert!(
            aty == bty && bty == cty,
            "fma type mismatch: {aty:?}, {bty:?}, {cty:?}"
        );
        assert!(aty.scalar.is_float(), "fma on {:?}", aty.scalar);
        let owned;
        let dst = match dst {
            Some(d) => d,
            None => {
                owned = self.regs(aty);
                &owned
            }
        };
        self.math_into(MathK::Fma, aty, &[&aops, &bops, &cops], dst);
        Val::Data {
            ops: dst.iter().cloned().map(Operand::Reg).collect(),
            ty: aty,
        }
    }

    pub(super) fn select(
        &mut self,
        pred: ValueId,
        a: ValueId,
        b: ValueId,
        dst: Option<&[String]>,
    ) -> Val {
        let pred = self.eval(pred);
        let a = self.eval(a);
        let b = self.eval(b);
        self.select_evaluated(pred, a, b, dst)
    }

    fn select_evaluated(&mut self, pred: Val, a: Val, b: Val, dst: Option<&[String]>) -> Val {
        let (pops, pty) = pred.data();
        assert_eq!(pty.scalar, Scalar::Pred, "select on {:?}", pty.scalar);
        let (aops, aty) = a.data();
        let (bops, bty) = b.data();
        assert_eq!(aty, bty, "select arm type mismatch: {aty:?} vs {bty:?}");
        assert!(
            pty.lanes == 1 || pty.lanes == aty.lanes,
            "select predicate lanes {} on {:?}",
            pty.lanes,
            aty
        );
        let owned;
        let dst = match dst {
            Some(d) => d,
            None => {
                owned = self.regs(aty);
                &owned
            }
        };
        for lane in 0..aty.lanes as usize {
            let p = self.materialize(&pops[lane.min(pops.len() - 1)], Scalar::Pred);
            let (sa, sb) = (
                self.operand(&aops[lane], aty.scalar),
                self.operand(&bops[lane], aty.scalar),
            );
            if aty.scalar == Scalar::Pred {
                let selected_a = self.reg(Scalar::Pred);
                let not_p = self.reg(Scalar::Pred);
                let selected_b = self.reg(Scalar::Pred);
                self.ins(format_args!("and.pred {selected_a}, {p}, {sa};"));
                self.ins(format_args!("not.pred {not_p}, {p};"));
                self.ins(format_args!("and.pred {selected_b}, {not_p}, {sb};"));
                self.ins(format_args!(
                    "or.pred {}, {selected_a}, {selected_b};",
                    dst[lane]
                ));
            } else {
                self.ins(format_args!(
                    "selp.{} {}, {sa}, {sb}, {p};",
                    mem(aty.scalar),
                    dst[lane]
                ));
            }
        }
        Val::Data {
            ops: dst.iter().cloned().map(Operand::Reg).collect(),
            ty: aty,
        }
    }

    /// A loop standing for one of the values it reduces to, evaluated where it
    /// is read.
    fn operand_loop(&mut self, op: OpId, value: ValueId) -> Val {
        let nothing = std::collections::BTreeSet::new();
        let regs = self.counted_loop(op, &[], &[], &nothing, None, None);
        let index = self
            .f
            .op(op)
            .results
            .iter()
            .position(|&result| result == value)
            .expect("the value is a result of this loop");
        Val::Data {
            ops: regs[index].iter().cloned().map(Operand::Reg).collect(),
            ty: self.f.data_ty(value),
        }
    }
}
