// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversions between scalars, warp shuffles, and the float operations whose
//! form depends on what the target packs.

use super::kernel::K;
use super::types::{FloatLower, MathK, Operand, arith, mem, reg_class, subword_wide};
use crate::{BinOp, Scalar, Ty};

impl<T: super::text::Text> K<'_, T> {
    /// One f32 approximation instruction per lane. The approximate instructions
    /// exist for f32 alone, so f16 and bf16 widen, compute, and narrow back.
    pub(super) fn approx_f32_lanes(
        &mut self,
        instruction: &str,
        ops: &[Operand],
        ty: Ty,
    ) -> Vec<String> {
        let regs = self.regs(ty);
        for (dst, src) in regs.iter().cloned().zip(ops) {
            if ty.scalar == Scalar::F32 {
                let s = self.operand(src, ty.scalar);
                self.ins(format_args!("{instruction}.f32 {dst}, {s};"));
            } else {
                let wide = self.up_f32(src, ty.scalar);
                let r = self.reg(Scalar::F32);
                self.ins(format_args!("{instruction}.f32 {r}, {wide};"));
                self.ins(format_args!("cvt.rn.{}.f32 {dst}, {r};", arith(ty.scalar)));
            }
        }
        regs
    }

    /// The hyperbolic tangent, per lane. `tanh.approx.f32` exists from sm_75;
    /// below it the same value is composed from `ex2.approx.f32` as
    /// `1 - 2 / (2^(2 x log2 e) + 1)`, which is exact at both infinities.
    pub(super) fn tanh_lanes(&mut self, ops: &[Operand], ty: Ty) -> Vec<String> {
        if self.sm >= 75 {
            return self.approx_f32_lanes("tanh.approx", ops, ty);
        }
        let regs = self.regs(ty);
        for (dst, src) in regs.iter().cloned().zip(ops) {
            let wide = if ty.scalar == Scalar::F32 {
                self.materialize(src, ty.scalar)
            } else {
                self.up_f32(src, ty.scalar)
            };
            let scaled = self.reg(Scalar::F32);
            // 2 log2 e, the argument scale that turns exp(2x) into 2^(2 x log2 e).
            self.ins(format_args!("mul.f32 {scaled}, {wide}, 0f4038AA3B;"));
            let raised = self.reg(Scalar::F32);
            self.ins(format_args!("ex2.approx.f32 {raised}, {scaled};"));
            let denominator = self.reg(Scalar::F32);
            self.ins(format_args!("add.f32 {denominator}, {raised}, 0f3F800000;"));
            let ratio = self.reg(Scalar::F32);
            self.ins(format_args!(
                "div.rn.f32 {ratio}, 0f40000000, {denominator};"
            ));
            let value = self.reg(Scalar::F32);
            self.ins(format_args!("sub.f32 {value}, 0f3F800000, {ratio};"));
            if ty.scalar == Scalar::F32 {
                self.ins(format_args!("mov.f32 {dst}, {value};"));
            } else {
                self.ins(format_args!(
                    "cvt.rn.{}.f32 {dst}, {value};",
                    arith(ty.scalar)
                ));
            }
        }
        regs
    }

    /// One float-to-float rounding per lane, in the operand's own type: PTX
    /// spells the mode in the conversion, so no widening is needed.
    pub(super) fn round_lanes(&mut self, mode: &str, ops: &[Operand], ty: Ty) -> Vec<String> {
        let regs = self.regs(ty);
        for (dst, src) in regs.iter().cloned().zip(ops) {
            let s = self.operand(src, ty.scalar);
            let t = arith(ty.scalar);
            self.ins(format_args!("cvt.{mode}.{t}.{t} {dst}, {s};"));
        }
        regs
    }

    /// One bit count per lane. PTX counts a whole register and answers in `u32`,
    /// so a type narrower than its register has the register's extra leading
    /// zeros subtracted, and a signed one is masked to its own width first so
    /// sign extension contributes no ones.
    pub(super) fn bit_count_lanes(
        &mut self,
        instruction: &str,
        ops: &[Operand],
        ty: Ty,
    ) -> Vec<String> {
        let regs = self.regs(ty);
        let bits = ty.scalar.size() * 8;
        let wide = subword_wide(ty.scalar).unwrap_or(ty.scalar);
        let width = if wide.size() == 8 { "b64" } else { "b32" };
        for (dst, src) in regs.iter().cloned().zip(ops) {
            let mut value = if wide == ty.scalar {
                self.materialize(src, ty.scalar)
            } else {
                let held = self.reg(wide);
                self.cvt_one(wide, ty.scalar, &held, src);
                held
            };
            if wide != ty.scalar {
                let masked = self.reg(wide);
                let mask = (1i64 << bits) - 1;
                self.ins(format_args!("and.{width} {masked}, {value}, {mask};"));
                value = masked;
            }
            let counted = self.reg(Scalar::U32);
            self.ins(format_args!("{instruction}.{width} {counted}, {value};"));
            let counted = if instruction == "clz" && wide != ty.scalar {
                let own = self.reg(Scalar::U32);
                self.ins(format_args!(
                    "sub.u32 {own}, {counted}, {};",
                    wide.size() * 8 - bits
                ));
                own
            } else {
                counted
            };
            self.cvt_one(ty.scalar, Scalar::U32, &dst, &Operand::Reg(counted));
        }
        regs
    }

    /// `src` widened to an f32 register.
    pub(super) fn up_f32(&mut self, src: &Operand, s: Scalar) -> String {
        let r = self.reg(Scalar::F32);
        let sv = self.materialize(src, s);
        self.ins(format_args!("cvt.f32.{} {r}, {sv};", arith(s)));
        r
    }

    pub(super) fn cvt_one(&mut self, dst_s: Scalar, src_s: Scalar, dst: &str, src: &Operand) {
        use Scalar::*;
        if dst_s == src_s || (reg_class(dst_s) == reg_class(src_s) && !dst_s.is_float()) {
            let s = self.operand(src, src_s);
            self.ins(format_args!("mov.{} {dst}, {s};", mem(dst_s)));
            return;
        }
        // f16 <-> bf16 have no direct cvt everywhere; hop through f32.
        if matches!((dst_s, src_s), (F16, BF16) | (BF16, F16)) {
            let mid = self.up_f32(src, src_s);
            self.ins(format_args!("cvt.rn.{}.f32 {dst}, {mid};", arith(dst_s)));
            return;
        }
        let op = match (dst_s, src_s) {
            (S64, S32) => "cvt.s64.s32".into(),
            (S64, U32) | (U64, U32) | (U64, S32) => "cvt.u64.u32".into(),
            (S32, S64) | (S32, U64) | (U32, S64) | (U32, U64) => "cvt.u32.u64".into(),
            (F32, F16) => "cvt.f32.f16".into(),
            (F32, BF16) => "cvt.f32.bf16".into(),
            (F16, F32) => "cvt.rn.f16.f32".into(),
            (BF16, F32) => "cvt.rn.bf16.f32".into(),
            (F64, F32) => "cvt.f64.f32".into(),
            (F32, F64) => "cvt.rn.f32.f64".into(),
            (F64, F16) => "cvt.f64.f16".into(),
            (F16, F64) => "cvt.rn.f16.f64".into(),
            (F64, BF16) => "cvt.f64.bf16".into(),
            (BF16, F64) => "cvt.rn.bf16.f64".into(),
            (d, s) if d.is_float() && !s.is_float() => {
                format!("cvt.rn.{}.{}", arith(d), arith(s))
            }
            (d, s) if !d.is_float() && s.is_float() => {
                format!("cvt.rzi.{}.{}", arith(d), arith(s))
            }
            (d, s) if !d.is_float() && !s.is_float() => {
                format!("cvt.{}.{}", arith(d), arith(s))
            }
            // Typing admits a conversion with a predicate on both sides or on
            // neither, and the arms above cover every other pair.
            (d, s) => unreachable!("a conversion of {s:?} to {d:?}"),
        };
        let s = self.materialize(src, src_s);
        self.ins(format_args!("{op} {dst}, {s};"));
    }

    fn shfl_one(
        &mut self,
        scalar: Scalar,
        destination: &str,
        source: &Operand,
        mut shuffle_word: impl FnMut(&mut Self, String) -> String,
    ) {
        let source = self.materialize(source, scalar);
        match scalar.size() {
            1 | 2 => {
                let word = self.reg(Scalar::U32);
                self.ins(format_args!("mov.b32 {word}, {{{source}, {source}}};"));
                let shuffled = shuffle_word(self, word);
                let unused = self.reg(scalar);
                self.ins(format_args!(
                    "mov.b32 {{{destination}, {unused}}}, {shuffled};"
                ));
            }
            4 => {
                let word = self.reg(Scalar::U32);
                self.ins(format_args!("mov.b32 {word}, {source};"));
                let shuffled = shuffle_word(self, word);
                self.ins(format_args!("mov.b32 {destination}, {shuffled};"));
            }
            8 => {
                let (low, high) = (self.reg(Scalar::U32), self.reg(Scalar::U32));
                self.ins(format_args!("mov.b64 {{{low}, {high}}}, {source};"));
                let low = shuffle_word(self, low);
                let high = shuffle_word(self, high);
                self.ins(format_args!("mov.b64 {destination}, {{{low}, {high}}};"));
            }
            _ => unreachable!("a shuffle of {scalar:?}"),
        }
    }

    pub(super) fn shfl_down_one(&mut self, s: Scalar, dst: &str, src: &Operand, delta: &str) {
        self.shfl_one(s, dst, src, |me, word| {
            let out = me.reg(Scalar::U32);
            me.ins(format_args!(
                "shfl.sync.down.b32 {out}, {word}, {delta}, 31, 0xffffffff;"
            ));
            out
        });
    }

    /// One value read out of `lane`, over exactly `mask`'s lanes.
    ///
    /// A shuffle moves 32 bits, so a value is carried through as many `b32`
    /// words as it occupies: a sub-word scalar rides the low half of one, and a
    /// 64-bit one is split and put back together.
    pub(super) fn shfl_from_one(
        &mut self,
        s: Scalar,
        dst: &str,
        src: &Operand,
        lane: &str,
        mask: &str,
    ) {
        self.shfl_one(s, dst, src, |me, word| {
            let out = me.reg(Scalar::U32);
            me.ins(format_args!(
                "shfl.sync.idx.b32 {out}, {word}, {lane}, 31, {mask};"
            ));
            out
        });
    }

    // ---- math legalization -------------------------------------------------

    /// Packed pair mnemonic when the target has one for this op and scalar.
    pub(super) fn packed_pair(&self, k: MathK, s: Scalar) -> Option<&'static str> {
        match (k, s) {
            (MathK::Fma, Scalar::F32) if self.sm >= 100 => Some("fma.rn.f32x2"),
            (MathK::Bin(BinOp::Mul), Scalar::F32) if self.sm >= 100 => Some("mul.rn.f32x2"),
            (MathK::Bin(BinOp::Add), Scalar::F32) if self.sm >= 100 => Some("add.rn.f32x2"),
            (MathK::Fma, Scalar::F16) => Some("fma.rn.f16x2"),
            (MathK::Bin(BinOp::Mul), Scalar::F16) => Some("mul.rn.f16x2"),
            (MathK::Bin(BinOp::Add), Scalar::F16) => Some("add.rn.f16x2"),
            // bf16 fma is native on Ampere/Ada (HFMA2.BF16_V2); add/mul need sm_90.
            (MathK::Fma, Scalar::BF16) if self.sm >= 80 => Some("fma.rn.bf16x2"),
            (MathK::Bin(BinOp::Mul), Scalar::BF16) if self.sm >= 90 => Some("mul.rn.bf16x2"),
            (MathK::Bin(BinOp::Add), Scalar::BF16) if self.sm >= 90 => Some("add.rn.bf16x2"),
            _ => None,
        }
    }

    /// Scalar float lowering: native instruction or the f32 detour.
    pub(super) fn float_lower(&self, k: MathK, s: Scalar) -> FloatLower {
        let native = match (k, s) {
            (_, Scalar::F32 | Scalar::F64) => !matches!(k, MathK::Bin(BinOp::Rem)),
            (MathK::Fma, Scalar::F16) => true,
            (MathK::Bin(BinOp::Add | BinOp::Mul), Scalar::F16) => true,
            (MathK::Fma, Scalar::BF16) => self.sm >= 80,
            (MathK::Bin(BinOp::Add | BinOp::Mul), Scalar::BF16) => self.sm >= 90,
            _ => false,
        };
        if native {
            FloatLower::Native
        } else {
            FloatLower::ViaF32
        }
    }

    pub(super) fn scalar_float_instr(k: MathK, s: Scalar) -> String {
        match k {
            MathK::Fma => format!("fma.rn.{}", arith(s)),
            MathK::Bin(op) => {
                let name = match op {
                    BinOp::Add => "add.rn",
                    BinOp::Sub => "sub.rn",
                    BinOp::Mul => "mul.rn",
                    BinOp::Div => "div.rn",
                    BinOp::Min => "min",
                    BinOp::Max => "max",
                    // Typing admits only the float operations named above.
                    other => unreachable!("{other:?} on floats"),
                };
                format!("{name}.{}", arith(s))
            }
        }
    }

    /// Elementwise float/int math over lanes into `dst` registers: packed
    /// pair instructions where the target has them, scalar otherwise, the
    /// f32 detour where the scalar type has no native instruction.
    pub(super) fn math_into(&mut self, k: MathK, ty: Ty, srcs: &[&[Operand]], dst: &[String]) {
        let s = ty.scalar;
        if !s.is_float() {
            let MathK::Bin(op) = k else {
                unreachable!("a fused multiply-add on {s:?}")
            };
            if matches!(op, BinOp::Div | BinOp::Rem)
                && s != Scalar::Pred
                && srcs[1].iter().all(|d| matches!(d, Operand::ImmI(_)))
            {
                for lane in 0..ty.lanes as usize {
                    let n = srcs[0][lane.min(srcs[0].len() - 1)].clone();
                    let Operand::ImmI(d) = srcs[1][lane.min(srcs[1].len() - 1)] else {
                        unreachable!()
                    };
                    let q = self.int_div_rem_const(op, s, &n, d);
                    self.ins(format_args!("mov.{} {}, {q};", mem(s), dst[lane]));
                }
                return;
            }
            if let Some(wide) = subword_wide(s) {
                let name = match op {
                    BinOp::Add => format!("add.{}", arith(wide)),
                    BinOp::Sub => format!("sub.{}", arith(wide)),
                    BinOp::Mul => format!("mul.lo.{}", arith(wide)),
                    BinOp::Div => format!("div.{}", arith(wide)),
                    BinOp::Rem => format!("rem.{}", arith(wide)),
                    BinOp::Min => format!("min.{}", arith(wide)),
                    BinOp::Max => format!("max.{}", arith(wide)),
                    BinOp::And => "and.b32".into(),
                    BinOp::Or => "or.b32".into(),
                    BinOp::Xor => "xor.b32".into(),
                    BinOp::Shl => "shl.b32".into(),
                    BinOp::Shr => format!("shr.{}", arith(wide)),
                };
                for lane in 0..ty.lanes as usize {
                    let mut texts = Vec::with_capacity(srcs.len());
                    for (source, ops) in srcs.iter().enumerate() {
                        let operand = &ops[lane.min(ops.len() - 1)];
                        if source == 1 && matches!(op, BinOp::Shl | BinOp::Shr) {
                            texts.push(self.operand(operand, Scalar::U32));
                        } else {
                            let widened = self.reg(wide);
                            self.cvt_one(wide, s, &widened, operand);
                            texts.push(widened);
                        }
                    }
                    let result = self.reg(wide);
                    self.ins(format_args!("{name} {result}, {};", texts.join(", ")));
                    self.cvt_one(s, wide, &dst[lane], &Operand::Reg(result));
                }
                return;
            }
            let name = match op {
                BinOp::Add => format!("add.{}", arith(s)),
                BinOp::Sub => format!("sub.{}", arith(s)),
                BinOp::Mul => format!("mul.lo.{}", arith(s)),
                BinOp::Div => format!("div.{}", arith(s)),
                BinOp::Rem => format!("rem.{}", arith(s)),
                BinOp::Min => format!("min.{}", arith(s)),
                BinOp::Max => format!("max.{}", arith(s)),
                BinOp::And | BinOp::Or => {
                    let name = if op == BinOp::And { "and" } else { "or" };
                    let sfx = if s == Scalar::Pred { "pred" } else { mem(s) };
                    format!("{name}.{sfx}")
                }
                BinOp::Xor => format!("xor.{}", mem(s)),
                BinOp::Shl => format!("shl.{}", mem(s)),
                BinOp::Shr => format!("shr.{}", arith(s)),
            };
            for lane in 0..ty.lanes as usize {
                let texts: Vec<String> = srcs
                    .iter()
                    .enumerate()
                    .map(|(source, ops)| {
                        let scalar = if source == 1 && matches!(op, BinOp::Shl | BinOp::Shr) {
                            Scalar::U32
                        } else {
                            s
                        };
                        self.operand(&ops[lane.min(ops.len() - 1)], scalar)
                    })
                    .collect();
                self.ins(format_args!("{name} {}, {};", dst[lane], texts.join(", ")));
            }
            return;
        }

        let mut lane = 0usize;
        if ty.lanes >= 2
            && let Some(packed) = self.packed_pair(k, s)
        {
            let carrier = if s == Scalar::F32 {
                Scalar::U64
            } else {
                Scalar::U32
            };
            while lane + 1 < ty.lanes as usize {
                let mut caps = Vec::with_capacity(srcs.len());
                for ops in srcs {
                    let lo = self.materialize(&ops[lane], s);
                    let hi = self.materialize(&ops[lane + 1], s);
                    let c = self.reg(carrier);
                    self.ins(format_args!("mov.{} {c}, {{{lo}, {hi}}};", mem(carrier)));
                    caps.push(c);
                }
                let out = self.reg(carrier);
                self.ins(format_args!("{packed} {out}, {};", caps.join(", ")));
                self.ins(format_args!(
                    "mov.{} {{{}, {}}}, {out};",
                    mem(carrier),
                    dst[lane],
                    dst[lane + 1]
                ));
                lane += 2;
            }
        }
        for lane in lane..ty.lanes as usize {
            match self.float_lower(k, s) {
                FloatLower::Native => {
                    let instr = Self::scalar_float_instr(k, s);
                    let texts: Vec<String> =
                        srcs.iter().map(|ops| self.operand(&ops[lane], s)).collect();
                    self.ins(format_args!("{instr} {}, {};", dst[lane], texts.join(", ")));
                }
                FloatLower::ViaF32 => {
                    let ups: Vec<Operand> = srcs
                        .iter()
                        .map(|ops| Operand::Reg(self.up_f32(&ops[lane], s)))
                        .collect();
                    let instr = Self::scalar_float_instr(k, Scalar::F32);
                    let texts: Vec<String> =
                        ups.iter().map(|op| self.operand(op, Scalar::F32)).collect();
                    let mid = self.reg(Scalar::F32);
                    self.ins(format_args!("{instr} {mid}, {};", texts.join(", ")));
                    self.ins(format_args!(
                        "cvt.rn.{}.f32 {}, {mid};",
                        arith(s),
                        dst[lane]
                    ));
                }
            }
        }
    }
}
