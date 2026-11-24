// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer division and remainder by a constant divisor.
//!
//! ptxas keeps 64-bit `div`/`rem` as ABI subroutine calls even with an
//! immediate divisor, so the strength reduction happens here, where the
//! constant is still visible.

use super::kernel::K;
use super::types::{Operand, arith, mem, subword_wide};
use crate::{BinOp, Scalar};

/// sign-extended to i64.
pub(super) fn magic_signed(d: i64, bits: u32) -> (i64, u32) {
    let two_p = 1u128 << (bits - 1);
    let mask = (1u128 << bits) - 1;
    let ad = u128::from(d.unsigned_abs()) & mask;
    debug_assert!(2 <= ad && ad < two_p);
    let t = two_p + u128::from(d < 0);
    let anc = t - 1 - t % ad;
    let mut p = bits - 1;
    let (mut q1, mut r1) = (two_p / anc, two_p % anc);
    let (mut q2, mut r2) = (two_p / ad, two_p % ad);
    loop {
        p += 1;
        q1 *= 2;
        r1 *= 2;
        if r1 >= anc {
            q1 += 1;
            r1 -= anc;
        }
        q2 *= 2;
        r2 *= 2;
        if r2 >= ad {
            q2 += 1;
            r2 -= ad;
        }
        let delta = ad - r2;
        if !(q1 < delta || (q1 == delta && r1 == 0)) {
            break;
        }
    }
    let m = (((q2 + 1) & mask) as i64) << (64 - bits) >> (64 - bits);
    (if d < 0 { m.wrapping_neg() } else { m }, p - bits)
}

/// Multiplier, add-indicator, and shift for unsigned `bits`-wide division by
/// `d` (Hacker's Delight 10-8): `n / d = shr(mulhi(M, n), s)`, or with the
/// add-indicator `shr(shr(n - q0, 1) + q0, s - 1)`. Requires non-power-of-two
/// `d >= 3`.
pub(super) fn magic_unsigned(d: u64, bits: u32) -> (u64, bool, u32) {
    let mask = (1u128 << bits) - 1;
    let d = u128::from(d) & mask;
    debug_assert!(d >= 3 && !d.is_power_of_two());
    let two_p = 1u128 << (bits - 1);
    let nc = mask - (mask + 1 - d) % d;
    let mut p = bits - 1;
    let (mut q1, mut r1) = (two_p / nc, two_p % nc);
    let (mut q2, mut r2) = ((two_p - 1) / d, (two_p - 1) % d);
    let mut add = false;
    loop {
        p += 1;
        if r1 >= nc - r1 {
            q1 = 2 * q1 + 1;
            r1 = 2 * r1 - nc;
        } else {
            q1 *= 2;
            r1 *= 2;
        }
        if r2 + 1 >= d - r2 {
            if q2 >= two_p - 1 {
                add = true;
            }
            q2 = 2 * q2 + 1;
            r2 = 2 * r2 + 1 - d;
        } else {
            if q2 >= two_p {
                add = true;
            }
            q2 *= 2;
            r2 = 2 * r2 + 1;
        }
        let delta = d - 1 - r2;
        if !(p < 2 * bits && (q1 < delta || (q1 == delta && r1 == 0))) {
            break;
        }
    }
    (((q2 + 1) & mask) as u64, add, p - bits)
}

impl<T: super::text::Text> K<'_, T> {
    /// Integer `/` or `%` by a compile-time constant, as multiply-shift
    /// (Hacker's Delight 10-4 / 10-8). ptxas keeps 64-bit `div`/`rem` as ABI
    /// subroutine calls even with an immediate divisor, so the strength
    /// reduction happens here, where the constant is still visible. Returns
    /// the result register.
    pub(super) fn int_div_rem_const(
        &mut self,
        op: BinOp,
        s: Scalar,
        n: &Operand,
        d: i64,
    ) -> String {
        if let Some(wide) = subword_wide(s) {
            let widened = self.reg(wide);
            self.cvt_one(wide, s, &widened, n);
            let q = self.int_div_rem_const(op, wide, &Operand::Reg(widened), d);
            let out = self.reg(s);
            self.cvt_one(s, wide, &out, &Operand::Reg(q));
            return out;
        }
        let bits = s.size() as u32 * 8;
        let signed = matches!(s, Scalar::S32 | Scalar::S64);
        // Normalize the immediate to the scalar's width and signedness.
        let d = if signed {
            (d << (64 - bits)) >> (64 - bits)
        } else if bits < 64 {
            d & ((1i64 << bits) - 1)
        } else {
            d
        };
        if d == 0 {
            // Division by zero keeps the hardware instruction's behavior.
            let n = self.operand(n, s);
            let out = self.reg(s);
            let name = if op == BinOp::Div { "div" } else { "rem" };
            self.ins(format_args!("{name}.{} {out}, {n}, 0;", arith(s)));
            return out;
        }
        if op == BinOp::Rem && signed && (d == 1 || d == -1) {
            return self.materialize(&Operand::ImmI(0), s);
        }
        if op == BinOp::Rem && !signed && (d as u64).is_power_of_two() {
            let n = self.operand(n, s);
            let out = self.reg(s);
            self.ins(format_args!(
                "and.{} {out}, {n}, 0x{:x};",
                mem(s),
                (d as u64) - 1
            ));
            return out;
        }
        let q = self.int_quotient_const(s, bits, signed, n, d);
        if op == BinOp::Div {
            return q;
        }
        // n % d = n - (n / d) * d.
        let n = self.operand(n, s);
        let prod = self.reg(s);
        let mask = (u64::MAX) >> (64 - bits);
        self.ins(format_args!(
            "mul.lo.{} {prod}, {q}, 0x{:x};",
            arith(s),
            (d as u64) & mask
        ));
        let out = self.reg(s);
        self.ins(format_args!("sub.{} {out}, {n}, {prod};", arith(s)));
        out
    }

    pub(super) fn int_quotient_const(
        &mut self,
        s: Scalar,
        bits: u32,
        signed: bool,
        n: &Operand,
        d: i64,
    ) -> String {
        let ua = arith(if s == Scalar::S64 {
            Scalar::U64
        } else if s == Scalar::S32 {
            Scalar::U32
        } else {
            s
        });
        if !signed {
            let n = self.operand(n, s);
            let out = self.reg(s);
            let d = d as u64;
            if d == 1 {
                self.ins(format_args!("mov.{} {out}, {n};", mem(s)));
            } else if d.is_power_of_two() {
                self.ins(format_args!("shr.{ua} {out}, {n}, {};", d.trailing_zeros()));
            } else {
                let (m, add, sh) = magic_unsigned(d, bits);
                self.ins(format_args!("mul.hi.{ua} {out}, {n}, 0x{m:x};"));
                if add {
                    debug_assert!(sh >= 1);
                    let t = self.reg(s);
                    self.ins(format_args!("sub.{ua} {t}, {n}, {out};"));
                    self.ins(format_args!("shr.{ua} {t}, {t}, 1;"));
                    self.ins(format_args!("add.{ua} {t}, {t}, {out};"));
                    self.ins(format_args!("shr.{ua} {out}, {t}, {};", sh - 1));
                } else if sh > 0 {
                    self.ins(format_args!("shr.{ua} {out}, {out}, {sh};"));
                }
            }
            return out;
        }
        let sa = arith(s);
        let n = self.operand(n, s);
        let out = self.reg(s);
        if d == 1 {
            self.ins(format_args!("mov.{} {out}, {n};", mem(s)));
        } else if d == -1 {
            self.ins(format_args!("sub.{sa} {out}, 0, {n};"));
        } else if d == (-1i64) << (bits - 1) {
            // Only MIN / MIN is 1; every other |n| < |MIN|.
            let p = self.reg(Scalar::Pred);
            self.ins(format_args!("setp.eq.{sa} {p}, {n}, {d};"));
            self.ins(format_args!("selp.{} {out}, 1, 0, {p};", mem(s)));
        } else if d.unsigned_abs().is_power_of_two() {
            // Round toward zero: bias negative dividends by |d| - 1.
            let k = d.unsigned_abs().trailing_zeros();
            let sign = self.reg(s);
            self.ins(format_args!("shr.{sa} {sign}, {n}, {};", bits - 1));
            self.ins(format_args!("shr.{ua} {sign}, {sign}, {};", bits - k));
            self.ins(format_args!("add.{sa} {sign}, {n}, {sign};"));
            self.ins(format_args!("shr.{sa} {out}, {sign}, {k};"));
            if d < 0 {
                self.ins(format_args!("sub.{sa} {out}, 0, {out};"));
            }
        } else {
            let mask = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let (m, sh) = magic_signed(d, bits);
            self.ins(format_args!(
                "mul.hi.{sa} {out}, {n}, 0x{:x};",
                (m as u64) & mask
            ));
            if d > 0 && m < 0 {
                self.ins(format_args!("add.{sa} {out}, {out}, {n};"));
            } else if d < 0 && m > 0 {
                self.ins(format_args!("sub.{sa} {out}, {out}, {n};"));
            }
            if sh > 0 {
                self.ins(format_args!("shr.{sa} {out}, {out}, {sh};"));
            }
            // A negative quotient was rounded toward -inf; its sign bit is
            // the +1 that rounds it back toward zero.
            let sign = self.reg(s);
            self.ins(format_args!("shr.{ua} {sign}, {out}, {};", bits - 1));
            self.ins(format_args!("add.{sa} {out}, {out}, {sign};"));
        }
        out
    }
}
