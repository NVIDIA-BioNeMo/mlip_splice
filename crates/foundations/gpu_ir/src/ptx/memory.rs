// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Addressing, and the load and store forms a width and a space admit.

use std::collections::BTreeSet;

use super::kernel::K;
use super::types::evict_qualifier;
use super::types::{Binding, Operand, Val, load_type, mem, store_type};
use crate::arena::{OpId, ValueId};
use crate::op::OpKind;
use crate::{Evict, PtrTy, Scalar, Space, Ty};

/// The type an indivisible add is spelled at. Signed and unsigned integers
/// share one form: two's-complement addition is the same bits either way.
fn atomic_add_suffix(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "f32",
        Scalar::F64 => "f64",
        Scalar::F16 => "noftz.f16",
        Scalar::BF16 => "noftz.bf16",
        Scalar::S32 | Scalar::U32 => "u32",
        Scalar::S64 | Scalar::U64 => "u64",
        // The verifier admits only the scalars above.
        other => unreachable!("an atomic add of {other:?}"),
    }
}

impl<T: super::text::Text> K<'_, T> {
    /// `[reg]` or `[reg+off]` addressing text, `bump` bytes past the value.
    pub(super) fn addr(&self, p: &Val, bump: i64) -> String {
        let Val::Ptr { reg, byte_off, .. } = p else {
            unreachable!("addressing a data value");
        };
        let off = byte_off.wrapping_add(bump);
        if off == 0 {
            format!("[{reg}]")
        } else if off > 0 {
            format!("[{reg}+{off}]")
        } else {
            format!("[{reg}{off}]")
        }
    }

    pub(super) fn ld_op(&self, space: Space, nc: bool) -> &'static str {
        match (space, nc) {
            (Space::Global, false) => "ld.global",
            (Space::Global, true) => "ld.global.nc",
            (Space::Shared, false) => "ld.shared",
            (Space::Shared, true) => unreachable!("a non-coherent read of shared memory"),
        }
    }

    pub(super) fn load_into(&mut self, p: &Val, nc: bool, evict: Evict, regs: &[String], ty: Ty) {
        let Val::Ptr { ty: pty, .. } = p else {
            unreachable!("a load through a data value");
        };
        assert_eq!(pty.elem, ty.scalar, "load element mismatch");
        assert!(
            evict == Evict::Normal || pty.space == Space::Global,
            "eviction hint on a shared-memory load"
        );
        let ld = format!("{}{}", self.ld_op(pty.space, nc), evict_qualifier(evict));
        let size = ty.scalar.size();
        let bytes = size * ty.lanes as i64;
        match (ty.lanes, size, bytes) {
            (1, _, _) => {
                let a = self.addr(p, 0);
                self.ins(format_args!(
                    "{ld}.{} {}, {a};",
                    load_type(ty.scalar),
                    regs[0]
                ));
            }
            // 2-byte lanes ride b32 words: one word, or a v2/v4 word vector.
            (_, 2, 4 | 8 | 16) => {
                let words: Vec<String> = (0..bytes / 4).map(|_| self.reg(Scalar::U32)).collect();
                let a = self.addr(p, 0);
                match words.len() {
                    1 => self.ins(format_args!("{ld}.b32 {}, {a};", words[0])),
                    n => self.ins(format_args!("{ld}.v{n}.b32 {{{}}}, {a};", words.join(", "))),
                }
                for (i, w) in words.iter().enumerate() {
                    self.ins(format_args!(
                        "mov.b32 {{{}, {}}}, {w};",
                        regs[2 * i],
                        regs[2 * i + 1]
                    ));
                }
            }
            (lanes, 4, 8 | 16) => {
                let a = self.addr(p, 0);
                self.ins(format_args!(
                    "{ld}.v{lanes}.{} {{{}}}, {a};",
                    mem(ty.scalar),
                    regs.join(", ")
                ));
            }
            (2, 8, 16) => {
                let a = self.addr(p, 0);
                self.ins(format_args!(
                    "{ld}.v2.{} {{{}}}, {a};",
                    mem(ty.scalar),
                    regs.join(", ")
                ));
            }
            // No vector form: split into scalar loads.
            _ => {
                for (i, r) in regs.iter().enumerate() {
                    let a = self.addr(p, i as i64 * size);
                    self.ins(format_args!("{ld}.{} {r}, {a};", load_type(ty.scalar)));
                }
            }
        }
    }

    pub(super) fn store(&mut self, p: &Val, ops: &[Operand], ty: Ty, evict: Evict) {
        let Val::Ptr { ty: pty, .. } = p else {
            unreachable!("a store through a data value");
        };
        assert_eq!(pty.elem, ty.scalar, "store element mismatch");
        assert!(
            evict == Evict::Normal || pty.space == Space::Global,
            "eviction hint on a shared-memory store"
        );
        let st = format!(
            "{}{}",
            match pty.space {
                Space::Global => "st.global",
                Space::Shared => "st.shared",
            },
            evict_qualifier(evict)
        );
        let size = ty.scalar.size();
        let bytes = size * ty.lanes as i64;
        match (ty.lanes, size, bytes) {
            (1, _, _) => {
                let s = self.operand(&ops[0], ty.scalar);
                let a = self.addr(p, 0);
                self.ins(format_args!("{st}.{} {a}, {s};", store_type(ty.scalar)));
            }
            (_, 2, 4 | 8 | 16) => {
                let words: Vec<String> = (0..bytes / 4)
                    .map(|i| {
                        let lo = self.materialize(&ops[2 * i as usize], ty.scalar);
                        let hi = self.materialize(&ops[2 * i as usize + 1], ty.scalar);
                        let w = self.reg(Scalar::U32);
                        self.ins(format_args!("mov.b32 {w}, {{{lo}, {hi}}};"));
                        w
                    })
                    .collect();
                let a = self.addr(p, 0);
                match words.len() {
                    1 => self.ins(format_args!("{st}.b32 {a}, {};", words[0])),
                    n => self.ins(format_args!("{st}.v{n}.b32 {a}, {{{}}};", words.join(", "))),
                }
            }
            (lanes, 4, 8 | 16) | (lanes @ 2, 8, 16) => {
                let regs: Vec<String> = ops
                    .iter()
                    .map(|op| self.materialize(op, ty.scalar))
                    .collect();
                let a = self.addr(p, 0);
                self.ins(format_args!(
                    "{st}.v{lanes}.{} {a}, {{{}}};",
                    mem(ty.scalar),
                    regs.join(", ")
                ));
            }
            _ => {
                for (i, op) in ops.iter().enumerate() {
                    let s = self.operand(op, ty.scalar);
                    let a = self.addr(p, i as i64 * size);
                    self.ins(format_args!("{st}.{} {a}, {s};", store_type(ty.scalar)));
                }
            }
        }
    }

    /// One `red.global.add` per lane at consecutive addresses. The reduction
    /// takes no eviction hint and no vector form, so a wide value is a run of
    /// element-wide reductions and asks nothing wider of the base.
    pub(super) fn reduce_add(&mut self, p: &Val, ops: &[Operand], ty: Ty) {
        let sfx = atomic_add_suffix(ty.scalar);
        for (lane, held) in ops.iter().enumerate() {
            let value = self.materialize(held, ty.scalar);
            let at = self.addr(p, lane as i64 * ty.scalar.size());
            self.ins(format_args!("red.global.add.{sfx} {at}, {value};"));
        }
    }

    /// One indivisible add at one address, in the form the result asks for.
    ///
    /// `atom` and `red` are the same addition; they differ in whether the
    /// previous value comes back. So `into` decides which is emitted, and a
    /// caller with nothing to read the result into spends no register on one.
    pub(super) fn fetch_add(&mut self, p: &Val, value: &Operand, ty: Ty, into: Option<&str>) {
        let sfx = atomic_add_suffix(ty.scalar);
        let value = self.materialize(value, ty.scalar);
        let at = self.addr(p, 0);
        match into {
            Some(dst) => self.ins(format_args!("atom.global.add.{sfx} {dst}, {at}, {value};")),
            None => self.ins(format_args!("red.global.add.{sfx} {at}, {value};")),
        }
    }

    // ---- statements -------------------------------------------------------

    /// Evaluates `value` writing the result into `dst` registers. Same-lane
    /// operations write their destination directly; everything else goes
    /// through temporaries and lane movs (safe under dst/src aliasing).
    pub(super) fn eval_into(&mut self, value: ValueId, dst: &[String], ty: Ty) {
        let f = self.f;
        let operand = f.operand_op(value);
        let kind = operand.map(|op| f.op(op).kind.clone());
        match (operand, kind) {
            (Some(op), Some(OpKind::Bin(binop))) => {
                let v = self.bin(binop, f.op(op).operands[0], f.op(op).operands[1], Some(dst));
                let (_, vty) = v.data();
                assert_eq!(vty, ty, "binding type mismatch");
            }
            (Some(op), Some(OpKind::Fma)) => {
                let operands = &f.op(op).operands;
                let v = self.fma(operands[0], operands[1], operands[2], Some(dst));
                let (_, vty) = v.data();
                assert_eq!(vty, ty, "binding type mismatch");
            }
            (Some(op), Some(OpKind::Select)) => {
                let operands = &f.op(op).operands;
                let v = self.select(operands[0], operands[1], operands[2], Some(dst));
                let (_, vty) = v.data();
                assert_eq!(vty, ty, "binding type mismatch");
            }
            (Some(op), Some(OpKind::Load { lanes, nc, evict })) => {
                let p = self.eval(f.op(op).operands[0]);
                let lty = Ty::vec(p.ty().ptr().elem, lanes);
                assert_eq!(lty, ty, "binding type mismatch");
                self.load_into(&p, nc, evict, dst, ty);
            }
            (Some(op), Some(OpKind::Pack)) => self.pack_into(op, dst, ty),
            _ => {
                let (ops, vty) = self.eval(value).data();
                assert_eq!(vty, ty, "binding type mismatch: {vty:?} vs {ty:?}");
                for (d, op) in dst.iter().zip(&ops) {
                    let s = self.operand(op, ty.scalar);
                    if s != *d {
                        self.ins(format_args!("mov.{} {d}, {s};", mem(ty.scalar)));
                    }
                }
            }
        }
    }

    /// Writes a pack: its parts end to end, into the registers they occupy.
    ///
    /// The parts land in order, so a part reading a register an earlier one
    /// writes would read what that part put there — a permutation would
    /// collapse instead of permuting. Landing in place is therefore allowed
    /// only where the whole pack is proven lane-local, and otherwise every part
    /// is evaluated into registers of its own first, in the order the source
    /// holds them, before any destination moves.
    fn pack_into(&mut self, op: OpId, dst: &[String], ty: Ty) {
        let f = self.f;
        let parts = f.op(op).operands.clone();
        let mut slices = Vec::with_capacity(parts.len());
        let mut at = 0;
        for &part in &parts {
            let lanes = f.data_ty(part).lanes;
            slices.push((at, lanes));
            at += lanes as usize;
        }
        assert_eq!(at, dst.len(), "a pack of {at} lanes wrote {ty:?}");

        if self.lands_lane_by_lane(&parts, &slices, dst) {
            for (&part, &(at, lanes)) in parts.iter().zip(&slices) {
                let held = Ty::vec(ty.scalar, lanes);
                self.eval_into(part, &dst[at..at + lanes as usize], held);
            }
            return;
        }

        // Every part first, in source order — a load or a walk inside one runs
        // where the program says it does — and nothing lands until all of them
        // have. The destinations are reserved meanwhile, so nothing evaluated
        // on the way writes one of them.
        let mark = self.reserve(dst);
        let mut staged = Vec::with_capacity(parts.len());
        for (&part, &(_, lanes)) in parts.iter().zip(&slices) {
            let held = Ty::vec(ty.scalar, lanes);
            let regs = self.regs(held);
            self.eval_into(part, &regs, held);
            staged.push(regs);
        }
        self.release(mark);
        for (&(at, lanes), regs) in slices.iter().zip(&staged) {
            for (d, s) in dst[at..at + lanes as usize].iter().zip(regs) {
                if s != d {
                    self.ins(format_args!("mov.{} {d}, {s};", mem(ty.scalar)));
                }
            }
        }
    }

    /// Whether every part of a pack may be written straight into the registers
    /// it lands in.
    ///
    /// A part qualifies one of two ways. It reaches no destination register at
    /// all, so nothing it reads can be clobbered and it writes nothing but its
    /// own landing; or it is lane-structured, and every one of its lanes reads,
    /// among the destinations, at most the very register that lane lands in.
    /// One part failing decides the whole pack, because the parts share the
    /// destination they are proving safety against.
    fn lands_lane_by_lane(
        &self,
        parts: &[ValueId],
        slices: &[(usize, u8)],
        dst: &[String],
    ) -> bool {
        let landing = dst.iter().cloned().collect::<BTreeSet<_>>();
        parts.iter().zip(slices).all(|(&part, &(at, lanes))| {
            // Nothing may write a destination on the way to landing in one.
            if self.writes_any(part, &landing) {
                return false;
            }
            let mut reached = BTreeSet::new();
            self.reads_registers(part, &mut reached);
            if reached.is_disjoint(&landing) {
                return true;
            }
            (0..lanes).all(|lane| {
                let read = self.lane_registers(part, lane);
                let mine = &dst[at + lane as usize];
                read.intersection(&landing).all(|reg| reg == mine)
            })
        })
    }

    /// Whether evaluating a value writes any of `landing`.
    ///
    /// Only a result produced into a place writes a register the expression did
    /// not allocate, and only a region holds one, so this is a walk for the
    /// destinations named inside.
    fn writes_any(&self, value: ValueId, landing: &BTreeSet<String>) -> bool {
        let Some(op) = self.f.operand_op(value) else {
            return false;
        };
        self.op_writes_any(op, landing)
    }

    fn op_writes_any(&self, op: OpId, landing: &BTreeSet<String>) -> bool {
        let f = self.f;
        for index in 0..f.op(op).results.len() {
            let Some(place) = f.op(op).destination(index) else {
                continue;
            };
            if self
                .held(place)
                .is_some_and(|binding| binding.registers().iter().any(|reg| landing.contains(reg)))
            {
                return true;
            }
        }
        if f.op(op)
            .operands
            .iter()
            .any(|&operand| self.writes_any(operand, landing))
        {
            return true;
        }
        for index in 0..f.op(op).regions.len() {
            let region = f.op(op).regions[index];
            for slot in 0..f.region(region).blocks.len() {
                let block = f.region(region).blocks[slot];
                for step in f.block_steps(block) {
                    if self.op_writes_any(step, landing) {
                        return true;
                    }
                }
                for yielded in 0..f.block(block).terminator.yields.len() {
                    if self.writes_any(f.block(block).terminator.yields[yielded], landing) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The registers one lane of a value reads.
    ///
    /// Exact where the value is a lane of lanes — a value in registers, and the
    /// operations that carry lanes through one for one — and the whole
    /// footprint of everything else, because a load, a shuffle, a broadcast or
    /// a region reads what it reads for every lane alike. Over-stating costs a
    /// staging copy; under-stating would let a write land on a register still
    /// to be read.
    fn lane_registers(&self, value: ValueId, lane: u8) -> BTreeSet<String> {
        if let Some(binding) = self.held(value) {
            return match binding {
                Binding::Data { regs, .. } | Binding::LoopState { regs, .. } => regs
                    .get(lane as usize)
                    .map(|reg| BTreeSet::from([reg.clone()]))
                    .unwrap_or_default(),
                Binding::Ptr { reg, .. } => BTreeSet::from([reg.clone()]),
            };
        }
        let whole = |held: &Self| {
            let mut read = BTreeSet::new();
            held.reads_registers(value, &mut read);
            read
        };
        let Some(op) = self.f.operand_op(value) else {
            return whole(self);
        };
        if !self.f.op(op).regions.is_empty() {
            return whole(self);
        }
        let operands = self.f.op(op).operands.clone();
        match self.f.op(op).kind {
            OpKind::IntConst { .. } | OpKind::FloatConst { .. } => BTreeSet::new(),
            // Lane by lane: lane `l` of the result is lane `l` of each operand,
            // and a one-lane operand is the same for every lane.
            OpKind::Un(_)
            | OpKind::Bin(_)
            | OpKind::Fma
            | OpKind::Cmp(_)
            | OpKind::Select
            | OpKind::Cast(_) => {
                let mut read = BTreeSet::new();
                for operand in operands {
                    let held = self.f.data_ty(operand).lanes;
                    let at = if held == 1 { 0 } else { lane };
                    read.extend(self.lane_registers(operand, at));
                }
                read
            }
            OpKind::Pack => {
                let mut at = 0;
                for part in operands {
                    let held = self.f.data_ty(part).lanes;
                    if lane < at + held {
                        return self.lane_registers(part, lane - at);
                    }
                    at += held;
                }
                whole(self)
            }
            OpKind::Extract(from) if lane == 0 => self.lane_registers(operands[0], from),
            _ => whole(self),
        }
    }

    pub(super) fn assign_ptr(&mut self, reg: &str, ty: PtrTy, value: ValueId) {
        let v = self.eval(value);
        let Val::Ptr {
            reg: src,
            byte_off,
            ty: vty,
        } = v
        else {
            unreachable!("a data value bound to pointer {reg}");
        };
        assert_eq!(vty, ty, "pointer binding type mismatch");
        let (word, add) = match ty.space {
            Space::Global => ("b64", "add.s64"),
            Space::Shared => ("b32", "add.s32"),
        };
        if byte_off != 0 {
            self.ins(format_args!("{add} {reg}, {src}, {byte_off};"));
        } else if src != reg {
            self.ins(format_args!("mov.{word} {reg}, {src};"));
        }
    }
}
