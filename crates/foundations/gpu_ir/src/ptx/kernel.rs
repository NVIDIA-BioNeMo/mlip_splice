// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `.visible .entry` frame around a kernel body, and the register, label
//! and storage bookkeeping every lowering step draws on.

use std::collections::{BTreeSet, HashMap};

use super::text::Text;
use super::types::{Binding, Operand, mem, reg_class};
use crate::arena::{Declaration, Function, OpId, ValueId};
use crate::bits::{f32_bits, f64_bits};
use crate::func::Launch;
use crate::{ParamTy, PtrTy, Scalar, Space, Ty};

pub(super) struct K<'a, T: Text> {
    pub(super) f: &'a Function,
    pub(super) sm: i32,
    body: T,
    pub(super) shared: Vec<String>,
    pub(super) counts: HashMap<&'static str, usize>,
    /// Where each value lives, indexed by value. A value with no entry is an
    /// operand: it has no storage of its own and is evaluated wherever it is
    /// read, which is what makes an operand read twice cost twice.
    pub(super) regs: Vec<Option<Binding>>,
    /// How deep the region was that gave a value its storage, so a result can
    /// tell a place of the region it stands in from one that outlives it.
    pub(super) depth: Vec<u32>,
    pub(super) open: u32,
    pub(super) labels: usize,
    /// How many calls this body has emitted, so each one's `.param` variables
    /// are named apart from every other's.
    pub(super) calls: usize,
    pub(super) loop_ends: Vec<String>,
    /// Per running loop, the registers it carries. A loop writes them at the
    /// backedge, where no operation shows, so what stands in them is the loop's
    /// own whatever name a chain of updates has left it under.
    carried: Vec<Vec<String>>,
    /// Registers a write in progress is about to land in. Nothing evaluated on
    /// the way there may write one of them, so a place holding one is not a
    /// place a loop or a branch may produce into while it is reserved.
    reserved: Vec<String>,
}

impl<'a, T: Text> K<'a, T> {
    pub(super) fn new(f: &'a Function, sm: i32, body: T) -> Self {
        K {
            f,
            sm,
            body,
            shared: Vec::new(),
            counts: HashMap::new(),
            regs: vec![None; f.num_values()],
            depth: vec![0; f.num_values()],
            open: 0,
            labels: 0,
            calls: 0,
            loop_ends: Vec::new(),
            carried: Vec::new(),
            reserved: Vec::new(),
        }
    }

    pub(super) fn kernel(&mut self, kernel: &Launch, out: &mut T) -> usize {
        let declarations = self.f.declarations().to_vec();
        for (index, declared) in declarations.iter().enumerate() {
            let Declaration::Shared(arr) = declared else {
                continue;
            };
            let sym = format!("shm_{}", arr.name);
            let bytes = arr.elem.size() * arr.len;
            // 16 covers the widest vector access a shared array can host.
            self.shared
                .push(format!(".shared .align 16 .b8 {sym}[{bytes}];"));
            let reg = self.reg(Scalar::U32);
            self.ins(format_args!("mov.u32 {reg}, {sym};"));
            self.bind(
                self.f.declared(index),
                Binding::Ptr {
                    reg,
                    ty: PtrTy {
                        elem: arr.elem,
                        space: Space::Shared,
                    },
                },
            );
        }
        let mut position = 0;
        for (index, declared) in declarations.iter().enumerate() {
            let Declaration::Param(p) = declared else {
                continue;
            };
            let i = position;
            position += 1;
            match p.ty {
                ParamTy::Ptr(elem) => {
                    let raw = self.reg(Scalar::U64);
                    self.ins(format_args!("ld.param.b64 {raw}, [{}_param_{i}];", p.name));
                    let global = self.reg(Scalar::U64);
                    self.ins(format_args!("cvta.to.global.u64 {global}, {raw};"));
                    self.bind(
                        self.f.declared(index),
                        Binding::Ptr {
                            reg: global,
                            ty: PtrTy {
                                elem,
                                space: Space::Global,
                            },
                        },
                    );
                }
                ParamTy::Scalar(s) => {
                    let r = self.reg(s);
                    self.ins(format_args!(
                        "ld.param.{} {r}, [{}_param_{i}];",
                        super::types::load_type(s),
                        p.name
                    ));
                    self.bind(
                        self.f.declared(index),
                        Binding::Data {
                            regs: vec![r],
                            ty: Ty::scalar(s),
                        },
                    );
                }
            }
        }
        let entry = self.f.entry();
        let ops = self.f.block(entry).ops.clone();
        let nothing = std::collections::BTreeSet::new();
        self.statements(&ops, &[], &nothing, &nothing);

        let params = kernel.function.parameters();
        let _ = writeln!(out, ".visible .entry {}(", kernel.function.name);
        let plist: Vec<String> = params
            .iter()
            .enumerate()
            .map(|(i, p)| match p.ty {
                ParamTy::Ptr(_) => format!("    .param .u64 {}_param_{i}", p.name),
                ParamTy::Scalar(s) => format!(
                    "    .param .{} {}_param_{i}",
                    super::param_space(crate::ValTy::Data(Ty::scalar(s))),
                    p.name
                ),
            })
            .collect();
        let _ = writeln!(out, "{}", plist.join(",\n"));
        let _ = writeln!(out, ")");
        let _ = writeln!(out, ".maxntid {}, 1, 1", kernel.max_threads);
        if let Some(max_registers) = kernel.max_registers {
            let _ = writeln!(out, ".maxnreg {max_registers}");
        }
        let _ = writeln!(out, "{{");
        for class in ["%p", "%rs", "%r", "%rd", "%h", "%f", "%fd"] {
            let n = *self.counts.get(class).unwrap_or(&0);
            if n > 0 {
                let kind = match class {
                    "%p" => ".pred",
                    "%rs" => ".b16",
                    "%r" => ".b32",
                    "%rd" => ".b64",
                    "%h" => ".b16",
                    "%f" => ".f32",
                    "%fd" => ".f64",
                    _ => unreachable!(),
                };
                let _ = writeln!(out, "    .reg {kind} {class}<{}>;", n + 1);
            }
        }
        for s in &self.shared {
            let _ = writeln!(out, "    {s}");
        }
        out.append(&self.body);
        let _ = writeln!(out, "$EXIT:");
        let _ = writeln!(out, "    ret;");
        let _ = writeln!(out, "}}");
        self.body.len()
    }

    pub(super) fn ins(&mut self, instruction: std::fmt::Arguments<'_>) {
        let _ = writeln!(self.body, "    {instruction}");
    }

    pub(super) fn write_label(&mut self, label: &str) {
        let _ = writeln!(self.body, "{label}:");
    }

    pub(super) fn reg(&mut self, s: Scalar) -> String {
        let class = reg_class(s);
        let c = self.counts.entry(class).or_insert(0);
        *c += 1;
        format!("{class}{c}")
    }

    pub(super) fn regs(&mut self, ty: Ty) -> Vec<String> {
        (0..ty.lanes).map(|_| self.reg(ty.scalar)).collect()
    }

    pub(super) fn label(&mut self, tag: &str) -> String {
        self.labels += 1;
        format!("$L{}_{}", tag, self.labels)
    }

    pub(super) fn bind(&mut self, value: ValueId, b: Binding) {
        let index = value.index();
        self.regs[index] = Some(b);
        self.depth[index] = self.open;
    }

    /// Where a value lives, for a value that has storage.
    pub(super) fn lookup(&self, value: ValueId) -> &Binding {
        self.regs[value.index()]
            .as_ref()
            .unwrap_or_else(|| unreachable!("%{} has no storage", value.index()))
    }

    /// Where a value lives, if it has storage at all.
    pub(super) fn held(&self, value: ValueId) -> Option<&Binding> {
        self.regs[value.index()].as_ref()
    }

    /// Reserves `regs` for the duration of a write, giving back the mark that
    /// releases them.
    pub(super) fn reserve(&mut self, regs: &[String]) -> usize {
        let mark = self.reserved.len();
        self.reserved.extend(regs.iter().cloned());
        mark
    }

    pub(super) fn release(&mut self, mark: usize) {
        self.reserved.truncate(mark);
    }

    /// Whether any of `regs` is reserved by a write in progress.
    pub(super) fn reserves(&self, regs: &[String]) -> bool {
        regs.iter().any(|reg| self.reserved.contains(reg))
    }

    /// The registers a value occupies, for a value held in some.
    pub(super) fn occupies(&self, value: ValueId) -> Option<&[String]> {
        match self.held(value)? {
            Binding::Data { regs, .. } | Binding::LoopState { regs, .. } => Some(regs),
            Binding::Ptr { .. } => None,
        }
    }

    /// The registers evaluating a value would read.
    ///
    /// Lane-exact where a lane is what is selected — an extract of a value with
    /// storage reads the one register it names — and the union of what it
    /// reaches everywhere else. Over-stating costs a copy; under-stating would
    /// let a write land on a register still to be read.
    pub(super) fn reads_registers(&self, value: ValueId, into: &mut BTreeSet<String>) {
        if let Some(binding) = self.held(value) {
            into.extend(binding.registers().iter().cloned());
            return;
        }
        let Some(op) = self.f.operand_op(value) else {
            return;
        };
        if let crate::op::OpKind::Extract(lane) = self.f.op(op).kind {
            let source = self.f.op(op).operands[0];
            if let Some(regs) = self.occupies(source)
                && let Some(reg) = regs.get(lane as usize)
            {
                into.insert(reg.clone());
                return;
            }
        }
        self.op_reads_registers(op, into);
    }

    /// The registers running an operation would read: its operands, the places
    /// it produces into, and — all the way down — every step of every block of
    /// every region it owns, those steps' own regions included.
    pub(super) fn op_reads_registers(&self, op: OpId, into: &mut BTreeSet<String>) {
        let f = self.f;
        for index in 0..f.op(op).operands.len() {
            self.reads_registers(f.op(op).operands[index], into);
        }
        // A value produced into a place is a use of what that place held.
        for index in 0..f.op(op).results.len() {
            if let Some(place) = f.op(op).destination(index)
                && let Some(binding) = self.held(place)
            {
                into.extend(binding.registers().iter().cloned());
            }
        }
        for index in 0..f.op(op).regions.len() {
            let region = f.op(op).regions[index];
            for slot in 0..f.region(region).blocks.len() {
                let block = f.region(region).blocks[slot];
                for step in f.block_steps(block) {
                    self.op_reads_registers(step, into);
                }
                for yielded in 0..f.block(block).terminator.yields.len() {
                    self.reads_registers(f.block(block).terminator.yields[yielded], into);
                }
            }
        }
    }

    /// Where a value lives, if it is a plain data place — not a running loop's
    /// own value, whose registers the loop writes where no operation shows.
    pub(super) fn data_place(&self, value: ValueId) -> Option<(&[String], Ty)> {
        match self.regs[value.index()].as_ref() {
            Some(Binding::Data { regs, ty }) => Some((regs, *ty)),
            _ => None,
        }
    }

    /// Marks the registers a loop about to run carries, and gives back the mark
    /// that releases them.
    pub(super) fn carry(&mut self, regs: &[Vec<String>]) -> usize {
        let mark = self.carried.len();
        self.carried.push(regs.iter().flatten().cloned().collect());
        mark
    }

    pub(super) fn released(&mut self, mark: usize) {
        self.carried.truncate(mark);
    }

    /// Where a value lives, when it lives in what a running loop carries.
    ///
    /// The question is storage and not the name it goes by: an accumulator the
    /// loop carries, a binding folded into it, and a branch's result produced
    /// into that binding are one register under three names, and the loop
    /// writes that register at the backedge whichever name it is called by. So
    /// a result produced into any of them is the update the body performs.
    pub(super) fn loop_carried(&self, value: ValueId) -> Option<(&[String], Ty)> {
        let (regs, ty) = match self.regs[value.index()].as_ref()? {
            Binding::Data { regs, ty } | Binding::LoopState { regs, ty } => (regs.as_slice(), *ty),
            Binding::Ptr { .. } => return None,
        };
        self.carried
            .iter()
            .any(|held| regs.iter().all(|reg| held.contains(reg)))
            .then_some((regs, ty))
    }

    /// As [`K::data_place`], for a value the region being lowered gave storage
    /// to rather than one that outlives it.
    pub(super) fn data_place_here(&self, value: ValueId) -> Option<(&[String], Ty)> {
        (self.depth[value.index()] == self.open)
            .then(|| self.data_place(value))
            .flatten()
    }

    /// The registers a place holds, whatever kind of place it is.
    pub(super) fn place_regs(&self, value: ValueId) -> (Vec<String>, Ty) {
        match self.lookup(value) {
            Binding::Data { regs, ty } | Binding::LoopState { regs, ty } => (regs.clone(), *ty),
            Binding::Ptr { ty, .. } => {
                unreachable!("a data place is an address into {:?}", ty.elem)
            }
        }
    }

    pub(super) fn place_pointer(&self, value: ValueId) -> (String, PtrTy) {
        match self.lookup(value) {
            Binding::Ptr { reg, ty } => (reg.clone(), *ty),
            Binding::Data { ty, .. } | Binding::LoopState { ty, .. } => {
                unreachable!("a pointer place holds {ty:?}")
            }
        }
    }

    /// Instruction operand text: register name or typed immediate.
    pub(super) fn operand(&self, op: &Operand, s: Scalar) -> String {
        match op {
            Operand::Reg(r) => r.clone(),
            Operand::ImmI(v) => format!("{v}"),
            Operand::ImmF(x) => match s {
                Scalar::F32 => f32_bits(*x),
                Scalar::F64 => f64_bits(*x),
                other => unreachable!("inline float immediate of {other:?}"),
            },
        }
    }

    /// The operand in a register (movs an immediate).
    pub(super) fn materialize(&mut self, op: &Operand, s: Scalar) -> String {
        match op {
            Operand::Reg(r) => r.clone(),
            _ => {
                let r = self.reg(s);
                let text = self.operand(op, s);
                self.ins(format_args!("mov.{} {r}, {text};", mem(s)));
                r
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::K;

    #[test]
    fn body_formats_instructions_and_labels_without_changing_indentation() {
        let function = crate::Builder::new().into_function();
        let mut renderer = K::new(&function, 120, String::new());
        renderer.ins(format_args!("mov.u32 %r{}, {};", 1, 42));
        renderer.write_label("$Lloop_1");
        renderer.ins(format_args!("{{"));
        renderer.ins(format_args!("bar.sync 0;"));
        renderer.ins(format_args!("}}"));
        assert_eq!(
            renderer.body,
            "    mov.u32 %r1, 42;\n$Lloop_1:\n    {\n    bar.sync 0;\n    }\n",
        );
    }
}
