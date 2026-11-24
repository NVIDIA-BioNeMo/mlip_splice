// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `gpu_ir` → PTX. Lowering is total: every well-typed program renders for every
//! `sm`. Instruction selection happens here — packed pair forms (`f32x2` on
//! sm_100+, `f16x2`/`bf16x2`) where the target has them, lane-split scalar
//! instructions everywhere else — so no candidate is ever lost to the device
//! generation. Type mismatches are construction bugs and panic.

mod divide;
mod expr;
mod kernel;
mod math;
mod memory;
mod stmt;
mod takeover;
#[cfg(test)]
mod tests;
mod text;
mod types;

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use kernel::K;
use text::{ByteCount, Text};

pub use types::vector_form;

use crate::arena::{BlockId, Function, OpId};
use crate::func::{Launch, Signature};
use crate::op::OpKind;
use crate::verify::Diagnostic;
use crate::{Scalar, VerifiedModule, inline_calls, verified_module};

/// The lowest PTX ISA `.version` that knows the target, as `[major, minor]`:
/// the family-specific Blackwell archs (sm_103, sm_121) entered the ISA at 8.8
/// (CUDA 12.9); everything else we emit is covered by 8.7.
pub fn isa_version(sm: i32) -> [u8; 2] {
    match sm {
        110 => [9, 0],
        103 | 121 => [8, 8],
        _ => [8, 7],
    }
}

/// Every operation a body runs or reads, in the order the body reaches them.
///
/// A body is a list of the operations it runs; an operand it never reads is
/// nothing the kernel does, so what a slice left behind in the arena is not
/// emitted and does not declare a constant.
fn reachable_ops(f: &Function, block: BlockId, seen: &mut BTreeSet<usize>, out: &mut Vec<OpId>) {
    for &op in &f.block(block).ops {
        if f.op(op).placed {
            visit_op(f, op, seen, out);
        }
    }
    for &value in &f.block(block).terminator.yields {
        visit_value(f, value, seen, out);
    }
}

fn visit_op(f: &Function, op: OpId, seen: &mut BTreeSet<usize>, out: &mut Vec<OpId>) {
    if !seen.insert(op.index()) {
        return;
    }
    out.push(op);
    for &operand in &f.op(op).operands {
        visit_value(f, operand, seen, out);
    }
    for &region in &f.op(op).regions {
        for &block in &f.region(region).blocks {
            reachable_ops(f, block, seen, out);
        }
    }
}

fn visit_value(
    f: &Function,
    value: crate::arena::ValueId,
    seen: &mut BTreeSet<usize>,
    out: &mut Vec<OpId>,
) {
    if let Some(op) = f.defining_op(value) {
        visit_op(f, op, seen, out);
    }
}

/// The PTX for a set of launches, which are the module's entry points.
///
/// A call to a function this language defines is run where it stands; one to a
/// function declared elsewhere stays a call, and the module declares the symbol
/// for the linker to resolve. The program is verified before a byte is emitted,
/// so what the renderer meets is a program every rule below already holds for. A
/// program that breaks one is refused here, in the program's own words, rather
/// than halfway through the text.
pub fn render_ptx_module(launches: &[Launch], sm: i32) -> String {
    match render_module(launches, sm) {
        Ok(text) => text,
        Err(why) => panic!("{why}"),
    }
}

/// [`render_ptx_module`], for a caller that answers for an unverified program
/// itself.
pub fn render_module(launches: &[Launch], sm: i32) -> Result<String, Diagnostic> {
    with_verified_module(launches, sm, render_verified)
}

/// Exact capacities requested for the module and kernel-body text buffers.
///
/// This excludes IR, register bookkeeping, formatting temporaries, allocator
/// overhead and residency. It is not a bound on renderer or process memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextSize {
    /// The complete module's UTF-8 bytes, retained by the returned string.
    pub module_bytes: usize,
    /// Module buffer plus the largest simultaneously live kernel-body buffer.
    pub peak_buffer_bytes: usize,
}

/// Count UTF-8 bytes before constructing any module or kernel-body text.
///
/// Verification, inlining and instruction selection still run and allocate
/// their own working state. The counter follows the same operations as the
/// text writer; it does not estimate bytes from statement counts.
pub fn measure_module(launches: &[Launch], sm: i32) -> Result<TextSize, Diagnostic> {
    with_verified_module(launches, sm, measure_verified)
}

/// [`measure_module`] for a module whose verification is already established.
/// Instruction selection still runs; module and kernel-body text are not built.
pub fn measure_verified(module: &VerifiedModule<'_>) -> TextSize {
    text_layout(module).size()
}

fn with_verified_module<R>(
    launches: &[Launch],
    sm: i32,
    take: impl FnOnce(&VerifiedModule<'_>) -> R,
) -> Result<R, Diagnostic> {
    // Only a launch with a body to run in it is rebuilt. A body the builder
    // already wrote is rendered where it stands, so a pass that has nothing to
    // do costs nothing.
    let mut expanded = Vec::with_capacity(launches.len());
    for launch in launches {
        let inlines = launch
            .function
            .body()
            .is_some_and(crate::inline::inlines_anywhere);
        expanded.push(match inlines {
            false => None,
            true => {
                let mut copy = launch.clone();
                inline_calls(&mut copy.function).map_err(|why| Diagnostic {
                    function: copy.function.name.clone(),
                    rule: crate::verify::Rule::Region,
                    at: crate::verify::Site::Function,
                    detail: why,
                })?;
                Some(copy)
            }
        });
    }
    let resolved = launches
        .iter()
        .zip(&expanded)
        .map(|(launch, inlined)| inlined.as_ref().unwrap_or(launch))
        .collect::<Vec<_>>();
    let verified = verified_module(&resolved, sm)?;
    Ok(take(&verified))
}

struct TextLayout {
    module_bytes: usize,
    body_bytes: Vec<usize>,
}

impl TextLayout {
    fn size(&self) -> TextSize {
        TextSize {
            module_bytes: self.module_bytes,
            peak_buffer_bytes: self
                .module_bytes
                .checked_add(self.body_bytes.iter().copied().max().unwrap_or(0))
                .expect("PTX text buffers exceed usize"),
        }
    }
}

fn text_layout(module: &VerifiedModule<'_>) -> TextLayout {
    let mut out = ByteCount::default();
    let body_bytes = write_module(module, &mut out, |_| ByteCount::default());
    TextLayout {
        module_bytes: out.0,
        body_bytes,
    }
}

/// Render into exactly sized text buffers, after counting their bytes.
/// Non-text working allocations are not included in [`TextSize`].
pub fn render_verified(module: &VerifiedModule<'_>) -> String {
    let layout = text_layout(module);
    let mut out = String::with_capacity(layout.module_bytes);
    let body_bytes = write_module(module, &mut out, |index| {
        String::with_capacity(layout.body_bytes[index])
    });
    assert_eq!(
        body_bytes, layout.body_bytes,
        "PTX body size changed after counting"
    );
    assert_eq!(
        out.len(),
        layout.module_bytes,
        "PTX module size changed after counting"
    );
    assert_eq!(
        out.capacity(),
        layout.module_bytes,
        "PTX module buffer grew"
    );
    out
}

fn write_module<T: Text>(
    module: &VerifiedModule<'_>,
    out: &mut T,
    mut body: impl FnMut(usize) -> T,
) -> Vec<usize> {
    let kernels = module.launches();
    let sm = module.target();
    let [major, minor] = isa_version(sm);
    let _ = writeln!(out, ".version {major}.{minor}");
    let _ = writeln!(out, ".target sm_{sm}");
    let _ = writeln!(out, ".address_size 64");
    let mut constants = HashMap::<String, (Arc<[u8]>, Scalar)>::new();
    let mut declared = HashMap::<String, Signature>::new();
    for kernel in kernels {
        let f = kernel.body();
        let mut seen = BTreeSet::new();
        let mut ops = Vec::new();
        reachable_ops(f, f.entry(), &mut seen, &mut ops);
        for op in ops {
            match &f.op(op).kind {
                OpKind::ConstLoad(array) => {
                    constants
                        .entry(array.name.clone())
                        .or_insert((array.data.clone(), array.scalar));
                }
                OpKind::Call(callee) => {
                    if let Some(signature) = callee.signature() {
                        declared
                            .entry(callee.name.clone())
                            .or_insert_with(|| signature.clone());
                    }
                }
                _ => {}
            }
        }
    }
    let mut constants = constants.into_iter().collect::<Vec<_>>();
    constants.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, (data, elem)) in constants {
        let align = elem.size().min(16);
        let _ = write!(
            out,
            ".global .align {align} .b8 const_{name}[{}] = {{",
            data.len()
        );
        for (index, byte) in data.iter().enumerate() {
            if index != 0 {
                let _ = out.write_str(", ");
            }
            let _ = write!(out, "0x{byte:02x}");
        }
        let _ = writeln!(out, "}};");
    }
    let mut declared = declared.into_iter().collect::<Vec<_>>();
    declared.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, signature) in declared {
        let _ = out.write_str(&extern_function(&name, &signature));
    }
    kernels
        .iter()
        .enumerate()
        .map(|(index, kernel)| {
            let mut k = K::new(kernel.body(), sm, body(index));
            k.kernel(kernel, out)
        })
        .collect()
}

/// One symbol this module calls and does not define, as the prototype the
/// linker resolves against.
///
/// The parameter list is what the declaration states and nothing else: no width
/// is promoted and no argument is inferred, because the contract is the caller's
/// and the definition on the other side of the link was written to it.
fn extern_function(name: &str, signature: &Signature) -> String {
    let mut out = String::new();
    let returns = match signature.results.first() {
        None => String::new(),
        Some(&result) => format!("(.param .{} {name}_retval0) ", param_space(result)),
    };
    let taken = signature
        .parameters
        .iter()
        .enumerate()
        .map(|(index, param)| {
            format!(
                ".param .{} {}_param_{index}",
                param_space(param.ty.val_ty()),
                param.name
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, ".extern .func {returns}{name}({taken});");
    out
}

/// The `.param` a value of this type is passed in: its own width, as bits. A
/// pointer is an address, and every address this language forms is 64 bits.
fn param_space(ty: crate::ValTy) -> &'static str {
    let size = match ty {
        crate::ValTy::Data(held) => held.scalar.size(),
        crate::ValTy::Ptr(_) => 8,
    };
    match size {
        1 => "b8",
        2 => "b16",
        4 => "b32",
        8 => "b64",
        other => unreachable!("a {other}-byte call parameter"),
    }
}
