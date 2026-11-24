// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One function, printed whole: the arena that is its body, or the contract it
//! states in place of one.
//!
//! Every list goes out in index order and every entity states its own id, so
//! what comes back reconstructs the same indices — which is what makes the
//! bytes a rendered arena finishes to the bytes it came from. A signature is
//! printed by the same rule: each of its lists goes out entry by entry, so two
//! that disagree in length are printed as they are rather than reconciled on
//! the way out.

use std::fmt::Write;
use std::sync::Arc;

use super::{
    binary_name, comparison_name, escaped, evict_name, hex, scalar_name, space_name, sreg_name,
    sync_name, trip_name, unary_name, write_mode_name,
};
use crate::arena::{Declaration, Def, Function, OpId};
use crate::func::{Access, DeviceFunction, Implementation, Signature};
use crate::op::{ConstArray, OpKind};
use crate::{Param, ParamTy, ValTy};

/// The constant arrays a body reads, by identity, in first-use order over its
/// operations. One array used twice is written once and read twice.
pub(crate) fn constants(f: &Function) -> Vec<Arc<ConstArray>> {
    let mut pool: Vec<Arc<ConstArray>> = Vec::new();
    for index in 0..f.num_ops() {
        if let OpKind::ConstLoad(array) = &f.op(f.op_at(index)).kind
            && !pool.iter().any(|held| Arc::ptr_eq(held, array))
        {
            pool.push(array.clone());
        }
    }
    pool
}

fn valty(ty: ValTy) -> String {
    match ty {
        ValTy::Data(held) => format!("data {} x{}", scalar_name(held.scalar), held.lanes),
        ValTy::Ptr(held) => format!("ptr {} {}", scalar_name(held.elem), space_name(held.space)),
    }
}

fn param(held: &Param) -> String {
    let ty = match held.ty {
        ParamTy::Ptr(elem) => format!("ptr {}", scalar_name(elem)),
        ParamTy::Scalar(scalar) => format!("scalar {}", scalar_name(scalar)),
    };
    format!("param {} {ty}", escaped(&held.name))
}

/// What a function does to the memory behind one parameter, as the two facts it
/// is: which way it is touched, and how wide the widest access through it is.
/// The width is stated even where nothing is touched, so what a signature holds
/// is what comes back.
fn access(held: Access) -> String {
    let touched = match (held.read, held.written) {
        (false, false) => "none",
        (true, false) => "read",
        (false, true) => "write",
        (true, true) => "read-write",
    };
    format!("access {touched} lanes {}", held.lanes)
}

/// One device function, whole: its name, and where its implementation is.
///
/// `symbol_of` answers which module function a call in the body runs; a
/// declaration has no body and asks it nothing.
pub fn render_device_function(
    f: &DeviceFunction,
    symbol_of: &dyn Fn(OpId) -> usize,
    indent: &str,
    out: &mut String,
) {
    match &f.implementation {
        Implementation::Body(body) => {
            let _ = writeln!(out, "{indent}function {} body", escaped(&f.name));
            render_function(body, symbol_of, indent, out);
        }
        Implementation::Declared(signature) => {
            let _ = writeln!(out, "{indent}function {} declared", escaped(&f.name));
            render_signature(signature, indent, out);
        }
    }
}

/// The whole contract of a function defined elsewhere.
///
/// Parameters, what each is touched by and what the function hands back are
/// three lists, printed one entry per line in order: the *n*-th access is what
/// the *n*-th parameter is touched by. They are printed apart rather than
/// paired because a signature may hold lists of different lengths — that is
/// something [`crate::verify()`] refuses and an editor is allowed to be holding,
/// so it is something this text has to be able to say.
pub fn render_signature(signature: &Signature, indent: &str, out: &mut String) {
    for held in &signature.parameters {
        let _ = writeln!(out, "{indent}{}", param(held));
    }
    for &held in &signature.accesses {
        let _ = writeln!(out, "{indent}{}", access(held));
    }
    for &result in &signature.results {
        let _ = writeln!(out, "{indent}result {}", valty(result));
    }
    let held = match signature.sync {
        None => "none",
        Some(sync) => sync_name(sync),
    };
    let _ = writeln!(out, "{indent}sync {held}");
}

fn kind(held: &OpKind, pool: &[Arc<ConstArray>], symbol: usize) -> String {
    match held {
        OpKind::IntConst { scalar, value } => {
            format!("intconst {} {value}", scalar_name(*scalar))
        }
        OpKind::FloatConst { scalar, value } => format!(
            "floatconst {} 0x{:016x}",
            scalar_name(*scalar),
            value.to_bits()
        ),
        OpKind::Sreg(sreg) => format!("sreg {}", sreg_name(*sreg)),
        OpKind::Load { lanes, nc, evict } => {
            format!("load lanes {lanes} nc {nc} evict {}", evict_name(*evict))
        }
        OpKind::Offset => "offset".to_string(),
        OpKind::Un(op) => format!("un {}", unary_name(*op)),
        OpKind::Bin(op) => format!("bin {}", binary_name(*op)),
        OpKind::Fma => "fma".to_string(),
        OpKind::Cmp(op) => format!("cmp {}", comparison_name(*op)),
        OpKind::Select => "select".to_string(),
        OpKind::Cast(scalar) => format!("cast {}", scalar_name(*scalar)),
        OpKind::Splat(lanes) => format!("splat {lanes}"),
        OpKind::Extract(lane) => format!("extract {lane}"),
        OpKind::Pack => "pack".to_string(),
        OpKind::ShflDown => "shfl-down".to_string(),
        OpKind::ActiveMask => "active-mask".to_string(),
        OpKind::ShflFrom => "shfl-from".to_string(),
        OpKind::WarpAny => "warp-any".to_string(),
        OpKind::ConstLoad(array) => format!(
            "const-load ${}",
            pool.iter()
                .position(|held| Arc::ptr_eq(held, array))
                .expect("the pool was built from these operations")
        ),
        OpKind::Bind => "bind".to_string(),
        OpKind::Write { mode, evict } => format!(
            "write {} evict {}",
            write_mode_name(*mode),
            evict_name(*evict)
        ),
        OpKind::FetchAdd => "fetch-add".to_string(),
        OpKind::If => "if".to_string(),
        OpKind::Loop { index, trip } => {
            format!("loop {} {}", scalar_name(*index), trip_name(*trip))
        }
        OpKind::Scope => "scope".to_string(),
        OpKind::Call(_) => format!("call @{symbol}"),
        OpKind::Break => "break".to_string(),
        OpKind::Ret => "ret".to_string(),
        OpKind::Bar => "bar".to_string(),
    }
}

fn ids<T: Copy>(prefix: char, held: &[T], index: impl Fn(T) -> usize) -> String {
    let mut text = String::from("[");
    for (position, &id) in held.iter().enumerate() {
        if position > 0 {
            text.push(' ');
        }
        let _ = write!(text, "{prefix}{}", index(id));
    }
    text.push(']');
    text
}

/// One arena, every line prefixed by `indent`.
///
/// `symbol_of` answers which module function a call runs, by the operation's id
/// *in the arena it was handed* — a call is a fact about the module a body
/// belongs to, and this file only knows about one body.
///
/// What is written down is the program, so the ids are the program's: an arena
/// an editor deleted from is written from [`Function::compacted`], where the
/// holes close up. The two numberings agree for every arena nothing was deleted
/// from, which is every arena that was only built.
pub fn render_function(
    f: &Function,
    symbol_of: &dyn Fn(OpId) -> usize,
    indent: &str,
    out: &mut String,
) {
    if f.has_deletions() {
        let live = f.live_ops();
        let compacted = f.compacted();
        // The symbol closure was handed this arena's ids, so a call in the
        // dense copy is asked about the one it came from.
        return render_dense(&compacted, &|op| symbol_of(live[op.index()]), indent, out);
    }
    render_dense(f, symbol_of, indent, out);
}

fn render_dense(f: &Function, symbol_of: &dyn Fn(OpId) -> usize, indent: &str, out: &mut String) {
    let pool = constants(f);
    for (index, array) in pool.iter().enumerate() {
        let _ = writeln!(
            out,
            "{indent}constant ${index} {} {} {}",
            escaped(&array.name),
            scalar_name(array.scalar),
            hex(&array.data)
        );
    }
    for declared in f.declarations() {
        let _ = match declared {
            Declaration::Param(held) => writeln!(out, "{indent}{}", param(held)),
            Declaration::Shared(array) => writeln!(
                out,
                "{indent}shared {} {} {}",
                escaped(&array.name),
                scalar_name(array.elem),
                array.len
            ),
        };
    }
    for index in 0..f.num_values() {
        let value = f.value_at(index);
        let held = f.value(value);
        let definition = match held.def {
            Def::Param { block, index } => format!("param ^{} {index}", block.index()),
            Def::Result { op, index } => format!("result *{} {index}", op.index()),
        };
        let _ = writeln!(
            out,
            "{indent}value %{index} {} {definition}",
            valty(held.ty)
        );
    }
    for index in 0..f.num_ops() {
        let id = f.op_at(index);
        let op = f.op(id);
        let into = {
            let mut text = String::from("[");
            for (position, destination) in op.into.iter().enumerate() {
                if position > 0 {
                    text.push(' ');
                }
                match destination {
                    None => text.push_str("none"),
                    Some(place) => {
                        let _ = write!(text, "%{}", place.index());
                    }
                }
            }
            text.push(']');
            text
        };
        let symbol = match op.kind {
            OpKind::Call(_) => symbol_of(id),
            _ => 0,
        };
        let _ = writeln!(
            out,
            "{indent}op *{index} {} {} operands {} results {} into {into} regions {}",
            if op.placed { "step" } else { "operand" },
            kind(&op.kind, &pool, symbol),
            ids('%', &op.operands, |id| id.index()),
            ids('%', &op.results, |id| id.index()),
            ids('!', &op.regions, |id| id.index()),
        );
    }
    for index in 0..f.num_blocks() {
        let block = f.block(f.block_at(index));
        let region = match block.region {
            None => "none".to_string(),
            Some(held) => format!("!{}", held.index()),
        };
        let _ = writeln!(
            out,
            "{indent}block ^{index} region {region} params {} ops {} yields {}",
            ids('%', &block.params, |id| id.index()),
            ids('*', &block.ops, |id| id.index()),
            ids('%', &block.terminator.yields, |id| id.index()),
        );
    }
    for index in 0..f.num_regions() {
        let region = f.region(f.region_at(index));
        let owner = match region.owner {
            None => "none".to_string(),
            Some(held) => format!("*{}", held.index()),
        };
        let _ = writeln!(
            out,
            "{indent}region !{index} owner {owner} blocks {}",
            ids('^', &region.blocks, |id| id.index()),
        );
    }
    let _ = writeln!(out, "{indent}entry ^{}", f.entry().index());
    for (value, name) in f.names() {
        let _ = writeln!(out, "{indent}name %{} {}", value.index(), escaped(name));
    }
}
