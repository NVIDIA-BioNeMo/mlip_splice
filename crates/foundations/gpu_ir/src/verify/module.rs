// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a set of launches must satisfy together, and what the target they are
//! rendered for admits.
//!
//! A launch verifies on its own, but a module is one text: the entries share a
//! namespace, and a constant array is emitted once under its name, so two
//! launches naming one constant differently would silently agree to disagree.
//! Nothing here is decidable inside one function, which is why it is its own
//! pass rather than a rule repeated per launch.
//!
//! **Names are emitted, so names are checked.** Every name that reaches the
//! text — an entry, a parameter, a shared array, a constant — must be a PTX
//! identifier. A name that is not cannot be escaped into one; it would inject
//! whatever it says into the middle of a declaration, so it is refused.

use std::collections::BTreeMap;

use super::{Checking, Result, Rule, Site};
use crate::Scalar;
use crate::arena::{Declaration, Function};
use crate::func::{DeviceFunction, Launch, Signature};
use crate::op::{ConstArray, OpKind, WriteMode};

/// Whether a name is a PTX identifier: what the grammar admits after
/// `followsym*`, starting with a letter or one of `_$%`.
///
/// The renderer prints names into declarations and instruction operands, so a
/// name outside this set is not a name it can print — there is no escape for
/// one, only a refusal.
pub(super) fn is_ptx_identifier(name: &str) -> bool {
    let mut held = name.chars();
    let Some(first) = held.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || matches!(first, '_' | '$' | '%'))
        && held.all(|held| held.is_ascii_alphanumeric() || matches!(held, '_' | '$' | '%'))
}

/// Everything the launches of one module must satisfy together, on the target
/// they are rendered for.
pub(super) fn check(launches: &[&Launch], sm: i32) -> Result {
    if !matches!(
        sm,
        50 | 52
            | 53
            | 60
            | 61
            | 62
            | 70
            | 72
            | 75
            | 80
            | 86
            | 87
            | 89
            | 90
            | 100
            | 101
            | 103
            | 110
            | 120
            | 121
    ) {
        return Err(super::Diagnostic {
            function: launches.first().map_or_else(
                || "<module>".to_string(),
                |launch| launch.function.name.clone(),
            ),
            rule: Rule::Abi,
            at: Site::Function,
            detail: format!("sm_{sm} is not a supported PTX target"),
        });
    }
    let mut entries: BTreeMap<&str, usize> = BTreeMap::new();
    let mut declared: BTreeMap<&str, &Signature> = BTreeMap::new();
    let mut constants: BTreeMap<&str, (Scalar, &[u8])> = BTreeMap::new();
    for (index, launch) in launches.iter().enumerate() {
        let function = &launch.function;
        let body = launch.body();
        let checking = Checking {
            name: &function.name,
            f: body,
        };
        named(&checking, &function.name, "an entry")?;
        if let Some(held) = entries.insert(&function.name, index) {
            return checking.fail(
                Rule::Abi,
                Site::Function,
                format!(
                    "launch {index} and launch {held} are both entered as {}",
                    function.name
                ),
            );
        }
        declarations(&checking)?;
        target(&checking, sm)?;
        for callee in called(body) {
            let Some(stated) = callee.signature() else {
                continue;
            };
            let name = callee.name.as_str();
            if let Some(before) = declared.insert(name, stated)
                && before != stated
            {
                return checking.fail(
                    Rule::Abi,
                    Site::Function,
                    format!("{name} is called under two contracts and the module declares it once"),
                );
            }
        }
        for (name, array) in module_constants(body) {
            named(&checking, name, "a constant")?;
            if let Some(before) = constants.insert(name, array)
                && before != array
            {
                return checking.fail(
                    Rule::Memory,
                    Site::Function,
                    format!(
                        "constant {name} holds {} bytes of {:?} here and {} bytes of {:?} in \
                         another launch, and the module emits it once",
                        array.1.len(),
                        array.0,
                        before.1.len(),
                        before.0
                    ),
                );
            }
        }
    }
    // A symbol is defined here or resolved by the linker, never both: an entry,
    // an external function, and a constant must all have distinct emitted names.
    for (name, index) in &entries {
        if declared.contains_key(name) {
            let checking = Checking {
                name,
                f: launches[*index].body(),
            };
            return checking.fail(
                Rule::Abi,
                Site::Function,
                format!("{name} is entered here and called as a function defined elsewhere"),
            );
        }
    }
    for constant in constants.keys() {
        let symbol = format!("const_{constant}");
        if let Some(index) = entries.get(symbol.as_str()) {
            let checking = Checking {
                name: symbol.as_str(),
                f: launches[*index].body(),
            };
            return checking.fail(
                Rule::Abi,
                Site::Function,
                format!("{symbol} names both an entry and constant {constant}"),
            );
        }
        if declared.contains_key(symbol.as_str()) {
            let checking = Checking {
                name: launches[0].function.name.as_str(),
                f: launches[0].body(),
            };
            return checking.fail(
                Rule::Abi,
                Site::Function,
                format!("{symbol} names both an external function and constant {constant}"),
            );
        }
    }
    Ok(())
}

/// The functions one body calls, in arena order.
fn called(f: &Function) -> Vec<&DeviceFunction> {
    f.live_ops()
        .into_iter()
        .filter_map(|op| match &f.op(op).kind {
            OpKind::Call(callee) => Some(callee.as_ref()),
            _ => None,
        })
        .collect()
}

/// The names one launch declares: each a PTX identifier, and no two the same,
/// since a parameter and a shared array of one name would declare one symbol
/// twice.
fn declarations(c: &Checking<'_>) -> Result {
    let mut held: BTreeMap<&str, &'static str> = BTreeMap::new();
    for declared in c.f.declarations() {
        let (name, what) = match declared {
            Declaration::Param(param) => (param.name.as_str(), "a parameter"),
            Declaration::Shared(array) => (array.name.as_str(), "a shared array"),
        };
        named(c, name, what)?;
        if let Some(before) = held.insert(name, what) {
            return c.fail(
                Rule::Abi,
                Site::Function,
                format!("{name} is {what} and {before} of one kernel"),
            );
        }
    }
    Ok(())
}

fn named(c: &Checking<'_>, name: &str, what: &str) -> Result {
    c.require(is_ptx_identifier(name), Rule::Abi, Site::Function, || {
        format!("{name:?} is {what} and not a name PTX can be given")
    })
}

/// Every constant array a launch reads, by the name it is emitted under.
fn module_constants(f: &Function) -> Vec<(&str, (Scalar, &[u8]))> {
    let mut held = Vec::new();
    for op in f.live_ops() {
        if let OpKind::ConstLoad(array) = &f.op(op).kind {
            let ConstArray {
                name, scalar, data, ..
            } = array.as_ref();
            held.push((name.as_str(), (*scalar, data.as_ref())));
        }
    }
    held
}

/// The operations whose lowering a target has to have an instruction for.
///
/// Almost nothing is target-dependent: the backend selects packed pair forms
/// where they exist and lane-split scalars where they do not, so arithmetic
/// lowers for every target. Atomics are the exception — a reduction of a float
/// width the target has no `red` for cannot be split into anything equivalent,
/// so the program is refused for that target rather than lowered into
/// something with different rounding or different atomicity.
fn target(c: &Checking<'_>, sm: i32) -> Result {
    for op in c.f.live_ops() {
        let adds = matches!(
            c.f.op(op).kind,
            OpKind::Write {
                mode: WriteMode::Add,
                ..
            } | OpKind::FetchAdd
        );
        if adds {
            let scalar = c.f.data_ty(c.f.op(op).operands[1]).scalar;
            let wanted = atomic_add_since(scalar);
            c.require(sm >= wanted, Rule::Memory, Site::Op(op), || {
                format!(
                    "an atomic add of {scalar:?} needs sm_{wanted} and this renders for sm_{sm}"
                )
            })?;
        }
    }
    Ok(())
}

/// The first target with a `red.global.add` — and so an `atom.global.add` — of
/// this scalar.
fn atomic_add_since(scalar: Scalar) -> i32 {
    match scalar {
        Scalar::F64 => 60,
        Scalar::F16 => 70,
        Scalar::BF16 => 90,
        _ => 0,
    }
}
