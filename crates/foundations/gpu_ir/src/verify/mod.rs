// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a program must satisfy before it can be rendered.
//!
//! Rendering is total over *verified* programs: every rule the backend relies
//! on is decided here, once, and said in the program's own words. A renderer
//! that meets an unverified program has no way to explain what is wrong with
//! it — it is halfway through emitting text — so the boundary is that
//! [`crate::ptx::render_ptx_module`] verifies first and then renders code that
//! cannot fail.
//!
//! The rules are grouped by what they are about, not by where the backend
//! happens to need them: `structure` for ownership, regions, terminators and
//! dominance; `typing` for what each operation's operands and results must
//! hold and which arithmetic has a meaning at a scalar; `memory` for
//! addressing, access widths, alignment and the storage a result takes over;
//! `converge` for the threads a barrier needs; `signature` for what a
//! function with no body here must state instead.

mod converge;
mod memory;
mod module;
mod shape;
mod signature;
mod structure;
#[cfg(test)]
mod tests;
mod typing;

use std::fmt;

use crate::arena::{BlockId, Function, OpId, RegionId, ValueId};
use crate::func::{DeviceFunction, Implementation, Launch};
use crate::{Declaration, MAX_THREADS_PER_BLOCK, REGISTER_CAP};

/// Which rule a program broke.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Rule {
    /// A definition and the thing that claims it disagree.
    Ownership,
    /// A region, its blocks, or what one hands back.
    Region,
    /// A value read where it is not defined.
    Dominance,
    /// An operand or a result of the wrong type, or arithmetic with no meaning
    /// at its scalar.
    Type,
    /// What an operation does to memory, said where it cannot be done.
    Effect,
    /// What a function may take and hand back, at a launch or at a call.
    Abi,
    /// The storage a produced value takes over.
    Aliasing,
    /// Addressing and the widths a space admits.
    Memory,
    /// An access wider than the address is proven to be aligned to.
    Alignment,
    /// The threads that must arrive together.
    Convergence,
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Rule::Ownership => "ownership",
            Rule::Region => "region",
            Rule::Dominance => "dominance",
            Rule::Type => "type",
            Rule::Effect => "effect",
            Rule::Abi => "abi",
            Rule::Aliasing => "aliasing",
            Rule::Memory => "memory",
            Rule::Alignment => "alignment",
            Rule::Convergence => "convergence",
        };
        f.write_str(name)
    }
}

/// Where a program broke a rule. The ids index the function named by the
/// diagnostic, which is the one being verified.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Site {
    Function,
    Op(OpId),
    Value(ValueId),
    Block(BlockId),
    Region(RegionId),
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Site::Function => f.write_str("the function"),
            Site::Op(op) => write!(f, "operation {}", op.index()),
            Site::Value(value) => write!(f, "value %{}", value.index()),
            Site::Block(block) => write!(f, "block {}", block.index()),
            Site::Region(region) => write!(f, "region {}", region.index()),
        }
    }
}

/// One reason a program cannot be rendered: which rule, where, and what about
/// that place breaks it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub function: String,
    pub rule: Rule,
    pub at: Site,
    pub detail: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} at {}: {}",
            self.function, self.rule, self.at, self.detail
        )
    }
}

impl std::error::Error for Diagnostic {}

/// The function being verified, and the name every diagnostic carries.
pub(crate) struct Checking<'a> {
    pub(crate) name: &'a str,
    pub(crate) f: &'a Function,
}

impl Checking<'_> {
    pub(crate) fn fail<T>(&self, rule: Rule, at: Site, detail: impl Into<String>) -> Result<T> {
        Err(Diagnostic {
            function: self.name.to_string(),
            rule,
            at,
            detail: detail.into(),
        })
    }

    pub(crate) fn require(
        &self,
        held: bool,
        rule: Rule,
        at: Site,
        detail: impl FnOnce() -> String,
    ) -> Result<()> {
        if held {
            Ok(())
        } else {
            self.fail(rule, at, detail())
        }
    }
}

pub type Result<T = ()> = std::result::Result<T, Diagnostic>;

/// Everything a device function must satisfy on its own: structure, types,
/// memory and the threads its barriers need, or — where the definition is
/// elsewhere — that its declaration states a contract at all.
///
/// A callee is verified where it is called, and a callee cannot call its
/// caller — an [`crate::OpKind::Call`] holds the function it runs, so the
/// callee existed before the caller did — which is why this terminates without
/// a recursion guard.
pub fn verify(function: &DeviceFunction) -> Result {
    match diagnostics(function).into_iter().next() {
        None => Ok(()),
        Some(first) => Err(first),
    }
}

/// Every rule this function breaks that is worth stating, in one fixed order.
///
/// A first refusal is enough to stop a compilation and too little to repair a
/// program by: an editor holding a half-finished body wants the list. So the
/// passes run in groups, and a group whose prerequisite failed does not run —
/// what a broken shape makes of a type rule is noise, not a second finding.
///
/// The `shape` pass gates everything, because every later pass reads a program by its
/// positions. Past it the four groups are independent, so all four are asked
/// and each answers at most once.
pub fn diagnostics(function: &DeviceFunction) -> Vec<Diagnostic> {
    match &function.implementation {
        Implementation::Declared(stated) => signature::check(&function.name, stated)
            .err()
            .into_iter()
            .collect(),
        Implementation::Body(body) => body_diagnostics(&function.name, body),
    }
}

fn body_diagnostics(name: &str, body: &Function) -> Vec<Diagnostic> {
    let checking = Checking { name, f: body };
    if let Err(broken) = shape::check(&checking) {
        return vec![broken];
    }
    let mut found = Vec::new();
    found.extend(structure::check(&checking).err());
    found.extend(memory::effects(&checking).err());
    match typing::check(&checking) {
        Err(broken) => found.push(broken),
        Ok(()) => found.extend(memory::check(&checking).err()),
    }
    found.extend(converge::check(&checking).err());
    found
}

/// A frozen function that has passed every function-local rule.
pub struct VerifiedFunction {
    name: String,
    function: crate::FrozenFunction,
}

impl VerifiedFunction {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn function(&self) -> &crate::FrozenFunction {
        &self.function
    }
}

pub fn verify_frozen_function(
    name: impl Into<String>,
    function: crate::FrozenFunction,
) -> Result<VerifiedFunction> {
    let name = name.into();
    match body_diagnostics(&name, function.function())
        .into_iter()
        .next()
    {
        None => Ok(VerifiedFunction { name, function }),
        Some(first) => Err(first),
    }
}

/// Everything a module of launches must satisfy before it is rendered for
/// `sm`: each launch on its own, what they must agree on together, and what
/// that target has an instruction for.
///
/// This is what [`crate::ptx::render_ptx_module`] verifies. A launch checked
/// on its own is not enough: the entries share a namespace and a constant is
/// emitted once under its name, so two launches disagreeing about one is a
/// fault of the module and of neither function.
pub struct VerifiedModule<'a> {
    launches: Vec<&'a Launch>,
    sm: i32,
}

impl VerifiedModule<'_> {
    pub(crate) fn launches(&self) -> &[&Launch] {
        &self.launches
    }

    pub(crate) fn target(&self) -> i32 {
        self.sm
    }
}

pub fn verified_module<'a>(launches: &[&'a Launch], sm: i32) -> Result<VerifiedModule<'a>> {
    for launch in launches {
        verify_launch(launch)?;
    }
    module::check(launches, sm)?;
    Ok(VerifiedModule {
        launches: launches.to_vec(),
        sm,
    })
}

pub fn verify_module(launches: &[&Launch], sm: i32) -> Result {
    verified_module(launches, sm).map(drop)
}

/// As [`verify`], plus what launching one asks: a kernel is a body, hands
/// nothing back, takes what a host can pass, and states a block and a register
/// cap a target admits.
pub fn verify_launch(launch: &Launch) -> Result {
    verify(&launch.function)?;
    let name = launch.function.name.as_str();
    let Some(body) = launch.function.body() else {
        return Err(Diagnostic {
            function: name.to_string(),
            rule: Rule::Abi,
            at: Site::Function,
            detail: "it is declared elsewhere, and a launch runs a body".to_string(),
        });
    };
    let checking = Checking { name, f: body };
    let returned = checking.f.returned().len();
    checking.require(returned == 0, Rule::Abi, Site::Function, || {
        format!("a launched kernel hands back nothing and this hands back {returned}")
    })?;
    checking.require(
        (1..=MAX_THREADS_PER_BLOCK).contains(&launch.max_threads),
        Rule::Abi,
        Site::Function,
        || {
            format!(
                "a launch promises {} threads per block, outside the 1..={MAX_THREADS_PER_BLOCK} \
                 every target admits",
                launch.max_threads
            )
        },
    )?;
    if let Some(cap) = launch.max_registers {
        checking.require(
            REGISTER_CAP.contains(&cap),
            Rule::Abi,
            Site::Function,
            || {
                format!(
                    "a register cap of {cap} is outside the {}..={} `.maxnreg` admits",
                    REGISTER_CAP.start(),
                    REGISTER_CAP.end()
                )
            },
        )?;
    }
    for (index, declared) in checking.f.declarations().iter().enumerate() {
        match declared {
            Declaration::Param(param) => {
                checking.require(
                    !matches!(param.ty, crate::ParamTy::Scalar(crate::Scalar::Pred)),
                    Rule::Abi,
                    Site::Function,
                    || format!("argument {index} is a predicate, which no kernel parameter holds"),
                )?;
            }
            Declaration::Shared(array) => {
                checking.require(array.len > 0, Rule::Abi, Site::Function, || {
                    format!("shared array {} holds {} elements", array.name, array.len)
                })?;
                checking.require(
                    array.elem.size().checked_mul(array.len).is_some(),
                    Rule::Abi,
                    Site::Function,
                    || format!("shared array {} has a byte size that overflows", array.name),
                )?;
            }
        }
    }
    Ok(())
}
