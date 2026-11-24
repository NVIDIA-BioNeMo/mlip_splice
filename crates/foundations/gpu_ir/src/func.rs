// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One kind of function, the two ways a program uses one, and the two ways one
//! is given.
//!
//! A [`DeviceFunction`] is a name and what it takes, hands back and does. There
//! is no second function type: a [`Launch`] is a *use* of one at a block
//! geometry, and [`crate::OpKind::Call`] is a use of one inside another body.
//! Launch bounds and the register cap belong to the launch, so one function
//! serves every geometry a program wants to try it at.
//!
//! What the two uses ask of a function differs, and the verifier is where that
//! is said: a launched function is a kernel — no results, and every parameter a
//! pointer or a scalar the host can pass — while a called one may produce
//! values and declares no shared memory of its own, because shared memory is
//! the launched kernel's.
//!
//! A function is given one of two ways, and that is the whole difference
//! between a body this language can read and one it cannot. An
//! [`Implementation::Body`] is an arena: what it takes, hands back and does is
//! *read off* it, and a call to one is run where it stands. An
//! [`Implementation::Declared`] is a [`Signature`]: the same facts *stated*,
//! for a function whose definition is in another language, so a call to one is
//! retained and lowers to a PTX `call`.

use crate::arena::Function;
use crate::effect::{Effect, Effects, Sync};
use crate::{Param, SharedArr, ValTy};

/// What a function does to the memory behind one of its parameters.
///
/// Nothing is what a scalar parameter is touched by, and it is also what a
/// pointer parameter the body never dereferences is touched by; the two are the
/// same fact and are stated the same way.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Access {
    pub read: bool,
    pub written: bool,
    /// The widest contiguous access it makes through that parameter, in
    /// elements. Zero where it touches nothing.
    pub lanes: u8,
}

impl Access {
    pub fn touched(self) -> bool {
        self.read || self.written
    }

    /// What touching the operand at `ptr` this way is, as an operation's effect.
    pub fn effect(self, ptr: usize) -> Option<Effect> {
        Effect::touching(ptr, self.read, self.written)
    }
}

/// What a function takes, hands back and does, stated rather than read off a
/// body.
///
/// This is the whole contract of a function defined in another language. It is
/// trusted, because there is no arena to prove it against; what it must be is
/// internally consistent and renderable, which [`crate::verify()`] decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub parameters: Vec<Param>,
    /// What it hands back. PTX gives a `.func` one return parameter, so this
    /// holds at most one value.
    pub results: Vec<ValTy>,
    /// What it does to the memory behind each parameter, in declaration order.
    pub accesses: Vec<Access>,
    /// The threads its body needs to arrive together, where it needs any.
    pub sync: Option<Sync>,
}

/// Where a function's implementation is.
#[derive(Clone, Debug)]
pub enum Implementation {
    /// Here: the arena that is its body.
    Body(Function),
    /// Elsewhere: what it takes, hands back and does, because there is no body
    /// to read those off.
    Declared(Signature),
}

/// One device function: a name and where its implementation is.
#[derive(Clone, Debug)]
pub struct DeviceFunction {
    pub name: String,
    pub implementation: Implementation,
}

impl DeviceFunction {
    /// A function this language defines, given by the arena that is its body.
    pub fn new(name: impl Into<String>, body: Function) -> DeviceFunction {
        DeviceFunction {
            name: name.into(),
            implementation: Implementation::Body(body),
        }
    }

    /// A function defined elsewhere, given by the contract a caller may rely on.
    pub fn declared(name: impl Into<String>, signature: Signature) -> DeviceFunction {
        DeviceFunction {
            name: name.into(),
            implementation: Implementation::Declared(signature),
        }
    }

    /// The arena that is its body, for a function this language defines.
    pub fn body(&self) -> Option<&Function> {
        match &self.implementation {
            Implementation::Body(body) => Some(body),
            Implementation::Declared(_) => None,
        }
    }

    pub fn body_mut(&mut self) -> Option<&mut Function> {
        match &mut self.implementation {
            Implementation::Body(body) => Some(body),
            Implementation::Declared(_) => None,
        }
    }

    pub fn into_body(self) -> Option<Function> {
        match self.implementation {
            Implementation::Body(body) => Some(body),
            Implementation::Declared(_) => None,
        }
    }

    /// What it takes, hands back and does, for a function defined elsewhere.
    pub fn signature(&self) -> Option<&Signature> {
        match &self.implementation {
            Implementation::Body(_) => None,
            Implementation::Declared(signature) => Some(signature),
        }
    }

    /// What this function takes, in declaration order.
    pub fn parameters(&self) -> Vec<Param> {
        match &self.implementation {
            Implementation::Body(body) => body.parameters(),
            Implementation::Declared(signature) => signature.parameters.clone(),
        }
    }

    /// The type of each thing it takes, in declaration order — the types an
    /// argument list is checked against.
    ///
    /// A body declares its shared arrays in the same list as its parameters, so
    /// this is what one is entered with rather than what a caller may pass; that
    /// a callee declares no shared memory is a rule of its own.
    pub fn parameter_tys(&self) -> Vec<ValTy> {
        match &self.implementation {
            Implementation::Body(body) => (0..body.declarations().len())
                .map(|index| body.ty(body.declared(index)))
                .collect(),
            Implementation::Declared(signature) => signature
                .parameters
                .iter()
                .map(|param| param.ty.val_ty())
                .collect(),
        }
    }

    /// The shared arrays it declares. A declaration has none: shared memory is
    /// the launched kernel's, and nothing outside a body can name it.
    pub fn shared_arrays(&self) -> Vec<SharedArr> {
        match &self.implementation {
            Implementation::Body(body) => body.shared_arrays(),
            Implementation::Declared(_) => Vec::new(),
        }
    }

    /// What it produces: the types its entry block hands back, or the ones its
    /// declaration states.
    pub fn result_tys(&self) -> Vec<ValTy> {
        match &self.implementation {
            Implementation::Body(body) => body
                .returned()
                .iter()
                .map(|&value| body.ty(value))
                .collect(),
            Implementation::Declared(signature) => signature.results.clone(),
        }
    }

    /// The threads running it needs to arrive together, where it needs any.
    pub fn synchronisation(&self) -> Option<Sync> {
        match &self.implementation {
            Implementation::Body(body) => body.block_synchronisation(body.entry()),
            Implementation::Declared(signature) => signature.sync,
        }
    }

    /// What running this function does, as the effects of a call to it.
    ///
    /// A declaration states what it touches, because nothing else can say. A
    /// body's accesses are not restated here: a call to one is run where it
    /// stands, so what it does to memory is what its own steps do, read where
    /// they land. What both kinds owe a caller either way is the threads they
    /// make it wait for, which no inlining decides.
    pub fn effects(&self) -> Effects {
        let mut held = Effects::none();
        if let Implementation::Declared(signature) = &self.implementation {
            for (slot, access) in signature.accesses.iter().enumerate() {
                if let Some(effect) = access.effect(slot) {
                    held = held.with(effect);
                }
            }
        }
        if let Some(sync) = self.synchronisation() {
            held = held.with(Effect::Barrier(sync));
        }
        held.with(Effect::Control)
    }
}

/// A launch of one device function: the block it promises to run in, and the
/// per-thread register cap it is compiled under.
#[derive(Clone, Debug)]
pub struct Launch {
    pub function: DeviceFunction,
    /// `.maxntid` (launch bounds); the launch block size.
    pub max_threads: u32,
    /// `.maxnreg`: a per-thread register cap, `None` for the assembler's own
    /// allocation. Capped variants trade registers for occupancy and are
    /// autotuned like any other candidate.
    pub max_registers: Option<u32>,
}

impl Launch {
    pub fn new(function: DeviceFunction, max_threads: u32, max_registers: Option<u32>) -> Launch {
        Launch {
            function,
            max_threads,
            max_registers,
        }
    }

    /// The arena it launches. A launch runs a body — [`crate::verify_launch`]
    /// refuses a declaration, which is callable and not launchable.
    pub fn body(&self) -> &Function {
        self.function.body().unwrap_or_else(|| {
            panic!(
                "{} is declared, and a launch runs a body",
                self.function.name
            )
        })
    }

    pub fn into_body(self) -> Function {
        let name = self.function.name.clone();
        self.function
            .into_body()
            .unwrap_or_else(|| panic!("{name} is declared, and a launch runs a body"))
    }
}
