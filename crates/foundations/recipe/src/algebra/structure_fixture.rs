// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A second computation family, and the values the structural tests are written
//! over.
//!
//! It shares nothing with the executable family but the structural contracts,
//! so anything both are held to is a fact about the core rather than about
//! kernels. Its one kind of call stores through the single address it is
//! handed, which is the least a family can be and still have effects worth
//! ordering.

use super::*;
use crate::algebra::computation::DeclaredEffects;
use crate::arg::Arg;
use crate::buffer::{Address, BufRef};
use crate::handles::Buf;
use crate::signature::RecipeSignature;

/// A family whose computations store, and hold nothing else at all: no root
/// data, no body data, no handles to rebase. What it shares with the executable
/// family is exactly the structural contracts, and no more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stores;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Store {
    pub(super) at: Address<BufRef>,
}

impl Addresses for Store {
    fn addresses(&self, visit: &mut dyn FnMut(Address<BufRef>)) {
        visit(self.at);
    }

    fn rebind(&mut self, bind: &mut dyn FnMut(Address<BufRef>) -> Address<BufRef>) {
        self.at = bind(self.at);
    }
}

/// A store says what it does by where it stores, so this family can be edited
/// by effect: it keeps its calls' effects in the call.
impl DeclaredEffects for Stores {
    fn call_effects(call: &Store) -> Effects {
        Effects::of([(call.at.buf, Access::Write)])
    }
}

impl Computation for Stores {
    type Authorization = ();

    type Contract = RecipeSignature;
    type RootContext = ();
    type BodyContext = ();
    type Call = Store;
    type RootRemap = ();
    type BodyRemap = ();
    type Refusal = String;

    fn merge_root_context(_into: &mut (), _incoming: ()) {}
    fn merge_body_context(_into: &mut (), _incoming: ()) {}
    fn rebase_body_context(_context: &mut (), _remap: &()) {}
    fn rebase_call(_call: &mut Store, _remap: &()) {}

    fn validate_root_context(_context: &()) -> Result<(), String> {
        Ok(())
    }

    fn validate_body_context(_context: &(), _root: &()) -> Result<(), String> {
        Ok(())
    }

    fn validate_call(_call: &Store, _context: &()) -> Result<(), String> {
        Ok(())
    }

    fn launch_arity(_call: &Store) -> usize {
        1
    }

    fn validate_geometry(
        _call: &Store,
        _launch: usize,
        _geometry: &Geometry,
    ) -> Result<(), String> {
        Ok(())
    }
}

pub(super) fn ports(input: usize, output: usize) -> RecipeSignature {
    RecipeSignature::new(vec![input], vec![output])
}

pub(super) fn storing_at(at: Address<BufRef>) -> Work<Stores> {
    Work::Call {
        call: Store { at },
        geometry: vec![Geometry::single()],
    }
}

/// A recipe over `Stores` whose one node stores at `at`.
pub(super) fn storing(at: Address<BufRef>) -> Result<Recipe<Stores>, RecipeError<String>> {
    Recipe::stating_edges(
        ports(64, 64),
        (),
        RecipeBody::new(1, 1, (), Vec::new(), vec![Node::new(storing_at(at), [])]),
    )
}

/// The same, as a fragment built the way a producer builds one — so it carries
/// the proof of what it does, which is what a placement needs from it.
pub(super) fn storing_fragment(at: Address<BufRef>) -> RecipeFragment<Stores> {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder
        .call(Store { at }, vec![Geometry::single()])
        .writes([named(at.buf)])
        .add()
        .expect("a store declares the buffer it stores into");
    builder.finish_fragment()
}

pub(super) fn named(buf: BufRef) -> Buf {
    match buf {
        BufRef::Input(index) => Buf::Input(index),
        BufRef::Output(index) => Buf::Output(index),
        BufRef::Workspace(index) => Buf::Workspace(crate::handles::WorkspaceId::from_index(index)),
    }
}

pub(super) fn launch(module: usize, entry: &str, args: Vec<Arg>) -> KernelLaunch {
    KernelLaunch {
        module,
        entry: entry.to_string(),
        args,
    }
}
