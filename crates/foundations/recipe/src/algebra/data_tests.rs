// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::algebra::builder::RecipeBuilder;
use crate::algebra::computation::Addresses;
use crate::algebra::executable::{CompileRequests, Executable, KernelLaunch};
use crate::arg::Arg;
use crate::buffer::BufRef;
use crate::code::{
    CompileRequest, CompileRequestId, KernelModule, PtxCompileOptions, PtxTarget, ResourceBounds,
    SourceImage, SourceKind,
};
use crate::effects::{Access, Effects};
use crate::signature::RecipeSignature;

fn request(ptx: &str) -> CompileRequest {
    CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 7],
                minimum_sm: 90,
            },
            blob: crate::Blob::from_vec(ptx.as_bytes().to_vec()),
        }],
        PtxCompileOptions::default(),
    )
}

fn launch(entry: &str) -> Work<Executable> {
    Work::Call {
        call: KernelLaunch {
            module: 0,
            entry: entry.to_string(),
            args: vec![Arg::input(0), Arg::output(0)],
        },
        geometry: vec![Geometry::single()],
    }
}

fn body(entry: &str) -> RecipeBody<Executable> {
    RecipeBody::new(
        1,
        1,
        vec![KernelModule::from_request(
            CompileRequestId(0),
            Vec::new(),
            ResourceBounds::default(),
        )],
        Vec::new(),
        vec![Node::new(launch(entry), [])],
    )
}

fn recipe(entry: &str) -> Recipe<Executable> {
    Recipe::stating_edges(
        RecipeSignature::new(vec![64], vec![32]),
        CompileRequests::new(vec![request("kernel")]),
        body(entry),
    )
    .unwrap()
}

/// One recipe is one value: two built the same way are equal, and each of the
/// three parts is part of that equality.
#[test]
fn a_recipe_is_the_three_parts_it_states() {
    assert_eq!(recipe("loom_a"), recipe("loom_a"));
    assert_ne!(recipe("loom_a"), recipe("loom_b"));

    let held = recipe("loom_a");
    assert_ne!(
        held,
        Recipe::stating_edges(
            RecipeSignature::new(vec![65], vec![32]),
            CompileRequests::new(vec![request("kernel")]),
            body("loom_a"),
        )
        .unwrap()
    );
    assert_ne!(
        held,
        Recipe::stating_edges(
            RecipeSignature::new(vec![64], vec![32]),
            CompileRequests::new(vec![request("other")]),
            body("loom_a"),
        )
        .unwrap()
    );
}

/// Which recipe a value is, is not part of what it says: two equal recipes are
/// equal, and each still hands out paths only it will answer for.
#[test]
fn identity_is_not_equality() {
    let one = recipe("loom_a");
    let other = recipe("loom_a");
    assert_eq!(one, other);
    assert_ne!(one.shape(), other.shape());
    assert_eq!(
        one.shape(),
        one.clone().shape(),
        "a clone is the same recipe"
    );
}

/// Every part comes out of decomposition, and the recipe it came from is gone.
#[test]
fn decomposition_hands_over_all_three_parts() {
    let (contract, context, body) = recipe("loom_a").into_parts();
    assert_eq!(contract, RecipeSignature::new(vec![64], vec![32]));
    assert_eq!(context.len(), 1);
    assert_eq!(body.nodes().len(), 1);
}

/// Dropping the ports is what `into_fragment` says it is: root data and work
/// survive so the handles into them still reach.
#[test]
fn a_fragment_keeps_the_root_data_its_bodies_name() {
    let fragment = recipe("loom_a").into_fragment();
    assert_eq!(fragment.context().len(), 1);
    assert_eq!(fragment.body(), &body("loom_a"));
}

/// A fragment that still carries root data has no body of its own to give:
/// dropping the table would strand the handles the modules name.
#[test]
fn a_fragment_carrying_root_data_refuses_to_give_up_its_body() {
    let fragment = recipe("loom_a").into_fragment();
    assert_eq!(
        fragment.into_body(),
        Err(RecipeError::FragmentLeavesRootContext)
    );

    let free = RecipeFragment::<Executable>::from_body(RecipeBody::default());
    assert_eq!(free.into_body(), Ok(RecipeBody::default()));
}

/// Checking a body proves what it can: a call naming a module the body it sits
/// in does not have is refused wherever that body arrives from.
#[test]
fn a_body_is_checked_against_the_context_beside_it() {
    let dangling = RecipeBody::new(
        1,
        1,
        Vec::new(),
        Vec::new(),
        vec![Node::new(launch("loom_a"), [])],
    );
    assert_eq!(
        Recipe::stating_edges(
            RecipeSignature::new(vec![64], vec![32]),
            CompileRequests::new(vec![request("kernel")]),
            dangling,
        ),
        Err(RecipeError::Call {
            node: 0,
            refusal: crate::algebra::ExecutableRefusal::ModuleOutOfRange { module: 0 }
        })
    );
}

/// A fragment taken from a recipe carries no proof, because a recipe holds
/// none: the edges are what survived, and nothing recovers the statement they
/// came from. Silence is not a claim that it touches nothing.
#[test]
fn a_recipe_gives_up_no_proof_with_its_work() {
    let fragment = recipe("loom_a").into_fragment();
    assert!(!fragment.is_proved());
    assert_eq!(fragment.boundary(), None);

    let built = RecipeBuilder::<Executable>::new(0, 0).finish_fragment();
    assert!(built.is_proved(), "having built it is the proof");
    assert_eq!(
        built.boundary(),
        Some(&Effects::none()),
        "work that touches nothing proved that it touches nothing"
    );
}

/// The contract's default is an empty context, not an unstated one: a recipe
/// built from a body alone says its bodies need no root data.
#[test]
fn a_recipe_from_a_body_alone_states_an_empty_context() {
    let held = Recipe::<Executable>::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::default(),
        RecipeBody::<Executable>::default(),
    )
    .unwrap();
    assert!(held.context().is_empty());
    assert_eq!(held.context(), &CompileRequests::default());
}

/// A call is opaque to the core except for the addresses it is handed. What it
/// does through them is the placement's, and is not read off the arguments.
#[test]
fn a_call_states_the_addresses_it_is_handed() {
    let call = KernelLaunch {
        module: 0,
        entry: "loom_a".to_string(),
        args: vec![Arg::input(1), Arg::scalar_u32(7), Arg::workspace(2)],
    };
    let mut seen = Vec::new();
    call.addresses(&mut |address| seen.push(address));
    assert_eq!(
        seen,
        vec![BufRef::Input(1).at(0), BufRef::Workspace(2).at(0)]
    );
}

/// Rebinding replaces the buffers a call names and leaves its scalar bytes
/// where they are.
#[test]
fn rebinding_moves_only_the_addresses() {
    let mut call = KernelLaunch {
        module: 0,
        entry: "loom_a".to_string(),
        args: vec![Arg::input(0), Arg::scalar_i32(-3)],
    };
    call.rebind(&mut |address| BufRef::Workspace(9).at(address.byte_offset + 4));
    assert_eq!(
        call.args,
        vec![
            Arg::pointer(BufRef::Workspace(9).at(4)),
            Arg::scalar_i32(-3)
        ]
    );
}

/// Effects are a set: stating one twice states it once, and stating a read and
/// a write states both.
#[test]
fn effects_are_what_was_said_however_it_was_said() {
    let held = Effects::of([
        (BufRef::Output(0), Access::Read),
        (BufRef::Input(0), Access::Read),
        (BufRef::Output(0), Access::Write),
        (BufRef::Input(0), Access::Read),
    ]);
    assert_eq!(held.access(BufRef::Output(0)), Some(Access::ReadWrite));
    assert_eq!(held.access(BufRef::Input(0)), Some(Access::Read));
    assert_eq!(held.access(BufRef::Workspace(0)), None);
    assert_eq!(
        held,
        Effects::of([
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::ReadWrite),
        ]),
        "how they were discovered is not part of them"
    );
    assert_eq!(held.writes().collect::<Vec<_>>(), vec![BufRef::Output(0)]);
}

/// Memory work states its effects by being what it is; there is nothing to
/// disagree with.
#[test]
fn memory_work_states_its_own_effects() {
    let memcpy = Work::<Executable>::Memcpy {
        source: BufRef::Input(0).at(0),
        destination: crate::buffer::WritableBuf::Output(0).at(0),
        nbytes: 8,
    };
    assert_eq!(
        memcpy.memory_effects(),
        Some(Effects::of([
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::Write)
        ]))
    );
    assert_eq!(
        launch("loom_a").memory_effects(),
        None,
        "a computation's effects are the placement's statement, not the work"
    );
}
