// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declarations that do not fit in an address.
//!
//! A body is whatever a producer assembled: it may declare a workspace no
//! allocation could back, and the arithmetic that sums one is fallible at every
//! step — rounding each buffer up to the alignment, adding them, and adding the
//! worst candidate of each choice. Nothing here may answer such a body with a
//! number, and nothing here may answer it by panicking. Asking is how a recipe
//! comes to exist at all, so the answer above is the only one a caller holding a
//! recipe can get.

use super::*;
use crate::algebra::structure_fixture::{Store, Stores, named, ports, storing_at};
use crate::buffer::{BufRef, WorkspaceBuf};
use crate::layout::WORKSPACE_ALIGN;

fn scratch(nbytes: usize) -> WorkspaceBuf {
    WorkspaceBuf { nbytes, init: None }
}

/// A body declaring `workspace`, whose one node stores into its own scratch.
fn declaring(workspace: Vec<WorkspaceBuf>) -> RecipeBody<Stores> {
    let node = Node::new(storing_at(BufRef::Workspace(0).at(0)), []);
    RecipeBody::new(1, 1, (), workspace, vec![node])
}

/// A body over `candidates`, bound to nothing, so the choice contributes only
/// what its worst candidate declares.
fn choosing(
    workspace: Vec<WorkspaceBuf>,
    candidates: Vec<RecipeBody<Stores>>,
) -> RecipeBody<Stores> {
    let node = Node::new(
        Work::Choice {
            candidates,
            input_binding: Vec::new(),
            output_binding: Vec::new(),
        },
        [],
    );
    RecipeBody::new(1, 1, (), workspace, vec![node])
}

/// A candidate declaring `workspace` and nothing else. It binds no slots, so it
/// answers for no ports.
fn candidate(workspace: Vec<WorkspaceBuf>) -> RecipeBody<Stores> {
    RecipeBody::new(0, 0, (), workspace, Vec::new())
}

/// Every way the sum can leave the address space is a `None`, at whatever step
/// it happens: rounding one buffer up past the end, adding two that fit
/// separately, or rounding the total up past the end after they are added.
#[test]
fn a_workspace_that_does_not_fit_in_an_address_has_no_bound() {
    assert_eq!(
        checked_body_workspace_upper_bound(&declaring(vec![scratch(usize::MAX)])),
        None,
        "one buffer spanning the whole address space leaves no room to align \
         the end of it"
    );
    assert_eq!(
        checked_body_workspace_upper_bound(&declaring(vec![
            scratch(usize::MAX / 2),
            scratch(usize::MAX / 2),
        ])),
        None,
        "each fits; together they do not"
    );
    assert_eq!(
        checked_body_workspace_upper_bound(&declaring(vec![scratch(
            usize::MAX - WORKSPACE_ALIGN / 2
        )])),
        None,
        "it fits, and the alignment of its end does not"
    );
}

/// A choice adds what its worst candidate declares, so a candidate with no
/// bound leaves the body with none — however deep it sits, and however
/// comfortably every level above it fits on its own.
#[test]
fn a_candidate_that_does_not_fit_leaves_the_body_it_is_offered_in_with_no_bound() {
    let body = choosing(
        vec![scratch(64)],
        vec![candidate(vec![scratch(usize::MAX)])],
    );
    assert_eq!(checked_body_workspace_upper_bound(&body), None);

    let nested = choosing(
        vec![scratch(64)],
        vec![candidate(vec![scratch(64)]), choosing(vec![], vec![body])],
    );
    assert_eq!(checked_body_workspace_upper_bound(&nested), None);
}

/// And the addition itself is checked: two halves that each have a bound have
/// none together, because a body runs its own work as well as whatever it
/// chooses.
#[test]
fn a_body_and_its_choice_are_added_with_the_end_of_the_address_space_in_mind() {
    // Half the address space, aligned, so each side lays out exactly and only
    // their sum leaves.
    let half = usize::MAX / 2 + 1;
    let candidate = candidate(vec![scratch(half)]);
    assert!(checked_body_workspace_upper_bound(&candidate).is_some());

    let own = declaring(vec![scratch(half)]);
    assert!(checked_body_workspace_upper_bound(&own).is_some());

    assert_eq!(
        checked_body_workspace_upper_bound(&choosing(vec![scratch(half)], vec![candidate])),
        None
    );
}

/// So no such body is ever a recipe: the one door that turns work into a recipe
/// asks this question, and a `None` is where it stops.
///
/// That is what makes the bound of a recipe a number rather than a maybe —
/// there is no body left that could be behind one and no answer to give for it.
#[test]
fn work_whose_workspace_has_no_layout_never_becomes_a_recipe() {
    assert_eq!(
        Recipe::<Stores>::stating_edges(ports(64, 64), (), declaring(vec![scratch(usize::MAX)]))
            .map(|_| ()),
        Err(RecipeError::WorkspaceLayoutOverflow)
    );

    let recipe = Recipe::<Stores>::stating_edges(ports(64, 64), (), declaring(vec![scratch(300)]))
        .expect("300 bytes of scratch fit in an address");
    assert_eq!(workspace_upper_bound(&recipe), 2 * WORKSPACE_ALIGN);
    assert_eq!(
        checked_workspace_upper_bound(&recipe),
        Some(workspace_upper_bound(&recipe)),
        "the proved form answers what the checked one does, wherever both apply"
    );
}

/// Work that leaves takes the memory carved for it: releasing those buffers
/// renumbers what named the ones that stayed, and touches nothing below.
#[test]
fn releasing_a_bodys_workspace_renumbers_what_is_left() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    let first = builder.scratch(64);
    let leaving = builder.scratch(1024);
    let last = builder.scratch(64);
    for buf in [first, last] {
        builder
            .call(
                Store {
                    at: BufRef::Workspace(buf.index()).at(0),
                },
                vec![Geometry::single()],
            )
            .writes([named(BufRef::Workspace(buf.index()))])
            .add()
            .unwrap();
    }
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();
    assert_eq!(workspace_upper_bound(&recipe), WORKSPACE_ALIGN * 6);

    let released = recipe
        .releasing_workspace(
            &recipe.root_path(),
            &[leaving.index()].into_iter().collect(),
        )
        .expect("nothing reaches the buffer that left");
    assert_eq!(released.body().workspace().len(), 2);
    assert_eq!(
        workspace_upper_bound(&released),
        WORKSPACE_ALIGN * 2,
        "what stayed is all there is to allocate"
    );
    let stored: Vec<BufRef> = released
        .body()
        .nodes()
        .iter()
        .map(|node| match node.work() {
            Work::Call { call, .. } => call.at.buf,
            _ => panic!("both nodes store"),
        })
        .collect();
    assert_eq!(
        stored,
        [BufRef::Workspace(0), BufRef::Workspace(1)],
        "the buffer that was last is now second, and the work says so"
    );
    assert_eq!(
        recipe.body().workspace().len(),
        3,
        "and the recipe it was released from is the value it was"
    );
}

/// A buffer something still reaches is not released: the memory would be gone
/// and the work would name whatever moved into its place.
#[test]
fn releasing_workspace_something_still_reaches_is_refused() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    let held = builder.scratch(64);
    builder
        .call(
            Store {
                at: BufRef::Workspace(held.index()).at(0),
            },
            vec![Geometry::single()],
        )
        .writes([named(BufRef::Workspace(held.index()))])
        .add()
        .unwrap();
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();
    assert_eq!(
        recipe.releasing_workspace(&recipe.root_path(), &[0].into_iter().collect()),
        Err(RecipeError::WorkspaceStillReached { node: 0, index: 0 })
    );
}
