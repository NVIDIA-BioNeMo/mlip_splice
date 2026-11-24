// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What ordering and editing prove.
//!
//! Effects are what order work, so the edges a builder derives are the subject
//! here: what a placement acquires, what a landed region does not, and what a
//! substitution has to answer for before it may replace a call. The proof a
//! fragment carries runs through all of it — only work someone vouched for can
//! be held to anything.

use std::collections::BTreeSet;

use super::structure_fixture::*;
use super::*;
use crate::buffer::{BufKind, BufRef, WritableBuf};
use crate::handles::{Buf, NodeId, Writable};
use crate::signature::RecipeSignature;

/// One allocation covers every way the choices can go: a body's own buffers,
/// plus the worst candidate of each choice it holds, because it runs them all.
#[test]
fn the_workspace_bound_dominates_every_choice() {
    let candidate = |nbytes: usize| {
        let mut builder = RecipeBuilder::<Stores>::new(1, 1);
        builder.scratch(nbytes);
        builder.finish_fragment()
    };
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder.scratch(100);
    builder.scratch(40);
    builder
        .choice(vec![candidate(10), candidate(300)])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    builder
        .choice(vec![candidate(500), candidate(20)])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();

    // parent (100 -> 256, then 40 -> +256 = 512) + worst of {10,300}=300->512
    // + worst of {500,20}=500->512.
    assert_eq!(workspace_upper_bound(&recipe), 512 + 512 + 512);
}

/// Effects order the work; what a node is told to wait for is added to what it
/// must wait for, never instead of it.
#[test]
fn effects_and_prerequisites_both_order() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    let scratch = builder.scratch(64);
    let first = builder.memset(Writable::Workspace(scratch).at(0), 0, 64);
    let aside = builder.memset(Writable::Output(0).at(0), 0, 64);
    let reader = builder
        .call(
            Store {
                at: BufRef::Workspace(scratch.index()).at(0),
            },
            vec![Geometry::single()],
        )
        .read_writes([Buf::Workspace(scratch)])
        .after(&[aside])
        .add()
        .unwrap();
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();

    assert_eq!(
        recipe.body().nodes()[reader.index()].deps(),
        &BTreeSet::from([first.index(), aside.index()])
    );
    assert_eq!(recipe.waits_for(reader.index()), Some(vec![0, 1]));
    assert_eq!(recipe.concurrent(), vec![(0, 1)]);
}

/// A landing is ordered by what the caller says it follows, and by nothing
/// else. The region keeps the edges it already had, its sources follow the
/// prerequisites given here, and it does not join the history that orders later
/// placements — a caller that wants something to follow the region names its
/// sinks. That is what makes a recipe assembled from regions the same recipe
/// those regions always produced.
#[test]
fn a_landing_is_ordered_by_what_the_caller_states() {
    let mut source = RecipeBuilder::<Stores>::new(1, 1);
    source
        .call(
            Store {
                at: BufRef::Output(0).at(0),
            },
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    source
        .call(
            Store {
                at: BufRef::Output(0).at(8),
            },
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    let fragment = source.finish_fragment();

    let mut host = RecipeBuilder::<Stores>::new(1, 1);
    let earlier = host.memset(Writable::Output(0).at(0), 0, 8);
    let sinks = host
        .splice(
            fragment,
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
            &[earlier],
        )
        .unwrap();
    assert_eq!(sinks, vec![NodeId::from_index(2)]);

    let after = host.memcpy(Buf::Output(0).at(0), Writable::Output(0).at(16), 8);
    let recipe = host.finish(ports(64, 64)).unwrap().into_recipe();

    assert_eq!(
        recipe.body().nodes()[1].deps(),
        &BTreeSet::from([earlier.index()]),
        "a dep-less incoming node follows what the caller said the region follows"
    );
    assert_eq!(
        recipe.body().nodes()[2].deps(),
        &BTreeSet::from([1]),
        "and the rest of the region keeps the order it arrived with"
    );
    assert!(
        !recipe.body().nodes()[after.index()].deps().contains(&2),
        "the region did not join the history: ordering after it is named, not derived"
    );
}

/// The other half of the same rule: a landing does not acquire edges from what
/// its work happens to touch. This region reads a buffer the host wrote and
/// still follows nothing, because the caller named nothing — which is the only
/// answer that keeps a landed region byte-identical to the one that was built.
#[test]
fn a_landing_acquires_no_edges_of_its_own() {
    let mut source = RecipeBuilder::<Stores>::new(1, 1);
    let scratch = source.scratch(64);
    source.memcpy(Buf::Output(0).at(0), Writable::Workspace(scratch).at(0), 8);
    let fragment = source.finish_fragment();
    assert!(
        fragment.body().nodes()[0].deps().is_empty(),
        "it is a source node in its own body"
    );

    let mut host = RecipeBuilder::<Stores>::new(1, 1);
    host.memset(Writable::Output(0).at(0), 0, 8);
    host.splice(fragment, &[BufRef::Input(0)], &[BufRef::Output(0)], &[])
        .unwrap();
    let recipe = host.finish(ports(64, 64)).unwrap().into_recipe();

    assert!(
        recipe.body().nodes()[1].deps().is_empty(),
        "nothing was named, so nothing is followed"
    );

    // Naming it is how a caller orders it, and that is all it takes.
    let mut ordered = RecipeBuilder::<Stores>::new(1, 1);
    let wrote = ordered.memset(Writable::Output(0).at(0), 0, 8);
    let reader = {
        let mut source = RecipeBuilder::<Stores>::new(1, 1);
        let scratch = source.scratch(64);
        source.memcpy(Buf::Output(0).at(0), Writable::Workspace(scratch).at(0), 8);
        source.finish_fragment()
    };
    ordered
        .splice(reader, &[BufRef::Input(0)], &[BufRef::Output(0)], &[wrote])
        .unwrap();
    let recipe = ordered.finish(ports(64, 64)).unwrap().into_recipe();
    assert_eq!(recipe.body().nodes()[1].deps(), &BTreeSet::from([0]));
}

/// A slot a choice only carries is a slot nothing under it may touch — reading
/// it is as much a claim as writing it, and neither was made.
#[test]
fn a_candidate_may_not_read_what_the_choice_only_carries() {
    let mut source = RecipeBuilder::<Stores>::new(1, 1);
    let scratch = source.scratch(64);
    source.memcpy(Buf::Output(0).at(0), Writable::Workspace(scratch).at(0), 8);
    let reader = source.finish_fragment();

    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    assert_eq!(
        builder
            .choice(vec![reader])
            .inputs([Buf::Input(0)])
            .does([Buf::Input(0)], Access::Read)
            .outputs([Buf::Output(0)])
            .add(),
        Err(RecipeError::UndeclaredEffect {
            node: 0,
            kind: BufKind::Output,
            index: 0
        })
    );
}

/// The edges around a call were derived from what that call does. Work
/// substituted into its place inherits those edges, so it is refused unless
/// they answer for it too.
#[test]
fn substituted_work_may_not_reach_past_the_call_it_replaces() {
    let host = storing(BufRef::Output(0).at(0)).unwrap();
    let path = host.call_paths().remove(0);

    let mut wider = RecipeBuilder::<Stores>::new(1, 1);
    let scratch = wider.scratch(64);
    wider.memcpy(Buf::Input(0).at(0), Writable::Workspace(scratch).at(0), 8);
    wider
        .call(
            Store {
                at: BufRef::Output(0).at(0),
            },
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();

    assert_eq!(
        host.substitute(
            &path,
            wider.finish_fragment(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)]
        ),
        Err(RecipeError::UndeclaredEffect {
            node: 0,
            kind: BufKind::Input,
            index: 0
        }),
        "the replaced call never read the input, so nothing ordered a reader of it"
    );

    // The same work, minus what the call it replaces does not answer for.
    let mut narrow = RecipeBuilder::<Stores>::new(1, 1);
    narrow
        .call(
            Store {
                at: BufRef::Output(0).at(8),
            },
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    assert!(
        host.substitute(
            &path,
            narrow.finish_fragment(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)]
        )
        .is_ok()
    );
}

/// A fragment nobody proved cannot be a candidate, and cannot be substituted
/// into a placement: both hold it to what it does, and its silence is not a
/// claim that it does nothing. Landing is different — that asks for no proof,
/// because the caller states the ordering itself.
#[test]
fn an_unproved_fragment_cannot_be_held_to_anything() {
    let unproved = RecipeFragment::<Stores>::from_body(RecipeBody::new(
        1,
        1,
        (),
        Vec::new(),
        vec![Node::new(storing_at(BufRef::Output(0).at(0)), [])],
    ));
    assert!(!unproved.is_proved());

    let mut host = RecipeBuilder::<Stores>::new(1, 1);
    assert!(
        host.splice(
            unproved.clone(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
            &[]
        )
        .is_ok(),
        "landing asks for no proof: the caller states the ordering"
    );
    assert_eq!(
        host.choice(vec![unproved.clone()])
            .inputs([Buf::Input(0)])
            .does([Buf::Input(0)], Access::Read)
            .outputs([Buf::Output(0)])
            .does([Buf::Output(0)], Access::Write)
            .add(),
        Err(RecipeError::UnprovedFragment)
    );

    let recipe = storing(BufRef::Output(0).at(0)).unwrap();
    let path = recipe.call_paths().remove(0);
    assert_eq!(
        recipe.substitute(&path, unproved, &[BufRef::Input(0)], &[BufRef::Output(0)]),
        Err(RecipeError::UnprovedFragment)
    );
}

/// A recipe assembled from parts is not a proved recipe, and there is no public
/// way to make one look like one.
///
/// These two memsets overwrite each other and state no edge between them. The
/// body is well-formed — every access is in range, every dep names an earlier
/// node — so checking it says yes, which is exactly why checking is not the
/// boundary. Building the same two memsets derives the edge that the raw body
/// left out.
#[test]
fn raw_work_with_a_missing_edge_is_not_a_proved_recipe() {
    let raw = Recipe::<Stores>::stating_edges(
        ports(64, 64),
        (),
        RecipeBody::new(
            1,
            1,
            (),
            Vec::new(),
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 1,
                        nbytes: 8,
                    },
                    [],
                ),
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 2,
                        nbytes: 8,
                    },
                    [],
                ),
            ],
        ),
    )
    .expect("well-formed, and wrong");
    assert!(
        raw.body().nodes()[1].deps().is_empty(),
        "nothing orders the second store after the first"
    );
    assert_eq!(
        raw.concurrent(),
        vec![(0, 1)],
        "so the recipe says they may run at once"
    );

    let mut built = RecipeBuilder::<Stores>::new(1, 1);
    built.memset(Writable::Output(0).at(0), 1, 8);
    built.memset(Writable::Output(0).at(0), 2, 8);
    let built = built.finish(ports(64, 64)).unwrap().into_recipe();
    assert_eq!(
        built.body().nodes()[1].deps(),
        &BTreeSet::from([0]),
        "building the same work derives the edge the raw body left out"
    );
    assert!(built.concurrent().is_empty());
}

/// What a choice does is what its candidates do, slot by slot. A slot none of
/// them reaches is carried, so nothing is ordered against the buffer behind it;
/// one they reach differently is answered for by everything any of them does.
#[test]
fn a_choice_may_answer_for_exactly_what_its_candidates_do() {
    // Two candidates over three slots: one input neither reads, one output both
    // write, and one output only the second reaches.
    let untouched = RecipeBuilder::<Stores>::new(1, 2).finish_fragment();
    let reaching = {
        let mut builder = RecipeBuilder::<Stores>::new(1, 2);
        builder
            .call(
                Store {
                    at: BufRef::Output(1).at(0),
                },
                vec![Geometry::single()],
            )
            .writes([Buf::Output(1)])
            .add()
            .unwrap();
        builder.finish_fragment()
    };

    let mut builder = RecipeBuilder::<Stores>::new(1, 2);
    let scratch = builder.scratch(64);
    builder.memset(Writable::Workspace(scratch).at(0), 0, 64);
    builder
        .choice(vec![untouched, reaching])
        .inputs([Buf::Workspace(scratch)])
        .outputs([Buf::Output(0), Buf::Output(1)])
        .does_what_its_candidates_do()
        .add()
        .unwrap();
    let recipe = builder
        .finish(RecipeSignature::new(vec![64], vec![64, 64]))
        .unwrap();

    assert!(
        recipe.recipe().body().nodes()[1].deps().is_empty(),
        "no candidate reads the buffer the input slot selects, so the clearing \
         of it orders nothing"
    );
    assert_eq!(
        recipe.boundary().access(BufRef::Output(0)),
        None,
        "an output slot nothing reaches is carried"
    );
    assert_eq!(
        recipe.boundary().access(BufRef::Output(1)),
        Some(Access::Write),
        "and one a candidate stores through is written"
    );
}
