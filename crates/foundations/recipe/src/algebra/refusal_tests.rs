// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a finished recipe refuses about itself.
//!
//! Extents against the ports that size them, a workspace that has to fit in an
//! address, and paths that name a place in one recipe and nowhere else. These
//! are the proofs a value carries rather than ones a producer made: they hold
//! for a recipe however it arrived.

use super::structure_fixture::*;
use super::*;
use crate::buffer::{Address, BufKind, BufRef, WorkspaceBuf, WritableBuf};
use crate::handles::{Buf, Writable};

#[test]
fn structure_refuses_what_it_owns() {
    let held = |nodes: Vec<Node<Stores>>, workspace: Vec<WorkspaceBuf>| {
        Recipe::stating_edges(
            ports(64, 64),
            (),
            RecipeBody::new(1, 1, (), workspace, nodes),
        )
    };
    let store = |at: Address<BufRef>, deps: Vec<usize>| Node::new(storing_at(at), deps);

    assert_eq!(
        held(vec![store(BufRef::Output(0).at(0), vec![0])], Vec::new()),
        Err(RecipeError::DepNotEarlier { node: 0, dep: 0 })
    );
    assert_eq!(
        held(vec![store(BufRef::Workspace(2).at(0), vec![])], Vec::new()),
        Err(RecipeError::BufOutOfRange {
            node: 0,
            kind: BufKind::Workspace,
            index: 2
        })
    );
    assert_eq!(
        held(
            vec![Node::new(
                Work::Memcpy {
                    source: BufRef::Input(0).at(usize::MAX),
                    destination: WritableBuf::Output(0).at(0),
                    nbytes: 8,
                },
                [],
            )],
            Vec::new()
        ),
        Err(RecipeError::AddressOverflow {
            node: 0,
            kind: BufKind::Input,
            index: 0,
            byte_offset: usize::MAX,
            nbytes: 8
        })
    );
    assert_eq!(
        held(
            Vec::new(),
            vec![WorkspaceBuf {
                nbytes: 4,
                init: Some(crate::Blob::from_vec(vec![0; 8]))
            }]
        ),
        Err(RecipeError::WorkspaceInitTooLarge {
            index: 0,
            init: 8,
            nbytes: 4
        })
    );
    assert_eq!(
        held(
            vec![Node::new(
                Work::Choice {
                    candidates: Vec::new(),
                    input_binding: Vec::new(),
                    output_binding: Vec::new(),
                },
                [],
            )],
            Vec::new()
        ),
        Err(RecipeError::EmptyChoice { node: 0 })
    );
}

/// The workspace a body declares has to have a layout. One that does not is
/// refused where the recipe is proved, so asking a recipe that exists for its
/// bound is asking for a number.
#[test]
fn a_workspace_with_no_layout_is_refused() {
    let vast = vec![
        WorkspaceBuf {
            nbytes: usize::MAX,
            init: None,
        },
        WorkspaceBuf {
            nbytes: 1,
            init: None,
        },
    ];
    assert_eq!(
        Recipe::<Stores>::stating_edges(
            ports(64, 64),
            (),
            RecipeBody::new(1, 1, (), vast, Vec::new())
        ),
        Err(RecipeError::WorkspaceLayoutOverflow)
    );

    // The same, one choice deep: the bound sums over what a body runs, and that
    // sum has to exist too.
    let huge = |nbytes: usize| {
        let mut builder = RecipeBuilder::<Stores>::new(1, 1);
        builder.scratch(nbytes);
        builder.finish_fragment()
    };
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder.scratch(usize::MAX / 2);
    builder
        .choice(vec![huge(usize::MAX / 2)])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    assert_eq!(
        builder.finish(ports(64, 64)),
        Err(RecipeError::WorkspaceLayoutOverflow)
    );
}

/// A path names one place in one recipe. One from anywhere else is refused,
/// however well its numbers happen to fit.
#[test]
fn a_path_from_another_recipe_is_refused() {
    let one = storing(BufRef::Output(0).at(0)).unwrap();
    let other = storing(BufRef::Output(0).at(0)).unwrap();
    assert_eq!(one, other, "the same recipe, built twice");

    let path = one.call_paths().remove(0);
    assert!(one.call_at(&path).is_ok());
    assert_eq!(other.call_at(&path), Err(RecipeError::PathNotOwned));
    assert_eq!(other.node_at(&path), Err(RecipeError::PathNotOwned));
    assert_eq!(other.body_at(path.body()), Err(RecipeError::PathNotOwned));
    assert_eq!(
        other.replace_call(
            &path,
            Store {
                at: BufRef::Output(0).at(0)
            }
        ),
        Err(RecipeError::PathNotOwned)
    );
    assert_eq!(
        other.substitute(
            &path,
            storing_fragment(BufRef::Output(0).at(0)),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)]
        ),
        Err(RecipeError::PathNotOwned)
    );
}

/// Moving work is a new shape, so a path taken before it names nothing
/// afterwards — rather than whatever moved into its place.
#[test]
fn a_path_from_before_an_edit_is_refused_after_it() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder
        .call(
            Store {
                at: BufRef::Output(0).at(0),
            },
            vec![Geometry::single()],
        )
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    builder.memset(Writable::Output(0).at(8), 0, 8);
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();

    let path = recipe.call_paths().remove(0);
    let grown = recipe
        .substitute(
            &path,
            storing_fragment(BufRef::Output(0).at(0)),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
        )
        .unwrap();
    assert_eq!(grown.node_at(&path), Err(RecipeError::PathNotOwned));

    // Replacing what one node computes leaves the shape alone, so paths keep
    // naming what they named.
    let same_shape = recipe
        .replace_call(
            &path,
            Store {
                at: BufRef::Output(0).at(16),
            },
        )
        .unwrap();
    assert!(same_shape.call_at(&path).is_ok());
}

/// A path that names nothing here is refused rather than resolved to a
/// neighbour.
#[test]
fn a_path_that_names_nothing_is_refused() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder
        .choice(vec![storing_fragment(BufRef::Output(0).at(0))])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    builder.memset(Writable::Output(0).at(0), 0, 8);
    let recipe = builder.finish(ports(64, 64)).unwrap().into_recipe();

    let inside = recipe.root_path().descend(0, 0);
    assert!(recipe.call_at(&inside.node(0)).is_ok());
    assert_eq!(recipe.call_paths(), vec![inside.node(0)]);
    assert_eq!(recipe.node_paths().len(), 3);

    assert_eq!(
        recipe.body_at(&recipe.root_path().descend(1, 0)),
        Err(RecipeError::NoSuchBody { at: 0 }),
        "node 1 is a memset, not a choice"
    );
    assert_eq!(
        recipe.body_at(&recipe.root_path().descend(0, 4)),
        Err(RecipeError::NoSuchBody { at: 0 })
    );
    assert_eq!(
        recipe.node_at(&recipe.root_path().node(9)),
        Err(RecipeError::NoSuchNode {
            path: recipe.root_path().node(9)
        })
    );
    assert_eq!(
        recipe.call_at(&recipe.root_path().node(1)),
        Err(RecipeError::NotACall {
            path: recipe.root_path().node(1)
        })
    );
    assert_eq!(
        recipe.substitute(
            &recipe.root_path().node(1),
            RecipeBuilder::<Stores>::new(1, 1).finish_fragment(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
        ),
        Err(RecipeError::NotACall {
            path: recipe.root_path().node(1)
        })
    );
}

/// A fragment is proved before anything indexes into it, so a body whose deps
/// name nodes it does not have is a refusal rather than a walk off the end.
#[test]
fn a_malformed_fragment_is_refused_before_it_is_reached_into() {
    let malformed = RecipeFragment::<Stores>::from_body(RecipeBody::new(
        1,
        1,
        (),
        Vec::new(),
        vec![Node::new(storing_at(BufRef::Output(0).at(0)), [7])],
    ));

    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    assert_eq!(
        builder.splice(
            malformed.clone(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
            &[]
        ),
        Err(RecipeError::DepNotEarlier { node: 0, dep: 7 })
    );

    let host = storing(BufRef::Output(0).at(0)).unwrap();
    let path = host.call_paths().remove(0);
    assert_eq!(
        host.substitute(&path, malformed, &[BufRef::Input(0)], &[BufRef::Output(0)]),
        Err(RecipeError::DepNotEarlier { node: 0, dep: 7 })
    );
}

/// A replacement is a computation, and it answers for no more than the one it
/// replaces: the edges around it were derived from that, and they are kept.
#[test]
fn replacement_work_is_proved() {
    let host = storing(BufRef::Output(0).at(0)).unwrap();
    let path = host.call_paths().remove(0);
    assert_eq!(
        host.replace_call(
            &path,
            Store {
                at: BufRef::Input(0).at(0)
            }
        ),
        Err(RecipeError::UndeclaredEffect {
            node: 0,
            kind: BufKind::Input,
            index: 0
        }),
        "a replacement that would store somewhere the old one did not is refused \
         before it can rest on edges derived from the old one"
    );
}

/// Flatness is decidable, so it is decided: a recipe that still bears a choice
/// cannot become a [`FlatRecipe`] however it is offered.
///
/// This is the whole of the public way in — there is no door that takes a
/// producer's word for it — so nothing downstream of a resolution meets a
/// choice at the point where there is nothing left to do about it.
#[test]
fn a_choice_bearing_recipe_cannot_become_flat() {
    let mut builder = RecipeBuilder::<Stores>::new(1, 1);
    builder
        .choice(vec![storing_fragment(BufRef::Output(0).at(0))])
        .inputs([Buf::Input(0)])
        .does([Buf::Input(0)], Access::Read)
        .outputs([Buf::Output(0)])
        .does([Buf::Output(0)], Access::Write)
        .add()
        .unwrap();
    let bearing = builder.finish(ports(64, 64)).unwrap();
    assert_eq!(
        FlatRecipe::new(bearing.recipe().clone()),
        Err(RecipeError::UnresolvedChoice { node: 0 })
    );

    // And the choice-free recipe beside it is flat, so the refusal is about the
    // choice rather than about this fixture.
    let flat = storing(BufRef::Output(0).at(0)).unwrap();
    assert_eq!(
        FlatRecipe::new(flat.clone()).map(FlatRecipe::into_recipe),
        Ok(flat)
    );
}
