// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a candidate is bound through is the binding that selects it, at every
//! depth — never its position among the root's ports.
//!
//! The fixtures here bind unequal extents through permuted slots and give each
//! candidate fewer formal slots than the root has ports, so anything that read
//! a candidate's slot `k` as the root's port `k` would either refuse work that
//! fits or admit work that does not.

use super::*;
use recipe::{
    Arg, BufRef, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, Recipe, RecipeBody,
    RecipeSignature, WorkspaceBuf, WritableBuf,
};

/// Two inputs an order of magnitude apart, so binding one where the other
/// belongs is a different answer rather than a coincidence.
const WIDE: usize = 4096;
const NARROW: usize = 8;
const OUT: usize = 16;

fn ports() -> RecipeSignature {
    RecipeSignature::new(vec![WIDE, NARROW], vec![OUT])
}

fn module() -> KernelModule {
    KernelModule::from_cuda_loadable(vec![0xAA])
}

/// A body of one input and one output that copies `nbytes` out of the far
/// end of its input: legal only where its slot is bound to a buffer that
/// holds them.
fn reaches(nbytes: usize) -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        vec![module()],
        Vec::new(),
        vec![Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: "candidate".into(),
                    args: vec![
                        Arg::pointer(BufRef::Input(0).at(nbytes - 1)),
                        Arg::output(0),
                    ],
                },
                geometry: vec![Geometry::single()],
            },
            [],
        )],
    )
}

fn rooted(node: Node) -> Result<Recipe, RecipeError<ExecutableRefusal>> {
    Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(2, 1, Vec::new(), Vec::new(), vec![node]),
    )
}

fn choice(candidates: Vec<RecipeBody>, input: BufRef) -> Node {
    Node::new(
        Work::Choice {
            candidates,
            input_binding: vec![input],
            output_binding: vec![BufRef::Output(0)],
        },
        [],
    )
}

/// A candidate reaching to the end of the wide buffer is a recipe when the
/// binding hands it that buffer, and is not one when the binding hands it
/// the narrow one — whichever slot index either sits at.
#[test]
fn a_candidate_is_checked_against_the_buffer_its_binding_selects() {
    assert_eq!(
        rooted(choice(vec![reaches(WIDE)], BufRef::Input(0))).map(|_| ()),
        Ok(()),
        "slot 0 is bound to the wide input, which holds what the candidate reaches"
    );
    assert!(
        matches!(
            rooted(choice(vec![reaches(WIDE)], BufRef::Input(1))),
            Err(RecipeError::AccessPastEnd {
                capacity: NARROW,
                ..
            })
        ),
        "the same candidate bound to the narrow input reaches past its end"
    );
    assert_eq!(
        rooted(choice(vec![reaches(NARROW)], BufRef::Input(1))).map(|_| ()),
        Ok(()),
        "and a candidate that fits the narrow input is a recipe there"
    );
}

/// The same, one level further down: the inner choice's binding is over its
/// parent candidate's formal slots, which are themselves bound.
#[test]
fn a_nested_candidate_is_checked_through_every_binding_above_it() {
    let inner = |nbytes: usize| {
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![choice(vec![reaches(nbytes)], BufRef::Input(0))],
        )
    };
    // The outer choice binds its one slot to the wide input, so the inner
    // candidate reaching the wide extent is in range through both.
    assert_eq!(
        rooted(choice(vec![inner(WIDE)], BufRef::Input(0))).map(|_| ()),
        Ok(()),
        "both bindings select the wide input"
    );
    assert!(
        matches!(
            rooted(choice(vec![inner(WIDE)], BufRef::Input(1))),
            Err(RecipeError::AccessPastEnd {
                capacity: NARROW,
                ..
            })
        ),
        "the outer binding decides what the inner candidate is checked against"
    );
}

/// Inlining a permuted binding hands the candidate's work the buffer the
/// binding named, and the result is a recipe over the root's own ports.
#[test]
fn inlining_moves_a_candidate_onto_the_buffer_its_binding_named() {
    let recipe = rooted(choice(vec![reaches(WIDE)], BufRef::Input(0))).unwrap();
    let flat = inline(&recipe, &body_default_selection(recipe.body()), &()).unwrap();
    assert_eq!(flat.contract(), recipe.contract());
    assert_eq!(
        flat.body().inputs(),
        2,
        "the inlined body is the root's, bound through the root's ports"
    );
    let Work::Call {
        call: KernelLaunch { args, .. },
        ..
    } = flat.body().nodes()[0].work()
    else {
        panic!("the candidate inlined to its launch")
    };
    assert_eq!(
        args[0].segments(),
        [recipe::Segment::Pointer(BufRef::Input(0).at(WIDE - 1))],
        "the candidate's slot 0 became the input the binding named"
    );
    assert_eq!(flat.validate(), Ok(()));
}

/// A candidate that has fewer formal slots than the root has ports is
/// still a candidate: what it is bound through is the binding's length,
/// and the root's port count is nothing to do with it.
#[test]
fn a_candidates_arity_is_its_bindings_and_not_the_roots() {
    let recipe = rooted(choice(vec![reaches(NARROW)], BufRef::Input(1))).unwrap();
    let Work::Choice { candidates, .. } = recipe.body().nodes()[0].work() else {
        panic!("a choice")
    };
    assert_eq!(recipe.body().inputs(), 2);
    assert_eq!(
        candidates[0].inputs(),
        1,
        "one bound slot, whatever the root binds"
    );
    // And a candidate that restated the root's arity is refused, because
    // the binding fills exactly the slots the body has.
    let wrong = RecipeBody::new(
        2,
        1,
        vec![module()],
        Vec::new(),
        vec![Node::new(
            Work::Memset {
                destination: WritableBuf::Output(0).at(0),
                value: 0,
                nbytes: 4,
            },
            [],
        )],
    );
    assert!(matches!(
        rooted(choice(vec![wrong], BufRef::Input(1))),
        Err(RecipeError::BindingArity {
            bound: 1,
            formal: 2,
            ..
        })
    ));
}

/// A candidate's own workspace is its own: it is carved after the parent's
/// and never confused with a port.
#[test]
fn a_candidates_workspace_stays_below_its_binding() {
    let candidate = RecipeBody::new(
        1,
        1,
        vec![module()],
        vec![WorkspaceBuf {
            nbytes: 32,
            init: None,
        }],
        vec![Node::new(
            Work::Memset {
                destination: WritableBuf::Workspace(0).at(0),
                value: 0,
                nbytes: 32,
            },
            [],
        )],
    );
    let recipe = rooted(choice(vec![candidate], BufRef::Input(1))).unwrap();
    assert_eq!(recipe.validate(), Ok(()));
    let flat = inline(&recipe, &body_default_selection(recipe.body()), &()).unwrap();
    assert_eq!(
        flat.body().workspace().len(),
        1,
        "the candidate's private buffer came up with its work"
    );
}
