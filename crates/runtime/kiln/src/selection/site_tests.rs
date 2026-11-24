// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where a candidate sits, lifted back to the recipe that contains it.
//!
//! A selection made inside one candidate says nothing on its own: what runs
//! is the whole recipe with that candidate in place, so every level's
//! descended choice is replaced by what was built below it.

use super::*;

fn leaf(candidate: usize) -> Selection {
    Selection {
        candidate,
        nested: Vec::new(),
    }
}

/// At the root, a candidate's own selection is already the recipe's.
#[test]
fn the_root_lifts_to_itself() {
    let local = vec![leaf(2), leaf(0)];
    assert_eq!(root_selection(None, local.clone()), local);
}

/// One level down: the recipe's selection is the parent's, with the choice
/// that was descended through carrying the candidate under test and the
/// selection built inside it. The parent's other choices keep what they
/// were given — decided winners before, the warm-up's filler after.
#[test]
fn a_nested_candidate_lifts_into_the_whole_recipe() {
    let site = Site {
        parent: None,
        position: 1,
        candidate: 3,
        siblings: vec![leaf(7), leaf(0), leaf(5)],
    };
    assert_eq!(
        root_selection(Some(&site), vec![leaf(4)]),
        vec![
            leaf(7),
            Selection {
                candidate: 3,
                nested: vec![leaf(4)],
            },
            leaf(5),
        ],
        "the descended choice takes the candidate under test; its siblings stand"
    );
}

/// Depth composes: each frame replaces its own choice with what was built
/// below it, so a candidate three levels down still names one program.
#[test]
fn lifting_composes_through_every_ancestor() {
    let outer = Site {
        parent: None,
        position: 0,
        candidate: 1,
        siblings: vec![leaf(0)],
    };
    let inner = Site {
        parent: Some(&outer),
        position: 1,
        candidate: 2,
        siblings: vec![leaf(9), leaf(0)],
    };
    assert_eq!(
        root_selection(Some(&inner), vec![leaf(6)]),
        vec![Selection {
            candidate: 1,
            nested: vec![
                leaf(9),
                Selection {
                    candidate: 2,
                    nested: vec![leaf(6)],
                },
            ],
        }]
    );
}

/// The lifted selection is one the recipe accepts: `inline_body` requires
/// exactly one selection per choice, at every depth.
#[test]
fn a_lifted_selection_inlines_the_whole_recipe() {
    use recipe::{Arg, BufRef, KernelModule, Node, RecipeBody};

    let leaf_body = |entry: &str| {
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(vec![1])],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: entry.into(),
                        args: vec![Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [1, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        )
    };
    let choice = |candidates: Vec<RecipeBody>| {
        Node::new(
            Work::Choice {
                candidates,
                input_binding: Vec::new(),
                output_binding: vec![BufRef::Output(0)],
            },
            [],
        )
    };
    let inner = RecipeBody::new(
        0,
        1,
        Vec::new(),
        Vec::new(),
        vec![choice(vec![leaf_body("deep_a"), leaf_body("deep_b")])],
    );
    let root = RecipeBody::new(
        0,
        1,
        Vec::new(),
        Vec::new(),
        vec![choice(vec![leaf_body("shallow"), inner])],
    );

    // The site of the inner choice's candidate 1, reached through the
    // root's candidate 1.
    let outer = Site {
        parent: None,
        position: 0,
        candidate: 1,
        siblings: vec![leaf(0)],
    };
    let flat = inline_body(
        &CompileRequests::default(),
        &root,
        &root_selection(Some(&outer), vec![leaf(1)]),
        &(),
    )
    .unwrap();
    let Work::Call {
        call: KernelLaunch { entry, .. },
        ..
    } = flat.nodes()[0].work()
    else {
        panic!("the lifted selection inlines to one launch");
    };
    assert_eq!(entry, "deep_b");
}
