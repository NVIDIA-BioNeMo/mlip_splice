// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Choosing which candidate runs, and refusing a selection that is not this
//! recipe's.
//!
//! Inlining changes what a recipe executes and not what it is bound through,
//! so the ports and the compile table come across untouched — and a selection
//! is proved against the recipe before anything is read from it.

use super::*;
use recipe::{Arg, BufRef, KernelModule, Node, RecipeSignature, WorkspaceBuf};

/// The one word every candidate here reads and writes.
const WORD: usize = 4;

/// What these fixtures are bound through: one input and one output. Every
/// candidate takes `Arg::input(0)` and `Arg::output(0)`, so the recipe is
/// not portless and does not get `RecipeSignature::empty()` — inlining is
/// checked for carrying exactly these across.
fn port_signature() -> RecipeSignature {
    RecipeSignature::new(vec![WORD], vec![WORD])
}

fn candidate(entry: &str, byte: u8, workspace: usize) -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        vec![KernelModule::from_cuda_loadable(vec![byte])],
        vec![WorkspaceBuf {
            nbytes: workspace,
            init: None,
        }],
        vec![Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
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
}

fn choice_recipe() -> Recipe {
    // The choice binds input 0 and output 0; the candidates fill one word.
    Recipe::stating_edges(
        port_signature(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Choice {
                    candidates: vec![candidate("first", 1, 16), candidate("second", 2, 32)],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// Inlining changes what a recipe executes, not what it is bound through:
/// the ports come out exactly as they went in, and the result validates
/// against them. The scratch recipe the inliner builds to carry a body is
/// consumed on the spot and is never what a caller receives.
#[test]
fn inlining_hands_back_the_ports_it_was_given() {
    let recipe = choice_recipe();
    let flat = inline(&recipe, &body_default_selection(recipe.body()), &()).unwrap();
    assert_eq!(flat.contract(), recipe.contract());
    assert_eq!(*flat.contract(), port_signature());
    flat.validate()
        .expect("an inlined recipe is bound through its own ports");
    assert!(
        !flat
            .body()
            .nodes()
            .iter()
            .any(|n| matches!(n.work(), Work::Choice { .. })),
        "inlining leaves no choice behind"
    );
}

/// A recipe whose chosen candidate holds a choice of its own, for the
/// malformed cases that only appear below the first level.
fn nested_recipe() -> Recipe {
    let inner = RecipeBody::new(
        1,
        1,
        Vec::new(),
        Vec::new(),
        vec![Node::new(
            Work::Choice {
                candidates: vec![
                    candidate("inner first", 3, 8),
                    candidate("inner second", 4, 8),
                ],
                input_binding: vec![BufRef::Input(0)],
                output_binding: vec![BufRef::Output(0)],
            },
            [],
        )],
    );
    Recipe::stating_edges(
        port_signature(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Choice {
                    candidates: vec![candidate("outer", 1, 16), inner],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// A selection is proved against the recipe before anything is read from
/// it. Every way of not naming this recipe's choices is a refusal that
/// says which — never an index into whatever happens to sit there.
#[test]
fn a_selection_that_is_not_this_recipe_s_is_refused() {
    let recipe = choice_recipe();
    let one = |candidate| Selection {
        candidate,
        nested: Vec::new(),
    };
    assert!(matches!(
        inline(&recipe, &[], &()),
        Err(RecipeError::SelectionArity {
            stated: 0,
            choices: 1
        })
    ));
    assert!(matches!(
        inline(&recipe, &[one(0), one(0)], &()),
        Err(RecipeError::SelectionArity {
            stated: 2,
            choices: 1
        })
    ));
    assert!(matches!(
        inline(&recipe, &[one(2)], &()),
        Err(RecipeError::NoSuchCandidate {
            node: 0,
            candidate: 2,
            offered: 2
        })
    ));
    assert!(matches!(
        inline(&recipe, &[one(usize::MAX)], &()),
        Err(RecipeError::NoSuchCandidate { .. })
    ));
    // A candidate that holds no choice is named by an empty selection, so
    // stating one for it is a selection of some other recipe.
    assert!(matches!(
        inline(
            &recipe,
            &[Selection {
                candidate: 0,
                nested: vec![one(0)],
            }],
            &(),
        ),
        Err(RecipeError::SelectionArity {
            stated: 1,
            choices: 0
        })
    ));
}

/// The same, one level down: a well-formed outer entry does not excuse a
/// malformed inner one.
#[test]
fn a_malformed_nested_selection_is_refused_where_it_is_malformed() {
    let recipe = nested_recipe();
    let one = |candidate| Selection {
        candidate,
        nested: Vec::new(),
    };
    let outer = |nested: Vec<Selection>| {
        vec![Selection {
            candidate: 1,
            nested,
        }]
    };
    assert!(matches!(
        inline(&recipe, &outer(Vec::new()), &()),
        Err(RecipeError::SelectionArity {
            stated: 0,
            choices: 1
        })
    ));
    assert!(matches!(
        inline(&recipe, &outer(vec![one(0), one(1)]), &()),
        Err(RecipeError::SelectionArity {
            stated: 2,
            choices: 1
        })
    ));
    assert!(matches!(
        inline(&recipe, &outer(vec![one(7)]), &()),
        Err(RecipeError::NoSuchCandidate {
            candidate: 7,
            offered: 2,
            ..
        })
    ));
    // And the one that names it correctly still resolves.
    assert!(inline(&recipe, &outer(vec![one(1)]), &()).is_ok());
}

#[test]
fn default_selects_first_candidate() {
    let recipe = choice_recipe();
    let selection = body_default_selection(recipe.body());
    assert_eq!(selection, vec![Selection::leaf(0)]);
    let flat = inline(&recipe, &selection, &()).unwrap();
    assert_eq!(flat.body().context().len(), 1);
    assert_eq!(flat.body().workspace()[0].nbytes, 16);
    let Work::Call {
        call: KernelLaunch { entry, .. },
        ..
    } = flat.body().nodes()[0].work()
    else {
        panic!("choice did not inline to a launch")
    };
    assert_eq!(entry, "first");
}

#[test]
fn explicit_selection_reindexes_candidate() {
    let recipe = choice_recipe();
    let flat = inline(&recipe, &[Selection::leaf(1)], &()).unwrap();
    assert_eq!(flat.body().workspace()[0].nbytes, 32);
    assert!(matches!(
        flat.body().nodes()[0].work(),
        Work::Call { call: KernelLaunch { entry, module: 0, .. }, .. } if entry == "second"
    ));
    assert!(recipe::FlatRecipe::new(flat).is_ok());
}

#[test]
fn nested_choices_are_resolved_recursively() {
    let inner = RecipeBody::new(
        1,
        1,
        Vec::new(),
        Vec::new(),
        vec![Node::new(
            Work::Choice {
                candidates: vec![candidate("inner_a", 3, 8), candidate("inner_b", 4, 8)],
                input_binding: vec![BufRef::Input(0)],
                output_binding: vec![BufRef::Output(0)],
            },
            [],
        )],
    );
    let recipe = Recipe::stating_edges(
        port_signature(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Choice {
                    candidates: vec![inner],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap();
    let flat = inline(
        &recipe,
        &[Selection {
            candidate: 0,
            nested: vec![Selection::leaf(1)],
        }],
        &(),
    )
    .unwrap();
    assert!(matches!(
        flat.body().nodes()[0].work(),
        Work::Call { call: KernelLaunch { entry, .. }, .. } if entry == "inner_b"
    ));
}

#[test]
fn covering_recipes_include_each_candidate() {
    let recipe = choice_recipe();
    let mut entries = Vec::new();
    for selection in covering_selections(&recipe) {
        let covering = inline(&recipe, &selection, &()).unwrap();
        let Work::Call {
            call: KernelLaunch { entry, .. },
            ..
        } = covering.body().nodes()[0].work()
        else {
            panic!("covering recipe is not flat")
        };
        entries.push(entry.clone());
    }
    entries.sort();
    assert_eq!(entries, vec!["first", "second"]);
}

/// Realization rewrites the body and nothing else. The ports must come
/// across untouched, because the resident recipe is what every later
/// replay is checked against — a signature dropped here would leave the
/// execution boundary comparing bindings to an empty statement and
/// accepting anything.
#[test]
fn inlining_carries_the_ports_to_the_resident_recipe() {
    let recipe = choice_recipe();
    // The same work, bound through wider ports than the fixture states.
    let signature = recipe::RecipeSignature::new(vec![WORD * 4], vec![WORD * 2]);
    let (_, table, body) = recipe.into_parts();
    let recipe = Recipe::stating_edges(signature.clone(), table, body).unwrap();
    assert!(
        recipe
            .body()
            .nodes()
            .iter()
            .any(|n| matches!(n.work(), Work::Choice { .. })),
        "the fixture must have a choice to inline"
    );

    let selections = covering_selections(&recipe);
    assert!(!selections.is_empty(), "a choice offers a selection");
    for selection in &selections {
        let flat = inline(&recipe, selection, &()).unwrap();
        assert_eq!(
            *flat.contract(),
            signature,
            "inlining {selection:?} kept a different signature"
        );
        assert!(
            !flat
                .body()
                .nodes()
                .iter()
                .any(|n| matches!(n.work(), Work::Choice { .. })),
            "and it really did inline"
        );
    }
}
