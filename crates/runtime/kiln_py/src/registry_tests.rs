// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! That a name resolves to what was loaded under it, and to nothing otherwise.

use recipe::{Node, RecipeBody, RecipeSignature, Work, WritableBuf};

use super::*;

/// A recipe that touches no device: it is loaded and resolved, never executed,
/// so every test here runs without a GPU.
fn recipe(nbytes: usize) -> recipe::Recipe {
    recipe::Recipe::stating_edges(
        RecipeSignature::new(vec![nbytes], vec![nbytes]),
        recipe::CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Memset {
                    destination: WritableBuf::Output(0).at(0),
                    value: 0,
                    nbytes,
                },
                [],
            )],
        ),
    )
    .expect("the fixture is a recipe")
}

fn pending(recipe: recipe::Recipe) -> RunnableRecipe {
    RunnableRecipe::Pending {
        source: recipe.into(),
        persist_to: None,
        report_to: None,
        scheduling_time_limit_ms: None,
    }
}

fn loaded_identity(recipe: recipe::Recipe) -> RecipeIdentity {
    let identity = RecipeIdentity::of(&recipe).expect("identity");
    load(identity, pending(recipe)).expect("load");
    identity
}

/// The name a program carries finds the recipe it was loaded with.
#[test]
fn a_loaded_recipe_resolves_under_its_own_identity() {
    let identity = loaded_identity(recipe(16));
    let handle = resolve(identity.as_bytes()).expect("resolves");
    let RunnableRecipe::Pending { source, .. } = &*handle.lock().expect("lock") else {
        panic!("a recipe loaded and never executed is pending");
    };
    assert_eq!(source.contract(), recipe(16).contract());
}

/// Loading the same recipe again is the same entry, not a second one.
///
/// This is what keeps the table bounded by how many distinct programs a process
/// runs: every trace of one computation loads its recipe again, and none of them
/// adds anything. It is also what carries a tuning result from the trace that
/// realized it to the next one.
#[test]
fn loading_one_recipe_twice_yields_one_entry() {
    let first = loaded_identity(recipe(32));
    let second = loaded_identity(recipe(32));
    assert_eq!(first, second, "one recipe has one identity");
    assert!(
        Arc::ptr_eq(
            &resolve(first.as_bytes()).expect("first"),
            &resolve(second.as_bytes()).expect("second"),
        ),
        "the second load must find the first's entry rather than replace it"
    );
}

/// Two different recipes are two entries, so neither can be handed the other's.
#[test]
fn two_recipes_resolve_to_their_own_entries() {
    let small = loaded_identity(recipe(64));
    let large = loaded_identity(recipe(128));
    assert_ne!(small, large);
    assert!(!Arc::ptr_eq(
        &resolve(small.as_bytes()).expect("small"),
        &resolve(large.as_bytes()).expect("large"),
    ));
}

/// A name nothing was loaded under is refused, rather than resolving to
/// whatever else is in the table.
#[test]
fn an_unloaded_identity_is_refused() {
    loaded_identity(recipe(256));
    let error = resolve(&[0xAB; 32])
        .err()
        .expect("nothing is loaded under this");
    assert!(
        error.contains("no recipe is loaded under identity"),
        "got: {error}"
    );
}

/// A digest that is not 32 bytes is refused as malformed, not resolved by the
/// part of it that arrived.
///
/// A prefix names a program only by accident. Accepting one would turn a
/// truncation upstream into the wrong kernels running against a caller's
/// buffers, which no later check could catch — the recipe found would be a
/// consistent recipe, just not this one.
#[test]
fn an_identity_of_the_wrong_width_is_refused() {
    let identity = loaded_identity(recipe(512));
    for width in [0usize, 8, 16, 31, 33, 64] {
        let mut stated = identity.as_bytes().to_vec();
        stated.resize(width, 0);
        let error = resolve(&stated)
            .err()
            .unwrap_or_else(|| panic!("a {width}-byte identity is not one"));
        assert!(
            error.contains("a recipe identity is 32 bytes"),
            "width {width} got: {error}"
        );
    }
}
