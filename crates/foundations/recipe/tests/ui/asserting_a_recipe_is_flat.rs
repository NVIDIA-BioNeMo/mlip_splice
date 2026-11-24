// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flatness is decided from the graph, so a choice-bearing recipe cannot become
//! a `FlatRecipe`: there is no unchecked constructor, and the recipe it wraps is
//! not a field a caller reaches.

use recipe::{FlatRecipe, Recipe};

fn wrapping_without_asking(held: Recipe) -> FlatRecipe {
    FlatRecipe::assume_flat(held)
}

fn wrapping_by_hand(held: Recipe) -> FlatRecipe {
    FlatRecipe(held)
}

fn unwrapping_to_put_a_choice_back(held: &mut FlatRecipe) {
    held.0 = held.0.clone();
}

fn main() {
    let _ = wrapping_without_asking;
    let _ = wrapping_by_hand;
    let _ = unwrapping_to_put_a_choice_back;
}
