// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which recipes this process has loaded, by content and tuning policy.
//!
//! A compiled program names the recipe it runs, and the name has to survive
//! being written into a compiler artifact and read back by a later process. An
//! address does not survive that: it means something only inside the process
//! that produced it, so a module carrying one is a different module in every
//! process and every cache keyed on the module misses. What the artifact
//! carries is therefore the recipe's [`RecipeIdentity`] — a digest of the
//! program itself, scoped to the policy selecting its tuned winner — and this
//! is where that name resolves to the loaded recipe at execution time.
//!
//! Resolution is the only direction that exists here. Nothing hands an address
//! back out, so there is no route by which one reaches an artifact again.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use kiln::RecipeIdentity;

use crate::handler::RunnableRecipe;

/// One entry per distinct recipe this process loaded, for as long as it runs.
///
/// The lifetime is the process's, and it has to be: XLA may execute an
/// executable it holds at any later time and says nothing when it stops holding
/// one, so a recipe released earlier than this would be one a live executable
/// could still name. What bounds the table is the identity — loading the same
/// recipe again finds its entry instead of adding one — so it grows with the
/// number of *distinct programs* the process runs, not with how many times each
/// is traced. That is the difference from leaking a box per load, which grew
/// with the latter.
fn loaded() -> &'static RwLock<HashMap<RecipeIdentity, Arc<Mutex<RunnableRecipe>>>> {
    static LOADED: OnceLock<RwLock<HashMap<RecipeIdentity, Arc<Mutex<RunnableRecipe>>>>> =
        OnceLock::new();
    LOADED.get_or_init(|| RwLock::new(HashMap::new()))
}

const POISONED: &str = "the loaded-recipe table is poisoned";

/// Loads `pending` under `identity`, unless that recipe is already loaded.
///
/// Two traces of one recipe share one loaded copy, which is what makes them
/// share a tuning result: the second finds the first's entry, and whatever the
/// first execution realized is already in it. `pending` is dropped in that
/// case, so a recipe arriving a second time costs nothing past the identity
/// that recognized it.
pub(crate) fn load(identity: RecipeIdentity, pending: RunnableRecipe) -> Result<(), String> {
    let mut table = loaded().write().map_err(|_| POISONED.to_string())?;
    table
        .entry(identity)
        .or_insert_with(|| Arc::new(Mutex::new(pending)));
    Ok(())
}

/// The recipe `identity` names, or a refusal saying so.
///
/// The handle is cloned out under a read lock which is then released, so a
/// realization that takes seconds blocks no other name from resolving: the
/// recipe's own `Mutex` is what serializes work on that recipe, and this table
/// is only how it was found.
pub(crate) fn resolve(identity: &[u8]) -> Result<Arc<Mutex<RunnableRecipe>>, String> {
    let identity = stated_identity(identity)?;
    let table = loaded().read().map_err(|_| POISONED.to_string())?;
    table.get(&identity).cloned().ok_or_else(|| {
        format!(
            "no recipe is loaded under identity {identity:?}: a program naming it \
             was compiled against a recipe this process never loaded"
        )
    })
}

/// The whole digest an attribute states, refusing anything shorter or longer.
///
/// A prefix is not a name. Resolving a program by part of its identity would
/// make a truncation anywhere upstream present as running the wrong kernels
/// against a caller's buffers, rather than as this refusal.
fn stated_identity(bytes: &[u8]) -> Result<RecipeIdentity, String> {
    <[u8; 32]>::try_from(bytes)
        .map(RecipeIdentity::from_bytes)
        .map_err(|_| {
            format!(
                "a recipe identity is 32 bytes; this attribute states {}",
                bytes.len()
            )
        })
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
