// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every payload a recipe carries, in one walk.
//!
//! A recipe's bytes live in four places — a compile's source, a module image, a
//! workspace initializer, and the same three inside every `Choice` candidate —
//! and three things need all of them: detaching from a decoded buffer, pooling
//! into an artifact, hashing an identity. Each of those knowing the list
//! separately is how a newly added payload field comes to be handled by two of
//! them and silently skipped by the third.
//!
//! So the list is here once, and each entry states what it *is*. The role is
//! not decoration: it decides whether the artifact compresses the bytes, and it
//! is what a digest frames them by. A consumer that inferred either from where
//! it happened to be standing in the structure would be a second statement of
//! this one, free to drift from it.
//!
//! What varies between the consumers is what they do with a payload, never
//! which payloads there are or what each one is.

use crate::algebra::{FlatRecipe, Launches, ModuleTable, Recipe, RecipeBody, Work};
use crate::blob::Blob;
use crate::code::CompileRequest;

/// What a payload is, in the recipe that carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PayloadRole {
    /// The text a compile request assembles. PTX is highly compressible and
    /// often the largest thing in a recipe, so an artifact stores it zstd.
    CompileSource,
    /// A module image: a cubin, or bytes the CUDA loader accepts. Already
    /// compact, and read back by a loader that wants them as they are, so an
    /// artifact stores it raw.
    ModuleImage,
    /// The bytes a workspace buffer starts life holding. Stored raw, for the
    /// same reason as an image.
    WorkspaceInit,
}

impl PayloadRole {
    /// Whether an artifact compresses payloads of this role.
    ///
    /// The one statement of it: the wire reads this rather than deciding per
    /// field, so a role added above cannot be stored one way by the writer and
    /// another by whatever else learns about it.
    pub fn is_compressed(self) -> bool {
        match self {
            PayloadRole::CompileSource => true,
            PayloadRole::ModuleImage | PayloadRole::WorkspaceInit => false,
        }
    }

    /// The byte a digest frames a payload of this role by, so that identical
    /// bytes serving two purposes are two programs rather than one.
    pub fn tag(self) -> u8 {
        match self {
            PayloadRole::CompileSource => 0,
            PayloadRole::ModuleImage => 1,
            PayloadRole::WorkspaceInit => 2,
        }
    }
}

/// One payload and what it is.
#[derive(Clone, Copy, Debug)]
pub struct Payload<'a> {
    pub role: PayloadRole,
    pub blob: &'a Blob,
}

/// The same, for a walk that replaces payloads.
#[derive(Debug)]
pub struct PayloadMut<'a> {
    pub role: PayloadRole,
    pub blob: &'a mut Blob,
}

/// Every payload in `body`, and in every candidate below it, in encounter
/// order: modules before workspace before nested candidates, and within a
/// module its images in priority order.
///
/// That order is the artifact's. The wire writes payload-bearing fields in it,
/// so a pool built from this walk numbers its entries the way the fields that
/// reference them are written, and priority — which image a loader tries
/// first, which candidate a tuner meets first — survives unchanged.
pub fn body_payloads<F: Launches>(body: &RecipeBody<F>) -> impl Iterator<Item = Payload<'_>> {
    let modules = body.context().modules().iter().flat_map(|module| {
        module.code.images().map(|image| Payload {
            role: PayloadRole::ModuleImage,
            blob: image.bytes(),
        })
    });
    let workspace = body.workspace().iter().filter_map(|buffer| {
        buffer.init.as_ref().map(|init| Payload {
            role: PayloadRole::WorkspaceInit,
            blob: init,
        })
    });
    let nested = body
        .nodes()
        .iter()
        .flat_map(|node| -> Box<dyn Iterator<Item = Payload<'_>>> {
            match node.work() {
                Work::Choice { candidates, .. } => {
                    Box::new(candidates.iter().flat_map(body_payloads))
                }
                Work::Call { .. } | Work::Memset { .. } | Work::Memcpy { .. } => {
                    Box::new(std::iter::empty())
                }
            }
        });
    modules.chain(workspace).chain(nested)
}

/// The same walk over a whole recipe: its compile table first, then its body.
pub fn recipe_payloads<F: Launches>(recipe: &Recipe<F>) -> impl Iterator<Item = Payload<'_>> {
    request_payloads(recipe.context().as_slice()).chain(body_payloads(recipe.body()))
}

/// Every source of every request, each request's unit in link order.
fn request_payloads(requests: &[CompileRequest]) -> impl Iterator<Item = Payload<'_>> {
    requests.iter().flat_map(|request| {
        request.sources.iter().map(|source| Payload {
            role: PayloadRole::CompileSource,
            blob: &source.blob,
        })
    })
}

/// Apply `visit` to every payload in `body`, in the same order
/// [`body_payloads`] yields them.
///
/// Private because the mutable walk exists for whole recipes: a caller holding
/// a body alone reads payloads, it does not replace them.
fn visit_body_payloads<F: Launches>(
    body: &mut RecipeBody<F>,
    visit: &mut impl FnMut(PayloadMut<'_>),
) {
    let (modules, workspace, nodes) = body.parts_mut();
    for module in modules.modules_mut().iter_mut() {
        for image in module.code.images_mut() {
            visit(PayloadMut {
                role: PayloadRole::ModuleImage,
                blob: image.bytes_mut(),
            });
        }
    }
    for buffer in workspace.iter_mut() {
        if let Some(init) = &mut buffer.init {
            visit(PayloadMut {
                role: PayloadRole::WorkspaceInit,
                blob: init,
            });
        }
    }
    for node in nodes.iter_mut() {
        if let Work::Choice { candidates, .. } = node.work_mut() {
            for candidate in candidates {
                visit_body_payloads(candidate, visit);
            }
        }
    }
}

/// The same, over a whole recipe.
pub fn visit_recipe_payloads<F: Launches>(
    recipe: &mut Recipe<F>,
    visit: &mut impl FnMut(PayloadMut<'_>),
) {
    let (requests, body) = recipe.parts_mut();
    for request in requests.as_mut_slice() {
        for source in &mut request.sources {
            visit(PayloadMut {
                role: PayloadRole::CompileSource,
                blob: &mut source.blob,
            });
        }
    }
    visit_body_payloads(body, visit);
}

impl<F: Launches> Recipe<F> {
    /// Re-owns every payload blob so the recipe stops referencing (and pinning)
    /// the shared buffer it was decoded from.
    ///
    /// Which payloads there are is [`recipe_payloads`]'s to say, so a field
    /// added there is detached here without this being touched.
    pub fn detach_blobs(&mut self) {
        visit_recipe_payloads(self, &mut |payload| {
            *payload.blob = payload.blob.detached();
        });
    }
}

impl<F: Launches> RecipeBody<F> {
    /// The same, for a body held on its own — a resolved candidate, which is a
    /// body and not a recipe.
    pub fn detach_blobs(&mut self) {
        visit_body_payloads(self, &mut |payload| {
            *payload.blob = payload.blob.detached();
        });
    }
}

impl<F: Launches> FlatRecipe<F> {
    /// The same, for a recipe already known to have no choice left. Detaching
    /// touches payload bytes and not the node graph, so what was flat stays
    /// flat.
    pub fn detach_blobs(&mut self) {
        self.recipe_mut().detach_blobs();
    }
}

#[cfg(test)]
#[path = "payloads_tests.rs"]
mod tests;
