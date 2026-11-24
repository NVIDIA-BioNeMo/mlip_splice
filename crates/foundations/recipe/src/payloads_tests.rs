// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! That the one walk reaches every payload a recipe can carry, says what each
//! one is, and is the walk its consumers actually use.
//!
//! The fixture holds a distinct blob in each place one can live, at the root
//! and again inside a candidate. Asserting the walk yields exactly those bytes
//! is what makes a payload field added without a line here fail rather than be
//! quietly skipped by everything built on this walk.
//!
//! Coverage alone would not be enough, because a consumer is free to ignore a
//! walk it does not depend on. Detaching is tied to this enumeration here; the
//! artifact's blob pool is tied to it in `wire_pool_tests`, and a program's
//! identity in kiln.

use std::collections::BTreeSet;

use super::*;
use crate::algebra::{CompileRequests, Geometry, KernelLaunch};
use crate::code::{
    CompileRequest, CompileRequestId, ImagePersistence, KernelCode, KernelModule, ModuleImage,
    PtxCompileOptions, PtxTarget, ResourceBounds, SourceImage, SourceKind,
};
use crate::{Arg, Node, RecipeSignature, Work, WorkspaceBuf};

fn blob(byte: u8) -> Blob {
    Blob::from_vec(vec![byte])
}

/// A body with a payload in every place a body has one: a cubin image, a
/// CUDA-loadable image, and a workspace initializer.
fn body_with(first: u8) -> RecipeBody {
    RecipeBody::new(
        0,
        0,
        vec![
            KernelModule {
                code: KernelCode::ImageOnly {
                    preferred_image: ModuleImage::Cubin {
                        target_sm: 100,
                        bytes: blob(first),
                        provenance: None,
                    },
                    alternative_images: vec![ModuleImage::CudaLoadable {
                        bytes: blob(first + 1),
                        persistence: ImagePersistence::Persistable,
                        provenance: None,
                    }],
                },
                resource_bounds: ResourceBounds::default(),
            },
            KernelModule::from_request(
                CompileRequestId(0),
                vec![ModuleImage::Cubin {
                    target_sm: 90,
                    bytes: blob(first + 2),
                    provenance: None,
                }],
                ResourceBounds::default(),
            ),
        ],
        vec![WorkspaceBuf {
            nbytes: 1,
            init: Some(blob(first + 3)),
        }],
        Vec::new(),
    )
}

/// The same, with a candidate carrying its own of each below a `Choice`.
fn nested_recipe() -> Recipe {
    let root = body_with(0x10);
    let nodes = vec![
        Node::new(
            Work::Choice {
                candidates: vec![body_with(0x20)],
                input_binding: Vec::new(),
                output_binding: Vec::new(),
            },
            [],
        ),
        Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: "k".into(),
                    args: vec![Arg::scalar(vec![1, 2, 3, 4])],
                },
                geometry: vec![Geometry::single()],
            },
            [0],
        ),
    ];
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![CompileRequest::unguarded(
            vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: blob(0x01),
            }],
            PtxCompileOptions::default(),
        )]),
        RecipeBody::new(
            0,
            0,
            root.context().clone(),
            root.workspace().to_vec(),
            nodes,
        ),
    )
    .unwrap()
}

/// Each payload as `(what it is, the byte that tells it apart)`.
fn walk(recipe: &Recipe) -> Vec<(PayloadRole, u8)> {
    crate::payloads::recipe_payloads(recipe)
        .map(|payload| (payload.role, payload.blob[0]))
        .collect()
}

/// What the fixture carries, in the order the artifact writes it.
fn expected_walk() -> Vec<(PayloadRole, u8)> {
    use PayloadRole::{CompileSource, ModuleImage, WorkspaceInit};
    vec![
        (CompileSource, 0x01),
        // The root's three module images, in the order a loader would try
        // them: the preferred image, then its alternative, then the next
        // module's.
        (ModuleImage, 0x10),
        (ModuleImage, 0x11),
        (ModuleImage, 0x12),
        (WorkspaceInit, 0x13),
        // And the candidate's four, below its Choice.
        (ModuleImage, 0x20),
        (ModuleImage, 0x21),
        (ModuleImage, 0x22),
        (WorkspaceInit, 0x23),
    ]
}

/// Every payload, once, in the order the wire writes them, each saying what it
/// is: the compile table, then each body's modules, workspace, and nested
/// candidates.
#[test]
fn the_walk_reaches_every_payload_a_recipe_carries() {
    let recipe = nested_recipe();
    let seen = walk(&recipe);
    assert_eq!(seen, expected_walk());
    assert_eq!(
        seen.iter()
            .map(|(_, byte)| *byte)
            .collect::<BTreeSet<_>>()
            .len(),
        seen.len(),
        "the fixture states a distinct payload per place, so a repeat is a walk visiting one twice"
    );
}

/// A role decides how the artifact stores the bytes, and it decides it here
/// rather than at each field: sources are compressed, images and initializers
/// are not.
#[test]
fn a_role_states_how_the_artifact_stores_it() {
    assert!(PayloadRole::CompileSource.is_compressed());
    assert!(!PayloadRole::ModuleImage.is_compressed());
    assert!(!PayloadRole::WorkspaceInit.is_compressed());
    let tags: BTreeSet<u8> = [
        PayloadRole::CompileSource,
        PayloadRole::ModuleImage,
        PayloadRole::WorkspaceInit,
    ]
    .into_iter()
    .map(PayloadRole::tag)
    .collect();
    assert_eq!(tags.len(), 3, "a digest must tell the three roles apart");
}

/// The mutable walk visits the same payloads, as the same roles, in the same
/// order, so what detaching reaches is what pooling and hashing reach.
#[test]
fn the_mutable_walk_agrees_with_the_reading_one() {
    let mut recipe = nested_recipe();
    let expected = walk(&recipe);
    let mut seen = Vec::new();
    crate::payloads::visit_recipe_payloads(&mut recipe, &mut |payload| {
        seen.push((payload.role, payload.blob[0]));
    });
    assert_eq!(seen, expected);
}

/// An artifact read back carries the same payloads, as the same roles, in the
/// same order — so the numbering the pool gave them survives the round trip and
/// image priority is where it was.
#[test]
fn an_artifact_gives_the_walk_back_unchanged() {
    let wire = crate::serialize(&nested_recipe()).unwrap();
    assert_eq!(walk(&crate::deserialize(&wire).unwrap()), expected_walk());
}

/// Which payloads of `recipe` still have their bytes inside `artifact`.
///
/// A raw payload holds them as a view into the buffer; a compressed one holds
/// the compressed run the buffer carries, which decoding copies out of. Both
/// pin the artifact, and detaching is what stops them — so both have to be
/// looked for, or the roles the artifact compresses would appear detached from
/// the moment they were decoded.
fn bound_to(recipe: &Recipe, artifact: &Blob) -> Vec<PayloadRole> {
    let source = artifact.as_ptr_range();
    crate::payloads::recipe_payloads(recipe)
        .filter(|payload| !payload.blob.is_empty())
        .filter(|payload| match payload.blob.stored_encoding() {
            Some(crate::blob::StoredEncoding::Zstd { payload: run, .. }) => {
                source.contains(&run.as_ptr())
            }
            _ => source.contains(&payload.blob.as_ptr()),
        })
        .map(|payload| payload.role)
        .collect()
}

/// Detaching is that walk with one thing done to each payload, so every
/// payload — including a candidate's, three levels in — stops referencing the
/// buffer it was decoded from.
#[test]
fn detaching_reaches_every_payload() {
    let wire = Blob::from_vec(crate::serialize(&nested_recipe()).unwrap());
    let mut decoded = crate::deserialize_shared(wire.clone()).unwrap();
    assert_eq!(
        bound_to(&decoded, &wire).len(),
        expected_walk().len(),
        "a decoded recipe holds every payload out of the buffer it came from, \
         or this proves nothing"
    );
    decoded.detach_blobs();
    assert_eq!(
        bound_to(&decoded, &wire),
        Vec::new(),
        "after detaching, nothing is still held out of that buffer"
    );
}

/// And it reaches them *because* the walk does. A pass that skipped a role
/// leaves payloads of that role still bound to the artifact — which is the
/// failure detaching exists to prevent, arriving the moment the enumeration
/// stops being the one everything else uses.
#[test]
fn detaching_leaves_behind_exactly_what_the_walk_leaves_out() {
    let wire = Blob::from_vec(crate::serialize(&nested_recipe()).unwrap());
    for skipped in [
        PayloadRole::CompileSource,
        PayloadRole::ModuleImage,
        PayloadRole::WorkspaceInit,
    ] {
        let mut decoded = crate::deserialize_shared(wire.clone()).unwrap();
        crate::payloads::visit_recipe_payloads(&mut decoded, &mut |payload| {
            if payload.role != skipped {
                *payload.blob = payload.blob.detached();
            }
        });
        let left = bound_to(&decoded, &wire);
        assert!(
            !left.is_empty() && left.iter().all(|role| *role == skipped),
            "skipping {skipped:?} left {left:?} bound to the artifact"
        );
    }
}
