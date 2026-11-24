// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! That the artifact's blob pool *is* the payload walk.
//!
//! The pool could have been filled by the writer discovering payloads as it
//! reached them, and for a while it was. Then two answers existed to "what does
//! this recipe carry", and the day they disagreed the artifact would have been
//! the one that was wrong. So the pool is built from
//! [`crate::payloads::recipe_payloads`] and the writer may only address it, and
//! these tests hold that: same entries, same numbering, same stored form, and a
//! loud failure in either direction when the two are made to differ.

use crate::algebra::CompileRequests;
use crate::code::{
    CompileRequest, CompileRequestId, ImagePersistence, KernelCode, KernelModule, ModuleImage,
    PtxCompileOptions, PtxTarget, ResourceBounds, SourceImage, SourceKind,
};
use crate::payloads::{Payload, PayloadRole, recipe_payloads};
use crate::{Blob, WireError};
use crate::{Node, Recipe, RecipeBody, RecipeSignature, Work, WorkspaceBuf};

use super::{Bucket, PayloadEncoding, Pool, write_sections};

fn blob(byte: u8) -> Blob {
    Blob::from_vec(vec![byte])
}

/// A payload in every place one can live, at the root and again inside a
/// candidate: two images in one module, an image in another, an initializer,
/// and a compile source above them all.
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

fn nested_recipe() -> Recipe {
    let root = body_with(0x10);
    let (inputs, outputs, modules, workspace, mut nodes) = root.into_parts();
    nodes.push(Node::new(
        Work::Choice {
            candidates: vec![body_with(0x20)],
            input_binding: Vec::new(),
            output_binding: Vec::new(),
        },
        [],
    ));
    let root = RecipeBody::new(inputs, outputs, modules, workspace, nodes);
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![CompileRequest::unguarded(
            vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(b".version 8.8\n.target sm_100\n".to_vec()),
            }],
            PtxCompileOptions::default(),
        )]),
        root,
    )
    .unwrap()
}

/// One entry per walk entry, in walk order, each stored the way its role says.
#[test]
fn the_pool_is_the_walk_in_order_and_in_the_form_each_role_states() {
    let recipe = nested_recipe();
    let roles: Vec<PayloadRole> = recipe_payloads(&recipe)
        .map(|payload| payload.role)
        .collect();
    assert_eq!(roles.len(), 9, "the fixture carries nine distinct payloads");

    let pool = Pool::from_payloads(recipe_payloads(&recipe)).unwrap();
    assert_eq!(
        pool.blobs.len(),
        roles.len(),
        "the payloads are all distinct, so nothing deduplicated"
    );
    for (entry, role) in pool.blobs.iter().zip(&roles) {
        assert_eq!(
            entry.encoding == PayloadEncoding::Zstd,
            role.is_compressed(),
            "{role:?} is not stored the way its role states"
        );
    }
}

/// Equal bytes in two places are one entry, so the pool is the walk with
/// duplicates collapsed rather than the walk verbatim.
#[test]
fn content_equal_payloads_share_one_entry() {
    // The candidate's initializer becomes a separate blob holding the root's
    // bytes: two places, one content.
    let recipe = {
        let (ports, table, body) = nested_recipe().into_parts();
        let (inputs, outputs, modules, workspace, mut nodes) = body.into_parts();
        let Work::Choice { candidates, .. } = nodes[0].work_mut() else {
            panic!("the fixture's first node is the Choice");
        };
        let (kin, kout, kmodules, mut kworkspace, knodes) = candidates[0].clone().into_parts();
        kworkspace[0].init = Some(Blob::from_vec(vec![0x13]));
        candidates[0] = RecipeBody::new(kin, kout, kmodules, kworkspace, knodes);
        Recipe::stating_edges(
            ports,
            table,
            RecipeBody::new(inputs, outputs, modules, workspace, nodes),
        )
        .unwrap()
    };

    assert_eq!(recipe_payloads(&recipe).count(), 9);
    let pool = Pool::from_payloads(recipe_payloads(&recipe)).unwrap();
    assert_eq!(pool.blobs.len(), 8, "the repeated content is pooled once");
    assert!(
        write_sections(
            &recipe,
            &mut Pool::from_payloads(recipe_payloads(&recipe)).unwrap()
        )
        .is_ok(),
        "and both places still name that one entry"
    );
}

/// A walk that skipped a role would write an artifact missing those bytes. The
/// writer refuses instead: it can only name entries the walk produced, so the
/// disagreement surfaces here rather than in whoever loads the artifact later.
#[test]
fn a_walk_that_omits_a_role_cannot_be_written() {
    let recipe = nested_recipe();
    for omitted in [
        PayloadRole::CompileSource,
        PayloadRole::ModuleImage,
        PayloadRole::WorkspaceInit,
    ] {
        let partial = recipe_payloads(&recipe).filter(|payload| payload.role != omitted);
        let mut pool = Pool::from_payloads(partial).unwrap();
        assert_eq!(
            write_sections(&recipe, &mut pool).map(|_| ()),
            Err(WireError::PayloadWalkDisagrees {
                what: "a field names bytes the walk did not reach",
            }),
            "an artifact was written without its {omitted:?} payloads"
        );
    }
}

/// The other direction: a walk reaching bytes no field names would leave an
/// entry in the artifact that nothing addresses.
#[test]
fn a_walk_that_reaches_more_than_the_fields_do_cannot_be_written() {
    let recipe = nested_recipe();
    let stray = blob(0xFF);
    let extra = recipe_payloads(&recipe).chain(std::iter::once(Payload {
        role: PayloadRole::ModuleImage,
        blob: &stray,
    }));
    let mut pool = Pool::from_payloads(extra).unwrap();
    assert_eq!(
        write_sections(&recipe, &mut pool).map(|_| ()),
        Err(WireError::PayloadWalkDisagrees {
            what: "the walk reached bytes no field names",
        })
    );
}

/// Reordering the walk renumbers the pool, and the fields follow — so priority
/// is preserved by the walk stating it, not by the writer restating it.
#[test]
fn the_walks_order_is_the_pools_numbering() {
    let recipe = nested_recipe();
    let forward = Pool::from_payloads(recipe_payloads(&recipe)).unwrap();
    let backward: Vec<Payload<'_>> = recipe_payloads(&recipe).collect();
    let reversed = Pool::from_payloads(backward.into_iter().rev()).unwrap();
    let digests = |pool: &Pool<'_>| pool.blobs.iter().map(|e| e.digest).collect::<Vec<_>>();
    let mut forward_digests = digests(&forward);
    let reversed_digests = digests(&reversed);
    assert_ne!(
        forward_digests, reversed_digests,
        "the fixture's payloads differ, so a reversed walk numbers them differently"
    );
    forward_digests.reverse();
    assert_eq!(forward_digests, reversed_digests);
}

/// A recipe carrying `bytes` in each of the three roles at once.
fn every_role_holding(source: Blob, image: Blob, init: Blob) -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::new(vec![CompileRequest::unguarded(
            vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: source,
            }],
            PtxCompileOptions::default(),
        )]),
        RecipeBody::new(
            0,
            0,
            vec![KernelModule {
                code: KernelCode::ImageOnly {
                    preferred_image: ModuleImage::CudaLoadable {
                        bytes: image,
                        persistence: ImagePersistence::Persistable,
                        provenance: None,
                    },
                    alternative_images: Vec::new(),
                },
                resource_bounds: ResourceBounds::default(),
            }],
            vec![WorkspaceBuf {
                nbytes: SHARED.len(),
                init: Some(init),
            }],
            Vec::new(),
        ),
    )
    .unwrap()
}

/// Long enough that zstd storage is plainly not the raw bytes.
const SHARED: &[u8] = b"the same bytes in three places, over and over and over and over";

/// The pool entry each of the three fields ends up naming, and the form that
/// entry is stored in.
fn fields_and_forms(recipe: &Recipe) -> (Vec<u32>, Vec<PayloadEncoding>) {
    let mut pool = Pool::from_payloads(recipe_payloads(recipe)).unwrap();
    write_sections(recipe, &mut pool).expect("the fixture writes");
    let named: Vec<u32> = recipe_payloads(recipe)
        .map(|payload| payload.role)
        .zip(recipe_payloads(recipe).map(|payload| payload.blob))
        .map(|(role, blob)| pool.index_of(blob, role).unwrap())
        .collect();
    let forms = pool.blobs.iter().map(|entry| entry.encoding).collect();
    (named, forms)
}

/// Equal content in a source and in an image is one content stored two ways, so
/// it is two entries. Sharing one would give a field bytes in a form its role
/// does not state — an image the artifact compressed, or a source it did not.
#[test]
fn equal_content_in_two_roles_is_two_entries_stored_two_ways() {
    let recipe = every_role_holding(
        Blob::from_vec(SHARED.to_vec()),
        Blob::from_vec(SHARED.to_vec()),
        Blob::from_vec(SHARED.to_vec()),
    );
    let (named, forms) = fields_and_forms(&recipe);
    assert_eq!(
        forms,
        vec![PayloadEncoding::Zstd, PayloadEncoding::Raw],
        "the source is compressed and the two raw roles share one entry"
    );
    assert_eq!(
        named,
        vec![0, 1, 1],
        "source names the compressed entry; image and initializer share the raw one"
    );

    let wire = crate::serialize(&recipe).unwrap();
    assert_eq!(crate::deserialize(&wire).unwrap(), recipe);
}

/// And the same holds when the three roles are literally one `Blob` — one place,
/// so a pool that keyed placement by place alone would have lost one of them.
#[test]
fn one_blob_reused_across_roles_still_gets_the_entry_its_role_needs() {
    let shared = Blob::from_vec(SHARED.to_vec());
    let recipe = every_role_holding(shared.clone(), shared.clone(), shared.clone());
    assert_eq!(
        recipe_payloads(&recipe)
            .map(|payload| payload.blob.place())
            .collect::<std::collections::HashSet<_>>()
            .len(),
        1,
        "all three roles hold the one blob, or this tests nothing"
    );
    let (named, forms) = fields_and_forms(&recipe);
    assert_eq!(forms, vec![PayloadEncoding::Zstd, PayloadEncoding::Raw]);
    assert_eq!(named, vec![0, 1, 1]);

    let wire = crate::serialize(&recipe).unwrap();
    let back = crate::deserialize(&wire).unwrap();
    assert_eq!(back, recipe);
    // Read back, the source is the compressed entry and the image the raw one,
    // which is what "stored the way its role says" means once the bytes are on
    // disk rather than in a pool.
    let stored: Vec<bool> = recipe_payloads(&back)
        .map(|payload| {
            matches!(
                payload.blob.stored_encoding(),
                Some(crate::blob::StoredEncoding::Zstd { .. })
            )
        })
        .collect();
    assert_eq!(stored, vec![true, false, false]);
}

/// Two payloads that share a digest bucket but not their bytes are two entries.
///
/// A real BLAKE3 collision cannot be constructed, so the bucket is made coarse
/// instead: every payload lands in one bucket, and only the comparison keeps
/// them apart. The digests written into the artifact are still each payload's
/// own, so what this produces is an ordinary artifact — one that decodes back
/// to the recipe it came from.
#[test]
fn payloads_sharing_a_digest_bucket_are_separated_by_their_bytes() {
    let recipe = every_role_holding(
        Blob::from_vec(b"one source, eight and forty bytes long ...".to_vec()),
        Blob::from_vec(b"an image of some other length entirely".to_vec()),
        Blob::from_vec(b"an image of some other length entirel!".to_vec()),
    );
    let one_bucket: Bucket = |_| [0; 32];
    let mut pool = Pool::from_payloads_bucketed(recipe_payloads(&recipe), one_bucket).unwrap();
    assert_eq!(
        pool.index.len(),
        1,
        "the coarse bucket must actually collide them, or this tests nothing"
    );
    assert_eq!(
        pool.blobs.len(),
        3,
        "three contents, so three entries — a bucket is not the bytes"
    );
    // Including the pair that differs in one byte at equal length, which no
    // length check would have separated.
    assert_ne!(pool.blobs[1].digest, pool.blobs[2].digest);
    assert_eq!(pool.blobs[1].decoded_len, pool.blobs[2].decoded_len);

    write_sections(&recipe, &mut pool).expect("all three entries are named");
}

/// Equal content in one role still shares an entry — the exactness above must
/// not have cost the deduplication the pool exists for.
#[test]
fn equal_content_in_one_role_is_still_one_entry() {
    let repeated = |bytes: &[u8]| ModuleImage::CudaLoadable {
        bytes: Blob::from_vec(bytes.to_vec()),
        persistence: ImagePersistence::Persistable,
        provenance: None,
    };
    let recipe = Recipe::stating_edges(
        RecipeSignature::empty(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            0,
            vec![KernelModule {
                code: KernelCode::ImageOnly {
                    preferred_image: repeated(SHARED),
                    alternative_images: vec![repeated(SHARED), repeated(b"different")],
                },
                resource_bounds: ResourceBounds::default(),
            }],
            Vec::new(),
            Vec::new(),
        ),
    )
    .unwrap();
    let pool = Pool::from_payloads(recipe_payloads(&recipe)).unwrap();
    assert_eq!(pool.blobs.len(), 2, "three images, two contents");
    let wire = crate::serialize(&recipe).unwrap();
    assert_eq!(crate::deserialize(&wire).unwrap(), recipe);
}
