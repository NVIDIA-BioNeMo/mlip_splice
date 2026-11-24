// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the code model states without anyone checking: that a module has
//! somewhere to get its kernels from, which image is preferred, and whether an
//! image is worth keeping.

use super::*;

fn cubin(sm: u16, byte: u8) -> ModuleImage {
    ModuleImage::Cubin {
        target_sm: sm,
        bytes: Blob::from_vec(vec![byte]),
        provenance: None,
    }
}

fn loadable(byte: u8, persistence: ImagePersistence) -> ModuleImage {
    ModuleImage::CudaLoadable {
        bytes: Blob::from_vec(vec![byte]),
        persistence,
        provenance: None,
    }
}

/// The order images are offered in is the order they were stated in, and the
/// preferred one of an image-only module comes first however many alternatives
/// follow it.
#[test]
fn images_are_offered_most_preferred_first() {
    let ptx_backed = KernelCode::PtxBacked {
        compile_request: CompileRequestId(3),
        images: vec![cubin(100, 1), cubin(90, 2)],
    };
    assert_eq!(
        ptx_backed.images().collect::<Vec<_>>(),
        vec![&cubin(100, 1), &cubin(90, 2)]
    );
    assert_eq!(ptx_backed.compile_request(), Some(CompileRequestId(3)));

    let image_only = KernelCode::ImageOnly {
        preferred_image: cubin(100, 1),
        alternative_images: vec![cubin(90, 2), loadable(3, ImagePersistence::Persistable)],
    };
    assert_eq!(
        image_only.images().collect::<Vec<_>>(),
        vec![
            &cubin(100, 1),
            &cubin(90, 2),
            &loadable(3, ImagePersistence::Persistable)
        ]
    );
    assert_eq!(image_only.compile_request(), None);
}

/// A module that must be compiled offers no image, which is a shape the type
/// admits — unlike a module with neither a compile nor an image, which it does
/// not.
#[test]
fn a_compile_with_no_image_is_a_module_that_must_be_compiled() {
    let code = KernelCode::PtxBacked {
        compile_request: CompileRequestId(0),
        images: Vec::new(),
    };
    assert_eq!(code.images().count(), 0);
    assert_eq!(code.compile_request(), Some(CompileRequestId(0)));
    assert_eq!(code.sole_image(), None);
}

/// One image and nothing behind it is what a loader can act on without
/// choosing. Anything a caller would have to choose between, or that might yet
/// be compiled, is not.
#[test]
fn only_one_image_with_no_compile_behind_it_is_sole() {
    assert_eq!(
        KernelModule::from_cuda_loadable(vec![0xAA]).sole_image(),
        Some(&loadable(0xAA, ImagePersistence::Persistable))
    );
    assert_eq!(
        KernelCode::ImageOnly {
            preferred_image: cubin(100, 1),
            alternative_images: vec![cubin(90, 2)],
        }
        .sole_image(),
        None,
        "two images are a choice, not an answer"
    );
    assert_eq!(
        KernelCode::PtxBacked {
            compile_request: CompileRequestId(0),
            images: vec![cubin(100, 1)],
        }
        .sole_image(),
        None,
        "an image that may still lose to a compile is not the answer either"
    );
}

/// Whether bytes are worth keeping is the image's own statement. A cubin
/// always is; a loadable image says which it is, and nothing reads an
/// assembler's name to find out.
#[test]
fn an_image_states_whether_it_outlives_its_session() {
    assert!(cubin(100, 1).is_persistable());
    assert!(loadable(1, ImagePersistence::Persistable).is_persistable());
    assert!(!loadable(1, ImagePersistence::SessionOnly).is_persistable());
}

/// A producer handing over loadable bytes is handing over bytes that keep, and
/// the module it gets is the module a `Recipe` can be written from.
#[test]
fn a_handed_image_is_one_persistable_image_and_no_compile() {
    let module = KernelModule::from_cuda_loadable(vec![1, 2, 3]);
    assert_eq!(module.resource_bounds, ResourceBounds::default());
    assert_eq!(module.code.compile_request(), None);
    let [image] = module.code.images().collect::<Vec<_>>()[..] else {
        panic!("one image");
    };
    assert!(image.is_persistable());
    assert_eq!(image.bytes().as_ref(), [1, 2, 3]);
}

/// A prerequisite is a statement about another request, so two statements
/// about the same one are the same entry rather than two the reader has to
/// reconcile.
#[test]
fn a_prerequisite_is_keyed_by_the_request_it_is_about() {
    let mut request = CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(b"ptx".to_vec()),
        }],
        PtxCompileOptions::default(),
    );
    assert!(request.requires.is_empty());

    let floor = ResourceBounds {
        min_registers: Some(64),
        ..ResourceBounds::default()
    };
    let ceiling = ResourceBounds {
        max_local_bytes: Some(0),
        ..ResourceBounds::default()
    };
    request.require(CompileRequestId(1), floor);
    request.require(CompileRequestId(1), ceiling);
    assert_eq!(request.requires.len(), 1);
    assert_eq!(request.requires[&CompileRequestId(1)], floor.and(ceiling));
}

/// Two guards about one request are one guard that says both. A floor
/// conjoins upward and a ceiling downward, and an absent bound constrains
/// nothing — so it is the identity for its own coordinate, not a floor of zero
/// or a ceiling of none.
#[test]
fn conjoined_bounds_are_the_tighter_of_each() {
    let floor = |min_registers| ResourceBounds {
        min_registers,
        max_local_bytes: None,
    };
    let ceiling = |max_local_bytes| ResourceBounds {
        min_registers: None,
        max_local_bytes,
    };

    assert_eq!(
        floor(Some(64)).and(ceiling(Some(32))),
        ResourceBounds {
            min_registers: Some(64),
            max_local_bytes: Some(32),
        },
        "complementary bounds keep both"
    );
    assert_eq!(floor(Some(64)).and(floor(Some(128))), floor(Some(128)));
    assert_eq!(floor(Some(128)).and(floor(Some(64))), floor(Some(128)));
    assert_eq!(ceiling(Some(64)).and(ceiling(Some(0))), ceiling(Some(0)));
    assert_eq!(ceiling(Some(0)).and(ceiling(Some(64))), ceiling(Some(0)));

    let stated = ResourceBounds {
        min_registers: Some(64),
        max_local_bytes: Some(32),
    };
    assert_eq!(
        stated.and(ResourceBounds::default()),
        stated,
        "nothing conjoined with something is that something"
    );
    assert_eq!(ResourceBounds::default().and(stated), stated);
    assert_eq!(stated.and(stated), stated, "and it is idempotent");
    assert_eq!(
        floor(Some(64)).and(ceiling(Some(32))),
        ceiling(Some(32)).and(floor(Some(64))),
        "and it does not depend on which came first"
    );
}

/// Adding a guard keeps the one already there, so a request cannot be
/// weakened by stating a second fact about the same prerequisite.
#[test]
fn requiring_the_same_request_twice_keeps_both_facts() {
    let mut request = CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(b"ptx".to_vec()),
        }],
        PtxCompileOptions::default(),
    );
    request.require(
        CompileRequestId(1),
        ResourceBounds {
            min_registers: Some(64),
            max_local_bytes: None,
        },
    );
    request.require(
        CompileRequestId(1),
        ResourceBounds {
            min_registers: None,
            max_local_bytes: Some(0),
        },
    );
    assert_eq!(request.requires.len(), 1);
    assert_eq!(
        request.requires[&CompileRequestId(1)],
        ResourceBounds {
            min_registers: Some(64),
            max_local_bytes: Some(0),
        }
    );
}

/// The collector every renumbering path uses conjoins collisions and keeps the
/// map's one entry per request.
#[test]
fn conjoined_requires_merges_what_lands_on_one_request() {
    let collected = conjoined_requires([
        (
            CompileRequestId(2),
            ResourceBounds {
                min_registers: Some(129),
                max_local_bytes: None,
            },
        ),
        (
            CompileRequestId(2),
            ResourceBounds {
                min_registers: None,
                max_local_bytes: Some(0),
            },
        ),
        (
            CompileRequestId(1),
            ResourceBounds {
                min_registers: Some(32),
                max_local_bytes: None,
            },
        ),
    ]);
    assert_eq!(
        collected.keys().copied().collect::<Vec<_>>(),
        vec![CompileRequestId(1), CompileRequestId(2)]
    );
    assert_eq!(
        collected[&CompileRequestId(2)],
        ResourceBounds {
            min_registers: Some(129),
            max_local_bytes: Some(0),
        }
    );
}
