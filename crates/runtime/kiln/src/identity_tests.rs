// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a program's identity must see, and what it must not.
//!
//! Every field a recipe carries either names the program or does not. One that
//! names it and is missing from the digest lets two programs share a cache
//! entry; one that does not name it and is in the digest makes a producer pay
//! for a compile it already has. Each test below pins one field to one side.

use recipe::payloads::PayloadRole;
use recipe::{
    Arg, BufRef, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, RecipeBuilder,
    RecipeSignature, Work,
};

use super::*;

/// One input and one output, which is what every recipe here launches over.
fn ports() -> RecipeSignature {
    RecipeSignature::new(vec![64], vec![64])
}

fn launch(entry: &str) -> Node {
    Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: entry.into(),
                args: vec![Arg::input(0), Arg::output(0)],
            },
            geometry: vec![Geometry {
                grid: [1, 1, 1],
                block: [64, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )
}

fn recipe(image: &[u8], entry: &str) -> Recipe {
    Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(image.to_vec())],
            Vec::new(),
            vec![launch(entry)],
        ),
    )
    .unwrap()
}

#[test]
fn identity_follows_program_content() {
    let base = RecipeIdentity::of(&recipe(b"image", "kernel")).unwrap();
    assert_eq!(
        base,
        RecipeIdentity::of(&recipe(b"image", "kernel")).unwrap()
    );
    assert_ne!(
        base,
        RecipeIdentity::of(&recipe(b"other", "kernel")).unwrap()
    );
    assert_ne!(
        base,
        RecipeIdentity::of(&recipe(b"image", "other")).unwrap()
    );
}

/// Identity is a function of the program, and a program does not remember
/// which order its edges were found in or how a producer cut up an
/// argument's bytes. Two producers reaching the same program by different
/// routes must key one cache entry, or every route pays for its own.
#[test]
fn identity_does_not_see_how_a_producer_stated_the_program() {
    let program = |deps: Vec<usize>, segments: Vec<Segment>| {
        Recipe::stating_edges(
            ports(),
            CompileRequests::default(),
            RecipeBody::new(
                1,
                1,
                vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
                Vec::new(),
                vec![
                    launch("first"),
                    Node::new(
                        Work::Call {
                            call: KernelLaunch {
                                module: 0,
                                entry: "second".into(),
                                args: vec![Arg::from_segments(segments)],
                            },
                            geometry: vec![Geometry {
                                grid: [1, 1, 1],
                                block: [64, 1, 1],
                                dynamic_shmem: 0,
                            }],
                        },
                        deps,
                    ),
                ],
            ),
        )
        .unwrap()
    };
    let stated = program(vec![0], vec![Segment::Bytes(vec![7, 8])]);
    let restated = program(
        vec![0, 0],
        vec![
            Segment::Bytes(Vec::new()),
            Segment::Bytes(vec![7]),
            Segment::Bytes(vec![8]),
        ],
    );
    assert_eq!(
        RecipeIdentity::of(&stated).unwrap(),
        RecipeIdentity::of(&restated).unwrap()
    );
    assert_ne!(
        RecipeIdentity::of(&stated).unwrap(),
        RecipeIdentity::of(&program(Vec::new(), vec![Segment::Bytes(vec![7, 8])])).unwrap(),
        "dropping the edge is a different program, not a different spelling"
    );
    assert_ne!(
        RecipeIdentity::of(&stated).unwrap(),
        RecipeIdentity::of(&program(vec![0], vec![Segment::Bytes(vec![8, 7])])).unwrap(),
        "different bytes are a different program"
    );
}

/// Every payload a recipe carries reaches the digest.
///
/// The list of places a payload can be is stated once, in
/// [`recipe::payloads`], and this walks that list rather than a copy of it: a
/// payload field added there and not hashed here fails this test instead of
/// silently letting two programs share one identity.
#[test]
fn identity_covers_every_payload_the_recipe_carries() {
    let carrier = || {
        let mut request = CompileRequest::unguarded(
            vec![recipe::SourceImage {
                kind: recipe::SourceKind::Ptx,
                target: recipe::PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(b".version 8.8\n".to_vec()),
            }],
            PtxCompileOptions::default(),
        );
        request.options.max_registers = Some(64);
        Recipe::stating_edges(
            ports(),
            CompileRequests::new(vec![request]),
            RecipeBody::new(
                1,
                1,
                vec![
                    KernelModule::from_request(
                        recipe::CompileRequestId(0),
                        vec![ModuleImage::Cubin {
                            bytes: Blob::from_vec(b"cubin".to_vec()),
                            target_sm: 100,
                            provenance: None,
                        }],
                        ResourceBounds::default(),
                    ),
                    KernelModule::from_cuda_loadable(b"loadable".to_vec()),
                ],
                vec![recipe::WorkspaceBuf {
                    nbytes: 8,
                    init: Some(Blob::from_vec(b"seedseed".to_vec())),
                }],
                vec![Node::new(
                    Work::Choice {
                        candidates: vec![RecipeBody::new(
                            1,
                            1,
                            vec![KernelModule::from_cuda_loadable(b"candidate".to_vec())],
                            vec![recipe::WorkspaceBuf {
                                nbytes: 4,
                                init: Some(Blob::from_vec(b"seed".to_vec())),
                            }],
                            vec![launch("inner")],
                        )],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    [],
                )],
            ),
        )
        .unwrap()
    };
    let roles: Vec<PayloadRole> = recipe::payloads::recipe_payloads(&carrier())
        .map(|payload| payload.role)
        .collect();
    assert_eq!(
        roles,
        vec![
            PayloadRole::CompileSource,
            PayloadRole::ModuleImage,
            PayloadRole::ModuleImage,
            PayloadRole::WorkspaceInit,
            PayloadRole::ModuleImage,
            PayloadRole::WorkspaceInit,
        ],
        "the fixture states one payload per place, of every role"
    );

    let base = RecipeIdentity::of(&carrier()).unwrap();
    for nth in 0..roles.len() {
        let mut altered = carrier();
        let mut seen = 0;
        recipe::payloads::visit_recipe_payloads(&mut altered, &mut |payload| {
            if seen == nth {
                let mut bytes = payload.blob.to_vec();
                bytes.push(0xFF);
                *payload.blob = Blob::from_vec(bytes);
            }
            seen += 1;
        });
        assert_ne!(
            base,
            RecipeIdentity::of(&altered).unwrap(),
            "payload {nth} does not reach the digest"
        );
    }
}

/// Two programs differing in exactly one payload, of the named role, and
/// nothing else.
fn differing_in(role: PayloadRole) -> (Recipe, Recipe) {
    let build = |bytes: &[u8]| {
        let source = |text: &[u8]| recipe::SourceImage {
            kind: recipe::SourceKind::Ptx,
            target: recipe::PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(text.to_vec()),
        };
        let vary = |mine: PayloadRole, fixed: &'static [u8]| {
            if mine == role { bytes } else { fixed }
        };
        Recipe::stating_edges(
            ports(),
            CompileRequests::new(vec![CompileRequest::unguarded(
                vec![source(vary(PayloadRole::CompileSource, b".version 8.8\n"))],
                PtxCompileOptions::default(),
            )]),
            RecipeBody::new(
                1,
                1,
                vec![KernelModule::from_cuda_loadable(
                    vary(PayloadRole::ModuleImage, b"image").to_vec(),
                )],
                vec![recipe::WorkspaceBuf {
                    nbytes: 8,
                    init: Some(Blob::from_vec(
                        vary(PayloadRole::WorkspaceInit, b"seedseed").to_vec(),
                    )),
                }],
                vec![launch("kernel")],
            ),
        )
        .unwrap()
    };
    (build(b"aaaaaaaa"), build(b"bbbbbbbb"))
}

/// An identity built on a walk that misses a role stops telling programs apart.
///
/// This is the coupling stated as the thing it prevents. `of` reads the one
/// walk, so it separates the two recipes below; the same digest over a walk
/// with that role filtered out gives them one identity — which is a cache
/// handing a caller the wrong realization, with nothing at the point of the
/// omission to say so.
#[test]
fn an_identity_over_a_walk_that_omits_a_role_stops_telling_programs_apart() {
    for role in [
        PayloadRole::CompileSource,
        PayloadRole::ModuleImage,
        PayloadRole::WorkspaceInit,
    ] {
        let (one, other) = differing_in(role);
        assert_ne!(
            RecipeIdentity::of(&one).unwrap(),
            RecipeIdentity::of(&other).unwrap(),
            "the one walk reaches {role:?}, so these are two programs"
        );
        let without = |recipe: &Recipe| {
            RecipeIdentity::over(
                recipe,
                recipe::payloads::recipe_payloads(recipe).filter(|payload| payload.role != role),
            )
            .unwrap()
        };
        assert_eq!(
            without(&one),
            without(&other),
            "a walk that missed {role:?} would have to collide here, or this \
             test is not watching the coupling it claims to"
        );
    }
}

/// The digest counts the payloads the walk produced, so a walk that stopped
/// reaching a role would change every identity rather than quietly agreeing
/// with one that still reached it.
///
/// This is the coupling itself, not a consequence of it: two recipes that
/// differ only in a payload the walk omitted would key one cache entry, and
/// nothing at the point of the omission would say so.
#[test]
fn identity_is_the_walks_payloads_and_only_those() {
    let with_seed = |init: Option<&[u8]>| {
        Recipe::stating_edges(
            ports(),
            CompileRequests::default(),
            RecipeBody::new(
                1,
                1,
                vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
                vec![recipe::WorkspaceBuf {
                    nbytes: 8,
                    init: init.map(|bytes| Blob::from_vec(bytes.to_vec())),
                }],
                vec![launch("kernel")],
            ),
        )
        .unwrap()
    };
    // The role that a structure-only pass would most easily drop: the bytes
    // live behind an `Option` the structure already reports the presence of.
    assert_ne!(
        RecipeIdentity::of(&with_seed(Some(b"seedseed"))).unwrap(),
        RecipeIdentity::of(&with_seed(Some(b"otherpad"))).unwrap(),
        "two seeds of one length are two programs"
    );
    assert_ne!(
        RecipeIdentity::of(&with_seed(Some(b"seedseed"))).unwrap(),
        RecipeIdentity::of(&with_seed(None)).unwrap(),
        "a seeded buffer and a bare one are two programs"
    );

    // Bytes serving two purposes are two programs, which is what the role tag
    // in the payload framing buys.
    let as_image = Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(b"abcdefgh".to_vec())],
            vec![recipe::WorkspaceBuf {
                nbytes: 8,
                init: None,
            }],
            vec![launch("kernel")],
        ),
    )
    .unwrap();
    let as_seed = Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(b"abcdefgh".to_vec())],
            vec![recipe::WorkspaceBuf {
                nbytes: 8,
                init: Some(Blob::from_vec(b"abcdefgh".to_vec())),
            }],
            vec![launch("kernel")],
        ),
    )
    .unwrap();
    assert_ne!(
        RecipeIdentity::of(&as_image).unwrap(),
        RecipeIdentity::of(&as_seed).unwrap()
    );
}

#[test]
fn identity_survives_reencoding_the_same_bytes() {
    let spilled = Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(
                Blob::zstd_spilled(b"image").unwrap(),
            )],
            Vec::new(),
            vec![launch("kernel")],
        ),
    )
    .unwrap();
    assert_eq!(
        RecipeIdentity::of(&spilled).unwrap(),
        RecipeIdentity::of(&recipe(b"image", "kernel")).unwrap(),
        "a zstd blob and a raw blob over the same bytes are the same program"
    );
}

#[test]
fn identity_separates_adjacent_fields() {
    // Two names that concatenate identically must not hash identically.
    let named = |entry: &str| {
        let mut builder = <RecipeBuilder>::new(1, 1);
        builder.module_definition(KernelModule::from_cuda_loadable(b"i".to_vec()));
        builder.land(launch(entry).work().clone(), Vec::new());
        builder.finish(ports()).unwrap().into_recipe()
    };
    assert_ne!(
        RecipeIdentity::of(&named("ab")).unwrap(),
        RecipeIdentity::of(&named("a")).unwrap()
    );
}

#[test]
fn identity_covers_choice_candidates() {
    let candidate = |entry: &str| {
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
            Vec::new(),
            vec![launch(entry)],
        )
    };
    let choice = |second: &str| {
        Recipe::stating_edges(
            ports(),
            CompileRequests::default(),
            RecipeBody::new(
                1,
                1,
                Vec::new(),
                Vec::new(),
                vec![Node::new(
                    Work::Choice {
                        candidates: vec![candidate("first"), candidate(second)],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    [],
                )],
            ),
        )
        .unwrap()
    };
    assert_ne!(
        RecipeIdentity::of(&choice("second")).unwrap(),
        RecipeIdentity::of(&choice("third")).unwrap(),
        "a losing candidate is still part of the program the tuner explores"
    );
}

fn bounded_module(bounds: ResourceBounds) -> Recipe {
    Recipe::stating_edges(
        ports(),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule {
                code: KernelCode::ImageOnly {
                    preferred_image: ModuleImage::CudaLoadable {
                        bytes: Blob::from_vec(b"image".to_vec()),
                        persistence: recipe::ImagePersistence::Persistable,
                        provenance: None,
                    },
                    alternative_images: Vec::new(),
                },
                resource_bounds: bounds,
            }],
            Vec::new(),
            vec![launch("kernel")],
        ),
    )
    .unwrap()
}

fn bounded_prerequisite(bounds: ResourceBounds) -> Recipe {
    let source = recipe::SourceImage {
        kind: recipe::SourceKind::Ptx,
        target: recipe::PtxTarget {
            isa: [8, 8],
            minimum_sm: 100,
        },
        blob: Blob::from_vec(b"ptx".to_vec()),
    };
    let mut guarded = CompileRequest::unguarded(vec![source.clone()], PtxCompileOptions::default());
    guarded.require(recipe::CompileRequestId(0), bounds);
    let (ports, _table, body) = recipe(b"image", "kernel").into_parts();
    Recipe::stating_edges(
        ports,
        CompileRequests::new(vec![
            CompileRequest::unguarded(vec![source], PtxCompileOptions::default()),
            guarded,
        ]),
        body,
    )
    .unwrap()
}

/// A register floor is what removes a spilling register-capped candidate,
/// so two programs whose modules admit different floors are two programs.
#[test]
fn identity_covers_a_modules_register_floor() {
    let floor = |min_registers| {
        RecipeIdentity::of(&bounded_module(ResourceBounds {
            min_registers,
            max_local_bytes: None,
        }))
        .unwrap()
    };
    assert_eq!(floor(Some(64)), floor(Some(64)));
    assert_ne!(floor(Some(64)), floor(Some(128)));
    assert_ne!(floor(Some(64)), floor(None));
}

#[test]
fn identity_covers_a_prerequisites_register_floor() {
    let floor = |min_registers| {
        RecipeIdentity::of(&bounded_prerequisite(ResourceBounds {
            min_registers,
            max_local_bytes: None,
        }))
        .unwrap()
    };
    assert_eq!(floor(Some(64)), floor(Some(64)));
    assert_ne!(floor(Some(64)), floor(Some(128)));
    assert_ne!(floor(Some(64)), floor(None));
}

/// The two bounds are different questions about a kernel, so the same
/// number in each is two different programs — which is what framing each
/// field in declaration order buys.
#[test]
fn a_register_floor_is_not_a_local_bound() {
    let bounds = |min_registers, max_local_bytes| {
        RecipeIdentity::of(&bounded_module(ResourceBounds {
            min_registers,
            max_local_bytes,
        }))
        .unwrap()
    };
    assert_ne!(bounds(Some(64), None), bounds(None, Some(64)));
    assert_ne!(bounds(Some(64), Some(64)), bounds(None, Some(64)));
    assert_ne!(bounds(Some(64), Some(64)), bounds(Some(64), None));
}

/// The ports are program content. Two recipes agreeing everywhere else and
/// differing in one port extent are two programs, so an artifact realized
/// for one is never reused for the other.
#[test]
fn identity_covers_the_external_ports() {
    // The work is the same work whatever ports the recipe states, so what
    // differs below is the ports and nothing else.
    let over = |signature: RecipeSignature| {
        let (inputs, outputs) = (signature.num_inputs(), signature.num_outputs());
        RecipeIdentity::of(
            &<Recipe>::stating_edges(
                signature,
                CompileRequests::default(),
                RecipeBody::new(
                    inputs,
                    outputs,
                    vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
                    Vec::new(),
                    Vec::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let base = over(RecipeSignature::new(vec![16], vec![32]));
    assert_eq!(base, over(RecipeSignature::new(vec![16], vec![32])));
    assert_ne!(base, over(RecipeSignature::new(vec![17], vec![32])));
    assert_ne!(base, over(RecipeSignature::new(vec![16], vec![33])));
    assert_ne!(
        base,
        over(RecipeSignature::new(vec![32], vec![16])),
        "the two sides are not one multiset"
    );
    assert_ne!(
        base,
        over(RecipeSignature::new(vec![16, 16], vec![32])),
        "a port count is content"
    );
    assert_ne!(base, over(RecipeSignature::empty()));
}

/// Where an operation starts is as much of what it does as which buffer it
/// names, so two recipes that differ only in a byte offset are two
/// programs. A digest that skipped an offset would let a tuning result
/// measured against one be replayed as the other.
///
/// Each case below changes exactly one offset from a common base, so a
/// separation that held for an unrelated reason would not show here.
#[test]
fn a_byte_offset_separates_two_programs() {
    let body = |source: usize, destination: usize, fill: usize, pointer: usize| -> RecipeBody {
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(b"image".to_vec())],
            vec![recipe::WorkspaceBuf {
                nbytes: 64,
                init: None,
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: recipe::WritableBuf::Workspace(0).at(fill),
                        value: 0,
                        nbytes: 8,
                    },
                    [],
                ),
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Input(0).at(source),
                        destination: recipe::WritableBuf::Output(0).at(destination),
                        nbytes: 8,
                    },
                    [0],
                ),
                Node::new(
                    Work::Call {
                        call: KernelLaunch {
                            module: 0,
                            entry: "k".into(),
                            args: vec![Arg::pointer(BufRef::Output(0).at(pointer))],
                        },
                        geometry: vec![Geometry {
                            grid: [1, 1, 1],
                            block: [64, 1, 1],
                            dynamic_shmem: 0,
                        }],
                    },
                    [1],
                ),
            ],
        )
    };
    let identity = |source, destination, fill, pointer| {
        RecipeIdentity::of(
            &<Recipe>::stating_edges(
                ports(),
                CompileRequests::default(),
                body(source, destination, fill, pointer),
            )
            .unwrap(),
        )
        .unwrap()
    };

    let base = identity(0, 0, 0, 0);
    assert_eq!(
        base,
        identity(0, 0, 0, 0),
        "the same program is one program"
    );
    for (what, moved) in [
        ("memcpy source", identity(8, 0, 0, 0)),
        ("memcpy destination", identity(0, 8, 0, 0)),
        ("memset destination", identity(0, 0, 8, 0)),
        ("kernel pointer", identity(0, 0, 0, 8)),
    ] {
        assert_ne!(base, moved, "{what} moved and the identity did not");
    }

    // Every one of the four is separate from every other, so no two
    // offsets are being hashed into the same place.
    let moved = [
        identity(8, 0, 0, 0),
        identity(0, 8, 0, 0),
        identity(0, 0, 8, 0),
        identity(0, 0, 0, 8),
    ];
    for (i, a) in moved.iter().enumerate() {
        for b in &moved[i + 1..] {
            assert_ne!(a, b, "two different offsets share one identity");
        }
    }
}

/// A copy out of an input is a program a copy out of the workspace is not.
#[test]
fn a_copy_source_capability_separates_two_programs() {
    let copies_from = |source: BufRef| {
        RecipeIdentity::of(
            &<Recipe>::stating_edges(
                ports(),
                CompileRequests::default(),
                RecipeBody::new(
                    1,
                    1,
                    Vec::new(),
                    vec![recipe::WorkspaceBuf {
                        nbytes: 64,
                        init: None,
                    }],
                    vec![Node::new(
                        Work::Memcpy {
                            source: source.at(0),
                            destination: recipe::WritableBuf::Output(0).at(0),
                            nbytes: 8,
                        },
                        [],
                    )],
                ),
            )
            .unwrap(),
        )
        .unwrap()
    };
    assert_ne!(
        copies_from(BufRef::Input(0)),
        copies_from(BufRef::Workspace(0))
    );
}

/// Everything a module's code states is part of what the program is: a
/// program that would load different bytes, or the same bytes under a
/// different rule, is not the program a realization was measured for.
#[test]
fn identity_covers_every_field_of_a_modules_code() {
    // Two compiles in the table, so a module may name either one and neither
    // spelling is a request the recipe does not hold.
    let table = || {
        let source = |bytes: &[u8]| recipe::SourceImage {
            kind: recipe::SourceKind::Ptx,
            target: recipe::PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(bytes.to_vec()),
        };
        CompileRequests::new(vec![
            CompileRequest::unguarded(vec![source(b"first")], PtxCompileOptions::default()),
            CompileRequest::unguarded(vec![source(b"second")], PtxCompileOptions::default()),
        ])
    };
    let coded = |code: KernelCode| {
        RecipeIdentity::of(
            &<Recipe>::stating_edges(
                ports(),
                table(),
                RecipeBody::new(
                    1,
                    1,
                    vec![KernelModule {
                        code,
                        resource_bounds: ResourceBounds::default(),
                    }],
                    Vec::new(),
                    vec![launch("kernel")],
                ),
            )
            .unwrap(),
        )
        .unwrap()
    };
    let cubin = |target_sm, byte| ModuleImage::Cubin {
        target_sm,
        bytes: Blob::from_vec(vec![byte]),
        provenance: None,
    };
    let loadable = |persistence| ModuleImage::CudaLoadable {
        bytes: Blob::from_vec(vec![0xAA]),
        persistence,
        provenance: None,
    };

    let base = KernelCode::ImageOnly {
        preferred_image: cubin(100, 1),
        alternative_images: vec![cubin(90, 2)],
    };
    assert_eq!(coded(base.clone()), coded(base.clone()));

    // Which architecture an image is for.
    assert_ne!(
        coded(base.clone()),
        coded(KernelCode::ImageOnly {
            preferred_image: cubin(101, 1),
            alternative_images: vec![cubin(90, 2)],
        })
    );
    // Which image is preferred. The set is the same; the order is not, and
    // the order is what decides which one this machine loads.
    assert_ne!(
        coded(base.clone()),
        coded(KernelCode::ImageOnly {
            preferred_image: cubin(90, 2),
            alternative_images: vec![cubin(100, 1)],
        }),
        "reordering two images changes which one runs"
    );
    // How many images there are.
    assert_ne!(
        coded(base.clone()),
        coded(KernelCode::ImageOnly {
            preferred_image: cubin(100, 1),
            alternative_images: Vec::new(),
        })
    );
    // Whether an image outlives its session, which decides what an
    // artifact may carry.
    assert_ne!(
        coded(KernelCode::ImageOnly {
            preferred_image: loadable(recipe::ImagePersistence::Persistable),
            alternative_images: Vec::new(),
        }),
        coded(KernelCode::ImageOnly {
            preferred_image: loadable(recipe::ImagePersistence::SessionOnly),
            alternative_images: Vec::new(),
        })
    );
    // Whether there is a compile behind the images at all, and which one.
    assert_ne!(
        coded(KernelCode::ImageOnly {
            preferred_image: cubin(100, 1),
            alternative_images: Vec::new(),
        }),
        coded(KernelCode::PtxBacked {
            compile_request: recipe::CompileRequestId(0),
            images: vec![cubin(100, 1)],
        })
    );
    assert_ne!(
        coded(KernelCode::PtxBacked {
            compile_request: recipe::CompileRequestId(0),
            images: Vec::new(),
        }),
        coded(KernelCode::PtxBacked {
            compile_request: recipe::CompileRequestId(1),
            images: Vec::new(),
        })
    );
}

/// How the PTX is to be assembled decides which assembler runs and so
/// which image a realization produces.
#[test]
fn identity_covers_the_assembly_mode() {
    let assembled = |assembly_mode| {
        let (ports, table, body) = recipe(b"image", "kernel").into_parts();
        let mut table = table.into_vec();
        table.push(CompileRequest::unguarded(
            vec![recipe::SourceImage {
                kind: recipe::SourceKind::Ptx,
                target: recipe::PtxTarget {
                    isa: [8, 8],
                    minimum_sm: 100,
                },
                blob: Blob::from_vec(b"ptx".to_vec()),
            }],
            PtxCompileOptions {
                max_registers: None,
                assembly_mode,
            },
        ));
        RecipeIdentity::of(
            &<Recipe>::stating_edges(ports, CompileRequests::new(table), body).unwrap(),
        )
        .unwrap()
    };
    let default = assembled(recipe::PtxAssemblyMode::EnvironmentDefault);
    assert_eq!(
        default,
        assembled(recipe::PtxAssemblyMode::EnvironmentDefault)
    );
    assert_ne!(default, assembled(recipe::PtxAssemblyMode::Relocatable));
    assert_ne!(default, assembled(recipe::PtxAssemblyMode::WholeProgram));
    assert_ne!(
        assembled(recipe::PtxAssemblyMode::Relocatable),
        assembled(recipe::PtxAssemblyMode::WholeProgram)
    );
}

/// Which request a prerequisite is about, not only what it demands.
#[test]
fn identity_covers_which_request_a_prerequisite_names() {
    let guarding = |request: usize| {
        let (ports, table, body) = recipe(b"image", "kernel").into_parts();
        let mut table = table.into_vec();
        let source = recipe::SourceImage {
            kind: recipe::SourceKind::Ptx,
            target: recipe::PtxTarget {
                isa: [8, 8],
                minimum_sm: 100,
            },
            blob: Blob::from_vec(b"ptx".to_vec()),
        };
        // Two entries a guard can be about, so both spellings name a request
        // the table holds and neither is a guard about itself.
        table.push(CompileRequest::unguarded(
            vec![source.clone()],
            PtxCompileOptions::default(),
        ));
        table.push(CompileRequest::unguarded(
            vec![source.clone(), source.clone()],
            PtxCompileOptions::default(),
        ));
        let mut guarded = CompileRequest::unguarded(vec![source], PtxCompileOptions::default());
        guarded.require(
            recipe::CompileRequestId(request),
            ResourceBounds {
                min_registers: Some(64),
                max_local_bytes: None,
            },
        );
        table.push(guarded);
        RecipeIdentity::of(
            &<Recipe>::stating_edges(ports, CompileRequests::new(table), body).unwrap(),
        )
        .unwrap()
    };
    assert_eq!(guarding(0), guarding(0));
    assert_ne!(guarding(0), guarding(1));
}

/// A compile unit is a sequence, and its identity is that sequence.
///
/// Two programs that link the same modules the other way round resolve their
/// symbols against different definitions, and one image is not the other's.
/// A digest over the set — or over only the first source — would hand the
/// second program the first one's realized artifact.
#[test]
fn a_compile_unit_is_identified_in_link_order() {
    let source = |ptx: &[u8], minimum_sm| recipe::SourceImage {
        kind: recipe::SourceKind::Ptx,
        target: recipe::PtxTarget {
            isa: [8, 8],
            minimum_sm,
        },
        blob: Blob::from_vec(ptx.to_vec()),
    };
    let program = |sources: Vec<recipe::SourceImage>| {
        Recipe::stating_edges(
            ports(),
            CompileRequests::new(vec![CompileRequest::unguarded(
                sources,
                PtxCompileOptions::default(),
            )]),
            RecipeBody::new(
                1,
                1,
                vec![KernelModule::from_request(
                    recipe::CompileRequestId(0),
                    Vec::new(),
                    recipe::ResourceBounds::default(),
                )],
                Vec::new(),
                vec![launch("caller")],
            ),
        )
        .unwrap()
    };
    let caller = source(b"caller", 100);
    let callee = source(b"callee", 90);
    let identity = |recipe: &Recipe| RecipeIdentity::of(recipe).unwrap();

    let forward = program(vec![caller.clone(), callee.clone()]);
    assert_eq!(
        identity(&forward),
        identity(&program(vec![caller.clone(), callee.clone()]))
    );
    assert_ne!(
        identity(&forward),
        identity(&program(vec![callee.clone(), caller.clone()])),
        "link order names the program"
    );
    assert_ne!(
        identity(&forward),
        identity(&program(vec![caller.clone()])),
        "dropping a module names another program"
    );
    assert_ne!(
        identity(&forward),
        identity(&program(vec![caller, source(b"callee", 100)])),
        "a source's own target names the program"
    );
}
