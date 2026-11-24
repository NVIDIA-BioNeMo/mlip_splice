// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One fixed recipe, its recipe-v13 bytes, and its kiln identity.
//!
//! Both are protocol: an artifact written by one build is read by another, and
//! a realization cached under an identity is handed back to whatever computes
//! the same one. Neither may move without a version, so both are pinned here to
//! the values this format has always produced rather than to whatever the
//! current code happens to compute.
//!
//! The fixture reaches every field that carries into either: two ports on one
//! side and one on the other, a compile table with a prerequisite between its
//! entries, a module carrying an image, a seeded workspace, a memset, a copy,
//! and a choice whose candidates carry their own modules, workspace and launch
//! geometry.

use recipe::{
    Arg, Blob, BufRef, CompileRequest, CompileRequestId, CompileRequests, Geometry,
    ImagePersistence, KernelLaunch, KernelModule, ModuleImage, Node, PtxAssemblyMode,
    PtxCompileOptions, PtxTarget, Recipe, RecipeBody, RecipeSignature, ResourceBounds, SourceImage,
    SourceKind, Work, WorkspaceBuf, WritableBuf,
};

/// What this recipe encodes to, and what it is identified by.
const WIRE_LEN: usize = 1103;
const WIRE_DIGEST: &str = "6909c72ac3d0028a772fa2f8fdbeb38026e6915e4ffc59f6eeb6aaf3a1a24c29";
const IDENTITY: &str = "9962db0354a939a171a97586b5cdc18b68d6fc1e1f5d45a5c031760c9ac85e0b";

/// The same, for the looped landing below.
const SPLICED_LEN: usize = 588;
const SPLICED_WIRE: &str = "9c19e3b1137a7cd9ed847b167722f7c07207c9ad3b9eff4618bd52b42a7882df";
const SPLICED_IDENTITY: &str = "34810032c8d69244d78f62eb2c36712a369ac02f7faa517351f624acab206c1f";

fn candidate(entry: &str, seed: u8) -> RecipeBody {
    RecipeBody::new(
        1,
        1,
        vec![KernelModule::from_request(
            CompileRequestId(0),
            vec![ModuleImage::Cubin {
                target_sm: 100,
                bytes: Blob::from_vec(vec![seed, seed + 1]),
                provenance: None,
            }],
            ResourceBounds {
                min_registers: Some(32),
                max_local_bytes: Some(64),
            },
        )],
        vec![WorkspaceBuf {
            nbytes: 32,
            init: Some(Blob::from_vec(vec![seed; 32])),
        }],
        vec![Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    args: vec![
                        Arg::input(0),
                        Arg::workspace(0),
                        Arg::output(0),
                        Arg::scalar_i32(-7),
                    ],
                },
                geometry: vec![Geometry::new([4, 2, 1], [64, 1, 1], 48)],
            },
            [],
        )],
    )
}

fn fixture() -> Recipe {
    let source = SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget {
            isa: [8, 8],
            minimum_sm: 100,
        },
        blob: Blob::from_vec(b".version 8.8\n.target sm_100\n".to_vec()),
    };
    let mut guarded = CompileRequest::unguarded(
        vec![source.clone()],
        PtxCompileOptions {
            max_registers: Some(128),
            assembly_mode: PtxAssemblyMode::WholeProgram,
        },
    );
    guarded.require(
        CompileRequestId(0),
        ResourceBounds {
            min_registers: Some(129),
            max_local_bytes: Some(0),
        },
    );
    Recipe::stating_edges(
        RecipeSignature::new(vec![64, 32], vec![16]),
        CompileRequests::new(vec![
            CompileRequest::unguarded(vec![source], PtxCompileOptions::default()),
            guarded,
        ]),
        RecipeBody::new(
            2,
            1,
            vec![KernelModule {
                code: recipe::KernelCode::ImageOnly {
                    preferred_image: ModuleImage::CudaLoadable {
                        bytes: Blob::from_vec(vec![0xCA, 0xFE]),
                        persistence: ImagePersistence::Persistable,
                        provenance: None,
                    },
                    alternative_images: Vec::new(),
                },
                resource_bounds: ResourceBounds::default(),
            }],
            vec![WorkspaceBuf {
                nbytes: 16,
                init: Some(Blob::from_vec(vec![1, 2, 3, 4])),
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Workspace(0).at(4),
                        value: 9,
                        nbytes: 8,
                    },
                    [],
                ),
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Input(0).at(8),
                        destination: WritableBuf::Output(0).at(0),
                        nbytes: 16,
                    },
                    [0],
                ),
                Node::new(
                    Work::Choice {
                        candidates: vec![candidate("alpha", 0x10), candidate("beta", 0x20)],
                        input_binding: vec![BufRef::Input(1)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    [1],
                ),
            ],
        ),
    )
    .expect("the fixture is a recipe")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The artifact is the same bytes it has always been, so version 13 still
/// means what every reader of it — including `include/recipe.h` — was written
/// against.
#[test]
fn the_fixture_encodes_to_the_bytes_version_twelve_has_always_meant() {
    let wire = recipe::serialize(&fixture()).expect("encode");
    assert_eq!(wire.len(), WIRE_LEN);
    assert_eq!(hex(blake3::hash(&wire).as_bytes()), WIRE_DIGEST);
    assert_eq!(recipe::wire::VERSION, 13);
    assert_eq!(
        recipe::deserialize(&wire).expect("decode"),
        fixture(),
        "and the bytes come back as the recipe they were written from"
    );
}

/// The identity is the same digest it has always been, so a realization cached
/// by an earlier build is still handed to the recipe that produced it.
#[test]
fn the_fixture_identifies_as_it_has_always_identified() {
    assert_eq!(
        hex(kiln::RecipeIdentity::of(&fixture())
            .expect("identity")
            .as_bytes()),
        IDENTITY
    );
}

/// Three regions landed one after another, every one of them writing the same
/// buffer, each following the sinks of the one before — the shape
/// `weldx_assemble` lowers a chain of ops into.
///
/// This is where a landing that quietly derived its own hazards would show:
/// each region's source node would acquire an edge to every earlier writer of
/// that buffer on top of the one the caller stated, and the artifact would stop
/// being the artifact this format has always produced.
fn looped_splice() -> Recipe {
    let region = |entry: &str| {
        let mut b = <recipe::RecipeBuilder>::new(1, 1);
        let m = b.module(vec![0xBB]);
        b.kernel(m, entry)
            .read(recipe::Buf::Input(0))
            .read_write(recipe::Writable::Output(0))
            .add();
        b.memset(recipe::Writable::Output(0).at(0), 0, 4);
        b.finish_fragment()
    };
    let mut host = <recipe::RecipeBuilder>::new(1, 1);
    let mut sinks = Vec::new();
    for entry in ["first", "second", "third"] {
        sinks = host
            .splice(
                region(entry),
                &[BufRef::Input(0)],
                &[BufRef::Output(0)],
                &sinks,
            )
            .expect("the region lands");
    }
    host.finish(RecipeSignature::new(vec![64], vec![64]))
        .expect("finish")
        .into_recipe()
}

/// A landing is ordered by what the caller states and by nothing else, so the
/// same chain of regions is the same artifact and the same identity it has
/// always been.
#[test]
fn a_looped_landing_is_the_artifact_it_has_always_been() {
    let recipe = looped_splice();
    let edges: Vec<Vec<usize>> = recipe
        .body()
        .nodes()
        .iter()
        .map(|node| node.deps().iter().copied().collect())
        .collect();
    assert_eq!(
        edges,
        vec![vec![], vec![0], vec![1], vec![2], vec![3], vec![4]],
        "each region keeps its own edge and follows the caller's prerequisite; \
         nothing was derived across the landings"
    );

    let wire = recipe::serialize(&recipe).expect("encode");
    assert_eq!(wire.len(), SPLICED_LEN);
    assert_eq!(hex(blake3::hash(&wire).as_bytes()), SPLICED_WIRE);
    assert_eq!(
        hex(kiln::RecipeIdentity::of(&recipe)
            .expect("identity")
            .as_bytes()),
        SPLICED_IDENTITY
    );
}
