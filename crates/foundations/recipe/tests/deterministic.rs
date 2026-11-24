// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! That one recipe is one byte string, in this process and in any other.
//!
//! A round trip says a reader accepts what a writer emitted, which two writers
//! disagreeing byte for byte would still satisfy. What a content-addressed
//! cache rests on is the stronger statement — the same recipe encodes to the
//! same bytes wherever it is encoded — and that is what is held here.
//!
//! The risk is concrete rather than theoretical. The blob pool carries a
//! `HashMap` index, Rust seeds every `HashMap` differently, and it seeds them
//! differently *within* one process as well as between two. So a write order
//! taken from that index would already differ between two encodes in one
//! process, and the run below would catch it; a second process is run anyway,
//! because that is the property being claimed and an in-process argument about
//! seeds is not it.

use std::process::Command;

use recipe::{
    Arg, Blob, BufRef, CompileRequest, CompileRequestId, CompileRequests, Geometry,
    ImagePersistence, KernelLaunch, KernelModule, ModuleImage, Node, PtxAssemblyMode,
    PtxCompileOptions, PtxTarget, Recipe, RecipeBody, RecipeSignature, ResourceBounds, SourceImage,
    SourceKind, Work, WorkspaceBuf, WritableBuf,
};

/// A recipe carrying a payload in every place one can live, and carrying the
/// same bytes in more than one of them.
///
/// Repetition is the point: a payload that appears twice is interned once, so
/// the pool's second visit to it is a *lookup* through the index rather than a
/// push, and the entry that lookup returns is the number a field is written
/// with. That is the one path by which a hashed order could reach the output,
/// so the fixture walks it — several times, under several roles, at both the
/// root and inside candidates.
fn fixture() -> Recipe {
    let shared = vec![0xAB, 0xCD, 0xEF];
    let ptx = ".version 8.8\n.target sm_100\n.address_size 64\n"
        .repeat(64)
        .into_bytes();

    let candidate = |entry: &str, image: Vec<u8>, seed: u8| {
        RecipeBody::new(
            1,
            1,
            vec![
                KernelModule::from_request(
                    CompileRequestId(0),
                    vec![
                        ModuleImage::Cubin {
                            target_sm: 100,
                            bytes: Blob::from_vec(image.clone()),
                            provenance: None,
                        },
                        ModuleImage::CudaLoadable {
                            bytes: Blob::from_vec(shared.clone()),
                            persistence: ImagePersistence::Persistable,
                            provenance: None,
                        },
                    ],
                    ResourceBounds::default(),
                ),
                KernelModule::from_cuda_loadable(image),
            ],
            vec![
                WorkspaceBuf {
                    nbytes: 3,
                    init: Some(Blob::from_vec(shared.clone())),
                },
                WorkspaceBuf {
                    nbytes: 4,
                    init: Some(Blob::from_vec(vec![seed; 4])),
                },
            ],
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: entry.into(),
                        args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry::new([2, 1, 1], [32, 1, 1], 0)],
                },
                [],
            )],
        )
    };

    Recipe::stating_edges(
        RecipeSignature::new(vec![4, 8], vec![4]),
        CompileRequests::new(vec![CompileRequest {
            sources: vec![
                SourceImage {
                    kind: SourceKind::Ptx,
                    target: PtxTarget {
                        isa: [8, 8],
                        minimum_sm: 100,
                    },
                    blob: Blob::from_vec(ptx.clone()),
                },
                SourceImage {
                    kind: SourceKind::Ptx,
                    target: PtxTarget {
                        isa: [8, 8],
                        minimum_sm: 90,
                    },
                    blob: Blob::from_vec(ptx),
                },
            ],
            options: PtxCompileOptions {
                max_registers: Some(64),
                assembly_mode: PtxAssemblyMode::WholeProgram,
            },
            requires: Default::default(),
        }]),
        RecipeBody::new(
            2,
            1,
            vec![KernelModule::from_cuda_loadable(shared.clone())],
            vec![WorkspaceBuf {
                nbytes: 3,
                init: Some(Blob::from_vec(shared.clone())),
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: 4,
                    },
                    [],
                ),
                Node::new(
                    Work::Choice {
                        candidates: vec![
                            candidate("alpha", vec![1, 2, 3, 4], 0x10),
                            candidate("beta", vec![1, 2, 3, 4], 0x20),
                            candidate("gamma", vec![5, 6, 7, 8], 0x10),
                        ],
                        input_binding: vec![BufRef::Input(1)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    [0],
                ),
            ],
        ),
    )
    .expect("the fixture is a recipe")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// What this process says the fixture encodes to.
fn encoding() -> String {
    hex(blake3::hash(&recipe::serialize(&fixture()).expect("encode")).as_bytes())
}

/// The environment variable a child answers under, and the test that reads it.
///
/// A `#[test]` re-invoking its own binary is how a second process is reached
/// with the same fixture and no separate build product: the child runs this one
/// test, prints what it computed, and exits.
const REPORT: &str = "RECIPE_DETERMINISM_REPORT";

#[test]
fn one_recipe_encodes_to_one_byte_string_in_every_process() {
    if std::env::var_os(REPORT).is_some() {
        println!("{REPORT}={}", encoding());
        return;
    }

    let mine = encoding();
    assert_eq!(
        mine,
        encoding(),
        "two encodes in one process disagree, so a write order is being taken \
         from something reseeded per instance"
    );

    // A fresh binary invocation reseeds every hasher in it, so a child's answer
    // is this process's answer only if nothing seeded reached the bytes.
    let exe = std::env::current_exe().expect("this test binary");
    let output = Command::new(&exe)
        .args([
            "--exact",
            "one_recipe_encodes_to_one_byte_string_in_every_process",
            "--nocapture",
        ])
        .env(REPORT, "1")
        .output()
        .expect("re-invoke this test binary");
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("child prints UTF-8");
    let theirs = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{REPORT}=")))
        .expect("child reports an encoding");

    assert_eq!(
        theirs, mine,
        "the same recipe encoded to different bytes in two processes"
    );
}

/// The same, for the digest an identity would be built from rather than for the
/// bytes.
///
/// Serialization being stable is not on its own the claim a cache key needs: a
/// key is computed over a recipe, and a recipe reaches one either from a
/// producer or from a decode of somebody else's bytes. So the decoded recipe is
/// required to encode to what it was decoded from, which is what makes one
/// digest survive the round the runtime actually performs.
#[test]
fn a_decoded_recipe_re_encodes_to_the_bytes_it_came_from() {
    let bytes = recipe::serialize(&fixture()).expect("encode");
    let decoded = recipe::deserialize(&bytes).expect("decode");
    assert_eq!(
        hex(blake3::hash(&recipe::serialize(&decoded).expect("re-encode")).as_bytes()),
        hex(blake3::hash(&bytes).as_bytes()),
        "a recipe that came off the wire encodes to something else, so a key \
         computed after a decode would not be the key computed before one"
    );
}
