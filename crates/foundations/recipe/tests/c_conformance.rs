// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use recipe::Work;
use recipe::algebra::{CompileRequests, Geometry, KernelLaunch};
use recipe::{
    Arg, Blob, BufRef, CompileProvenance, CompileRequest, CompileRequestId, ImagePersistence,
    KernelModule, ModuleImage, Node, PtxAssemblyMode, PtxCompileOptions, PtxTarget, RecipeBody,
    RecipeSignature, ResourceBounds, SourceImage, SourceKind, WorkspaceBuf, WritableBuf,
};
use recipe::{Recipe, serialize};
use std::path::PathBuf;
use std::process::Command;

fn canonical_recipe() -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(vec![48, 96], vec![16]),
        CompileRequests::default(),
        RecipeBody::new(
            2,
            1,
            vec![KernelModule::from_cuda_loadable(Blob::from_vec(vec![
                0xDE, 0xAD, 0xBE, 0xEF,
            ]))],
            vec![WorkspaceBuf {
                nbytes: 4,
                init: Some(Blob::from_vec(vec![10, 0, 0, 0])),
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: 16,
                    },
                    [],
                ),
                Node::new(
                    Work::Call {
                        call: KernelLaunch {
                            module: 0,
                            entry: "fill".into(),
                            args: vec![
                                Arg::input(0),
                                Arg::workspace(0),
                                Arg::pointer(BufRef::Output(0).at(4)),
                            ],
                        },
                        geometry: vec![Geometry {
                            grid: [1, 1, 1],
                            block: [256, 1, 1],
                            dynamic_shmem: 0,
                        }],
                    },
                    [0],
                ),
                // A copy out of an input, both ends offset: the reader has to
                // carry the two capabilities and the two offsets separately.
                // Two dependencies, so the artifact states an order a reader
                // can be shown to check.
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Input(1).at(8),
                        destination: WritableBuf::Output(0).at(4),
                        nbytes: 8,
                    },
                    [0, 1],
                ),
            ],
        ),
    )
    .unwrap()
}

fn candidate_recipe(entry: &str) -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(vec![32], vec![32]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            1,
            vec![KernelModule::from_cuda_loadable(Blob::from_vec(vec![
                0xAA, 0xBB,
            ]))],
            vec![WorkspaceBuf {
                nbytes: 32,
                init: None,
            }],
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: entry.into(),
                        args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [4, 1, 1],
                        block: [64, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

fn choice_recipe() -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(vec![64], vec![]),
        CompileRequests::default(),
        RecipeBody::new(
            1,
            0,
            vec![],
            vec![WorkspaceBuf {
                nbytes: 64,
                init: None,
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Workspace(0).at(16),
                        value: 0,
                        nbytes: 48,
                    },
                    [],
                ),
                Node::new(
                    Work::Choice {
                        candidates: vec![
                            candidate_recipe("alpha")
                                .into_fragment()
                                .into_body()
                                .unwrap(),
                            candidate_recipe("beta")
                                .into_fragment()
                                .into_body()
                                .unwrap(),
                        ],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Workspace(0)],
                    },
                    [0],
                ),
            ],
        ),
    )
    .unwrap()
}

fn source_recipe() -> Recipe {
    let ptx = ".version 8.8\n.target sm_100\n.address_size 64\n"
        .repeat(256)
        .into_bytes();
    // One request per assembly mode, so the C reader states each of the three
    // rather than the one this fixture happens to prefer. The last is guarded
    // by both of the others, with different bounds, so a reader that dropped a
    // prerequisite or its bounds would say something else.
    let source = |isa: [u8; 2], minimum_sm, blob: Vec<u8>| SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget { isa, minimum_sm },
        blob: Blob::from_vec(blob),
    };
    let request = |assembly_mode, max_registers, sources| {
        CompileRequest::unguarded(
            sources,
            PtxCompileOptions {
                max_registers,
                assembly_mode,
            },
        )
    };
    let mut guarded = request(
        PtxAssemblyMode::WholeProgram,
        Some(128),
        vec![source([8, 8], 100, ptx.clone())],
    );
    guarded.require(
        CompileRequestId(0),
        ResourceBounds {
            min_registers: Some(129),
            max_local_bytes: None,
        },
    );
    guarded.require(
        CompileRequestId(1),
        ResourceBounds {
            min_registers: None,
            max_local_bytes: Some(0),
        },
    );

    Recipe::stating_edges(
        RecipeSignature::new(vec![8], vec![8]),
        CompileRequests::new(vec![
            request(
                PtxAssemblyMode::EnvironmentDefault,
                None,
                vec![source([8, 8], 100, b"default".to_vec())],
            ),
            // Two modules in one unit, at two different targets: the C reader
            // has to walk the sequence and state each source's own facts,
            // rather than read one and stop.
            request(
                PtxAssemblyMode::Relocatable,
                Some(64),
                vec![
                    source([8, 8], 100, b"relocatable".to_vec()),
                    source([8, 7], 90, b"callable".to_vec()),
                ],
            ),
            guarded,
        ]),
        RecipeBody::new(
            1,
            1,
            // Three images ahead of the compile, in the order a reader must
            // try them: an exact cubin carrying full provenance, bytes only
            // this session can use, then bytes that keep. Every one of those
            // facts is the image's own, so the C reader states them without
            // consulting a provenance string.
            vec![KernelModule::from_request(
                CompileRequestId(2),
                vec![
                    ModuleImage::Cubin {
                        target_sm: 100,
                        bytes: Blob::from_vec(vec![0xC0, 0xDE]),
                        provenance: Some(CompileProvenance {
                            unit_digest: [7; 32],
                            options: PtxCompileOptions {
                                max_registers: Some(96),
                                assembly_mode: PtxAssemblyMode::Relocatable,
                            },
                            target_sm: 100,
                            assembler: "ptxas".into(),
                            assembler_version: "13.0".into(),
                            driver_version: Some(13000),
                            toolkit_version: Some(12080),
                        }),
                    },
                    ModuleImage::CudaLoadable {
                        bytes: Blob::from_vec(vec![0x5E, 0x55]),
                        persistence: ImagePersistence::SessionOnly,
                        provenance: None,
                    },
                    ModuleImage::CudaLoadable {
                        bytes: Blob::from_vec(vec![0xCA, 0xFE]),
                        persistence: ImagePersistence::Persistable,
                        provenance: None,
                    },
                ],
                ResourceBounds {
                    min_registers: Some(32),
                    max_local_bytes: Some(64),
                },
            )],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "source".into(),
                        args: vec![Arg::input(0), Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [32, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

fn find_cc() -> Option<&'static str> {
    ["cc", "clang", "gcc"].into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

#[test]
fn c_reader_matches_serializer() {
    let Some(cc) = find_cc() else {
        eprintln!("c_conformance: no C compiler (cc/clang/gcc); skipping");
        return;
    };

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = manifest.join("include");
    let source = manifest.join("tests/c_conformance.c");

    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let recipe_path = tmp.join(format!("recipe_canonical_{pid}.bin"));
    let choice_path = tmp.join(format!("recipe_choice_{pid}.bin"));
    let source_path = tmp.join(format!("recipe_source_{pid}.bin"));
    let exe_path = tmp.join(format!("recipe_c_conformance_{pid}"));

    std::fs::write(&recipe_path, serialize(&canonical_recipe()).unwrap())
        .expect("write recipe bytes");
    std::fs::write(&choice_path, serialize(&choice_recipe()).unwrap()).expect("write choice bytes");
    std::fs::write(&source_path, serialize(&source_recipe()).unwrap()).expect("write source bytes");

    let compile = Command::new(cc)
        .arg("-I")
        .arg(&include)
        .arg("-std=c99")
        .arg("-O2")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg(&source)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .expect("invoke C compiler");
    assert!(
        compile.status.success(),
        "compiling c_conformance.c failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&exe_path)
        .arg(&recipe_path)
        .arg(&choice_path)
        .arg(&source_path)
        .output()
        .expect("run C conformance harness");

    let _ = std::fs::remove_file(&recipe_path);
    let _ = std::fs::remove_file(&choice_path);
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);

    assert!(
        run.status.success(),
        "C conformance harness failed with exit code {:?}",
        run.status.code()
    );
}
