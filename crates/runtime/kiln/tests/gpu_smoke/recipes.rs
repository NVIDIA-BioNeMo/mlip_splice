// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use recipe::{
    Arg, Blob, BufRef, CompileRequest, CompileRequestId, CompileRequests, Geometry, KernelLaunch,
    KernelModule, Node, PtxAssemblyMode, PtxCompileOptions, PtxTarget, Recipe, RecipeBody,
    RecipeSignature, ResourceBounds, SourceImage, SourceKind, Work, WorkspaceBuf, WritableBuf,
};

pub(super) const FILL_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry fill(
    .param .u64 fill_param_0,
    .param .f32 fill_param_1,
    .param .u32 fill_param_2
)
{
    .reg .pred %p1;
    .reg .f32 %f1;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [fill_param_0];
    ld.param.f32 %f1, [fill_param_1];
    ld.param.u32 %r1, [fill_param_2];
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra $L_DONE;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f1;
$L_DONE:
    ret;
}
"#;

pub(super) const N: usize = 256;
pub(super) const NBYTES: usize = N * 4;
/// Every fixture here writes one `N`-float output and reads no input.
pub(super) fn one_output() -> RecipeSignature {
    RecipeSignature::new(Vec::new(), vec![NBYTES])
}
pub(super) const FILL_VALUE: f32 = 7.0;

pub(super) fn fill_ptx_cubin() -> Vec<u8> {
    let mut bytes = FILL_PTX.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

pub(super) fn fill_recipe(grid: [u32; 3]) -> Recipe {
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
            vec![WorkspaceBuf {
                nbytes: NBYTES,
                init: None,
            }],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: NBYTES,
                    },
                    [],
                ),
                Node::new(
                    Work::Call {
                        call: KernelLaunch {
                            module: 0,
                            entry: "fill".into(),
                            args: vec![
                                Arg::output(0),
                                Arg::scalar(FILL_VALUE.to_ne_bytes().to_vec()),
                                Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                            ],
                        },
                        geometry: vec![Geometry {
                            grid,
                            block: [256, 1, 1],
                            dynamic_shmem: 0,
                        }],
                    },
                    [0],
                ),
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Output(0).at(0),
                        destination: WritableBuf::Workspace(0).at(0),
                        nbytes: NBYTES,
                    },
                    [1],
                ),
            ],
        ),
    )
    .unwrap()
}

pub(super) fn source_fill_recipe(include_native: bool) -> Recipe {
    let images = include_native
        .then(|| recipe::ModuleImage::CudaLoadable {
            bytes: fill_ptx_cubin().into(),
            persistence: recipe::ImagePersistence::Persistable,
            provenance: None,
        })
        .into_iter()
        .collect();
    Recipe::stating_edges(
        one_output(),
        CompileRequests::new(vec![CompileRequest {
            sources: vec![SourceImage {
                kind: SourceKind::Ptx,
                target: PtxTarget {
                    isa: [7, 0],
                    minimum_sm: 52,
                },
                blob: Blob::zstd_spilled(FILL_PTX.as_bytes()).unwrap(),
            }],
            options: PtxCompileOptions {
                max_registers: None,
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
            requires: Default::default(),
        }]),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_request(
                CompileRequestId(0),
                images,
                ResourceBounds::default(),
            )],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "fill".into(),
                        args: vec![
                            Arg::output(0),
                            Arg::scalar(FILL_VALUE.to_ne_bytes().to_vec()),
                            Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                        ],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [256, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// The two halves of the linked compile unit, as the hand-written `.ptx`
/// files they are: one declares `scaled_value` and launches, the other defines
/// it and has no entry at all. Neither text alone is a runnable image, which
/// is what makes the link the thing under test.
pub(super) const LINKED_FILL_PTX: &str = include_str!("../fixtures/linked_fill.ptx");
pub(super) const SCALE_PTX: &str = include_str!("../fixtures/scaled_value.ptx");

/// What `scaled_value` multiplies by, so the output distinguishes a linked run
/// from one that somehow reached the caller's input untouched.
pub(super) const SCALE: f32 = 2.0;

/// One compile over two PTX modules: the launched entry lives in the first and
/// calls a `.func` only the second defines, so nothing runs unless both
/// modules reached the linker together.
pub(super) fn linked_fill_recipe() -> Recipe {
    let source = |ptx: &str| SourceImage {
        kind: SourceKind::Ptx,
        target: PtxTarget {
            isa: [7, 0],
            minimum_sm: 52,
        },
        blob: Blob::zstd_spilled(ptx.as_bytes()).unwrap(),
    };
    Recipe::stating_edges(
        one_output(),
        CompileRequests::new(vec![CompileRequest::unguarded(
            vec![source(LINKED_FILL_PTX), source(SCALE_PTX)],
            PtxCompileOptions {
                max_registers: None,
                assembly_mode: PtxAssemblyMode::Relocatable,
            },
        )]),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_request(
                CompileRequestId(0),
                Vec::new(),
                ResourceBounds::default(),
            )],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "linked_fill".into(),
                        args: vec![
                            Arg::output(0),
                            Arg::scalar(FILL_VALUE.to_ne_bytes().to_vec()),
                            Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                        ],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [256, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

pub(super) fn one_foreign_arm_recipe(foreign_sm: u16) -> Recipe {
    let foreign = RecipeBody::new(
        0,
        1,
        vec![KernelModule {
            code: recipe::KernelCode::ImageOnly {
                preferred_image: recipe::ModuleImage::Cubin {
                    target_sm: foreign_sm,
                    bytes: fill_ptx_cubin().into(),
                    provenance: None,
                },
                alternative_images: Vec::new(),
            },
            resource_bounds: ResourceBounds::default(),
        }],
        vec![],
        vec![fill_launch_node(FOREIGN_ARM_GRID)],
    );
    let working = |grid: [u32; 3]| {
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
            vec![],
            vec![fill_launch_node(grid)],
        )
    };
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![],
            vec![],
            vec![Node::new(
                Work::Choice {
                    candidates: vec![foreign, working([1, 1, 1]), working([8, 1, 1])],
                    input_binding: vec![],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// Distinct from every working arm's, so a visit to the foreign arm shows up
/// as itself rather than hiding behind an identical sibling.
pub(super) const FOREIGN_ARM_GRID: [u32; 3] = [4, 1, 1];

pub(super) fn fill_launch_node(grid: [u32; 3]) -> Node {
    Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: "fill".into(),
                args: vec![
                    Arg::output(0),
                    Arg::scalar(FILL_VALUE.to_ne_bytes().to_vec()),
                    Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                ],
            },
            geometry: vec![Geometry {
                grid,
                block: [256, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )
}

pub(super) fn seeded_recipe(seed: f32) -> Recipe {
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
            vec![WorkspaceBuf {
                nbytes: NBYTES,
                init: Some(vec_f32_bytes(seed).into()),
            }],
            vec![
                // Copy the seeded workspace into the bound output, proving the
                // graph path seeded workspace[0] before the nodes ran.
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Workspace(0).at(0),
                        destination: WritableBuf::Output(0).at(0),
                        nbytes: NBYTES,
                    },
                    [],
                ),
            ],
        ),
    )
    .unwrap()
}

pub(super) fn choice_fill_recipe() -> Recipe {
    let candidate = |grid: [u32; 3]| {
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
            vec![],
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "fill".into(),
                        args: vec![
                            Arg::output(0),
                            Arg::scalar(FILL_VALUE.to_ne_bytes().to_vec()),
                            Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                        ],
                    },
                    geometry: vec![Geometry {
                        grid,
                        block: [256, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        )
    };
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![],
            vec![],
            vec![
                Node::new(
                    Work::Memset {
                        destination: WritableBuf::Output(0).at(0),
                        value: 0,
                        nbytes: NBYTES,
                    },
                    [],
                ),
                Node::new(
                    Work::Choice {
                        candidates: vec![candidate([1, 1, 1]), candidate([8, 1, 1])],
                        input_binding: vec![],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    [0],
                ),
            ],
        ),
    )
    .unwrap()
}

pub(super) fn vec_f32_bytes(value: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(NBYTES);
    for _ in 0..N {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

pub(super) const SPIN_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry spin(
    .param .u64 spin_param_0,
    .param .u64 spin_param_1
)
{
    .reg .pred %p1;
    .reg .b64 %rd<7>;
    ld.param.u64 %rd1, [spin_param_0];
    ld.param.u64 %rd2, [spin_param_1];
    cvta.to.global.u64 %rd3, %rd1;
    mov.u64 %rd4, %clock64;
$L_SPIN:
    mov.u64 %rd5, %clock64;
    sub.s64 %rd6, %rd5, %rd4;
    setp.lt.s64 %p1, %rd6, %rd2;
    @%p1 bra $L_SPIN;
    st.global.u64 [%rd3], %rd6;
    ret;
}
"#;

pub(super) fn spin_recipe(kernels: usize, cycles: u64) -> Recipe {
    let mut bytes = SPIN_PTX.as_bytes().to_vec();
    bytes.push(0);
    let node = |output: usize| {
        Node::new(
            Work::Call {
                call: KernelLaunch {
                    module: 0,
                    entry: "spin".into(),
                    args: vec![
                        Arg::output(output),
                        Arg::scalar(cycles.to_ne_bytes().to_vec()),
                    ],
                },
                geometry: vec![Geometry {
                    grid: [16, 1, 1],
                    block: [32, 1, 1],
                    dynamic_shmem: 0,
                }],
            },
            [],
        )
    };
    Recipe::stating_edges(
        // One output per spinning kernel, each an `N`-float buffer.
        RecipeSignature::new(Vec::new(), vec![NBYTES; kernels]),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            kernels,
            vec![KernelModule::from_cuda_loadable(bytes)],
            Vec::new(),
            (0..kernels).map(node).collect(),
        ),
    )
    .unwrap()
}

/// Every fixture in this file is a recipe, on a machine with a device and on
/// one without.
///
/// The tests that run these need a GPU and are `#[ignore]`d, so nothing else
/// here builds them: a fixture whose body states a different arity than its
/// ports, or a candidate that does not fill the slots its binding has, would be
/// refused only where no ordinary test run reaches. Building them all is what
/// makes that refusal arrive at the same place as every other.
#[test]
fn every_fixture_here_is_a_recipe() {
    let fixtures = [
        fill_recipe([1, 1, 1]),
        source_fill_recipe(false),
        source_fill_recipe(true),
        linked_fill_recipe(),
        one_foreign_arm_recipe(52),
        seeded_recipe(1.5),
        choice_fill_recipe(),
        spin_recipe(3, 1_000),
    ];
    for recipe in fixtures {
        assert_eq!(recipe.validate(), Ok(()));
        assert_eq!(
            recipe.body().outputs(),
            recipe.contract().outputs().len(),
            "the body is bound through the ports it states"
        );
    }
}
