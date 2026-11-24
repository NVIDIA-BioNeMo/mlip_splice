// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which image or compile a module is realized from, on a real device.
//!
//! A module states where its kernels come from and, when it offers more than
//! one image, the order to try them in. What this asks is that kiln honours
//! that order, refuses a cubin the device cannot run, falls back to the
//! compile only when no image serves, and writes back what the artifact policy
//! and the image's own persistence allow.

use cuda_driver::{Context, CudaDriver, DeviceBuffer, check};
use kiln::{
    ArtifactPolicy, Bindings, RealizationArgs, RealizationPolicy, Realizer, RealizerOptions,
    execute,
};
use recipe::{
    Arg, Blob, CompileRequest, CompileRequestId, CompileRequests, Geometry, ImagePersistence,
    KernelCode, KernelLaunch, KernelModule, ModuleImage, Node, PtxAssemblyMode, PtxCompileOptions,
    PtxTarget, Recipe, RecipeBody, RecipeSignature, ResourceBounds, SourceImage, SourceKind, Work,
};

const FILL_PTX: &str = r#".version 7.0
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

const N: usize = 256;
const FILL_VALUE: f32 = 7.0;
/// An architecture no device under test is, so a cubin naming it is one this
/// machine must refuse without loading.
const FOREIGN_SM: u16 = 37;

unsafe fn init_context(drv: &CudaDriver) {
    check(unsafe { (drv.cuInit)(0) }, "cuInit").unwrap();
    let mut device = 0;
    check(unsafe { (drv.cuDeviceGet)(&mut device, 0) }, "cuDeviceGet").unwrap();
    let mut ctx = std::ptr::null_mut();
    check(
        unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, device) },
        "cuCtxCreate",
    )
    .unwrap();
    check(unsafe { (drv.cuCtxSetCurrent)(ctx) }, "cuCtxSetCurrent").unwrap();
}

/// PTX the CUDA loader accepts directly, which is what a producer that hands
/// over bytes without labelling an architecture is handing over.
fn loadable_fill() -> Vec<u8> {
    let mut bytes = FILL_PTX.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

fn fill_request() -> CompileRequest {
    CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [7, 0],
                minimum_sm: 52,
            },
            blob: Blob::from_vec(FILL_PTX.as_bytes().to_vec()),
        }],
        PtxCompileOptions {
            max_registers: None,
            assembly_mode: PtxAssemblyMode::Relocatable,
        },
    )
}

fn fill_node() -> Node {
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
                grid: [(N as u32).div_ceil(256), 1, 1],
                block: [256, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )
}

fn recipe_with(code: KernelCode, requests: Vec<CompileRequest>) -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(Vec::new(), vec![N * 4]),
        CompileRequests::new(requests),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule {
                code,
                resource_bounds: ResourceBounds::default(),
            }],
            Vec::new(),
            vec![fill_node()],
        ),
    )
    .unwrap()
}

fn usable_image() -> ModuleImage {
    ModuleImage::CudaLoadable {
        bytes: loadable_fill().into(),
        persistence: ImagePersistence::Persistable,
        provenance: None,
    }
}

fn foreign_cubin() -> ModuleImage {
    ModuleImage::Cubin {
        target_sm: FOREIGN_SM,
        bytes: loadable_fill().into(),
        provenance: None,
    }
}

/// Realize `recipe` under `policy`, run it, and hand back what was realized
/// alongside the values the device wrote.
fn realize_and_run(recipe: &Recipe, policy: ArtifactPolicy) -> (Recipe, usize, Vec<f32>) {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let mut realized = unsafe {
        realizer.realize(RealizationArgs::new(
            recipe,
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy {
                artifact: policy,
                ..RealizationPolicy::default()
            },
        ))
    }
    .unwrap();
    unsafe { execute(&mut realized.resident, &bindings, std::ptr::null_mut()) }.unwrap();
    let compiles = realized.report.requests.len();
    (realized.artifact, compiles, out.to_host().unwrap())
}

fn assert_filled(values: &[f32]) {
    assert!(
        values.iter().all(|&value| value == FILL_VALUE),
        "the device did not run the fill: {:?}",
        &values[..4.min(values.len())]
    );
}

/// A module that is images and nothing else realizes from the one it carries,
/// with no toolchain involved at all.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn an_image_only_module_realizes_without_a_compile() {
    let recipe = recipe_with(
        KernelCode::ImageOnly {
            preferred_image: usable_image(),
            alternative_images: Vec::new(),
        },
        Vec::new(),
    );
    let (artifact, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(compiles, 0, "an image-only module compiles nothing");
    assert_eq!(artifact.context().as_slice().len(), 0);
    assert_filled(&values);
}

/// A module that is a compile and nothing else realizes by running it, and
/// what comes back is a cubin for this device that another process could load.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_source_only_module_compiles_and_yields_a_persistable_cubin() {
    let recipe = recipe_with(
        KernelCode::PtxBacked {
            compile_request: CompileRequestId(0),
            images: Vec::new(),
        },
        vec![fill_request()],
    );
    let (artifact, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(compiles, 1, "the one request was compiled");
    assert_filled(&values);

    let [image] = artifact.body().context()[0]
        .code
        .images()
        .collect::<Vec<_>>()[..]
    else {
        panic!("a realized module is one image");
    };
    let sm = cuda_driver::Device::current().sm_arch().unwrap() as u16;
    assert!(
        matches!(image, ModuleImage::Cubin { target_sm, .. } if *target_sm == sm),
        "the compile states the architecture it assembled for"
    );
    assert!(
        image.is_persistable(),
        "an assembled cubin outlives the session that produced it"
    );
}

/// An image ahead of a compile is used instead of it. The compile is still
/// stated, so the artifact could be realized elsewhere; nothing runs it here.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn an_image_ahead_of_a_compile_is_used_instead_of_it() {
    let recipe = recipe_with(
        KernelCode::PtxBacked {
            compile_request: CompileRequestId(0),
            images: vec![usable_image()],
        },
        vec![fill_request()],
    );
    let (_, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(compiles, 0, "the image answered, so nothing was compiled");
    assert_filled(&values);
}

/// Order is priority, and a cubin naming another architecture is not a
/// candidate on this one. The usable image is reached only because the list is
/// walked past the one that cannot serve.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn a_cubin_for_another_architecture_is_passed_over_for_the_next_image() {
    let recipe = recipe_with(
        KernelCode::ImageOnly {
            preferred_image: foreign_cubin(),
            alternative_images: vec![usable_image()],
        },
        Vec::new(),
    );
    let (artifact, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(compiles, 0);
    assert_filled(&values);
    let [image] = artifact.body().context()[0]
        .code
        .images()
        .collect::<Vec<_>>()[..]
    else {
        panic!("a realized module is one image");
    };
    assert!(
        matches!(image, ModuleImage::CudaLoadable { .. }),
        "the alternative was taken, not the cubin this device cannot run"
    );
}

/// The same list with nothing usable in it falls through to the compile. A
/// module whose only image names a foreign architecture must not be realized
/// from that image, and must not fail either while it still has a compile.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn an_unusable_image_falls_through_to_the_compile() {
    let recipe = recipe_with(
        KernelCode::PtxBacked {
            compile_request: CompileRequestId(0),
            images: vec![foreign_cubin()],
        },
        vec![fill_request()],
    );
    let (_, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(compiles, 1, "no image served, so the compile ran");
    assert_filled(&values);
}

/// A compile guarded by a prerequisite resolves that prerequisite first.
///
/// The guard is a statement about the base *image*, so the base has to be
/// launched from for there to be anything to measure — a module of its own,
/// here running the same fill. The bound is one any image satisfies, so what
/// this shows is that the prerequisite was reached and cleared rather than
/// ignored.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_guarded_compile_resolves_its_prerequisite_first() {
    let mut guarded = fill_request();
    guarded.require(
        CompileRequestId(0),
        ResourceBounds {
            min_registers: Some(1),
            max_local_bytes: None,
        },
    );
    // The base is capped so it is a different compile from the one it guards.
    let mut base = fill_request();
    base.options.max_registers = Some(64);

    let module = |request: usize| {
        KernelModule::from_request(
            CompileRequestId(request),
            Vec::new(),
            ResourceBounds::default(),
        )
    };
    let launch = |module: usize, deps: Vec<usize>| {
        Node::new(
            match fill_node().work().clone() {
                Work::Call { call, geometry } => Work::Call {
                    call: KernelLaunch { module, ..call },
                    geometry,
                },
                work => work,
            },
            deps,
        )
    };
    let recipe = Recipe::stating_edges(
        RecipeSignature::new(Vec::new(), vec![N * 4]),
        CompileRequests::new(vec![base, guarded]),
        RecipeBody::new(
            0,
            1,
            vec![module(0), module(1)],
            Vec::new(),
            vec![launch(0, Vec::new()), launch(1, vec![0])],
        ),
    )
    .unwrap();

    let (_, compiles, values) = realize_and_run(&recipe, ArtifactPolicy::NativeOnly);
    assert_eq!(
        compiles, 2,
        "the guard and the request it guards were both compiled"
    );
    assert_filled(&values);
}

/// What a source-backed artifact keeps: the compile, and no image at all, so
/// the bytes it carries are the source rather than one machine's assembly of
/// it.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_source_backed_artifact_keeps_the_compile_and_drops_the_image() {
    let recipe = recipe_with(
        KernelCode::PtxBacked {
            compile_request: CompileRequestId(0),
            images: Vec::new(),
        },
        vec![fill_request()],
    );
    let (artifact, _, values) = realize_and_run(&recipe, ArtifactPolicy::SourceBacked);
    assert_filled(&values);
    assert_eq!(artifact.context().as_slice().len(), 1);
    assert_eq!(artifact.body().context()[0].code.images().count(), 0);
    assert_eq!(
        artifact.body().context()[0].code.compile_request(),
        Some(CompileRequestId(0))
    );
    artifact
        .validate()
        .expect("a source-backed artifact is one");
}

/// The fixtures here are recipes on a machine without a device too: every test
/// that runs one needs a GPU and is `#[ignore]`d, so nothing else builds them.
#[test]
fn every_fixture_here_is_a_recipe() {
    let image = recipe_with(
        KernelCode::ImageOnly {
            preferred_image: ModuleImage::CudaLoadable {
                bytes: Blob::from_vec(vec![1, 2, 3]),
                persistence: ImagePersistence::Persistable,
                provenance: None,
            },
            alternative_images: Vec::new(),
        },
        Vec::new(),
    );
    assert_eq!(image.validate(), Ok(()));
    assert_eq!(image.body().outputs(), image.contract().outputs().len());
}
