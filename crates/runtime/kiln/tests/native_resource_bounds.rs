// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a resource bound decides about an image the recipe already carries.
//!
//! A bound is producer data that removes a candidate this machine cannot use.
//! It says the same thing about an image the recipe carries as about one this
//! machine
//! assembles, so these apply one to the other and read back what it decided.

use cuda_driver::{CUcontext, CUdevice, CUdeviceptr, Context, CudaDriver, DeviceBuffer, check};
use kiln::{Bindings, RealizationArgs, RealizationPolicy, Realizer, RealizerOptions};
use recipe::{
    Arg, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, Recipe, RecipeBody,
    RecipeSignature, ResourceBounds, Work,
};

/// A kernel that writes one word, so its register allocation is a handful and
/// any floor above that is one it cannot meet.
const PROBE_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry probe(
    .param .u64 probe_param_0
)
{
    .reg .b32 %r1;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [probe_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0;
    st.global.u32 [%rd2], %r1;
    ret;
}
"#;

fn probe_image() -> Vec<u8> {
    let mut bytes = PROBE_PTX.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

unsafe fn init_context(drv: &CudaDriver) {
    let r = unsafe { (drv.cuInit)(0) };
    check(r, "cuInit").unwrap();
    let mut dev: CUdevice = 0;
    let r = unsafe { (drv.cuDeviceGet)(&mut dev, 0) };
    check(r, "cuDeviceGet").unwrap();
    let mut ctx: CUcontext = std::ptr::null_mut();
    let r = unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, dev) };
    check(r, "cuCtxCreate_v2").unwrap();
    let r = unsafe { (drv.cuCtxSetCurrent)(ctx) };
    check(r, "cuCtxSetCurrent").unwrap();
}

/// One module, one image, native only — so what resolution decides about it is
/// the bound and nothing else. There is no compile request to fall back to,
/// which is what makes a refusal visible as a failed realization.
fn admitting_native(bounds: ResourceBounds) -> Recipe {
    // One output, the single `u32` the probe writes.
    Recipe::stating_edges(
        RecipeSignature::new(Vec::new(), vec![std::mem::size_of::<u32>()]),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule {
                code: recipe::KernelCode::ImageOnly {
                    preferred_image: recipe::ModuleImage::CudaLoadable {
                        bytes: probe_image().into(),
                        persistence: recipe::ImagePersistence::Persistable,
                        provenance: None,
                    },
                    alternative_images: Vec::new(),
                },
                resource_bounds: bounds,
            }],
            Vec::new(),
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "probe".into(),
                        args: vec![Arg::output(0)],
                    },
                    geometry: vec![Geometry {
                        grid: [1, 1, 1],
                        block: [1, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

fn realize_admitting(
    ctx: CUcontext,
    out: CUdeviceptr,
    bounds: ResourceBounds,
) -> Result<(), String> {
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out],
        workspace: 0,
    };
    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    unsafe {
        realizer.realize(RealizationArgs::new(
            &admitting_native(bounds),
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy::default(),
        ))
    }
    .map(|_| ())
    .map_err(|error| format!("{error:?}"))
}

/// A register floor is what removes a spilling register-capped candidate. It
/// says the same thing about an image the recipe carries as about one this
/// machine compiles,
/// so an image that does not meet it is refused, not accepted because nothing
/// else in the bound was set.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn a_native_image_below_the_register_floor_is_refused() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<u32>::alloc(ctx, 1).unwrap();

    let error = realize_admitting(
        ctx,
        out.ptr(),
        ResourceBounds {
            min_registers: Some(240),
            max_local_bytes: None,
        },
    )
    .expect_err("a floor the image cannot meet refuses it");
    assert!(
        error.contains("below required 240 registers"),
        "refused for the wrong reason: {error}"
    );

    // A floor it does meet leaves it accepted, so the refusal above is the
    // bound talking and not the native path failing outright.
    realize_admitting(
        ctx,
        out.ptr(),
        ResourceBounds {
            min_registers: Some(1),
            max_local_bytes: None,
        },
    )
    .expect("a floor the image meets admits it");
}

/// The fixture is a recipe on a machine without a device too: the test that
/// runs it needs a GPU and is `#[ignore]`d, so nothing else builds it.
#[test]
fn the_fixture_is_a_recipe() {
    let recipe = admitting_native(ResourceBounds::default());
    assert_eq!(recipe.validate(), Ok(()));
    assert_eq!(recipe.body().outputs(), recipe.contract().outputs().len());
}
