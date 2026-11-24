// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The binding check at the boundary a Torch caller reaches, on a real device.
//!
//! The unit tests cover the pending branch, which rejects before any CUDA call
//! and so needs no GPU. What they cannot reach is the branch after it: once a
//! recipe has been realized and tuned, the handle holds a `ResidentRecipe` and
//! every later call replays it. That branch checks a *different* signature —
//! the resident recipe's — and this is what proves the two agree and that a
//! bad binding is refused there too, before anything launches.

use std::sync::Mutex;

use cuda_driver::ffi::CUstream;
use cuda_driver::{CUcontext, CUdevice, Context, CudaDriver, DeviceBuffer, check};
use kiln_py::{BoundSide, RunnableRecipe, run_recipe_core};
use recipe::{
    Arg, BufRef, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, Recipe, RecipeBody,
    RecipeSignature, Work,
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
const NBYTES: usize = N * 4;
const FILL_VALUE: f32 = 7.0;

fn fill_ptx_cubin() -> Vec<u8> {
    let mut bytes = FILL_PTX.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

fn fill_launch(grid: [u32; 3]) -> Node {
    Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: "fill".into(),
                args: vec![
                    Arg::output(0),
                    Arg::scalar(FILL_VALUE.to_le_bytes().to_vec()),
                    Arg::scalar((N as u32).to_le_bytes().to_vec()),
                ],
            },
            geometry: vec![Geometry {
                grid,
                block: [N as u32, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        [],
    )
}

fn candidate(grid: [u32; 3]) -> RecipeBody {
    RecipeBody::new(
        0,
        1,
        vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
        vec![],
        vec![fill_launch(grid)],
    )
}

/// One output of `NBYTES` and no inputs, behind a Choice so realization has a
/// real decision to make and the resident recipe differs from the pending one.
fn choice_recipe() -> Recipe {
    Recipe::stating_edges(
        RecipeSignature::new(Vec::new(), vec![NBYTES]),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![],
            vec![],
            // The fill writes every element, so there is no memset to clear
            // first: a graph with one repointable node is the smallest thing
            // that still exercises realize -> tune -> resident replay.
            vec![Node::new(
                Work::Choice {
                    candidates: vec![candidate([1, 1, 1]), candidate([8, 1, 1])],
                    input_binding: vec![],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap()
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

fn pending() -> Mutex<RunnableRecipe> {
    Mutex::new(RunnableRecipe::Pending {
        source: choice_recipe().into(),
        persist_to: None,
        report_to: None,
        scheduling_time_limit_ms: None,
    })
}

fn is_resident(handle: &Mutex<RunnableRecipe>) -> bool {
    matches!(&*handle.lock().unwrap(), RunnableRecipe::Resident(_))
}

/// The whole boundary, in the order a caller drives it: the first call realizes
/// and tunes, every later one replays the resident recipe, and the check holds
/// on both sides of that transition.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn the_resident_replay_checks_its_own_signature() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let stream: CUstream = std::ptr::null_mut();

    let workspace = DeviceBuffer::<u8>::alloc(ctx, NBYTES * 4).unwrap();
    let output = DeviceBuffer::from_host(ctx, &vec![0f32; N]).unwrap();
    let workspace_size = NBYTES * 4;

    let handle = pending();
    let out_ptrs = [output.ptr()];
    let exact = |nbytes: &[usize]| -> std::result::Result<(), String> {
        unsafe {
            run_recipe_core(
                &handle,
                stream,
                workspace.ptr(),
                workspace_size,
                BoundSide {
                    ptrs: &[],
                    nbytes: &[],
                },
                BoundSide {
                    ptrs: &out_ptrs,
                    nbytes,
                },
            )
        }
    };

    // First call: still Pending, so this realizes and tunes.
    assert!(!is_resident(&handle), "the handle starts pending");
    exact(&[NBYTES]).expect("the first call realizes and runs");
    assert!(
        is_resident(&handle),
        "realization is what makes the second call take the resident branch"
    );
    assert_eq!(
        output.to_host().unwrap(),
        vec![FILL_VALUE; N],
        "and it actually ran"
    );

    // Second call with the same exact binding: the resident replay path.
    exact(&[NBYTES]).expect("an exact binding replays");
    assert_eq!(output.to_host().unwrap(), vec![FILL_VALUE; N]);

    // A larger buffer is still a view that covers the port.
    exact(&[NBYTES * 2]).expect("an oversized binding replays");

    // Now the failures, all on the resident path — the handle never goes back
    // to pending, so these are checked against the resident signature.
    let undersized = exact(&[NBYTES - 1]).unwrap_err();
    assert_eq!(
        undersized,
        format!(
            "resident recipe: output 0 is {} bytes, recipe addresses {NBYTES}",
            NBYTES - 1
        ),
        "an undersized output is refused against the resident signature"
    );
    assert!(is_resident(&handle), "and the handle is untouched by it");

    let too_many = unsafe {
        run_recipe_core(
            &handle,
            stream,
            workspace.ptr(),
            workspace_size,
            BoundSide {
                ptrs: &[],
                nbytes: &[],
            },
            BoundSide {
                ptrs: &[output.ptr(), output.ptr()],
                nbytes: &[NBYTES, NBYTES],
            },
        )
    }
    .unwrap_err();
    assert_eq!(
        too_many,
        "resident recipe: recipe states 1 output buffer(s), 2 bound"
    );

    let too_few_inputs = unsafe {
        run_recipe_core(
            &handle,
            stream,
            workspace.ptr(),
            workspace_size,
            BoundSide {
                ptrs: &[output.ptr()],
                nbytes: &[NBYTES],
            },
            BoundSide {
                ptrs: &out_ptrs,
                nbytes: &[NBYTES],
            },
        )
    }
    .unwrap_err();
    assert_eq!(
        too_few_inputs, "resident recipe: recipe states 0 input buffer(s), 1 bound",
        "a recipe with no inputs refuses one bound anyway"
    );

    // The rejections left the resident recipe usable: nothing was consumed.
    exact(&[NBYTES]).expect("an exact binding still replays after the refusals");
    assert_eq!(output.to_host().unwrap(), vec![FILL_VALUE; N]);
}

/// The resident recipe states the same ports the pending one did. Realization
/// rewrites the body and the workspace; if it dropped the signature the replay
/// check above would be comparing against nothing and would pass for free.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn realization_carries_the_ports_onto_the_resident_recipe() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let stream: CUstream = std::ptr::null_mut();

    let workspace = DeviceBuffer::<u8>::alloc(ctx, NBYTES * 4).unwrap();
    let output = DeviceBuffer::from_host(ctx, &vec![0f32; N]).unwrap();
    let handle = pending();
    let stated = choice_recipe().contract().clone();

    unsafe {
        run_recipe_core(
            &handle,
            stream,
            workspace.ptr(),
            NBYTES * 4,
            BoundSide {
                ptrs: &[],
                nbytes: &[],
            },
            BoundSide {
                ptrs: &[output.ptr()],
                nbytes: &[NBYTES],
            },
        )
    }
    .expect("realize");

    let state = handle.lock().unwrap();
    let RunnableRecipe::Resident(resident) = &*state else {
        panic!("the call realized the recipe");
    };
    assert_eq!(
        *resident
            .contract()
            .expect("a realized residency states its ports"),
        stated,
        "the ports survived tuning and flattening"
    );
    assert!(
        !resident
            .body()
            .nodes()
            .iter()
            .any(|n| matches!(n.work(), Work::Choice { .. })),
        "and the choice really was resolved, so this is the tuned artifact"
    );
}

/// A device pointer bound where the recipe has no port at all: the count check
/// is what stops it, and it stops it before the launch rather than after.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn a_count_mismatch_is_refused_before_the_first_launch() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let workspace = DeviceBuffer::<u8>::alloc(ctx, NBYTES * 4).unwrap();

    let handle = pending();
    let error = unsafe {
        run_recipe_core(
            &handle,
            std::ptr::null_mut(),
            workspace.ptr(),
            NBYTES * 4,
            BoundSide {
                ptrs: &[],
                nbytes: &[],
            },
            BoundSide {
                ptrs: &[],
                nbytes: &[],
            },
        )
    }
    .unwrap_err();
    assert_eq!(
        error,
        "pending recipe: recipe states 1 output buffer(s), 0 bound"
    );
    assert!(
        !is_resident(&handle),
        "a refused binding never realizes the recipe, so nothing was compiled or launched"
    );
}

/// The fixture is a recipe on a machine without a device too: every test that
/// runs it needs a GPU and is `#[ignore]`d, so nothing else builds it.
#[test]
fn the_fixture_is_a_recipe() {
    let recipe = choice_recipe();
    assert_eq!(recipe.validate(), Ok(()));
    assert_eq!(recipe.body().outputs(), recipe.contract().outputs().len());
}
