// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a measured candidate is put back to, when the program reads what it
//! writes — and which storage that means once tuning has descended into a
//! nested choice.

use cuda_driver::{CUcontext, CUdevice};
use cuda_driver::{CUdeviceptr, CUstream, Context, CudaDriver, DeviceBuffer, check};
use kiln::{Bindings, RealizationArgs, RealizationPolicy, Realizer, RealizerOptions};
use recipe::{
    Arg, BufRef, CompileRequests, Geometry, KernelLaunch, KernelModule, Node, Recipe, RecipeBody,
    RecipeSignature, Work, WorkspaceBuf, WritableBuf,
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

/// A recipe whose outer choice carries a nested one, bound against the outer
/// candidate's ports *in another order* with a workspace slice among them.
///
/// This is the shape a position-indexed starting state gets wrong: at the
/// nested level, "output 0" is the caller's output 1, "output 1" is scratch,
/// and "output 2" is the caller's output 0.
fn nested_choice_recipe() -> Recipe {
    let inner = |grid: [u32; 3], port: usize| {
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
                            Arg::output(port),
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
    // One outer candidate, whose body is itself a choice — and which binds that
    // inner choice against scratch of its own rather than against a port. What
    // the inner level is timed over is therefore storage the caller never
    // supplied and never named.
    let outer = |port: usize| {
        RecipeBody::new(
            0,
            3,
            vec![],
            vec![WorkspaceBuf {
                nbytes: NBYTES,
                init: None,
            }],
            vec![
                Node::new(
                    Work::Choice {
                        candidates: vec![inner([1, 1, 1], 0), inner([8, 1, 1], 0)],
                        input_binding: vec![],
                        output_binding: vec![BufRef::Workspace(0)],
                    },
                    [],
                ),
                Node::new(
                    Work::Memcpy {
                        source: BufRef::Workspace(0).at(0),
                        destination: WritableBuf::Output(port).at(0),
                        nbytes: NBYTES,
                    },
                    [0],
                ),
            ],
        )
    };
    Recipe::stating_edges(
        RecipeSignature::new(Vec::new(), vec![NBYTES, NBYTES]),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            2,
            vec![],
            vec![WorkspaceBuf {
                nbytes: NBYTES,
                init: None,
            }],
            vec![Node::new(
                Work::Choice {
                    candidates: vec![outer(2), outer(0)],
                    input_binding: vec![],
                    // Reordered, with scratch in the middle.
                    output_binding: vec![
                        BufRef::Output(1),
                        BufRef::Workspace(0),
                        BufRef::Output(0),
                    ],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

/// What the caller supplied, by the address it was supplied at, and a record of
/// every address tuning offered.
struct SavedByAddress {
    saved: Vec<(CUdeviceptr, usize)>,
    offered: std::sync::Mutex<Vec<CUdeviceptr>>,
    restored: std::sync::Mutex<Vec<CUdeviceptr>>,
}

impl kiln::StartingState for SavedByAddress {
    fn restore(&self, buffer: CUdeviceptr, _stream: CUstream) -> Result<(), String> {
        self.offered.lock().unwrap().push(buffer);
        if self.saved.iter().any(|(address, _)| *address == buffer) {
            self.restored.lock().unwrap().push(buffer);
        }
        Ok(())
    }
}

/// Tuning descends into the nested choice and offers what *that* level binds —
/// which is the caller's ports in the child's order, with a workspace slice
/// among them. Reading those by position would name a different allocation at
/// every one, so the starting state is asked by address and answers for the two
/// it was given.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_nested_choice_offers_its_own_bound_storage_to_the_starting_state() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let stream: CUstream = std::ptr::null_mut();

    let first = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let second = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let ws = DeviceBuffer::<f32>::alloc(ctx, 4 * N).unwrap();
    let outputs = [first.ptr(), second.ptr()];
    let bindings = Bindings {
        inputs: &[],
        outputs: &outputs,
        workspace: ws.ptr(),
    };

    let state = SavedByAddress {
        saved: vec![(first.ptr(), NBYTES), (second.ptr(), NBYTES)],
        offered: std::sync::Mutex::new(Vec::new()),
        restored: std::sync::Mutex::new(Vec::new()),
    };

    let recipe = nested_choice_recipe();
    assert_eq!(recipe.validate(), Ok(()));
    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let mut args = RealizationArgs::new(&recipe, &bindings, stream, RealizationPolicy::default());
    args.starting_state = Some(&state);
    let _ = unsafe { realizer.realize(args) }.expect("realize");

    let offered = state.offered.lock().unwrap().clone();
    let restored = state.restored.lock().unwrap().clone();
    assert!(!offered.is_empty(), "tuning timed at least one candidate");

    // The decisive one: what tuning offers at the nested level is what *that*
    // level binds — here, scratch the caller never supplied. A starting state
    // handed the root's outputs at every depth would never see anything else,
    // and would have written the caller's bytes into this.
    assert!(
        offered.iter().any(|held| !outputs.contains(held)),
        "only the root's own outputs were ever offered, so the descent bound \
         nothing of its own: {offered:#x?}"
    );
    // And what came back is only what the caller supplied — never the scratch.
    assert!(
        restored.iter().all(|held| outputs.contains(held)),
        "something that was not the caller's state was restored: {restored:#x?}"
    );
    assert!(
        !restored.contains(&ws.ptr()),
        "the workspace is not the caller's state"
    );
}

/// The fixture is a recipe on a machine without a device too.
///
/// The test that runs it needs a GPU and is `#[ignore]`d, so nothing else here
/// builds it: a candidate whose formal slots do not match the binding that
/// selects it — which is the whole point of this fixture's reordering — would
/// otherwise be refused only where no ordinary run reaches.
#[test]
fn the_nested_fixture_is_a_recipe() {
    let recipe = nested_choice_recipe();
    assert_eq!(recipe.validate(), Ok(()));
    assert_eq!(recipe.body().outputs(), 2);
}
