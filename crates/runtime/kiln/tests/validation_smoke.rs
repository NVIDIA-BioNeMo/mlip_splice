// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate validation end to end: a candidate that computes the wrong thing
//! is rejected before it can win on speed, and validation writes only the
//! caller's candidate storage.

use cuda_driver::{
    CUdevice, CUdeviceptr, CUstream, Context, CudaDriver, DeviceBuffer, StreamRef, check,
};
use kiln::{
    Bindings, RealizationArgs, RealizationPolicy, Realizer, RealizerOptions, ResidentRecipe,
    execute, tune,
};
use recipe::{
    Arg, BufRef, CompileRequests, FlatRecipe, Geometry, KernelLaunch, KernelModule, Node, Recipe,
    RecipeBody, RecipeSignature, Segment, Work, WritableBuf,
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
/// Every program here writes one `N`-float output and reads no input.
fn one_output() -> RecipeSignature {
    RecipeSignature::new(Vec::new(), vec![NBYTES])
}
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
    let mut ctx = std::ptr::null_mut();
    let r = unsafe { (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev) };
    check(r, "cuDevicePrimaryCtxRetain").unwrap();
    let r = unsafe { (drv.cuCtxSetCurrent)(ctx) };
    check(r, "cuCtxSetCurrent").unwrap();
}

unsafe fn memset(drv: &CudaDriver, dev: CUdeviceptr, value: u8, stream: CUstream) {
    let r = unsafe { (drv.cuMemsetD8Async)(dev, value, NBYTES, stream) };
    check(r, "cuMemsetD8Async").unwrap();
}

fn clear_output() -> Node {
    Node::new(
        Work::Memset {
            destination: WritableBuf::Output(0).at(0),
            value: 0,
            nbytes: NBYTES,
        },
        [],
    )
}

fn fill_launch(value: f32, grid: [u32; 3], deps: Vec<usize>) -> Node {
    Node::new(
        Work::Call {
            call: KernelLaunch {
                module: 0,
                entry: "fill".into(),
                args: vec![
                    Arg::output(0),
                    Arg::scalar(value.to_ne_bytes().to_vec()),
                    Arg::scalar((N as i32).to_ne_bytes().to_vec()),
                ],
            },
            geometry: vec![Geometry {
                grid,
                block: [256, 1, 1],
                dynamic_shmem: 0,
            }],
        },
        deps,
    )
}

/// One candidate: the fill alone, in the choice's own index space.
fn fill_candidate(value: f32, grid: [u32; 3]) -> RecipeBody {
    RecipeBody::new(
        0,
        1,
        vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
        vec![],
        vec![fill_launch(value, grid, vec![])],
    )
}

/// The grid the *correct* arm launches. `N` elements need one block, so every
/// block past the first exits on the bounds guard and buys nothing but launch
/// overhead — which is the point: it makes the correct arm reliably the slower
/// one, by a margin that is scheduling work rather than a timing coincidence.
const SLOW_GRID: [u32; 3] = [4096, 1, 1];

/// The grid the *wrong* arm launches: the least work that still covers `N`.
const FAST_GRID: [u32; 3] = [1, 1, 1];

/// The program stated once, with nothing to choose: what a reference must come
/// from.
///
/// A reference derived by tuning the recipe *under test* is not a reference. An
/// unvalidated tuner selects by speed alone, so it would pick the very
/// candidate this test exists to reject and then compare that candidate against
/// itself — the comparison would pass for the wrong reason, or, as it did,
/// contradict its own premise.
fn reference_recipe() -> Recipe {
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![KernelModule::from_cuda_loadable(fill_ptx_cubin())],
            vec![],
            vec![clear_output(), fill_launch(FILL_VALUE, SLOW_GRID, vec![0])],
        ),
    )
    .unwrap()
}

/// A choice offering the same fill at two grids: both compute the program.
fn choice_fill_recipe() -> Recipe {
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![],
            vec![],
            vec![
                clear_output(),
                Node::new(
                    Work::Choice {
                        candidates: vec![
                            fill_candidate(FILL_VALUE, [1, 1, 1]),
                            fill_candidate(FILL_VALUE, [8, 1, 1]),
                        ],
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

/// The constant the selected fill writes, read back out of a choice-free
/// recipe's launch argument — which candidate won, in the units the test cares
/// about.
fn selected_fill_value(flat: &Recipe) -> f32 {
    flat.body()
        .nodes()
        .iter()
        .find_map(|node| match node.work() {
            Work::Call {
                call: KernelLaunch { args, .. },
                ..
            } => match args[1].segments() {
                [Segment::Bytes(bytes)] => Some(f32::from_ne_bytes(bytes[..4].try_into().unwrap())),
                _ => None,
            },
            _ => None,
        })
        .expect("a realized fill recipe launches exactly one fill")
}

/// Exact bytes: what settles agreement for a kernel that stores a constant.
struct ExactBytes;

impl kiln::OutputComparator for ExactBytes {
    fn agrees(
        &self,
        _output: usize,
        reference: CUdeviceptr,
        candidate: CUdeviceptr,
        nbytes: usize,
        stream: CUstream,
    ) -> bool {
        let read = |ptr| unsafe { to_host_bytes(CudaDriver::get(), ptr, nbytes, stream) };
        read(reference) == read(candidate)
    }
}

unsafe fn to_host_bytes(
    drv: &CudaDriver,
    src: CUdeviceptr,
    nbytes: usize,
    stream: CUstream,
) -> Vec<u8> {
    let mut host = vec![0u8; nbytes];
    let r = unsafe { (drv.cuMemcpyDtoHAsync_v2)(host.as_mut_ptr().cast(), src, nbytes, stream) };
    check(r, "cuMemcpyDtoHAsync").unwrap();
    StreamRef::from_raw(stream).synchronize().unwrap();
    host
}

/// A choice whose second candidate fills a different value: it is a real
/// kernel, it runs cleanly, and it computes the wrong thing.
fn wrong_candidate_recipe(wrong_value: f32) -> Recipe {
    Recipe::stating_edges(
        one_output(),
        CompileRequests::default(),
        RecipeBody::new(
            0,
            1,
            vec![],
            vec![],
            vec![
                clear_output(),
                Node::new(
                    Work::Choice {
                        // The wrong one launches a single block against the right
                        // one's 4096, so it is also the one that wins on speed.
                        candidates: vec![
                            fill_candidate(FILL_VALUE, SLOW_GRID),
                            fill_candidate(wrong_value, FAST_GRID),
                        ],
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

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn a_wrong_candidate_is_rejected_before_it_can_win_on_speed() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let reference = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let candidate = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let stream: CUstream = std::ptr::null_mut();

    let wrong_value = FILL_VALUE + 1.0;
    let recipe = wrong_candidate_recipe(wrong_value);

    // The caller establishes what the program is supposed to produce, from a
    // recipe that states one program and therefore needs no tuner.
    let reference_bindings = Bindings {
        inputs: &[],
        outputs: &[reference.ptr()],
        workspace: 0,
    };
    let mut resident =
        unsafe { ResidentRecipe::new(ctx, FlatRecipe::new(reference_recipe()).unwrap()) }.unwrap();
    unsafe { execute(&mut resident, &reference_bindings, stream) }.unwrap();
    StreamRef::from_raw(stream).synchronize().unwrap();
    assert!(
        reference
            .to_host()
            .unwrap()
            .iter()
            .all(|&v| v == FILL_VALUE),
        "the reference is the program's own result"
    );

    let live = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };

    // The premise: unvalidated, speed decides, and the wrong candidate is the
    // fast one. Without this the validated run below would prove less —
    // agreeing with a tuner that was going to pick the right arm anyway is not
    // evidence that validation changed an outcome.
    //
    // The margin is 4096 blocks against one, so this is launch overhead rather
    // than a timing coincidence. It is still a statement about a measurement,
    // which is why the rejection evidence below stands on its own and does not
    // lean on it.
    let unvalidated = unsafe { tune(&recipe, &live, stream) }.unwrap();
    assert_eq!(
        selected_fill_value(unvalidated.recipe()),
        wrong_value,
        "the wrong candidate launches one block against the right one's 4096, \
         so an unvalidated tuner takes it"
    );

    let outputs = [kiln::ValidationOutput {
        nbytes: NBYTES,
        reference: reference.ptr(),
        candidate: candidate.ptr(),
    }];
    let contract = kiln::ValidationContract {
        outputs: &outputs,
        // These recipes bind no input; a contract still states that exactly.
        input_extents: &[],
        comparator: &ExactBytes,
        initializer: &kiln::ZeroInitialized,
    };
    unsafe { memset(drv, out.ptr(), 0xEE, stream) };
    StreamRef::from_raw(stream).synchronize().unwrap();
    let untouched = out.to_host().unwrap();

    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let mut args = RealizationArgs::new(&recipe, &live, stream, RealizationPolicy::default());
    args.validation = Some(contract);
    let realized = unsafe { realizer.realize(args) }.unwrap();

    match realized.report.validation {
        kiln::ValidationStatus::Checked {
            outputs,
            validated,
            rejected,
        } => {
            assert_eq!(outputs, 1);
            assert_eq!(validated, 2, "both candidates got one clean run");
            assert_eq!(rejected, 1, "the wrong one was rejected");
        }
        other => panic!("expected a checked validation, got {other:?}"),
    }
    let tuned = kiln::kernel_entries(&realized.artifact);
    assert_eq!(tuned.len(), 1, "the surviving candidate was selected");
    assert_eq!(
        selected_fill_value(&realized.artifact),
        FILL_VALUE,
        "the same choice the tuner resolved to the wrong candidate on speed \
         alone resolves to the right one once a contract judges it"
    );

    // The live outputs carry the winner's result. Timing writes them — that is
    // what the `0xEE` was there to show — and every validation run went to the
    // caller's candidate storage instead, so nothing the rejected candidate
    // computed reached them.
    let live_result = out.to_host().unwrap();
    assert_ne!(
        live_result, untouched,
        "timing still uses the live outputs; validation is what must not"
    );
    assert!(
        live_result.iter().all(|&v| v == FILL_VALUE),
        "the live outputs hold the winner's result, never the rejected value"
    );
}

/// Validation writes the caller's candidate storage and nothing else. A
/// contract whose candidate buffer is the live output is the caller's mistake
/// to make, so the contract refuses it rather than kiln discovering it late.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn validation_never_writes_the_live_outputs() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let reference = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let candidate = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let stream: CUstream = std::ptr::null_mut();
    let recipe = choice_fill_recipe();

    // Establish the reference by running the recipe normally.
    let reference_bindings = Bindings {
        inputs: &[],
        outputs: &[reference.ptr()],
        workspace: 0,
    };
    let flat = unsafe { tune(&recipe, &reference_bindings, stream) }.unwrap();
    let mut resident = unsafe { ResidentRecipe::new(ctx, flat) }.unwrap();
    unsafe { execute(&mut resident, &reference_bindings, stream) }.unwrap();
    StreamRef::from_raw(stream).synchronize().unwrap();

    let outputs = [kiln::ValidationOutput {
        nbytes: NBYTES,
        reference: reference.ptr(),
        candidate: candidate.ptr(),
    }];
    let contract = kiln::ValidationContract {
        outputs: &outputs,
        // These recipes bind no input; a contract still states that exactly.
        input_extents: &[],
        comparator: &ExactBytes,
        initializer: &kiln::ZeroInitialized,
    };
    let live = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let mut args = RealizationArgs::new(&recipe, &live, stream, RealizationPolicy::default());
    args.validation = Some(contract);
    let realized = unsafe { realizer.realize(args) }.unwrap();
    assert!(matches!(
        realized.report.validation,
        kiln::ValidationStatus::Checked { rejected: 0, .. }
    ));

    // The live output is untouched by validation: only timing writes it, and
    // what it holds is the winning candidate's result, not a comparison's.
    assert!(
        out.to_host().unwrap().iter().all(|&v| v == FILL_VALUE),
        "timing wrote the live output; validation wrote only candidate storage"
    );

    // A contract that aims candidate storage at the caller's live output is
    // refused before anything runs — that buffer is the one validation exists
    // to keep out of.
    for aimed_at_live in [
        kiln::ValidationOutput {
            nbytes: NBYTES,
            reference: reference.ptr(),
            candidate: out.ptr(),
        },
        // And a reference kept there, which tuning would overwrite before the
        // first comparison.
        kiln::ValidationOutput {
            nbytes: NBYTES,
            reference: out.ptr(),
            candidate: candidate.ptr(),
        },
        // Overlap, not just identity: a candidate one element into the live
        // buffer is written just as thoroughly.
        kiln::ValidationOutput {
            nbytes: NBYTES,
            reference: reference.ptr(),
            candidate: out.ptr() + 4,
        },
    ] {
        let outputs = [aimed_at_live];
        let mut args = RealizationArgs::new(&recipe, &live, stream, RealizationPolicy::default());
        args.validation = Some(kiln::ValidationContract {
            outputs: &outputs,
            // These recipes bind no input; a contract still states that exactly.
            input_extents: &[],
            comparator: &ExactBytes,
            initializer: &kiln::ZeroInitialized,
        });
        assert!(
            matches!(
                unsafe { realizer.realize(args) },
                Err(kiln::Error::ValidationContract(_))
            ),
            "{aimed_at_live:?} overlaps a live buffer and must be refused"
        );
    }
}

/// Every fixture here is a recipe on a machine without a device too.
///
/// The tests that run them need a GPU and are `#[ignore]`d, so nothing else
/// builds them: a candidate that does not fill the slots its choice binds would
/// otherwise be refused only where no ordinary run reaches.
#[test]
fn every_fixture_here_is_a_recipe() {
    for recipe in [
        reference_recipe(),
        choice_fill_recipe(),
        wrong_candidate_recipe(FILL_VALUE + 1.0),
    ] {
        assert_eq!(recipe.validate(), Ok(()));
    }
}
