// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Progressive realization end to end: a deadline still yields an executable
//! selection, and resuming uses the completed compile instead of assembling it
//! again.

use cuda_driver::{CUdevice, CUdeviceptr, Context, CudaDriver, DeviceBuffer, StreamRef, check};
use kiln::{
    Bindings, RealizationArgs, RealizationPolicy, Realizer, RealizerOptions, TuningPolicy, execute,
};
use recipe::{
    Arg, Blob, BufRef, CompileRequest, CompileRequestId, CompileRequests, Geometry, KernelLaunch,
    KernelModule, Node, PtxAssemblyMode, PtxCompileOptions, PtxTarget, Recipe, RecipeBody,
    RecipeSignature, ResourceBounds, SourceImage, SourceKind, Work,
};

const FILL_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry fill(
    .param .u64 fill_param_0,
    .param .f32 fill_param_1
)
{
    .reg .f32 %f1;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [fill_param_0];
    ld.param.f32 %f1, [fill_param_1];
    cvta.to.global.u64 %rd2, %rd1;
    st.global.f32 [%rd2], %f1;
    ret;
}
"#;

const FILL_VALUE: f32 = 7.0;

fn request(source: String) -> CompileRequest {
    CompileRequest {
        sources: vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [7, 0],
                minimum_sm: 52,
            },
            blob: Blob::zstd_spilled(source.as_bytes()).unwrap(),
        }],
        options: PtxCompileOptions {
            max_registers: None,
            assembly_mode: PtxAssemblyMode::Relocatable,
        },
        requires: Default::default(),
    }
}

fn candidate(request: usize) -> RecipeBody {
    RecipeBody::new(
        0,
        1,
        vec![KernelModule::from_request(
            CompileRequestId(request),
            Vec::new(),
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
                    ],
                },
                geometry: vec![Geometry {
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                    dynamic_shmem: 0,
                }],
            },
            [],
        )],
    )
}

fn two_source_candidates() -> Recipe {
    Recipe::stating_edges(
        // One output, the single `f32` each candidate fills.
        RecipeSignature::new(Vec::new(), vec![std::mem::size_of::<f32>()]),
        CompileRequests::new(vec![
            request(FILL_PTX.to_string()),
            request(format!("{FILL_PTX}\n// second candidate\n")),
        ]),
        RecipeBody::new(
            0,
            1,
            Vec::new(),
            Vec::new(),
            vec![Node::new(
                Work::Choice {
                    candidates: vec![candidate(0), candidate(1)],
                    input_binding: Vec::new(),
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

unsafe fn init_context() {
    let driver = CudaDriver::get();
    check(unsafe { (driver.cuInit)(0) }, "cuInit").unwrap();
    let mut device: CUdevice = 0;
    check(
        unsafe { (driver.cuDeviceGet)(&mut device, 0) },
        "cuDeviceGet",
    )
    .unwrap();
    let mut context = std::ptr::null_mut();
    check(
        unsafe { (driver.cuDevicePrimaryCtxRetain)(&mut context, device) },
        "cuDevicePrimaryCtxRetain",
    )
    .unwrap();
    check(
        unsafe { (driver.cuCtxSetCurrent)(context) },
        "cuCtxSetCurrent",
    )
    .unwrap();
}

fn assert_filled(output: &DeviceBuffer<f32>) {
    assert_eq!(output.to_host().unwrap(), vec![FILL_VALUE]);
}

#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn deadline_returns_executable_baseline_and_resume_reuses_completed_compile() {
    unsafe { init_context() };
    let context = Context::current().raw();
    let output = DeviceBuffer::<f32>::alloc(context, 1).unwrap();
    let outputs: [CUdeviceptr; 1] = [output.ptr()];
    let bindings = Bindings {
        inputs: &[],
        outputs: &outputs,
        workspace: 0,
    };
    let recipe = two_source_candidates();
    let mut realizer = Realizer::new(context, RealizerOptions::default()).unwrap();

    let first_policy = RealizationPolicy {
        tuning: TuningPolicy {
            scheduling_time_limit_ms: Some(0.0),
        },
        ..RealizationPolicy::default()
    };
    let first = unsafe {
        realizer.realize(RealizationArgs::new(
            &recipe,
            &bindings,
            std::ptr::null_mut(),
            first_policy,
        ))
    }
    .unwrap();
    assert!(!first.progress.is_complete());
    assert_eq!(first.progress.choices[0].unscheduled, 1);
    assert_eq!(first.report.requests.len(), 1);
    let progress = first.progress.clone();
    let mut baseline = first.resident;
    unsafe { execute(&mut baseline, &bindings, std::ptr::null_mut()) }.unwrap();
    StreamRef::from_raw(std::ptr::null_mut())
        .synchronize()
        .unwrap();
    assert_filled(&output);

    let mut resumed_args = RealizationArgs::new(
        &recipe,
        &bindings,
        std::ptr::null_mut(),
        RealizationPolicy::default(),
    );
    resumed_args.resume = Some(&progress);
    let mut resumed = unsafe { realizer.realize(resumed_args) }.unwrap();
    assert!(resumed.progress.is_complete());
    let incumbent = resumed
        .report
        .requests
        .iter()
        .find(|request| request.request == 0)
        .expect("resume resolves its incumbent first");
    assert!(
        matches!(incumbent.cache_source.as_deref(), Some("memory" | "disk")),
        "the completed incumbent compile must come from the compile cache, got {incumbent:?}"
    );
    unsafe { execute(&mut resumed.resident, &bindings, std::ptr::null_mut()) }.unwrap();
    StreamRef::from_raw(std::ptr::null_mut())
        .synchronize()
        .unwrap();
    assert_filled(&output);
}

/// The fixture is a recipe on a machine without a device too: the test that
/// runs it needs a GPU and nvJitLink and is `#[ignore]`d.
#[test]
fn the_fixture_is_a_recipe() {
    let recipe = two_source_candidates();
    assert_eq!(recipe.validate(), Ok(()));
    assert_eq!(recipe.body().outputs(), recipe.contract().outputs().len());
}
