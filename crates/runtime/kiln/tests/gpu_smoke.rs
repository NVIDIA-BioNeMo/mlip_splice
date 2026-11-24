// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_driver::ffi::{
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
    CU_MEM_LOCATION_TYPE_DEVICE, CUmemAccessDesc, CUmemAllocationProp, CUmemLocation,
};
use cuda_driver::{
    CUcontext, CUdevice, CUdeviceptr, CUstream, Context, CudaDriver, DeviceBuffer, Event,
    StreamRef, check,
};
use kiln::{
    ArtifactCache, ArtifactCacheConfig, ArtifactCacheOutcome, Bindings, RealizationArgs,
    RealizationPolicy, Realizer, RealizerOptions, ResidentRecipe, WorkloadIdentity, execute, tune,
};
use recipe::{FlatRecipe, Geometry, Work, deserialize, serialize};

#[path = "gpu_smoke/recipes.rs"]
mod recipes;

use recipes::*;

/// The exit of this action, on a real device: a recipe whose one compile
/// request states two PTX modules round-trips through the wire, is realized as
/// *one* compile rather than two, and runs the `.entry` of one module against
/// the `.func` of the other.
#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_two_module_compile_unit_links_and_runs() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };

    let pending = linked_fill_recipe();
    assert_eq!(pending.validate(), Ok(()));
    // Through the wire first, so the unit that reaches the linker is the one
    // the artifact states rather than the one this process built.
    let pending = deserialize(&serialize(&pending).unwrap()).unwrap();
    assert_eq!(pending.context().as_slice()[0].sources.len(), 2);

    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let mut realized = unsafe {
        realizer.realize(RealizationArgs::new(
            &pending,
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy::default(),
        ))
    }
    .expect("two PTX modules link into one image");

    assert_eq!(
        realized.report.requests.len(),
        1,
        "two modules are one compile, not two"
    );
    let image = realized.artifact.body().context()[0]
        .code
        .images()
        .next()
        .expect("the linked image");
    let provenance = image
        .provenance()
        .expect("a compiled image records its own");
    assert_eq!(
        provenance.assembler, "nvjitlink",
        "a link is nvJitLink's, whatever the environment's default assembler is"
    );

    unsafe { execute(&mut realized.resident, &bindings, std::ptr::null_mut()) }.unwrap();
    let host = out.to_host().unwrap();
    assert!(
        host.iter().all(|&value| value == FILL_VALUE * SCALE),
        "the entry never reached the callee: {host:?}"
    );
    println!(
        "linked unit: out[0]={}, all {N} == {}",
        host[0],
        FILL_VALUE * SCALE
    );
}

/// A choice whose first candidate carries a cubin for another architecture,
/// followed by two that carry this one's. The arm that cannot be realized comes
/// first, so a sweep that stopped at it would visit nothing at all, and its
/// grid is its own, so which arms were visited is never in doubt.
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn an_unrealizable_arm_is_skipped_and_reported() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let recipe = one_foreign_arm_recipe(realizer.target_sm() + 1);

    let mut grids = Vec::new();
    let unrealized = realizer
        .for_each_covering_recipe(&recipe, &(), |flat| {
            let Work::Call { geometry, .. } = flat.recipe().body().nodes()[0].work() else {
                panic!("a realized covering recipe launches its fill")
            };
            let [Geometry { grid, .. }] = geometry[..] else {
                panic!("a kernel launch is one launch")
            };
            grids.push(grid);
        })
        .expect("the recipe under test is valid");

    assert!(
        !grids.contains(&FOREIGN_ARM_GRID),
        "an arm with no loadable image is never visited"
    );
    assert_eq!(
        grids,
        vec![[1, 1, 1], [8, 1, 1]],
        "the two working arms are still visited, in order, past the foreign first one"
    );
    assert_eq!(unrealized.len(), 1, "exactly one arm did not realize");
    assert_eq!(unrealized[0].arm, 0, "and it is the one that comes first");
    assert!(
        matches!(
            unrealized[0].error,
            kiln::Error::ModuleResolution { module: 0, .. }
        ),
        "the cause names the module with no image for this target, got {:?}",
        unrealized[0].error
    );

    // The whole recipe still realizes: tuning selects among the arms that are
    // feasible, and the broken one simply is not one of them.
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    let flat = unsafe { tune(&recipe, &bindings, std::ptr::null_mut()) }.expect("tune");
    let mut resident = unsafe { ResidentRecipe::new(ctx, flat) }.unwrap();
    unsafe { execute(&mut resident, &bindings, std::ptr::null_mut()) }.unwrap();
    assert!(out.to_host().unwrap().iter().all(|&v| v == FILL_VALUE));
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

unsafe fn memset(drv: &CudaDriver, dev: CUdeviceptr, value: u8, stream: CUstream) {
    let r = unsafe { (drv.cuMemsetD8Async)(dev, value, NBYTES, stream) };
    check(r, "cuMemsetD8Async (reset)").unwrap();
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (reset)").unwrap();
}

fn assert_all_filled(label: &str, out: &[f32], ws: &[f32]) {
    assert!(
        out.iter().all(|&x| x == FILL_VALUE),
        "{label}: output buffer not uniformly {FILL_VALUE}: {out:?}"
    );
    assert!(
        ws.iter().all(|&x| x == FILL_VALUE),
        "{label}: workspace (memcpy dst) not uniformly {FILL_VALUE}: {ws:?}"
    );
    println!(
        "{label}: out[0]={}, ws[0]={}, all {N} == {FILL_VALUE}",
        out[0], ws[0]
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn fill_recipe_runs() {
    let drv = CudaDriver::get();

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

    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let stream: CUstream = std::ptr::null_mut();

    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: ws.ptr(),
    };

    let recipe = fill_recipe([1, 1, 1]);
    assert_eq!(recipe.validate(), Ok(()));
    let mut loaded = unsafe {
        ResidentRecipe::new(
            ctx,
            FlatRecipe::new(deserialize(&serialize(&recipe).unwrap()).unwrap()).unwrap(),
        )
    }
    .expect("ResidentRecipe::new");

    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (exec)").unwrap();
    assert_all_filled(
        "execute/Static",
        &out.to_host().unwrap(),
        &ws.to_host().unwrap(),
    );

    unsafe { memset(drv, out.ptr(), 0, stream) };
    unsafe { memset(drv, ws.ptr(), 0, stream) };
    let wide_recipe = fill_recipe([8, 1, 1]);
    assert_eq!(wide_recipe.validate(), Ok(()));
    let mut wide_loaded = unsafe {
        ResidentRecipe::new(
            ctx,
            FlatRecipe::new(deserialize(&serialize(&wide_recipe).unwrap()).unwrap()).unwrap(),
        )
    }
    .expect("ResidentRecipe::new (wide grid)");
    unsafe { execute(&mut wide_loaded, &bindings, stream) }.expect("execute (wide grid)");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (wide grid)").unwrap();
    assert_all_filled(
        "execute/wide-grid",
        &out.to_host().unwrap(),
        &ws.to_host().unwrap(),
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn realizer_compiles_ptx_fallback() {
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
            &source_fill_recipe(false),
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy::default(),
        ))
    }
    .unwrap();
    assert!(realized.artifact.context().as_slice().is_empty());
    assert_eq!(realized.report.requests.len(), 1);
    unsafe { execute(&mut realized.resident, &bindings, std::ptr::null_mut()) }.unwrap();
    assert!(
        out.to_host()
            .unwrap()
            .iter()
            .all(|&value| value == FILL_VALUE)
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU and nvJitLink"]
fn a_warm_artifact_is_loaded_without_compiling_or_tuning() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    let cache = ArtifactCache::open(ArtifactCacheConfig {
        dir: None,
        max_bytes: None,
    });
    let workload = WorkloadIdentity::new("fill-n");
    let pending = source_fill_recipe(false);
    let mut realizer = Realizer::new(ctx, RealizerOptions::default()).unwrap();
    let realize = |realizer: &mut Realizer| {
        let mut args = RealizationArgs::new(
            &pending,
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy::default(),
        );
        args.cache = Some(&cache);
        args.workload = Some(&workload);
        unsafe { realizer.realize(args) }.unwrap()
    };

    let cold = realize(&mut realizer);
    assert_eq!(cold.report.artifact_cache, ArtifactCacheOutcome::Stored);
    assert_eq!(cold.report.requests.len(), 1);

    let mut warm = realize(&mut realizer);
    assert_eq!(
        warm.report.artifact_cache,
        ArtifactCacheOutcome::Hit { source: "memory" }
    );
    assert!(
        warm.report.requests.is_empty(),
        "a warm artifact compiles nothing"
    );
    assert!(warm.tuning.is_none(), "a warm artifact measures nothing");
    unsafe { execute(&mut warm.resident, &bindings, std::ptr::null_mut()) }.unwrap();
    assert!(
        out.to_host()
            .unwrap()
            .iter()
            .all(|&value| value == FILL_VALUE)
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn realizer_reuses_loader_image_without_compiling_source() {
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
    let realized = unsafe {
        realizer.realize(RealizationArgs::new(
            &source_fill_recipe(true),
            &bindings,
            std::ptr::null_mut(),
            RealizationPolicy::default(),
        ))
    }
    .unwrap();
    assert!(realized.report.requests.is_empty());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_is_repointed_across_buffers() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let recipe = fill_recipe([1, 1, 1]);
    assert_eq!(recipe.validate(), Ok(()));
    let mut loaded = unsafe {
        ResidentRecipe::new(
            ctx,
            FlatRecipe::new(deserialize(&serialize(&recipe).unwrap()).unwrap()).unwrap(),
        )
    }
    .expect("ResidentRecipe::new");

    let outs: Vec<DeviceBuffer<f32>> = (0..3)
        .map(|_| DeviceBuffer::<f32>::alloc(ctx, N).unwrap())
        .collect();
    for out in &outs {
        unsafe { memset(drv, out.ptr(), 0, stream) };
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: ws.ptr(),
        };
        unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (re-point)");
        let r = unsafe { (drv.cuStreamSynchronize)(stream) };
        check(r, "cuStreamSynchronize (re-point)").unwrap();
    }

    for (i, out) in outs.iter().enumerate() {
        let out = out.to_host().unwrap();
        assert!(
            out.iter().all(|&x| x == FILL_VALUE),
            "buffer {i} not uniformly {FILL_VALUE}: {out:?}"
        );
        println!(
            "re-point buffer {i}: out[0]={}, all {N} == {FILL_VALUE}",
            out[0]
        );
    }
}

/// Reserve, back, map and enable a VMM region (`cuMemCreate`/`cuMemMap`) —
/// the mapping kind XLA uses for some of its arenas, distinct from
/// `cuMemAlloc`. Leaked at test end.
unsafe fn vmm_alloc(drv: &CudaDriver, min_size: usize) -> CUdeviceptr {
    let prop = CUmemAllocationProp {
        type_: CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CU_MEM_HANDLE_TYPE_NONE,
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: 0,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let mut gran = 0usize;
    let r = unsafe { (drv.cuMemGetAllocationGranularity)(&mut gran, &prop, 0) };
    check(r, "cuMemGetAllocationGranularity").unwrap();
    let size = min_size.next_multiple_of(gran);
    let mut base: CUdeviceptr = 0;
    let r = unsafe { (drv.cuMemAddressReserve)(&mut base, size, 0, 0, 0) };
    check(r, "cuMemAddressReserve").unwrap();
    let mut handle = 0u64;
    let r = unsafe { (drv.cuMemCreate)(&mut handle, size, &prop, 0) };
    check(r, "cuMemCreate").unwrap();
    let r = unsafe { (drv.cuMemMap)(base, size, 0, handle, 0) };
    check(r, "cuMemMap").unwrap();
    let desc = CUmemAccessDesc {
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: 0,
        },
        flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let r = unsafe { (drv.cuMemSetAccess)(base, size, &desc, 1) };
    check(r, "cuMemSetAccess").unwrap();
    base
}

unsafe fn to_host_f32(drv: &CudaDriver, src: CUdeviceptr, stream: CUstream) -> Vec<f32> {
    let mut host = vec![0f32; N];
    let r = unsafe { (drv.cuMemcpyDtoHAsync_v2)(host.as_mut_ptr() as *mut _, src, NBYTES, stream) };
    check(r, "cuMemcpyDtoHAsync_v2").unwrap();
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (to_host)").unwrap();
    host
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_repoints_to_vmm_mapped_buffers() {
    // The resident graph is built on cuMemAlloc'd placeholders; XLA can then
    // bind buffers from a cuMemMap'd arena. Driver memcpy nodes reject that
    // re-point (mapping kind changed), which is why `Work::Memcpy` runs as a
    // copy kernel.
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();

    let region = unsafe { vmm_alloc(drv, 2 * NBYTES) };
    let out = region;
    let ws = region + NBYTES as CUdeviceptr;

    let recipe = fill_recipe([1, 1, 1]);
    assert_eq!(recipe.validate(), Ok(()));
    let mut loaded = unsafe {
        ResidentRecipe::new(
            ctx,
            FlatRecipe::new(deserialize(&serialize(&recipe).unwrap()).unwrap()).unwrap(),
        )
    }
    .expect("ResidentRecipe::new (vmm)");

    unsafe { memset(drv, out, 0, stream) };
    unsafe { memset(drv, ws, 0, stream) };
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out],
        workspace: ws,
    };
    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (vmm)");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (vmm)").unwrap();

    let out = unsafe { to_host_f32(drv, out, stream) };
    let ws = unsafe { to_host_f32(drv, ws, stream) };
    assert_all_filled("execute/vmm", &out, &ws);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_seeds_workspace_init() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    const SEED: f32 = -3.5;
    let recipe = seeded_recipe(SEED);
    assert_eq!(recipe.validate(), Ok(()));
    let mut loaded = unsafe {
        ResidentRecipe::new(
            ctx,
            FlatRecipe::new(deserialize(&serialize(&recipe).unwrap()).unwrap()).unwrap(),
        )
    }
    .expect("ResidentRecipe::new (seeded)");

    unsafe { memset(drv, ws.ptr(), 0xff, stream) };
    unsafe { memset(drv, out.ptr(), 0xff, stream) };

    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: ws.ptr(),
    };
    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (seeded)");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (seeded)").unwrap();

    let out = out.to_host().unwrap();
    assert!(
        out.iter().all(|&x| x == SEED),
        "execute did not seed workspace init: output is {out:?}, expected all {SEED}"
    );
    println!("workspace-init: out[0]={}, all {N} == {SEED}", out[0]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_repointing_preserves_pending_buffer_bindings() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream = StreamRef::from_raw(std::ptr::null_mut());
    let ctx = Context::current().raw();
    const CALLS: usize = 8;
    let outs: Vec<_> = (0..CALLS)
        .map(|_| DeviceBuffer::<f32>::alloc(ctx, N).unwrap())
        .collect();
    let workspaces: Vec<_> = (0..CALLS)
        .map(|_| DeviceBuffer::<f32>::alloc(ctx, N).unwrap())
        .collect();
    let delay_out = DeviceBuffer::<u64>::alloc(ctx, NBYTES / 8).unwrap();
    let mut delay = unsafe {
        ResidentRecipe::new(ctx, FlatRecipe::new(spin_recipe(1, 3_000_000_000)).unwrap())
    }
    .unwrap();
    let mut loaded =
        unsafe { ResidentRecipe::new(ctx, FlatRecipe::new(fill_recipe([1, 1, 1])).unwrap()) }
            .unwrap();
    let start = Event::new().unwrap();
    let end = Event::new().unwrap();

    for (out, ws) in outs.iter().zip(&workspaces) {
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: ws.ptr(),
        };
        unsafe { execute(&mut loaded, &bindings, stream.raw()) }.unwrap();
    }
    stream.synchronize().unwrap();
    for (out, ws) in outs.iter().zip(&workspaces) {
        unsafe { memset(drv, out.ptr(), 0xff, stream.raw()) };
        unsafe { memset(drv, ws.ptr(), 0xff, stream.raw()) };
    }

    start.record(stream).unwrap();
    let delay_bindings = Bindings {
        inputs: &[],
        outputs: &[delay_out.ptr()],
        workspace: 0,
    };
    unsafe { execute(&mut delay, &delay_bindings, stream.raw()) }.unwrap();
    end.record(stream).unwrap();
    for (out, ws) in outs.iter().zip(&workspaces) {
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: ws.ptr(),
        };
        unsafe { execute(&mut loaded, &bindings, stream.raw()) }.unwrap();
    }
    let pending = end.elapsed_since(&start);
    stream.synchronize().unwrap();
    const CUDA_ERROR_NOT_READY: u32 = 600;
    assert!(
        matches!(&pending, Err(error) if error.code == CUDA_ERROR_NOT_READY),
        "repointing waited for the queued GPU work: {pending:?}"
    );
    for (out, ws) in outs.iter().zip(&workspaces) {
        assert_all_filled(
            "pending replay",
            &out.to_host().unwrap(),
            &ws.to_host().unwrap(),
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn independent_nodes_overlap_in_the_graph() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();

    const CYCLES: u64 = 3_000_000;
    let outs: Vec<DeviceBuffer<u64>> = (0..2)
        .map(|_| DeviceBuffer::<u64>::alloc(ctx, 1).unwrap())
        .collect();
    let out_ptrs: Vec<CUdeviceptr> = outs.iter().map(|out| out.ptr()).collect();

    let time = |kernels: usize| {
        let recipe = spin_recipe(kernels, CYCLES);
        assert_eq!(recipe.validate(), Ok(()));
        let mut loaded =
            unsafe { ResidentRecipe::new(ctx, FlatRecipe::new(recipe).unwrap()) }.unwrap();
        let bindings = Bindings {
            inputs: &[],
            outputs: &out_ptrs[..kernels],
            workspace: 0,
        };
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let start = std::time::Instant::now();
            unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (spin)");
            let r = unsafe { (drv.cuStreamSynchronize)(stream) };
            check(r, "cuStreamSynchronize (spin)").unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    };

    let single = time(1);
    let pair = time(2);
    println!(
        "spin single {:.3}ms, independent pair {:.3}ms",
        single * 1e3,
        pair * 1e3
    );
    assert!(
        pair < single * 1.5,
        "independent graph nodes serialized: single {:.3}ms, pair {:.3}ms",
        single * 1e3,
        pair * 1e3
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn choice_tunes_and_runs() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let stream: CUstream = std::ptr::null_mut();

    let recipe = choice_fill_recipe();
    assert_eq!(recipe.validate(), Ok(()));

    let tune_out = DeviceBuffer::<f32>::alloc(Context::current().raw(), N).unwrap();
    let tune_bindings = Bindings {
        inputs: &[],
        outputs: &[tune_out.ptr()],
        workspace: 0,
    };
    let flat = unsafe { tune(&recipe, &tune_bindings, stream) }.expect("tune");
    assert!(
        flat.recipe()
            .body()
            .nodes()
            .iter()
            .all(|n| !matches!(n.work(), Work::Choice { .. })),
        "tuned recipe is Choice-free"
    );
    let ctx = Context::current().raw();
    let mut loaded = unsafe { ResidentRecipe::new(ctx, flat) }.expect("new (flat)");

    let mut run = |out: &DeviceBuffer<f32>| {
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: 0,
        };
        // Dirty the real output so only a real run can leave the fill value.
        unsafe { memset(drv, out.ptr(), 0xff, stream) };
        unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (choice)");
        let r = unsafe { (drv.cuStreamSynchronize)(stream) };
        check(r, "cuStreamSynchronize (choice)").unwrap();
        out.to_host().unwrap()
    };

    let a = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let b = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    for (label, out_buf) in [("first", &a), ("re-point", &b)] {
        let out = run(out_buf);
        assert!(
            out.iter().all(|&x| x == FILL_VALUE),
            "choice/{label}: output not uniformly {FILL_VALUE}: {out:?}"
        );
        println!("choice/{label}: out[0]={}, all {N} == {FILL_VALUE}", out[0]);
    }
}
