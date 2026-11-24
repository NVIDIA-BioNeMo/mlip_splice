// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_driver::{CUstream, DeviceBuffer};
use cuda_jit::{CudaJit, CudaJitConfig};
use kiln::{Bindings, ResidentRecipe, execute, tune};
use recipe::{
    Arg, BufRef, CompileRequests, FlatRecipe, Geometry, KernelLaunch, KernelModule, Node, Recipe,
    RecipeBody, RecipeSignature, Work, deserialize, serialize,
};

/// `poly` reads two `f32` arrays of `n` and writes a third, so every port of
/// this problem holds the same bytes whatever block size runs it.
fn ports(n: usize) -> RecipeSignature {
    let nbytes = n * std::mem::size_of::<f32>();
    RecipeSignature::new(vec![nbytes, nbytes], vec![nbytes])
}

/// One candidate of the choice below: the same launch as a body, reached
/// through the checked exit, which is what proves nothing is left naming a
/// compile table the candidate would not carry.
fn candidate_at(cubin: &[u8], block: u32, n: usize) -> RecipeBody {
    launch_at(cubin, block, n)
        .into_fragment()
        .into_body()
        .expect("a candidate names no compile table")
}

fn launch_at(cubin: &[u8], block: u32, n: usize) -> Recipe {
    Recipe::stating_edges(
        ports(n),
        CompileRequests::default(),
        RecipeBody::new(
            2,
            1,
            vec![KernelModule::from_cuda_loadable(cubin.to_vec())],
            vec![],
            vec![Node::new(
                Work::Call {
                    call: KernelLaunch {
                        module: 0,
                        entry: "poly".into(),
                        args: vec![
                            Arg::scalar((n as i32).to_ne_bytes().to_vec()),
                            Arg::input(0),
                            Arg::input(1),
                            Arg::output(0),
                        ],
                    },
                    geometry: vec![Geometry {
                        grid: [(n as u32).div_ceil(block), 1, 1],
                        block: [block, 1, 1],
                        dynamic_shmem: 0,
                    }],
                },
                [],
            )],
        ),
    )
    .unwrap()
}

#[test]
#[serial_test::serial(gpu)]
fn poly_autotunes_block_size() {
    if cuda_driver::ensure_context().is_err() {
        eprintln!("poly_autotunes_block_size: no GPU, skipping");
        return;
    }
    let ctx = cuda_driver::Context::current().raw();
    let sm = cuda_driver::Device::current().sm_arch().unwrap_or(0);
    if sm <= 0 {
        eprintln!("poly_autotunes_block_size: no GPU, skipping");
        return;
    }
    let stream: CUstream = std::ptr::null_mut();

    let module = r#"
extern "C" __global__ void poly(
    int n, const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = x[i];
    out[i] = ((0.5f * value - 1.0f) * value) * value + 2.0f + y[i];
}
"#;
    let jit = CudaJit::load(CudaJitConfig::default()).expect("load CUDA toolchain");
    let cubin = jit
        .to_cubin(module.as_bytes(), None, &[], &[], sm)
        .expect("nvrtc compile");

    const N: usize = 1 << 20;

    let recipe = Recipe::stating_edges(
        ports(N),
        CompileRequests::default(),
        RecipeBody::new(
            2,
            1,
            vec![],
            vec![],
            vec![Node::new(
                Work::Choice {
                    candidates: vec![
                        candidate_at(&cubin, 64, N),
                        candidate_at(&cubin, 256, N),
                        candidate_at(&cubin, 1024, N),
                    ],
                    input_binding: vec![BufRef::Input(0), BufRef::Input(1)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .unwrap();

    // Inputs on the GPU, allocated in `ctx`: x = 1, y = 2, so p(1) + y = 3.5.
    let x = DeviceBuffer::from_host(ctx, &vec![1.0f32; N]).unwrap();
    let y = DeviceBuffer::from_host(ctx, &vec![2.0f32; N]).unwrap();
    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let tune_bindings = Bindings {
        inputs: &[x.ptr(), y.ptr()],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    let flat = unsafe { tune(&recipe, &tune_bindings, stream) }.expect("tune");
    let won = flat
        .recipe()
        .body()
        .nodes()
        .iter()
        .find_map(|n| match n.work() {
            Work::Call { geometry, .. } => {
                let [Geometry { block, .. }] = geometry[..] else {
                    panic!("a kernel launch is one launch")
                };
                Some(block[0])
            }
            _ => None,
        })
        .expect("a launch survives tuning");
    eprintln!("autotune picked block size {won}");

    let bytes = serialize(flat.recipe()).expect("serialize");
    let reloaded = deserialize(&bytes).expect("deserialize");
    assert_eq!(
        &reloaded,
        flat.recipe(),
        "the tuned recipe survives a bytes round-trip"
    );
    eprintln!(
        "tuned recipe dumped to {} bytes and loaded back",
        bytes.len()
    );

    let flat = FlatRecipe::new(reloaded).expect("tuned recipe is flat");
    let mut loaded = unsafe { ResidentRecipe::new(ctx, flat) }.expect("load");
    let bindings = Bindings {
        inputs: &[x.ptr(), y.ptr()],
        outputs: &[out.ptr()],
        workspace: 0,
    };
    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute");

    assert!(
        out.to_host()
            .unwrap()
            .iter()
            .all(|&v| (v - 3.5).abs() < 1e-6)
    );
}

/// The candidate bodies are bound through the ports their choice binds, and the
/// recipe holding them is a recipe — on a machine without a device too, where
/// the tuning test above is skipped and nothing else builds them.
#[test]
fn the_candidates_fill_the_slots_the_choice_binds() {
    const N: usize = 64;
    let cubin = [0u8; 4];
    let recipe = Recipe::stating_edges(
        ports(N),
        CompileRequests::default(),
        RecipeBody::new(
            2,
            1,
            vec![],
            vec![],
            vec![Node::new(
                Work::Choice {
                    candidates: vec![candidate_at(&cubin, 64, N), candidate_at(&cubin, 256, N)],
                    input_binding: vec![BufRef::Input(0), BufRef::Input(1)],
                    output_binding: vec![BufRef::Output(0)],
                },
                [],
            )],
        ),
    )
    .expect("the fixture is a recipe");
    assert_eq!(recipe.validate(), Ok(()));
}
