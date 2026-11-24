// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What one Loom compilation costs, on the CPU alone.
//!
//! Compilation reads its render target from a device, so a context is opened;
//! nothing is launched. The kernel and its size come from the command line, so
//! a growth exponent is fitted from runs of this one binary rather than from a
//! harness rewritten each time the question comes up.
//!
//! ```text
//! cargo bench -p loom --bench compile_cost -- broadcast 128 4096
//! LOOM_BENCH_PTX=corpus cargo bench -p loom --bench compile_cost -- projection 128 1024
//! ```
//!
//! The printed signature is the per-unit PTX hash. It is not a discriminating
//! check of emission order: the register allocator renumbers, so a wrong order
//! can render identical PTX. `cargo test -p gpu_ir --lib` is what catches that.

use std::time::Instant;

mod support;

use loom::{CompileOptions, compile};

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let (named, request) = support::request(&arguments);
    support::prepare();
    let started = Instant::now();
    let (_, profile) = compile(
        request,
        CompileOptions {
            keep_ptx: true,
            ..Default::default()
        },
    )
    .expect("compile");
    let elapsed = started.elapsed();
    let bytes: usize = profile.kernels.iter().map(|unit| unit.ptx_bytes).sum();
    let signature = profile
        .kernels
        .iter()
        .map(|unit| unit.ptx_hash.as_str())
        .collect::<Vec<_>>()
        .join(",");
    if let Some(path) = std::env::var_os("LOOM_BENCH_PTX") {
        let mut ptx = Vec::new();
        for unit in &profile.kernels {
            let text = unit.ptx.as_deref().expect("keep_ptx retained source");
            ptx.extend_from_slice(&(text.len() as u64).to_le_bytes());
            ptx.extend_from_slice(text.as_bytes());
        }
        std::fs::write(path, ptx).expect("write PTX corpus");
    }
    println!(
        "kernel={named} wall_s={:.6} simplify_s={:.6} plan_s={:.6} render_ptx_s={:.6} state_requests_s={:.6} build_recipe_s={:.6} kernels={} ptx_bytes={} signature={}",
        elapsed.as_secs_f64(),
        profile.timings.simplify.as_secs_f64(),
        profile.timings.plan.as_secs_f64(),
        profile.timings.render_ptx.as_secs_f64(),
        profile.timings.state_requests.as_secs_f64(),
        profile.timings.build_recipe.as_secs_f64(),
        profile.kernels.len(),
        bytes,
        signature
    );
}
