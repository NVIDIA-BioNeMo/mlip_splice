// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-RAM accounting for the Recipe lifecycle, no GPU required.
//!
//! Models a MACE-like problem: `UNITS` Choice nodes, each with `CANDIDATES`
//! candidates; every candidate carries a unique ~200 KB cubin plus a ~100 KB
//! helper cubin shared by all candidates. Reports heap (counting allocator)
//! and RSS at each lifecycle stage: forge, wire, pending, tune churn,
//! post-tune resident.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAlloc;

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn note_alloc(n: usize) {
    let cur = CUR.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(cur, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CUR.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            note_alloc(new_size - layout.size());
        } else {
            CUR.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn rss_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let field = |key: &str| -> f64 {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (field("VmRSS:"), field("VmHWM:"))
}

fn stage(name: &str) {
    let heap = CUR.load(Ordering::Relaxed) as f64 / 1e6;
    let peak = PEAK.load(Ordering::Relaxed) as f64 / 1e6;
    let (rss, hwm) = rss_mb();
    println!(
        "{name:<28} heap {heap:8.1} MB  peak {peak:8.1} MB  rss {rss:8.1} MB  hwm {hwm:8.1} MB"
    );
}

fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}

const UNITS: usize = 12;
const CANDIDATES: usize = 32;
const CUBIN_BYTES: usize = 200 * 1024;
const HELPER_BYTES: usize = 100 * 1024;
const INIT_BYTES: usize = 4 * 1024;
/// What the modelled program is bound through. This example measures the RAM a
/// recipe tree costs and never reaches a device, so the extent is stated at the
/// scale of the work rather than read off a buffer — but it is stated, because
/// every unit binds input 0 and output 0.
const PORT_BYTES: usize = INIT_BYTES;

fn filled(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn candidate(unit: usize, cand: usize, helper: &recipe::Blob) -> recipe::ProvedRecipe {
    let mut b = recipe::RecipeBuilder::new(1, 1);
    let seed = (unit * CANDIDATES + cand) as u8;
    let main = b.module(filled(CUBIN_BYTES, seed));
    let help = b.module(helper.clone());
    let ws = b.scratch_init(filled(INIT_BYTES, seed));
    b.kernel(main, &format!("k_u{unit}_c{cand}"))
        .grid([64, 1, 1])
        .block([256, 1, 1])
        .read(recipe::Buf::Input(0))
        .write(ws)
        .write(recipe::Writable::Output(0))
        .add();
    b.kernel(help, "k_helper")
        .grid([8, 1, 1])
        .block([128, 1, 1])
        .read(ws)
        .write(recipe::Writable::Output(0))
        .add();
    // A candidate's ports are the parent buffers its bindings select, and this
    // one is bound onto the root's input 0 and output 0. It is not portless, so
    // it states those extents rather than claiming none.
    b.finish(port_signature()).unwrap()
}

/// The ports every recipe here is bound through: one input and one output of
/// `PORT_BYTES`, at the root and at every candidate the root's bindings map
/// onto it.
fn port_signature() -> recipe::RecipeSignature {
    recipe::RecipeSignature::new(vec![PORT_BYTES], vec![PORT_BYTES])
}

fn forge() -> recipe::Recipe {
    let helper = recipe::Blob::from_vec(filled(HELPER_BYTES, 0xEE));
    let mut b = recipe::RecipeBuilder::new(1, 1);
    for u in 0..UNITS {
        let candidates: Vec<recipe::ProvedRecipe> =
            (0..CANDIDATES).map(|c| candidate(u, c, &helper)).collect();
        for c in &candidates {
            assert_eq!(
                *c.recipe().contract(),
                port_signature(),
                "a candidate states the ports the root binds it onto"
            );
        }
        // Each candidate states the root's ports; `into_fragment` leaves them
        // behind here, where the choice's own bindings take over.
        b.choice(candidates.into_iter().map(|c| c.into_fragment()).collect())
            .inputs([recipe::Buf::Input(0)])
            .does([recipe::Buf::Input(0)], recipe::Access::Read)
            .outputs([recipe::Buf::Output(0)])
            .does([recipe::Buf::Output(0)], recipe::Access::Write)
            .add()
            .unwrap();
    }
    b.finish(port_signature()).unwrap().into_recipe()
}

fn main() {
    stage("start");

    let tree = forge();
    stage("forged tree");

    let bytes = recipe::serialize(&tree).unwrap();
    println!("wire bytes: {:.1} MB", bytes.len() as f64 / 1e6);
    stage("serialized (tree+wire)");

    drop(tree);
    stage("forge tree dropped");

    let spilled = recipe::Blob::spilled(&bytes).unwrap();
    let pending = recipe::deserialize_shared(spilled).unwrap();
    stage("pending decoded (+wire)");
    drop(bytes);
    stage("wire bytes dropped");

    reset_peak();
    // Structure only — this example has no device, so the covering recipes are
    // built but never realized.
    let mut winner: Option<recipe::Recipe> = None;
    let mut n = 0usize;
    for selection in kiln::covering_selections(&pending) {
        let covering = kiln::inline(&pending, &selection, &()).expect("a validated recipe inlines");
        if winner.is_none() {
            winner = Some(covering);
        }
        n += 1;
    }
    println!("tune churn: {n} covering flats");
    stage("tune churn peak");

    let mut winner = winner.unwrap();
    winner.detach_blobs();
    drop(pending);
    stage("post-tune resident");

    println!(
        "winner: {} modules, {} nodes",
        winner.body().context().len(),
        winner.body().nodes().len()
    );
    stage("end");
}
