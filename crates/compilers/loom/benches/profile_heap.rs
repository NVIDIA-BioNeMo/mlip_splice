// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tracks allocations of one Loom compilation in-process and writes `dhat-heap.json`.

mod support;

#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

fn main() {
    let _profiler = dhat::Profiler::new_heap();
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let _profile = support::compile(&arguments);
}
