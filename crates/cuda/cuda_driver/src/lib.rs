// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `cuda_driver` — a safe, idiomatic Rust layer over libcuda.
//!
//! Implement nothing: just use the safe types. [`Error`]/[`Result`] wrap every
//! `CUresult`; [`Device`]/[`Context`] name what you run on; and the RAII handles
//! ([`Stream`]/[`StreamRef`], [`Event`], [`Module`]/[`Kernel`], [`Graph`]/
//! [`GraphExec`], [`DeviceBuffer`]) free on drop and carry their owning context,
//! so the API is multi-GPU-ready. Contexts/devices are always passed explicitly
//! — nothing reads or mutates a hidden "current" global behind your back.
//!
//! The raw `dlopen` function table lives in [`ffi`]; reach for it only for
//! operations this layer doesn't cover.

pub mod ffi;

mod context;
mod device;
mod error;
mod event;
mod graph;
mod memory;
mod module;
mod stream;

// Bring the raw substrate into crate scope (crate-internal) so the safe modules
// can name `crate::CudaDriver`, `crate::CUstream`, the constants, etc.
pub(crate) use ffi::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    CUDA_KERNEL_NODE_PARAMS, CUDA_MEMSET_NODE_PARAMS, CUDA_SUCCESS, CUkernel,
};

// The only raw items in the *public* root surface: the handle/result aliases the
// safe signatures expose, plus the `CudaDriver` table as an escape hatch.
// Everything else raw (constants, `CUDA_*_NODE_PARAMS`, …) is reached via `ffi`.
pub use ffi::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode,
    CUlibrary, CUmodule, CUresult, CUstream, CudaDriver,
};

pub use context::{Context, ContextGuard, ensure_context};
pub use device::Device;
pub use error::{Error, Result, check};
pub use event::Event;
pub use graph::{Graph, GraphExec, GraphNode, KernelNode, KernelParams, MemsetNode};
pub use memory::{DeviceBuffer, Span};
pub use module::{Function, Kernel, Module};
pub use stream::{LaunchConfig, Stream, StreamRef};
