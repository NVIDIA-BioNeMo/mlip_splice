// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One orchestration algebra, and the family of it a device runs.
//!
//! [`algebra`] is the general form: recipes over opaque computations, with
//! structure — bindings, hazards, choices, workspace, paths, mapping,
//! substitution — written once and holding for every family. What this crate's
//! root exposes is one specialization of it, [`algebra::Executable`]: kernel
//! launches over a compile-request graph and per-body module tables, which is
//! what the wire format, the C reader and the runtime consume.
//!
//! So `recipe::Recipe` *is* `algebra::Recipe<Executable>`, not a second model
//! beside it. A producer of device work never names the family; a consumer that
//! means to work over any family names [`algebra`] and takes its own parameter.

pub mod algebra;
mod aliases;
mod arg;
mod blob;
mod buffer;
mod code;
mod compilation_cost;
mod effects;
mod handles;
pub mod layout;
pub mod payloads;
mod requests;
mod signature;
pub mod wire;

mod hazards;

pub use algebra::{
    Choice, CompileRequests, Executable, ExecutableRefusal, Geometry, Kernel, KernelLaunch,
    Launches, Placement, count_covering_blocks,
};
pub use aliases::PortAliases;
pub use arg::{Arg, Segment};
pub use blob::{Blob, BlobDecodeError};
pub use buffer::{Address, BufKind, BufRef, WorkspaceBuf, WritableBuf};
pub use code::{
    CompileProvenance, CompileRequest, CompileRequestId, ImagePersistence, KernelCode,
    KernelModule, ModuleImage, PtxAssemblyMode, PtxCompileOptions, PtxTarget, ResourceBounds,
    SourceImage, SourceKind, conjoined_requires,
};
pub use compilation_cost::CompilationCostModel;
pub use effects::{Access, Effects};
pub use handles::{Buf, ModuleId, NodeId, WorkspaceId, Writable};
pub use signature::RecipeSignature;
pub use wire::{WireError, deserialize, deserialize_shared, serialize, serialize_spilled};

/// A recipe of kernel launches: what the wire format carries and what kiln
/// runs.
///
/// Every one of these names the general form at its executable specialization,
/// and defaults to the one family that has a wire format. A consumer of device
/// work writes `Recipe` and means this; one written against several families
/// writes `Recipe<F>` and means the same type.
pub type Recipe<F = Executable> = algebra::Recipe<F>;
pub type RecipeBody<F = Executable> = algebra::RecipeBody<F>;
pub type RecipeFragment<F = Executable> = algebra::RecipeFragment<F>;
pub type ProvedRecipe<F = Executable> = algebra::ProvedRecipe<F>;
pub type FlatRecipe<F = Executable> = algebra::FlatRecipe<F>;
pub type RecipeBuilder<F = Executable> = algebra::RecipeBuilder<F>;
pub type Node<F = Executable> = algebra::Node<F>;
pub type Work<F = Executable> = algebra::Work<F>;
pub type RecipeError<R = ExecutableRefusal> = algebra::RecipeError<R>;
