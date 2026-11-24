// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One orchestration algebra over opaque computations.
//!
//! A recipe says what runs, in what order, through which buffers, and under
//! which alternatives. That is all it says. What a computation *is* — a kernel
//! and its module, a fragment of IR, an image — belongs to a computation
//! family, and this core carries the family's data without ever reading inside
//! it.
//!
//! The split is the point. Structure is written once and holds for every
//! family: bindings, hazards, workspace composition, nested choices,
//! validation, paths, mapping, substitution. A family states its own contract,
//! contexts and calls, and the merges those need when two recipes become one.
//! Neither half restates the other's facts, so there is no second truth to
//! disagree.
//!
//! [`Executable`] is the family the CUDA runtime consumes: a compile-request
//! graph for a root context, a module table for each body context, and a kernel
//! launch for each call.

mod builder;
mod computation;
mod data;
mod dependencies;
mod error;
mod executable;
mod launching;
mod map;
mod path;
mod substitute;
mod validate;
mod workspace;

pub use crate::effects::{Access, Effects};
pub use builder::{Choice, Placement, RecipeBuilder};
pub use computation::{
    Addresses, BytePortContract, Computation, DeclaredEffects, Geometry, Identify, IdentityWalk,
    count_covering_blocks,
};
pub use data::{FlatRecipe, Node, ProvedRecipe, Recipe, RecipeBody, RecipeFragment, Shape, Work};
pub use error::RecipeError;
pub use executable::{
    CompileRequests, Executable, ExecutableRefusal, KernelLaunch, Launches, ModuleBase,
    ModuleTable, RequestRemap,
};
pub use launching::Kernel;
pub use map::Mapping;
pub use path::{BodyPath, Descent, NodePath};
pub use workspace::{
    checked_body_workspace_upper_bound, checked_workspace_upper_bound, workspace_sizes,
    workspace_upper_bound,
};

#[cfg(test)]
#[path = "launching_tests.rs"]
mod launching_tests;

#[cfg(test)]
#[path = "laws_tests.rs"]
mod laws_tests;

#[cfg(test)]
#[path = "structure_fixture.rs"]
mod structure_fixture;

#[cfg(test)]
#[path = "placement_tests.rs"]
mod placement_tests;

#[cfg(test)]
#[path = "refusal_tests.rs"]
mod refusal_tests;

#[cfg(test)]
#[path = "ordering_tests.rs"]
mod ordering_tests;

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod workspace_tests;
