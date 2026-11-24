// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use domain_index::{AlignedView, Domain, IndexMap, SlotKind};
use recipe::{Address, BufRef, Geometry, KernelLaunch, Segment, Work, WritableBuf};
use xir::builder::ModuleBuilder;
use xir::dtype::DType as Dtype;
use xir::dtype::DType;
use xir::ops::{Literal, Select};
use xir::shape::Shape;

use super::super::partition::Settings as PartitionSettings;
use super::*;
use crate::FlatOperand;
use crate::profile::ElementOrderSummary;

/// What one candidate of a launch choice does: the entry it launches, the block
/// it launches at, and the register cap its module's request states. Module
/// requests index the whole recipe's table, so `recipe` is the root.
fn candidate_launch(
    recipe: &::recipe::Recipe,
    candidate: &::recipe::RecipeBody,
) -> Option<(String, u32, Option<u32>)> {
    candidate.nodes().iter().find_map(|node| match node.work() {
        Work::Call {
            call: KernelLaunch { module, entry, .. },
            geometry,
            ..
        } => {
            let [Geometry { block, .. }] = geometry[..] else {
                panic!("a kernel launch is one launch")
            };
            let request = candidate.context()[*module]
                .code
                .compile_request()
                .expect("a loom candidate module resolves from a compile request");
            Some((
                entry.clone(),
                block[0],
                recipe.context().as_slice()[request.0].options.max_registers,
            ))
        }
        Work::Choice { candidates, .. } => candidates
            .iter()
            .find_map(|candidate| candidate_launch(recipe, candidate)),
        _ => None,
    })
}

include!("compile.rs");
include!("sources.rs");
include!("candidates.rs");
include!("lowering.rs");
include!("orders.rs");
mod structure;
