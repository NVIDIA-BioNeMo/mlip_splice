// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::OnceCell;
use std::collections::BTreeSet;

use domain_index::{Domain, DomainIndex, IndexMap};
use gpu_ir::{Function, OpId, OpKind, Scalar};
use xir::builder::ModuleBuilder;
use xir::dim_numbers::DotDimensionNumbers;
use xir::dtype::DType as Dtype;
use xir::dtype::DType;
use xir::ops::{DotAlgorithm, Literal, Select};
use xir::shape::Shape;

use super::super::codegen::{
    BLOCK_SIZES, Flavor, emit_consumer_kernel, emit_consumer_kernel_flavor, emit_kernel,
    emit_producer_kernel,
};
use super::super::compiler::emit_planned_kernel;
use super::super::iteration::Plan as DomainPlan;
use super::super::partition::{
    IncidenceProfile, KernelPart, Settings as PartitionSettings, Term,
    evaluation_profile_with_provenance, partition_cost, plan, plan_shared_work,
    selected_term_values, slice_bindings, statement_count,
};
use super::{Launch, PlanIr, ReductionProvenance, WriteMode};
use crate::{FlatOperand, LiftedProgram};

mod fixtures;
mod guard_expansion;
mod guards;
mod lane_hoist;
mod materialization;
mod reductions;
mod scheduling;
mod support;

use fixtures::*;
use support::*;
