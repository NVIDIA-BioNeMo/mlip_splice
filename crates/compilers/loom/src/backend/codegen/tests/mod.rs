// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{AlignedView, Domain, IndexMap};
use xir::builder::ModuleBuilder;
use xir::dtype::DType as Dtype;
use xir::dtype::DType;
use xir::ops::{Convert, Literal, Reduce};
use xir::shape::Shape;

use super::super::partition::{DEFAULT_MAX_KERNEL_STATEMENTS, plan};
use super::*;
use crate::FlatOperand;

mod support;
use support::*;

mod basic;
mod block_reduction;
mod expansion;
mod fiber;
mod mixed_outputs;
mod render;
mod scatter;
mod vector;
mod wellformed;
