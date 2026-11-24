// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, verified whole-module optimization for [`xir`].

pub(crate) use xir::{analysis, dim_numbers, dtype, ir, op, ops, shape, verify};

#[cfg(test)]
pub(crate) use xir::builder;

mod optimize;

pub use optimize::*;
