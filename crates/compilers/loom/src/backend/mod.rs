// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod codegen;
pub(super) mod compiler;
mod facts;
mod iteration;
mod name;
mod partition;
mod reduction;
mod schedule;

pub use partition::ElementOrder;

#[derive(Clone, Copy, Debug, Default)]
struct BackendSettings {
    compiler: compiler::Settings,
    partition: partition::Settings,
    schedule: schedule::Settings,
}
