// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod buffer;
mod decode;
mod dispatch;
mod dtype;
mod error;
mod remaining;
mod state;

pub use buffer::{AnyBuffer, Buffer, Ret};
pub use decode::{DecodeState, Stream};
pub use dispatch::{dispatch, dispatch_stage};
pub use dtype::{Dtype, dtype_size};
pub use error::{Error, Result};
pub use remaining::{RemainingArgs, RemainingArgsIter, RemainingRets, RemainingRetsIter};
pub use state::{RegisteredType, StateType};

pub use xla_ffi_sys as sys;

pub use xla_ffi_macros::handler;

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;
