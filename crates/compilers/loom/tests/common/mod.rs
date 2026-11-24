// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a Loom GPU test needs before it can launch anything.

use std::ptr;

use cuda_driver::CUstream;

pub(crate) fn gpu_available() -> bool {
    cuda_toolchain::populate_from_python_for_tests();
    cuda_driver::ensure_context().is_ok()
        && cuda_driver::Device::current().sm_arch().unwrap_or(0) > 0
}

pub(crate) fn default_stream() -> CUstream {
    ptr::null_mut()
}
