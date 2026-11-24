// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    cuda_toolchain_py::register(module)?;
    kiln_py::register(module)?;
    loom_py::register(module)?;
    recipe_py::register(module)?;
    Ok(())
}
